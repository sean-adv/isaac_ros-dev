"""Linux resource sampling independent of inference, Torch and ROS."""
import copy
import csv
import glob
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time


class ResourceSampler:
    """CPU percentages are interval averages normalized to online system capacity."""

    def __init__(self, proc=Path('/proc'), root_pid=None):
        self.proc = Path(proc)
        self.root_pid = os.getpid() if root_pid is None else root_pid
        self.ticks = os.sysconf('SC_CLK_TCK')
        self.page = os.sysconf('SC_PAGE_SIZE')
        self.previous = {}
        self.previous_cpu = None
        self.gpu_path = None

    def _processes(self):
        records = {}
        for path in self.proc.glob('[0-9]*/stat'):
            try:
                # comm can contain spaces and parentheses.
                fields = path.read_text().rsplit(')', 1)[1].split()
                pid = int(path.parent.name)
                records[pid] = dict(
                    ppid=int(fields[1]), ticks=int(fields[11]) + int(fields[12]),
                    start=int(fields[19]), rss=int(fields[21]) * self.page,
                    command=(path.parent / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'),
                )
            except (OSError, ValueError, IndexError):
                continue  # Process can exit at any point during sampling.
        descendants = {self.root_pid}
        while True:
            added = {pid for pid, row in records.items() if row['ppid'] in descendants} - descendants
            if not added:
                break
            descendants.update(added)
        return {pid: row for pid, row in records.items() if pid in descendants}

    def _gpu(self):
        if self.gpu_path is None:
            candidates = []
            for pattern in (
                '/sys/devices/platform/*gpu*/load',
                '/sys/devices/platform/bus@0/*gpu*/load',
                '/sys/class/devfreq/*gpu*/load',
                '/sys/class/devfreq/*gpu*/device/load',
            ):
                candidates.extend(glob.glob(pattern))
            self.gpu_path = next(iter(candidates), None)
        try:
            value = float(Path(self.gpu_path).read_text()) / 10 if self.gpu_path else None
            return value if value is not None and 0 <= value <= 100 else None
        except (OSError, ValueError):
            self.gpu_path = None
            return None

    def sample(self):
        now = time.monotonic()
        cpu_lines = (self.proc / 'stat').read_text().splitlines()
        # Match the system CPU denominator, including online CPUs outside affinity.
        # Read each sample so CPU hotplug does not leave a hardcoded divisor.
        cpu_count = sum(bool(re.match(r'^cpu\d+\s', line)) for line in cpu_lines)
        if not cpu_count:
            cpu_count = max(1, os.sysconf('SC_NPROCESSORS_ONLN'))
        groups = {name: dict(pids=[], cpu_percent=None, cpu_complete=True, rss_bytes=0)
                  for name in ('dinov3_main', 'sam3', 'nvdcf')}
        previous = {}
        for pid, row in self._processes().items():
            if pid == self.root_pid:
                name = 'dinov3_main'
            elif 'dinov3_cases.grounding_ipc' in row['command']:
                name = 'sam3'
            elif 'nvdcf_bridge' in row['command'] or 'dinov3-nvdcf-bridge' in row['command']:
                name = 'nvdcf'
            else:
                continue
            group = groups[name]
            group['pids'].append(pid)
            group['rss_bytes'] += max(0, row['rss'])
            key = (pid, row['start'])  # Protect against PID reuse.
            old = self.previous.get(key)
            previous[key] = (now, row['ticks'])
            if old is not None and now > old[0]:
                cpu = max(0, row['ticks'] - old[1]) / self.ticks / (now - old[0]) * 100
                group['cpu_percent'] = (group['cpu_percent'] or 0) + cpu
            else:
                group['cpu_complete'] = False
        self.previous = previous
        system = dict(cpu_percent=None, cpu_count=cpu_count, ram_total_bytes=None, ram_used_bytes=None,
                      gpu_percent=self._gpu(), gpu_source='sysfs' if self.gpu_path else None)
        fields = [int(v) for v in cpu_lines[0].split()[1:]]
        # guest times are already included in user/nice.
        total, idle = sum(fields[:8]), fields[3] + fields[4]
        if self.previous_cpu is not None:
            dt, di = total - self.previous_cpu[0], idle - self.previous_cpu[1]
            if dt > 0:
                system['cpu_percent'] = max(0, min(100, 100 * (dt - di) / dt))
        self.previous_cpu = (total, idle)
        memory = {}
        for line in (self.proc / 'meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            memory[key] = int(value.split()[0]) * 1024
        system['ram_total_bytes'] = memory.get('MemTotal')
        if 'MemAvailable' in memory and 'MemTotal' in memory:
            system['ram_used_bytes'] = memory['MemTotal'] - memory['MemAvailable']
        for group in groups.values():
            group['pids'].sort()
            if not group['cpu_complete']:
                group['cpu_percent'] = None
            group['cpu_core_percent'] = group['cpu_percent']
            if group['cpu_percent'] is not None:
                group['cpu_percent'] /= cpu_count
        return dict(timestamp=time.time(), monotonic=now, system=system, processes=groups)


class ResourceMonitor:
    def __init__(self, interval=0.5, csv_dir='', publish=None, on_error=None):
        if isinstance(interval, bool) or not math.isfinite(interval) or interval < 0.1:
            raise ValueError('resource_interval must be finite and >= 0.1 seconds.')
        self.interval = interval
        self.sampler = ResourceSampler()
        self.publish = publish
        self.on_error = on_error or (lambda message: None)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.csv_error = None
        self.csv_path = None
        self.file = None
        self.writer = None
        self.tegrastats = None
        self.smi = None
        self.smi_attempted = False
        self.smi_buffer = b''
        self.smi_sample = None
        self.gpu_buffer = b''
        self.gpu_sample = None
        self.thread = None
        if csv_dir:
            try:
                directory = Path(csv_dir).expanduser()
                directory.mkdir(parents=True, exist_ok=True)
                self.csv_path = str(directory / f'resources-{time.time_ns()}.csv')
                self.file = open(self.csv_path, 'x', newline='')
                self.writer = csv.writer(self.file)
                self.writer.writerow(['timestamp', 'component', 'pids', 'cpu_percent',
                                      'rss_bytes', 'ram_total_bytes', 'ram_used_bytes',
                                      'gpu_percent', 'gpu_source', 'cpu_count', 'cpu_core_percent'])
            except OSError as error:
                self.csv_error = str(error)
                self.on_error(f'Resource CSV unavailable: {error}')
        executable = shutil.which('tegrastats')
        if executable:
            try:
                self.tegrastats = subprocess.Popen(
                    [executable, '--interval', str(round(interval * 1000))],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                os.set_blocking(self.tegrastats.stdout.fileno(), False)
            except OSError as error:
                self.on_error(f'tegrastats unavailable; using GPU sysfs: {error}')

    def snapshot(self):
        with self.lock:
            return dict(enabled=True, interval_seconds=self.interval, csv_path=self.csv_path,
                        csv_error=self.csv_error, error=self.error, sample=copy.deepcopy(self.latest))

    def _smi_gpu(self):
        # Thor may omit GR3D_FREQ and expose NVML instead. Keep one query process.
        if not self.smi_attempted:
            self.smi_attempted = True
            executable = shutil.which('nvidia-smi')
            if executable:
                try:
                    self.smi = subprocess.Popen(
                        [executable, '--id=0', '--query-gpu=utilization.gpu',
                         '--format=csv,noheader,nounits', f'--loop-ms={round(self.interval * 1000)}'],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    )
                    os.set_blocking(self.smi.stdout.fileno(), False)
                except OSError as error:
                    self.on_error(f'nvidia-smi unavailable: {error}')
        if self.smi is not None:
            try:
                self.smi_buffer += os.read(self.smi.stdout.fileno(), 65536)
                lines = self.smi_buffer.split(b'\n')
                self.smi_buffer = lines.pop()[-65536:]
                for line in lines:
                    value = float(line.strip())
                    if 0 <= value <= 100:
                        self.smi_sample = (time.monotonic(), value)
            except (OSError, ValueError):
                pass
        if self.smi_sample and time.monotonic() - self.smi_sample[0] <= max(2, 3 * self.interval):
            return self.smi_sample[1]
        return None

    def _gpu(self, sample):
        if self.tegrastats is not None:
            try:
                chunk = os.read(self.tegrastats.stdout.fileno(), 65536)
                self.gpu_buffer += chunk
                lines = self.gpu_buffer.split(b'\n')
                self.gpu_buffer = lines.pop()[-65536:]
                for line in lines:
                    match = re.search(rb'GR3D_FREQ\s+(\d+)%', line)
                    if match:
                        self.gpu_sample = (time.monotonic(), int(match[1]))
            except (OSError, ValueError):
                pass
        if self.gpu_sample and time.monotonic() - self.gpu_sample[0] <= max(2, 3 * self.interval):
            sample['system'].update(gpu_percent=self.gpu_sample[1], gpu_source='tegrastats')
        if sample['system']['gpu_percent'] is None:
            value = self._smi_gpu()
            if value is not None:
                sample['system'].update(gpu_percent=value, gpu_source='nvidia-smi')
        sample['system']['gpu_unavailable_reason'] = (
            'No readable GPU load source; check tegrastats, sysfs or nvidia-smi access.'
            if sample['system']['gpu_percent'] is None else None)

    def _write_csv(self, sample):
        if self.writer is None:
            return
        try:
            system = sample['system']
            self.writer.writerow([sample['timestamp'], 'system', '', system['cpu_percent'], '',
                                  system['ram_total_bytes'], system['ram_used_bytes'],
                                  system['gpu_percent'], system['gpu_source'], system['cpu_count'], ''])
            for name, row in sample['processes'].items():
                self.writer.writerow([sample['timestamp'], name, ';'.join(map(str, row['pids'])),
                                      row['cpu_percent'], row['rss_bytes'], '', '', '', '',
                                      system['cpu_count'], row['cpu_core_percent']])
            self.file.flush()
        except OSError as error:
            with self.lock:
                self.csv_error = str(error)
            self.writer = None
            self.on_error(f'Resource CSV stopped: {error}')

    def start(self):
        self.thread = threading.Thread(target=self._run, name='resource-monitor', daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                sample = self.sampler.sample()
                self._gpu(sample)
                with self.lock:
                    self.latest, self.error = sample, None
                self._write_csv(sample)
                if self.publish is not None:
                    self.publish(json.dumps(sample, allow_nan=False))
            except Exception as error:
                with self.lock:
                    changed = self.error != str(error)
                    self.error = str(error)
                if changed:
                    self.on_error(f'Resource monitoring: {error}')
            self.stop.wait(max(0, self.interval - (time.monotonic() - started)))

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        for process in (self.tegrastats, self.smi):
            if process is None:
                continue
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            process.stdout.close()
        if self.file is not None:
            self.file.close()
