(() => {
  const status = document.getElementById('resource-status');
  const rows = document.getElementById('resource-rows');
  const number = value => Number.isFinite(value) ? value.toFixed(1) : '—';
  const mib = value => Number.isFinite(value) ? number(value / 1048576) : '—';
  async function refresh() {
    let delay = 1000;
    try {
      const response = await fetch('/api/resources', {cache: 'no-store'});
      if (!response.ok) throw Error(response.statusText);
      const data = await response.json();
      rows.replaceChildren();
      if (!data.enabled) {
        status.textContent = 'Resource monitoring is disabled.';
      } else if (!data.sample) {
        status.textContent = data.error || 'Waiting for first sample…';
      } else {
        delay = Math.max(100, data.interval_seconds * 1000);
        const sample = data.sample;
        const age = Math.max(0, Date.now() / 1000 - sample.timestamp);
        const messages = [new Date(sample.timestamp * 1000).toLocaleTimeString(),
          sample.system.cpu_count + ' online logical CPUs'];
        if (age > Math.max(2, data.interval_seconds * 3)) messages.push('Stale sample');
        if (data.csv_path) messages.push('CSV: ' + data.csv_path);
        if (data.error) messages.push(data.error);
        if (data.csv_error) messages.push('CSV: ' + data.csv_error);
        if (sample.system.gpu_unavailable_reason) messages.push(sample.system.gpu_unavailable_reason);
        status.textContent = messages.join(' · ');
        const items = [
          ['System', sample.system.cpu_percent, sample.system.ram_used_bytes, sample.system.gpu_percent],
          ...Object.entries(sample.processes).map(([name, value]) => [
            name + (value.pids.length ? ' (PID ' + value.pids.join(', ') + ')' : ' (inactive)'),
            value.cpu_percent, value.rss_bytes, null
          ])
        ];
        for (const [name, cpu, ram, gpu] of items) {
          const row = document.createElement('tr');
          for (const value of [name, number(cpu), mib(ram), number(gpu)]) {
            const cell = document.createElement('td');
            cell.textContent = value;
            row.append(cell);
          }
          rows.append(row);
        }
      }
    } catch (error) {
      status.textContent = 'Resource monitor unavailable: ' + error.message;
    } finally {
      setTimeout(refresh, delay);
    }
  }
  refresh();
})();
