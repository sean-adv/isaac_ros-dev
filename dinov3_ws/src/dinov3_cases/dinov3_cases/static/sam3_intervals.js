/* Server timestamps avoid counting HTTP polling/network latency as inference time. */
const sam3Intervals = (() => {
  let stream = null;
  let previous = null;
  let samples = [];
  let skipped = 0;
  const grid = document.getElementById('sam3-interval-grid');
  const status = document.getElementById('sam3-interval-status');

  function render() {
    const values = samples.map(sample => sample.ms);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 0;
    const cells = [];
    for (let i = 0; i < 100; i++) {
      const cell = document.createElement('span');
      cell.className = 'sam3-interval-cell';
      const sample = samples[i];
      if (sample) {
        cell.dataset.level = String(max === min ? 2 : 1 + Math.min(3,
          Math.floor(4 * (sample.ms - min) / (max - min))));
        cell.title = `SAM3 #${sample.sequence - 1} → #${sample.sequence}: ${sample.ms.toFixed(1)} ms`;
        cell.setAttribute('aria-label', cell.title);
        cell.tabIndex = 0;
      } else {
        cell.title = 'Waiting for inference interval';
        cell.setAttribute('aria-hidden', 'true');
      }
      cells.push(cell);
    }
    grid.replaceChildren(...cells);
    status.textContent = samples.length
      ? `${samples.length}/100 intervals · latest ${values[values.length - 1].toFixed(1)} ms · average ${(values.reduce((sum, ms) => sum + ms, 0) / values.length).toFixed(1)} ms · min ${min.toFixed(1)} / max ${max.toFixed(1)} ms`
      : 'Waiting for two SAM3 inferences…';
    if (skipped) status.textContent += ` · ${skipped} unavailable intervals skipped after a connection gap`;
  }

  document.getElementById('sam3-interval-clear').onclick = () => {
    samples = [];
    skipped = 0;
    // Keep the cursor so the next poll cannot replay cleared server events.
    render();
  };

  function update(data) {
    if (!data || !data.stream || !Array.isArray(data.events)) return;
    let changed = false;
    if (stream !== data.stream) {
      stream = data.stream;
      previous = null;
      samples = [];
      skipped = 0;
      changed = true;
    }
    for (const event of data.events) {
      if (!Number.isInteger(event.sequence) || event.sequence < 1 ||
          !Number.isFinite(event.started_at)) continue;
      if (previous && event.sequence <= previous.sequence) continue;
      if (previous) {
        const gap = event.sequence - previous.sequence;
        const ms = (event.started_at - previous.started_at) * 1000;
        if (gap === 1 && ms >= 0) {
          samples.push({sequence: event.sequence, ms});
          if (samples.length > 100) samples.shift();
        } else {
          // Never turn missed samples into one misleadingly long interval.
          skipped += gap;
        }
      }
      previous = event;
      changed = true;
    }
    if (changed) render();
  }

  render();
  return {update};
})();
