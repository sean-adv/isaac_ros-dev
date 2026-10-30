/* Native camera coordinates -> encoded video -> object-fit: contain display. */
(function () {
  function regionBounds(native, values) {
    function axis(length, size, center) {
      if (size === 0) return [0, length];
      const start = center === -1 ? Math.floor((length - size) / 2) : center - Math.floor(size / 2);
      return [Math.max(0, Math.min(length, start)), Math.max(0, Math.min(length, start + size))];
    }
    const [left, right] = axis(native[0], values.width, values.center_x);
    const [top, bottom] = axis(native[1], values.height, values.center_y);
    return [left, top, right, bottom];
  }

  function previewRect(bounds, native, viewport, encoded) {
    const scale = Math.min(viewport[0] / encoded[0], viewport[1] / encoded[1]);
    const width = encoded[0] * scale, height = encoded[1] * scale;
    return {
      left: (viewport[0] - width) / 2 + bounds[0] / native[0] * width,
      top: (viewport[1] - height) / 2 + bounds[1] / native[1] * height,
      width: (bounds[2] - bounds[0]) / native[0] * width,
      height: (bounds[3] - bounds[1]) / native[1] * height
    };
  }

  if (typeof module !== 'undefined') module.exports = {regionBounds, previewRect};
  if (typeof document === 'undefined') return;
  const form = document.getElementById('roi-form');
  const video = document.getElementById('overlay');
  const box = document.getElementById('roi-preview');
  const toggle = document.getElementById('roi-preview-enabled');
  const status = document.getElementById('roi-status');
  const keys = ['width', 'height', 'center_x', 'center_y'];
  let state = null, source = 'camera', dirty = false, saving = false;

  function values() {
    const result = Object.fromEntries(keys.map(key => [key, Number(form.elements.namedItem(key).value)]));
    return keys.every(key => form.elements.namedItem(key).value !== '' &&
      Number.isSafeInteger(result[key]) && result[key] >= (key.startsWith('center_') ? -1 : 0)) ? result : null;
  }

  function draw() {
    box.hidden = true;
    const region = values();
    if (!state?.native_size) {
      status.textContent = 'Waiting for camera dimensions…';
      return;
    }
    if (!region) {
      status.textContent = 'Enter integer dimensions ≥ 0 and centers ≥ −1.';
      return;
    }
    const bounds = regionBounds(state.native_size, region);
    const enabled = region.width !== 0 || region.height !== 0;
    status.textContent = 'Camera ' + state.native_size.join(' × ') + ' px · region ' +
      (bounds[2] - bounds[0]) + ' × ' + (bounds[3] - bounds[1]) + ' px at (' +
      bounds[0] + ', ' + bounds[1] + ')' + (dirty ? ' · preview draft (not applied)' : ' · applied') +
      (enabled ? '' : ' · filtering disabled') + (source === 'camera' ? '' : ' · camera source only');
    if (!toggle.checked || source !== 'camera' || bounds[2] <= bounds[0] || bounds[3] <= bounds[1]) return;
    const rect = previewRect(bounds, state.native_size, [video.clientWidth, video.clientHeight],
      [video.videoWidth || 960, video.videoHeight || 540]);
    for (const key of ['left', 'top', 'width', 'height']) box.style[key] = rect[key] + 'px';
    box.hidden = false;
  }

  function update(next, nextSource) {
    source = nextSource;
    if (!saving && next) {
      state = next;
      if (!dirty) for (const key of keys) form.elements.namedItem(key).value = next[key];
    }
    draw();
  }

  form.oninput = () => { dirty = true; draw(); };
  toggle.onchange = draw;
  form.onsubmit = async event => {
    event.preventDefault();
    const region = values();
    if (!region || saving) return;
    saving = true;
    for (const input of form.elements) input.disabled = true;
    document.getElementById('error').textContent = '';
    try {
      const response = await fetch('/api/sam3-roi', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(region)
      });
      const data = await response.json();
      if (!response.ok) throw Error(data.error || response.statusText);
      state = data;
      dirty = false;
    } catch (error) {
      document.getElementById('error').textContent = error.message;
    } finally {
      saving = false;
      for (const input of form.elements) input.disabled = false;
      update(state, source);
    }
  };
  document.getElementById('roi-reset').onclick = () => {
    if (saving) return;
    const defaults = {width: 0, height: 0, center_x: -1, center_y: -1};
    for (const key of keys) form.elements.namedItem(key).value = defaults[key];
    dirty = true;
    draw();
    form.requestSubmit();
  };
  new ResizeObserver(draw).observe(video);
  video.addEventListener('loadedmetadata', draw);
  video.addEventListener('resize', draw);
  globalThis.sam3Region = {update};
})();
