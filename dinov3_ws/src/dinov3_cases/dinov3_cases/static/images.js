const el = id => document.getElementById(id);
let selected = null,
  point = null,
  previewUrl = null,
  previewController = null,
  previewVersion = 0,
  loadVersion = 0,
  timer = null,
  busy = false;
let cameraLabel = '', cameraBboxUrl = null;
let knownLabels = new Set();
// Running Flask services may still have the previous HTML template cached.
if (el('camera-section') && !el('camera-capture-help')) {
  const help = document.createElement('p');
  help.id = 'camera-capture-help';
  help.setAttribute('role', 'status');
  el('camera-capture').after(help);
  el('camera-capture').setAttribute('aria-describedby', help.id);
}

if (el('camera-section') && !el('camera-bbox-preview')) {
  const button = document.createElement('button');
  button.id = 'camera-bbox-preview';
  button.textContent = 'Preview Bounding Boxes';
  const status = document.createElement('p');
  status.id = 'camera-bbox-status';
  status.textContent = 'Select a prompt to preview. The preview is a snapshot; saving captures a new frame.';
  status.setAttribute('role', 'status');
  const image = document.createElement('img');
  image.id = 'camera-bbox-image';
  image.alt = 'SAM3 bounding box preview';
  image.hidden = true;
  el('camera-status').after(button, status, image);
}

function message(text, error = false) {
  el('message').textContent = text;
  el('message').className = error ? 'error' : '';
}
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw Error(data.error || 'Request failed: ' + response.status);
  }
  return response;
}

function fileUrl(group, name) {
  return '/api/images/file/' + encodeURIComponent(group) + '/' + encodeURIComponent(name);
}

function buttons() {
  el('save').disabled = busy || !selected || !point || !previewUrl;
  el('delete').disabled = busy || !selected;
  el('upload-button').disabled = busy;
  if (el('camera-section')) {
    // Keep validation actionable; a disabled button cannot explain what is missing.
    el('camera-capture').disabled = busy;
    el('camera-bbox-preview').disabled = busy;
    el('camera-create').disabled = busy;
    el('camera-label').disabled = busy;
    el('camera-prompt').disabled = busy || !Array.from(el('camera-prompt').options).some(option => option.value);
    el('camera-capture-help').textContent = busy ? 'Processing, please wait…' : cameraLabel !== el('camera-label').value || !knownLabels.has(cameraLabel) ? 'Create or select a label first.' : !el('camera-prompt').value ? 'Select an SAM3 prompt first.' : 'Click to capture and save to ' + cameraLabel + '.';
  }
}

function clearPreview() {
  previewVersion++;
  if (previewController) previewController.abort();
  clearTimeout(timer);
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  el('preview').hidden = true;
  el('preview').removeAttribute('src');
  el('download').hidden = true;
  el('download').removeAttribute('href');
  buttons();
}

function clearSelection() {
  loadVersion++;
  selected = null;
  point = null;
  clearPreview();
  el('original').hidden = true;
  el('filename').textContent = 'Select an image from the gallery first.';
  el('sample-text').textContent = 'No background color selected';
  el('sample').style.background = 'transparent';
  buttons();
}
async function refresh(group = el('cases').value) {
  const data = await (await api('/api/images' + (group ? '?case=' + encodeURIComponent(group) : ''))).json();
  knownLabels = new Set(data.cases);
  if (el('camera-section')) syncCameraLabel(el('camera-label').value);
  el('cases').replaceChildren(new Option('Select a label', ''), ...data.cases.map(c => new Option(c, c)));
  el('cases').value = group;
  el('case-list').replaceChildren(...data.cases.map(c => new Option(c, c)));
  el('count').textContent = group ? data.images.length + ' images' : 'Select a label or upload images to get started';
  el('gallery').replaceChildren();
  for (const item of data.images) {
    const button = document.createElement('button');
    button.className = 'tile';
    button.setAttribute('aria-pressed', String(selected?.group === group && selected?.name === item.name));
    const image = document.createElement('img');
    image.src = fileUrl(group, item.name);
    image.loading = 'lazy';
    image.alt = item.name;
    image.className = 'checker';
    const name = document.createElement('span');
    name.textContent = item.name;
    button.append(image, name);
    button.onclick = () => choose(group, item.name).catch(e => message(e.message, true));
    el('gallery').append(button);
  }
}
async function choose(group, name) {
  clearSelection();
  const version = loadVersion;
  const image = new Image();
  image.src = fileUrl(group, name);
  await image.decode();
  if (version !== loadVersion) return;
  const canvas = el('original');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  canvas.getContext('2d').drawImage(image, 0, 0);
  canvas.hidden = false;
  selected = {
    group,
    name
  };
  el('filename').textContent = group + ' / ' + name + ' · ' + canvas.width + ' × ' + canvas.height;
  el('target-case').value = group;
  el('upload-case').value = group;
  if (el('camera-section')) syncCameraLabel(group);
  for (const tile of el('gallery').children) tile.setAttribute('aria-pressed', String(tile.querySelector('span').textContent === name));
  message('Click the background color you want to remove in the original image.');
  buttons();
}

function options() {
  return {
    x: point.x,
    y: point.y,
    hue: Number(el('hue').value),
    saturation: Number(el('saturation').value),
    value: Number(el('value').value),
    feather: Number(el('feather').value),
    connected: el('connected').checked
  };
}

function body() {
  return {
    case: selected.group,
    name: selected.name,
    options: options(),
    target_case: el('target-case').value || selected.group
  };
}

function schedulePreview() {
  clearPreview();
  if (!point) return;
  timer = setTimeout(() => preview(), 250);
}
async function preview() {
  if (!selected || !point) return;
  const version = previewVersion;
  previewController = new AbortController();
  message('Generating background removal preview…');
  try {
    const response = await api('/api/images/preview', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body()),
      signal: previewController.signal
    });
    const blob = await response.blob();
    if (version !== previewVersion) return;
    previewUrl = URL.createObjectURL(blob);
    el('preview').src = previewUrl;
    el('preview').hidden = false;
    el('download').href = previewUrl;
    el('download').hidden = false;
    message('Preview updated. Adjust the tolerances or save a new image.');
    buttons();
  } catch (e) {
    if (e.name !== 'AbortError' && version === previewVersion) message(e.message, true);
  }
}
el('original').onclick = event => {
  if (!selected || busy) return;
  const canvas = el('original'),
    rect = canvas.getBoundingClientRect();
  const x = Math.min(canvas.width - 1, Math.max(0, Math.floor((event.clientX - rect.left) * canvas.width / rect.width))),
    y = Math.min(canvas.height - 1, Math.max(0, Math.floor((event.clientY - rect.top) * canvas.height / rect.height)));
  const rgba = canvas.getContext('2d').getImageData(x, y, 1, 1).data;
  if (rgba[3] === 0) {
    message('This point is already transparent. Click a colored background region.', true);
    return;
  }
  point = {
    x,
    y
  };
  el('sample').style.background = 'rgb(' + Array.from(rgba).slice(0, 3).join(',') + ')';
  el('sample-text').textContent = 'RGB (' + Array.from(rgba).slice(0, 3).join(', ') + ') · coordinates (' + x + ', ' + y + ')';
  schedulePreview();
};
for (const name of ['hue', 'saturation', 'value', 'feather']) el(name).oninput = () => {
  el(name + '-out').textContent = el(name).value;
  schedulePreview();
};
el('connected').onchange = schedulePreview;
el('cases').onchange = () => {
  clearSelection();
  el('upload-case').value = el('cases').value;
  if (el('camera-section')) syncCameraLabel(el('cases').value);
  refresh().catch(e => message(e.message, true));
};
el('reload').onclick = () => {
  clearSelection();
  refresh().catch(e => message(e.message, true));
};
el('upload').onsubmit = async event => {
  event.preventDefault();
  if (busy) return;
  busy = true;
  buttons();
  try {
    const form = new FormData(event.target),
      group = form.get('case');
    await api('/api/images/upload', {
      method: 'POST',
      body: form
    });
    clearSelection();
    await refresh(group);
    event.target.querySelector('input[type=file]').value = '';
    message('Images uploaded to ' + group + '.');
  } catch (e) {
    message(e.message, true);
  } finally {
    busy = false;
    buttons();
  }
};
el('save').onclick = async () => {
  if (busy || !point || !previewUrl) return;
  const data = body();
  if (!/^[A-Za-z0-9_-]{1,80}$/.test(data.target_case)) {
    message('Labels must contain 1–80 letters, numbers, underscores, or hyphens.', true);
    return;
  }
  busy = true;
  buttons();
  try {
    const result = await (await api('/api/images/cutout', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    })).json();
    await refresh(result.case);
    await choose(result.case, result.name);
    message('Transparent PNG created. The original is preserved in ' + data.case+'.');
  } catch (e) {
    message(e.message, true);
  } finally {
    busy = false;
    buttons();
  }
};
el('delete').onclick = async () => {
  if (busy || !selected) return;
  const item = {
    ...selected
  };
  if (!confirm('Permanently delete ' + item.group + ' / ' + item.name + '? This cannot be undone.')) return;
  busy = true;
  buttons();
  try {
    await api(fileUrl(item.group, item.name), {
      method: 'DELETE'
    });
    clearSelection();
    await refresh(item.group);
    message('Image deleted.');
  } catch (e) {
    message(e.message, true);
  } finally {
    busy = false;
    buttons();
  }
};
refresh().catch(e => message(e.message, true));

function syncCameraLabel(group) {
  el('camera-label').value = group;
  cameraLabel = knownLabels.has(group) ? group : '';
  el('camera-label-status').textContent = cameraLabel ? 'Captured images will be saved to: ' + cameraLabel : 'For a new label, click "Create / Select Label" first.';
  buttons();
}

if (el('camera-section')) {
  el('camera-label').oninput = () => syncCameraLabel(el('camera-label').value);
  function clearCameraBbox() {
    if (cameraBboxUrl) URL.revokeObjectURL(cameraBboxUrl);
    cameraBboxUrl = null;
    el('camera-bbox-image').hidden = true;
    el('camera-bbox-image').removeAttribute('src');
  }
  el('camera-prompt').onchange = () => {
    clearCameraBbox();
    el('camera-bbox-status').textContent = 'Prompt updated. Preview the bounding boxes again.';
    buttons();
  };
  el('camera-bbox-preview').onclick = async () => {
    if (busy) return;
    const prompt = el('camera-prompt').value;
    if (!prompt) {
      el('camera-bbox-status').textContent = 'Select an SAM3 prompt first.';
      el('camera-prompt').focus();
      return;
    }
    busy = true;
    buttons();
    clearCameraBbox();
    el('camera-bbox-status').textContent = 'Detecting objects in the camera frame…';
    try {
      const response = await api('/api/images/camera-preview', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt})
      });
      cameraBboxUrl = URL.createObjectURL(await response.blob());
      el('camera-bbox-image').src = cameraBboxUrl;
      el('camera-bbox-image').hidden = false;
      const count = Number(response.headers.get('X-Detection-Count'));
      el('camera-bbox-status').textContent = count ? 'Detected ' + count + ' bounding boxes · ' + prompt : 'No objects matching the prompt were detected.';
    } catch (e) {
      el('camera-bbox-status').textContent = e.message;
      message(e.message, true);
    } finally {
      busy = false;
      buttons();
    }
  };
  el('camera-label-form').onsubmit = async event => {
    event.preventDefault();
    if (busy) return;
    busy = true;
    buttons();
    try {
      const result = await (await api('/api/images/labels', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({case: el('camera-label').value})
      })).json();
      knownLabels.add(result.case);
      syncCameraLabel(result.case);
      clearSelection();
      await refresh(cameraLabel);
      message('Label ready. Select a prompt, then capture.');
    } catch (e) {
      message(e.message, true);
    } finally {
      busy = false;
      buttons();
    }
  };
  async function cameraStatus() {
    try {
      const status = await (await api('/api/status')).json();
      const prompts = status.grounding_offline_prompts || [];
      const previous = el('camera-prompt').value;
      const current = Array.from(el('camera-prompt').options).map(option => option.value).filter(Boolean);
      if (JSON.stringify(current) !== JSON.stringify(prompts)) {
        el('camera-prompt').replaceChildren(new Option('Select a prompt', ''), ...prompts.map(p => new Option(p, p)));
        el('camera-prompt').value = prompts.includes(previous) ? previous : '';
        if (previous !== el('camera-prompt').value) clearCameraBbox();
      }
      el('camera-status').textContent = !status.camera_connected ? 'Camera disconnected or frame is stale' : !prompts.length ? 'SAM3 is loading…' : 'Camera connected. Ready to capture.';
    } catch (e) {
      el('camera-status').textContent = e.message;
    } finally {
      buttons();
      setTimeout(cameraStatus, 3000);
    }
  }
  el('camera-capture').onclick = async () => {
    if (busy) return;
    syncCameraLabel(el('camera-label').value);
    if (!cameraLabel) {
      el('camera-capture-help').textContent = 'Create or select a label before capturing images.';
      message('Create or select a label before capturing images.', true);
      el('camera-label').focus();
      return;
    }
    if (!el('camera-prompt').value) {
      el('camera-capture-help').textContent = 'Select an SAM3 prompt first. If no options are available, wait for the model to load.';
      message(el('camera-capture-help').textContent, true);
      el('camera-prompt').focus();
      return;
    }
    busy = true;
    buttons();
    message('Capturing the camera frame and running SAM3 detection…');
    try {
      const result = await (await api('/api/images/capture', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({case: cameraLabel, prompt: el('camera-prompt').value})
      })).json();
      if (result.names.length) {
        await refresh(result.case);
        await choose(result.case, result.names[0]);
        message('Saved ' + result.names.length + ' bounding box images to ' + result.case + '. Click the background in the original image to remove it.');
      } else {
        message('No objects matching the prompt were detected. No images were saved. Adjust the position or prompt and try again.');
      }
    } catch (e) {
      el('camera-status').textContent = e.message;
      message(e.message, true);
    } finally {
      busy = false;
      buttons();
    }
  };
  buttons();
  cameraStatus();
}
