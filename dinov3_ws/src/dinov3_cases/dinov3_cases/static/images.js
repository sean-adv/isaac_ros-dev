const el = id => document.getElementById(id);
let selected = null,
  point = null,
  previewUrl = null,
  previewController = null,
  previewVersion = 0,
  loadVersion = 0,
  timer = null,
  busy = false;
let cameraLabel = '';
let knownLabels = new Set();
// Running Flask services may still have the previous HTML template cached.
if (el('camera-section') && !el('camera-capture-help')) {
  const help = document.createElement('p');
  help.id = 'camera-capture-help';
  help.setAttribute('role', 'status');
  el('camera-capture').after(help);
  el('camera-capture').setAttribute('aria-describedby', help.id);
}

function message(text, error = false) {
  el('message').textContent = text;
  el('message').className = error ? 'error' : '';
}
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw Error(data.error || '請求失敗：' + response.status);
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
    el('camera-create').disabled = busy;
    el('camera-label').disabled = busy;
    el('camera-prompt').disabled = busy || !Array.from(el('camera-prompt').options).some(option => option.value);
    el('camera-capture-help').textContent = busy ? '處理中，請稍候…' : cameraLabel !== el('camera-label').value || !knownLabels.has(cameraLabel) ? '請先建立或選擇 label。' : !el('camera-prompt').value ? '請先選擇 OmDet prompt。' : '按下按鈕即可擷取並儲存至 ' + cameraLabel + '。';
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
  el('filename').textContent = '先從左側選擇圖片。';
  el('sample-text').textContent = '尚未選擇背景色';
  el('sample').style.background = 'transparent';
  buttons();
}
async function refresh(group = el('cases').value) {
  const data = await (await api('/api/images' + (group ? '?case=' + encodeURIComponent(group) : ''))).json();
  knownLabels = new Set(data.cases);
  if (el('camera-section')) syncCameraLabel(el('camera-label').value);
  el('cases').replaceChildren(new Option('請選擇標籤', ''), ...data.cases.map(c => new Option(c, c)));
  el('cases').value = group;
  el('case-list').replaceChildren(...data.cases.map(c => new Option(c, c)));
  el('count').textContent = group ? data.images.length + ' 張圖片' : '選擇標籤或上傳圖片以開始';
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
  message('請在原圖上點選要去除的背景色。');
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
  message('正在產生去背預覽…');
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
    message('預覽已更新。可調整容差，或儲存新圖片。');
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
    message('此位置已透明，請點選有顏色的背景。', true);
    return;
  }
  point = {
    x,
    y
  };
  el('sample').style.background = 'rgb(' + Array.from(rgba).slice(0, 3).join(',') + ')';
  el('sample-text').textContent = 'RGB (' + Array.from(rgba).slice(0, 3).join(', ') + ') · 座標 (' + x + ', ' + y + ')';
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
    message('圖片已上傳至 ' + group + '。');
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
    message('標籤限 1–80 個英文字母、數字、底線或連字號。', true);
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
    message('已建立透明 PNG，原圖保留於 ' + data.case+'。');
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
  if (!confirm('確定永久刪除 ' + item.group + ' / ' + item.name + '？此操作無法復原。')) return;
  busy = true;
  buttons();
  try {
    await api(fileUrl(item.group, item.name), {
      method: 'DELETE'
    });
    clearSelection();
    await refresh(item.group);
    message('圖片已刪除。');
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
  el('camera-label-status').textContent = cameraLabel ? '擷取圖片將存入：' + cameraLabel : '新 label 請先按「建立／選擇 label」';
  buttons();
}

if (el('camera-section')) {
  el('camera-label').oninput = () => syncCameraLabel(el('camera-label').value);
  el('camera-prompt').onchange = buttons;
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
      message('標籤已就緒，請選擇 prompt 後擷取。');
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
        el('camera-prompt').replaceChildren(new Option('請選擇 prompt', ''), ...prompts.map(p => new Option(p, p)));
        el('camera-prompt').value = prompts.includes(previous) ? previous : '';
      }
      el('camera-status').textContent = !status.camera_connected ? '相機未連線或畫面已過期' : !prompts.length ? 'OmDet 載入中…' : '相機已連線，可手動擷取';
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
      el('camera-capture-help').textContent = '請先建立或選擇 label，再擷取圖片。';
      message('請先建立或選擇 label，再擷取圖片。', true);
      el('camera-label').focus();
      return;
    }
    if (!el('camera-prompt').value) {
      el('camera-capture-help').textContent = '請先選擇 OmDet prompt；若無選項，請等候模型載入。';
      message(el('camera-capture-help').textContent, true);
      el('camera-prompt').focus();
      return;
    }
    busy = true;
    buttons();
    message('正在擷取相機畫面並交給 OmDet 偵測…');
    try {
      const result = await (await api('/api/images/capture', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({case: cameraLabel, prompt: el('camera-prompt').value})
      })).json();
      if (result.names.length) {
        await refresh(result.case);
        await choose(result.case, result.names[0]);
        message('已儲存 ' + result.names.length + ' 張 bbox 圖片至 ' + result.case + '。可在原圖點選背景進行去背。');
      } else {
        message('此畫面未偵測到符合 prompt 的物件，未新增圖片。請調整位置或 prompt 後重試。');
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
