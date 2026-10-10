const el = id => document.getElementById(id);
let selected = null,
  point = null,
  previewUrl = null,
  previewController = null,
  previewVersion = 0,
  loadVersion = 0,
  timer = null,
  busy = false;

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
