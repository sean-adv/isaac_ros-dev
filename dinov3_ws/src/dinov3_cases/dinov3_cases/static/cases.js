const el = id => document.getElementById(id);
let referenceKey = '',
  referenceVersion = 0;
let groundingLoaded = false;
el('grounding-form').onsubmit = e => {
  e.preventDefault();
  const values = Object.fromEntries(new FormData(e.target));
  for (const key of Object.keys(values)) {
    if (key !== 'prompt') values[key] = Number(values[key]);
  }
  send('/api/grounding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(values)
  });
};

function updateGrounding(d) {
  if (!groundingLoaded && d.grounding && d.grounding_offline_prompts?.length) {
    el('prompt').replaceChildren(...d.grounding_offline_prompts.map(prompt => {
      const option = document.createElement('option');
      option.value = option.textContent = prompt;
      return option;
    }));
    for (const [key, value] of Object.entries(d.grounding)) el('grounding-form').elements.namedItem(key).value = value;
    groundingLoaded = true;
  }
}
async function refreshDetections() {
  const r = await fetch('/api/detections');
  if (!r.ok) return;
  const data = await r.json();
  const list = el('recognition-list');
  list.replaceChildren();
  for (const item of data.detections) {
    if (!item.label) continue;
    const row = document.createElement('p');
    row.textContent = (item.track_id == null ? '' : 'ID ' + item.track_id + ' · ') + item.label + ' · ' + item.detection_label + ' · det ' + item.detection_score.toFixed(2) + ' · sim ' + (item.score === null ? '—' : item.score.toFixed(3)) + ' · ' + item.reason + (item.label === 'unknown' && item.candidate ? ' · 最接近 ' + item.candidate : '') + (item.margin === null ? '' : ' · 差距 ' + item.margin.toFixed(3));
    list.append(row);
  }
}
async function refreshReferences(group, revision, stride) {
  const key = JSON.stringify([group, revision, stride]);
  if (key === referenceKey) return;
  referenceKey = key;
  const version = ++referenceVersion;
  el('reference-gallery').replaceChildren();
  el('reference-status').textContent = group ? '載入 ' + group + ' 的範例照片…' : '請選擇 case 以查看每張範例照片。';
  if (!group) return;
  try {
    const r = await fetch('/api/references?case=' + encodeURIComponent(group));
    const d = await r.json();
    if (version !== referenceVersion) return;
    if (!r.ok) throw Error(d.error || r.statusText);
    el('reference-status').textContent = d.case+' · ' + d.images.length + ' 張範例照片 · 每格 ' + d.patch_size + ' × ' + d.patch_size + ' 像素';
    for (const item of d.images) {
      const card = document.createElement('figure');
      card.className = 'reference-card';
      const caption = document.createElement('figcaption');
      caption.textContent = item.name;
      if (item.error) {
        const error = document.createElement('p');
        error.textContent = item.error;
        card.append(error);
      } else {
        const image = document.createElement('img');
        image.loading = 'lazy';
        image.alt = item.name + ' 的 patch 分割示意圖';
        image.src = '/api/references/grid/' + encodeURIComponent(group) + '/' + encodeURIComponent(item.name) + '?v=' + encodeURIComponent(item.version);
        image.onerror = () => {
          if (version === referenceVersion) caption.textContent += ' · 示意圖載入失敗';
        };
        card.append(image);
        const details = document.createElement('p');
        details.textContent = '原圖 ' + item.original_size.join(' × ') + ' → 模型輸入 ' + item.size.join(' × ') + ' · ' + item.rows + ' 列 × ' + item.columns + ' 欄（' + item.rows * item.columns + ' patches）';
        card.append(details);
      }
      card.append(caption);
      el('reference-gallery').append(card);
    }
  } catch (e) {
    if (version === referenceVersion) {
      el('reference-status').textContent = e.message;
      referenceKey = '';
    }
  }
}
el('cases').onchange = () => {
  referenceKey = '';
  refresh().catch(() => {});
};
async function send(url, options) {
  el('error').textContent = '';
  try {
    const r = await fetch(url, options);
    const d = await r.json();
    if (!r.ok) throw Error(d.error || r.statusText);
    await refresh();
  } catch (e) {
    el('error').textContent = e.message;
  }
}
el('upload').onsubmit = e => {
  e.preventDefault();
  send('/api/cases', {
    method: 'POST',
    body: new FormData(e.target)
  });
};
el('select').onclick = () => send('/api/select', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    case: el('cases').value
  })
});
el('target').onsubmit = e => {
  e.preventDefault();
  let f = new FormData(e.target);
  f.set('source', 'upload');
  send('/api/source', {
    method: 'POST',
    body: f
  });
};
el('camera').onclick = () => {
  let f = new FormData();
  f.set('source', 'camera');
  send('/api/source', {
    method: 'POST',
    body: f
  });
};
async function refresh() {
  const r = await fetch('/api/status');
  const d = await r.json();
  updateGrounding(d);
  el('status').textContent = d.status + ' | source: ' + d.source + (d.overlay_age === null ? '' : ' | result age: ' + d.overlay_age.toFixed(1) + 's');
  el('camera-status').textContent = d.preview_error || (d.camera_connected ? 'Live · ' + d.camera_topic : 'No recent camera frames · ' + d.camera_topic + ' · start the Orbbec driver');
  el('preview').style.opacity = d.camera_connected ? '1' : '0.3';
  const s = el('cases');
  if (JSON.stringify(d.cases) !== s.dataset.list) {
    const value = s.value || d.selected;
    s.replaceChildren(new Option('Select case', ''), ...d.cases.map(c => new Option(c, c)));
    s.value = value;
    s.dataset.list = JSON.stringify(d.cases);
  }
  refreshReferences(s.value || d.selected, d.reference_revision, d.reference_stride);
  el('overlay').style.display = d.overlay_ready ? '' : 'none';
  if (d.performance && d.performance.result_fps) el('status').textContent += ' | ' + d.performance.result_fps.toFixed(1) + ' FPS | callback→result ' + (d.performance.callback_to_result_ms === null ? '—' : d.performance.callback_to_result_ms.toFixed(0) + ' ms');
}

const objectHelp = {
  heatmap: 'ROI 原圖疊加局部 patch 相似度。固定色階：0（深藍）→ 0.5（綠）→ 1（紅）；負值顯示深藍。',
  crop: 'OmDet-Turbo-Tiny 在原圖座標中找到的物體裁切，保持原長寬比。',
  grid: '實際 DINOv3 模型輸入：黃色為 patch 邊界，藍框外為右側／底部補邊。',
  patches: '每個 ROI patch 與最佳參考圖所有 patch 的最高 cosine similarity；顯示完整模型網格（含補邊）。',
  similarity: '移除補邊並映射回 ROI 的純相似度圖。固定色階 0–1；負值顯示深藍。',
  reference: '整體商品特徵比對分數最高的那張參考圖，使用與辨識相同的縮放與補邊。'
};
let objectBusy = false,
  objectKey = '',
  objectUrls = [];

function clearObjectUrls() {
  for (const url of objectUrls) URL.revokeObjectURL(url);
  objectUrls = [];
}

function updateObjectReferences(cases) {
  const select = el('object-reference');
  const list = JSON.stringify(cases);
  if (select.dataset.list === list) return;
  const selected = select.value;
  select.replaceChildren(new Option('自動最佳參考圖', ''), ...cases.map(name => new Option(name, name)));
  select.value = cases.includes(selected) ? selected : '';
  select.dataset.list = list;
}
async function refreshObjects() {
  if (objectBusy || el('object-section').hidden) return;
  if (!el('object-enabled').checked) {
    el('object-status').textContent = '診斷圖片已關閉；勾選「開啟診斷圖片」以顯示物體熱圖。';
    return;
  }
  objectBusy = true;
  const view = el('object-view').value;
  el('object-help').textContent = objectHelp[view];
  const referenceCase = el('object-reference').value;
  const comparisonQuery = referenceCase ? '?reference_case=' + encodeURIComponent(referenceCase) : '';
  if (referenceCase) el('object-help').textContent = '比對標籤 ' + referenceCase + ' 的全部參考圖片。' + objectHelp[view].replace('最佳參考圖所有 patch', '指定標籤全部參考圖的 patch');
  el('object-scale').hidden = !['heatmap', 'patches', 'similarity'].includes(view);
  try {
    const response = await fetch('/api/object_visualizations' + comparisonQuery);
    const data = await response.json();
    if (data.ready || !response.ok && data.reference_cases) updateObjectReferences(data.reference_cases);
    if (!response.ok) throw Error(data.error || '無法讀取物體視覺化');
    const key = JSON.stringify([data.revision, view, referenceCase]);
    if (key === objectKey) return;
    if (!data.objects.length) {
      clearObjectUrls();
      el('object-gallery').replaceChildren();
      el('object-status').textContent = '目前沒有物體結果；請提供目標影像並檢查文字提示與偵測門檻。';
      objectKey = key;
      return;
    }
    const images = await Promise.all(data.objects.map(async item => {
      if (!item.views.includes(view)) return null;
      const image = await fetch('/debug/objects/' + data.revision + '/' + item.index + '/' + view + '.jpg' + comparisonQuery);
      if (image.status === 410) throw Error('結果已更新，正在取得最新視覺化…');
      if (!image.ok) throw Error('物體 ' + (item.index + 1) + ' 視覺化讀取失敗');
      return image.blob();
    }));
    if (!el('object-enabled').checked || view !== el('object-view').value || referenceCase !== el('object-reference').value) return;
    clearObjectUrls();
    const gallery = el('object-gallery');
    gallery.replaceChildren();
    data.objects.forEach((item, index) => {
      const card = document.createElement('figure');
      card.className = 'reference-card';
      const title = document.createElement('figcaption');
      title.textContent = (item.track_id == null ? '#' + (item.index + 1) : 'ID ' + item.track_id) + ' · ' + item.label + ' · ' + item.detection_label;
      card.append(title);
      const detail = document.createElement('p');
      detail.textContent = 'bbox [' + item.bbox.join(', ') + '] · ROI ' + item.roi_size.join(' × ') + ' → 輸入 ' + item.input_size.join(' × ') + ' · ' + item.patch_grid.join(' × ') + ' patches';
      card.append(detail);
      const score = document.createElement('p');
      score.textContent = 'det ' + item.detection_score.toFixed(3) + ' · 商品 sim ' + (item.score === null ? '—' : item.score.toFixed(3)) + ' · ' + item.reason;
      card.append(score);
      const comparison = document.createElement('p');
      comparison.textContent = item.comparison_reference ? '比對 ' + item.comparison_case + ' · 標籤 sim ' + item.comparison_score.toFixed(3) + ' · ' + item.comparison_reference_count + ' 張參考圖 · 最佳 ' + item.comparison_reference.name : '沒有可用參考圖';
      card.append(comparison);
      if (images[index]) {
        const image = document.createElement('img');
        image.alt = title.textContent + ' ' + objectHelp[view];
        const url = URL.createObjectURL(images[index]);
        objectUrls.push(url);
        image.src = url;
        card.append(image);
      } else {
        const empty = document.createElement('p');
        empty.textContent = '沒有參考圖庫，無法計算相似度；可切換裁切原圖或 patch 網格。';
        card.append(empty);
      }
      if (item.heatmap_available) {
        const range = document.createElement('small');
        range.textContent = 'ROI 局部 cosine 範圍 ' + item.similarity_min.toFixed(3) + '–' + item.similarity_max.toFixed(3);
        card.append(range);
      }
      gallery.append(card);
    });
    el('object-status').textContent = data.objects.length + ' 個物體 · ' + (referenceCase ? '比對 ' + referenceCase : '自動最佳參考圖') + ' · 同一幀結果 #' + data.revision;
    objectKey = key;
  } catch (error) {
    el('object-status').textContent = error.message;
  } finally {
    objectBusy = false;
  }
}
el('object-view').onchange = () => {
  objectKey = '';
  refreshObjects();
};
el('object-reference').onchange = () => {
  objectKey = '';
  clearObjectUrls();
  el('object-gallery').replaceChildren();
  el('object-status').textContent = '正在更新標籤熱圖…';
  refreshObjects();
};
el('object-enabled').onchange = () => {
  objectKey = '';
  if (!el('object-enabled').checked) {
    clearObjectUrls();
    el('object-gallery').replaceChildren();
  }
  refreshObjects();
};
setInterval(refreshObjects, 1000);
async function poll() {
  try {
    await refresh();
    await refreshDetections();
  } catch (e) {
    el('status').textContent = 'Connection lost: ' + e.message;
  }
  setTimeout(poll, 250);
}
poll();
