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
  if (!groundingLoaded && d.grounding) {
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
    row.textContent = (item.track_id == null ? '' : 'ID ' + item.track_id + ' · ') + item.label + ' · ' + item.detection_label + ' · det ' + item.detection_score.toFixed(2) + ' · DINO ' + (item.score === null ? '—' : item.score.toFixed(3)) + ' · ' + item.reason + (item.label === 'unknown' && item.candidate ? ' · closest match ' + item.candidate : '') + (item.margin === null ? '' : ' · Lab gap ' + item.margin.toFixed(3));
    if (item.track_status) {
      row.textContent += ' · track: ' + item.track_status;
      if (Number.isFinite(item.sam3_age_seconds)) row.textContent += ' · SAM3 ' + item.sam3_age_seconds.toFixed(1) + 's ago';
      if (item.sam3_misses) row.textContent += ' · missed ' + item.sam3_misses + ' refresh(es)';
    }
    if (item.bbox3d_status) {
      row.textContent += ' · 3D: ' + item.bbox3d_status;
      if (Number.isFinite(item.bbox3d_age_seconds)) row.textContent += ' · depth ' + item.bbox3d_age_seconds.toFixed(1) + 's ago';
    }
    list.append(row);
  }
}
async function refreshReferences(group, revision, stride) {
  const key = JSON.stringify([group, revision, stride]);
  if (key === referenceKey) return;
  referenceKey = key;
  const version = ++referenceVersion;
  el('reference-gallery').replaceChildren();
  el('reference-status').textContent = group ? 'Loading ' + group + ' reference images…' : 'Select a case to view its reference images.';
  if (!group) return;
  try {
    const r = await fetch('/api/references?case=' + encodeURIComponent(group));
    const d = await r.json();
    if (version !== referenceVersion) return;
    if (!r.ok) throw Error(d.error || r.statusText);
    el('reference-status').textContent = d.case+' · ' + d.images.length + ' reference images · patch size ' + d.patch_size + ' × ' + d.patch_size + ' pixels';
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
        image.alt = item.name + ' patch grid';
        image.src = '/api/references/grid/' + encodeURIComponent(group) + '/' + encodeURIComponent(item.name) + '?v=' + encodeURIComponent(item.version);
        image.onerror = () => {
          if (version === referenceVersion) caption.textContent += ' · Failed to load patch grid';
        };
        card.append(image);
        const details = document.createElement('p');
        details.textContent = 'Original ' + item.original_size.join(' × ') + ' → model input ' + item.size.join(' × ') + ' · ' + item.rows + ' rows × ' + item.columns + ' columns (' + item.rows * item.columns + ' patches)';
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
  sam3Region.update(d.sam3_roi, d.source);
  sam3Intervals.update(d.sam3_inference_events);
  el('model-timings').textContent = [
    ['nvdcf_ms', 'NvDCF tracking'], ['dinov3_ms', 'DINOv3 recognition'], ['sam3_ms', 'SAM3 detection']
  ].map(([key, label]) => {
    const timing = d.model_timings?.[key];
    const duration = Number.isFinite(timing) ? timing.toFixed(1) + ' ms' : '—';
    return label + ': ' + duration;
  }).join(' | ');
  el('status').textContent = d.status + ' | source: ' + d.source + (d.overlay_age === null ? '' : ' | result age: ' + d.overlay_age.toFixed(1) + 's');
  el('camera-status').textContent = d.video_error || (d.camera_connected ? 'Live · ' + d.camera_topic : 'No recent camera frames · ' + d.camera_topic + ' · start the Orbbec driver');
  if (d.camera_connected && Number.isFinite(d.camera_fps)) el('camera-status').textContent += ' · ' + d.camera_fps.toFixed(1) + ' input FPS';
  el('overlay').style.opacity = d.source === 'upload' || d.camera_connected ? '1' : '0.3';
  const s = el('cases');
  if (JSON.stringify(d.cases) !== s.dataset.list) {
    const value = s.value || d.selected;
    s.replaceChildren(new Option('Select case', ''), ...d.cases.map(c => new Option(c, c)));
    s.value = value;
    s.dataset.list = JSON.stringify(d.cases);
  }
  refreshReferences(s.value || d.selected, d.reference_revision, d.reference_stride);
  if (d.performance && d.performance.result_fps) el('status').textContent += ' | ' + d.performance.result_fps.toFixed(1) + ' processed RGB FPS (3s avg) | callback→result ' + (d.performance.callback_to_result_ms === null ? '—' : d.performance.callback_to_result_ms.toFixed(0) + ' ms');
}

const objectHelp = {
  heatmap: 'Local patch similarity overlaid on the original ROI. Fixed scale: 0 (dark blue) → 0.5 (green) → 1 (red). Negative values appear dark blue.',
  crop: 'Object crop detected by SAM3 in original image coordinates, preserving its aspect ratio.',
  grid: 'Actual DINOv3 model input. Yellow lines mark patch boundaries; the area outside the blue box is right/bottom padding.',
  patches: 'Maximum cosine similarity between each ROI patch and all patches of the best reference image. Shows the full model grid, including padding.',
  similarity: 'Similarity map with padding removed, mapped back to the ROI. Fixed scale of 0–1; negative values appear dark blue.',
  reference: 'Reference image with the highest overall product feature similarity, using the same resizing and padding as recognition.'
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
  select.replaceChildren(new Option('Automatic best reference', ''), ...cases.map(name => new Option(name, name)));
  select.value = cases.includes(selected) ? selected : '';
  select.dataset.list = list;
}
async function refreshObjects() {
  if (objectBusy || el('object-section').hidden) return;
  if (!el('object-enabled').checked) {
    el('object-status').textContent = 'Diagnostic images are disabled. Check "Enable diagnostic images" to show object heatmaps.';
    return;
  }
  objectBusy = true;
  const view = el('object-view').value;
  el('object-help').textContent = objectHelp[view];
  const referenceCase = el('object-reference').value;
  const comparisonQuery = referenceCase ? '?reference_case=' + encodeURIComponent(referenceCase) : '';
  if (referenceCase) el('object-help').textContent = 'Comparing all reference images for label ' + referenceCase + '. ' + objectHelp[view].replace('all patches of the best reference image', 'all patches of all reference images under the selected label');
  el('object-scale').hidden = !['heatmap', 'patches', 'similarity'].includes(view);
  try {
    const response = await fetch('/api/object_visualizations' + comparisonQuery);
    const data = await response.json();
    if (data.ready || !response.ok && data.reference_cases) updateObjectReferences(data.reference_cases);
    if (!response.ok) throw Error(data.error || 'Unable to load object visualizations');
    const key = JSON.stringify([data.revision, view, referenceCase]);
    if (key === objectKey) return;
    if (!data.objects.length) {
      clearObjectUrls();
      el('object-gallery').replaceChildren();
      el('object-status').textContent = 'No object results yet. Provide a target image and check the text prompt and detection threshold.';
      objectKey = key;
      return;
    }
    const images = await Promise.all(data.objects.map(async item => {
      if (!item.views.includes(view)) return null;
      const image = await fetch('/debug/objects/' + data.revision + '/' + item.index + '/' + view + '.jpg' + comparisonQuery);
      if (image.status === 410) throw Error('Results have changed. Fetching the latest visualizations…');
      if (!image.ok) throw Error('Object ' + (item.index + 1) + ' visualization failed to load');
      return image.blob();
    }));
    if (!el('object-enabled').checked || view !== el('object-view').value || referenceCase !== el('object-reference').value) return;
    clearObjectUrls();
    const gallery = el('object-gallery');
    gallery.replaceChildren();
    const similarity = value => Number.isFinite(value) ? value.toFixed(3) : '—';
    data.objects.forEach((item, index) => {
      const card = document.createElement('figure');
      card.className = 'reference-card';
      const title = document.createElement('figcaption');
      title.textContent = (item.track_id == null ? '#' + (item.index + 1) : 'ID ' + item.track_id) + ' · ' + item.label + ' · ' + item.detection_label;
      card.append(title);
      const detail = document.createElement('p');
      detail.textContent = 'bbox [' + item.bbox.join(', ') + '] · ROI ' + item.roi_size.join(' × ') + ' → input ' + item.input_size.join(' × ') + ' · ' + item.patch_grid.join(' × ') + ' patches';
      card.append(detail);
      const score = document.createElement('p');
      score.textContent = 'det ' + item.detection_score.toFixed(3) + ' · DINO ' + similarity(item.dino_score) + ' · Lab ' + similarity(item.lab_score) + ' · ' + item.reason;
      card.append(score);
      const comparison = document.createElement('p');
      comparison.textContent = item.comparison_reference ? 'Comparing ' + item.comparison_case + ' · DINO ' + similarity(item.comparison_dino_score) + ' · Lab ' + similarity(item.comparison_lab_score) + ' · ' + item.comparison_reference_count + ' reference images · best ' + item.comparison_reference.name : 'No reference images available';
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
        empty.textContent = 'No reference gallery is available to calculate similarity. Switch to the original crop or patch grid view.';
        card.append(empty);
      }
      if (item.heatmap_available) {
        const range = document.createElement('small');
        range.textContent = 'ROI local cosine range ' + item.similarity_min.toFixed(3) + '–' + item.similarity_max.toFixed(3);
        card.append(range);
      }
      gallery.append(card);
    });
    el('object-status').textContent = data.objects.length + ' objects · ' + (referenceCase ? 'Comparing ' + referenceCase : 'Automatic best reference') + ' · recognition snapshot #' + data.revision;
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
  el('object-status').textContent = 'Updating label heatmaps…';
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
