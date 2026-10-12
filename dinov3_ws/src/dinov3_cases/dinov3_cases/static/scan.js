/* No front-end framework or 3D dependency; bounded XYZ previews use Canvas. */
const scanEl = id => document.getElementById(id);
const scanStages = {
  segmentation:'② 深度 ROI mask', cloud:'③ RGB-D 點雲', preprocess:'④ 深度清理',
  geometry:'⑤ PCA／表面法向', candidates:'⑥ 抓取候選', scoring:'⑦ 接觸評分',
  collision:'⑧ 碰撞／未知區域', confidence:'⑨ 信心判斷', ready:'⑩ Grasp pose'
};
const scanStates = {...scanStages, rescan:'信心不足，等待補掃', moving:'相機移動中',
  fusion:'多視角融合中', failed:'掃描失敗'};
let scanObjects = [], scanRevision = null, scanSession = null, scanResult = null;
let cloudPoints = [], cloudCandidates = [], cloudPose = null;
let yaw = 0.35, pitch = -0.2, zoom = 1, dragging = null;

function updateScanObjects(data) {
  const select = scanEl('scan-object');
  const selected = select.value;
  scanObjects = data.detections;
  scanRevision = data.revision;
  select.replaceChildren(...scanObjects.map((item, index) => {
    const option = new Option(`#${index+1} ${item.label} · ${item.scan?.point_count || 0} 點`, String(index));
    option.disabled = item.label === 'unknown' || !item.scan?.points?.length;
    return option;
  }));
  if (!select.options.length) select.add(new Option('等待 RGB-D…', ''));
  if ([...select.options].some(o => o.value === selected && !o.disabled)) select.value = selected;
  else select.value = [...select.options].find(o => !o.disabled)?.value ?? '';
  scanEl('scan-start').disabled = !scanObjects[Number(select.value)]?.scan?.points?.length || select.value === '';
  if (!scanSession) showScanPreview();
}
function showScanPreview() {
  const selected = scanEl('scan-object').value;
  const item = selected === '' ? null : scanObjects[Number(selected)];
  renderScan(item?.scan ? {...item.scan, state:item.scan.stage, label:item.label,
    view_count:1, reason:item.pose_status === 'ok' ? '點雲已就緒，請開始掃描以等待抓取規劃結果。' : item.pose_status,
    completed:Object.keys(scanStages).slice(0, Object.keys(scanStages).indexOf(item.scan.stage))} : null);
}
function renderScan(data) {
  scanResult = data;
  const state = data?.state;
  scanEl('scan-state').textContent = data ? (scanSession ? '' : '即時預覽 · ') + (scanStates[state] || state) : '等待 RGB-D';
  scanEl('scan-steps').replaceChildren(...Object.entries(scanStages).map(([key, label]) => {
    const li = document.createElement('li');
    li.textContent = (data?.completed?.includes(key) ? '✓ ' : '') + label;
    li.className = data?.completed?.includes(key) ? 'done' : key === state ? 'active' : '';
    return li;
  }));
  scanEl('scan-metrics').textContent = data ? `${data.label} · ${data.point_count || 0} 點 · ${data.view_count || 1} 視角\n${data.frame_id || '—'}${data.stamp ? ' · RGB-D 時戳 ' + data.stamp.toFixed(3) : ''}${data.elapsed_seconds != null ? ' · 已經過 ' + data.elapsed_seconds.toFixed(1) + 's · 回報 ' + data.age_seconds.toFixed(1) + 's 前' : ''}` : '尚無點雲';
  scanEl('scan-confidence').value = data?.confidence ?? 0;
  scanEl('scan-confidence-text').textContent = data?.confidence == null ? '尚未評估' : `${(100*data.confidence).toFixed(0)}% / 門檻 ${(100*data.threshold).toFixed(0)}%`;
  const reasons = {ambiguous_surface_axes:'PCA 主軸不明確，需換視角補掃。', insufficient_depth_points:'有效深度點不足，請調整視角或物體距離。', no_valid_center_depth:'ROI 中心沒有有效深度。'};
  scanEl('scan-reason').textContent = reasons[data?.reason] || data?.reason || '等待同步 RGB-D 資料。';
  const labels = {antipodal:'對向接觸', contact:'接觸', clearance:'間隙', collision_free:'無碰撞', unknown_space_clear:'未知空間已排除'};
  scanEl('scan-checks').textContent = Object.entries(labels).map(([key,label]) => `${label}：${data?.checks?.[key] === true ? '通過' : data?.checks?.[key] === false ? '未通過' : '未檢查'}`).join(' · ');
  const pose = data?.state === 'ready' ? data.grasp_pose : null;
  scanEl('scan-pose').textContent = pose ? `Grasp pose（${pose.frame_id}）\nXYZ (m): ${pose.position.map(v=>v.toFixed(4)).join(', ')}\nQuaternion (x,y,z,w): ${pose.quaternion.map(v=>v.toFixed(4)).join(', ')}\n夾爪寬度 (m): ${pose.width.toFixed(4)}` : 'Grasp pose：待驗證';
  scanEl('scan-download').disabled = !pose;
  cloudPoints = data?.points || [];
  cloudCandidates = data?.candidates || [];
  cloudPose = pose;
  drawCloud();
}
async function scanRequest(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '掃描介面錯誤');
  return data;
}
async function refreshScan() {
  const data = await scanRequest('/api/scan');
  scanSession = data.session;
  if (scanSession) renderScan(scanSession);
  else showScanPreview();
}
scanEl('scan-object').onchange = () => { if (!scanSession) showScanPreview(); };
scanEl('scan-start').onclick = async () => {
  scanEl('scan-start').disabled = true;
  try {
    await scanRequest('/api/scan', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({index:Number(scanEl('scan-object').value), revision:scanRevision})});
    await refreshScan();
  } catch(e) { scanEl('scan-reason').textContent = e.message; }
};
scanEl('scan-reset').onclick = async () => {
  try { await scanRequest('/api/scan', {method:'DELETE'}); await refreshScan(); }
  catch(e) { scanEl('scan-reason').textContent = e.message; }
};
scanEl('scan-download').onclick = () => {
  if (scanResult?.state !== 'ready' || !scanResult.grasp_pose) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify({id:scanResult.id,
    confidence:scanResult.confidence, checks:scanResult.checks, grasp_pose:scanResult.grasp_pose}, null, 2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'grasp-pose.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
const cloudCanvas = scanEl('scan-cloud');
function drawCloud() {
  const ctx = cloudCanvas.getContext('2d'), w = cloudCanvas.width, h = cloudCanvas.height;
  ctx.clearRect(0,0,w,h);
  if (!cloudPoints.length) { ctx.fillStyle='#b0becd'; ctx.font='18px system-ui'; ctx.fillText('等待物體點雲', 24, 40); return; }
  const center = [0,0,0]; cloudPoints.forEach(p=>p.forEach((v,i)=>center[i]+=v/cloudPoints.length));
  let radius = 0.01; cloudPoints.forEach(p=>{ radius=Math.max(radius,Math.hypot(...p.map((v,i)=>v-center[i]))); });
  const scale = Math.min(w,h)*0.38/radius*zoom;
  const rotate = p => {
    const [x,y,z] = p.map((v,i)=>v-center[i]);
    const a = Math.cos(yaw)*x+Math.sin(yaw)*z, b = -Math.sin(yaw)*x+Math.cos(yaw)*z;
    return [w/2+a*scale, h/2+(Math.cos(pitch)*y-Math.sin(pitch)*b)*scale, Math.sin(pitch)*y+Math.cos(pitch)*b];
  };
  cloudPoints.map(rotate).sort((a,b)=>b[2]-a[2]).forEach(p=>{
    ctx.fillStyle=`hsl(${175+Math.max(-1,Math.min(1,p[2]/radius))*35},65%,65%)`; ctx.fillRect(p[0],p[1],2.5,2.5);
  });
  cloudCandidates.forEach(candidate=>{
    const [x,y]=rotate(candidate.position); ctx.strokeStyle='#f6c76d'; ctx.strokeRect(x-6,y-6,12,12);
  });
  const origin = cloudPose?.position || center;
  const q = cloudPose?.quaternion || [0,0,0,1];
  const [x,y,z,t] = q;
  const axes = [[1-2*(y*y+z*z),2*(x*y+t*z),2*(x*z-t*y)],
    [2*(x*y-t*z),1-2*(x*x+z*z),2*(y*z+t*x)],
    [2*(x*z+t*y),2*(y*z-t*x),1-2*(x*x+y*y)]];
  const start=rotate(origin);
  axes.forEach((axis,i)=>{
    const end=rotate(origin.map((v,j)=>v+axis[j]*radius*0.6));
    ctx.strokeStyle=['#ff7272','#8ddc9b','#83b8ff'][i]; ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(start[0],start[1]);ctx.lineTo(end[0],end[1]);ctx.stroke();
    ctx.fillStyle=ctx.strokeStyle;ctx.font='15px system-ui';ctx.fillText(['X','Y','Z'][i],end[0]+4,end[1]);
  });
  ctx.fillStyle='#b0becd';ctx.font='14px system-ui';ctx.fillText(cloudPose ? '已驗證 grasp 座標軸' : '相機座標軸 · 尚無最終 grasp',16,24);
}
cloudCanvas.onpointerdown = e => {dragging=[e.clientX,e.clientY];cloudCanvas.setPointerCapture(e.pointerId);};
cloudCanvas.onpointermove = e => {if(!dragging)return;yaw+=(e.clientX-dragging[0])*0.01;pitch+=(e.clientY-dragging[1])*0.01;dragging=[e.clientX,e.clientY];drawCloud();};
cloudCanvas.onpointerup = cloudCanvas.onpointercancel = () => {dragging=null;};
cloudCanvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(0.3,Math.min(4,zoom*Math.exp(-e.deltaY*0.001)));drawCloud();},{passive:false});
cloudCanvas.onkeydown = e => {if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();yaw+=e.key==='ArrowLeft'?-0.1:e.key==='ArrowRight'?0.1:0;pitch+=e.key==='ArrowUp'?-0.1:e.key==='ArrowDown'?0.1:0;drawCloud();};
