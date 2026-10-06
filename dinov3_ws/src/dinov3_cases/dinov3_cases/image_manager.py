"""Local image gallery and HSV background removal, independent of ROS/model loading."""
import argparse
import io
import re
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageOps

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def group_path(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
        raise ValueError('標籤限 1–80 個英文字母、數字、底線或連字號。')
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError('無效的標籤路徑。')
    return path


def image_path(root, group, filename):
    folder = group_path(root, group)
    if (not isinstance(filename, str) or not filename or '\\' in filename
            or '/' in filename or Path(filename).name != filename):
        raise ValueError('無效的圖片名稱。')
    path = folder / filename
    if (path.is_symlink() or path.resolve().parent != folder.resolve()
            or path.suffix.lower() not in EXTENSIONS or not path.is_file()):
        raise ValueError('圖片不存在或格式不支援。')
    return path


def load_rgba(source):
    try:
        with Image.open(source) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError('圖片不得超過 2500 萬像素。')
            return ImageOps.exif_transpose(image).convert('RGBA')
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('無法讀取圖片。') from error


def remove_background(image, options):
    """Match circular hue and S/V, optionally flood-fill only the clicked region."""
    if not isinstance(options, dict):
        raise ValueError('請提供去背參數。')

    def number(name, default, maximum):
        value = options.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or not 0 <= value <= maximum):
            raise ValueError(f'{name} 必須介於 0 到 {maximum}。')
        return float(value)

    x = number('x', None, image.width - 1)
    y = number('y', None, image.height - 1)
    if not x.is_integer() or not y.is_integer():
        raise ValueError('取色座標必須是整數。')
    x, y = int(x), int(y)
    h = number('hue', 15, 180) / 2  # UI degrees; OpenCV uses half degrees.
    s = number('saturation', 50, 255)
    v = number('value', 50, 255)
    feather = number('feather', 0, 10)
    connected = options.get('connected', True)
    if not isinstance(connected, bool):
        raise ValueError('connected 必須是布林值。')
    rgba = np.array(image.convert('RGBA'))
    if rgba[y, x, 3] == 0:
        raise ValueError('請點選尚未透明的背景位置。')
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV).astype(np.float32)
    sample = hsv[y, x]
    hue_distance = np.abs(hsv[..., 0] - sample[0])
    hue_distance = np.minimum(hue_distance, 180 - hue_distance)
    # Hue is unstable near grayscale; S and V still constrain those matches.
    hue_match = (hue_distance <= h) | ((sample[1] <= 20) & (hsv[..., 1] <= 20))
    mask = (hue_match & (np.abs(hsv[..., 1] - sample[1]) <= s)
            & (np.abs(hsv[..., 2] - sample[2]) <= v) & (rgba[..., 3] > 0)).astype(np.uint8)
    if connected:
        cv2.floodFill(mask, None, (x, y), 2, flags=4)
        mask = (mask == 2).astype(np.uint8)
    removal = mask.astype(np.float32)
    if feather > 0:
        removal = cv2.GaussianBlur(removal, (0, 0), feather)
    rgba[..., 3] = np.rint(rgba[..., 3] * (1 - removal)).astype(np.uint8)
    # Erase hidden background RGB as well; RGB-only consumers otherwise recover it.
    rgba[rgba[..., 3] == 0, :3] = 0
    return Image.fromarray(rgba)


def png_response(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    output.seek(0)
    response = send_file(output, mimetype='image/png')
    response.headers['Cache-Control'] = 'no-store'
    return response


def register_image_manager(app, root, on_change=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mutation_lock = threading.RLock()

    def changed(group):
        if on_change:
            on_change(group)

    def payload():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError('請提供 JSON 物件。')
        return data

    def save(image, folder, suffix=''):
        folder.mkdir(exist_ok=True)
        destination = folder / f'{uuid.uuid4().hex}{suffix}.png'
        temporary = destination.with_suffix('.tmp')
        try:
            image.save(temporary, format='PNG')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination.name

    @app.get('/images')
    def manager_page():
        return app.config.get('IMAGE_MANAGER_PAGE', MANAGER_PAGE)

    @app.get('/api/images')
    def list_images():
        with mutation_lock:
            groups = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                            and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', p.name))
            group = request.args.get('case', '')
            images = []
            if group:
                folder = group_path(root, group)
                if not folder.is_dir():
                    raise ValueError('標籤不存在。')
                images = [dict(name=p.name, bytes=p.stat().st_size) for p in sorted(folder.iterdir())
                          if p.suffix.lower() in EXTENSIONS and p.is_file() and not p.is_symlink()]
        return jsonify(cases=groups, images=images)

    @app.get('/api/images/file/<group>/<filename>')
    def picture(group, filename):
        # Normalize EXIF orientation so gallery, click coordinates and HSV agree.
        with mutation_lock:
            image = load_rgba(image_path(root, group, filename))
        return png_response(image)

    @app.post('/api/images/upload')
    def upload_images():
        group = request.form.get('case', '')
        folder = group_path(root, group)
        files = request.files.getlist('images')
        if not files:
            raise ValueError('請選擇圖片。')
        images = [load_rgba(file.stream) for file in files]
        with mutation_lock:
            names = [save(image, folder) for image in images]
            changed(group)
        return jsonify(ok=True, names=names)

    @app.post('/api/images/preview')
    def preview():
        data = payload()
        with mutation_lock:
            image = load_rgba(image_path(root, data.get('case'), data.get('name')))
        return png_response(remove_background(image, data.get('options')))

    @app.post('/api/images/cutout')
    def cutout():
        data = payload()
        target = data.get('target_case', data.get('case'))
        folder = group_path(root, target)
        with mutation_lock:
            image = load_rgba(image_path(root, data.get('case'), data.get('name')))
            result = remove_background(image, data.get('options'))
            name = save(result, folder, '_cutout')
            changed(target)
        return jsonify(ok=True, case=target, name=name)

    @app.delete('/api/images/file/<group>/<filename>')
    def delete_picture(group, filename):
        with mutation_lock:
            image_path(root, group, filename).unlink()
            changed(group)
        return jsonify(ok=True)


def create_app(root):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
    app.config['IMAGE_MANAGER_PAGE'] = MANAGER_PAGE.replace('DINOv3 首頁', '圖片管理首頁')

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='上傳總大小不得超過 64 MB。'), 413

    @app.get('/')
    def index():
        return app.config['IMAGE_MANAGER_PAGE']

    register_image_manager(app, root)
    return app


def main():
    parser = argparse.ArgumentParser(description='本機圖片管理與 HSV 去背')
    parser.add_argument('--cases-dir', default='/workspaces/isaac_ros-dev/dinov3_cases_data')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8091)
    args = parser.parse_args()
    from werkzeug.serving import make_server
    server = make_server(args.host, args.port, create_app(args.cases_dir), threaded=True)
    print(f'圖片管理：http://{args.host}:{args.port}/images', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


MANAGER_PAGE = r'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>圖片管理 · HSV 去背</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101821;color:#e7eef8;font:16px system-ui}main{max-width:1400px;margin:auto;padding:24px}a{color:#8ddcc4}header{display:flex;align-items:center;justify-content:space-between;gap:12px}h1{font-size:26px}h2{font-size:19px;margin-top:0}section{background:#1b2836;border:1px solid #34485b;border-radius:12px;padding:20px;margin:16px 0}button,input,select{font:inherit;padding:9px;border-radius:6px;border:1px solid #496075}input,select{background:#101821;color:inherit;max-width:100%}button{background:#8ddcc4;color:#101821;cursor:pointer}button:disabled{opacity:.45;cursor:default}button.danger{background:#f6a6a6}label{display:block;margin:12px 0}small,.muted{color:#b0becd}form,.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.layout{display:grid;grid-template-columns:320px 1fr;gap:16px}.gallery{max-height:720px;overflow:auto;display:grid;grid-template-columns:1fr 1fr;gap:10px}.tile{background:#101821;color:inherit;padding:8px;text-align:left;overflow:hidden;border:2px solid transparent}.tile[aria-pressed=true]{border-color:#8ddcc4}.tile img{width:100%;height:110px;object-fit:contain}.tile span{display:block;font-size:12px;overflow-wrap:anywhere}.checker{background-color:#fff;background-image:linear-gradient(45deg,#ccd2d8 25%,transparent 25%),linear-gradient(-45deg,#ccd2d8 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ccd2d8 75%),linear-gradient(-45deg,transparent 75%,#ccd2d8 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}.view{min-width:0}.view canvas,.view img{max-width:100%;height:auto;display:block;margin:auto}#original{cursor:crosshair}.imagebox{max-height:520px;overflow:auto;border-radius:8px}.controls{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.controls input{width:100%;padding:0}#message{min-height:24px;color:#8ddcc4}#message.error{color:#ffb5b5}#sample{display:inline-block;width:22px;height:22px;vertical-align:middle;border:1px solid #ccc;border-radius:4px}#filename{overflow-wrap:anywhere} [hidden]{display:none!important}@media(max-width:850px){.layout{grid-template-columns:1fr}.gallery{max-height:280px;grid-template-columns:repeat(3,1fr)}.controls{grid-template-columns:1fr 1fr}.views{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}}
</style></head><body><main>
<header><div><h1>圖片管理 · HSV 去背</h1><p class="muted">依標籤整理圖片，點選背景取色，再產生透明 PNG。</p></div><a href="/">DINOv3 首頁</a></header>
<section><h2>匯入圖片</h2><form id="upload"><input name="case" id="upload-case" aria-label="上傳標籤" placeholder="標籤，例如 medicine_01" required pattern="[A-Za-z0-9_-]{1,80}" list="case-list"><input name="images" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" multiple required><button id="upload-button">上傳到標籤</button></form><datalist id="case-list"></datalist><p><small>標籤支援英文、數字、底線、連字號。去背會新增圖片，原圖保留。</small></p></section>
<p id="message" role="status" aria-live="polite"></p>
<div class="layout"><section><h2>標籤與圖片</h2><div class="toolbar"><select id="cases" aria-label="篩選標籤"><option value="">請選擇標籤</option></select><button id="reload">重新整理</button></div><p id="count" class="muted"></p><div id="gallery" class="gallery"></div></section>
<section><h2>點選背景取色</h2><p id="filename" class="muted">先從左側選擇圖片。</p><div class="views"><div class="view"><p>原圖 · 點選背景</p><div class="imagebox checker"><canvas id="original" hidden aria-label="點選原圖背景取色"></canvas></div></div><div class="view"><p>去背預覽 · 棋盤格表示透明</p><div class="imagebox checker"><img id="preview" hidden alt="透明背景預覽"></div></div></div>
<p><span id="sample"></span> <span id="sample-text">尚未選擇背景色</span></p>
<div class="controls"><label>H 色相容差：<output id="hue-out">15</output>°<input id="hue" type="range" min="0" max="180" value="15"></label><label>S 飽和度容差：<output id="saturation-out">50</output><input id="saturation" type="range" min="0" max="255" value="50"></label><label>V 亮度容差：<output id="value-out">50</output><input id="value" type="range" min="0" max="255" value="50"></label><label>邊緣柔化：<output id="feather-out">0</output> px<input id="feather" type="range" min="0" max="10" step="0.5" value="0"></label></div>
<label><input id="connected" type="checkbox" checked> 僅去除與取色位置連通的背景區域</label><p><small>取消勾選會去除全圖相近顏色。物體與背景顏色相近時，請降低容差；斷開的背景可儲存後再點選去除。</small></p>
<label>新圖片的標籤 <input id="target-case" placeholder="預設使用原圖標籤" pattern="[A-Za-z0-9_-]{1,80}" list="case-list"></label>
<div class="toolbar"><button id="save" disabled>儲存新的透明 PNG</button><a id="download" hidden download="cutout.png">下載預覽</a><button id="delete" class="danger" disabled>刪除此圖片</button></div></section></div></main>
<script>
const el=id=>document.getElementById(id);
let selected=null,point=null,previewUrl=null,previewController=null,previewVersion=0,loadVersion=0,timer=null,busy=false;
function message(text,error=false){el('message').textContent=text;el('message').className=error?'error':'';}
async function api(url,options){const response=await fetch(url,options);if(!response.ok){const data=await response.json().catch(()=>({}));throw Error(data.error||'請求失敗：'+response.status);}return response;}
function fileUrl(group,name){return '/api/images/file/'+encodeURIComponent(group)+'/'+encodeURIComponent(name);}
function buttons(){el('save').disabled=busy||!selected||!point||!previewUrl;el('delete').disabled=busy||!selected;el('upload-button').disabled=busy;}
function clearPreview(){previewVersion++;if(previewController)previewController.abort();clearTimeout(timer);if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;el('preview').hidden=true;el('preview').removeAttribute('src');el('download').hidden=true;el('download').removeAttribute('href');buttons();}
function clearSelection(){loadVersion++;selected=null;point=null;clearPreview();el('original').hidden=true;el('filename').textContent='先從左側選擇圖片。';el('sample-text').textContent='尚未選擇背景色';el('sample').style.background='transparent';buttons();}
async function refresh(group=el('cases').value){const data=await (await api('/api/images'+(group?'?case='+encodeURIComponent(group):''))).json();el('cases').replaceChildren(new Option('請選擇標籤',''),...data.cases.map(c=>new Option(c,c)));el('cases').value=group;el('case-list').replaceChildren(...data.cases.map(c=>new Option(c,c)));el('count').textContent=group?data.images.length+' 張圖片':'選擇標籤或上傳圖片以開始';el('gallery').replaceChildren();for(const item of data.images){const button=document.createElement('button');button.className='tile';button.setAttribute('aria-pressed',String(selected?.group===group&&selected?.name===item.name));const image=document.createElement('img');image.src=fileUrl(group,item.name);image.loading='lazy';image.alt=item.name;image.className='checker';const name=document.createElement('span');name.textContent=item.name;button.append(image,name);button.onclick=()=>choose(group,item.name).catch(e=>message(e.message,true));el('gallery').append(button);}}
async function choose(group,name){clearSelection();const version=loadVersion;const image=new Image();image.src=fileUrl(group,name);await image.decode();if(version!==loadVersion)return;const canvas=el('original');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;canvas.getContext('2d').drawImage(image,0,0);canvas.hidden=false;selected={group,name};el('filename').textContent=group+' / '+name+' · '+canvas.width+' × '+canvas.height;el('target-case').value=group;el('upload-case').value=group;for(const tile of el('gallery').children)tile.setAttribute('aria-pressed',String(tile.querySelector('span').textContent===name));message('請在原圖上點選要去除的背景色。');buttons();}
function options(){return {x:point.x,y:point.y,hue:Number(el('hue').value),saturation:Number(el('saturation').value),value:Number(el('value').value),feather:Number(el('feather').value),connected:el('connected').checked};}
function body(){return {case:selected.group,name:selected.name,options:options(),target_case:el('target-case').value||selected.group};}
function schedulePreview(){clearPreview();if(!point)return;timer=setTimeout(()=>preview(),250);}
async function preview(){if(!selected||!point)return;const version=previewVersion;previewController=new AbortController();message('正在產生去背預覽…');try{const response=await api('/api/images/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body()),signal:previewController.signal});const blob=await response.blob();if(version!==previewVersion)return;previewUrl=URL.createObjectURL(blob);el('preview').src=previewUrl;el('preview').hidden=false;el('download').href=previewUrl;el('download').hidden=false;message('預覽已更新。可調整容差，或儲存新圖片。');buttons();}catch(e){if(e.name!=='AbortError'&&version===previewVersion)message(e.message,true);}}
el('original').onclick=event=>{if(!selected||busy)return;const canvas=el('original'),rect=canvas.getBoundingClientRect();const x=Math.min(canvas.width-1,Math.max(0,Math.floor((event.clientX-rect.left)*canvas.width/rect.width))),y=Math.min(canvas.height-1,Math.max(0,Math.floor((event.clientY-rect.top)*canvas.height/rect.height)));const rgba=canvas.getContext('2d').getImageData(x,y,1,1).data;if(rgba[3]===0){message('此位置已透明，請點選有顏色的背景。',true);return;}point={x,y};el('sample').style.background='rgb('+Array.from(rgba).slice(0,3).join(',')+')';el('sample-text').textContent='RGB ('+Array.from(rgba).slice(0,3).join(', ')+') · 座標 ('+x+', '+y+')';schedulePreview();};
for(const name of ['hue','saturation','value','feather'])el(name).oninput=()=>{el(name+'-out').textContent=el(name).value;schedulePreview();};el('connected').onchange=schedulePreview;
el('cases').onchange=()=>{clearSelection();el('upload-case').value=el('cases').value;refresh().catch(e=>message(e.message,true));};el('reload').onclick=()=>{clearSelection();refresh().catch(e=>message(e.message,true));};
el('upload').onsubmit=async event=>{event.preventDefault();if(busy)return;busy=true;buttons();try{const form=new FormData(event.target),group=form.get('case');await api('/api/images/upload',{method:'POST',body:form});clearSelection();await refresh(group);event.target.querySelector('input[type=file]').value='';message('圖片已上傳至 '+group+'。');}catch(e){message(e.message,true);}finally{busy=false;buttons();}};
el('save').onclick=async()=>{if(busy||!point||!previewUrl)return;const data=body();if(!/^[A-Za-z0-9_-]{1,80}$/.test(data.target_case)){message('標籤限 1–80 個英文字母、數字、底線或連字號。',true);return;}busy=true;buttons();try{const result=await (await api('/api/images/cutout',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})).json();await refresh(result.case);await choose(result.case,result.name);message('已建立透明 PNG，原圖保留於 '+data.case+'。');}catch(e){message(e.message,true);}finally{busy=false;buttons();}};
el('delete').onclick=async()=>{if(busy||!selected)return;const item={...selected};if(!confirm('確定永久刪除 '+item.group+' / '+item.name+'？此操作無法復原。'))return;busy=true;buttons();try{await api(fileUrl(item.group,item.name),{method:'DELETE'});clearSelection();await refresh(item.group);message('圖片已刪除。');}catch(e){message(e.message,true);}finally{busy=false;buttons();}};
refresh().catch(e=>message(e.message,true));
</script></body></html>
'''

if __name__ == '__main__':
    main()
