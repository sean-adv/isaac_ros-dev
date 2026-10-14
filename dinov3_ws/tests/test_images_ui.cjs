// Run with node --test dinov3_ws/tests/test_images_ui.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../src/dinov3_cases/dinov3_cases/static/images.js'), 'utf8');

function page() {
  const elements = new Map();
  function element(id = '') {
    return {id, value: '', textContent: '', disabled: false, style: {}, children: [], options: [],
      setAttribute() {}, removeAttribute() {}, focus() { this.focused = true; },
      replaceChildren(...items) { this.children = items; this.options = items; this.replacements = (this.replacements || 0) + 1; },
      append(item) { this.children.push(item); },
      after(item) { elements.set(item.id, item); },
    };
  }
  const ids = [...source.matchAll(/el\('([^']+)'\)/g)].map(match => match[1]);
  for (const id of [...ids, 'hue', 'saturation', 'value', 'feather']) elements.set(id, element(id));
  elements.delete('camera-capture-help'); // Reproduce a service with cached old HTML.
  const requests = [], timers = [];
  const state = {status: {camera_connected: true, grounding_offline_prompts: ['Detect box.']}};
  const context = vm.createContext({
    document: {getElementById: id => elements.get(id), createElement: () => element()},
    Option: function(text, value) { return {textContent: text, value}; },
    setTimeout: fn => { timers.push(fn); }, clearTimeout() {}, URL, AbortController,
    fetch: async (url, options) => {
      requests.push({url, options});
      if (url === '/api/status' && state.error) throw Error('status unavailable');
      const data = url === '/api/status' ? state.status : url === '/api/images/capture' ?
        {case: 'medicine', names: []} : url === '/api/images/labels' ?
        {case: JSON.parse(options.body).case} : {cases: ['medicine'], images: []};
      return {ok: true, json: async () => data};
    },
  });
  vm.runInContext(source, context);
  return {el: id => elements.get(id), requests, timers, state};
}
const settled = () => new Promise(resolve => setImmediate(resolve));

test('capture remains actionable and explains missing label and prompt', async () => {
  const p = page();
  await settled();
  assert.equal(p.el('camera-capture').disabled, false);
  await p.el('camera-capture').onclick();
  assert.match(p.el('camera-capture-help').textContent, /label/);
  assert.equal(p.el('camera-label').focused, true);
  p.el('camera-label').value = 'medicine';
  p.el('camera-label').oninput();
  await p.el('camera-capture').onclick();
  assert.match(p.el('camera-capture-help').textContent, /prompt/);
  assert.equal(p.requests.some(r => r.url === '/api/images/capture'), false);
});

test('selecting an existing gallery label allows capture without creating it again', async () => {
  const p = page();
  await settled();
  p.el('cases').value = 'medicine';
  p.el('cases').onchange();
  await settled();
  assert.equal(p.el('camera-label').value, 'medicine');
  p.el('camera-prompt').value = 'Detect box.';
  p.el('camera-prompt').onchange();
  await p.el('camera-capture').onclick();
  const request = p.requests.find(r => r.url === '/api/images/capture');
  assert.deepEqual(JSON.parse(request.options.body), {case: 'medicine', prompt: 'Detect box.'});
  assert.equal(p.requests.some(r => r.url === '/api/images/labels'), false);
});

test('status polling preserves prompt menu and does not block capture on a transient status failure', async () => {
  const p = page();
  await settled();
  p.el('camera-label').value = 'medicine';
  p.el('camera-label').oninput();
  p.el('camera-prompt').value = 'Detect box.';
  const replacements = p.el('camera-prompt').replacements;
  await p.timers.shift()();
  assert.equal(p.el('camera-prompt').replacements, replacements);
  assert.equal(p.el('camera-prompt').value, 'Detect box.');
  p.state.error = true;
  await p.timers.shift()();
  await p.el('camera-capture').onclick();
  assert.equal(p.requests.filter(r => r.url === '/api/images/capture').length, 1);
});

test('an uncreated new label prevents capture and duplicate clicks are ignored while saving', async () => {
  const p = page();
  await settled();
  p.el('camera-label').value = 'new_label';
  p.el('camera-label').oninput();
  p.el('camera-prompt').value = 'Detect box.';
  await p.el('camera-capture').onclick();
  assert.equal(p.requests.some(r => r.url === '/api/images/capture'), false);
  p.el('camera-label').value = 'medicine';
  p.el('camera-label').oninput();
  const pending = p.el('camera-capture').onclick();
  assert.equal(p.el('camera-capture').disabled, true);
  await p.el('camera-capture').onclick();
  await pending;
  assert.equal(p.requests.filter(r => r.url === '/api/images/capture').length, 1);
  assert.equal(p.el('camera-capture').disabled, false);
});
