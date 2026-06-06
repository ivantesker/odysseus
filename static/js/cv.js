// ============================================================
// CV pipeline panel — dataset lint/stats/split, eval, convert.
// Self-contained tool modal (mirrors calendar.js): builds its own DOM,
// registers with modalManager for minimize/restore, talks to /api/cv/*.
// ============================================================

let _modal = null;
let _open = false;

const _ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>';

async function _post(path, payload) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify(payload || {}),
  });
  let data;
  try { data = await res.json(); } catch { data = { error: 'bad response' }; }
  if (!res.ok) throw new Error(data.error || data.detail || `HTTP ${res.status}`);
  return data;
}

function _field(label, id, value, ph) {
  return `<label class="cv-field"><span>${label}</span>`
    + `<input id="${id}" value="${value || ''}" placeholder="${ph || ''}" spellcheck="false"></label>`;
}

// Example values match a RiderDome-style YOLO project (Front/Back/Side @256).
const _TABS = {
  dataset: () => `
    <div class="cv-row">
      <select id="cv-ds-action">
        <option value="lint">lint — find bad/empty labels</option>
        <option value="stats">stats — class balance + orphans</option>
        <option value="split">split — deterministic train/val</option>
      </select>
    </div>
    ${_field('labels dir', 'cv-ds-labels', '', 'D:/rider_dome/yolo_dataset/labels')}
    ${_field('images dir (stats/split)', 'cv-ds-images', '', 'D:/rider_dome/yolo_dataset/images')}
    ${_field('out dir (split)', 'cv-ds-out', '', 'D:/rider_dome/yolo_dataset/split')}
    <div class="cv-row">
      ${_field('num_classes', 'cv-ds-nc', '3', '3')}
      ${_field('class_names', 'cv-ds-names', 'Front,Back,Side', 'Front,Back,Side')}
      ${_field('val_frac', 'cv-ds-val', '0.2', '0.2')}
    </div>
    <button class="cv-run" data-action="dataset">Run dataset tool</button>`,
  eval: () => `
    ${_field('model (.pt/.onnx)', 'cv-ev-model', '', 'D:/rider_dome/yolov10_training/.../best.pt')}
    ${_field('data (yaml or images)', 'cv-ev-data', '', 'D:/rider_dome/yolo_dataset/data.yaml')}
    <div class="cv-row">${_field('imgsz', 'cv-ev-imgsz', '256', '256')}${_field('iou', 'cv-ev-iou', '0.5', '0.5')}</div>
    <button class="cv-run" data-action="eval">Run eval (mAP + latency)</button>
    <p class="cv-hint">needs <code>ultralytics</code> — see requirements-optional.txt</p>`,
  convert: () => `
    ${_field('source (.pt)', 'cv-cv-src', '', 'D:/rider_dome/yolov10_training/.../best.pt')}
    <div class="cv-row">
      <label class="cv-field"><span>formats</span>
        <span><label><input type="checkbox" class="cv-fmt" value="onnx" checked> onnx</label>
        <label><input type="checkbox" class="cv-fmt" value="fp16"> fp16</label>
        <label><input type="checkbox" class="cv-fmt" value="rknn"> rknn</label></span></label>
    </div>
    <div class="cv-row">${_field('imgsz', 'cv-cv-imgsz', '256', '256')}${_field('opset', 'cv-cv-opset', '12', '12')}${_field('target', 'cv-cv-target', 'rk3588', 'rk3588')}</div>
    <label class="cv-chk"><input type="checkbox" id="cv-cv-parity" checked> PT-vs-ONNX parity check</label>
    <button class="cv-run" data-action="convert">Convert (PT→ONNX→FP16→RKNN)</button>
    <p class="cv-hint">onnx/rknn steps need <code>ultralytics, onnx, rknn-toolkit2</code></p>`,
};

function _getModal() {
  if (_modal) return _modal;
  _modal = document.createElement('div');
  _modal.id = 'cv-modal';
  _modal.className = 'modal';
  _modal.style.display = 'none';
  _modal.innerHTML = `
    <div class="modal-content" style="width:min(760px,94vw);height:90vh;max-height:90vh;display:flex;flex-direction:column;">
      <div class="modal-header">
        <h4>${_ICON}CV pipeline <span class="cv-tag">YOLO → edge</span></h4>
        <button class="close-btn" id="cv-close" aria-label="Close CV">✖</button>
      </div>
      <div class="cv-tabs">
        <button class="cv-tab active" data-tab="dataset">Dataset</button>
        <button class="cv-tab" data-tab="eval">Eval / bench</button>
        <button class="cv-tab" data-tab="convert">Convert</button>
      </div>
      <div class="modal-body" id="cv-body" style="flex:1;overflow:auto;padding:14px 16px;">
        <div id="cv-form"></div>
        <pre id="cv-out" class="cv-out" hidden></pre>
      </div>
    </div>`;
  document.body.appendChild(_modal);
  _modal.querySelector('#cv-close').addEventListener('click', closeCv);
  _modal.addEventListener('click', (e) => { if (e.target === _modal) closeCv(); });
  _modal.querySelectorAll('.cv-tab').forEach(t => {
    t.addEventListener('click', () => _showTab(t.dataset.tab));
  });
  _modal.querySelector('#cv-form').addEventListener('click', (e) => {
    const btn = e.target.closest('.cv-run');
    if (btn) _run(btn.dataset.action);
  });
  _showTab('dataset');
  return _modal;
}

function _showTab(name) {
  _getModal();
  _modal.querySelectorAll('.cv-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  _modal.querySelector('#cv-form').innerHTML = _TABS[name]();
  const out = _modal.querySelector('#cv-out');
  out.hidden = true; out.textContent = '';
}

function _val(id) { return (_modal.querySelector('#' + id)?.value || '').trim(); }

function _render(out, data) {
  out.hidden = false;
  out.classList.remove('cv-err');
  out.textContent = JSON.stringify(data, null, 2);
}

async function _run(action) {
  const out = _modal.querySelector('#cv-out');
  out.hidden = false; out.classList.remove('cv-err'); out.textContent = 'Running…';
  try {
    let data;
    if (action === 'dataset') {
      const act = _val('cv-ds-action');
      const names = _val('cv-ds-names');
      const payload = {
        action: act,
        labels_dir: _val('cv-ds-labels'),
        images_dir: _val('cv-ds-images'),
        out_dir: _val('cv-ds-out'),
        num_classes: _val('cv-ds-nc') ? parseInt(_val('cv-ds-nc'), 10) : null,
        class_names: names ? names.split(',').map(s => s.trim()).filter(Boolean) : null,
        val_frac: parseFloat(_val('cv-ds-val') || '0.2'),
      };
      data = await _post(`/api/cv/dataset/${act}`, payload);
    } else if (action === 'eval') {
      data = await _post('/api/cv/eval', {
        model: _val('cv-ev-model'), data: _val('cv-ev-data'),
        imgsz: parseInt(_val('cv-ev-imgsz') || '256', 10), iou: parseFloat(_val('cv-ev-iou') || '0.5'),
      });
    } else if (action === 'convert') {
      const formats = Array.from(_modal.querySelectorAll('.cv-fmt:checked')).map(c => c.value);
      data = await _post('/api/cv/convert', {
        source: _val('cv-cv-src'), formats,
        imgsz: parseInt(_val('cv-cv-imgsz') || '256', 10),
        opset: parseInt(_val('cv-cv-opset') || '12', 10),
        target_platform: _val('cv-cv-target') || 'rk3588',
        parity: _modal.querySelector('#cv-cv-parity')?.checked !== false,
      });
    }
    _render(out, data);
  } catch (e) {
    out.hidden = false; out.classList.add('cv-err'); out.textContent = 'Error: ' + e.message;
  }
}

export function openCv() {
  _getModal();
  _modal.style.display = 'flex';
  _open = true;
  import('./modalManager.js').then(M => {
    if (!M.isRegistered('cv-modal')) {
      M.register('cv-modal', {
        sidebarBtnId: 'tool-cv-btn',
        label: 'CV',
        icon: _ICON,
        restoreFn: () => { _modal.style.display = 'flex'; _open = true; },
        closeFn: () => { _modal.style.display = 'none'; _open = false; },
      });
      M.injectMinimizeButton(_modal, 'cv-modal');
    }
  }).catch(() => {});
}

export function closeCv() {
  if (_modal) _modal.style.display = 'none';
  _open = false;
}

export function isCvOpen() { return _open; }

export default { openCv, closeCv, isCvOpen };
