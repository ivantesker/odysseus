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
    <div class="cv-inspect">
      ${_field('dataset root (auto-fills from data.yaml)', 'cv-ds-root', _recall('root'), 'D:/rider_dome/yolo_dataset')}
      <button class="cv-load" data-action="inspect">Load ↻</button>
    </div>
    <div class="cv-row">
      <select id="cv-ds-action">
        <option value="eda">eda — full visual report ↗</option>
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
    <label class="cv-chk"><input type="checkbox" id="cv-ds-scan"> image-quality scan (brightness/blur/dups — slower, decodes images)</label>
    <button class="cv-run" data-action="dataset">Run dataset tool</button>`,
  review: () => `
    <p class="cv-hint">Compare predictions vs ground-truth (YOLO .txt, preds add a conf column). Report: failure gallery, PR curves + per-class threshold, suspect labels, confusion matrix, A/B model diff.</p>
    ${_field('ground-truth labels dir', 'cv-rv-gt', '', 'D:/rider_dome/yolo_dataset/labels')}
    ${_field('predictions dir (model A)', 'cv-rv-pa', '', 'D:/rider_dome/preds_pt')}
    ${_field('images dir (enables failure gallery)', 'cv-rv-img', '', 'D:/rider_dome/yolo_dataset/images')}
    ${_field('predictions dir (model B, optional A/B)', 'cv-rv-pb', '', 'D:/rider_dome/preds_rknn_int8')}
    <div class="cv-row">${_field('class_names', 'cv-rv-names', 'Front,Back,Side', '')}${_field('iou', 'cv-rv-iou', '0.5', '0.5')}</div>
    <button class="cv-run" data-action="review">Run review ↗</button>`,
  eval: () => `
    ${_field('model (.pt/.onnx)', 'cv-ev-model', '', 'D:/rider_dome/yolov10_training/.../best.pt')}
    ${_field('data (yaml or images)', 'cv-ev-data', '', 'D:/rider_dome/yolo_dataset/data.yaml')}
    <div class="cv-row">${_field('imgsz', 'cv-ev-imgsz', '256', '256')}${_field('iou', 'cv-ev-iou', '0.5', '0.5')}</div>
    <button class="cv-run" data-action="eval">Run eval (mAP + latency)</button>
    <p class="cv-hint">needs <code>ultralytics</code> — see requirements-optional.txt</p>
    <hr style="border:none;border-top:1px solid var(--border,#355a66);margin:16px 0">
    <p class="cv-hint">Aggregate scattered eval CSVs (results.csv across dated folders) into one comparison ↗ — no deps.</p>
    ${_field('eval results root', 'cv-ag-root', '', 'D:/rider_dome/eval_results')}
    <button class="cv-run" data-action="aggregate">Aggregate eval CSVs ↗</button>`,
  import: () => `
    <p class="cv-hint">Convert COCO / Pascal-VOC / Label-Studio annotations to YOLO.</p>
    <div class="cv-row">
      <select id="cv-im-fmt">
        <option value="coco">COCO (instances.json)</option>
        <option value="labelstudio">Label Studio (export.json)</option>
        <option value="voc">Pascal VOC (xml dir)</option>
      </select>
    </div>
    ${_field('source (json file or xml dir)', 'cv-im-src', '', 'D:/export/instances.json')}
    ${_field('out labels dir', 'cv-im-out', '', 'D:/rider_dome/converted/labels')}
    <button class="cv-run" data-action="convert_data">Convert → YOLO</button>
    <hr style="border:none;border-top:1px solid var(--border,#355a66);margin:16px 0">
    <p class="cv-hint">Leak-free stratified split — keeps each source/video's frames together (regex captures the source key from the filename).</p>
    ${_field('labels dir', 'cv-sp-labels', '', 'D:/rider_dome/yolo_dataset/labels')}
    <div class="cv-row">${_field('source regex (group 1 = source)', 'cv-sp-rx', '', '(.+)_frame')}${_field('val_frac', 'cv-sp-val', '0.2', '0.2')}</div>
    <button class="cv-run" data-action="stratified">Stratified split (report balance)</button>`,
  runs: () => `
    <p class="cv-hint">Compare training runs (Ultralytics dirs with results.csv) — overlay metric curves, final table, config diff.</p>
    ${_field('run dirs (comma-separated)', 'cv-rn-dirs', '', 'D:/rider_dome/_yolov10_training/run_a, .../run_b')}
    <button class="cv-run" data-action="compare_runs">Compare runs ↗</button>
    <hr style="border:none;border-top:1px solid var(--border,#355a66);margin:16px 0">
    <p class="cv-hint">Drift — model prediction distribution on new (unlabeled) data vs a baseline (PSI on conf / boxes-per-image / class freq).</p>
    ${_field('baseline predictions dir', 'cv-dr-base', '', 'D:/rider_dome/preds_eval')}
    ${_field('new predictions dir', 'cv-dr-new', '', 'D:/rider_dome/preds_production')}
    <button class="cv-run" data-action="drift">Check drift ↗</button>`,
  deploy: () => `
    <p class="cv-hint">Generate DeepStream + Triton configs from an ONNX model (I/O shapes auto-read).</p>
    ${_field('onnx model', 'cv-dp-onnx', '', 'D:/rider_dome/batch_yolox/exports/model.onnx')}
    <div class="cv-row">${_field('class_names', 'cv-dp-names', '', 'car,bus,truck')}
      <label class="cv-field"><span>precision</span><select id="cv-dp-mode"><option value="2">FP16</option><option value="1">INT8</option><option value="0">FP32</option></select></label></div>
    <button class="cv-run" data-action="deploy">Generate configs ↗</button>
    <hr style="border:none;border-top:1px solid var(--border,#355a66);margin:16px 0">
    <p class="cv-hint">INT8 calibration set — pick a diverse subset (aHash farthest-point) so quantization doesn't tank on near-identical frames.</p>
    ${_field('images dir', 'cv-cal-img', '', 'D:/rider_dome/yolo_dataset/images')}
    <div class="cv-row">${_field('n images', 'cv-cal-n', '200', '200')}${_field('out file (optional)', 'cv-cal-out', '', 'D:/calib_list.txt')}</div>
    <button class="cv-run" data-action="calibration">Select calibration set</button>`,
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

function _injectStyles() {
  if (document.getElementById('cv-styles')) return;
  const s = document.createElement('style');
  s.id = 'cv-styles';
  s.textContent = `
  #cv-modal .cv-tabs{display:flex;flex-wrap:wrap;gap:4px;padding:8px 14px 0;border-bottom:1px solid var(--border,#355a66)}
  #cv-modal .cv-tab{background:none;border:none;border-bottom:2px solid transparent;color:var(--color-muted,#888);padding:8px 12px;cursor:pointer;font-size:13px}
  #cv-modal .cv-tab.active{color:var(--fg,#9cdef2);border-bottom-color:var(--accent,#e06c75)}
  #cv-modal .cv-tag{font-size:10px;color:var(--color-muted,#888);border:1px solid var(--border,#355a66);border-radius:8px;padding:1px 6px;margin-left:6px}
  #cv-modal .cv-field{display:flex;flex-direction:column;gap:3px;margin:8px 0;font-size:12px;flex:1}
  #cv-modal .cv-field span{color:var(--color-muted,#888)}
  #cv-modal .cv-field input,#cv-modal select{background:var(--panel,#111);border:1px solid var(--border,#355a66);border-radius:6px;color:var(--fg,#9cdef2);padding:7px 9px;font-size:12px;font-family:ui-monospace,monospace;width:100%}
  #cv-modal .cv-row{display:flex;gap:10px}
  #cv-modal .cv-inspect{display:flex;gap:8px;align-items:flex-end;margin-bottom:6px}
  #cv-modal .cv-load,#cv-modal .cv-run{background:var(--accent,#e06c75);color:#111;border:none;border-radius:7px;padding:9px 14px;cursor:pointer;font-weight:600;font-size:13px;white-space:nowrap}
  #cv-modal .cv-load{align-self:flex-end;margin-bottom:8px;background:var(--hl-function,#61afef)}
  #cv-modal .cv-run{margin-top:10px;width:100%}
  #cv-modal .cv-chk{display:flex;gap:7px;align-items:center;font-size:12px;color:var(--color-muted,#888);margin:8px 0}
  #cv-modal .cv-hint{font-size:11px;color:var(--color-muted,#888);margin:4px 0 8px}
  #cv-modal .cv-report-link{display:inline-block;background:var(--accent,#e06c75);color:#111;border-radius:7px;padding:8px 12px;text-decoration:none;font-weight:600;margin:10px 0 8px}
  #cv-modal .cv-report-frame{width:100%;height:60vh;border:1px solid var(--border,#355a66);border-radius:8px;background:#282c34;display:block}
  #cv-modal .cv-out{background:var(--panel,#111);border:1px solid var(--border,#355a66);border-radius:8px;padding:10px;font-size:11px;color:var(--color-muted,#888);max-height:240px;overflow:auto;white-space:pre-wrap;margin-top:10px}
  #cv-modal .cv-out.cv-err{color:var(--red,#e06c75);border-color:var(--red,#e06c75)}
  #cv-modal .cv-run:disabled,#cv-modal .cv-load:disabled{opacity:0.6;cursor:wait}
  #cv-modal .cv-spin{display:inline-block;width:11px;height:11px;border:2px solid #1118;border-top-color:#111;border-radius:50%;animation:cv-spin .7s linear infinite;vertical-align:-1px;margin-right:4px}
  @keyframes cv-spin{to{transform:rotate(360deg)}}
  #cv-modal .cv-history{margin-top:18px;border-top:1px solid var(--border,#355a66);padding-top:10px}
  #cv-modal .cv-hist-h{font-size:11px;color:var(--color-muted,#888);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
  #cv-modal .cv-hist-item{display:block;font-size:12px;color:var(--hl-function,#61afef);text-decoration:none;padding:3px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #cv-modal .cv-hist-item:hover{text-decoration:underline}`;
  document.head.appendChild(s);
}

function _getModal() {
  if (_modal) return _modal;
  _injectStyles();
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
        <button class="cv-tab" data-tab="import">Import</button>
        <button class="cv-tab" data-tab="review">Review / QA</button>
        <button class="cv-tab" data-tab="eval">Eval / bench</button>
        <button class="cv-tab" data-tab="runs">Runs / Drift</button>
        <button class="cv-tab" data-tab="convert">Convert</button>
        <button class="cv-tab" data-tab="deploy">Deploy</button>
      </div>
      <div class="modal-body" id="cv-body" style="flex:1;overflow:auto;padding:14px 16px;">
        <div id="cv-form"></div>
        <pre id="cv-out" class="cv-out" hidden></pre>
        <div id="cv-history" class="cv-history"></div>
      </div>
    </div>`;
  document.body.appendChild(_modal);
  _modal.querySelector('#cv-close').addEventListener('click', closeCv);
  _modal.addEventListener('click', (e) => { if (e.target === _modal) closeCv(); });
  _modal.querySelectorAll('.cv-tab').forEach(t => {
    t.addEventListener('click', () => _showTab(t.dataset.tab));
  });
  _modal.querySelector('#cv-form').addEventListener('click', (e) => {
    const run = e.target.closest('.cv-run');
    if (run) { _run(run.dataset.action, run); return; }
    const load = e.target.closest('.cv-load');
    if (load) _inspect(load);
  });
  // Esc closes when on top (Odysseus modal convention).
  _onKey = (e) => { if (e.key === 'Escape' && _open && _modal.style.display !== 'none') closeCv(); };
  document.addEventListener('keydown', _onKey);
  _showTab(_recall('tab') || 'dataset');
  return _modal;
}

let _onKey = null;

function _showTab(name) {
  _getModal();
  if (!_TABS[name]) name = 'dataset';
  _remember('tab', name);
  _modal.querySelectorAll('.cv-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  _modal.querySelector('#cv-form').innerHTML = _TABS[name]();
  const out = _modal.querySelector('#cv-out');
  out.hidden = true; out.textContent = '';
  _modal.querySelector('#cv-report-wrap')?.remove();
}

function _val(id) { return (_modal.querySelector('#' + id)?.value || '').trim(); }
function _set(id, v) { const el = _modal.querySelector('#' + id); if (el && v != null) el.value = v; }
function _recall(key) { try { return localStorage.getItem('cv:' + key) || ''; } catch { return ''; } }
function _remember(key, v) { try { if (v) localStorage.setItem('cv:' + key, v); } catch {} }

async function _inspect(btn) {
  const root = _val('cv-ds-root');
  const out = _modal.querySelector('#cv-out');
  if (!root) { out.hidden = false; out.classList.add('cv-err'); out.textContent = 'Enter a dataset root path.'; return; }
  out.hidden = false; out.classList.remove('cv-err'); out.textContent = 'Inspecting…';
  _busy(btn, true);
  try {
    const d = await _post('/api/cv/inspect', { root });
    _set('cv-ds-labels', d.labels_dir);
    _set('cv-ds-images', d.images_dir);
    if (d.num_classes != null) _set('cv-ds-nc', d.num_classes);
    if (d.class_names && d.class_names.length) _set('cv-ds-names', d.class_names.join(','));
    _remember('root', root);
    out.textContent = `Loaded: ${d.class_names?.length || 0} classes${d.yaml ? ' from ' + d.yaml.split(/[\\/]/).pop() : ''}` +
      (d.splits?.length ? `, splits: ${d.splits.join(', ')}` : '');
  } catch (e) {
    out.classList.add('cv-err'); out.textContent = 'Inspect failed: ' + e.message;
  } finally {
    _busy(btn, false);
  }
}

function _render(out, data) {
  out.hidden = false;
  out.classList.remove('cv-err');
  // A generated report → inline preview (iframe) + open-in-new-tab link.
  _modal.querySelector('#cv-report-wrap')?.remove();
  if (data && data.report_url) {
    const wrap = document.createElement('div');
    wrap.id = 'cv-report-wrap';
    wrap.innerHTML =
      `<a class="cv-report-link" href="${data.report_url}" target="_blank" rel="noopener">📊 Open report in new tab ↗</a>`
      + `<iframe class="cv-report-frame" src="${data.report_url}" title="report"></iframe>`;
    out.parentNode.insertBefore(wrap, out);
  }
  out.textContent = JSON.stringify(data, null, 2);
}

function _busy(btn, on) {
  if (!btn) return;
  if (on) {
    btn.dataset.label = btn.textContent;
    btn.disabled = true;
    btn.innerHTML = '<span class="cv-spin"></span> working…';
  } else {
    btn.disabled = false;
    if (btn.dataset.label) btn.textContent = btn.dataset.label;
  }
}

async function _run(action, btn) {
  const out = _modal.querySelector('#cv-out');
  out.hidden = false; out.classList.remove('cv-err'); out.textContent = 'Running…';
  _busy(btn, true);
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
        image_scan: _modal.querySelector('#cv-ds-scan')?.checked || false,
      };
      // 'eda' has its own endpoint that returns a report URL; others post to /dataset/<act>.
      data = await _post(act === 'eda' ? '/api/cv/eda' : `/api/cv/dataset/${act}`, payload);
    } else if (action === 'review') {
      const names = _val('cv-rv-names');
      data = await _post('/api/cv/review', {
        labels_dir: _val('cv-rv-gt'),
        preds_dir: _val('cv-rv-pa'),
        images_dir: _val('cv-rv-img'),
        preds_b_dir: _val('cv-rv-pb'),
        class_names: names ? names.split(',').map(s => s.trim()).filter(Boolean) : null,
        iou: parseFloat(_val('cv-rv-iou') || '0.5'),
      });
    } else if (action === 'aggregate') {
      data = await _post('/api/cv/eval-aggregate', { root: _val('cv-ag-root') });
    } else if (action === 'convert_data') {
      data = await _post('/api/cv/convert-data', {
        src: _val('cv-im-src'), format: _val('cv-im-fmt'), out_dir: _val('cv-im-out'),
      });
    } else if (action === 'stratified') {
      data = await _post('/api/cv/split-stratified', {
        labels_dir: _val('cv-sp-labels'), source_regex: _val('cv-sp-rx'),
        val_frac: parseFloat(_val('cv-sp-val') || '0.2'),
      });
    } else if (action === 'deploy') {
      const names = _val('cv-dp-names');
      data = await _post('/api/cv/deploy', {
        onnx: _val('cv-dp-onnx'),
        class_names: names ? names.split(',').map(s => s.trim()).filter(Boolean) : null,
        network_mode: parseInt(_val('cv-dp-mode') || '2', 10),
      });
    } else if (action === 'calibration') {
      data = await _post('/api/cv/calibration', {
        images_dir: _val('cv-cal-img'), n: parseInt(_val('cv-cal-n') || '200', 10),
        out_file: _val('cv-cal-out'),
      });
    } else if (action === 'compare_runs') {
      const dirs = _val('cv-rn-dirs').split(',').map(s => s.trim()).filter(Boolean);
      data = await _post('/api/cv/runs/compare', { run_dirs: dirs });
    } else if (action === 'drift') {
      data = await _post('/api/cv/drift', {
        baseline_preds: _val('cv-dr-base'), new_preds: _val('cv-dr-new'),
      });
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
    _loadHistory();
  } catch (e) {
    out.hidden = false; out.classList.add('cv-err'); out.textContent = 'Error: ' + e.message;
  } finally {
    _busy(btn, false);
  }
}

async function _loadHistory() {
  const el = _modal?.querySelector('#cv-history');
  if (!el) return;
  try {
    const res = await fetch('/api/cv/reports', { credentials: 'same-origin' });
    if (!res.ok) return;
    const { reports } = await res.json();
    if (!reports || !reports.length) { el.innerHTML = ''; return; }
    el.innerHTML = '<div class="cv-hist-h">Recent reports</div>'
      + reports.slice(0, 12).map(r =>
        `<a class="cv-hist-item" href="${r.url}" target="_blank" rel="noopener">${(r.title || r.id)}</a>`).join('');
  } catch { /* ignore */ }
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
  _loadHistory();
}

export function closeCv() {
  if (_modal) _modal.style.display = 'none';
  _open = false;
}

export function isCvOpen() { return _open; }

export default { openCv, closeCv, isCvOpen };
