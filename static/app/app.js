/* 钢带表面质检台 —— 前端单页应用
   数据全部来自 /api/*，不依赖模板引擎。 */

const app = document.getElementById('app');
const tipEl = document.getElementById('tip');

const STATUS = { pass: '合格', review: '建议复检', fail: '不合格' };
const SOURCE = { upload: '上传', web: '网页', batch: '批量', demo: '示例', api: '接口' };

const state = {
  view: 'detect',
  mode: 'single',
  params: { conf: 0.25, conf_high: 0.60, min_area: 0 },
  file: null,
  preview: '',
  files: [],
  filePreviews: [],
  result: null,
  resultNew: false,
  batch: null,
  busy: false,
  samples: [],
  model: null,
  history: { status: '', q: '', page: 1, data: null },
  stats: null,
};

let navToken = 0;
let searchTimer = 0;

/* ---------------------------------------------------------------- 工具 */

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

const pct = (v) => (Number(v) * 100).toFixed(0);

const fmtTs = (ts) => (ts ? String(ts).replace('T', ' ').slice(5, 16) : '—');

const fmtBox = (b) => `${Math.round(b[0])}, ${Math.round(b[1])} – ${Math.round(b[2])}, ${Math.round(b[3])}`;

function statusTag(s) {
  const k = STATUS[s] ? s : 'pass';
  return `<span class="status status--${k}"><i></i>${STATUS[k]}</span>`;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data;
  try {
    data = await res.json();
  } catch (e) {
    throw new Error(`服务返回了非 JSON 内容（HTTP ${res.status}）`);
  }
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
  }
  return data;
}

let toastTimer = 0;
function toast(message, kind = 'info') {
  document.querySelectorAll('.toast').forEach((n) => n.remove());
  const el = document.createElement('div');
  el.className = 'toast';
  el.dataset.kind = kind;
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  el.textContent = message;
  document.body.appendChild(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), 4200);
}

function fillStrip(m) {
  const set = (k, v) => {
    const el = document.querySelector(`[data-strip="${k}"]`);
    if (el) el.textContent = v;
  };
  if (m.model) set('model', m.model.name);
  set('ready', m.ready ? '已加载' : '未找到');
  set('imgsz', `${m.imgsz} × ${m.imgsz}`);
  set('device', m.model && m.model.device ? m.model.device : 'CPU');
  if (m.test_metrics) set('map', Number(m.test_metrics.mAP50).toFixed(3));
  const item = document.querySelector('[data-strip="ready"]');
  if (item) item.parentElement.classList.toggle('is-off', !m.ready);
}

/* ---------------------------------------------------------------- 路由 */

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [name, arg] = raw.split('/');
  return { name: name || 'detect', arg: arg || '' };
}

async function route() {
  const { name, arg } = parseHash();
  const token = ++navToken;
  state.view = name;

  document.querySelectorAll('#nav a').forEach((a) => {
    if (a.dataset.view === name) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });

  if (name === 'detect') {
    renderDetect();
  } else if (name === 'history') {
    await renderHistory(token);
  } else if (name === 'stats') {
    await renderStats(token);
  } else if (name === 'model') {
    await renderModel(token);
  } else if (name === 'record') {
    await renderRecord(arg, token);
  } else {
    location.replace('#/detect');
  }
  window.scrollTo(0, 0);
}

/* ---------------------------------------------------------------- 检测页 */

function renderDetect() {
  app.innerHTML = `
    <section class="detect">
      <aside class="intake panel">
        <div class="intake-head">
          <h1>送检</h1>
          <div class="seg" role="group" aria-label="检测模式">
            <button data-act="mode" data-mode="single"
              aria-pressed="${state.mode === 'single'}">单张</button>
            <button data-act="mode" data-mode="batch"
              aria-pressed="${state.mode === 'batch'}">批量</button>
          </div>
        </div>

        ${dropHTML()}

        <div class="params">
          ${paramHTML('conf', '检出阈值', '低于此值的框不输出', 0.01, 0.90, 0.01)}
          ${paramHTML('conf_high', '不合格阈值', '最高置信度达到即判不合格', 0.10, 0.99, 0.01)}
          ${paramHTML('min_area', '最小面积', '过滤小于该像素面积的框', 0, 4000, 10)}
        </div>

        <button class="btn btn--block" data-act="run" ${state.busy ? 'disabled' : ''}>
          ${state.busy ? '正在推理…' : (state.mode === 'batch' ? '开始批量检测' : '开始检测')}
        </button>

        <div class="samples">
          <div class="samples-head">
            <b>示例钢板</b>
            <span>点击直接检测</span>
          </div>
          <div class="sample-grid">
            ${state.samples.length ? state.samples.map(sampleHTML).join('')
              : '<p class="hint">示例图加载中…</p>'}
          </div>
        </div>
      </aside>

      <div class="stage" id="stage">${stageHTML()}</div>
    </section>`;

  bindDrop();
  fitViewport();

  if (state.resultNew) {
    setTimeout(() => { state.resultNew = false; }, 1600);
  }
}

/* 原图小于 360px 时按 360px 显示，否则保持原始比例 */
function fitViewport() {
  const fig = app.querySelector('.viewport');
  const img = fig && fig.querySelector('img');
  if (!fig || !img) return;
  const apply = () => {
    if (img.naturalWidth && img.naturalWidth < 360) fig.classList.add('is-small');
  };
  if (img.complete && img.naturalWidth) apply();
  else img.addEventListener('load', apply, { once: true });
}

function sampleHTML(s) {
  return `<button class="sample" data-act="sample" data-name="${esc(s.file)}"
            title="检测示例：${esc(s.zh)}">
      <img src="/static/samples/${esc(s.file)}" alt="${esc(s.zh)}" loading="lazy">
      <span>${esc(s.zh)}</span>
    </button>`;
}

function paramHTML(key, label, hint, min, max, step) {
  const v = state.params[key];
  return `<div class="param">
    <label for="p-${key}">${label}</label>
    <output id="o-${key}" for="p-${key}">${key === 'min_area' ? v : v.toFixed(2)}</output>
    <input type="range" id="p-${key}" data-param="${key}"
      min="${min}" max="${max}" step="${step}" value="${v}"
      aria-describedby="h-${key}" title="${hint}">
  </div>`;
}

function dropHTML() {
  if (state.mode === 'batch') {
    const n = state.files.length;
    return `<div class="drop ${n ? 'has-file' : ''}" id="drop" role="button" tabindex="0"
              aria-label="选择图片，支持一次多张">
      <input type="file" id="file" accept="image/*" multiple hidden>
      ${n ? `<div class="drop-grid">
               ${state.filePreviews.slice(0, 6).map((src) =>
                 `<img src="${src}" alt="">`).join('')}
               ${n > 6 ? `<span class="drop-more">+${n - 6}</span>` : ''}
             </div>
             <div class="drop-file">
               <b>已选择 ${n} 张图像</b>
               <button class="drop-clear" data-act="clear" type="button">清空</button>
             </div>`
          : `<svg class="drop-mark" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="1.4" aria-hidden="true">
               <rect x="3" y="4" width="18" height="16" rx="1"/>
               <path d="M3 15l5-4 4 3 3-2 6 4"/>
               <circle cx="9" cy="9" r="1.4"/>
             </svg>
             <p class="drop-title">拖入多张图像，或点击选择</p>
             <p class="drop-hint">一次最多 20 张，适合成批抽检</p>`}
    </div>`;
  }

  const has = !!state.file;
  return `<div class="drop ${has ? 'has-file' : ''}" id="drop" role="button" tabindex="0"
            aria-label="选择一张图片">
    <input type="file" id="file" accept="image/*" hidden>
    ${has ? `
      <img class="drop-preview" src="${state.preview}" alt="待检测图像预览">
      <div class="drop-file">
        <b>${esc(state.file.name)}</b>
        <button class="drop-clear" data-act="clear" type="button">更换</button>
      </div>`
    : `
      <svg class="drop-mark" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.4" aria-hidden="true">
        <rect x="3" y="4" width="18" height="16" rx="1"/>
        <path d="M3 15l5-4 4 3 3-2 6 4"/>
        <circle cx="9" cy="9" r="1.4"/>
      </svg>
      <p class="drop-title">拖入图像，或点击选择</p>
      <p class="drop-hint">JPG / PNG / BMP / TIF　单张不超过 64 MB</p>`}
  </div>`;
}

function stageHTML() {
  if (state.busy) {
    return `<div class="sheet loading-stage">
      <div class="scan-bar"></div>
      <p class="loading-title">正在推理</p>
      <p class="loading-text">${state.mode === 'batch' ? '逐张检测，稍候' : '模型正在扫描钢板表面'}</p>
    </div>`;
  }
  if (state.mode === 'batch') {
    return state.batch ? batchSheetHTML(state.batch) : idleHTML(
      '批量抽检',
      '选择多张钢板图像一次送检，结果按判定分类列出，可逐条打开查看标注图。',
    );
  }
  if (state.result) return sheetHTML(state.result, state.resultNew);
  return idleHTML(
    '等待送检',
    '把一张钢板表面图像拖到左侧，或点选示例钢板。系统会框出缺陷位置，并给出合格、建议复检或不合格判定。',
  );
}

function idleHTML(title, text) {
  return `<div class="sheet sheet--idle">
    <div class="idle-stage">
      <div class="idle-plate" aria-hidden="true">
        <div class="idle-ruler"></div>
        <div class="idle-screen">
          <span class="corner c-tl"></span><span class="corner c-tr"></span>
          <span class="corner c-bl"></span><span class="corner c-br"></span>
          <span class="idle-word">${title}</span>
        </div>
        <div class="vcap">
          <span><b>输入</b> 320 × 320</span>
          <span><b>类别</b> 6</span>
        </div>
      </div>
      <p class="idle-text">${text}</p>
    </div>
  </div>`;
}

function sheetHTML(r, isNew) {
  const defects = r.defects || [];
  const n = defects.length;
  const imgsz = r.image_size ? `${r.image_size[0]} × ${r.image_size[1]}` : '—';

  const rows = defects.map((d, i) => `
    <tr>
      <td class="idx">${String(i + 1).padStart(2, '0')}</td>
      <td>${esc(d.name_zh)}<span class="lat">${esc(d.name)}</span></td>
      <td class="num">
        <div class="conf">
          <div class="conf-track"><div class="conf-fill" style="width:${Math.round(d.conf * 100)}%"></div></div>
          <span class="conf-val">${Number(d.conf).toFixed(2)}</span>
        </div>
      </td>
      <td class="hide-sm pos">${fmtBox(d.bbox)}</td>
      <td class="num hide-sm">${Math.round(d.area)}</td>
    </tr>`).join('');

  return `<article class="sheet ${isNew ? 'is-new' : ''}" data-status="${r.status}">
    <header class="sheet-head">
      <div class="sheet-lead">
        <div class="verdict">
          <div class="verdict-word">${esc(r.label)}</div>
          <div class="verdict-sub">缺陷 ${n} 处</div>
        </div>
        <p class="verdict-reason">${esc(r.reason)}</p>
      </div>
      <dl class="facts">
        <div><dt>缺陷</dt><dd>${n}</dd></div>
        <div><dt>耗时</dt><dd>${Math.round(r.elapsed_ms)}<small>ms</small></dd></div>
        <div><dt>尺寸</dt><dd>${imgsz.replace(' × ', '<small>×</small>')}</dd></div>
        <div><dt>记录</dt><dd>#${String(r.record_id).padStart(5, '0')}</dd></div>
      </dl>
    </header>

    <figure class="viewport ${isNew ? 'is-new' : ''}">
      <div class="ruler"></div>
      <div class="plate">
        <img src="${esc(r.annotated_url)}" alt="缺陷标注图：${esc(r.filename)}">
        <span class="corner c-tl"></span><span class="corner c-tr"></span>
        <span class="corner c-bl"></span><span class="corner c-br"></span>
      </div>
      <figcaption class="vcap">
        <span><b>检测框</b> ${n}</span>
        <span><b>输入</b> ${imgsz}</span>
        <span class="hide-sm">${fmtTs(r.recorded_at || '')}</span>
      </figcaption>
    </figure>

    <section class="defects">
      <h2>缺陷明细</h2>
      ${n ? `<table class="grid">
        <thead>
          <tr>
            <th class="idx">编号</th>
            <th>类型</th>
            <th class="num">置信度</th>
            <th class="hide-sm pos">位置 x1, y1 – x2, y2</th>
            <th class="num hide-sm">面积 px²</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`
      : `<p class="hint">未检出缺陷，判定为合格。</p>`}
    </section>

    <footer class="sheet-foot">
      <span class="sheet-note">判定依据：最高置信度与不合格阈值比较，详见模型页。</span>
      <span class="pager-btns">
        <a class="btn btn--ghost btn--sm" href="/report/${r.record_id}" target="_blank" rel="noopener">打印检测报告</a>
        <button class="btn btn--sm" data-act="again">检测下一张</button>
      </span>
    </footer>
  </article>`;
}

function batchSheetHTML(b) {
  const s = b.summary;
  const rows = b.results.map((r) => `
    <tr data-act="open" data-id="${r.record_id}">
      <td><img class="thumb" src="${esc(r.annotated_url)}" alt="" loading="lazy"></td>
      <td><span class="fname">${esc(r.filename)}</span></td>
      <td>${statusTag(r.status)}</td>
      <td class="num">${r.count}</td>
      <td class="num">${r.count ? Math.max(...r.defects.map((d) => d.conf)).toFixed(2) : '—'}</td>
      <td class="num hide-sm">${Math.round(r.elapsed_ms)} ms</td>
      <td class="rid">#${String(r.record_id).padStart(5, '0')}</td>
    </tr>`).join('');

  const segs = ['pass', 'review', 'fail']
    .filter((k) => s[k] > 0)
    .map((k) => {
      const w = (s[k] / s.total) * 100;
      return `<div class="stack-seg" data-s="${k}" style="width:${w}%"
                title="${STATUS[k]} ${s[k]} 张">${w > 12 ? `${STATUS[k]} ${s[k]}` : ''}</div>`;
    }).join('');

  return `<article class="sheet">
    <header class="sheet-head">
      <div class="sheet-lead">
        <div class="verdict" data-status="${s.fail ? 'fail' : (s.review ? 'review' : 'pass')}"
             style="color:var(--${s.fail ? 'fail' : (s.review ? 'review' : 'pass')});
                    border-color:currentColor;background:transparent">
          <div class="verdict-word">${s.total} 张</div>
          <div class="verdict-sub">批量抽检</div>
        </div>
        <p class="verdict-reason">合格 ${s.pass} 张，建议复检 ${s.review} 张，不合格 ${s.fail} 张。
          <span>合格率 ${s.pass_rate}%</span>
        </p>
      </div>
      <dl class="facts">
        <div><dt>送检</dt><dd>${s.total}</dd></div>
        <div><dt>合格</dt><dd>${s.pass}</dd></div>
        <div><dt>复检</dt><dd>${s.review}</dd></div>
        <div><dt>不合格</dt><dd>${s.fail}</dd></div>
      </dl>
    </header>

    <div class="stack" style="margin-top:18px">${segs}</div>

    <div class="records">
      <table class="records-t">
        <thead>
          <tr>
            <th style="width:52px">图</th>
            <th>文件</th>
            <th>判定</th>
            <th class="num" style="text-align:right">缺陷</th>
            <th class="num" style="text-align:right">最高置信度</th>
            <th class="num hide-sm" style="text-align:right">耗时</th>
            <th>记录</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>

    ${b.errors && b.errors.length
      ? `<p class="hint">${b.errors.map(esc).join('<br>')}</p>` : ''}

    <footer class="sheet-foot">
      <span class="sheet-note">点击任意一行查看标注图与检测报告。</span>
      <span class="pager-btns">
        <button class="btn btn--sm" data-act="again">继续检测</button>
      </span>
    </footer>
  </article>`;
}

/* 检测页交互 */

function bindDrop() {
  const drop = document.getElementById('drop');
  if (!drop) return;
  const input = document.getElementById('file');

  drop.addEventListener('click', (e) => {
    if (e.target.closest('[data-act="clear"]')) return;
    input && input.click();
  });

  drop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      input && input.click();
    }
  });

  ['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => {
    e.preventDefault();
    drop.classList.add('is-over');
  }));

  ['dragleave', 'dragend'].forEach((t) => drop.addEventListener(t, () => {
    drop.classList.remove('is-over');
  }));

  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('is-over');
    const files = [...(e.dataTransfer.files || [])].filter((f) => f.type.startsWith('image/'));
    if (!files.length) return toast('请拖入图像文件', 'error');
    acceptFiles(files);
  });

  input && input.addEventListener('change', () => {
    const files = [...input.files];
    if (files.length) acceptFiles(files);
    input.value = '';
  });
}

function acceptFiles(files) {
  if (state.mode === 'batch') {
    state.filePreviews.forEach((u) => URL.revokeObjectURL(u));
    state.files = files.slice(0, 20);
    state.filePreviews = state.files.map((f) => URL.createObjectURL(f));
    if (files.length > 20) toast('一次最多 20 张，已保留前 20 张', 'error');
    state.batch = null;
    renderDetect();
    return;
  }
  const file = files[0];
  if (state.preview) URL.revokeObjectURL(state.preview);
  state.file = file;
  state.preview = URL.createObjectURL(file);
  state.result = null;
  renderDetect();
}

function appendParams(fd) {
  Object.entries(state.params).forEach(([k, v]) => fd.append(k, v));
}

function resetDetect() {
  state.result = null;
  state.batch = null;
  state.file = null;
  state.files = [];
  state.filePreviews.forEach((u) => URL.revokeObjectURL(u));
  state.filePreviews = [];
  if (state.preview) URL.revokeObjectURL(state.preview);
  state.preview = '';
  renderDetect();
}

async function runDetect() {
  if (state.mode === 'batch') return runBatch();
  if (!state.file) return toast('先选择一张图像，或点选下方示例', 'error');

  state.busy = true;
  state.result = null;
  renderDetect();

  const fd = new FormData();
  fd.append('image', state.file);
  fd.append('source', 'web');
  appendParams(fd);

  try {
    state.result = await api('/api/predict', { method: 'POST', body: fd });
    state.resultNew = true;
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    state.busy = false;
    renderDetect();
  }
}

async function runBatch() {
  if (!state.files.length) return toast('先选择图像，可一次选多张', 'error');

  state.busy = true;
  state.batch = null;
  renderDetect();

  const fd = new FormData();
  state.files.forEach((f) => fd.append('images', f));
  appendParams(fd);

  try {
    state.batch = await api('/api/predict-batch', { method: 'POST', body: fd });
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    state.busy = false;
    renderDetect();
  }
}

async function runDemo(name) {
  state.busy = true;
  state.mode = 'single';
  state.result = null;
  renderDetect();

  const qs = new URLSearchParams(state.params).toString();
  try {
    state.result = await api(`/api/demo/${encodeURIComponent(name)}?${qs}`, { method: 'POST' });
    state.resultNew = true;
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    state.busy = false;
    renderDetect();
  }
}

/* ---------------------------------------------------------------- 历史 */

async function renderHistory(token) {
  const h = state.history;
  app.innerHTML = `<div class="records"><p class="empty">读取记录…</p></div>`;

  const qs = new URLSearchParams({ page: h.page });
  if (h.status) qs.set('status', h.status);
  if (h.q) qs.set('q', h.q);

  let data;
  try {
    data = await api(`/api/history?${qs}`);
  } catch (e) {
    if (token === navToken) app.innerHTML = `<div class="records"><p class="empty">${esc(e.message)}</p></div>`;
    return;
  }
  if (token !== navToken) return;

  h.data = data;
  const rows = data.records.map((r) => `
    <tr data-act="open" data-id="${r.id}">
      <td class="rid">#${String(r.id).padStart(5, '0')}</td>
      <td><span class="fcell"><span class="fname">${esc(r.filename)}</span>
        <span class="fsrc hide-sm">${SOURCE[r.source] || r.source}</span></span></td>
      <td>${statusTag(r.status)}</td>
      <td class="num" style="text-align:right">${r.n_defects}</td>
      <td class="num hide-sm" style="text-align:right">
        ${r.n_defects ? Number(r.top_conf).toFixed(2) : '—'}</td>
      <td class="hide-sm">${esc(r.top_class) || '—'}</td>
      <td class="num hide-sm" style="text-align:right">${Math.round(r.elapsed_ms)} ms</td>
      <td class="num" style="text-align:right">${fmtTs(r.ts)}</td>
    </tr>`).join('');

  app.innerHTML = `
    <div class="toolbar">
      <div class="seg" role="group" aria-label="按判定筛选">
        ${[['', '全部'], ['pass', '合格'], ['review', '建议复检'], ['fail', '不合格']]
          .map(([k, label]) => `<button data-act="fstatus" data-status="${k}"
              aria-pressed="${h.status === k}">${label}</button>`).join('')}
      </div>
      <form class="search" data-act="search" role="search">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" aria-hidden="true">
          <circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>
        </svg>
        <input type="search" name="q" value="${esc(h.q)}" placeholder="按文件名查找"
               aria-label="按文件名查找">
      </form>
    </div>

    <div class="records">
      ${data.records.length ? `<table class="records-t">
        <thead>
          <tr>
            <th>记录</th>
            <th>文件</th>
            <th>判定</th>
            <th class="num" style="text-align:right">缺陷</th>
            <th class="num hide-sm" style="text-align:right">最高置信度</th>
            <th class="hide-sm">主要类型</th>
            <th class="num hide-sm" style="text-align:right">耗时</th>
            <th class="num" style="text-align:right">时间</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`
      : `<p class="empty">${h.q || h.status ? '没有符合条件的记录。' : '还没有检测记录，先去检测一张。'}</p>`}
    </div>

    <div class="pager">
      <span>共 ${data.total} 条，第 ${data.page} / ${data.pages} 页</span>
      <span class="pager-btns">
        <button class="btn btn--ghost btn--sm" data-act="page" data-page="${data.page - 1}"
          ${data.page <= 1 ? 'disabled' : ''}>上一页</button>
        <button class="btn btn--ghost btn--sm" data-act="page" data-page="${data.page + 1}"
          ${data.page >= data.pages ? 'disabled' : ''}>下一页</button>
      </span>
    </div>`;

  const form = app.querySelector('[data-act="search"]');
  const field = form.querySelector('input');
  form.addEventListener('submit', (e) => e.preventDefault());
  field.addEventListener('input', () => {
    clearTimeout(searchTimer);
    const q = field.value.trim();
    searchTimer = setTimeout(() => {
      state.history.q = q;
      state.history.page = 1;
      state.history.focus = true;
      renderHistory(++navToken);
    }, 280);
  });
  if (state.history.focus) {
    state.history.focus = false;
    field.focus();
    field.setSelectionRange(field.value.length, field.value.length);
  }
}

/* ---------------------------------------------------------------- 记录详情 */

async function renderRecord(rid, token) {
  app.innerHTML = `<div class="records"><p class="empty">读取记录…</p></div>`;
  let r;
  try {
    r = await api(`/api/record/${encodeURIComponent(rid)}`);
  } catch (e) {
    if (token === navToken) {
      app.innerHTML = `<div class="records"><p class="empty">${esc(e.message)}</p></div>`;
    }
    return;
  }
  if (token !== navToken) return;

  app.innerHTML = `
    <div class="toolbar">
      <a class="btn btn--ghost btn--sm" href="#/history">返回历史记录</a>
      <span class="note" style="font-size:11.5px;color:var(--muted)">
        记录 #${String(r.record_id).padStart(5, '0')}　${esc(r.recorded_at.replace('T', ' '))}
      </span>
    </div>
    ${sheetHTML(r, false)}`;
  fitViewport();
}

/* ---------------------------------------------------------------- 统计 */

async function renderStats(token) {
  app.innerHTML = `<div class="records"><p class="empty">读取统计…</p></div>`;
  let s;
  try {
    s = await api('/api/stats');
  } catch (e) {
    if (token === navToken) app.innerHTML = `<div class="records"><p class="empty">${esc(e.message)}</p></div>`;
    return;
  }
  if (token !== navToken) return;
  state.stats = s;

  const peak = Math.max(...s.days.map((d) => d.count), 1);
  const cols = s.days.map((d) => {
    const h = d.count ? Math.max(4, Math.round((d.count / peak) * 128)) : 2;
    return `<div class="col ${d.count ? '' : 'is-zero'}" data-tip="${d.date}　${d.count} 次检测">
      <span class="col-val">${d.count || ''}</span>
      <div class="col-bar" style="height:${h}px"></div>
    </div>`;
  }).join('');

  const axis = s.days.map((d) => `<span>${d.label}</span>`).join('');

  const maxClass = Math.max(...s.classes.map((c) => c.c), 1);
  const bars = s.classes.length ? s.classes.map((c) => `
    <div class="bar-row" data-tip="${esc(c.name_zh)}　${c.c} 个框 / ${c.records} 张图">
      <span class="bar-label">${esc(c.name_zh)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(c.c / maxClass) * 100}%"></div></div>
      <span class="bar-val"><b>${c.c}</b></span>
    </div>`).join('')
    : `<p class="empty">还没有缺陷数据。</p>`;

  const total = s.total || 0;
  const segs = ['pass', 'review', 'fail'].filter((k) => s[k] > 0).map((k) => {
    const w = (s[k] / total) * 100;
    return `<div class="stack-seg" data-s="${k}" style="width:${w}%"
      data-tip="${STATUS[k]}　${s[k]} 条　${w.toFixed(0)}%">
      ${w > 12 ? `${STATUS[k]} ${s[k]}` : ''}</div>`;
  }).join('');

  const legend = ['pass', 'review', 'fail'].map((k) => `
    <span class="legend-item">
      <i style="background:var(--${k})"></i>${STATUS[k]}<b>${s[k]}</b>
    </span>`).join('');

  app.innerHTML = `
    <div class="kpis">
      <dl class="kpi"><dt>累计检测</dt><dd>${s.total}<small>次</small></dd></dl>
      <dl class="kpi"><dt>合格率</dt><dd>${s.pass_rate.toFixed(1)}<small>%</small></dd></dl>
      <dl class="kpi"><dt>平均耗时</dt><dd>${Math.round(s.avg_ms)}<small>ms / 张</small></dd></dl>
      <dl class="kpi"><dt>平均缺陷</dt><dd>${s.avg_defects.toFixed(1)}<small>处 / 张</small></dd></dl>
    </div>

    <div class="stats-grid">
      <section class="block">
        <div class="block-head"><h2>近 7 天检测量</h2><span class="note">单位：张</span></div>
        <div class="cols">${cols}</div>
        <div class="col-axis">${axis}</div>
      </section>

      <section class="block">
        <div class="block-head"><h2>判定占比</h2><span class="note">共 ${s.total} 条</span></div>
        ${total ? `<div class="stack">${segs}</div><div class="legend">${legend}</div>`
          : `<p class="empty">还没有检测记录。</p>`}
        <p class="tbl-note">合格率按「合格 / 全部记录」计算，建议复检不计入合格。</p>
      </section>
    </div>

    <section class="block">
      <div class="block-head"><h2>缺陷类型分布</h2><span class="note">按标注框数量统计</span></div>
      <div class="bars">${bars}</div>
    </section>`;

  bindTips();
}

/* ---------------------------------------------------------------- 模型 */

async function renderModel(token) {
  if (!state.model) {
    app.innerHTML = `<div class="records"><p class="empty">读取模型信息…</p></div>`;
    try {
      state.model = await api('/api/model');
    } catch (e) {
      if (token === navToken) app.innerHTML = `<div class="records"><p class="empty">${esc(e.message)}</p></div>`;
      return;
    }
  }
  if (token !== navToken) return;
  const m = state.model;
  const M = m.model, D = m.dataset, T = m.test_metrics, TR = m.train;

  const cls = m.per_class.map((c) => `
    <tr>
      <td>${esc(c.zh)}<span class="lat">${esc(c.name)}</span></td>
      <td class="num" style="text-align:right">
        <div class="conf">
          <div class="conf-track"><div class="conf-fill" style="width:${Math.round(c.map50 * 100)}%"></div></div>
          <span class="conf-val">${c.map50.toFixed(3)}</span>
        </div>
      </td>
      <td class="num" style="text-align:right">${c.p.toFixed(3)}</td>
      <td class="num" style="text-align:right">${c.r.toFixed(3)}</td>
    </tr>`).join('');

  const cmp = m.compare.map((c) => `
    <tr class="${c.model === M.name ? 'is-pick' : ''}">
      <td>${esc(c.model)}${c.model === M.name ? '<span class="tag" style="margin-left:8px">部署</span>' : ''}</td>
      <td class="num" style="text-align:right">${c.params_m.toFixed(2)}</td>
      <td class="num" style="text-align:right">${c.size_mb.toFixed(1)}</td>
      <td class="num" style="text-align:right">${c.map50.toFixed(3)}</td>
      <td class="num" style="text-align:right">${c.map50_95.toFixed(3)}</td>
      <td class="num" style="text-align:right">${c.p.toFixed(3)}</td>
      <td class="num" style="text-align:right">${c.r.toFixed(3)}</td>
    </tr>`).join('');

  app.innerHTML = `
    <div class="model-grid">
      <section class="block">
        <div class="block-head"><h2>部署配置</h2>
          <span class="note">${m.ready ? '权重已加载' : '权重未找到'}</span></div>
        <dl class="spec">
          <div class="spec-row"><dt>模型</dt><dd>${esc(M.name)}</dd></div>
          <div class="spec-row"><dt>框架</dt><dd>${esc(M.framework)}</dd></div>
          <div class="spec-row"><dt>参数量</dt><dd>${M.params_m} M</dd></div>
          <div class="spec-row"><dt>权重大小</dt><dd>${M.size_mb} MB</dd></div>
          <div class="spec-row"><dt>推理尺寸</dt><dd>${m.imgsz} × ${m.imgsz}</dd></div>
          <div class="spec-row"><dt>训练轮次</dt><dd>${TR.epochs} 轮（最佳 ${TR.best_epoch}）</dd></div>
          <div class="spec-row"><dt>批大小</dt><dd>${TR.batch}</dd></div>
          <div class="spec-row"><dt>运行设备</dt><dd>${esc(M.device)}</dd></div>
          <div class="spec-row"><dt>权重路径</dt><dd style="font-size:11px">${esc(m.weights)}</dd></div>
        </dl>
      </section>

      <section class="block">
        <div class="block-head"><h2>测试集指标</h2><span class="note">${D.test} 张 / 独立测试集</span></div>
        <div class="metric-row">
          <dl class="metric metric--hero"><dt>mAP50</dt><dd>${T.mAP50.toFixed(3)}</dd></dl>
          <dl class="metric"><dt>mAP50-95</dt><dd>${T.mAP50_95.toFixed(3)}</dd></dl>
          <dl class="metric"><dt>Precision</dt><dd>${T.precision.toFixed(3)}</dd></dl>
          <dl class="metric"><dt>Recall</dt><dd>${T.recall.toFixed(3)}</dd></dl>
        </div>
        <dl class="spec" style="margin-top:14px">
          <div class="spec-row"><dt>数据集</dt><dd>${esc(D.name)}</dd></div>
          <div class="spec-row"><dt>图像</dt><dd>${esc(D.resolution)}</dd></div>
          <div class="spec-row"><dt>划分</dt><dd>${D.train} / ${D.val} / ${D.test}</dd></div>
          <div class="spec-row"><dt>标注框</dt><dd>${D.boxes}</dd></div>
          <div class="spec-row"><dt>训练</dt><dd>${TR.epochs} 轮 · ${esc(TR.gpu)}</dd></div>
        </dl>
      </section>
    </div>

    <section class="block" style="margin-bottom:18px">
      <div class="block-head"><h2>每类表现</h2><span class="note">测试集 mAP50 / P / R</span></div>
      <table class="grid">
        <thead><tr>
          <th>类别</th><th class="num" style="text-align:right">mAP50</th>
          <th class="num" style="text-align:right">Precision</th>
          <th class="num" style="text-align:right">Recall</th>
        </tr></thead>
        <tbody>${cls}</tbody>
      </table>
      <p class="tbl-note">龟裂是细密网状微裂纹，在 200×200 灰度图上与轧制纹理高度相似，是 NEU-DET 公认最难的一类，也是本系统主要的漏检来源。</p>
    </section>

    <section class="block" style="margin-bottom:18px">
      <div class="block-head"><h2>骨干对比</h2><span class="note">同一套超参，只换骨干</span></div>
      <table class="grid">
        <thead><tr>
          <th>模型</th><th class="num" style="text-align:right">参数量 M</th>
          <th class="num" style="text-align:right">大小 MB</th>
          <th class="num" style="text-align:right">mAP50</th>
          <th class="num" style="text-align:right">mAP50-95</th>
          <th class="num" style="text-align:right">P</th>
          <th class="num" style="text-align:right">R</th>
        </tr></thead>
        <tbody>${cmp}</tbody>
      </table>
      <p class="tbl-note">三者差距在 1–3 个百分点，属于单次训练的波动范围。YOLO11s 参数量最少，精度损失可忽略，因此选作部署模型。</p>
    </section>

    <div class="model-grid">
      <section class="block">
        <div class="block-head"><h2>判定规则</h2></div>
        <ul class="rules">
          ${m.judge_rules.map((r) => `<li>
            <span class="cond">${esc(r.cond)}</span>
            ${statusTag(r.status)}
          </li>`).join('')}
        </ul>
      </section>

      <section class="block">
        <div class="block-head"><h2>接口调用</h2><span class="note">POST /api/predict</span></div>
        <pre class="code"><span class="c"># 返回 JSON：判定、缺陷框、耗时</span>
curl -X POST http://localhost:5000/api/predict \\
  -F "image=@plate.jpg" \\
  -F "conf=0.25" \\
  -F "conf_high=0.60"</pre>
        <p class="tbl-note">也支持 JSON + base64：字段 <code>image_base64</code>。</p>
      </section>
    </div>`;
}

/* ---------------------------------------------------------------- 图表提示 */

function bindTips() {
  app.querySelectorAll('[data-tip]').forEach((el) => {
    el.addEventListener('mouseenter', () => {
      tipEl.hidden = false;
      tipEl.textContent = el.dataset.tip;
    });
    el.addEventListener('mousemove', (e) => {
      tipEl.style.left = `${e.clientX}px`;
      tipEl.style.top = `${e.clientY - 8}px`;
    });
    el.addEventListener('mouseleave', () => { tipEl.hidden = true; });
  });
}

/* ---------------------------------------------------------------- 全局事件 */

app.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-act]');
  if (!btn) return;
  const act = btn.dataset.act;

  if (act === 'mode') {
    state.mode = btn.dataset.mode;
    state.result = null;
    state.batch = null;
    renderDetect();
  } else if (act === 'run') {
    runDetect();
  } else if (act === 'clear') {
    if (state.mode === 'batch') {
      state.files = [];
      state.filePreviews.forEach((u) => URL.revokeObjectURL(u));
      state.filePreviews = [];
      state.batch = null;
      renderDetect();
    } else {
      resetDetect();
    }
  } else if (act === 'sample') {
    runDemo(btn.dataset.name);
  } else if (act === 'again') {
    resetDetect();
  } else if (act === 'open') {
    const id = btn.dataset.id || btn.closest('[data-id]')?.dataset.id;
    if (id) location.hash = `#/record/${id}`;
  } else if (act === 'fstatus') {
    state.history.status = btn.dataset.status;
    state.history.page = 1;
    renderHistory(++navToken);
  } else if (act === 'page') {
    state.history.page = Number(btn.dataset.page);
    renderHistory(++navToken);
  }
});

app.addEventListener('input', (e) => {
  const input = e.target.closest('[data-param]');
  if (!input) return;
  const key = input.dataset.param;
  const v = Number(input.value);
  state.params[key] = v;
  const out = document.getElementById(`o-${key}`);
  if (out) out.textContent = key === 'min_area' ? String(v) : v.toFixed(2);
});

window.addEventListener('hashchange', route);

/* ---------------------------------------------------------------- 启动 */

async function boot() {
  try {
    const m = await api('/api/model');
    state.model = m;
    fillStrip(m);
  } catch (e) {
    toast(e.message, 'error');
  }

  try {
    const s = await api('/api/samples');
    state.samples = s.samples || [];
  } catch (e) {
    state.samples = [];
  }

  route();
}

boot();
