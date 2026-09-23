// Cat or Alligator? Main app: camera -> model -> layer visualisations -> verdict.
import { EfficientNetLite, CAT_CLASSES, GATOR_CLASSES } from './model.js';
import { IMAGENET_CLASSES } from './imagenet-classes.js';
import { STAGES, TILES_PER_STAGE, TILES_IN_DETAIL } from './stages.js';
import * as R from './render.js';

const tf = globalThis.tf;
const $ = (sel, el = document) => el.querySelector(sel);
const TAPS = STAGES.map(s => s.tap);
const INPUT = 224;

// ---------------------------------------------------------------- settings
const DEFAULTS = { mirror: true, zoom: 0.75, heat: true, mode: 'important', none: 0.12, grad: 'auto', stats: false, camera: '' };
const settings = { ...DEFAULTS };
try { Object.assign(settings, JSON.parse(localStorage.getItem('cat-or-gator-settings') || '{}')); } catch { /* storage unavailable */ }
function saveSettings() { try { localStorage.setItem('cat-or-gator-settings', JSON.stringify(settings)); } catch { /* ignore */ } }

// ---------------------------------------------------------------- state
let model = null;
let stream = null;
let source = 'camera';      // 'camera' | 'photo'
let frozen = false;
let running = false;
let frameNo = 0;
let lastNeitherSince = 0;
const timing = { analyse: 0, forward: 0, frame: 0, fps: 0, lastEnd: 0 };

const video = $('#video');
const still = $('#still');
const overlay = $('#cam-overlay');
const inputCanvas = document.createElement('canvas');
inputCanvas.width = inputCanvas.height = INPUT;
const inputCtx = inputCanvas.getContext('2d', { willReadFrequently: true });

const probsEma = new Float32Array(1000);
let probsInit = false;
const stages = [];            // per-stage runtime state + DOM
let detail = { open: false, stage: -1, sel: [], spot: -1, els: [] };

// ---------------------------------------------------------------- helpers
const clamp01 = v => Math.max(0, Math.min(1, v));
const pct = v => (v >= 0.995 ? '>99' : v < 0.005 ? '0' : Math.round(v * 100)) + '%';
const FRIENDLY = {
  281: 'Tabby cat', 282: 'Tiger cat', 283: 'Persian cat', 284: 'Siamese cat', 285: 'Egyptian cat',
  286: 'Cougar', 287: 'Lynx', 288: 'Leopard', 289: 'Snow leopard', 290: 'Jaguar', 291: 'Lion', 292: 'Tiger', 293: 'Cheetah',
  49: 'Nile crocodile', 50: 'American alligator',
};
function className(i) {
  if (FRIENDLY[i]) return FRIENDLY[i];
  const n = IMAGENET_CLASSES[i].split(',')[0].trim();
  return n.charAt(0).toUpperCase() + n.slice(1);
}
const CAT_SET = new Set(CAT_CLASSES), GATOR_SET = new Set(GATOR_CLASSES);

function argsortDesc(arr) {
  const idx = Array.from(arr, (_, i) => i);
  idx.sort((a, b) => arr[b] - arr[a]);
  return idx;
}

// Keep each detector in the same tile slot while it stays near the top,
// so tiles don't shuffle around every frame.
function stableTopK(scores, current, k, margin) {
  const order = argsortDesc(scores);
  const pool = new Set(order.slice(0, k + margin));
  const next = new Array(k).fill(-1);
  const used = new Set();
  for (let s = 0; s < k; s++) {
    const c = current[s];
    if (c !== undefined && c >= 0 && pool.has(c) && !used.has(c)) { next[s] = c; used.add(c); }
  }
  let oi = 0;
  for (let s = 0; s < k; s++) {
    if (next[s] >= 0) continue;
    while (used.has(order[oi])) oi++;
    next[s] = order[oi]; used.add(order[oi]);
  }
  return next;
}

function voteColour(v, maxAbs) {
  const s = maxAbs > 0 ? Math.abs(v) / maxAbs : 0;
  if (s < 0.12) return '#2a3350';
  const rgb = v > 0 ? R.CAT_RGB : R.GATOR_RGB;
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.45 + 0.55 * clamp01(s)).toFixed(2)})`;
}

function leaning(share) {
  const d = share - 0.5;
  if (Math.abs(d) < 0.12) return ['none', 'Mixed'];
  const who = d > 0 ? 'cat' : 'alligator';
  return [d > 0 ? 'cat' : 'gator', (Math.abs(d) > 0.3 ? 'Mostly ' : 'Leaning ') + who];
}

// ---------------------------------------------------------------- build the layer cards
function buildPipeline() {
  const root = $('#pipeline');
  root.innerHTML = '';
  STAGES.forEach((st, i) => {
    const [, H, W, C] = model.tapShapes[st.tap];
    const layer = model.layerIndex[st.tap];
    const el = document.createElement('article');
    el.className = 'stage';
    el.tabIndex = 0;
    el.setAttribute('aria-label', `Layer ${layer}: ${st.title}. Click to see more detectors.`);
    el.innerHTML = `
      <div class="stage-tag">Layer ${layer} of ${model.totalLayers}</div>
      <h3>${st.title}</h3>
      <div class="stage-meta"><b>${W}×${H}</b> grid · <b>${C.toLocaleString()}</b> detectors<span class="rf"><br>Each square sees ${st.short.charAt(0).toLowerCase() + st.short.slice(1)}</span></div>
      <div class="tiles">${Array.from({ length: TILES_PER_STAGE }, () => '<div class="tile"><canvas></canvas><i class="vote"></i><span class="cid"></span></div>').join('')}</div>
      <div class="sub-label">Evidence found at this layer</div>
      <canvas class="evidence" width="224" height="224"></canvas>
      <div class="tally"><div class="tally-bar"><i style="width:50%"></i></div><div class="tally-text">…</div></div>
      <p class="blurb">${st.blurb}</p>`;
    el.addEventListener('click', (e) => {
      const tile = e.target.closest('.tile');
      const k = tile ? [...el.querySelectorAll('.tile')].indexOf(tile) : -1;
      openDetail(i, k >= 0 ? stages[i].sel[k] : -1);
    });
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter') openDetail(i, -1); });
    root.append(el);
    stages[i] = {
      H, W, C, layer,
      el,
      tiles: [...el.querySelectorAll('.tile')].map(t => ({ el: t, canvas: $('canvas', t), vote: $('.vote', t), cid: $('.cid', t) })),
      evCanvas: $('.evidence', el),
      tallyFill: $('.tally-bar i', el),
      tallyText: $('.tally-text', el),
      emaVote: new Float32Array(C), emaMean: new Float32Array(C), hasVote: false, init: false,
      voteMaxAbs: 0, mapMax: 0, evMax: 0, share: 0.5,
      evidence: new Float32Array(H * W),
      sel: [],
    };
  });
  // Build-up chart in the verdict panel
  $('#buildup').innerHTML = STAGES.map((st, i) =>
    `<div class="bu" title="Layer ${stages[i].layer}: ${st.title}"><div class="up"><i style="height:0"></i></div><div class="down"><i style="height:0"></i></div><span>L${stages[i].layer}</span></div>`).join('');
}

function resetSmoothing() {
  probsInit = false;
  for (const s of stages) { s.init = false; s.hasVote = false; }
}

// ---------------------------------------------------------------- camera & photos
async function startCamera() {
  stopCamera();
  showCamMessage('');
  if (!navigator.mediaDevices?.getUserMedia) {
    showCamMessage(window.isSecureContext
      ? 'This browser can\'t use a camera. Try Chrome or Edge, or use a photo instead.'
      : 'The camera only works when the page is opened from https:// or http://localhost (e.g. with start-demo.bat).', true);
    return false;
  }
  const video_c = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if (settings.camera) video_c.deviceId = { exact: settings.camera }; else video_c.facingMode = 'user';
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: video_c, audio: false });
  } catch (err) {
    if (settings.camera && (err.name === 'OverconstrainedError' || err.name === 'NotFoundError')) {
      settings.camera = ''; saveSettings(); return startCamera();
    }
    const msg = err.name === 'NotAllowedError'
      ? 'Camera access was blocked. Click the camera icon in the address bar to allow it, then reload. Or use a photo instead.'
      : err.name === 'NotFoundError' ? 'No camera found. Plug in a webcam and reload, or use a photo instead.'
      : `Couldn't start the camera (${err.name}). Try reloading, or use a photo instead.`;
    showCamMessage(msg, true);
    return false;
  }
  video.srcObject = stream;
  await video.play().catch(() => {});
  applyMirror();
  listCameras();
  return true;
}
function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
}
async function listCameras() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    const sel = $('#set-camera');
    const active = stream?.getVideoTracks()[0]?.getSettings().deviceId;
    sel.innerHTML = devs.map((d, i) => `<option value="${d.deviceId}">${d.label || 'Camera ' + (i + 1)}</option>`).join('');
    sel.value = settings.camera || active || '';
  } catch { /* ignore */ }
}
function showCamMessage(text, withPhotoButton = false) {
  const m = $('#cam-message');
  if (!text) { m.hidden = true; return; }
  m.hidden = false;
  m.innerHTML = `<div><p>${text}</p>${withPhotoButton ? '<button class="btn primary" id="msg-photo">Use a photo</button>' : ''}</div>`;
  $('#msg-photo', m)?.addEventListener('click', () => $('#dlg-photo').showModal());
}
function applyMirror() { video.classList.toggle('mirror', settings.mirror); }

async function loadPhoto(src) {
  still.src = src;
  try { await still.decode(); } catch { showCamMessage('Sorry, that file couldn\'t be opened as a picture.'); return; }
  source = 'photo';
  frozen = false; updateFreezeUi();
  still.hidden = false; video.hidden = true;
  $('#photo-bar').hidden = false;
  showCamMessage('');
  resetSmoothing();
  $('#dlg-photo').close();
}
async function backToCamera() {
  source = 'camera';
  still.hidden = true; video.hidden = false;
  $('#photo-bar').hidden = true;
  frozen = false; updateFreezeUi();
  resetSmoothing();
  if (!stream) await startCamera();
}

// Copy the part of the picture inside the box into the 224x224 model input.
function grabFrame() {
  let src, sw, sh, zoom;
  if (source === 'camera') {
    if (!stream || video.readyState < 2 || !video.videoWidth) return false;
    src = video; sw = video.videoWidth; sh = video.videoHeight; zoom = settings.zoom;
  } else {
    if (!still.naturalWidth) return false;
    src = still; sw = still.naturalWidth; sh = still.naturalHeight; zoom = 1;
  }
  const s = Math.min(sw, sh) * zoom;
  inputCtx.save();
  if (source === 'camera' && settings.mirror) { inputCtx.translate(INPUT, 0); inputCtx.scale(-1, 1); }
  inputCtx.drawImage(src, (sw - s) / 2, (sh - s) / 2, s, s, 0, 0, INPUT, INPUT);
  inputCtx.restore();
  return true;
}

// ---------------------------------------------------------------- the main loop
function decideGrad() {
  frameNo++;
  if (!stages.some(s => s.hasVote)) return true;
  switch (settings.grad) {
    case 'every': return true;
    case 'half': return frameNo % 2 === 0;
    case 'quarter': return frameNo % 4 === 0;
    default:
      if (frozen || source === 'photo') return true;
      if (timing.analyse < 150) return true;
      if (timing.analyse < 320) return frameNo % 2 === 0;
      return frameNo % 4 === 0;
  }
}

async function step() {
  const t0 = performance.now();
  if (!frozen && !grabFrame()) return false;
  const live = source === 'camera' && !frozen;
  const doGrad = decideGrad();

  const x = model.preprocess(inputCanvas);
  let r;
  try {
    r = doGrad ? model.analyse(x, TAPS) : model.activations(x, TAPS);
  } finally { x.dispose(); }

  try {
    // --- 1. small summary numbers
    const statT = tf.tidy(() => {
      const o = { probs: tf.softmax(r.logits.squeeze([0])), means: r.acts.map(a => a.mean([0, 1, 2])) };
      if (doGrad) {
        const ga = r.acts.map((a, i) => a.mul(r.grads[i]));        // gradient x activation
        o.votes = ga.map(t => t.sum([0, 1, 2]));                     // one vote per detector
        o.evid = ga.map(t => t.sum(3).squeeze([0]));                 // one number per grid square
      }
      return o;
    });
    const list = [statT.probs, ...statT.means, ...(statT.votes || []), ...(statT.evid || [])];
    const data = await Promise.all(list.map(t => t.data()));
    list.forEach(t => t.dispose());
    const probs = data[0];
    const means = data.slice(1, 1 + TAPS.length);
    const votes = doGrad ? data.slice(1 + TAPS.length, 1 + 2 * TAPS.length) : null;
    const evid = doGrad ? data.slice(1 + 2 * TAPS.length) : null;

    // --- 2. smoothing (only while the live camera is moving)
    const a = live ? 0.45 : 1;
    for (let i = 0; i < 1000; i++) probsEma[i] = probsInit ? probsEma[i] + a * (probs[i] - probsEma[i]) : probs[i];
    probsInit = true;
    stages.forEach((s, i) => {
      const aa = live && s.init ? 0.4 : 1;
      const m = means[i];
      for (let c = 0; c < s.C; c++) s.emaMean[c] += aa * (m[c] - s.emaMean[c]);
      if (votes) {
        const v = votes[i];
        const av = live && s.hasVote ? 0.4 : 1;
        let pos = 0, neg = 0, mx = 0;
        for (let c = 0; c < s.C; c++) {
          s.emaVote[c] += av * (v[c] - s.emaVote[c]);
          if (v[c] > 0) pos += v[c]; else neg -= v[c];
          mx = Math.max(mx, Math.abs(s.emaVote[c]));
        }
        s.voteMaxAbs = mx;
        const share = pos + neg > 0 ? pos / (pos + neg) : 0.5;
        s.share = s.hasVote ? s.share + av * (share - s.share) : share;
        const e = evid[i];
        let em = 0;
        for (let k = 0; k < e.length; k++) { s.evidence[k] += (s.hasVote ? av : 1) * (e[k] - s.evidence[k]); em = Math.max(em, Math.abs(s.evidence[k])); }
        s.evMax = s.hasVote && live ? s.evMax + 0.3 * (em - s.evMax) : em;
        s.hasVote = true;
      }
      s.init = true;

      // --- 3. choose which detectors to show
      const useVotes = settings.mode === 'important' && s.hasVote;
      const score = useVotes ? s.emaVote.map(Math.abs) : s.emaMean;
      s.sel = stableTopK(score, s.sel, TILES_PER_STAGE, live ? 3 : 0);
      if (detail.open && detail.stage === i) {
        detail.sel = stableTopK(score, detail.sel, Math.min(TILES_IN_DETAIL, s.C), live ? 6 : 0);
      }
    });

    // --- 4. fetch the chosen detectors' maps
    const want = stages.map((s, i) => {
      const ids = [...s.sel];
      if (detail.open && detail.stage === i) {
        for (const c of detail.sel) if (!ids.includes(c)) ids.push(c);
        if (detail.spot >= 0 && !ids.includes(detail.spot)) ids.push(detail.spot);
      }
      return ids;
    });
    const mapT = tf.tidy(() => r.acts.map((act, i) => tf.gather(act.squeeze([0]), want[i], 2).transpose([2, 0, 1])));
    const maps = await Promise.all(mapT.map(t => t.data()));
    mapT.forEach(t => t.dispose());

    // --- 5. draw
    const t1 = performance.now();
    render(maps, want, live);
    const ms = t1 - t0;
    if (doGrad) timing.analyse = timing.analyse ? timing.analyse * 0.8 + ms * 0.2 : ms;
    else timing.forward = timing.forward ? timing.forward * 0.8 + ms * 0.2 : ms;
  } finally {
    [r.logits, r.score, ...r.acts, ...(r.grads || [])].forEach(t => t && t.dispose());
  }
  return true;
}

// ---------------------------------------------------------------- drawing
function render(maps, want, live) {
  const pc = CAT_CLASSES.reduce((s, i) => s + probsEma[i], 0);
  const pg = GATOR_CLASSES.reduce((s, i) => s + probsEma[i], 0);
  const both = pc + pg;
  const neither = both < settings.none;

  stages.forEach((s, i) => {
    const plane = s.H * s.W;
    const m = maps[i];
    // detector tiles
    let stageMax = 0;
    const tileMax = s.sel.map((c, k) => {
      let mx = 0; const o = k * plane;
      for (let p = 0; p < plane; p++) if (m[o + p] > mx) mx = m[o + p];
      stageMax = Math.max(stageMax, mx);
      return mx;
    });
    s.mapMax = live && s.mapMax ? s.mapMax + 0.3 * (stageMax - s.mapMax) : stageMax;
    s.sel.forEach((c, k) => {
      const t = s.tiles[k];
      const scale = 1 / Math.max(tileMax[k], 0.3 * s.mapMax, 1e-6);
      R.drawMap(t.canvas, m, s.W, s.H, { offset: k * plane, scale });
      const v = s.hasVote ? s.emaVote[c] : 0;
      t.el.style.borderColor = voteColour(v, s.voteMaxAbs);
      t.vote.style.width = (s.voteMaxAbs ? 100 * Math.abs(v) / s.voteMaxAbs : 0).toFixed(1) + '%';
      t.vote.style.color = v >= 0 ? 'var(--cat)' : 'var(--gator)';
      t.cid.textContent = '#' + c;
      t.el.title = `Detector #${c}` + (s.hasVote ? ` — voting ${v >= 0 ? 'cat' : 'alligator'}` : '');
    });
    // evidence
    R.drawEvidence(s.evCanvas, inputCanvas, s.hasVote ? s.evidence : null, s.W, s.H, s.evMax ? 1 / s.evMax : 0);
    // tally: how the detectors in this layer split their votes. The last
    // layer's votes are added up to make the decision, so show that directly.
    const isLast = i === stages.length - 1;
    const share = isLast ? (neither || both < 1e-6 ? 0.5 : pc / both) : s.share;
    const [kind, words] = leaning(share);
    s.tallyFill.style.width = (100 * share).toFixed(1) + '%';
    s.tallyText.innerHTML = !isLast ? `Detectors' votes: <b class="${kind}">${words}</b>`
      : neither ? 'Final vote: <b class="none">neither</b>'
      : `Final vote: <b class="${kind}">${share >= 0.5 ? 'cat' : 'alligator'} ${Math.round(100 * Math.max(share, 1 - share))}%</b>`;
    // build-up chart
    const bu = $('#buildup').children[i];
    bu.querySelector('.up i').style.height = (200 * Math.max(0, share - 0.5)).toFixed(1) + '%';
    bu.querySelector('.down i').style.height = (200 * Math.max(0, 0.5 - share)).toFixed(1) + '%';
  });

  if (detail.open) renderDetail(maps[detail.stage], want[detail.stage]);

  // camera overlay
  drawOverlay(neither ? 0.35 : 1);

  // verdict
  const top = argsortDesc(probsEma).slice(0, 5);
  const v = $('#verdict');
  let kind, word, sub;
  if (neither) {
    kind = 'none'; word = 'Not sure';
    sub = `I don't think that's a cat or an alligator. Maybe: ${className(top[0]).toLowerCase()}?`;
  } else {
    const share = pc / both;
    const bestCat = CAT_CLASSES.reduce((b, i) => probsEma[i] > probsEma[b] ? i : b, CAT_CLASSES[0]);
    const bestGator = GATOR_CLASSES.reduce((b, i) => probsEma[i] > probsEma[b] ? i : b, GATOR_CLASSES[0]);
    if (share > 0.65) { kind = 'cat'; word = 'CAT'; sub = `It thinks: ${className(bestCat).toLowerCase()}`; }
    else if (share < 0.35) { kind = 'gator'; word = 'ALLIGATOR'; sub = bestGator === 49 ? 'It thinks: crocodile (close enough!)' : 'It thinks: American alligator'; }
    else { kind = 'unsure'; word = 'Hmm…'; sub = 'It could be either!'; }
  }
  if (v.dataset.kind !== kind) v.dataset.kind = kind;
  $('#verdict-word').textContent = word;
  $('#verdict-sub').textContent = sub;
  const other = Math.max(0, 1 - both);
  $('#bar-cat').style.width = (100 * pc).toFixed(1) + '%';
  $('#bar-gator').style.width = (100 * pg).toFixed(1) + '%';
  $('#bar-other').style.width = (100 * other).toFixed(1) + '%';
  $('#pct-cat').textContent = pct(pc);
  $('#pct-gator').textContent = pct(pg);
  $('#pct-other').textContent = pct(other);
  $('#top5').innerHTML = top.map(i => {
    const cls = CAT_SET.has(i) ? 'cat' : GATOR_SET.has(i) ? 'gator' : '';
    return `<li class="${cls}"><span class="name">${className(i)}</span><span class="p">${pct(probsEma[i])}</span><span class="bar"><i style="width:${(100 * probsEma[i]).toFixed(1)}%"></i></span></li>`;
  }).join('');

  // prompt when nothing recognisable is in view for a while
  const now = performance.now();
  if (!neither || source !== 'camera' || frozen) lastNeitherSince = 0;
  else if (!lastNeitherSince) lastNeitherSince = now;
  $('#cam-prompt').hidden = !(lastNeitherSince && now - lastNeitherSince > 2500);
}

function camBox() {
  const rect = overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(rect.width * dpr), h = Math.round(rect.height * dpr);
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
  const zoom = source === 'camera' ? settings.zoom : 1;
  const sq = Math.min(w, h), s = sq * zoom;
  return { x: (w - s) / 2, y: (h - s) / 2, s, w, h, dpr };
}

function drawOverlay(heatStrength = 1) {
  const box = camBox();
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, box.w, box.h);
  // darken outside the box
  ctx.fillStyle = 'rgba(5,7,12,0.55)';
  ctx.beginPath();
  ctx.rect(0, 0, box.w, box.h);
  ctx.rect(box.x, box.y, box.s, box.s);
  ctx.fill('evenodd');
  // heat map: evidence from the deepest layer
  const last = stages[stages.length - 1];
  if (settings.heat && last?.hasVote && last.evMax > 0) {
    ctx.save(); ctx.globalAlpha = heatStrength;
    R.drawCameraHeat(ctx, last.evidence, last.W, last.H, 1 / last.evMax, box);
    ctx.restore();
  }
  // corner brackets
  const L = box.s * 0.12, lw = 3 * box.dpr;
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = lw; ctx.lineCap = 'round';
  ctx.beginPath();
  const x0 = box.x + lw / 2, y0 = box.y + lw / 2, x1 = box.x + box.s - lw / 2, y1 = box.y + box.s - lw / 2;
  ctx.moveTo(x0, y0 + L); ctx.lineTo(x0, y0); ctx.lineTo(x0 + L, y0);
  ctx.moveTo(x1 - L, y0); ctx.lineTo(x1, y0); ctx.lineTo(x1, y0 + L);
  ctx.moveTo(x1, y1 - L); ctx.lineTo(x1, y1); ctx.lineTo(x1 - L, y1);
  ctx.moveTo(x0 + L, y1); ctx.lineTo(x0, y1); ctx.lineTo(x0, y1 - L);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1 * box.dpr;
  ctx.strokeRect(box.x, box.y, box.s, box.s);
}

// ---------------------------------------------------------------- layer detail dialog
function openDetail(i, spot) {
  const s = stages[i], st = STAGES[i];
  detail = { open: true, stage: i, sel: [], spot: spot ?? -1, els: [] };
  $('#ld-eyebrow').textContent = `Layer ${s.layer} of ${model.totalLayers}`;
  $('#ld-title').textContent = st.title;
  $('#ld-meta').innerHTML = `${st.blurb} This layer has <b>${s.C.toLocaleString()}</b> detectors, each producing a <b>${s.W}×${s.H}</b> grid. ` +
    `Here are the ${Math.min(TILES_IN_DETAIL, s.C)} ${settings.mode === 'important' ? 'that mattered most for this decision' : 'firing most strongly'}.`;
  const grid = $('#ld-grid');
  const n = Math.min(TILES_IN_DETAIL, s.C);
  grid.innerHTML = Array.from({ length: n }, () => '<div class="tile"><canvas></canvas><i class="vote"></i><span class="cid"></span></div>').join('');
  detail.els = [...grid.children].map(t => ({ el: t, canvas: $('canvas', t), vote: $('.vote', t), cid: $('.cid', t) }));
  detail.els.forEach((t, k) => t.el.addEventListener('click', () => { detail.spot = detail.sel[k]; }));
  $('#ld-spot-text').textContent = 'Click any detector to see exactly where in the picture it\'s firing.';
  $('#dlg-layer').showModal();
}
$('#dlg-layer').addEventListener('close', () => { detail.open = false; });

function renderDetail(m, ids) {
  const s = stages[detail.stage];
  const plane = s.H * s.W;
  const off = c => ids.indexOf(c) * plane;
  let stageMax = 0;
  const maxOf = c => { let mx = 0; const o = off(c); for (let p = 0; p < plane; p++) if (m[o + p] > mx) mx = m[o + p]; return mx; };
  const mxs = detail.sel.map(c => { const v = maxOf(c); stageMax = Math.max(stageMax, v); return v; });
  if (detail.spot < 0 && detail.sel.length) detail.spot = detail.sel[0];
  detail.sel.forEach((c, k) => {
    const t = detail.els[k]; if (!t) return;
    R.drawMap(t.canvas, m, s.W, s.H, { offset: off(c), scale: 1 / Math.max(mxs[k], 0.3 * stageMax, 1e-6) });
    const v = s.hasVote ? s.emaVote[c] : 0;
    t.el.style.borderColor = voteColour(v, s.voteMaxAbs);
    t.vote.style.width = (s.voteMaxAbs ? 100 * Math.abs(v) / s.voteMaxAbs : 0).toFixed(1) + '%';
    t.vote.style.color = v >= 0 ? 'var(--cat)' : 'var(--gator)';
    t.cid.textContent = '#' + c;
    t.el.classList.toggle('sel', c === detail.spot);
  });
  // spotlight
  const c = detail.spot;
  if (c >= 0 && ids.includes(c)) {
    const mx = maxOf(c);
    const v = s.hasVote ? s.emaVote[c] : 0;
    const rel = s.voteMaxAbs ? Math.abs(v) / s.voteMaxAbs : 0;
    const rgb = rel < 0.12 ? [190, 180, 255] : v > 0 ? R.CAT_RGB : R.GATOR_RGB;
    R.drawActivationOverlay($('#ld-spot'), inputCanvas, m, s.W, s.H, { offset: off(c), scale: 1 / Math.max(mx, 1e-6), rgb });
    R.drawMap($('#ld-spot-raw'), m, s.W, s.H, { offset: off(c), scale: 1 / Math.max(mx, 1e-6) });
    const firing = mx < 0.05 * (s.mapMax || 1) ? 'is hardly firing at all' : mx > 0.6 * (s.mapMax || 1) ? 'is firing strongly on the highlighted areas' : 'is firing on the highlighted areas';
    const vote = !s.hasVote || rel < 0.12 ? 'It isn\'t making much difference to the answer.'
      : `It is voting <b class="${v > 0 ? 'c-cat' : 'c-gator'}">${v > 0 ? 'CAT' : 'ALLIGATOR'}</b> (${rel > 0.6 ? 'strongly' : rel > 0.3 ? 'quite strongly' : 'a little'}).`;
    $('#ld-spot-text').innerHTML = `Detector <b>#${c}</b> ${firing}. ${vote}`;
  }
}

// ---------------------------------------------------------------- loop
async function loop() {
  if (running && !document.hidden && !document.querySelector('dialog[open]:not(#dlg-layer):not(#dlg-settings)')) {
    const now = performance.now();
    const minGap = (frozen || source === 'photo') ? 200 : 0;
    if (now - timing.lastEnd >= minGap) {
      try {
        const ok = await step();
        if (ok) {
          const end = performance.now();
          const dt = end - timing.lastEnd;
          timing.fps = timing.fps ? timing.fps * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
          timing.lastEnd = end;
          window.__frames = (window.__frames || 0) + 1;   // handy for automated tests
          if (settings.stats) updateStats();
        } else {
          drawOverlay();
        }
      } catch (err) {
        console.error(err);
        showError(err);
      }
    }
  }
  requestAnimationFrame(loop);
}

let errorShown = false;
function showError(err) {
  if (errorShown) return;
  errorShown = true;
  showCamMessage('Something went wrong: ' + (err?.message || err) + '. Try reloading the page.');
}

function updateStats() {
  const mem = tf.memory();
  $('#stats').textContent =
    `backend   ${tf.getBackend()}\n` +
    `frame/s   ${timing.fps.toFixed(1)}\n` +
    `explain   ${timing.analyse.toFixed(0)} ms\n` +
    `quick     ${timing.forward.toFixed(0)} ms\n` +
    `tensors   ${mem.numTensors}\n` +
    `gpu MB    ${((mem.numBytesInGPU || 0) / 1e6).toFixed(0)}`;
}

// ---------------------------------------------------------------- UI wiring
function updateFreezeUi() {
  const b = $('#btn-freeze');
  b.classList.toggle('on', frozen);
  $('.lbl', b).textContent = frozen ? 'Carry on' : 'Freeze';
  $('#frozen-badge').hidden = !frozen;
}
function toggleFreeze() {
  if (source === 'photo') return;
  frozen = !frozen;
  updateFreezeUi();
}
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}
function setMode(mode) {
  settings.mode = mode; saveSettings();
  document.querySelectorAll('#seg-mode button').forEach(b => b.setAttribute('aria-checked', String(b.dataset.mode === mode)));
  stages.forEach(s => { s.sel = []; });
  detail.sel = [];
}

function wireUi() {
  $('#btn-freeze').addEventListener('click', toggleFreeze);
  $('#btn-full').addEventListener('click', toggleFullscreen);
  $('#btn-info').addEventListener('click', () => $('#dlg-info').showModal());
  $('#btn-photo').addEventListener('click', () => $('#dlg-photo').showModal());
  $('#btn-settings').addEventListener('click', () => { listCameras(); $('#dlg-settings').showModal(); });
  $('#btn-back-camera').addEventListener('click', backToCamera);
  $('#btn-photo-camera').addEventListener('click', () => { $('#dlg-photo').close(); backToCamera(); });
  document.querySelectorAll('#seg-mode button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  const heat = $('#chk-heat');
  heat.checked = settings.heat;
  heat.addEventListener('change', () => { settings.heat = heat.checked; saveSettings(); });

  // settings dialog
  const mirror = $('#set-mirror'), zoom = $('#set-zoom'), none = $('#set-none'), grad = $('#set-grad'), stats = $('#set-stats'), cam = $('#set-camera');
  const syncSettingsUi = () => {
    mirror.checked = settings.mirror; zoom.value = settings.zoom; none.value = settings.none; grad.value = settings.grad; stats.checked = settings.stats;
    $('#out-zoom').textContent = Math.round(settings.zoom * 100) + '% of the camera picture';
    $('#out-none').textContent = Math.round(settings.none * 100) + '%';
    $('#stats').hidden = !settings.stats;
  };
  syncSettingsUi();
  mirror.addEventListener('change', () => { settings.mirror = mirror.checked; applyMirror(); saveSettings(); resetSmoothing(); });
  zoom.addEventListener('input', () => { settings.zoom = +zoom.value; saveSettings(); syncSettingsUi(); });
  none.addEventListener('input', () => { settings.none = +none.value; saveSettings(); syncSettingsUi(); });
  grad.addEventListener('change', () => { settings.grad = grad.value; saveSettings(); });
  stats.addEventListener('change', () => { settings.stats = stats.checked; saveSettings(); syncSettingsUi(); });
  cam.addEventListener('change', async () => { settings.camera = cam.value; saveSettings(); if (source === 'camera') await startCamera(); });
  $('#set-reset').addEventListener('click', async () => {
    Object.assign(settings, DEFAULTS); saveSettings(); syncSettingsUi(); applyMirror(); setMode(settings.mode); heat.checked = settings.heat;
    if (source === 'camera') await startCamera();
  });

  // photos: samples, file picker, drag & drop, paste
  $('#file-input').addEventListener('change', (e) => {
    const f = e.target.files?.[0]; if (f) loadPhoto(URL.createObjectURL(f)); e.target.value = '';
  });
  let dragDepth = 0;
  window.addEventListener('dragenter', (e) => { if (e.dataTransfer?.types?.includes('Files')) { dragDepth++; $('#drop-hint').hidden = false; } });
  window.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $('#drop-hint').hidden = true; });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault(); dragDepth = 0; $('#drop-hint').hidden = true;
    const f = [...(e.dataTransfer?.files || [])].find(f => f.type.startsWith('image/'));
    if (f) loadPhoto(URL.createObjectURL(f));
  });
  window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
    if (item) loadPhoto(URL.createObjectURL(item.getAsFile()));
  });
  loadSamples();

  // keyboard shortcuts (for presenters)
  window.addEventListener('keydown', (e) => {
    if (e.target.closest?.('input, select, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
    const anyDialog = document.querySelector('dialog[open]');
    const k = e.key.toLowerCase();
    if (k === ' ' && !anyDialog) { e.preventDefault(); toggleFreeze(); }
    else if (k === 'f') toggleFullscreen();
    else if (anyDialog) return;
    else if (k === 'h') { heat.checked = !heat.checked; heat.dispatchEvent(new Event('change')); }
    else if (k === 'm') { mirror.checked = !mirror.checked; mirror.dispatchEvent(new Event('change')); }
    else if (k === 'p') $('#dlg-photo').showModal();
    else if (k === 'i') $('#dlg-info').showModal();
    else if (k === 's') { listCameras(); $('#dlg-settings').showModal(); }
    else if (k === 'd') { stats.checked = !stats.checked; stats.dispatchEvent(new Event('change')); }
    else if (k >= '1' && k <= String(STAGES.length)) openDetail(+k - 1, -1);
  });
  window.addEventListener('resize', () => drawOverlay());
}

async function loadSamples() {
  try {
    const list = await (await fetch('samples/samples.json')).json();
    const box = $('#samples');
    box.innerHTML = list.map((s, i) => `<button data-i="${i}" title="${s.label}"><img src="samples/${s.file}" alt="${s.label}" loading="lazy"></button>`).join('');
    box.querySelectorAll('button').forEach(b => b.addEventListener('click', () => loadPhoto('samples/' + list[+b.dataset.i].file)));
  } catch { $('#samples').hidden = true; }
}

// ---------------------------------------------------------------- start-up
async function main() {
  const setLoading = (text, frac, detailText) => {
    if (text) $('#loading-text').textContent = text;
    if (frac != null) $('#loading-bar').style.width = (100 * frac).toFixed(0) + '%';
    if (detailText != null) $('#loading-detail').innerHTML = detailText;
  };
  if (location.protocol === 'file:') {
    setLoading('Please open this demo through a web server', 0,
      'Browsers block cameras and model files for pages opened straight from disk. Double-click <b>start-demo.bat</b> instead, or use the GitHub Pages link.');
    return;
  }
  if (!tf) { setLoading('TensorFlow.js failed to load', 0, 'Check that vendor/tf.min.js exists.'); return; }
  try {
    await tf.setBackend('webgl');
  } catch { /* fall through */ }
  await tf.ready();
  if (tf.getBackend() !== 'webgl') {
    setLoading(null, null, 'Warning: no GPU acceleration (WebGL) available, so this will be slow. Try Chrome or Edge with hardware acceleration switched on.');
  }
  try {
    model = await EfficientNetLite.load('model/', { onProgress: f => setLoading('Loading the AI…', f * 0.9) });
  } catch (err) {
    console.error(err);
    setLoading('Couldn\'t load the AI model', 0, 'Check the <code>model/</code> folder is present, then reload. (' + err.message + ')');
    return;
  }
  setLoading('Warming up…', 0.95);
  // First run compiles the GPU programs, so do it behind the loading screen.
  await new Promise(r => setTimeout(r, 30));
  tf.tidy(() => { const r = model.analyse(tf.zeros([1, INPUT, INPUT, 3]), TAPS); [r.logits, r.score, ...r.acts, ...r.grads].forEach(t => t.dispose()); });
  buildPipeline();
  wireUi();
  setLoading('Starting the camera…', 1);
  await startCamera();
  $('#loading').hidden = true;
  running = true;
  requestAnimationFrame(loop);
}

main();
