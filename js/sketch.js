// Sketch mode helpers: pencil-sketch clean-up, capture quality checks, and scoring.
// normaliseSketch() mirrors tools/sketch/sketchlib.py normalise() exactly, because the
// sketch classifier was trained on images cleaned up that way.

const S = 224, B = 7, N = S / B;   // 32 x 32 blocks of 7 x 7 pixels

function upsampleBilinear(small, n, size) {
  const pos = new Float32Array(size), i0 = new Int32Array(size), i1 = new Int32Array(size), f = new Float32Array(size);
  for (let k = 0; k < size; k++) {
    let p = (k + 0.5) * n / size - 0.5;
    p = Math.min(Math.max(p, 0), n - 1);
    i0[k] = Math.floor(p); i1[k] = Math.min(i0[k] + 1, n - 1); f[k] = p - i0[k]; pos[k] = p;
  }
  const out = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    const ra = i0[y] * n, rb = i1[y] * n, fy = f[y];
    for (let x = 0; x < size; x++) {
      const a = small[ra + i0[x]] * (1 - f[x]) + small[ra + i1[x]] * f[x];
      const b = small[rb + i0[x]] * (1 - f[x]) + small[rb + i1[x]] * f[x];
      out[y * size + x] = a * (1 - fy) + b * fy;
    }
  }
  return out;
}

function greyOf(ctx) {
  const d = ctx.getImageData(0, 0, S, S).data;
  const g = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++) g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  return g;
}

function pctIndex(hist, total, q) { let c = 0; for (let k = 0; k < 256; k++) { c += hist[k]; if (c / total >= q) return k; } return 255; }

// 4-connected flood fill from the image border through pixels where mask[i] is 1.
function floodFromBorder(mask) {
  const out = new Uint8Array(S * S), stack = new Int32Array(S * S); let sp = 0;
  const push = (i) => { if (mask[i] && !out[i]) { out[i] = 1; stack[sp++] = i; } };
  for (let x = 0; x < S; x++) { push(x); push((S - 1) * S + x); }
  for (let y = 0; y < S; y++) { push(y * S); push(y * S + S - 1); }
  while (sp) {
    const i = stack[--sp], y = (i / S) | 0, x = i - y * S;
    if (y > 0) push(i - S); if (y < S - 1) push(i + S); if (x > 0) push(i - 1); if (x < S - 1) push(i + 1);
  }
  return out;
}

// Largest 4-connected region of mask (1s).
function largestComponent(mask) {
  const lab = new Int32Array(S * S), stack = new Int32Array(S * S);
  let cur = 0, best = 0, bestN = 0;
  for (let s0 = 0; s0 < S * S; s0++) {
    if (!mask[s0] || lab[s0]) continue;
    cur++; let n = 0, sp = 0; lab[s0] = cur; stack[sp++] = s0;
    while (sp) {
      const i = stack[--sp]; n++;
      const y = (i / S) | 0, x = i - y * S;
      if (y > 0 && mask[i - S] && !lab[i - S]) { lab[i - S] = cur; stack[sp++] = i - S; }
      if (y < S - 1 && mask[i + S] && !lab[i + S]) { lab[i + S] = cur; stack[sp++] = i + S; }
      if (x > 0 && mask[i - 1] && !lab[i - 1]) { lab[i - 1] = cur; stack[sp++] = i - 1; }
      if (x < S - 1 && mask[i + 1] && !lab[i + 1]) { lab[i + 1] = cur; stack[sp++] = i + 1; }
    }
    if (n > bestN) { best = cur; bestN = n; }
  }
  const out = new Uint8Array(S * S);
  if (!best) return mask.slice();
  for (let i = 0; i < S * S; i++) out[i] = lab[i] === best ? 1 : 0;
  return out;
}

function dilate(mask, r) {
  let cur = mask;
  for (let k = 0; k < r; k++) {
    const m = cur.slice();
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const i = y * S + x;
      if (cur[i]) continue;
      if ((y > 0 && cur[i - S]) || (y < S - 1 && cur[i + S]) || (x > 0 && cur[i - 1]) || (x < S - 1 && cur[i + 1])) m[i] = 1;
    }
    cur = m;
  }
  return cur;
}

function percentile(sorted, q) {            // numpy's default (linear) percentile
  const pos = q / 100 * (sorted.length - 1), lo = Math.floor(pos), hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// Clean up a 224x224 canvas in place (same steps as tools/sketch/sketchlib.py normalise()):
//  1. grey-scale;  2. blank out everything that isn't the sheet of paper (background, hands);
//  3. divide by a smooth estimate of the paper brightness (removes shadows);
//  4. stretch so the paper is white and the pencil is black;  5. zoom in on the drawing.
// Returns simple stats (used for the capture checks) and the zoom rectangle.
export function normaliseSketch(canvas, { crop = true } = {}) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const g = greyOf(ctx);
  // 2. background removal: keep the biggest bright region (the paper) and whatever it encloses
  const h0 = new Uint32Array(256);
  for (let i = 0; i < S * S; i++) h0[Math.min(255, Math.max(0, Math.floor(g[i])))]++;
  const P = pctIndex(h0, S * S, 0.90);
  const bright = new Uint8Array(S * S);
  for (let i = 0; i < S * S; i++) bright[i] = g[i] >= 0.8 * P ? 1 : 0;
  const paper = largestComponent(bright);
  const notPaper = new Uint8Array(S * S);
  for (let i = 0; i < S * S; i++) notPaper[i] = paper[i] ? 0 : 1;
  const outside = dilate(floodFromBorder(notPaper), 2);
  let bgFrac = 0;
  for (let i = 0; i < S * S; i++) if (outside[i]) { g[i] = P; bgFrac++; }
  // 3. paper brightness estimate: 7x7 block means, then a 3x3 max filter, then smooth upsample
  const small = new Float32Array(N * N);
  for (let by = 0; by < N; by++) for (let bx = 0; bx < N; bx++) {
    let s = 0;
    for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) s += g[(by * B + y) * S + bx * B + x];
    small[by * N + bx] = s / (B * B);
  }
  const mx = new Float32Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let m = -Infinity;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const yy = Math.min(N - 1, Math.max(0, y + dy)), xx = Math.min(N - 1, Math.max(0, x + dx));
      m = Math.max(m, small[yy * N + xx]);
    }
    mx[y * N + x] = m;
  }
  const bg = upsampleBilinear(mx, N, S);
  // 4. ratio to paper, then a percentile stretch
  const r = new Float32Array(S * S), hist = new Uint32Array(256);
  for (let i = 0; i < S * S; i++) {
    r[i] = g[i] / Math.max(bg[i], 1);
    hist[Math.min(255, Math.max(0, Math.floor(r[i] * 200)))]++;
  }
  let lo = pctIndex(hist, S * S, 0.01) / 200; const hi = pctIndex(hist, S * S, 0.90) / 200;
  const contrast = hi - lo;
  if (hi - lo < 0.12) lo = hi - 0.12;
  const img = ctx.createImageData(S, S);
  let ink = 0, meanBrightness = 0;
  const xs = [], ys = [];
  for (let i = 0; i < S * S; i++) {
    const v = Math.min(1, Math.max(0, (r[i] - lo) / (hi - lo)));
    const u = Math.floor(255 * v);
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = u; img.data[i * 4 + 3] = 255;
    if (v < 0.5) { ink++; xs.push(i % S); ys.push((i / S) | 0); }
    meanBrightness += g[i];
  }
  ctx.putImageData(img, 0, 0);
  const stats = { contrast, ink: ink / (S * S), brightness: meanBrightness / (S * S), background: bgFrac / (S * S), rect: { x: 0, y: 0, s: S } };
  // 5. zoom in on the drawing: a square around the pencil marks plus a margin
  if (crop && xs.length > 40) {
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const x0 = percentile(xs, 0.5), x1 = percentile(xs, 99.5), y0 = percentile(ys, 0.5), y1 = percentile(ys, 99.5);
    let side = Math.max(x1 - x0, y1 - y0) * 1.2 + 8;
    side = Math.max(side, 56);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const L = Math.round(cx - side / 2), T = Math.round(cy - side / 2); side = Math.round(side);
    const tmp = document.createElement('canvas'); tmp.width = tmp.height = S;
    tmp.getContext('2d').drawImage(canvas, 0, 0);
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, S, S);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'medium';
    ctx.drawImage(tmp, L, T, side, side, 0, 0, S, S);
    stats.rect = { x: L, y: T, s: side };
  }
  return stats;
}

// Sharpness of a raw (not yet cleaned) 224x224 frame: mean squared gradient.
export function sharpness(canvas) {
  const g = greyOf(canvas.getContext('2d', { willReadFrequently: true }));
  let s = 0;
  for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) {
    const gx = g[y * S + x + 1] - g[y * S + x - 1], gy = g[(y + 1) * S + x] - g[(y - 1) * S + x];
    s += gx * gx + gy * gy;
  }
  return s / ((S - 2) * (S - 2));
}

// Is there a drawing on paper in the box at all? (Blank scenes, people or rooms fail this.)
export function looksLikeADrawing(stats) {
  return stats.contrast >= 0.1 && stats.ink >= 0.005 && stats.background <= 0.78;
}

// Friendly warnings about a captured frame (empty list = looks fine).
export function captureWarnings(stats, sharp) {
  const w = [];
  if (stats.brightness < 45) w.push('It\'s quite dark. Try facing a window or a light.');
  if (stats.background > 0.78) w.push('I can\'t find a sheet of paper in the box. Hold your drawing closer, so it fills most of the box.');
  else if (stats.contrast < 0.1 || stats.ink < 0.005) w.push('I can hardly see any drawing. Hold it closer, or press harder with the pencil.');
  else if (stats.ink > 0.45) w.push('There\'s a lot of dark stuff in the box. Make sure it\'s just your drawing on white paper.');
  if (sharp < 25 && stats.ink >= 0.005) w.push('It looks a bit blurry. Keep the paper still.');
  return w;
}

// ---- the sketch classifier (a small extra layer trained on sketches) ----
export class SketchHead {
  constructor(spec) {
    this.spec = spec;
    this.classes = spec.classes;                 // ['cat', 'alligator', 'other']
    this.features = spec.features;               // layers whose average activations are used
    const tf = globalThis.tf;
    this.W = tf.tensor2d(spec.W, [spec.dim, this.classes.length]);
    this.b = tf.tensor1d(spec.b);
  }
  static async load(url) { return new SketchHead(await (await fetch(url)).json()); }

  // store: tap name -> activation tensor [1,H,W,C]. Returns logits tensor [3].
  logits(store) {
    const tf = globalThis.tf;
    const f = tf.concat(this.features.map(t => store[t].mean([1, 2])), 1);   // [1, dim]
    return f.matMul(this.W).add(this.b).squeeze([0]);
  }

  // The leaderboard score for the animal the student says they drew.
  // margin = how much more the AI believes that animal than the alternatives (log-odds);
  // dividing by a "temperature" before squashing to 0-100 stops good drawings all
  // bunching up at 99-100 (the top end is stretched out, like a log scale).
  score(logitsArr, animal) {
    const k = this.classes.indexOf(animal);
    const others = logitsArr.filter((_, i) => i !== k);
    const mo = Math.max(...others);
    const lse = mo + Math.log(others.reduce((s, v) => s + Math.exp(v - mo), 0));
    const margin = logitsArr[k] - lse;
    const score = 100 / (1 + Math.exp(-margin / this.spec.temperature));
    const ref = this.spec.reference?.[animal] || [];
    let below = 0; for (const m of ref) if (m < margin) below++;
    return { margin, score, percentile: ref.length ? 100 * below / ref.length : null };
  }

  probs(logitsArr) {
    const m = Math.max(...logitsArr);
    const e = logitsArr.map(v => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }
}
