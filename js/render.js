// Drawing helpers: colour maps, feature-map tiles and evidence overlays.

// Colours shared with the CSS (keep in sync with --cat / --gator).
export const CAT_RGB = [255, 159, 28];
export const GATOR_RGB = [45, 212, 191];

// "Ice" colour map for detector activations: black -> indigo -> lavender -> white.
// Kept free of orange/teal so those colours can mean "cat" and "alligator".
function buildLut(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k], [t1, c1] = stops[k + 1];
    const u = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * u;
  }
  return lut;
}
export const ICE = buildLut([
  [0.00, [6, 8, 18]],
  [0.25, [40, 30, 110]],
  [0.55, [108, 92, 231]],
  [0.80, [190, 180, 255]],
  [1.00, [255, 255, 255]],
]);

// Draw a single-channel map (values already scaled so 1 = brightest) onto a
// canvas whose size equals the map's size. CSS does the (pixelated) scaling.
export function drawMap(canvas, data, w, h, { offset = 0, stride = 1, scale = 1 } = {}) {
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  const px = img.data;
  for (let i = 0, n = w * h; i < n; i++) {
    let v = data[offset + i * stride] * scale;
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    const li = Math.round(Math.sqrt(v) * 255) * 3;   // sqrt = gentle gamma so faint activity is visible
    px[i * 4] = ICE[li]; px[i * 4 + 1] = ICE[li + 1]; px[i * 4 + 2] = ICE[li + 2]; px[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// Signed evidence map -> RGBA (orange = cat, teal = alligator), on a w x h canvas.
const scratch = typeof document !== 'undefined' ? document.createElement('canvas') : null;
function evidenceToCanvas(data, w, h, scale, maxAlpha = 0.85) {
  scratch.width = w; scratch.height = h;
  const ctx = scratch.getContext('2d');
  const img = ctx.createImageData(w, h);
  const px = img.data;
  for (let i = 0; i < w * h; i++) {
    const v = data[i] * scale;
    const c = v >= 0 ? CAT_RGB : GATOR_RGB;
    const a = Math.min(1, Math.abs(v));
    px[i * 4] = c[0]; px[i * 4 + 1] = c[1]; px[i * 4 + 2] = c[2];
    px[i * 4 + 3] = Math.round(255 * maxAlpha * Math.pow(a, 0.8));
  }
  ctx.putImageData(img, 0, 0);
  return scratch;
}

// Grey-scale copy of the model's input with the evidence map on top.
export function drawEvidence(canvas, inputCanvas, data, w, h, scale, { smooth = false, size = 224 } = {}) {
  if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.filter = 'grayscale(1) brightness(0.55)';
  ctx.drawImage(inputCanvas, 0, 0, size, size);
  ctx.restore();
  if (!data) return;
  ctx.imageSmoothingEnabled = smooth;
  ctx.drawImage(evidenceToCanvas(data, w, h, scale), 0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
}

// One detector's activation drawn over the grey-scale input (for the spotlight view).
export function drawActivationOverlay(canvas, inputCanvas, data, w, h, { offset = 0, stride = 1, scale = 1, rgb = [255, 255, 255], size = 448 } = {}) {
  if (canvas.width !== size) { canvas.width = size; canvas.height = size; }
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.filter = 'grayscale(1) brightness(0.45)';
  ctx.drawImage(inputCanvas, 0, 0, size, size);
  ctx.restore();
  scratch.width = w; scratch.height = h;
  const sctx = scratch.getContext('2d');
  const img = sctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.min(1, Math.max(0, data[offset + i * stride] * scale));
    img.data[i * 4] = rgb[0]; img.data[i * 4 + 1] = rgb[1]; img.data[i * 4 + 2] = rgb[2];
    img.data[i * 4 + 3] = Math.round(255 * 0.9 * Math.sqrt(v));
  }
  sctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch, 0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
}

// "Spotlight" heat-map on top of the live camera, drawn inside the crop box:
// areas with little evidence are dimmed, areas with strong evidence stay
// bright with a tint (orange = cat, teal = alligator). Dimming rather than just
// tinting keeps it readable on orange cats and green alligators.
export function drawCameraHeat(ctx, data, w, h, scale, box) {
  scratch.width = w; scratch.height = h;
  const sctx = scratch.getContext('2d');
  const img = sctx.createImageData(w, h);
  const px = img.data;
  for (let i = 0; i < w * h; i++) {
    const v = data[i] * scale;
    const a = Math.min(1, Math.abs(v));
    const c = v >= 0 ? CAT_RGB : GATOR_RGB;
    px[i * 4] = c[0] * a; px[i * 4 + 1] = c[1] * a; px[i * 4 + 2] = c[2] * a;
    px[i * 4 + 3] = Math.round(255 * (0.7 * (1 - a) + 0.3 * a));
  }
  sctx.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(scratch, box.x, box.y, box.s, box.s);
  ctx.restore();
}
