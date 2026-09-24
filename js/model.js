// EfficientNet-Lite0 written directly with TensorFlow.js ops, so that we can
// (a) grab the activations of any layer and (b) get gradients for any layer.
//
// Weights: ImageNet-trained EfficientNet-Lite0 (Apache-2.0, originally from
// Google's TensorFlow TPU repo, via the `efficientnet_lite0_pytorch_model`
// package). BatchNorm has been folded into the convolutions and the weights are
// stored as float16 to halve the download (see tools/convert_weights.py).

const tf = globalThis.tf;

// ImageNet class indices that we treat as "cat" and "alligator".
// 281-285 are pet cats (tabby, tiger cat, Persian, Siamese, Egyptian);
// 286-293 are wild cats (cougar, lynx, leopard, snow leopard, jaguar, lion, tiger, cheetah).
export const CAT_CLASSES = [281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293];
export const GATOR_CLASSES = [49, 50];                   // African crocodile, American alligator

// Normalisation used by the original TF EfficientNet-Lite models.
const MEAN = 127.0, STD = 128.0;

function float16ToFloat32(u16) {
  const out = new Float32Array(u16.length);
  // Lookup tables make this fast enough for ~4.6M values.
  const exp = new Float32Array(32);
  for (let e = 0; e < 32; e++) exp[e] = Math.pow(2, e - 15);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x03ff;
    if (e === 0) out[i] = s * Math.pow(2, -14) * (f / 1024);
    else if (e === 31) out[i] = f ? NaN : s * Infinity;
    else out[i] = s * exp[e] * (1 + f / 1024);
  }
  return out;
}

export class EfficientNetLite {
  constructor(manifest, weights) {
    this.arch = manifest.arch;
    this.w = weights;
    this.tapShapes = {};
    this.layerIndex = this._numberLayers();
  }

  static async load(baseUrl, { fetchFn = globalThis.fetch, onProgress } = {}) {
    const manifest = await (await fetchFn(baseUrl + 'model.json')).json();
    const res = await fetchFn(baseUrl + 'weights.bin');
    let buf;
    const total = Number(res.headers?.get?.('content-length')) || 0;
    if (onProgress && res.body && total) {
      const reader = res.body.getReader();
      const chunks = []; let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        onProgress(Math.min(1, got / total));
      }
      const all = new Uint8Array(got); let o = 0;
      for (const c of chunks) { all.set(c, o); o += c.length; }
      buf = all.buffer;
    } else {
      buf = await res.arrayBuffer();
    }
    const f32 = float16ToFloat32(new Uint16Array(buf));
    const weights = {};
    for (const t of manifest.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      weights[t.name] = tf.tensor(f32.subarray(t.offset, t.offset + n), t.shape, 'float32');
    }
    const model = new EfficientNetLite(manifest, weights);
    model._warmup();
    return model;
  }

  // Give every conv layer a number (1-based), the way you'd count layers in a diagram.
  _numberLayers() {
    const idx = {}; let n = 0;
    idx['stem'] = ++n;
    for (const b of this.arch.blocks) {
      if (b.e !== 1) idx[`b${b.i}.expand`] = ++n;
      idx[`b${b.i}.dw`] = ++n;
      idx[`b${b.i}.project`] = ++n;
    }
    idx['head'] = ++n;
    idx['fc'] = ++n;
    this.totalLayers = n;
    return idx;
  }

  // Receptive field (in input pixels) of one cell of a tap's feature map.
  receptiveField(tap) {
    let rf = 1, jump = 1;
    const step = (k, s) => { rf += (k - 1) * jump; jump *= s; };
    step(3, 2); if (tap === 'stem') return rf;
    for (const b of this.arch.blocks) {
      if (tap === `b${b.i}.expand`) return rf;
      step(b.k, b.s);
      if (tap === `b${b.i}.dw`) return rf;
    }
    return rf;
  }

  // Turn an image (canvas / video / ImageData, already 224x224) into a model input.
  preprocess(pixels) {
    return tf.tidy(() => tf.browser.fromPixels(pixels).toFloat().sub(MEAN).div(STD).expandDims(0));
  }

  // Fused ops are faster for a plain forward pass, but their gradient also
  // computes (unneeded) weight gradients, so the gradient pass uses plain ops.
  _conv(x, name, strides, act, fused) {
    const w = this.w[name + '.w'], b = this.w[name + '.b'];
    if (fused) return tf.fused.conv2d({ x, filter: w, bias: b, strides, pad: 'same', activation: act ? 'relu6' : 'linear' });
    const y = tf.conv2d(x, w, strides, 'same').add(b);
    return act ? tf.relu6(y) : y;
  }

  _dw(x, name, strides, fused) {
    const w = this.w[name + '.w'], b = this.w[name + '.b'];
    if (fused) return tf.fused.depthwiseConv2d({ x, filter: w, bias: b, strides, pad: 'same', activation: 'relu6' });
    return tf.relu6(tf.depthwiseConv2d(x, w, strides, 'same').add(b));
  }

  // Forward pass. `eps` (optional) maps tap name -> tensor added at that tap
  // (used to get gradients w.r.t. the tap). `store` (optional) receives the
  // tap activations (kept alive with tf.keep, caller must dispose).
  forward(x, { eps = {}, store = null, keep = false, fused = true } = {}) {
    const tap = (name, h) => {
      if (eps[name]) h = h.add(eps[name]);
      if (store && name in store) store[name] = keep ? tf.keep(h) : h;
      if (!(name in this.tapShapes)) this.tapShapes[name] = h.shape;
      return h;
    };
    let h = tap('stem', this._conv(x, 'stem', 2, true, fused));
    for (const b of this.arch.blocks) {
      const inp = h;
      if (b.e !== 1) h = tap(`b${b.i}.expand`, this._conv(h, `b${b.i}.expand`, 1, true, fused));
      h = tap(`b${b.i}.dw`, this._dw(h, `b${b.i}.dw`, b.s, fused));
      h = this._conv(h, `b${b.i}.project`, 1, false, fused);
      if (b.skip) h = h.add(inp);
    }
    h = tap('head', this._conv(h, 'head', 1, true, fused));
    const pooled = h.mean([1, 2]);                              // global average pool -> [1,1280]
    return pooled.matMul(this.w['fc.w']).add(this.w['fc.b']);    // logits [1,1000]
  }

  // The number the model uses to decide "cat or alligator":
  // log P(cat) - log P(alligator), where each is summed over its ImageNet classes.
  // (The softmax denominator cancels, so this only needs the logits.)
  score(logits) {
    const l = logits.squeeze([0]);
    return tf.sub(tf.logSumExp(tf.gather(l, CAT_CLASSES)), tf.logSumExp(tf.gather(l, GATOR_CLASSES)));
  }

  _warmup() {
    tf.tidy(() => {
      const x = tf.zeros([1, 224, 224, 3]);
      const store = {};
      // record every tap's shape
      this.forward(x, { store });
    });
  }

  // Full analysis for one image: activations + gradients at the requested taps.
  // `extra` lists more layers whose activations are wanted (e.g. for the sketch
  // classifier), and `scoreFn(logits, store)` can replace the cat-vs-alligator score
  // that the gradients are taken of. Returns tensors which the caller must dispose:
  // {logits, score, acts[], grads[], extra{}} (see disposeResult).
  analyse(x, taps, { extra = [], scoreFn = null } = {}) {
    const names = [...new Set([...taps, ...extra])];
    const store = Object.fromEntries(names.map(t => [t, null]));
    let logits = null;
    const zeros = taps.map(t => tf.zeros(this.tapShapes[t]));
    const f = (...e) => {
      const eps = Object.fromEntries(taps.map((t, i) => [t, e[i]]));
      const lg = this.forward(x, { eps, store, keep: true, fused: false });
      logits = tf.keep(lg);
      return scoreFn ? scoreFn(lg, store) : this.score(lg);
    };
    const { value, grads } = tf.valueAndGrads(f)(zeros);
    zeros.forEach(z => z.dispose());
    return { logits, score: value, acts: taps.map(t => store[t]), grads, extra: Object.fromEntries(extra.filter(t => !taps.includes(t)).map(t => [t, store[t]])), store };
  }

  // Forward pass only (cheaper) — used when gradients aren't needed.
  activations(x, taps, extra = []) {
    const names = [...new Set([...taps, ...extra])];
    const store = Object.fromEntries(names.map(t => [t, null]));
    const logits = tf.tidy(() => tf.keep(this.forward(x, { store, keep: true })));
    return { logits, acts: taps.map(t => store[t]), extra: Object.fromEntries(extra.filter(t => !taps.includes(t)).map(t => [t, store[t]])), store };
  }
}

// Free every tensor returned by analyse() / activations().
export function disposeResult(r) {
  if (!r) return;
  [r.logits, r.score, ...(r.acts || []), ...(r.grads || []), ...Object.values(r.extra || {})].forEach(t => t && !t.isDisposed && t.dispose());
}
