"""Re-train the sketch-mode classifier (model/sketch-head.json).

You don't need this to run the demo. It's here so the training is reproducible.

  pip install numpy pillow scipy scikit-learn onnx onnxruntime
  python tools/sketch/build_onnx.py                 # makes an ONNX copy of the app's network, for fast feature extraction
  python tools/sketch/train_sketch_head.py DATA_DIR  # writes model/sketch-head.json

DATA_DIR must contain (see README.md in this folder for how they were collected):
  quickdraw.json          {"cat": [drawing, ...], "crocodile": [...], "<other thing>": [...]}  (Quick, Draw! simplified strokes)
  isketch/meta.json       [{"file": "isketch/cat_281_14301.jpg", "k": "cat"|"gator"|"other", ...}]  (ImageNet-Sketch images)
  drawings/meta.json      optional extra drawings, same format as isketch (e.g. public-domain drawings from Wikimedia Commons)

Every image is shown to the network the way the app sees it: turned into a pretend webcam photo
(paper, lighting, blur, background, hands; see sketchlib.webcam_augment) and then cleaned up with
exactly the same steps the app uses (sketchlib.normalise).
"""
import json, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sketchlib import render_quickdraw, letterbox, webcam_augment
from features import extract

FEATS = ['b15.expand_gap', 'head_gap']   # average activations of these layers are the classifier's input
C = 0.02                                 # regularisation (smaller = simpler model)
W_REAL = 4.0                             # real pencil sketches count 4x more than quick doodles
OTHER_BIAS = -2.0                        # makes "neither" a little less trigger-happy
TEMPERATURE = 7.0                        # score = 100 * sigmoid(margin / T); median real sketch scores about 70
CL = ['cat', 'gator', 'other']
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main(data):
    random.seed(0); np.random.seed(0)
    items = []                           # (label, group, source, what)
    qd = json.load(open(os.path.join(data, 'quickdraw.json')))
    for k, v in qd.items():
        lab = 'cat' if k == 'cat' else 'gator' if k == 'crocodile' else 'other'
        items += [(lab, f'qd:{k}:{i}', 'quickdraw', d) for i, d in enumerate(v)]
    for sub, src in [('isketch', 'real'), ('drawings', 'real')]:
        mp = os.path.join(data, sub, 'meta.json')
        if os.path.exists(mp):
            items += [(m['k'], f'{sub}:{m["file"]}', src, os.path.join(data, m['file'])) for m in json.load(open(mp))]
    items += [('other', f'neg:{i}', 'synthetic', i) for i in range(240)]
    imgs, meta = [], []
    for lab, gid, src, obj in items:
        sk = render_quickdraw(obj) if src == 'quickdraw' else synthetic_page(obj) if src == 'synthetic' else letterbox(Image.open(obj))
        for v in ['clean', 'cam', 'cam2']:
            imgs.append(sk.convert('RGB') if v == 'clean' else webcam_augment(sk)); meta.append((lab, gid, src, v))
    print('images', len(imgs), flush=True)
    F = extract(imgs)
    X = np.concatenate([F[f] for f in FEATS], 1)
    y = np.array([CL.index(m[0]) for m in meta]); grp = np.array([m[1] for m in meta]); src = np.array([m[2] for m in meta]); ver = np.array([m[3] for m in meta])
    w = np.where(src == 'real', W_REAL, 1.0)
    def fit(idx):
        sc = StandardScaler().fit(X[idx])
        return sc, LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(sc.transform(X[idx]), y[idx], sample_weight=w[idx])
    def margin(L, k):
        o = np.delete(L, k, axis=1); mo = o.max(1, keepdims=True); return L[:, k] - (mo[:, 0] + np.log(np.exp(o - mo).sum(1)))
    # out-of-fold margins of the real sketches -> the "better than X% of sketches" reference
    real = np.where(src == 'real')[0]; oof = np.zeros((len(y), 3))
    for a, b in GroupKFold(n_splits=5).split(real, groups=grp[real]):
        te = real[b]; sc, clf = fit(np.setdiff1d(np.arange(len(y)), te)); oof[te] = clf.decision_function(sc.transform(X[te]))
    oof[:, 2] += OTHER_BIAS
    ref = {}
    for k, nm in [(0, 'cat'), (1, 'alligator')]:
        sel = (src == 'real') & (y == k) & (ver == 'cam')
        mg = margin(oof[sel], k); ref[nm] = [round(float(v), 2) for v in np.sort(mg)]
        print(f'{nm}: held-out accuracy {np.mean(oof[sel].argmax(1) == k):.2f}, cat-vs-alligator {np.mean((oof[sel, k] > oof[sel, 1 - k])):.2f}, median score {100 / (1 + np.exp(-np.median(mg) / TEMPERATURE)):.0f}', flush=True)
    # final model on everything, with the standardisation folded into the weights
    sc, clf = fit(np.arange(len(y)))
    W = clf.coef_.T / sc.scale_[:, None]; b = clf.intercept_ - (sc.mean_ / sc.scale_) @ clf.coef_.T; b[2] += OTHER_BIAS
    spec = {'version': 1, 'about': 'Sketch-mode classifier: logistic regression on the average activations of EfficientNet-Lite0 layers b15.expand and head. '
                                    'Trained by tools/sketch/train_sketch_head.py on ImageNet-Sketch, Quick, Draw! doodles, public-domain drawings and blank/scribble pages.',
            'classes': ['cat', 'alligator', 'other'], 'features': [f.replace('_gap', '') for f in FEATS], 'dim': int(W.shape[0]),
            'W': [round(float(v), 6) for v in W.ravel()], 'b': [round(float(v), 6) for v in b], 'temperature': TEMPERATURE, 'reference': ref}
    out = os.path.join(ROOT, 'model', 'sketch-head.json'); json.dump(spec, open(out, 'w'), separators=(',', ':')); print('wrote', out)

def synthetic_page(i):
    """Blank pages, scribbles, writing and simple shapes: things that should score 'neither'."""
    rnd = random.Random(i); im = Image.new('L', (224, 224), 255); d = ImageDraw.Draw(im); kind = i % 4
    if kind == 1:
        for _ in range(rnd.randint(1, 5)):
            d.line([(rnd.uniform(20, 204), rnd.uniform(20, 204)) for _ in range(rnd.randint(3, 12))], fill=0, width=rnd.randint(1, 4), joint='curve')
    elif kind == 2:
        try: font = ImageFont.load_default(size=rnd.randint(14, 30))
        except TypeError: font = ImageFont.load_default()
        for r in range(rnd.randint(1, 4)):
            d.text((rnd.uniform(10, 60), 30 + r * rnd.uniform(30, 45)), rnd.choice(['hello', 'my drawing', 'name:', 'Year 7', '2026']), fill=0, font=font)
    elif kind == 3:
        for _ in range(rnd.randint(1, 4)):
            x, yy, r = rnd.uniform(40, 184), rnd.uniform(40, 184), rnd.uniform(10, 60)
            (d.ellipse if rnd.random() < 0.5 else d.rectangle)([x - r, yy - r, x + r, yy + r], outline=0, width=rnd.randint(1, 4))
    return im

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data')
