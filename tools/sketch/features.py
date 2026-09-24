import os, numpy as np, onnxruntime as ort
from sketchlib import normalise, to_model_input
_sess = None
OUTS = ['stem_gap', 'b3.expand_gap', 'b5.expand_gap', 'b8.expand_gap', 'b11.expand_gap', 'b15.expand_gap', 'head_gap', 'logits']
def sess():
    global _sess
    if _sess is None:
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        _sess = ort.InferenceSession(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lite0.onnx'), so, providers=['CPUExecutionProvider'])
    return _sess
def extract(images_rgb, batch=32):
    """images: list of PIL RGB images (any size). Applies the app's sketch normalisation. Returns dict of arrays."""
    res = {k: [] for k in OUTS}
    for i in range(0, len(images_rgb), batch):
        x = np.stack([to_model_input(normalise(im)) for im in images_rgb[i:i + batch]]).astype(np.float32)
        o = sess().run(OUTS, {'input': x})
        for k, v in zip(OUTS, o): res[k].append(v)
    return {k: np.concatenate(v) for k, v in res.items()}
