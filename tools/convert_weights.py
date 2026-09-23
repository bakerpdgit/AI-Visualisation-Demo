"""Re-create model/model.json + model/weights.bin from the original weights.

You don't need to run this: the converted files are already in model/.
It's here so the conversion is reproducible and inspectable.

  pip install numpy efficientnet_lite0_pytorch_model
  python tools/convert_weights.py            # writes model/model.json and model/weights.bin

What it does:
  1. Reads the EfficientNet-Lite0 ImageNet weights (a PyTorch .pth file from the
     `efficientnet_lite0_pytorch_model` package, Apache-2.0) without needing PyTorch.
  2. Folds each BatchNorm layer into the convolution before it (same maths, fewer steps).
  3. Re-orders the weights into TensorFlow's layout and saves them as float16
     (halves the download; accuracy is unaffected).
"""
import argparse, json, os, pickle, struct, collections
import numpy as np

# ---------------------------------------------------------------- .pth reader

DTYPES = {'FloatStorage': np.float32, 'LongStorage': np.int64, 'IntStorage': np.int32,
          'HalfStorage': np.float16, 'DoubleStorage': np.float64, 'ByteStorage': np.uint8}

class StorageType:
    def __init__(self, name): self.name = name

class Storage:
    def __init__(self, dtype, numel): self.dtype, self.numel, self.data = dtype, numel, None

class LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)
    def numpy(self):
        d = self.storage.data
        itemsize = d.itemsize
        return np.lib.stride_tricks.as_strided(d[self.offset:], shape=self.size,
                                               strides=[s * itemsize for s in self.stride]).copy()

def rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, *a):
    return LazyTensor(storage, offset, size, stride)

def load(path):
    f = open(path, 'rb')
    magic = pickle.load(f); proto = pickle.load(f); sysinfo = pickle.load(f)
    storages = {}
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == 'torch._utils' and name == '_rebuild_tensor_v2': return rebuild_tensor_v2
            if mod == 'torch' and name.endswith('Storage'): return StorageType(name)
            if mod == 'collections' and name == 'OrderedDict': return collections.OrderedDict
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            typename, stype, key, location, numel, view = pid
            assert typename == 'storage' and view is None
            if key not in storages: storages[key] = Storage(DTYPES[stype.name], numel)
            return storages[key]
    sd = U(f).load()
    keys = pickle.load(f)
    for k in keys:
        s = storages[k]
        n = struct.unpack('<q', f.read(8))[0]
        assert n == s.numel, (n, s.numel)
        s.data = np.frombuffer(f.read(n * np.dtype(s.dtype).itemsize), dtype=s.dtype)
    return collections.OrderedDict((k, v.numpy()) for k, v in sd.items())


def find_pth():
    try:
        from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
        return EfficientnetLite0ModelFile.get_model_file_path()
    except ImportError:
        raise SystemExit('Install the weights first:  pip install efficientnet_lite0_pytorch_model  (or pass --pth)')

ap = argparse.ArgumentParser()
ap.add_argument('--pth', help='path to efficientnet-lite0-57934424.pth')
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))
args = ap.parse_args()
EPS = 1e-3  # BatchNorm epsilon used by EfficientNet
sd = load(args.pth or find_pth())


def fold(prefix_conv, prefix_bn, depthwise=False):
    W = sd[prefix_conv + '.weight']                 # [out, in/groups, kh, kw]
    g, b = sd[prefix_bn + '.weight'], sd[prefix_bn + '.bias']
    m, v = sd[prefix_bn + '.running_mean'], sd[prefix_bn + '.running_var']
    scale = g / np.sqrt(v + EPS)
    W = W * scale[:, None, None, None]
    bias = b - m * scale
    if depthwise:   # [C,1,kh,kw] -> [kh,kw,C,1]
        W = W.transpose(2, 3, 0, 1)
    else:           # [out,in,kh,kw] -> [kh,kw,in,out]
        W = W.transpose(2, 3, 1, 0)
    return W.astype(np.float32), bias.astype(np.float32)

# Block config for Lite0 (repeats, kernel, stride, expand, in, out)
stages = [(1,3,1,1,32,16),(2,3,2,6,16,24),(2,5,2,6,24,40),(3,3,2,6,40,80),
          (3,5,1,6,80,112),(4,5,2,6,112,192),(1,3,1,6,192,320)]
tensors = []   # (name, array)
arch = {'stem': {'k':3,'s':2,'out':32}, 'blocks': [], 'head': {'out':1280}, 'classes':1000}
tensors += list(zip(['stem.w','stem.b'], fold('_conv_stem','_bn0')))
bi = 0
for si,(r,k,s,e,cin,cout) in enumerate(stages):
    for j in range(r):
        stride = s if j == 0 else 1
        inp = cin if j == 0 else cout
        p = f'_blocks.{bi}'
        blk = {'i':bi,'stage':si+1,'k':k,'s':stride,'e':e,'in':inp,'out':cout,
               'skip': stride == 1 and inp == cout}
        if e != 1:
            tensors += list(zip([f'b{bi}.expand.w', f'b{bi}.expand.b'], fold(p+'._expand_conv', p+'._bn0')))
        tensors += list(zip([f'b{bi}.dw.w', f'b{bi}.dw.b'], fold(p+'._depthwise_conv', p+'._bn1', depthwise=True)))
        tensors += list(zip([f'b{bi}.project.w', f'b{bi}.project.b'], fold(p+'._project_conv', p+'._bn2')))
        arch['blocks'].append(blk); bi += 1
tensors += list(zip(['head.w','head.b'], fold('_conv_head','_bn1')))
tensors.append(('fc.w', sd['_fc.weight'].T.astype(np.float32)))
tensors.append(('fc.b', sd['_fc.bias'].astype(np.float32)))

manifest, offset, chunks = [], 0, []
for name, arr in tensors:
    h = arr.astype('<f2')
    assert np.isfinite(h).all(), name
    manifest.append({'name': name, 'shape': list(arr.shape), 'offset': offset})
    chunks.append(h.tobytes()); offset += h.size
open(os.path.join(args.out, 'weights.bin'),'wb').write(b''.join(chunks))
json.dump({'format':'float16-le','model':'EfficientNet-Lite0 (ImageNet), BatchNorm folded',
           'arch': arch, 'tensors': manifest}, open(os.path.join(args.out, 'model.json'),'w'), indent=1)
print('tensors', len(tensors), 'params', offset, 'bytes', offset*2)
# max relative error from fp16
err = max(float(np.abs(a.astype(np.float16).astype(np.float32)-a).max()/ (np.abs(a).max()+1e-9)) for _,a in tensors)
print('max rel fp16 err', err)
