"""Build an ONNX copy of the app's EfficientNet-Lite0 (same weights, same TF 'SAME' padding)
so we can extract features for thousands of training images quickly on the CPU."""
import json, os, numpy as np, onnx
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx import helper, TensorProto, numpy_helper
M = json.load(open(os.path.join(ROOT, 'model', 'model.json')))
raw = np.fromfile(os.path.join(ROOT, 'model', 'weights.bin'), dtype='<f2').astype(np.float32)
W = {t['name']: raw[t['offset']:t['offset'] + int(np.prod(t['shape']))].reshape(t['shape']) for t in M['tensors']}
nodes, inits, outs = [], [], []
def init(name, arr): inits.append(numpy_helper.from_array(arr.astype(np.float32), name)); return name
def same_pads(k, s, size):
    out = -(-size // s); total = max((out - 1) * s + k - size, 0); b = total // 2; return [b, b, total - b, total - b], out
cnt = [0]
def nm(p): cnt[0] += 1; return f'{p}_{cnt[0]}'
def conv(x, name, stride, act, size, depthwise=False):
    w = W[name + '.w']; b = W[name + '.b']
    if depthwise: kh, kw, C, _ = w.shape; wo = w.transpose(2, 3, 0, 1); group = C   # [C,1,kh,kw]
    else: kh, kw, cin, cout = w.shape; wo = w.transpose(3, 2, 0, 1); group = 1      # [out,in,kh,kw]
    pads, out = same_pads(kh, stride, size)
    y = nm(name)
    nodes.append(helper.make_node('Conv', [x, init(name + '.W', wo), init(name + '.B', b)], [y], strides=[stride, stride], pads=pads, group=group, kernel_shape=[kh, kw]))
    if act:
        y2 = nm(name + '_relu6'); nodes.append(helper.make_node('Clip', [y, init(nm('lo'), np.array(0.0)), init(nm('hi'), np.array(6.0))], [y2])); y = y2
    return y, out
def gap(x, name):
    y = name + '_gap'; nodes.append(helper.make_node('GlobalAveragePool', [x], [y + '_4d']))
    nodes.append(helper.make_node('Flatten', [y + '_4d'], [y])); outs.append(y)
TAPS = {'stem', 'b3.expand', 'b5.expand', 'b8.expand', 'b11.expand', 'b15.expand', 'head'}
h, size = conv('input', 'stem', 2, True, 224); gap(h, 'stem')
for b in M['arch']['blocks']:
    inp = h; i = b['i']
    if b['e'] != 1:
        h, _ = conv(h, f'b{i}.expand', 1, True, size)
        if f'b{i}.expand' in TAPS: gap(h, f'b{i}.expand')
    h, size = conv(h, f'b{i}.dw', b['s'], True, size, depthwise=True)
    h, _ = conv(h, f'b{i}.project', 1, False, size)
    if b['skip']: y = nm('add'); nodes.append(helper.make_node('Add', [h, inp], [y])); h = y
h, _ = conv(h, 'head', 1, True, size); gap(h, 'head')
nodes.append(helper.make_node('Gemm', ['head_gap', init('fc.W', W['fc.w']), init('fc.B', W['fc.b'])], ['logits'])); outs.append('logits')
g = helper.make_graph(nodes, 'effnet_lite0', [helper.make_tensor_value_info('input', TensorProto.FLOAT, ['N', 3, 224, 224])],
                      [helper.make_tensor_value_info(o, TensorProto.FLOAT, ['N', None]) for o in outs], inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
onnx.checker.check_model(m); onnx.save(m, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lite0.onnx')); print('saved', outs)
