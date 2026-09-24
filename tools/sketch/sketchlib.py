"""Shared helpers: QuickDraw rendering, 'webcam photo of pencil on paper' augmentation,
and the sketch normalisation that the web app also applies (keep in sync with js/sketch.js)."""
import numpy as np, io, random
from PIL import Image, ImageDraw, ImageFilter

S = 224

def render_quickdraw(drawing, size=S, width=None, rng=random):
    """Render QuickDraw strokes (0-255 coords) as dark lines on white, centred with margin."""
    xs = np.concatenate([np.array(s[0]) for s in drawing]); ys = np.concatenate([np.array(s[1]) for s in drawing])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    span = max(x1 - x0, y1 - y0, 1)
    fill = rng.uniform(0.62, 0.9)                     # how much of the frame the drawing fills
    scale = size * fill / span
    ox = (size - (x1 - x0) * scale) / 2 - x0 * scale; oy = (size - (y1 - y0) * scale) / 2 - y0 * scale
    big = 3                                           # supersample for smoother lines
    im = Image.new('L', (size * big, size * big), 255); d = ImageDraw.Draw(im)
    w = width if width else rng.uniform(1.4, 3.6)
    for s in drawing:
        pts = [((x * scale + ox) * big, (y * scale + oy) * big) for x, y in zip(s[0], s[1])]
        if len(pts) == 1: pts = pts * 2
        d.line(pts, fill=0, width=max(1, int(round(w * big))), joint='curve')
    return im.resize((size, size), Image.LANCZOS)

def letterbox(im, size=S, bg=255):
    im = im.convert('L'); w, h = im.size; r = size / max(w, h)
    im = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
    out = Image.new('L', (size, size), bg); out.paste(im, ((size - im.width) // 2, (size - im.height) // 2)); return out

def webcam_augment(sketch_L, rng=random):
    """Pretend the sketch was drawn in pencil on paper and held up to a webcam."""
    size = sketch_L.size[0]
    ink = 1.0 - np.asarray(sketch_L, np.float32) / 255.0            # 1 = full pencil
    # random zoom/rotation/shift of the drawing on the paper
    im = Image.fromarray((255 * (1 - ink)).astype(np.uint8))
    z = rng.uniform(0.72, 1.08); ang = rng.uniform(-10, 10)
    im = im.rotate(ang, resample=Image.BILINEAR, fillcolor=255)
    nw = int(size * z); im2 = im.resize((nw, nw), Image.BILINEAR)
    canvas = Image.new('L', (size, size), 255)
    dx = int(rng.uniform(-0.06, 0.06) * size) + (size - nw) // 2; dy = int(rng.uniform(-0.06, 0.06) * size) + (size - nw) // 2
    canvas.paste(im2, (dx, dy)) if nw <= size else canvas.paste(im2.crop(((nw - size) // 2 - dx, (nw - size) // 2 - dy, (nw - size) // 2 - dx + size, (nw - size) // 2 - dy + size)), (0, 0))
    ink = 1.0 - np.asarray(canvas, np.float32) / 255.0
    # paper: off-white with an uneven light gradient / shadow
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = rng.uniform(150, 235)
    grad = 1 + rng.uniform(-0.18, 0.18) * (xx - 0.5) + rng.uniform(-0.18, 0.18) * (yy - 0.5)
    vign = 1 - rng.uniform(0, 0.25) * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 2
    paper = base * grad * vign
    pencil = rng.uniform(0.35, 0.8)                                   # how dark the pencil is relative to paper
    img = paper * (1 - pencil * ink)
    img = np.stack([img * rng.uniform(0.95, 1.05), img, img * rng.uniform(0.9, 1.02)], -1)   # slight colour cast
    img += rng.uniform(1, 6) * np.random.randn(*img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.3)))
    if rng.random() < 0.6:                                             # paper held up in a room, with hands
        cover = rng.uniform(0.6, 0.98); ps = int(size * cover)
        scene = np.zeros((size, size, 3), np.float32)
        top, bot = np.array([rng.uniform(30, 200) for _ in range(3)]), np.array([rng.uniform(20, 160) for _ in range(3)])
        t = np.linspace(0, 1, size)[:, None, None]; scene[:] = top * (1 - t) + bot * t
        scene += rng.uniform(3, 10) * np.random.randn(size, size, 3)
        sc = Image.fromarray(np.clip(scene, 0, 255).astype(np.uint8))
        paper_im = im.resize((ps, ps), Image.BILINEAR).rotate(rng.uniform(-6, 6), resample=Image.BILINEAR, expand=True, fillcolor=(0, 0, 0))
        mask = Image.new('L', (ps, ps), 255).rotate(0, expand=False)
        mask = Image.new('L', (ps, ps), 255).rotate(rng.uniform(-6, 6) * 0, expand=True)
        m2 = Image.fromarray((np.asarray(paper_im).sum(-1) > 0).astype(np.uint8) * 255)
        ox = int((size - paper_im.width) / 2 + rng.uniform(-0.08, 0.08) * size); oy = int((size - paper_im.height) / 2 + rng.uniform(-0.08, 0.08) * size)
        sc.paste(paper_im, (ox, oy), m2)
        d = ImageDraw.Draw(sc); skin = tuple(int(rng.uniform(90, 230)) for _ in range(3))
        for hx in ([0.25, 0.75] if rng.random() < 0.7 else [rng.uniform(0.2, 0.8)]):
            cx, cy, rr = int(hx * size), int(rng.uniform(0.85, 1.05) * size), int(rng.uniform(0.07, 0.13) * size)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr * 1.3], fill=skin)
        im = sc
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=int(rng.uniform(55, 90))); buf.seek(0)
    return Image.open(buf).convert('RGB')

def _upsample_bilinear(small, size):
    """Bilinear upsample (pixel-centre convention) - mirrored exactly in js/sketch.js."""
    n = small.shape[0]
    pos = (np.arange(size) + 0.5) * n / size - 0.5
    pos = np.clip(pos, 0, n - 1)
    i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, n - 1); f = pos - i0
    rows = small[i0] * (1 - f)[:, None] + small[i1] * f[:, None]
    return rows[:, i0] * (1 - f)[None, :] + rows[:, i1] * f[None, :]

def _flood_border(dark):
    """Pixels of `dark` (bool HxW) connected to the image border (4-neighbour)."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(dark)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        edge = edge[edge > 0]
        return np.isin(lab, edge)
    except ImportError:
        pass
    from collections import deque
    H, W = dark.shape; out = np.zeros_like(dark); q = deque()
    for x in range(W):
        for y in (0, H - 1):
            if dark[y, x] and not out[y, x]: out[y, x] = True; q.append((y, x))
    for y in range(H):
        for x in (0, W - 1):
            if dark[y, x] and not out[y, x]: out[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < H and 0 <= xx < W and dark[yy, xx] and not out[yy, xx]: out[yy, xx] = True; q.append((yy, xx))
    return out

def _largest_component(mask):
    """Largest 4-connected component of a bool mask."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        if n == 0: return mask
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        return lab == sizes.argmax()
    except ImportError:
        pass
    from collections import deque
    H, W = mask.shape; lab = np.zeros((H, W), np.int32); best, best_n, cur = 0, 0, 0
    for y0 in range(H):
        for x0 in range(W):
            if not mask[y0, x0] or lab[y0, x0]: continue
            cur += 1; n = 0; q = deque([(y0, x0)]); lab[y0, x0] = cur
            while q:
                y, x = q.popleft(); n += 1
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]: lab[yy, xx] = cur; q.append((yy, xx))
            if n > best_n: best, best_n = cur, n
    return lab == best if best else mask

def _dilate(mask, r):
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        m[1:] |= out[:-1]; m[:-1] |= out[1:]; m[:, 1:] |= out[:, :-1]; m[:, :-1] |= out[:, 1:]
        out = m
    return out

def _pct_index(values_u8, q):
    hist = np.bincount(values_u8.ravel(), minlength=256); cdf = np.cumsum(hist) / hist.sum()
    return int(np.searchsorted(cdf, q))

def normalise(im_rgb, crop=True, return_info=False):
    """The sketch clean-up used by the app (keep in sync with js/sketch.js):
    1. grey-scale;  2. blank out dark areas touching the edge of the box (background, hands);
    3. divide by a smooth estimate of the paper brightness (removes shadows);
    4. stretch so paper -> white and pencil -> black;  5. zoom in on the drawing.  Returns uint8 224x224."""
    im = im_rgb.convert('RGB')
    if im.size != (S, S): im = im.resize((S, S), Image.BILINEAR)
    a = np.asarray(im, np.float32)
    g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    # --- 2. background removal: keep only the paper (the biggest bright region, plus
    #        everything enclosed by it, i.e. the drawing); blank out the rest.
    P = _pct_index(np.clip(g, 0, 255).astype(np.uint8), 0.90)
    bright = g >= 0.8 * P
    paper = _largest_component(bright)
    outside = _flood_border(~paper)
    outside = _dilate(outside, 2)
    bgmask = outside
    g = np.where(bgmask, P, g)
    # --- 3. paper brightness
    B = 7; n = S // B
    small = g.reshape(n, B, n, B).mean(axis=(1, 3))
    pad = np.pad(small, 1, mode='edge')
    small = np.max(np.stack([pad[dy:dy + n, dx:dx + n] for dy in range(3) for dx in range(3)]), 0)
    bg = _upsample_bilinear(small, S)
    r = g / np.maximum(bg, 1.0)
    # --- 4. stretch
    ri = np.clip(np.floor(r * 200), 0, 255).astype(np.uint8)
    lo = _pct_index(ri, 0.01) / 200; hi = _pct_index(ri, 0.90) / 200
    contrast = hi - lo
    if hi - lo < 0.12: lo = hi - 0.12
    v = np.clip((r - lo) / (hi - lo), 0, 1)
    out = (255 * v).astype(np.uint8)
    info = {'contrast': contrast, 'ink': float((v < 0.5).mean()), 'bg': float(bgmask.mean()), 'rect': (0, 0, S)}
    # --- 5. zoom in on the drawing (square box around the pencil marks, plus a margin)
    if crop:
        ys, xs = np.nonzero(v < 0.5)
        if len(xs) > 40:
            x0, x1 = np.percentile(xs, [0.5, 99.5]); y0, y1 = np.percentile(ys, [0.5, 99.5])
            side = max(x1 - x0, y1 - y0) * 1.2 + 8
            side = max(side, 56)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            L = int(round(cx - side / 2)); T = int(round(cy - side / 2)); side = int(round(side))
            canvas = Image.new('L', (side, side), 255)
            canvas.paste(Image.fromarray(out), (-L, -T))
            out = np.asarray(canvas.resize((S, S), Image.BILINEAR))
            info['rect'] = (L, T, side)
    return (out, info) if return_info else out

def to_model_input(norm_u8):
    x = (norm_u8.astype(np.float32) - 127) / 128
    return np.repeat(x[None], 3, 0)                                     # CHW, grey in all 3 channels
