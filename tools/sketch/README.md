# Sketch mode: how the extra layer was trained

Sketch mode uses the same EfficientNet-Lite0 network as photo mode, plus one small extra layer. That layer is a logistic regression with 3 outputs (cat, alligator, neither). It reads the average activations of two deep layers, `b15.expand` (1,152 numbers) and `head` (1,280 numbers), and is stored in `model/sketch-head.json`.

You don't need anything in this folder to run the demo.

## Why an extra layer?

The photo classifier learned from photographs, and ImageNet-trained networks rely heavily on texture. On pencil drawings it says "cat" for almost anything, and its confidence is tiny. We tested it on 51 real drawings: it got every cat right but only 42% of the alligators. With the extra layer, held-out accuracy on real sketches it had never seen is:

| Test set (not used for training) | Cat vs alligator |
|---|---|
| ImageNet-Sketch pencil sketches | 93% |
| Line drawings from Wikimedia Commons | 92% |
| Quick, Draw! doodles | 97% |
| Pencil-style versions of photos | 94% |

It's weaker on dense Victorian engravings (about 62%). Those aren't what visitors will draw. 5-fold cross-validation of the shipped model on all the real drawings (ImageNet-Sketch plus Commons) gives 94% for cats and 88% for alligators on cat vs alligator.

## Training data (about 3,700 drawings, each shown 3 ways)

- **ImageNet-Sketch** (Wang et al., 2019): 193 cat sketches (ImageNet classes 281–285), 100 alligator/crocodile sketches (classes 49–50) and 515 sketches of other things. They were fetched through the Hugging Face dataset viewer from `Huanyiiiii/Imagenet_Sketch`, which is sorted by label, so the cat and alligator rows can be read by offset.
- **Quick, Draw!** (Google, CC BY 4.0): 700 cats, 700 crocodiles, and 30 each of 40 other things (dog, snake, fish, house, face…), all ones the game recognised. They came from the `full/simplified/<name>.ndjson` files in the `quickdraw_dataset` Google Cloud Storage bucket.
- **Wikimedia Commons:** 51 freely licensed drawings of cats and crocodiles.
- **Synthetic "neither" pages:** blank paper, scribbles, writing and simple shapes.

Each drawing is used as-is and also as two "pretend webcam photos". `sketchlib.webcam_augment` puts it on off-white paper under uneven light, adds blur and camera noise, and usually adds a room background and hands. It is then cleaned up by `sketchlib.normalise`, which is **exactly** what the app does in `js/sketch.js` (checked to within 0.2 grey levels):

1. Grey-scale.
2. Keep only the sheet of paper (the largest bright region and what it encloses), removing background and hands.
3. Divide by a smooth estimate of the paper's brightness, which removes shadows.
4. Stretch so paper is white and pencil is black.
5. Zoom in on the pencil marks.

## The score

`margin` = the extra layer's log-odds for the animal the student says they drew, against the other two outputs.
`score = 100 / (1 + exp(-margin / 7))`

The temperature of 7 stretches the top of the scale, so good drawings don't all tie at 99. A typical real sketch scores about 70, the top 10% score 90 or more, and the best sketch the AI learned from scores about 99. "Better than X%" compares the margin with held-out margins of the ImageNet-Sketch cats or alligators, stored in `reference` in the JSON.

## Re-training

```
pip install numpy pillow scipy scikit-learn onnx onnxruntime
python tools/sketch/build_onnx.py                   # ONNX copy of the network (fast on CPU)
python tools/sketch/train_sketch_head.py DATA_DIR   # rebuilds model/sketch-head.json (~10 minutes)
```

`DATA_DIR` needs `quickdraw.json`, `isketch/meta.json` plus images, and optionally `drawings/meta.json`. The formats are described at the top of `train_sketch_head.py`.
