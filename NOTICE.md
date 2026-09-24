# Third-party notices

This project bundles the following third-party material.

## TensorFlow.js (`vendor/tf.min.js`)
TensorFlow.js 4.22.0, Copyright Google LLC. Licensed under the Apache License, Version 2.0.
https://github.com/tensorflow/tfjs

## EfficientNet-Lite0 weights (`model/`)
Pretrained ImageNet weights for EfficientNet-Lite0, originally released by Google in the TensorFlow TPU repository
(https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite) under the Apache License, Version 2.0.
They were obtained as a PyTorch conversion from the `efficientnet_lite0_pytorch_model` package
(Copyright 2020 ML Illustrated, Apache License 2.0; https://github.com/ml-illustrated/EfficientNet-Lite-PyTorch).
For this project they have been modified: BatchNorm layers were folded into the preceding convolutions,
tensors were re-ordered to TensorFlow layout, and values were stored as float16 (see `tools/convert_weights.py`).

## ImageNet class names (`js/imagenet-classes.js`)
From `@tensorflow-models/mobilenet`, Copyright Google LLC, Apache License, Version 2.0.

## Sketch classifier (`model/sketch-head.json`)
A small logistic-regression layer trained by this project. No training images are included. It was trained using:
- **The Quick, Draw! Dataset** by Google, licensed under CC BY 4.0 (https://github.com/googlecreativelab/quickdraw-dataset). Cat, crocodile and 40 other categories of doodles were used.
- **ImageNet-Sketch**: H. Wang, S. Ge, E. P. Xing and Z. C. Lipton, *Learning Robust Global Representations by Penalizing Local Predictive Power*, NeurIPS 2019 (https://github.com/HaohanWang/ImageNet-Sketch).
- Freely licensed drawings of cats and crocodiles from Wikimedia Commons.

## Sample photos (`samples/`)
Public domain or CC0 photos from Wikimedia Commons. Full details are in `samples/CREDITS.md`.

---

Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
