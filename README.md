# PADSNet

PADSNet is a three-class laryngeal video classification model that combines:

- an RGB+mask 3D Swin Transformer visual branch;
- an LSTM branch for vocal-fold distance sequences;
- wavelet cross-attention for visual/physical temporal interaction; and
- physics-aware gating based on mobility, closure, asymmetry, energy, and periodicity.

The main pipeline keeps each original sample as a 300-frame sequence. The dataset loader divides it into 32-frame clips during training and evaluation. All paths below are relative to the repository root and can be overridden through command-line arguments.

## Repository layout

```text
PADSNet/
├── padsnet/
│   ├── models/padsnet.py
│   └── data/
│       ├── video_dataset.py
│       └── vocal_fold_features.py
├── tools/quality_aware_segment_selection.py
├── splits/
│   └── README.md
├── weights/
│   └── README.md
├── train.py
├── evaluate.py
├── requirements.txt
└── docs/reference_performance.md
```

`tools/quality_aware_segment_selection.py` is an optional preprocessing experiment. It is not called by the default PADSNet training pipeline.

## Installation

Python 3.9 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected local layout

The dataset and weights are not part of the repository. Place them locally using the following layout, or pass alternative paths on the command line.

```text
PADSNet/
├── data/
│   ├── frames/
│   │   ├── train/
│   │   │   ├── high-risk/sample_001/*.png
│   │   │   ├── low-risk/sample_002/*.png
│   │   │   └── normal/sample_003/*.png
│   │   └── test/
│   │       └── <class>/<sample>/*.png
│   └── masks/
│       ├── sample_001_0001.png
│       └── ...
├── splits/
│   ├── train.txt
│   └── test.txt
├── weights/
│   ├── swin_tiny_kinetics400.pth
│   └── padsnet_model_epoch_100.pth
└── outputs/                  # generated automatically
```

Masks must use pixel value `128` for one vocal fold and `255` for the other. Frame filenames must contain a numeric frame index and must be globally unique when masks are stored in one flat directory.

Each split-file row contains a relative sample directory and an integer label:

```text
high-risk/sample_001 0
low-risk/sample_002 1
normal/sample_003 2
```

Do not publish real patient identifiers in split files.

## Training

The training command creates a stratified video-level validation split from the supplied training list.

```bash
python train.py
```

The default relative paths are:

```text
frames:     data/frames/train
masks:      data/masks
split:      splits/train.txt
pretrained: weights/swin_tiny_kinetics400.pth
output:     outputs/checkpoints/padsnet
```

Alternative locations can be supplied with `--data-root`, `--mask-root`, `--train-split`, `--pretrained`, and `--output-dir`. Defaults reproduce the main setup: 100 epochs, batch size 2, 32-frame non-overlapping training clips, AWB loss, and gradient accumulation over eight steps. Use `--balanced-sampling` only when weighted clip sampling is desired.

## Evaluation

Evaluation covers the complete sample using 32-frame clips and combines clip predictions with class-wise top-k mean pooling.

```bash
python evaluate.py
```

By default, evaluation reads `data/frames/test`, `data/masks`, `splits/test.txt`, and `weights/padsnet_model_epoch_100.pth`. Results are written to `outputs/evaluation/padsnet`. These locations can also be overridden by command-line arguments.

## Model inputs and output

`LaryngealRecognitionModel.forward(images, masks, distance_seq)` expects:

| Tensor | Shape | Description |
| --- | --- | --- |
| `images` | `[B, 3, 32, 224, 224]` | RGB clip |
| `masks` | `[B, 32, 224, 224]` | Vocal-fold segmentation masks |
| `distance_seq` | `[B, 32, 8]` | Per-frame bilateral distance features |

The model returns classification logits with shape `[B, 3]`. The evaluation script converts clip logits to probabilities and applies class-wise top-5 mean fusion to obtain one prediction per 300-frame sample.

## Model weights and data

Clinical data and trained weights are intentionally excluded. The existing checkpoints exceed GitHub's normal single-file limit; publish them through Git LFS, a GitHub Release, or a separate model host after completing privacy review.

## Optional quality-aware segment selection

This auxiliary tool scores candidate segments using duration, image sharpness, vibration periodicity, and bilateral symmetry:

```bash
python -m tools.quality_aware_segment_selection \
  --data-root data/frames/train \
  --mask-root data/masks \
  --output-root outputs/selected_segments
```

Its output is separate from the default 300-frame PADSNet training pipeline.
