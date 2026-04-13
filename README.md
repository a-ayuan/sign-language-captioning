# Sign Language Captioning Base Model

This repository implements a streaming-ready sign-to-gloss pipeline for isolated ASL clips using MediaPipe landmarks, temporal encoders, and a hybrid isolated-sign objective (clip classification + auxiliary CTC).

## What was wrong with the previous training path

The previous version had several issues that can easily collapse accuracy on WLASL-style isolated sign recognition:

1. the training script always instantiated the BiLSTM model even when the Transformer config was selected,
2. preprocessing used a very weak frame-mean normalization that destroyed meaningful body and hand geometry,
3. missing hands were encoded as all-zero frames with no reconstruction,
4. no augmentation was actually applied during training,
5. no class-balancing sampler was used despite WLASL's long-tail label distribution,
6. there was no learning-rate scheduler to stabilize CTC training.

## What changed in this revision

### Preprocessing

- Added signer-invariant normalization built around shoulder-centered body normalization.
- Added separate local hand normalization so hand shape is learned independently from global body position.
- Added temporal interpolation to reconstruct missing hand landmarks instead of leaving zero-only frames.
- Added light temporal smoothing before velocity computation.
- Preserved the original 450-D feature layout:
  - 225 normalized spatial features
  - 225 temporal delta features

### Training

- Added configurable training augmentations:
  - small in-plane rotation,
  - Gaussian landmark noise,
  - temporal speed perturbation,
  - frame dropout with interpolation recovery.
- Added optional weighted sampling for imbalanced gloss distributions.
- Added learning-rate scheduler support.
- Added richer checkpoint metadata.

### Models

- Added a model factory so `model.type` now actually controls whether training uses:
  - `bilstm`, or
  - `transformer`.
- Improved the BiLSTM path with:
  - input layer normalization,
  - learned input projection,
  - GELU activation,
  - input dropout.
- Improved the Transformer path with:
  - input layer normalization,
  - learned projection,
  - GELU feedforward blocks,
  - pre-norm encoder layers.

## Supported datasets

This repository intentionally supports only datasets explicitly named in the proposal:

1. SignAlphaSet
2. Images of American Sign Language Alphabet Gestures
3. World Level American Sign Language / WLASL-processed

The default training path uses WLASL-processed because it is the only listed dataset that naturally supports the temporal video pipeline.

## Repository layout

```text
sign-language-captioning/
├── configs/
│   ├── alphabet_debug.yaml
│   ├── base_wlasl_ctc.yaml
│   └── transformer_wlasl_ctc.yaml
├── scripts/
│   ├── evaluate_ctc.py
│   ├── prepare_alphabet.py
│   ├── prepare_wlasl.py
│   ├── smoke_test.py
│   ├── stream_infer.py
│   ├── train_ctc.py
│   └── visualize_dataset.py
├── src/
│   └── slc/
│       ├── config.py
│       ├── constants.py
│       ├── data/
│       ├── datasets/
│       ├── inference/
│       ├── models/
│       ├── preprocessing/
│       ├── training/
│       └── utils/
├── requirements.txt
├── setup.py
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Kaggle setup for automatic WLASL download

The WLASL pipeline can download and stage the Kaggle dataset automatically with `kagglehub`.

### Option 1: Kaggle API token file

1. Log into Kaggle.
2. Go to **Account**.
3. Create a new API token.
4. Save `kaggle.json` to:

macOS / Linux:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force $HOME\.kaggle
Move-Item $HOME\Downloads\kaggle.json $HOME\.kaggle\kaggle.json
```

### Option 2: Environment variables

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_key"
```

Windows PowerShell:

```powershell
$env:KAGGLE_USERNAME = "your_kaggle_username"
$env:KAGGLE_KEY = "your_kaggle_key"
```

## Prepare WLASL features

Run preprocessing again after pulling this revision because the feature normalization logic changed.

```bash
python scripts/prepare_wlasl.py --download --input-root data/wlasl_processed --output-root outputs/wlasl_features --max-frames 96 --top-k 100
```

This creates:

```text
outputs/
└── wlasl_features/
    ├── features/
    ├── manifests/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   └── vocab.json
    └── preprocessing_summary.csv
```

## Train

### BiLSTM CTC

```bash
python scripts/train_ctc.py --config configs/base_wlasl_ctc.yaml
```

### Transformer CTC

```bash
python scripts/train_ctc.py --config configs/transformer_wlasl_ctc.yaml
```

Training outputs include checkpoints, metrics history, validation predictions, and plots under the configured run directory.

## Evaluate

```bash
python scripts/evaluate_ctc.py --config configs/base_wlasl_ctc.yaml --checkpoint outputs/runs/base_wlasl_ctc/checkpoints/best.pt
python scripts/evaluate_ctc.py --config configs/transformer_wlasl_ctc.yaml --checkpoint outputs/runs/transformer_wlasl_ctc/checkpoints/best.pt
```

## Streaming inference

```bash
python scripts/stream_infer.py --config configs/base_wlasl_ctc.yaml --checkpoint outputs/runs/base_wlasl_ctc/checkpoints/best.pt --video path/to/sample.mp4
```

## Notes on expectations

This project still keeps the original architecture family intact:

- MediaPipe landmark extraction
- temporal sequence encoder
- CTC objective
- greedy streaming decode

The revision improves the weakest engineering points around those components rather than replacing the project with a completely different RGB-heavy model.

## Dependencies

No new third-party dependencies were required for this revision.


## Additional fixes in this revision

- Added a clip-level classification head on top of both temporal encoders.
- Switched isolated-sign prediction to use the clip classifier by default, while keeping CTC as an auxiliary temporal regularizer.
- Added top-5 evaluation because WLASL commonly reports top-k accuracy, especially for large-vocabulary subsets.
- Added `--top-k` preprocessing support so you can build WLASL100, WLASL300, WLASL1000, or full WLASL2000 style subsets for staged debugging and benchmarking.

### Why this matters

WLASL clips are isolated-sign classification examples. Pure greedy CTC decoding is a weak fit for that supervision because each video has one gloss label, while CTC spreads probability mass over many time steps and often collapses to blank-heavy outputs. Existing working WLASL systems generally optimize clip-level classification directly and report top-k classification metrics.
