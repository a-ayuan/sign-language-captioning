# Sign Language Captioning Base Model

This repository implements a **base streaming-ready sign-to-gloss model** that follows the proposal's stated design:

## Supported datasets

This repository intentionally supports **only datasets explicitly named in the proposal**:

1. **SignAlphaSet** (image dataset)
2. **Images of American Sign Language (ASL) Alphabet Gestures** (image dataset)
3. **World Level American Sign Language / WLASL-processed** (video dataset)

### What this starter model trains on by default

The default training path uses **WLASL-processed** because it is the only listed dataset that naturally supports the proposal's temporal video pipeline.

The alphabet datasets are supported as optional utilities for:

- preprocessing sanity checks,
- landmark extraction debugging,
- simple image-level baselines,
- label pipeline validation.

## What the base model does

### Main base model

- Input: ASL word video clip
- Preprocessing: MediaPipe pose + left hand + right hand landmarks per frame
- Temporal model: BiLSTM encoder
- Objective: CTC loss
- Decode: greedy CTC decoding
- Output: gloss token(s)

Even when the isolated WLASL clip contains one target gloss, training with CTC is still useful because it keeps the architecture compatible with the final streaming project.

### Streaming-ready components included

- sliding window chunking
- greedy prefix decoding
- caption churn / stability metrics
- latency and FPS measurement hooks

## Repository layout

```text
sign-language-captioning-base/
├── configs/
│   ├── base_wlasl_ctc.yaml
│   └── alphabet_debug.yaml
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── prepare_wlasl.py
│   ├── prepare_alphabet.py
│   ├── train_ctc.py
│   ├── evaluate_ctc.py
│   ├── visualize_dataset.py
│   ├── stream_infer.py
│   └── smoke_test.py
├── src/
│   └── slc/
│       ├── data/
│       │   ├── __init__.py
│       │   └── kaggle_wlasl.py
│       ├── datasets/
│       ├── inference/
│       ├── models/
│       ├── preprocessing/
│       ├── training/
│       └── utils/
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv    # Version 3.10+
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

The WLASL pipeline can now **download and stage the Kaggle dataset automatically**.

This uses `kagglehub`, which still requires Kaggle authentication.

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

## Automatic WLASL download and preprocessing

The `prepare_wlasl.py`:

1. download the proposal-approved Kaggle dataset,
2. detect whether the dataset already contains `train/val/test` folders,
3. if not, locate the WLASL metadata JSON and raw `videos/` directory,
4. build `train/val/test/<gloss>/...` locally under `data/wlasl_processed/`,
5. extract landmarks,
6. save local feature caches and manifests.

### One-command path

```bash
python scripts/prepare_wlasl.py --download --input-root data/wlasl_processed --output-root outputs/wlasl_features --max-frames 96
```

If the dataset has already been staged under `data/wlasl_processed/`, the script reuses it.

### What gets created locally

```text
data/
└── wlasl_processed/
    ├── train/
    ├── val/
    └── test/

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

## Optional manual dataset placement

If you already downloaded the dataset yourself, the script still supports manual placement.

Expected WLASL layout:

```text
data/
└── wlasl_processed/
    ├── train/
    │   ├── hello/
    │   │   ├── clip1.mp4
    │   │   └── clip2.mp4
    │   └── thanks/
    ├── val/
    └── test/
```

Alphabet image dataset examples:

```text
data/
├── signalphaset/
│   ├── A/
│   ├── B/
│   └── ...
└── asl_alphabet_images/
    ├── A/
    ├── B/
    ├── Delete/
    └── Space/
```

## End-to-end usage

### 1) Prepare WLASL landmark features

```bash
python scripts/prepare_wlasl.py --download --input-root data/wlasl_processed --output-root outputs/wlasl_features --max-frames 96
```

### 2) Visualize the prepared dataset

```bash
python scripts/visualize_dataset.py --manifest outputs/wlasl_features/manifests/train.csv --output-dir outputs/visualizations/dataset_train
```

This saves:

- class distribution bar chart
- sequence length histogram
- sample landmark trajectory figure
- dataset summary CSV

### 3) Train the base CTC model

```bash
python scripts/train_ctc.py --config configs/base_wlasl_ctc.yaml
```

Training saves locally:

- checkpoints
- train/val metrics CSV
- training curves PNG
- validation confusion matrix PNG
- decoded predictions CSV
- config snapshot

### 4) Evaluate the model

```bash
python scripts/evaluate_ctc.py --config configs/base_wlasl_ctc.yaml --checkpoint outputs/runs/base_wlasl_ctc/checkpoints/best.pt
```

### 5) Run streaming-style inference on a clip

```bash
python scripts/stream_infer.py --checkpoint outputs/runs/base_wlasl_ctc/checkpoints/best.pt --config configs/base_wlasl_ctc.yaml --video path/to/example.mp4
```

This saves local outputs for:

- frame-by-frame decoded prefixes
- chunk predictions
- streaming latency summary
- caption stability summary

## Visualizations included

### Dataset visualizations
- class distribution
- sequence-length distribution
- sample landmark trajectory plot
- preprocessing summary CSV

### Training visualizations
- loss curves
- clip accuracy curves
- token error rate curves
- confusion matrix
- confidence histogram

### Streaming visualizations
- prefix evolution over time
- caption churn over time
- chunk latency summary

## Git and local-data policy

Large local datasets and generated artifacts are ignored by default.

Ignored paths include:

- `data/`
- model checkpoints
- cached feature arrays
- local logs
- generated outputs

Keeps the repo lightweight for team collaboration while preserving reproducible local runs.
