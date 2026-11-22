# AEGIS-NET

AI-powered threat detection for offline network-flow records and authorized lab
traffic. The application is detection-only and does not automatically block
network traffic.

## Requirements

- Windows 11, 64-bit
- Python 3.15
- NVIDIA GPU and a compatible driver for CUDA training (the local setup uses a
  CUDA 13 PyTorch wheel)
- Git

## Local setup

From the folder containing the project, run in PowerShell:

```powershell
cd .\AEGIS-NET
py -3.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the actual runtime before training:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

The verified Windows/Python 3.15 CUDA 13 wheel is specified in
`requirements-cuda.txt`. On this machine, Windows Application Control blocked
its `torch\lib\c10.dll` at runtime (WinError 4551), so GPU training is currently
unavailable for PyTorch. Ask the administrator to review/approve the CUDA
package; do not disable or bypass Application Control. A small XGBoost CUDA
training smoke test did succeed, so the current training pipeline can use that
backend separately. Do not treat an NVIDIA driver alone as proof PyTorch is
using the GPU.

After administrator approval, install the CUDA build with:

```powershell
python -m pip install -r requirements-cuda.txt
```

## Train locally

The training pipeline downloads the public CIC-IDS-2017 flow dataset through
KaggleHub unless `--dataset-dir` points to an existing extracted dataset:

```powershell
python -m src.models.train_models --device cpu --xgboost-device cuda --overwrite-artifacts
```

The pipeline uses one shared fitted preprocessing contract, then exports the
XGBoost classifier, ANN, CNN, RNN, LSTM, Transformer, and benign-only
autoencoder with their model configurations, labels, anomaly threshold,
evaluation metrics, and feature order under `artifacts\`. The `--overwrite-artifacts`
flag replaces existing model artifacts.

Until PyTorch CUDA is approved, the command above trains the neural models and
autoencoder on CPU and XGBoost on CUDA. After approval and once
`torch.cuda.is_available()` returns `True`, use `--device cuda`; the `auto`
XGBoost setting will then also select CUDA when supported by its build. The
dataset contains fixed-width flow vectors, not time-indexed sequences: CNN
convolves over feature order, and RNN/LSTM/Transformer treat ordered feature
indices as tokens. These are not temporal sequence models; meaningful temporal
learning requires flow timestamps and grouping/windowing data.

The default evaluation is a stratified random split from the same dataset. It
does not establish generalization to a different day, network, or deployment,
and duplicate or correlated flows may make results optimistic. Read the
exported metrics and validate with a day- or capture-separated test set before
using results operationally. Model scores are not guarantees of correctness or
calibrated probabilities. Anomaly thresholds must also be assessed against
false-positive and attack-recall requirements.

## Validate and run

Run the automated tests:

```powershell
python -m pytest -q
```

The detection engine loads the complete artifact bundle and combines the
XGBoost and neural classifier outputs using `detection.classifier_weights` in
`config.yaml`. It reports model scores, anomaly score, configured risk, and
explanations; it does not take network-blocking actions.

## Project layout

- `data\`: raw, processed, and sample flow records
- `notebooks\`: local experiments
- `src\`: preprocessing, models, detection, evaluation, and utilities
- `artifacts\`: exported models and preprocessing contract
- `api\`: FastAPI application
- `dashboard\`: Streamlit interface
- `database\`: local SQLite database
- `tests\`: automated tests

Raw data, model artifacts, local database files, virtual environments, and
credentials are excluded from Git by default. Review dataset licensing and
remove sensitive information before adding samples to the repository.
