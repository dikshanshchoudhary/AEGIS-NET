# AEGIS-NET

AI-powered network threat detection for offline flow records and authorized lab
traffic. The application is intended for detection and analysis; it does not
automatically block network traffic.

## Requirements

- Windows 11
- Python 3.15 (64-bit)
- Git

## Stage 1: local setup

Run these commands in PowerShell from the directory containing this project:

```powershell
cd .\AEGIS-NET
py -3.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git init
git branch -M main
```

If PowerShell prevents virtual-environment activation, allow it only for the
current terminal with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The requirements file provides version ranges, but not every dependency may
publish a Python 3.15-compatible build yet. If installation fails, use the
reported package error to determine whether a compatible release is available;
do not silently change versions in a way that could break Colab artifact
compatibility. PyTorch is listed as a dependency; GPU/CUDA availability depends
on the installed PyTorch build and the local NVIDIA driver. Verify it after
setup with:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If an import fails with `An Application Control policy has blocked this file`,
ask your administrator to review the blocked package DLLs or use an
organization-approved Python environment. Do not disable or bypass Windows
application-control policies.

Raw and processed traffic data, model artifacts, local database files, virtual
environments, and credentials are excluded from Git by default. Review dataset
licenses and remove sensitive information before adding samples to the
repository.

## Project layout

- `data/`: raw, processed, and sample flow records
- `notebooks/`: experiments and model-integration notes
- `src/`: preprocessing, feature extraction, models, detection, evaluation, and utilities
- `artifacts/`: exported classifiers, autoencoders, and preprocessing artifacts
- `api/`: FastAPI application
- `dashboard/`: Streamlit interface
- `database/`: local SQLite database
- `tests/`: automated tests

Model integration will use the exported Colab artifacts and their exact
training-time preprocessing, feature names, and feature ordering. Do not
recreate or infer those transformations in this application.

## Stage 2: train and export in Colab

Open [notebooks/aegis_kaggle_train_export.ipynb](./notebooks/aegis_kaggle_train_export.ipynb)
in Google Colab and run its cells. The notebook downloads the Kaggle
CIC-IDS-2017 CSVs, fits preprocessing on its training partition, trains the
classifier and neural models, measures evaluation metrics, and downloads a
versioned artifact bundle. Training results are produced only when the notebook
runs; none are assumed here.

After downloading `aegis_artifact_bundle.zip`, copy it into the project root
and extract its contents into `artifacts`:

```powershell
Expand-Archive -LiteralPath .\aegis_artifact_bundle.zip -DestinationPath .\artifacts
```

Keep the exported manifest, feature order, preprocessing parameters, model
configs, and thresholds together. Application inference integration and
artifact-compatibility tests should use that exact bundle; do not substitute
locally inferred transformations. The local Windows runtime currently needs
administrator review of Application Control blocks before scikit-learn,
PyTorch, and FastAPI imports can be validated.
