# Digital Twin AI for ARDS Research

Research software for leakage-controlled prediction of acute respiratory distress syndrome (ARDS) from ICU physiology and pre-index clinical reports.

## Intended use

This repository supports retrospective methods research, cohort discovery, and possible clinical-trial screening. It is not a medical device, autonomous diagnostic system, treatment recommender, or substitute for clinical judgment.

## Model architecture

| Component | Model | Responsibility |
|---|---|---|
| Phenotype engine | Deterministic clinical rules | Construct auditable ARDS outcomes and onset times |
| Interpretable baseline | L2 logistic regression | Establish a transparent performance baseline |
| Structured model | XGBoost | Model nonlinear relationships in aggregated ICU features |
| Temporal model | GRU-D | Model hourly physiology with informative missingness |
| Text model | BioClinicalBERT | Extract imaging and cardiac-edema criteria from pre-index reports |
| Fusion model | Logistic regression | Calibrate and combine component probabilities |

The recommended implementation order is phenotype engine, logistic regression, XGBoost, GRU-D, BioClinicalBERT, and late fusion. Advanced components should be retained only when they add validated value over simpler baselines.

## Privacy boundary

This is a public repository. Never commit:

- MIMIC, eICU, or other credentialed clinical data;
- patient-level tables, notes, images, or identifiers;
- derived patient-level features or predictions;
- local databases or query exports;
- trained checkpoints or serialized preprocessing objects;
- API keys, access tokens, credentials, or `.env` files;
- experiment-tracking directories or logs containing patient-level metadata.

Only source code, configuration templates, synthetic examples, aggregate results, and documentation belong in Git. See [Data Governance](docs/data-governance.md).

## Repository structure

```text
configs/                  Experiment configurations
docs/                     Research and governance documentation
src/ards_ai/              Reusable Python package
tests/                    Unit and privacy-boundary tests
.github/workflows/        Public CI checks
```

Real data must remain outside the repository. Provide its location at runtime through a local environment variable or command-line argument.

## Installation

Create an isolated environment and install the baseline package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional temporal and clinical-NLP dependencies only when needed:

```bash
python -m pip install -e ".[temporal,nlp]"
```

## Expected tabular input

The baseline trainer expects a local Parquet file containing:

- `patient_id`: used only for grouped splitting and removed before training;
- `label`: binary target;
- `split`: one of `train`, `validation`, or `test`;
- numeric or boolean feature columns available before the prediction timestamp.

The input file must not be placed inside this repository.

## Train the baseline models

```bash
ards-train-tabular \
  --config configs/tabular.yaml \
  --input /approved/private/location/features.parquet \
  --output /approved/private/location/run-001
```

The command trains logistic regression and XGBoost, fits a probability calibrator on the validation partition, and writes aggregate metrics plus serialized models to the specified private output directory.

## Run tests

```bash
pytest
```

The test suite includes a privacy-boundary check that rejects commonly sensitive file types and credential patterns from the tracked working tree.

## Research safeguards

- Split by patient, not by row, image, report, or ICU hour.
- Freeze the temporal test set before model tuning.
- Exclude post-index information and discharge summaries.
- Report AUPRC with prevalence, AUROC, calibration, and operating-point metrics.
- Evaluate explicit leakage ablations.
- Validate the phenotype with independent clinical review.
- Keep treatment recommendation and reinforcement learning out of the initial scope.

## Project status

The repository provides privacy-safe model scaffolding and a functioning tabular baseline pipeline. Dataset-specific cohort SQL, clinical phenotype validation, and institution-approved data access must be completed before training on real ICU data.
