# Liver Cirrhosis — Survival Analysis

Data Science portfolio project (Stackfuel). Mayo Clinic clinical trial data
on primary biliary cirrhosis (PBC) — 418 patients, comparing the drug
D-penicillamine against placebo.

## Data

`data/cirrhosis.csv` — not included in this repository (download separately).

Source: [Kaggle — fedesoriano/cirrhosis-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset)

Download it and place the file at `data/cirrhosis.csv`.

## Structure

```
Portfolie_project_DS/
├── data/
│   └── cirrhosis.csv       # place downloaded file here
├── notebooks/
│   └── 01_survival.ipynb   # Kaplan-Meier + Cox PH
├── main.py
└── pyproject.toml
```

## Setup

```bash
uv sync
uv run jupyter notebook notebooks/01_survival.ipynb
```

## Plan

1. **Survival analysis** — Kaplan-Meier curves, Drug vs Placebo comparison, Cox Proportional Hazards
2. **Classification** — predict `Stage` (1-4) from biomarkers
3. **Interpretation** — feature importance (SHAP)
