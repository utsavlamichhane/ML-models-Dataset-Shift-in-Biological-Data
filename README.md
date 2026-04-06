# Microbiome-Based ADG Prediction Model 

A Random Forest regression pipeline that predicts **Average Daily Gain (ADG)** in beef cattle from rumen microbiome community profiles at three taxonomic levels (family, genus, species), with explicit evaluation of cross-farm generalisability through farm-stratified cross-validation.

---

## Background

The rumen microbiome is the primary engine of fermentative digestion in ruminants and plausibly influences feed efficiency and growth rate. This project investigates whether 16S rRNA amplicon-derived microbial relative abundances — in combination with nutritional metadata — can predict ADG across 1,168 animals from 10 commercial beef farms.

The key methodological contribution is the **parallel application of two cross-validation paradigms**:

| Paradigm | Strategy | What it measures |
|----------|----------|-----------------|
| **RandomCV** | 10-fold KFold (farm-blind) | Within-distribution interpolation |
| **FarmBatchCV** | Leave-One-Farm-Out | Out-of-distribution generalisation (dataset shift) |

This design directly quantifies the magnitude of **farm-level dataset shift** — a critical but frequently overlooked problem in multi-site biological prediction studies.

---

## Key Findings

| Condition | R² | RMSE (kg/day) | Pearson r |
|-----------|-----|---------------|-----------|
| RandomCV – Level 5 | **0.316** | 0.627 | 0.564 |
| RandomCV – Level 6 | 0.307 | 0.631 | 0.555 |
| RandomCV – Level 7 | 0.306 | 0.632 | 0.554 |
| FarmBatchCV – Level 5 | **−0.128** | 0.806 | −0.252 |
| FarmBatchCV – Level 6 | −0.094 | 0.793 | −0.255 |
| FarmBatchCV – Level 7 | −0.083 | 0.789 | −0.221 |

**Within farms**, the model achieves modest but significant predictive performance. **Across farms**, performance collapses to below-chance levels — Pearson correlations invert sign — demonstrating profound dataset shift driven by farm-specific microbiome compositions.

---

## Repository Structure

```
github_repo/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore
├── pseudo_code.txt                # Plain-English description of all scripts
│
├── scripts/
│   ├── 01_split_sheets.py         # Split multi-sheet Excel into level files
│   ├── 02_preprocessing.py        # Select metadata + microbiome columns
│   └── 03_run_pipeline.py         # Full RF training + evaluation pipeline
│
├── data/
    ├── sample_preprocessed_Level5.csv   # 10-row sample — family-level features
    ├── sample_preprocessed_Level6.csv   # 10-row sample — genus-level features
    └── sample_preprocessed_Level7.csv   # 10-row sample — species-level features

```

---

## Data Format

Full datasets are not included due to size constraints. The `data/` folder contains 10-row samples demonstrating the expected column layout.

**Required columns per preprocessed file:**

| Column | Type | Description |
|--------|------|-------------|
| `SampleID` | string | Unique animal identifier |
| `Farm_Code` | string | Farm label (A–J) |
| `Weight` | float | Body weight (kg) |
| `ADG` | float | **Target variable** — Average Daily Gain (kg/day) |
| `Crude_Protein` | float | Dietary crude protein (%) |
| `Calcium` | float | Dietary calcium (%) |
| `Phosphorous` | float | Dietary phosphorous (%) |
| `Magnesium` | float | Dietary magnesium (%) |
| `TDN` | float | Total digestible nutrients (%) |
| `[taxon columns]` | float | Relative abundances — one column per taxon |

**Taxonomic levels:**
- **Level 5** — family level (~480 taxa)
- **Level 6** — genus level (~1,350 taxa)
- **Level 7** — species level (~2,300 taxa)

---

## Pipeline

### Step 1 — Split raw Excel sheets

```bash
python scripts/01_split_sheets.py
```
Reads `data.xlsx` (multi-sheet workbook) and writes one file per sheet: `data_Level5.xlsx`, `data_Level6.xlsx`, `data_Level7.xlsx`.

### Step 2 — Preprocess

```bash
python scripts/02_preprocessing.py
```
Selects metadata and microbiome columns (everything after the `Profit` delimiter column), writes `preprocessed_data_Level5.xlsx`, `preprocessed_data_Level6.xlsx`, `preprocessed_data_Level7.xlsx`.

### Step 3 — Train and evaluate

```bash
python scripts/03_run_pipeline.py
```

Runs the full pipeline for all three taxonomic levels under both CV strategies. Outputs are written to:
```
randomCV/Level5/  randomCV/Level6/  randomCV/Level7/
FarmBatchCV/Level5/  FarmBatchCV/Level6/  FarmBatchCV/Level7/
```

Each output directory contains:

| File | Description |
|------|-------------|
| `best_params.txt` | Optimal hyperparameters and grid-search CV RMSE |
| `cv_results.csv` | Full GridSearchCV results table |
| `metrics.txt` | Out-of-fold test metrics |
| `oob_metrics.txt` | Out-of-bag metrics (final all-data model) |
| `feature_importances.csv` | MDI feature importance, all features, ranked |
| `extreme_sample_shap.csv` | SHAP values for the 20 most mis-predicted samples |
| `best_model.joblib` | Serialised final Random Forest |
| `*.png` | Diagnostic plots (see below) |

**Diagnostic plots generated per run:**
- `actual_vs_predicted.png` — scatter coloured by farm (FarmBatchCV)
- `residuals_hist.png` — residual distribution + Shapiro-Wilk test
- `residuals_vs_pred.png` — residuals vs predicted with LOWESS smoother
- `train_test_scatter.png` — training vs CV test predictions overlaid
- `feature_importances.png` — top-30 MDI importance bar chart
- `learning_curve.png` — training/validation RMSE vs training set size
- `shap_summary.png` — SHAP beeswarm summary (top-30 features)

**FarmBatchCV additionally produces:**
- `per_farm/farm_comparison_metrics.csv` — one row per farm with all metrics
- `per_farm/farm_comparison_barplot.png` — 2×2 metric grid across all farms
- `per_farm/Farm_A/` … `per_farm/Farm_J/` — individual farm diagnostic plots

---

## Hyperparameter Search Space

900 combinations evaluated via exhaustive `GridSearchCV`:

| Parameter | Values |
|-----------|--------|
| `n_estimators` | 500, 1000, 1500, 2000 |
| `max_depth` | None, 10, 20, 30, 50 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | sqrt, log2, 0.3, 0.5, None |

Grid search scoring: **negative mean squared error**. All results deterministic with `random_state=42`.

---

## Hardware Requirements

The full pipeline (3 levels × 2 paradigms × 900 parameter combos) is computationally intensive:

- RandomCV Level 7 grid search: ~11.5 hours
- Recommended: 32+ CPU cores, 64+ GB RAM
- GPU acceleration is not used (scikit-learn CPU-only)

For development/testing, reduce `n_estimators` and `max_depth` values in the `PARAM_GRID` dictionary in `scripts/03_run_pipeline.py`.

---

## Installation

```bash
git clone https://github.com/your-username/rumen-adg-prediction.git
cd rumen-adg-prediction
pip install -r requirements.txt
```

Python 3.9+ required.

---



## Pseudocode

`pseudo_code.txt` contains a complete plain-English walkthrough of every script — what each function does, step by step — without any code. Read this first to understand the pipeline logic before looking at the source.

---

## License

MIT License. See `LICENSE` for details.
