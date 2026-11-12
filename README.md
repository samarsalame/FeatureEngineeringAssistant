# Feature Engineering Assistant

An interactive desktop workbench for transparent feature engineering and impact evaluation.

Feature Engineering Assistant is an open-source Python/Tkinter application that brings the whole
feature-engineering workflow into one window: dataset profiling, automated cleaning, latent feature
generation (PCA, ICA and a deep autoencoder), manual pruning, supervised feature selection, a
Latent Features Explorer that explains what each latent feature represents, and cross-validated
evaluation of every pipeline stage with significance tests. Each session can be exported as a
DOCX report. It runs fully offline and requires no programming.

![Main window after loading and cleaning a dataset](paper_assets/figures/03_cleaning.png)

## Features

| Step | What it does |
|---|---|
| **Load dataset** | CSV, Excel or Parquet. Shows rows, columns, duplicates and, per column, type, inferred role (numeric, categorical, datetime, ID-like, ...), missing % and unique values. |
| **Profile columns** | Double-click a column: histogram, KDE, box, violin, scatter and line plots for numeric data; bar, pie, table, top-N and word cloud for categorical data. ydata-profiling and skimpy reports from the *Plots* tab. |
| **Data cleaning** | Removes empty, constant, mostly missing (>60 %) and ID-like columns and duplicate rows; imputes the rest (median / mode); logs every change. |
| **Feature creation** | Latent features in three presets: *Fast* (PCA), *Balanced* (PCA + ICA), *Deep* (PCA + ICA + tabular autoencoder). The target column is excluded; new columns are prefixed `latent_`. |
| **Data pruning** | Remove columns manually; they stay banned until reset. |
| **Feature selection** | Combines mutual information, model importance (Random Forest, XGBoost, LightGBM or linear) and RFE; the number of features K is chosen by cross-validation. |
| **Latent Features Explorer** | Distributions, latent correlation heatmap, original features most correlated with a latent feature, and importance of latent vs. original features. |
| **Evaluate impact** | Rebuilds the baseline, pruned, post-creation and selected datasets and scores each with the same model under k-fold CV (accuracy, F1, ROC-AUC or RMSE, MAE, R²), with paired t-tests, Cohen's d, confusion matrix and ROC / precision–recall curves. |
| **Export** | DOCX report of the session and CSV export of the current dataset. |

![Architecture](paper_assets/figures/fig_architecture.png)

## Installation

Requirements: Python 3.10 (recommended; also tested with 3.13), Windows 10/11. No GPU is needed.
Tkinter is included in the standard Python installer on Windows and macOS; on Linux install it
with your package manager (e.g. `sudo apt install python3-tk`).

```bash
git clone <repository URL>
cd <repository folder>

# create and activate a virtual environment with Python 3.10
py -3.10 -m venv venv            # Windows   (Linux/macOS: python3.10 -m venv venv)
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)

pip install -r FinalFeatureEng/requirements.txt
```

## Usage

```bash
cd FinalFeatureEng
python Main.py
```

A typical session, following the sidebar from top to bottom:

1. **Load Dataset** and choose a file (try `data.csv` in the repository root).
2. Double-click columns in the table to profile them.
3. **Data Cleaning**.
4. *Control Settings* tab: enter the target column, choose the task type and model, click **Apply Settings**.
5. **Feature Creation**: choose *Fast*, *Balanced* or *Deep* mode.
6. *Latent Features* tab: inspect, explain and compare the new features.
7. **Data Pruning** (optional) and **Feature Selection**.
8. **Evaluate Impact**; then *Confusion Matrix* and *ROC & PR Curves* in the *Results Panel*.
9. **Export Report** (DOCX) or *Export Current Dataset* (CSV).

Large files: loading, profiling, plotting and cleaning work with millions of rows on a standard
laptop (tested with 10.5 million rows). Feature creation in *Deep* mode and supervised feature
selection become slow beyond about 100,000 rows; for large datasets run them on a more powerful
machine or on a sample. See [paper_assets/scalability/README.md](paper_assets/scalability/README.md).

## Repository structure

```
FinalFeatureEng/          the application
  Main.py                 GUI and workflow (start here)
  dataset.py              loading and shared dataset state
  column_profile.py       column profile pop-ups
  datareport.py           ydata-profiling / skimpy reports
  data_preprocessing.py   data cleaning
  feature_creation.py     PCA, ICA and autoencoder latent features
  Pruning.py              manual pruning
  feature_selection.py    supervised feature selection
  see_latent.py           Latent Features Explorer
  evaluate_impact.py      stage-wise impact evaluation
  export_report.py        DOCX report
  dataset_downloads.py    CSV export
  requirements.txt
data.csv                  example dataset (Wisconsin Diagnostic Breast Cancer)
paper_assets/             reproducible capsule for the paper (scripts, results, figures)
```

## Reproducing the paper results

All results in the accompanying paper can be regenerated with the scripts in `paper_assets/`
(random seeds fixed to 42). Install the extra dependencies first:

```bash
pip install -r paper_assets/requirements.txt
```

Run from the repository root:

| Command | Result |
|---|---|
| `python paper_assets/capture_gui.py` | Drives the GUI through the full workflow on `data.csv`; saves screenshots and the numbers shown in the GUI (`paper_assets/gui_results.json`). *Windows only.* |
| `python paper_assets/benchmark.py` | Leakage-free benchmark: 10-fold CV × 3, four classifiers, all transformations refit inside each training fold (`paper_assets/results/`). About 7 min on a laptop CPU. |
| `python paper_assets/make_figures.py` and `python paper_assets/make_architecture.py` | Paper figures (`paper_assets/figures/`). |
| `python paper_assets/record_demo.py` | Narrated demo video of the workflow. *Windows only.* |
| `paper_assets/scalability/` | Large-dataset tests and a step-by-step manual test protocol; see its [README](paper_assets/scalability/README.md). |

## Datasets

* `data.csv`: Wisconsin Diagnostic Breast Cancer (WDBC). W. H. Wolberg, O. L. Mangasarian,
  N. Street, W. Street, UCI Machine Learning Repository, 1995, https://doi.org/10.24432/C5DW2B
  (CC BY 4.0).
* Large-dataset tests: *Temperature_IoT_on_GCP* (Kaggle,
  https://www.kaggle.com/datasets/mattpo/temperature-iot-on-gcp), not stored in this repository,
  and synthetic enlargements of WDBC that are regenerated by
  `paper_assets/scalability/make_large_dataset.py`.

## Citation

If you use this software, please cite the accompanying paper:

> S. Salame, A. Kanaan, H. Younis. Feature Engineering Assistant: An interactive desktop workbench
> for transparent feature engineering and impact evaluation. *(submitted to SoftwareX)*

and the archived software version: *(Zenodo DOI to be added)*.

## License

Released under the [MIT License](LICENSE).

## Authors

Samar Salame, Aya Kanaan and Hanin Younis, Master of Data Science Program, Lebanese American
University, Beirut, Lebanon. Questions and bug reports: please open an issue in this repository.
