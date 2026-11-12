"""
Insert the WDBC case study, screenshots, tables and fixes into the paper.

Reads   Feature Paper.docx (unchanged)
Writes  Feature Paper - WDBC case study.docx
Usage:  python paper_assets/build_paper.py   (after capture_gui / benchmark / make_figures)
"""
import os
import sys
import json
import shutil
import zipfile
import tempfile
from copy import deepcopy

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from docx_helpers import (Doc, para, h1, h2, caption, table, table_caption, run,
                          runs_from_markup, para_text, w)

SRC = os.path.join(PROJECT, "Feature Paper.docx")
DST = os.path.join(PROJECT, "Feature Paper - WDBC case study.docx")
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")

# ------------------------------------------------------------- numbers
G = json.load(open(os.path.join(HERE, "gui_results.json"), encoding="utf-8"))
M = json.load(open(os.path.join(RES, "benchmark_meta.json"), encoding="utf-8"))
S = pd.read_csv(os.path.join(RES, "summary.csv"))
freq = pd.read_csv(os.path.join(RES, "selection_frequency.csv"), index_col=0)["frequency"]
data = pd.read_csv(os.path.join(PROJECT, "data.csv"))

st = G["eval"]["stages"]
sig = G["eval"]["significance"]
f = [s["metrics"]["f1"]["mean"] for s in st]
acc = [s["metrics"]["accuracy"]["mean"] for s in st]
auc = [s["metrics"]["roc_auc"]["mean"] for s in st]
cm = G["final_confusion"]["matrix"]          # [[TN, FP], [FN, TP]] with labels B, M
sens = cm[1][1] / sum(cm[1]); spec = cm[0][0] / sum(cm[0])
sel = G["selected_features"]
sel_lat = [c for c in sel if c.startswith("latent_")]
rad = data["radius_mean"]


def srow(model, cfg):
    return S[(S.model == model) & (S.config == cfg)].iloc[0]


MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
CONFIGS = ["Raw", "Latent only", "Raw + Latent", "Selected"]
raw_f1 = {m: srow(m, "Raw").f1_mean for m in MODELS}
d_rl = {m: srow(m, "Raw + Latent").f1_delta for m in MODELS}
d_lo = {m: srow(m, "Latent only").f1_delta for m in MODELS}
d_sel = {m: srow(m, "Selected").f1_delta for m in MODELS}
nonraw = S[S.config != "Raw"]
min_p_f1, min_p_auc = nonraw.f1_p.min(), nonraw.roc_auc_p.min()
max_abs_delta = nonraw.f1_delta.abs().max()
n_lo_worse = sum(d_lo[m] < 0 for m in MODELS)
k_mean = S[S.config == "Selected"].n_features.mean()
best = S.loc[S.f1_mean.idxmax()]
k2025 = M["k_distribution"].get("20", 0) + M["k_distribution"].get("25", 0)
always = [k for k, v in freq.items() if v >= 1.0]
ica_ae = freq[[k for k in freq.index if k.startswith(("latent_ica", "latent_auto"))]]
rf_sel_vs_rl = srow("Random Forest", "Selected").f1_mean - srow("Random Forest", "Raw + Latent").f1_mean
assert all(d_rl[m] > 0 for m in ["Random Forest", "XGBoost", "LightGBM"]) and d_rl["Logistic Regression"] < 0
assert min_p_f1 > 0.05 and min_p_auc > 0.05


def fmt_p(p):
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def sg(x, nd=4):
    """Signed number with a typographic minus."""
    return f"{x:+.{nd}f}".replace("-", "−")


WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


# ---------------------------------------------------- unpack original
work = tempfile.mkdtemp(prefix="paper_")
with zipfile.ZipFile(SRC) as z:
    z.extractall(work)
D = Doc(work)
IMG = D.images.add
FIGN = {}


def fig(key, label_no, path, text, width=4.8, max_h=None):
    FIGN[key] = label_no
    return [IMG(path, width_in=width, max_height_in=max_h), caption(f"Figure {label_no}.", text)]


# ===================================================== 1. abstract/keywords
ab = D.find(startswith="Abstract")
D.replace_text(ab, "Users can export",
               "We demonstrate the system end-to-end on the Wisconsin Diagnostic Breast Cancer dataset "
               f"(569 samples, 30 features). In an interactive session, latent features and automated "
               f"selection raised the cross-validated F1-score of a Random Forest from {f[0]:.3f} to {f[2]:.3f} "
               f"with 25 of 46 candidate features; a leakage-free benchmark with four classifiers, in which every "
               f"transformation is refit inside the training folds, showed that such changes are small "
               f"(|ΔF1| < 0.01) and not statistically significant, illustrating why "
               f"engineered features must be validated rather than assumed. Users can export")
D.replace_text(ab, "rather than assumed. Users can export",
               "rather than assumed. Loading, profiling, cleaning and every visualization also remained usable on a "
               "real 10.5-million-row dataset on a standard laptop, whereas supervised feature selection at that "
               "scale requires more powerful hardware or sampling. Users can export")
kw = D.find(startswith="Keywords")
D.replace_text(kw, "model evaluation", "model evaluation, breast cancer diagnosis, data leakage, reproducibility")

# ===================================================== 2. introduction
D.insert_after(D.find(startswith="Together, these components"), para(
    "Beyond describing the system, this paper makes three contributions. First, it documents the complete "
    "workflow through screenshots of a real session. Second, it reports an end-to-end case study on the "
    "Wisconsin Diagnostic Breast Cancer (WDBC) dataset [7], [8]. Third, it contrasts the application’s "
    "built-in impact evaluation with a leakage-free benchmark in which all data-dependent steps are refit "
    "inside each training fold [9], [10], and tests differences with the corrected resampled t-test [11]. "
    "The scripts that regenerate every figure and table are released with the paper."))

# ===================================================== 3. methods
D.replace_text(D.find(startswith="The Feature Engineering Assistant is a standalone, offline"),
               "While AutoML frameworks", "While AutoML frameworks [1]–[3]")
D.replace_text(D.find(startswith="The system is implemented as a standalone desktop application"),
               "including those with millions of rows.",
               "including files with millions of rows: loading, profiling and visualization were verified on a "
               "real 10.5-million-row dataset (see Scalability to Large Datasets).")

fe = D.find(startswith="Feature construction uses PCA")
D.insert_after(
    fe,
    para("The generator offers three presets. *Fast* mode extracts four principal components from standardized "
         "inputs; *Balanced* mode adds five FastICA components; *Deep* mode min–max scales the inputs and "
         "produces five PCA [12], five ICA [13] and six autoencoder [14] components. The autoencoder uses a "
         "128–64–d encoder with a mirrored decoder, ReLU activations, a linear bottleneck, embeddings for "
         "categorical columns, mean-squared-error (numeric) and cross-entropy (categorical) reconstruction losses, "
         "and Adam optimization for up to 20 epochs with early stopping (patience 3). The target column is excluded "
         "from all latent generators, and generated columns carry the prefix latent_ so they remain identifiable."),
    para("Supervised selection aggregates three rankings: mutual information with the target [19], model-based "
         "importance (impurity-based for Random Forests [15], XGBoost [16] and LightGBM [17]; absolute coefficients "
         "for linear models), and recursive feature elimination (RFE) with a logistic- or linear-regression base "
         "learner [18]. Features are ordered by their mean rank, and the number retained, K, is chosen from "
         "{5, 10, 15, 20, 25} by five-fold (stratified) cross-validation, maximizing weighted F1 for classification "
         "or minimizing RMSE for regression. All models are implemented with scikit-learn [20] and the respective "
         "boosting libraries."),
    h2("Case-Study Dataset"),
    para("The WDBC dataset [7], [8] contains 569 fine-needle-aspirate images of breast masses, 357 benign (62.7%) "
         "and 212 malignant (37.3%). Thirty real-valued features describe the cell nuclei: the mean, standard error "
         "and “worst” (mean of the three largest values) of ten characteristics—radius, texture, "
         "perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. The "
         "CSV file used here also contains a patient identifier and an empty trailing column, which makes it a "
         "useful test of the cleaning module. The target is the binary diagnosis (M = malignant, B = benign)."),
    h2("Experimental Protocol"),
    para("**Interactive session.** The application was driven end-to-end by a script that invokes the same "
         "callbacks as the GUI buttons and captures each window (see the Case Study section). Settings: target = "
         "diagnosis, task = classification, model = Random Forest (200 trees), Deep feature-creation mode, no "
         "manual pruning, and automatic K. Evaluate Impact uses five-fold stratified cross-validation and reports a "
         "paired t-test and Cohen’s d between consecutive stages. Random seeds were fixed (42); repeated runs "
         "produced identical results."),
    para("**Leakage-free benchmark.** The interactive evaluation fits the latent generators and the feature ranking "
         "on all rows before cross-validation, so its estimates can be optimistic [9], [10]. We therefore re-ran the "
         "pipeline under 10-fold stratified cross-validation repeated three times (30 outer folds). In each outer "
         "fold, scaling, PCA, ICA and the autoencoder were fitted on the training part only and applied to the "
         "held-out part, and the application’s own selection routine—including its inner five-fold choice "
         "of K—was run on the training part only. Four classifiers with the application’s default "
         "hyperparameters (logistic regression; Random Forest, 200 trees; XGBoost, 200 trees of depth 5; LightGBM, "
         "200 trees), each preceded by standardization, were trained on four feature sets: *Raw* (30 cleaned "
         "features), *Latent only* (16), *Raw + Latent* (46) and *Selected* (top-K of Raw + Latent). We report "
         "accuracy, weighted F1, ROC-AUC and the Matthews correlation coefficient (MCC) [21]. Differences from "
         "*Raw* were tested with the Nadeau–Bengio corrected resampled t-test [11], which accounts for "
         "overlapping training sets; 95% confidence intervals use the same corrected variance."),
    para("**Software corrections.** Preparing the case study exposed two defects that were fixed before the "
         "experiments. (i) Under Keras 3 the autoencoder raised an error that was caught silently, so the "
         "latent_auto_* columns were copies of PCA components (correlation 1.0); building the model with dictionary "
         "outputs restored autoencoder training. (ii) ROC-AUC, and therefore the ROC and precision–recall "
         "plots, were silently skipped when class labels were strings; the score is now computed against the "
         "model’s positive class."),
)

ai = D.find(startswith="AI-based language assistance")
D.set_text(ai, "AI-based language assistance (ChatGPT, OpenAI) was used to support linguistic refinement, structural "
               "editing, and clarity of presentation. An AI coding assistant (Claude, Anthropic) was used to review the "
               "source code and correct the two software defects described above, to generate the evaluation and "
               "figure scripts in the reproducibility capsule, and to draft parts of the case-study text. The "
               "application itself was developed by the authors. All experiments were executed "
               "on the authors’ machine, and all code, results, interpretations and final content were reviewed "
               "and approved by the authors, who take full responsibility for the manuscript.")

# ===================================================== 4. existing figure captions
D.set_text(D.find(startswith="Figure 1 System interface"),
           "Figure 1. System interface overview of the Feature Engineering Assistant: the application provides "
           "unified access to dataset exploration, feature creation, pruning, selection, and evaluation.")
D.set_text(D.find(contains="Figure2: Successful Dataset Loading"), "Figure 2. Dataset-loading confirmation.")
D.set_text(D.find(startswith="Figure 3: Numerical variable"), "Figure 3. Numerical-variable profiling pop-up.")

prune = D.find(startswith="To maintain interpretability and prevent feature bloat")
D.replace_text(prune, "over the evolving feature space",
               "over the evolving feature space. Figure 4 shows the feature-creation dialog and the pruning panel.")
D.insert_after(prune, *fig("creation", 4, os.path.join(FIG, "fig_creation_pruning.png"),
                           "(a) Feature Creation dialog with the three latent generators and the Fast, Balanced "
                           "and Deep presets; (b) manual pruning panel with the current columns and the "
                           "banned-feature list (WDBC session).", width=4.2))

arch = D.find_by_blip("rId757210254")
for r in list(arch.iter(w("r"))):
    if any(b.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed") == "rId1920911093"
           for b in r.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip")):
        r.getparent().remove(r)
D.insert_after(arch, caption("Figure 5.", "High-level architecture of the Feature Engineering Assistant, "
                             "summarizing the main workflow components and their interactions."))

# ===================================================== 5. case study section
case = [
    h1("Case Study: Wisconsin Diagnostic Breast Cancer"),
    para("This section follows one complete session on the WDBC dataset and then reports the leakage-free "
         "benchmark. All screenshots were taken from the running application."),
    h2("Loading, Profiling and Cleaning"),
    para(f"Loading the CSV file reported 569 rows, 33 columns and no duplicate rows. Profiling showed moderately "
         f"imbalanced classes (357 benign, 212 malignant) and right-skewed size features; radius_mean, for example, "
         f"ranges from {rad.min():.2f} to {rad.max():.2f} with a mean of {rad.mean():.2f} and a long upper tail "
         f"(Fig. 6). The cleaning module removed the empty trailing column and the ID-like id column, leaving "
         f"{G['after_cleaning_shape'][0]} rows and {G['after_cleaning_shape'][1]} columns (30 features and the "
         f"target) with no missing values (Fig. 7)."),
    *fig("profiles", 6, os.path.join(FIG, "fig_profiles.png"),
         "Column-profiling pop-ups for the WDBC dataset: (a) numeric feature radius_mean with summary statistics "
         "and histogram; (b) categorical target diagnosis."),
    *fig("cleaning", 7, os.path.join(FIG, "03_cleaning.png"),
         "Dataset preview after automated cleaning (33 → 31 columns); the cleaning log appears below the "
         "data preview."),
    h2("Latent Feature Generation and Exploration"),
    para(f"Deep mode added 16 latent features (5 PCA, 5 ICA, 6 autoencoder) in about "
         f"{max(1, round(G['feature_creation_seconds']))} s, giving {G['post_creation_shape'][1]} columns. The five "
         f"principal components explain {M['pca_cumulative_5'] * 100:.1f}% of the variance of the scaled inputs, "
         f"the first alone {M['pca_explained_variance_mean'][0] * 100:.1f}% (means over the 30 benchmark training "
         f"folds). Figure 8 relates every latent feature to the original variables. latent_pca_1, latent_ica_2, "
         f"latent_auto_1 and latent_auto_5 summarize nucleus concavity and size (|r| = 0.76–0.96 with the "
         f"concave-points and concavity features); latent_pca_2 tracks fractal dimension, latent_pca_3 and "
         f"latent_ica_1 texture, and latent_auto_4 overall size (r = −0.83 with area_mean). The Latent Features "
         f"Explorer exposes the same information interactively (Fig. 9): the explainability view ranks the original "
         f"features most correlated with a chosen latent feature, and the importance view shows that latent_pca_1 "
         f"and latent_ica_2 rank first and fourth by Random Forest importance."),
    *fig("corr", 8, os.path.join(FIG, "fig_latent_correlation.png"),
         "Pearson correlation between the 16 latent features (rows) and the 30 original features (columns) "
         "in the WDBC session. Red: positive; blue: negative."),
    *fig("explorer", 9, os.path.join(FIG, "fig_latent_explorer.png"),
         "Latent Features Explorer: (a) original features most correlated with latent_auto_1; (b) Random Forest "
         "importance of original (blue) and latent (orange) features for the target diagnosis.", max_h=5.0),
    h2("Supervised Feature Selection"),
    para(f"With K chosen automatically, the selection routine retained K = {len(sel)} of the 46 candidate features "
         f"in about {round(G['feature_selection_seconds'])} s (Fig. 10). {WORDS[len(sel_lat)]} of the {len(sel)} are latent "
         f"({', '.join(sel_lat)}); the highest-ranked original features (area_worst, perimeter_worst, radius_worst "
         f"and concave points_worst) describe nucleus size and boundary concavity."),
    *fig("selection", 10, os.path.join(FIG, "08_feature_selection.png"),
         "Results panel after supervised feature selection: the retained features (K chosen by cross-validation), "
         "followed by the mutual-information and Random Forest rankings."),
    h2("In-App Impact Evaluation"),
    para(f"Table I and Fig. 11 summarize the Evaluate Impact output. Adding latent features raised the weighted "
         f"F1-score from {f[0]:.4f} to {f[1]:.4f} (p = {fmt_p(sig[0]['p_value'])}, Cohen’s d = "
         f"{sig[0]['effect_size']:.2f}), and selecting 25 features raised it marginally to {f[2]:.4f} "
         f"(p = {fmt_p(sig[1]['p_value'])}); ROC-AUC rose from {auc[0]:.4f} to {auc[2]:.4f}. Pooled over the five "
         f"folds, the final model misclassified {cm[0][1]} of {sum(cm[0])} benign and {cm[1][0]} of {sum(cm[1])} "
         f"malignant cases (sensitivity {sens * 100:.1f}%, specificity {spec * 100:.1f}%, ROC-AUC "
         f"{G['final_pooled_auc']:.3f}; Fig. 12). Two caveats apply: the baseline stage is the raw file, so it still "
         f"contains the id column and the empty column (32 inputs), and both the latent generators and the feature "
         f"ranking were fitted on all 569 rows before cross-validation. The benchmark below removes both effects."),
]
t1_rows = []
names = ["Baseline (raw file)", "+ Latent features", "+ Feature selection"]
for i, s in enumerate(st):
    m = s["metrics"]
    t1_rows.append([names[i], s["n_features"],
                    f"{m['accuracy']['mean']:.4f} ± {m['accuracy']['std']:.4f}",
                    f"{m['f1']['mean']:.4f} ± {m['f1']['std']:.4f}",
                    f"{m['roc_auc']['mean']:.4f} ± {m['roc_auc']['std']:.4f}",
                    "–" if i == 0 else fmt_p(sig[i - 1]["p_value"])])
case += [
    table_caption("TABLE I.", "In-app impact evaluation (Random Forest, five-fold stratified CV, mean ± SD; "
                  "p: paired t-test on per-fold F1 vs. the previous stage)."),
    table(["Stage", "Features", "Accuracy", "F1-score", "ROC-AUC", "p"], t1_rows,
          [1700, 800, 1250, 1250, 1250, 600]),
    *fig("impact", 11, os.path.join(FIG, "09_evaluate_impact.png"),
         "Evaluate Impact output in the Results panel: per-stage accuracy and F1-score, change relative to the "
         "previous stage, and paired t-tests with Cohen’s d."),
    *fig("cmroc", 12, os.path.join(FIG, "fig_confusion_roc.png"),
         "Final-stage diagnostics pooled over the cross-validation folds: (a) confusion matrix; (b) ROC and "
         "precision–recall curves."),
    h2("Leakage-Free Benchmark"),
    para(f"Table II and Fig. 13 report the benchmark (30 outer folds; total run time "
         f"{M['total_seconds'] / 60:.0f} min on a CPU-only Windows PC). All four classifiers performed strongly on the raw "
         f"features (weighted F1 {min(raw_f1.values()):.3f}–{max(raw_f1.values()):.3f}, ROC-AUC ≥ "
         f"{S[S.config == 'Raw'].roc_auc_mean.min():.3f}), leaving little room for improvement. Adding latent "
         f"features to the raw features increased F1 for the three tree ensembles (Random Forest "
         f"{sg(d_rl['Random Forest'])}, XGBoost {sg(d_rl['XGBoost'])}, LightGBM {sg(d_rl['LightGBM'])}) but "
         f"slightly decreased it for logistic regression ({sg(d_rl['Logistic Regression'])}). Latent features "
         f"alone scored below the raw features for {WORDS[n_lo_worse].lower()} of the four models. The selected subsets used "
         f"{k_mean:.1f} features on average—less than half of the 46 candidates—and stayed within "
         f"{max(abs(v) for v in d_sel.values()):.4f} F1 of the raw features. None of the differences from Raw was "
         f"statistically significant (corrected p ≥ {min_p_f1:.2f} for F1 and ≥ {min_p_auc:.2f} for "
         f"ROC-AUC). The best configuration overall was {best.model.lower() if best.model == 'Logistic Regression' else best.model} "
         f"on the {best.config.lower()} features (F1 = {best.f1_mean:.4f}, MCC = {best.mcc_mean:.3f})."),
    para(f"Feature selection was reasonably stable (Fig. 14). K = 20 or 25 was chosen in {k2025} of 30 folds, and "
         f"the mean pairwise Jaccard similarity between selected sets was {M['selection_jaccard_mean']:.2f} ± "
         f"{M['selection_jaccard_sd']:.2f}. {WORDS[len(always)]} features were selected in every fold, among them "
         f"latent_pca_1, the only latent feature in this group. Because ICA and autoencoder components are not "
         f"ordered, their indices do not denote the same component across folds; collectively, the most frequently "
         f"selected of them appeared in {ica_ae.head(6).min() * 100:.0f}–{ica_ae.max() * 100:.0f}% of folds. "
         f"On average {M['latent_share_in_selection'] * 100:.0f}% of the selected features were latent."),
]
t2_rows = []
for mdl in MODELS:
    for j, cfg in enumerate(CONFIGS):
        r = srow(mdl, cfg)
        t2_rows.append([mdl if j == 0 else "", cfg,
                        f"{r.n_features:.0f}" if cfg != "Selected" else f"{r.n_features:.1f}",
                        f"{r.accuracy_mean:.3f}", f"{r.f1_mean:.3f} ± {r.f1_ci:.3f}",
                        f"{r.roc_auc_mean:.3f}", f"{r.mcc_mean:.3f}",
                        "–" if cfg == "Raw" else f"{r.f1_p:.2f}"])
case += [
    table_caption("TABLE II.", "Leakage-free benchmark: means over 30 outer folds (10-fold CV × 3); "
                  "± is the corrected 95% CI half-width; p: corrected resampled t-test on F1 vs. Raw."),
    table(["Model", "Feature set", "n", "Accuracy", "F1-score", "ROC-AUC", "MCC", "p"], t2_rows,
          [1350, 1150, 450, 800, 1150, 800, 650, 500]),
    *fig("bench", 13, os.path.join(FIG, "fig_benchmark.png"),
         "Leakage-free benchmark: mean weighted F1-score and ROC-AUC with corrected 95% confidence intervals for "
         "each classifier and feature set (30 outer folds)."),
    *fig("stab", 14, os.path.join(FIG, "fig_selection_stability.png"),
         "Selection frequency of the 25 most frequently selected features across the 30 outer training folds "
         "(blue: original features; orange: latent features).", width=3.6),
]
# ----------------------------------------------- scalability subsection
SC = os.path.join(HERE, "scalability", "results")
pipe = pd.read_csv(os.path.join(SC, "scalability_pipeline.csv"))
viz = pd.read_csv(os.path.join(SC, "scalability_viz.csv"))


def vz(ds, action, group_contains=None):
    d = viz[(viz.dataset == ds) & (viz.action == action)]
    if group_contains:
        d = d[d.group.str.contains(group_contains, regex=False)]
    return d.iloc[0]


def ps(stage, n):
    d = pipe[(pipe.stage == stage) & (pipe.rows == n)]
    return d.iloc[0] if len(d) else None


def tfmt(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "–"
    return f"{s:.1f} s" if s < 60 else f"{s / 60:.1f} min"


W1, T10 = "WDBC-1M", "Temperature-10.5M"
peak = {ds: viz[viz.dataset == ds].peak_rss_mb.max() / 1000 for ds in (W1, T10)}
num_w = ["Open pop-up (Histogram + KDE)", "KDE", "Box", "Violin", "Scatter", "Line"]
t_num = [vz(T10, a, "numeric").seconds for a in num_w]
t_cat = [vz(T10, a, "categorical").seconds for a in ["Open pop-up (Bar)", "Horizontal", "Pie", "Table",
                                                       "Top-N Categories"]]
t_lat = [vz(T10, a, "Latent").seconds for a in ["Histogram", "KDE", "Scatter vs index", "Correlation heatmap",
                                               "Correlation with original features"]]
sel100, imp100, imp1m = ps("Feature selection", 100000), ps("Latent: RF importance", 100000), \
    ps("Latent: RF importance", 1000000)
fc1m, ev100 = ps("Feature creation, Deep (bg)", 1000000), ps("Evaluate impact (bg)", 100000)
sel1m = ps("Feature selection", 1000000)

scal = [
    h2("Scalability to Large Datasets"),
    para("Non-expert users often work with large exported files, so we tested whether the application remains "
         "usable with millions of rows. Two datasets were used. (i) *WDBC-scaled*: synthetic enlargements of the "
         "case-study file to 10,000, 100,000 and 1,000,000 rows (33 columns, 291 MB at 1M rows), created by "
         "bootstrap-resampling the 569 records and multiplying every numeric value by (1 + 0.01·N(0, 1)); the "
         "files are deterministic (seed 42) and valid for timing only, not for accuracy, because near-duplicate rows "
         "fall into both training and test folds. (ii) *Temperature*: a real, publicly available multi-sensor IoT dataset "
         "from Kaggle [22] with "
         "10,553,597 temperature readings from three devices and five columns (timestamp as text and as epoch, "
         "temperature in °F and °C, device id; 580 MB). Both files are loaded through the normal "
         "*Load Dataset* button. Each stage was driven through the same callbacks as the GUI buttons on a laptop "
         "with an Intel Core i7-8750H CPU (6 cores), 16 GB RAM and no GPU, with other applications open. We "
         "recorded the wall-clock time of each action and the longest period during which the window could not "
         "react, measured with a 20 ms timer in the Tk event loop."),
    para(f"Table III shows that every visualization completed on both datasets. At one million rows "
         f"(WDBC-1M), loading took {vz(W1, 'Load CSV + metadata + preview').seconds:.0f} s, each numeric "
         f"profiling plot (histogram, KDE, box, violin, scatter, line) "
         f"{min(vz(W1, a, 'numeric').seconds for a in num_w):.1f}–"
         f"{max(vz(W1, a, 'numeric').seconds for a in num_w):.0f} s, each categorical plot at most "
         f"{vz(W1, 'WordCloud').seconds:.0f} s, cleaning {vz(W1, 'Data cleaning').seconds:.0f} s and every "
         f"Latent-Explorer plot at most {max(vz(W1, a, 'Latent').seconds for a in ['Histogram', 'KDE', 'Scatter vs index', 'Correlation heatmap', 'Correlation with original features']):.0f} s. "
         f"On the real 10.55-million-row temperature file (Fig. 15), loading took "
         f"{vz(T10, 'Load CSV + metadata + preview').seconds:.0f} s, numeric plots "
         f"{min(t_num):.0f}–{max(t_num):.0f} s, categorical plots "
         f"{min(t_cat):.1f}–{max(t_cat):.0f} s, cleaning (which removed 228 duplicate rows) "
         f"{vz(T10, 'Data cleaning').seconds:.0f} s and Latent-Explorer plots {min(t_lat):.0f}–"
         f"{max(t_lat):.0f} s; peak memory was {peak[T10]:.1f} GB. These actions run on the GUI thread, so the "
         f"window is busy for the duration of each action, but it always returned to normal operation. Fast-mode "
         f"feature creation (PCA) runs in a background thread and kept the window responsive "
         f"({vz(T10, 'Fast mode: 4 PCA components (background thread)', 'Feature').seconds / 60:.1f} min on 10.55M "
         f"rows, longest pause {vz(T10, 'Fast mode: 4 PCA components (background thread)', 'Feature').max_freeze_s:.0f} s)."),
    para(f"The modeling stages scale less well (Table IV, Fig. 16). Supervised feature selection runs on the GUI "
         f"thread and took {sel100.seconds / 60:.1f} min at 100,000 rows, during which the window was frozen; at "
         f"one million rows it had not finished after {sel1m.max_freeze_s / 60:.0f} min and was stopped. The Random "
         f"Forest importance view took {imp100.seconds / 60:.1f} min at 100,000 rows and "
         f"{imp1m.seconds / 60:.0f} min at one million. Deep-mode feature creation ({fc1m.seconds / 60:.0f} min at "
         f"one million rows) and Evaluate Impact ({ev100.seconds / 60:.1f} min at 100,000 rows) are also slow, but "
         f"they run in background threads and the window stayed responsive. For datasets with millions of rows we "
         f"therefore recommend the application for loading, profiling, visualization and cleaning on a standard "
         f"computer, while latent-feature creation and supervised selection should be run on a more powerful "
         f"machine, on a representative sample, or in chunks. The test also exposed one robustness issue unrelated "
         f"to size: the word-cloud view raises an error for text columns that contain no words, such as timestamps."),
]
t3_rows = []
labels3 = [
    ("Load CSV + metadata + preview", None, "Load CSV and build preview"),
    ("Open pop-up (Histogram + KDE)", "numeric", "Numeric profile: histogram + KDE"),
    ("KDE", "numeric", "Numeric profile: KDE"),
    ("Box", "numeric", "Numeric profile: box"),
    ("Violin", "numeric", "Numeric profile: violin"),
    ("Scatter", "numeric", "Numeric profile: scatter"),
    ("Line", "numeric", "Numeric profile: line"),
    ("Open pop-up (Bar)", "categorical", "Categorical profile: bar"),
    ("Pie", "categorical", "Categorical profile: pie"),
    ("Top-N Categories", "categorical", "Categorical profile: top-N"),
    ("WordCloud", "categorical", "Categorical profile: word cloud"),
    ("Data cleaning", None, "Data cleaning"),
    ("Fast mode: 4 PCA components (background thread)", None, "Fast feature creation (PCA)†"),
    ("Histogram", "Latent", "Latent explorer: histogram"),
    ("KDE", "Latent", "Latent explorer: KDE"),
    ("Scatter vs index", "Latent", "Latent explorer: scatter"),
    ("Correlation heatmap", "Latent", "Latent explorer: heatmap"),
    ("Correlation with original features", "Latent", "Latent explorer: correlations"),
    ("Export current dataset to CSV", None, "Export dataset to CSV"),
]
for action, grp, label in labels3:
    a, b = vz(W1, action, grp), vz(T10, action, grp)
    tb = tfmt(b.seconds) if not (action == "WordCloud") else "error‡"
    t3_rows.append([label, tfmt(a.seconds), tb])
t3_rows.append(["Peak memory of the application", f"{peak[W1]:.1f} GB", f"{peak[T10]:.1f} GB"])

t4_rows = []
for stage, label in [("Load CSV", "Load CSV"), ("Profile: histogram + KDE", "Numeric profile (hist. + KDE)"),
                     ("Data cleaning", "Data cleaning"),
                     ("Latent: histogram", "Latent explorer: histogram"),
                     ("Feature creation, Deep (bg)", "Feature creation, Deep†"),
                     ("Latent: RF importance", "Latent explorer: RF importance"),
                     ("Feature selection", "Feature selection"),
                     ("Evaluate impact (bg)", "Evaluate impact†")]:
    cells = [label]
    for n in (10000, 100000, 1000000):
        r = ps(stage, n)
        if r is None:
            cells.append("not run")
        elif pd.isna(r.seconds):
            cells.append(f"> {r.max_freeze_s / 60:.0f} min (stopped)")
        else:
            cells.append(tfmt(r.seconds))
    t4_rows.append(cells)

scal += [
    table_caption("TABLE III.", "Time per visualization action on large datasets (WDBC-1M: 1,000,000 × 33, "
                  "synthetic; Temperature: 10,553,597 × 5, real). The window is busy for the duration of each "
                  "action except † (background thread). ‡ No words in the timestamp column."),
    table(["Action", "WDBC-1M", "Temperature (10.55M)"], t3_rows, [3700, 1500, 1700],
          align=["left", "center", "center"]),
    table_caption("TABLE IV.", "Run time of the full pipeline on WDBC-scaled data (33 columns). † Background "
                  "thread: the window stays responsive; all other stages freeze the window while they run."),
    table(["Stage", "10k rows", "100k rows", "1M rows"], t4_rows, [2700, 1300, 1300, 1600],
          align=["left", "center", "center", "center"]),
    *fig("scal_screens", 15, os.path.join(FIG, "fig_scalability_screens.png"),
         "The application holding the real 10,553,597-row temperature dataset: (a) dataset preview after "
         "loading; (b) violin plot of temp_c computed over all rows; (c) scatter of the first PCA feature "
         "against the row index.", max_h=5.2),
    *fig("scal_time", 16, os.path.join(FIG, "fig_scalability_time.png"),
         "Wall-clock time of pipeline stages versus dataset size (log–log). Dashed lines (†) run in a "
         "background thread; solid lines freeze the window while running. Feature selection did not finish at "
         f"1M rows (stopped after {sel1m.max_freeze_s / 60:.0f} min) and Evaluate Impact was not run at that size."),
]
case += scal

disc = D.find(startswith="Discussion of Core System Features")
for el in case:
    disc.addprevious(el)

# ===================================================== 6. discussion
lat_h = D.find(startswith="Latent Feature Exploration")
D.set_text(lat_h, "Latent Feature Exploration")
D.insert_after(D.find(startswith="Random Forest–based importance comparisons"),
               para("Figure 9 shows two of these views for the WDBC session, and Fig. 8 gives the complete "
                    "latent-to-original correlation map."))

cv_bullet = D.find(startswith="Cross-validation reduces variance")
D.replace_text(cv_bullet, "gives unbiased estimates of model performance.",
               "gives nearly unbiased estimates of generalization performance, provided that every data-dependent "
               "step (scaling, latent-feature fitting, feature selection) is fitted inside the training folds "
               "[9], [10].")

cls_cap = D.find_by_blip("rId2113058063")
cls_cap.addprevious(caption("Figure 17.", "Evaluate Impact output for an example classification dataset (not WDBC): "
                            "accuracy, F1-score, Δ-metrics and paired t-tests across pipeline stages."))
D.remove(cls_cap)
reg_cap = D.find_by_blip("rId763588605")
reg_cap.addprevious(caption("Figure 18.", "Evaluate Impact output for an example regression dataset: RMSE, R², "
                            "Δ-metrics and paired t-tests across pipeline stages."))
D.remove(reg_cap)

lessons = [
    h2("Lessons from the Case Study"),
    para("The case study supports the paper’s central claim that feature engineering must be validated rather "
         "than assumed. On WDBC the raw features are already highly informative, so latent features changed the "
         f"F1-score by at most {max_abs_delta * 100:.1f} percentage points, helped tree ensembles but not logistic "
         "regression, and never produced a statistically significant gain. Their main practical value here was "
         "compression and summarization: the selected subsets roughly halved the number of inputs without loss of "
         "accuracy, and latent_pca_1—a single summary of nucleus concavity and size—was selected in every "
         "fold."),
    para("Comparing the two evaluation protocols is equally instructive. For the Random Forest, adding latent "
         f"features gave similar gains in the application ({sg(f[1] - f[0])}) and in the benchmark "
         f"({sg(d_rl['Random Forest'])}). The selection step behaved differently: in the application, selecting 25 "
         f"features increased F1 further ({sg(f[2] - f[1])}), whereas in the benchmark, where selection was refit "
         f"inside each training fold, the selected subsets scored {abs(rf_sel_vs_rl):.4f} lower than Raw + Latent. "
         "This is the selection bias described by Ambroise and McLachlan [10]: a ranking computed on all rows has "
         "already seen the test folds. In addition, the ordinary paired t-test on five folds understates variance "
         "because training sets overlap [11]. Making these effects visible is itself useful to practitioners, and "
         "nested evaluation is the natural next step for the application."),
]
rep2 = D.find(startswith="Report Export", nth=1)
for el in lessons:
    rep2.addprevious(el)

# limitations / future work bullets (copy existing bullet formatting)
lim_last = D.find(startswith="No hyperparameter tuning pipeline")
new_lims = ["The case study uses a single, small and nearly separable dataset; multi-dataset benchmarks are needed "
            "to generalize the findings.",
            "The built-in Evaluate Impact fits latent generators and feature selection on the full dataset before "
            "cross-validation, which can make its estimates optimistic.",
            "ICA and autoencoder features are stochastic; results are reproducible only when random seeds are fixed.",
            "Supervised feature selection and the importance view run on the GUI thread and become impractical "
            "beyond about 100,000 rows on a standard laptop; the word-cloud view fails on text columns without words."]
cur = lim_last
for t in new_lims:
    b = deepcopy(lim_last); D.set_text(b, t); cur.addnext(b); cur = b
fw_last = D.find(startswith="Support automated pipeline export")
new_fw = ["Implement nested, leakage-free evaluation inside the application, as in the benchmark used here.",
          "Report corrected resampled t-tests and confidence intervals in the Results panel.",
          "Run feature selection on a stratified sample or in chunks, in a background thread, so that "
          "datasets with millions of rows can be modeled on standard hardware."]
cur = fw_last
for t in new_fw:
    b = deepcopy(fw_last); D.set_text(b, t); cur.addnext(b); cur = b

# ===================================================== 7. conclusion
D.replace_text(D.find(startswith="Overall, the system significantly improves"),
               "Overall, the system significantly improves", "Overall, the system improves")
D.replace_text(D.find(startswith="Overall, the system improves"),
               "with substantially reduced manual effort", "with reduced manual effort")
D.insert_after(D.find(startswith="Its modular architecture allows users"), para(
    f"A case study on the WDBC dataset illustrated the complete workflow and the value of explicit validation: "
    f"latent features and automated selection produced compact feature sets that performed on par with the raw "
    f"features (weighted F1 up to {S.f1_mean.max():.3f}, ROC-AUC up to {S.roc_auc_mean.max():.3f}), and a "
    f"leakage-free benchmark showed that the differences observed in the interactive session were within "
    f"statistical noise. A scalability test showed that a standard laptop loads, profiles, cleans and visualizes "
    f"a real 10.5-million-row dataset, which makes the tool practical for non-experts exploring large files, "
    f"while supervised modeling at that scale calls for more powerful hardware or sampling."))

# ===================================================== 8. references
refs = [
    "[7] W. N. Street, W. H. Wolberg, and O. L. Mangasarian, “Nuclear feature extraction for breast tumor "
    "diagnosis,” in *Proc. SPIE 1905, Biomedical Image Processing and Biomedical Visualization*, 1993, "
    "pp. 861–870.",
    "[8] W. H. Wolberg, O. L. Mangasarian, N. Street, and W. Street, “Breast Cancer Wisconsin (Diagnostic),” "
    "*UCI Machine Learning Repository*, 1995, doi: 10.24432/C5DW2B.",
    "[9] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, "
    "and avoidance,” *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, art. 15, 2012.",
    "[10] C. Ambroise and G. J. McLachlan, “Selection bias in gene extraction on the basis of microarray "
    "gene-expression data,” *Proceedings of the National Academy of Sciences*, vol. 99, no. 10, "
    "pp. 6562–6566, 2002.",
    "[11] C. Nadeau and Y. Bengio, “Inference for the generalization error,” *Machine Learning*, vol. 52, "
    "no. 3, pp. 239–281, 2003.",
    "[12] I. T. Jolliffe and J. Cadima, “Principal component analysis: A review and recent developments,” "
    "*Philosophical Transactions of the Royal Society A*, vol. 374, no. 2065, art. 20150202, 2016.",
    "[13] A. Hyvärinen and E. Oja, “Independent component analysis: Algorithms and applications,” "
    "*Neural Networks*, vol. 13, no. 4–5, pp. 411–430, 2000.",
    "[14] G. E. Hinton and R. R. Salakhutdinov, “Reducing the dimensionality of data with neural "
    "networks,” *Science*, vol. 313, no. 5786, pp. 504–507, 2006.",
    "[15] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.",
    "[16] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD "
    "Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.",
    "[17] G. Ke *et al.*, “LightGBM: A highly efficient gradient boosting decision tree,” in *Advances in "
    "Neural Information Processing Systems 30*, 2017, pp. 3146–3154.",
    "[18] I. Guyon, J. Weston, S. Barnhill, and V. Vapnik, “Gene selection for cancer classification using "
    "support vector machines,” *Machine Learning*, vol. 46, pp. 389–422, 2002.",
    "[19] A. Kraskov, H. Stögbauer, and P. Grassberger, “Estimating mutual information,” *Physical "
    "Review E*, vol. 69, no. 6, art. 066138, 2004.",
    "[20] F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning "
    "Research*, vol. 12, pp. 2825–2830, 2011.",
    "[21] D. Chicco and G. Jurman, “The advantages of the Matthews correlation coefficient (MCC) over F1 score "
    "and accuracy in binary classification evaluation,” *BMC Genomics*, vol. 21, art. 6, 2020.",
    "[22] mattpo, “Temperature_IoT_on_GCP,” *Kaggle* dataset. [Online]. Available: "
    "https://www.kaggle.com/datasets/mattpo/temperature-iot-on-gcp (accessed Nov. 12, 2026).",
]
D.insert_after(D.find(startswith="[6]"), *[para(t, before=240, after=240) for t in refs])

# ===================================================== 9. appendix
app_h = D.find(startswith="Appendix")
D.set_text(app_h, "Appendix: Reproducibility")
v = M["versions"]
D.insert_after(
    app_h,
    para("All case-study results can be regenerated from the paper_assets folder distributed with the source code: "
         "capture_gui.py drives the application on data.csv and saves the screenshots together with the numbers "
         "shown in the GUI (gui_results.json); benchmark.py runs the leakage-free benchmark (results/summary.csv, "
         "results/fold_results.csv); make_figures.py builds the composite and chart figures; record_demo.py "
         "records the demonstration video; and build_paper.py inserts the results into this manuscript. Random "
         "seeds were fixed to 42."),
    para(f"Software: Python {v['python']}, scikit-learn {v['scikit-learn']}, TensorFlow {v['tensorflow']}, "
         f"XGBoost {v['xgboost']}, LightGBM {v['lightgbm']}, NumPy {v['numpy']} and pandas {v['pandas']}, on "
         f"a CPU-only Windows 11 PC. The benchmark took {M['total_seconds'] / 60:.0f} min."),
    para("A narrated screen recording of the complete WDBC session (feature_engineering_assistant_demo.mp4, "
         "about 2 min) shows every step described in the Case Study section—loading, profiling, cleaning, "
         "latent-feature creation and exploration, pruning, feature selection, impact evaluation and report "
         "export—and is provided as supplementary material, together with the pipeline report exported by the "
         "application at the end of the session (pipeline_report.docx)."),
    para("The scalability datasets and a step-by-step manual test are provided in paper_assets/scalability "
         "(README.md), so that anyone can load the same files through the Load Dataset button and compare their "
         "timings with Tables III and IV. The synthetic WDBC files are regenerated byte-for-byte by "
         "make_large_dataset.py; SHA-256 checksums: wdbc_1000000.csv 6382d5b6…e5, temperature.csv "
         "2991bcc4…bd (full values in the README)."),
)

# ------------------------------------------------------------- save
D.save()
if os.path.exists(DST):
    os.remove(DST)
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(work):
        for fn in files:
            full = os.path.join(root, fn)
            z.write(full, os.path.relpath(full, work).replace(os.sep, "/"))
shutil.rmtree(work)
print("saved", DST)
print("figures:", FIGN)
