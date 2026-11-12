"""
SoftwareX Original Software Publication (OSP) version of the paper.

The content is written once (CONTENT section) and rendered twice:
  * an Overleaf-ready LaTeX project (elsarticle class)  -> SoftwareX_Overleaf/ and SoftwareX_Overleaf.zip
  * a Word version                                       -> Feature Engineering Assistant - SoftwareX.docx

Structure follows the SoftwareX OSP template as used in the sample paper (softwarex sample.pdf).
All numbers are read from the result files; unknown details are placeholders
(red in LaTeX via \\ph{...}, yellow highlight in Word).

Usage:  python paper_assets/build_softwarex.py
"""
import json
import os
import re
import shutil
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
SC = os.path.join(HERE, "scalability", "results")
OUT_DOCX = os.path.join(PROJECT, "Feature Engineering Assistant - SoftwareX.docx")
OUT_TEX_DIR = os.path.join(PROJECT, "SoftwareX_Overleaf")
OUT_ZIP = os.path.join(PROJECT, "SoftwareX_Overleaf.zip")

# ===================================================================== numbers
G = json.load(open(os.path.join(HERE, "gui_results.json"), encoding="utf-8"))
M = json.load(open(os.path.join(RES, "benchmark_meta.json"), encoding="utf-8"))
S = pd.read_csv(os.path.join(RES, "summary.csv"))
pipe = pd.read_csv(os.path.join(SC, "scalability_pipeline.csv"))
viz = pd.read_csv(os.path.join(SC, "scalability_viz.csv"))
st, sig = G["eval"]["stages"], G["eval"]["significance"]
f1 = [s["metrics"]["f1"]["mean"] for s in st]
cm = G["final_confusion"]["matrix"]
sens, spec = cm[1][1] / sum(cm[1]), cm[0][0] / sum(cm[0])
sel = G["selected_features"]
n_lat_sel = sum(c.startswith("latent_") for c in sel)
MODELS = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
CONFIGS = ["Raw", "Latent only", "Raw + Latent", "Selected"]


def srow(m, c):
    return S[(S.model == m) & (S.config == c)].iloc[0]


d_rl = {m: srow(m, "Raw + Latent").f1_delta for m in MODELS}
rf_sel_vs_rl = srow("Random Forest", "Selected").f1_mean - srow("Random Forest", "Raw + Latent").f1_mean
nonraw = S[S.config != "Raw"]
k_mean = S[S.config == "Selected"].n_features.mean()


def vz(ds, action, grp=None):
    d = viz[(viz.dataset == ds) & (viz.action == action)]
    if grp:
        d = d[d.group.str.contains(grp, regex=False)]
    return d.iloc[0]


def ps(stage, n):
    d = pipe[(pipe.stage == stage) & (pipe.rows == n)]
    return d.iloc[0] if len(d) else None


def sg(x, nd=4):
    return f"{x:+.{nd}f}".replace("-", "−")


def tf(s):
    if s is None or pd.isna(s):
        return "–"
    return f"{s:.1f} s" if s < 60 else f"{s / 60:.1f} min"


W1, T10 = "WDBC-1M", "Temperature-10.5M"
peak = {d: viz[viz.dataset == d].peak_rss_mb.max() / 1000 for d in (W1, T10)}

# ===================================================================== references
REFS = {
    "zoller": "Zöller M-A, Huber MF. Benchmark and survey of automated machine learning frameworks. "
              "J Artif Intell Res 2021;70:409–72.",
    "alcobaca": "Alcobaça E, de Carvalho ACPLF. A literature review on automated machine learning. "
                "Artif Intell Rev 2025.",
    "feurer": "Feurer M, Eggensperger K, Falkner S, Lindauer M, Hutter F. Auto-sklearn 2.0: hands-free AutoML via "
              "meta-learning. J Mach Learn Res 2022;23(261):1–61.",
    "azevedo": "Azevedo K, Torkar R, Martínez-Fernández S. A multivocal literature review on the benefits "
               "and limitations of AutoML tools. Inf Softw Technol 2024;166.",
    "yuan": "Yuan H, Yu K. Automated machine learning with interpretation: a systematic review. "
            "Wiley Interdiscip Rev Med AI 2024;6(1).",
    "eswa": "Automated data processing and feature engineering for machine learning pipelines: a review. "
            "Expert Syst Appl 2024;237.",
    "pandas": "McKinney W. Data structures for statistical computing in Python. In: Proceedings of the 9th Python "
              "in science conference; 2010. p. 56–61.",
    "sklearn": "Pedregosa F, et al. Scikit-learn: machine learning in Python. J Mach Learn Res 2011;12:2825–30.",
    "tensorflow": "Abadi M, et al. TensorFlow: a system for large-scale machine learning. In: 12th USENIX symposium "
                  "on operating systems design and implementation (OSDI 16); 2016. p. 265–83.",
    "xgboost": "Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD "
               "international conference on knowledge discovery and data mining; 2016. p. 785–94.",
    "lightgbm": "Ke G, et al. LightGBM: a highly efficient gradient boosting decision tree. In: Advances in neural "
                "information processing systems 30; 2017. p. 3146–54.",
    "matplotlib": "Hunter JD. Matplotlib: a 2D graphics environment. Comput Sci Eng 2007;9(3):90–5.",
    "orange": "Demšar J, et al. Orange: data mining toolbox in Python. J Mach Learn Res 2013;14:2349–53.",
    "knime": "Berthold MR, et al. KNIME: the Konstanz Information Miner. In: Data analysis, machine learning and "
             "applications. Springer; 2008. p. 319–26.",
    "weka": "Hall M, Frank E, Holmes G, Pfahringer B, Reutemann P, Witten IH. The WEKA data mining software: an "
            "update. SIGKDD Explor 2009;11(1):10–8.",
    "ydata": "YData. ydata-profiling: data profiling for pandas DataFrames; 2023. [Online]. Available: "
             "https://github.com/ydataai/ydata-profiling.",
    "featuretools": "Kanter JM, Veeramachaneni K. Deep feature synthesis: towards automating data science "
                    "endeavors. In: 2015 IEEE international conference on data science and advanced analytics "
                    "(DSAA); 2015. p. 1–10.",
    "pca": "Jolliffe IT, Cadima J. Principal component analysis: a review and recent developments. Philos Trans R "
           "Soc A 2016;374(2065):20150202.",
    "ica": "Hyvärinen A, Oja E. Independent component analysis: algorithms and applications. Neural Netw "
           "2000;13(4–5):411–30.",
    "ae": "Hinton GE, Salakhutdinov RR. Reducing the dimensionality of data with neural networks. Science "
          "2006;313(5786):504–7.",
    "mi": "Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. Phys Rev E "
          "2004;69(6):066138.",
    "rf": "Breiman L. Random forests. Mach Learn 2001;45(1):5–32.",
    "rfe": "Guyon I, Weston J, Barnhill S, Vapnik V. Gene selection for cancer classification using support vector "
           "machines. Mach Learn 2002;46:389–422.",
    "street": "Street WN, Wolberg WH, Mangasarian OL. Nuclear feature extraction for breast tumor diagnosis. In: "
              "Proc. SPIE 1905, biomedical image processing and biomedical visualization; 1993. p. 861–70.",
    "uci": "Wolberg WH, Mangasarian OL, Street N, Street W. Breast cancer Wisconsin (diagnostic). UCI Machine "
           "Learning Repository; 1995. https://doi.org/10.24432/C5DW2B.",
    "kaufman": "Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation, detection, and "
               "avoidance. ACM Trans Knowl Discov Data 2012;6(4):15.",
    "ambroise": "Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray "
                "gene-expression data. Proc Natl Acad Sci USA 2002;99(10):6562–6.",
    "nadeau": "Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn 2003;52(3):239–81.",
    "temperature": "mattpo. Temperature_IoT_on_GCP. Kaggle; [Online]. Available: "
                   "https://www.kaggle.com/datasets/mattpo/temperature-iot-on-gcp (accessed 12 November 2026).",
}

# ===================================================================== content model
# Inline markup: **bold**, *italic*, {{placeholder}}, [@key,@key] citations, `code`.
B = []


def P(t, **o): B.append(("p", t, o))
def H1(t): B.append(("h1", t))
def H2(t): B.append(("h2", t))
def STEPS(items): B.append(("steps", items))
def STATEMENT(h, t): B.append(("statement", h, t))


FIGN, TABN = [0], [0]


def FIGURE(path, caption, width=1.0):
    FIGN[0] += 1
    B.append(("fig", path, caption, width, FIGN[0]))
    return FIGN[0]


def TABLE(caption, header, rows, widths, align=None, size="small"):
    TABN[0] += 1
    B.append(("table", caption, header, rows, widths, align or (["l"] + ["c"] * (len(header) - 1)), size))
    return TABN[0]


def nextfig(k=1): return FIGN[0] + k
def nexttab(k=1): return TABN[0] + k


TITLE = ("Feature Engineering Assistant: An interactive desktop workbench for transparent feature engineering "
         "and impact evaluation")
AUTHORS = [("Samar Salame", True), ("Aya Kanaan", False), ("Hanin Younis", False)]
AFFIL = ("Master of Data Science Program, Department of Computer Science and Mathematics, Lebanese American "
         "University, Beirut, Lebanon")
EMAILS = [("{{samar.salam@lau.edu}}", "S. Salame"), ("{{aya.kenaan@lau.edu}}", "A. Kanaan"),
          ("hanin.younis@lau.edu", "H. Younis")]
ABSTRACT = (
    "Feature engineering strongly influences the accuracy and interpretability of machine-learning models, yet it is "
    "usually carried out across scattered scripts and tools. Feature Engineering Assistant is an open-source Python "
    "desktop application that integrates dataset profiling, automated cleaning, latent feature generation with "
    "principal and independent component analysis and a deep autoencoder, manual pruning, supervised feature "
    "selection, an explorer that explains latent features, and cross-validated evaluation of every pipeline stage in "
    "one graphical interface, and exports the session as a report. A case study on the Wisconsin Diagnostic Breast "
    "Cancer dataset demonstrates the complete workflow, and a leakage-free benchmark shows why engineered features "
    "must be validated rather than assumed. On a standard laptop, the application loads, profiles and visualizes a "
    "real 10.5-million-row dataset, making it practical for non-expert users who explore large files.")
KEYWORDS = ["Feature engineering", "Latent features", "Feature selection", "Model evaluation", "Data visualization",
            "Desktop application"]
METADATA = [
    ["C1", "Current code version", "v0.1"],
    ["C2", "Permanent link to code/repository used for this code version",
     "{{https://github.com/<user>/feature-engineering-assistant}}"],
    ["C3", "Permanent link to Reproducible Capsule", "{{<repository URL>/paper_assets}}"],
    ["C4", "Permanent software archive (Zenodo DOI)", "{{https://doi.org/10.5281/zenodo.<id>}}"],
    ["C5", "Legal Code License", "MIT License"],
    ["C6", "Code versioning system used", "Git"],
    ["C7", "Software code languages, tools, and services used",
     "Python; Tkinter; pandas; NumPy; SciPy; scikit-learn; TensorFlow/Keras; XGBoost; LightGBM; matplotlib; "
     "seaborn; python-docx; ydata-profiling; skimpy"],
    ["C8", "Compilation requirements, operating environments and dependencies",
     "Python 3.10 (recommended) and 3.13 (tested); Windows 10/11; dependencies in requirements.txt"],
    ["C9", "If available, link to developer documentation/manual", "{{README: <repository URL>}}"],
    ["C10", "Support email for questions", "{{samar.salam@lau.edu}}"],
]
TABN[0] = 1  # Table 1 = code metadata

# ---------------------------------------------------------------- 1
H1("Motivation and significance")
P("Feature engineering—turning raw variables into informative model inputs—often determines predictive "
  "performance more than the choice of algorithm, yet in practice it is fragmented: data are profiled in one tool, "
  "cleaned in scripts, compressed with dimensionality-reduction methods in notebooks and evaluated elsewhere. "
  "Automated machine learning (AutoML) frameworks reduce this effort [@zoller,@alcobaca,@feurer], but they treat "
  "preprocessing and feature construction largely as a black box, which limits interpretability and user trust "
  "[@azevedo,@yuan,@eswa].")
P("Feature Engineering Assistant takes a complementary approach. Rather than automating decisions, it makes them "
  "explicit: every transformation is visible, intermediate datasets are kept as snapshots, and the effect of each "
  "pipeline stage on predictive performance is measured by cross-validation before the user commits to it. The tool "
  "serves two groups: data scientists who need fine-grained control and interpretable features, and non-expert "
  "analysts who need visual and statistical summaries without writing code.")
P("To our knowledge, Feature Engineering Assistant is the first open-source desktop application that combines data "
  "profiling, cleaning, linear and nonlinear latent feature generation, an explorer that relates latent features to "
  "the original variables, supervised feature selection and stage-wise impact evaluation with significance testing "
  "in a single graphical interface. This paper describes its architecture and functionality, its installation, and "
  "three illustrative examples covering a complete workflow, a leakage-free validation and large datasets.")

# ---------------------------------------------------------------- 2
H1("Software description")
H2("Software architecture")
P(f"The application is written in Python with a Tkinter graphical user interface (`Main.py`). A sidebar exposes the "
  f"pipeline actions in their natural order—load, clean, create features, prune, select, evaluate and "
  f"export—and tabs host the dataset preview, the Latent Features Explorer, profiling reports, control "
  f"settings, the results panel and dataset export (Fig. {nextfig()}). All modules communicate through a shared "
  f"dataset state (`dataset.py`) that holds the working data frame, snapshots of the data after loading and after "
  f"feature creation, the list of pruned (banned) columns and the task type. Each module reads the current data and "
  f"writes its result back, so changes propagate immediately, while the snapshots let the evaluator rebuild the "
  f"dataset of every pipeline stage. Long computations (feature creation, impact evaluation and automatic choice of "
  f"the number of features) run in background threads with progress indicators, and evaluation results are cached "
  f"until the state changes. The back end relies on pandas [@pandas], scikit-learn [@sklearn], TensorFlow/Keras "
  f"[@tensorflow], XGBoost [@xgboost], LightGBM [@lightgbm] and matplotlib [@matplotlib]; reports are written with "
  f"python-docx.")
FIGURE(os.path.join(FIG, "fig_architecture.png"),
       "Architecture of Feature Engineering Assistant. Pipeline modules read from and write to a shared dataset "
       "state; its snapshots are used to rebuild and re-evaluate each pipeline stage.")

H2("Related software")
P(f"General-purpose visual data-mining environments such as Orange [@orange], KNIME [@knime] and Weka [@weka] "
  f"provide graphical preprocessing, principal component analysis and model evaluation, but a feature-engineering "
  f"workflow must be assembled from many separate components, and none combines autoencoder features with an "
  f"explanation view and stage-by-stage significance testing. ydata-profiling [@ydata] produces rich exploratory "
  f"reports but performs no transformation or modeling; Featuretools [@featuretools] synthesizes features from "
  f"relational data through code; and AutoML systems such as auto-sklearn [@feurer] search preprocessing pipelines "
  f"automatically but expose them only as code-level configurations. Feature Engineering Assistant occupies the space "
  f"between these tools: an interactive application dedicated to feature engineering in which every step is visible "
  f"and its effect is measured (Table {nexttab()}).")
Y, N, Pa = "✓", "×", "∼"
TABLE("Comparison of Feature Engineering Assistant with related software. ✓ = supported, × = not "
      "supported, ∼ = partial support.",
      ["Software", "Desktop GUI", "Open source", "Profiling & plots", "Cleaning", "Latent features (PCA/ICA/AE)",
       "Supervised selection", "Stage-wise impact test", "Report export"],
      [["Orange [@orange]", Y, Y, Y, Y, Pa, Y, Pa, Y],
       ["KNIME [@knime]", Y, Y, Y, Y, Pa, Y, Pa, Pa],
       ["Weka [@weka]", Y, Y, Pa, Y, Pa, Y, Pa, N],
       ["ydata-profiling [@ydata]", N, Y, Y, N, N, N, N, Y],
       ["Featuretools [@featuretools]", N, Y, N, N, N, Pa, N, N],
       ["auto-sklearn [@feurer]", N, Y, N, Pa, Pa, Y, N, N],
       ["**Feature Engineering Assistant**", Y, Y, Y, Y, Y, Y, Y, Y]],
      [3.4, 1.4, 1.4, 1.6, 1.4, 2.0, 1.6, 1.6, 1.4], size="footnotesize")

H2("Software functionalities")
P("**Data loading and profiling.** CSV, Excel and Parquet files are loaded and common missing-value tokens are "
  "normalized. The preview reports rows, columns and duplicates and, for each column, its type, inferred role "
  "(numeric, low-cardinality, categorical, datetime, boolean or ID-like), missing percentage and number of unique "
  "values. Double-clicking a column opens a profile with six plots for numeric data (histogram, KDE, box, violin, "
  "scatter, line) or for categorical data (bar, horizontal bar, pie, table, top-N, word cloud); ydata-profiling and "
  "skimpy reports can also be generated.")
P("**Data cleaning.** One action removes empty, constant, mostly missing (>60%) and ID-like columns (an ID-like name "
  "and >90% unique values), drops duplicate rows, imputes remaining gaps with the median or mode and logs every "
  "change.")
P("**Feature creation.** Latent features are generated with principal component analysis (PCA) [@pca], independent "
  "component analysis (ICA) [@ica] and a deep tabular autoencoder [@ae] in three presets: *Fast* (four PCA "
  "components), *Balanced* (five PCA and five ICA components) and *Deep* (five PCA, five ICA and six autoencoder "
  "components on min–max-scaled inputs). The autoencoder has a 128–64–d encoder with a mirrored "
  "decoder, embeddings for categorical columns and early stopping. The target column is excluded, and generated "
  "columns carry the prefix `latent_`.")
P("**Pruning and supervised selection.** Columns can be removed manually; they stay banned until reset to the "
  "post-creation snapshot. Supervised selection averages three rankings—mutual information [@mi], model "
  "importance (Random Forest [@rf], XGBoost, LightGBM or linear coefficients) and recursive feature elimination "
  "[@rfe]—and chooses the number of retained features from {5, 10, 15, 20, 25} by five-fold cross-validation.")
P("**Latent Features Explorer.** Distributions, a latent correlation heatmap, the original features most correlated "
  "with a chosen latent feature and Random Forest importance of latent versus original features help users judge "
  "what each latent dimension represents.")
P("**Impact evaluation and reporting.** The evaluator rebuilds the baseline, pruned, post-creation and selected "
  "datasets and scores each with the same model under k-fold (stratified) cross-validation, reporting accuracy, "
  "F1-score, precision, recall and ROC-AUC, or RMSE, MAE and R² for regression, together with the change "
  "between stages, a paired t-test and Cohen’s d. Confusion matrices and ROC and precision–recall curves "
  "are available, and the session can be exported as a DOCX report and the current dataset as CSV.")

# ---------------------------------------------------------------- 3
H1("Availability and installation")
P("Feature Engineering Assistant is open source and available at {{<repository URL>}} under the MIT "
  "license. It requires Python 3.10 (recommended in the installation guide) and was also tested with Python 3.13 on "
  "Windows 11; no GPU is required. After creating a virtual environment, dependencies are installed with "
  "`pip install -r requirements.txt` and the application is started with `python Main.py`. The repository also "
  "contains the scripts that reproduce every result in this paper (`paper_assets`): an automated GUI session with "
  "fixed random seeds, the leakage-free benchmark, the scalability tests with a manual test protocol and dataset "
  "checksums, and the figure scripts.")

# ---------------------------------------------------------------- 4
H1("Illustrative examples")
H2("Case study 1: End-to-end workflow on breast cancer diagnosis data")
f_clean, f_expl, f_imp = nextfig(1), nextfig(2), nextfig(3)
P("We applied the application to the Wisconsin Diagnostic Breast Cancer (WDBC) dataset [@street,@uci]: 569 "
  "fine-needle-aspirate samples (357 benign, 212 malignant) described by 30 nuclear features. The workflow, run with "
  "a fixed random seed of 42, proceeded as follows:")
STEPS([
    ("Loading and profiling", "The CSV file (569 rows, 33 columns) loaded without duplicates. Profiles showed "
     "moderate class imbalance and right-skewed size features."),
    ("Cleaning", f"The empty trailing column and the ID-like `id` column were removed, leaving 30 features and the "
     f"target (Fig. {f_clean})."),
    ("Feature creation", f"Deep mode added 16 latent features in about "
     f"{max(1, round(G['feature_creation_seconds']))} s. The explorer (Fig. {f_expl}) showed that `latent_pca_1` "
     f"and `latent_auto_1` summarize nuclear concavity and size (|r| up to 0.96 with the concave-points features) and "
     f"that `latent_pca_1` is the most important feature for the diagnosis."),
    ("Feature selection", f"With automatic K, {len(sel)} of the 46 candidates were retained, {n_lat_sel} of them "
     f"latent."),
    ("Impact evaluation", f"Under five-fold stratified cross-validation with a Random Forest, the weighted "
     f"F1-score rose from {f1[0]:.3f} (raw file) to {f1[1]:.3f} with latent features "
     f"(p = {sig[0]['p_value']:.2f}) and {f1[2]:.3f} after selection (p = {sig[1]['p_value']:.2f}); the final model "
     f"reached a sensitivity of {sens * 100:.1f}% and a specificity of {spec * 100:.1f}% "
     f"(Table {nexttab()}, Fig. {f_imp})."),
])
FIGURE(os.path.join(FIG, "03_cleaning.png"),
       "Main window after loading and cleaning the WDBC dataset: summary cards, column table with inferred roles, "
       "data preview and the cleaning log.")
FIGURE(os.path.join(FIG, "fig_latent_explorer.png"),
       "Latent Features Explorer: (a) original features most correlated with `latent_auto_1`; (b) Random Forest "
       "importance of original (blue) and latent (orange) features for the diagnosis.", width=0.75)
t3 = []
names = ["Baseline (raw file)", "+ Latent features", "+ Feature selection"]
for i, s_ in enumerate(st):
    m = s_["metrics"]
    t3.append([names[i], str(s_["n_features"]), f"{m['accuracy']['mean']:.3f}", f"{m['f1']['mean']:.3f}",
               f"{m['roc_auc']['mean']:.3f}", "–" if i == 0 else f"{sig[i - 1]['p_value']:.2f}"])
TABLE("Case study 1 – impact evaluation on WDBC (Random Forest, five-fold stratified cross-validation, "
      "means; p: paired t-test versus the previous stage).",
      ["Stage", "Features", "Accuracy", "F1-score", "ROC-AUC", "p"], t3, [4.4, 1.8, 2.0, 2.0, 2.0, 1.4])
FIGURE(os.path.join(FIG, "09_evaluate_impact.png"),
       "Results panel after Evaluate Impact: per-stage metrics, changes between stages and paired t-tests with "
       "Cohen’s d.")

H2("Case study 2: Validating engineered features without leakage")
P(f"The built-in evaluation fits the latent generators and the feature ranking on all rows before cross-validation, "
  f"which can make its estimates optimistic [@kaufman,@ambroise]. To show how the tool supports rigorous validation, "
  f"we re-ran the same pipeline under 10-fold stratified cross-validation repeated three times, refitting scaling, "
  f"PCA, ICA, the autoencoder and the application’s own selection routine inside each training fold, for four "
  f"classifiers and four feature sets, and tested differences with the corrected resampled t-test [@nadeau]. Adding "
  f"latent features increased the F1-score of the tree ensembles (Random Forest {sg(d_rl['Random Forest'])}, "
  f"XGBoost {sg(d_rl['XGBoost'])}, LightGBM {sg(d_rl['LightGBM'])}) and slightly decreased that of logistic "
  f"regression ({sg(d_rl['Logistic Regression'])}); the selected subsets used {k_mean:.1f} features on average "
  f"with unchanged accuracy. No difference was significant (all p ≥ {nonraw.f1_p.min():.2f}; "
  f"Table {nexttab()}, Fig. {nextfig()}). Selection in the interactive session raised F1 by {sg(f1[2] - f1[1])}, "
  f"whereas refitting it inside each fold lowered it by {abs(rf_sel_vs_rl):.4f}—the selection bias described "
  f"by Ambroise and McLachlan [@ambroise]. The tool therefore makes such effects visible instead of reporting gains "
  f"that are not real.")
t4 = []
for m in MODELS:
    for j, c in enumerate(CONFIGS):
        r = srow(m, c)
        t4.append([m if j == 0 else "", c, f"{r.n_features:.0f}" if c != "Selected" else f"{r.n_features:.1f}",
                   f"{r.f1_mean:.3f} ± {r.f1_ci:.3f}", f"{r.roc_auc_mean:.3f}",
                   "–" if c == "Raw" else f"{r.f1_p:.2f}"])
TABLE("Case study 2 – leakage-free benchmark on WDBC (means over 30 outer folds; ±: corrected 95% "
      "confidence interval; p: corrected resampled t-test on F1 versus Raw).",
      ["Model", "Feature set", "Features", "F1-score", "ROC-AUC", "p"], t4, [3.4, 2.6, 1.7, 2.8, 1.9, 1.4])
FIGURE(os.path.join(FIG, "fig_benchmark.png"),
       "Leakage-free benchmark: weighted F1-score and ROC-AUC with corrected 95% confidence intervals for each "
       "classifier and feature set.")

H2("Case study 3: Exploring datasets with millions of rows")
sel100 = ps("Feature selection", 100000)
P(f"Non-expert users often receive large exported files. We loaded the real Temperature_IoT_on_GCP dataset "
  f"[@temperature] (10,553,597 sensor readings, five columns, 580 MB) through the normal Load Dataset button on a "
  f"laptop with an Intel Core i7-8750H processor and 16 GB of memory, and repeated the test with a one-million-row "
  f"synthetic enlargement of WDBC. Every profiling plot, the cleaning step, Fast-mode feature creation and every "
  f"Latent-Explorer plot completed on both datasets (Table {nexttab()}, Fig. {nextfig()}); on 10.55 million rows "
  f"each plot took between 1 s and 1 min and peak memory was {peak[T10]:.1f} GB. Supervised modeling scales less "
  f"well: feature selection took {sel100.seconds / 60:.0f} min at 100,000 rows. For millions of rows we therefore "
  f"recommend the application for exploration, visualization and cleaning on standard hardware, and running feature "
  f"creation and selection on more powerful machines, on a sample or in chunks. The datasets, checksums and a "
  f"step-by-step manual test are provided in the repository.")
lab5 = [("Load CSV + metadata + preview", None, "Load and preview"),
        ("Open pop-up (Histogram + KDE)", "numeric", "Numeric profile: histogram + KDE"),
        ("Violin", "numeric", "Numeric profile: violin"),
        ("Open pop-up (Bar)", "categorical", "Categorical profile: bar"),
        ("Data cleaning", None, "Data cleaning"),
        ("Fast mode: 4 PCA components (background thread)", None, "Fast feature creation (PCA)†"),
        ("Scatter vs index", "Latent", "Latent explorer: scatter"),
        ("Export current dataset to CSV", None, "Export to CSV")]
t5 = [[lbl, tf(vz(W1, a, g).seconds), tf(vz(T10, a, g).seconds)] for a, g, lbl in lab5]
t5.append(["Peak memory", f"{peak[W1]:.1f} GB", f"{peak[T10]:.1f} GB"])
TABLE("Case study 3 – time per action on large datasets (WDBC-1M: 1,000,000 × 33, synthetic; "
      "Temperature: 10,553,597 × 5, real). † Runs in a background thread; the window stays responsive.",
      ["Action", "WDBC-1M", "Temperature (10.55M)"], t5, [7.0, 3.2, 4.0])
FIGURE(os.path.join(FIG, "fig_scalability_screens.png"),
       "The application holding the 10,553,597-row temperature dataset: (a) preview after loading; (b) violin plot "
       "of `temp_c` over all rows; (c) first PCA feature plotted against the row index.", width=0.85)

# ---------------------------------------------------------------- 5
H1("Impact")
P("Feature Engineering Assistant lowers the barrier to systematic feature engineering. Analysts without programming "
  "experience can profile and clean data, generate and interpret latent features, select features and verify their "
  "effect in one environment, and the exported report documents every decision. Because the application runs "
  "offline, sensitive data such as clinical records never leave the user’s machine.")
P("The software also supports research questions that are awkward to study with scattered scripts: when latent "
  "representations help a given family of models (Case study 2 shows they helped tree ensembles but not logistic "
  "regression), how strongly in-sample feature selection inflates performance estimates, and how stable selected "
  "feature sets are across resamples. Stage-wise evaluation with significance tests encourages users to treat "
  "feature engineering as a hypothesis to be tested rather than a guaranteed improvement.")
P("In education, the tool makes abstract concepts—variance captured by principal components, independent "
  "sources, nonlinear embeddings, information leakage—visible and interactive. It complements AutoML frameworks "
  "by giving users an interpretable view of the data and features before automated modeling is applied.")
P("**Adoption and reach.** The software was developed within the Master of Data Science program at the Lebanese "
  "American University and is released as open source for teaching and research. {{Usage in courses or projects, "
  "if any, can be reported here.}}")

# ---------------------------------------------------------------- 6
H1("Conclusions")
P("This paper presented Feature Engineering Assistant, a desktop workbench that unifies profiling, cleaning, latent "
  "feature generation, explanation, selection, stage-wise evaluation and reporting. The case studies showed a "
  "complete and reproducible workflow, demonstrated that engineered features must be validated without leakage, and "
  "showed that loading and visualization remain practical with more than ten million rows on standard hardware.")
P("The current version has limitations. Its built-in evaluation fits feature generators and selection on the full "
  "dataset before cross-validation; supervised selection runs on the interface thread and becomes impractical beyond "
  "about 100,000 rows; autoencoder and ICA features are stochastic unless seeds are fixed; and time-series features "
  "are not supported. Future work will add nested, leakage-free evaluation, sampling- or chunk-based selection in "
  "background threads, explanation methods such as SHAP, and support for temporal data.")

# ---------------------------------------------------------------- statements
STATEMENT("CRediT authorship contribution statement",
          "{{Samar Salame: Conceptualization, Software, Writing – original draft. Aya Kanaan: Software, "
          "Validation. Hanin Younis: Software, Visualization, Writing – review & editing. (Roles to be confirmed "
          "by the authors.)}}")
STATEMENT("Declaration of competing interest",
          "The authors declare that they have no known competing financial interests or personal relationships that "
          "could have appeared to influence the work reported in this paper.")
STATEMENT("Declaration of generative AI and AI-assisted technologies in the writing process",
          "During the preparation of this work the authors used ChatGPT (OpenAI) for language editing and Claude "
          "(Anthropic) to review the source code and correct two defects, to generate the evaluation and figure "
          "scripts in the reproducibility capsule, and to draft parts of the text. The application itself was "
          "developed by the authors. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for "
          "the content of the published article.")
STATEMENT("Data availability",
          "The WDBC dataset is available from the UCI Machine Learning Repository [@uci] and the temperature dataset "
          "from Kaggle [@temperature]. The scripts, results and synthetic datasets used in this paper are available "
          "in the software repository.")
STATEMENT("Acknowledgements", "{{To be added by the authors (e.g., course instructor, funding).}}")

# ===================================================================== citation order
ORDER = []
CITE = re.compile(r"\[(@[^\]]+)\]")


def _collect(text):
    for m in CITE.finditer(text):
        for k in m.group(1).split(","):
            k = k.strip().lstrip("@")
            if k not in REFS:
                raise KeyError(k)
            if k not in ORDER:
                ORDER.append(k)


for blk in B:
    kind = blk[0]
    if kind == "p":
        _collect(blk[1])
    elif kind == "steps":
        for t_, x in blk[1]:
            _collect(x)
    elif kind == "fig":
        _collect(blk[2])
    elif kind == "table":
        _collect(blk[1])
        for row in blk[3]:
            for c in row:
                _collect(str(c))
    elif kind == "statement":
        _collect(blk[2])


def num_cites(text):
    def rep(m):
        nums = sorted(ORDER.index(k.strip().lstrip("@")) + 1 for k in m.group(1).split(","))
        if len(nums) > 2 and nums == list(range(nums[0], nums[-1] + 1)):
            return f"[{nums[0]}–{nums[-1]}]"
        return "[" + ", ".join(map(str, nums)) + "]"
    return CITE.sub(rep, text)


# ===================================================================== word count
def _plain(t):
    return re.sub(r"[*{}`]|\[@[^\]]+\]", "", t)


WC = len(_plain(ABSTRACT).split())
for blk in B:
    if blk[0] == "p":
        WC += len(_plain(blk[1]).split())
    elif blk[0] == "steps":
        WC += sum(len(_plain(a + " " + b).split()) for a, b in blk[1])
    elif blk[0] == "fig":
        WC += len(_plain(blk[2]).split())
    elif blk[0] == "table":
        WC += len(_plain(blk[1]).split())

# ===================================================================== LaTeX renderer
UNI = {
    "✓": r"\checkmark{}", "×": r"$\times$", "∼": r"$\sim$", "≥": r"$\geq$", "≤": r"$\leq$",
    "−": r"$-$", "±": r"$\pm$", "→": r"$\rightarrow$", "·": r"$\cdot$", "†": r"\textdagger{}",
    "‡": r"\textdaggerdbl{}", "°": r"\textdegree{}", "–": "--", "—": "---", "’": "'",
    "‘": "`", "“": "``", "”": "''", "²": r"\textsuperscript{2}", "…": r"\ldots{}",
}
TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
               "<": r"\textless{}", ">": r"\textgreater{}"}
URL = re.compile(r"https?://[^\s,;)]+[^\s,;).]")


def _esc_plain(s):
    out = []
    pos = 0
    for m in URL.finditer(s):
        out.append(_esc_chars(s[pos:m.start()]))
        out.append(r"\url{" + m.group(0) + "}")
        pos = m.end()
    out.append(_esc_chars(s[pos:]))
    return "".join(out)


def _esc_chars(s):
    return "".join(UNI.get(ch, TEX_SPECIAL.get(ch, ch)) for ch in s)


def _cite_tex(m):
    return r"~\cite{" + ",".join(k.strip().lstrip("@") for k in m.group(1).split(",")) + "}"


INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\{\{.+?\}\}|`.+?`|\[@[^\]]+\])")


def tex(s):
    out = []
    for part in INLINE.split(s):
        if not part:
            continue
        if part.startswith("**"):
            out.append(r"\textbf{" + tex(part[2:-2]) + "}")
        elif part.startswith("{{"):
            inner = part[2:-2]
            out.append(r"\ph{" + (_esc_chars(inner) if "<" in inner else tex(inner)) + "}")
        elif part.startswith("[@"):
            out.append(_cite_tex(re.match(r"\[(@[^\]]+)\]", part)))
        elif part.startswith("`"):
            out.append(r"\texttt{" + _esc_chars(part[1:-1]) + "}")
        elif part.startswith("*"):
            out.append(r"\emph{" + tex(part[1:-1]) + "}")
        else:
            out.append(_esc_plain(part))
    txt = "".join(out)
    return txt.replace(" ~\\cite", "~\\cite")


def render_latex():
    if os.path.isdir(OUT_TEX_DIR):
        shutil.rmtree(OUT_TEX_DIR)
    os.makedirs(os.path.join(OUT_TEX_DIR, "figures"))
    L = []
    a = L.append
    a(r"\documentclass[preprint,12pt]{elsarticle}")
    a(r"% SoftwareX Original Software Publication -- compile with pdfLaTeX on Overleaf.")
    a(r"\usepackage[utf8]{inputenc}")
    a(r"\usepackage[T1]{fontenc}")
    a(r"\usepackage{amssymb,amsmath}")
    a(r"\usepackage{graphicx}")
    a(r"\usepackage{booktabs,tabularx,array}")
    a(r"\usepackage{textcomp}")
    a(r"\usepackage[table]{xcolor}")
    a(r"\usepackage{url}")
    a(r"\usepackage[hidelinks]{hyperref}")
    a(r"\usepackage{enumitem}")
    a(r"\newcolumntype{Y}{>{\raggedright\arraybackslash}X}")
    a(r"\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}")
    a(r"\newcommand{\ph}[1]{\textcolor{red}{#1}}  % placeholder: replace before submission")
    a(r"\journal{SoftwareX}")
    a(r"\begin{document}")
    a(r"\begin{frontmatter}")
    a(r"\title{" + tex(TITLE) + "}")
    for (name, corr), (email, _) in zip(AUTHORS, EMAILS):
        a(r"\author[lau]{" + tex(name) + (r"\corref{cor1}" if corr else "") + "}")
        a(r"\ead{" + email.replace("{{", "").replace("}}", "") + "}")
    a(r"\cortext[cor1]{Corresponding author. \ph{Corresponding author and e-mail addresses to be confirmed "
      r"(the original draft spells them samar.salam@lau.edu / samar.salame@lau.ed and aya.kenaan@lau.edu).}}")
    a(r"\affiliation[lau]{organization={" + tex(AFFIL) + "}}")
    a(r"\begin{abstract}")
    a(tex(ABSTRACT))
    a(r"\end{abstract}")
    a(r"\begin{keyword}")
    a(r" \sep ".join(tex(k) for k in KEYWORDS))
    a(r"\end{keyword}")
    a(r"\end{frontmatter}")
    a("")
    a(r"\section*{Metadata}")
    a(r"The metadata can be found in Table~\ref{tab:metadata}.")
    a(r"\begin{table}[!h]")
    a(r"\caption{Code metadata.}\label{tab:metadata}")
    a(r"\footnotesize")
    a(r"\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}p{5.4cm}Y@{}}")
    a(r"\toprule")
    a(r"\textbf{Nr.} & \textbf{Code metadata description} & \textbf{Metadata} \\")
    a(r"\midrule")
    for row in METADATA:
        a(" & ".join(tex(c) for c in row) + r" \\")
    a(r"\bottomrule")
    a(r"\end{tabularx}")
    a(r"\end{table}")
    a("")
    for blk in B:
        k = blk[0]
        if k == "h1":
            a(r"\section{" + tex(blk[1]) + "}")
        elif k == "h2":
            if blk[1].startswith("Case study"):
                a(r"\subsection*{" + tex(blk[1]) + "}")
            else:
                a(r"\subsection{" + tex(blk[1]) + "}")
        elif k == "p":
            a(tex(blk[1]))
            a("")
        elif k == "steps":
            a(r"\begin{enumerate}[leftmargin=*]")
            for t_, x in blk[1]:
                a(r"\item \textbf{" + tex(t_) + r":} " + tex(x))
            a(r"\end{enumerate}")
        elif k == "fig":
            path, cap, width, n = blk[1:]
            name = os.path.basename(path)
            shutil.copy(path, os.path.join(OUT_TEX_DIR, "figures", name))
            a(r"\begin{figure}[!htbp]")
            a(r"\centering")
            a(r"\includegraphics[width=" + f"{width:.2f}" + r"\textwidth]{figures/" + name + "}")
            a(r"\caption{" + tex(cap) + "}")
            a(r"\label{fig:" + str(n) + "}")
            a(r"\end{figure}")
            a("")
        elif k == "table":
            cap, header, rows, widths, align, size = blk[1:]
            tw = sum(widths)
            cols = []
            for w, al in zip(widths, align):
                frac = w / tw
                spec_ = r">{\raggedright\arraybackslash}" if al == "l" else r">{\centering\arraybackslash}"
                cols.append(spec_ + r"p{\dimexpr" + f"{frac:.3f}" + r"\textwidth-2\tabcolsep\relax}")
            a(r"\begin{table}[!htbp]")
            a(r"\caption{" + tex(cap) + "}")
            a("\\" + size)
            a(r"\begin{tabular}{@{}" + "".join(cols) + "@{}}")
            a(r"\toprule")
            a(" & ".join(r"\textbf{" + tex(h) + "}" for h in header) + r" \\")
            a(r"\midrule")
            for row in rows:
                a(" & ".join(tex(str(c)) for c in row) + r" \\")
            a(r"\bottomrule")
            a(r"\end{tabular}")
            a(r"\end{table}")
            a("")
        elif k == "statement":
            a(r"\section*{" + tex(blk[1]) + "}")
            a(tex(blk[2]))
            a("")
    a(r"\begin{thebibliography}{" + str(len(ORDER)) + "}")
    for key in ORDER:
        a(r"\bibitem{" + key + "} " + tex(REFS[key]))
    a(r"\end{thebibliography}")
    a(r"\end{document}")
    with open(os.path.join(OUT_TEX_DIR, "main.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(OUT_TEX_DIR, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("Overleaf: New Project > Upload Project > select SoftwareX_Overleaf.zip.\n"
                 "Compiler: pdfLaTeX (Menu > Settings > Compiler). Main document: main.tex.\n"
                 "Red text (\\ph{...}) marks placeholders to replace before submission.\n")
    if os.path.exists(OUT_ZIP):
        os.remove(OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT_TEX_DIR):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, OUT_TEX_DIR).replace(os.sep, "/"))


# ===================================================================== Word renderer
def render_docx():
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.2)
    text_w = 21.0 - 4.4
    base = doc.styles["Normal"]
    base.font.name = "Times New Roman"
    base.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    base.font.size = Pt(10.5)
    base.paragraph_format.space_after = Pt(4)
    base.paragraph_format.line_spacing = 1.15
    for name, size in (("Heading 1", 12), ("Heading 2", 10.5)):
        s = doc.styles[name]
        s.font.name = "Times New Roman"
        s.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        s.font.size = Pt(size); s.font.bold = True; s.font.italic = name == "Heading 2"
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.paragraph_format.space_before = Pt(10 if name == "Heading 1" else 6)
        s.paragraph_format.space_after = Pt(4)
        s.paragraph_format.keep_with_next = True
    J, Cn, Lf = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT

    def runs(p, text, size=None):
        text = num_cites(text)
        for part in INLINE.split(text):
            if not part:
                continue
            if part.startswith("{{"):
                r = p.add_run(part[2:-2]); r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            elif part.startswith("**"):
                r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith("`"):
                r = p.add_run(part[1:-1]); r.font.name = "Consolas"
            elif part.startswith("*"):
                r = p.add_run(part[1:-1]); r.italic = True
            else:
                r = p.add_run(part)
            if size:
                r.font.size = Pt(size)
        return p

    def para(text, align=J, size=None):
        p = doc.add_paragraph(); p.alignment = align
        return runs(p, text, size)

    def borders(cell, **kw):
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders"); tcPr.append(b)
        for edge in ("top", "left", "bottom", "right"):
            el = OxmlElement(f"w:{edge}")
            if edge in kw:
                el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(kw[edge])); el.set(qn("w:color"), "000000")
            else:
                el.set(qn("w:val"), "nil")
            b.append(el)

    def table(n, caption, header, rows, widths, align, size_pt):
        c = doc.add_paragraph(); c.paragraph_format.keep_with_next = True
        r = c.add_run(f"Table {n}. "); r.bold = True; r.font.size = Pt(9)
        runs(c, caption, 9)
        t = doc.add_table(rows=len(rows) + 1, cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
        tw = sum(widths)
        for i, row in enumerate([header] + rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Cm(text_w * widths[j] / tw)
                p = cell.paragraphs[0]
                p.alignment = Lf if align[j] == "l" else Cn
                p.paragraph_format.space_after = Pt(1); p.paragraph_format.line_spacing = 1.0
                if i == 0:
                    rr = p.add_run(str(val)); rr.bold = True; rr.font.size = Pt(size_pt)
                else:
                    runs(p, str(val), size_pt)
                kw = {"top": 8, "bottom": 4} if i == 0 else {}
                if i == len(rows):
                    kw["bottom"] = 8
                borders(cell, **kw)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)

    t = doc.add_paragraph(); t.alignment = Cn
    r = t.add_run(TITLE); r.bold = True; r.font.size = Pt(15)
    para(", ".join(n + ("{{*}}" if c else "") for n, c in AUTHORS), Cn)
    para("*" + AFFIL + "*", Cn, 9.5)
    para("{{* Corresponding author – to be confirmed.}} E-mail addresses: "
         + ", ".join(f"{e} ({n})" for e, n in EMAILS) + ".", Cn, 9)
    para("**Abstract**", Lf)
    para(ABSTRACT)
    para("**Keywords:** " + "; ".join(KEYWORDS), Lf)
    para("**Metadata**", Lf)
    para("The metadata can be found in Table 1.")
    table(1, "Code metadata.", ["Nr.", "Code metadata description", "Metadata"], METADATA,
          [1.2, 6.6, 8.8], ["l", "l", "l"], 8.5)
    sec_no, sub_no = 0, 0
    for blk in B:
        k = blk[0]
        if k == "h1":
            sec_no += 1; sub_no = 0
            doc.add_heading(f"{sec_no}. {blk[1]}", level=1)
        elif k == "h2":
            if blk[1].startswith("Case study"):
                doc.add_heading(blk[1], level=2)
            else:
                sub_no += 1
                doc.add_heading(f"{sec_no}.{sub_no}. {blk[1]}", level=2)
        elif k == "p":
            para(blk[1])
        elif k == "steps":
            for i, (a_, b_) in enumerate(blk[1], 1):
                p = doc.add_paragraph(); p.alignment = J
                p.paragraph_format.left_indent = Cm(0.6); p.paragraph_format.first_line_indent = Cm(-0.6)
                runs(p, f"{i}. **{a_}:** {b_}")
        elif k == "fig":
            path, cap, width, n = blk[1:]
            p = doc.add_paragraph(); p.alignment = Cn; p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(path, width=Cm(text_w * width))
            c = doc.add_paragraph(); c.alignment = J
            rr = c.add_run(f"Fig. {n}. "); rr.bold = True; rr.font.size = Pt(9)
            runs(c, cap, 9)
        elif k == "table":
            cap, header, rows, widths, align, size = blk[1:]
            n = 1 + sum(1 for b in B[:B.index(blk) + 1] if b[0] == "table")
            table(n, cap, header, rows, widths, align, 8 if size == "footnotesize" else 8.5)
        elif k == "statement":
            doc.add_heading(blk[1], level=1)
            para(blk[2])
    doc.add_heading("References", level=1)
    for i, key in enumerate(ORDER, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.7); p.paragraph_format.first_line_indent = Cm(-0.7)
        p.paragraph_format.space_after = Pt(2)
        rr = p.add_run(f"[{i}] {REFS[key]}"); rr.font.size = Pt(9)
    doc.save(OUT_DOCX)


render_latex()
render_docx()
unused = [k for k in REFS if k not in ORDER]
print("saved", OUT_TEX_DIR, "and", OUT_ZIP)
print("saved", OUT_DOCX)
print(f"words (abstract + body + captions + table captions): {WC}")
print(f"figures {FIGN[0]}, tables {TABN[0]}, references {len(ORDER)}; unused refs: {unused}")
