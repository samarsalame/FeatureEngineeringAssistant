"""
Drive the Feature Engineering Assistant GUI end-to-end on data.csv and capture
screenshots of every stage for the paper.

Usage (from the project root):
    python paper_assets/capture_gui.py

Outputs:
    paper_assets/screenshots/*.png
    paper_assets/gui_results.json      (numbers shown in the GUI)
    paper_assets/pipeline_report.docx  (report produced by "Export Report")
"""
import ctypes
import json
import os
import sys
import time
import traceback

# Real pixel coordinates on HiDPI screens (must happen before Tk starts)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
APP_DIR = os.path.join(PROJECT, "FinalFeatureEng")
DATA = os.path.join(PROJECT, "data.csv")
SHOTS = os.path.join(HERE, "screenshots")
os.makedirs(SHOTS, exist_ok=True)
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageGrab

# ---- non-blocking dialogs -------------------------------------------------
DIALOG_LOG = []
def _log(kind):
    def f(title="", message="", **kw):
        DIALOG_LOG.append((kind, title, message))
        print(f"[{kind}] {title}: {message}")
        return True
    return f
for name in ("showinfo", "showwarning", "showerror"):
    setattr(messagebox, name, _log(name))
filedialog.askopenfilename = lambda **kw: DATA
filedialog.asksaveasfilename = lambda **kw: os.path.join(HERE, "pipeline_report.docx")

# Render the UI at 100 % scale (96 dpi) inside the DPI-aware process: crisp pixels,
# and matplotlib's Tk backend then sizes figures to their widgets (at 125 % it
# oversizes them by the device-pixel ratio and the plots get clipped).
_orig_tk_init = tk.Tk.__init__
def _tk_init(self, *a, **k):
    _orig_tk_init(self, *a, **k)
    self.tk.call("tk", "scaling", 96 / 72)
tk.Tk.__init__ = _tk_init

import Main  # builds the UI (root created at import)
import tensorflow as tf
tf.keras.utils.set_random_seed(42)  # seeds python, numpy and TF -> reproducible GUI run
import dataset
import evaluate_impact

root = Main.root
root.geometry("1440x860+0+0")
PROFILE_GEOMETRY = "920x800+300+0"

TARGET = "diagnosis"
RESULTS = {}


# ---- helpers ------------------------------------------------------------
class RECT(ctypes.Structure):
    _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long),
                ("r", ctypes.c_long), ("b", ctypes.c_long)]


def window_rect(w):
    hwnd = int(w.wm_frame(), 16)
    rect = RECT()
    # DWMWA_EXTENDED_FRAME_BOUNDS = 9 (excludes drop shadow)
    ok = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect))
    if ok != 0:
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.l, rect.t, rect.r, rect.b


def shot(w, name):
    w.update()
    w.lift()
    w.attributes("-topmost", True)
    w.update()
    time.sleep(0.8)
    w.update()
    img = ImageGrab.grab(bbox=window_rect(w), all_screens=True)
    path = os.path.join(SHOTS, name + ".png")
    img.save(path)
    w.attributes("-topmost", False)
    print("saved", path, img.size)


def newest_toplevel(title_contains=None):
    tops = [c for c in root.winfo_children() if isinstance(c, tk.Toplevel)]
    if title_contains:
        tops = [t for t in tops if title_contains in t.title()]
    return tops[-1] if tops else None


def find_widget(parent, cls, text):
    for c in parent.winfo_children():
        if isinstance(c, cls):
            try:
                if c.cget("text") == text:
                    return c
            except tk.TclError:
                pass
        found = find_widget(c, cls, text)
        if found is not None:
            return found
    return None


def find_all(parent, cls):
    out = []
    for c in parent.winfo_children():
        if isinstance(c, cls):
            out.append(c)
        out.extend(find_all(c, cls))
    return out


# ---- scripted session (generator; yields ms to wait or a predicate) -----
def session():
    nb = Main.notebook

    # 1. Load dataset
    Main.on_load_dataset()
    yield 500
    shot(root, "01_load_preview")

    # 2. Column profiles (numeric + categorical)
    Main.column_profile.show_column_profile(root, "radius_mean")
    yield 300
    w = newest_toplevel("Profile")
    w.geometry(PROFILE_GEOMETRY)
    yield 800
    shot(w, "02a_profile_numeric")
    w.grab_release(); w.destroy()

    Main.column_profile.show_column_profile(root, TARGET)
    yield 300
    w = newest_toplevel("Profile")
    w.geometry(PROFILE_GEOMETRY)
    yield 800
    shot(w, "02b_profile_categorical")
    w.grab_release(); w.destroy()

    # 3. Data cleaning
    Main.on_data_cleaning()
    yield 400
    Main.preview_text.configure(state="normal")
    Main.preview_text.see("end")
    yield 300
    shot(root, "03_cleaning")
    RESULTS["after_cleaning_shape"] = list(dataset.CURRENT_DF.shape)
    RESULTS["after_cleaning_columns"] = list(dataset.CURRENT_DF.columns)

    # 4. Control settings
    nb.select(Main.tab_controls)
    Main.entry_target.delete(0, "end")
    Main.entry_target.insert(0, TARGET)
    Main.task_type_var.set("Classification")
    Main.model_var.set("Random Forest")
    Main.on_apply_settings()
    yield 300
    shot(root, "04_control_settings")

    # 5. Feature creation popup, then Deep mode (PCA + ICA + AE)
    Main.open_feature_creation_popup()
    yield 600
    w = newest_toplevel("Feature Creation")
    w.geometry("520x560")
    yield 500
    shot(w, "05a_feature_creation_popup")
    w.grab_release(); w.destroy()

    Main.start_feature_creation_with_progress(
        True, True, True, 5, 5, 6, "MinMaxScaler", target_column=TARGET)
    t0 = time.time()
    yield lambda: dataset.POST_CREATION_DF is not None and newest_toplevel("Creating") is None
    RESULTS["feature_creation_seconds"] = round(time.time() - t0, 1)
    yield 500
    shot(root, "05b_feature_creation_results")
    RESULTS["post_creation_shape"] = list(dataset.POST_CREATION_DF.shape)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    dataset.POST_CREATION_DF.to_csv(os.path.join(HERE, "results", "gui_post_creation.csv"), index=False)

    # 6. Latent explorer
    nb.select(Main.latent_tab)
    yield 300
    find_widget(Main.latent_tab, tk.Button, "Refresh list").invoke()
    combos = find_all(Main.latent_tab, ttk.Combobox)
    latent_combo, target_combo = combos[0], combos[1]
    latent_combo.set("latent_pca_1")
    find_widget(Main.latent_tab, tk.Button, "Plot selected latent").invoke()
    yield 600
    shot(root, "06a_latent_hist")
    find_widget(Main.latent_tab, tk.Button, "Latent Corr Heatmap").invoke()
    yield 600
    shot(root, "06b_latent_heatmap")
    latent_combo.set("latent_auto_1")
    find_widget(Main.latent_tab, tk.Button, "Explain selected latent").invoke()
    yield 600
    shot(root, "06c_latent_explain")
    find_widget(Main.latent_tab, tk.Button, "Refresh targets").invoke()
    target_combo.set(TARGET)
    find_widget(Main.latent_tab, tk.Button, "Compare importance").invoke()
    yield 1500
    shot(root, "06d_latent_importance")

    # 7. Pruning window (inspection only; no columns removed in this run)
    Main.on_transformations()
    yield 800
    w = newest_toplevel("Pruning")
    shot(w, "07_pruning")
    w.destroy()

    # 8. Supervised feature selection (K tuned by CV)
    t0 = time.time()
    Main.on_feature_selection()
    RESULTS["feature_selection_seconds"] = round(time.time() - t0, 1)
    yield 500
    shot(root, "08_feature_selection")
    RESULTS["selected_features"] = [c for c in dataset.CURRENT_DF.columns if c != TARGET]
    RESULTS["results_text_feature_selection"] = Main.results_text.get("1.0", "end")

    # 9. Evaluate impact
    t0 = time.time()
    Main.on_eval_impact()
    yield 500
    yield lambda: Main.LAST_EVAL_RESULTS is not None and (
        Main.EVAL_THREAD is None or not Main.EVAL_THREAD.is_alive())
    RESULTS["evaluation_seconds"] = round(time.time() - t0, 1)
    yield 800
    shot(root, "09_evaluate_impact")
    RESULTS["results_text_eval"] = Main.results_text.get("1.0", "end")
    ev = evaluate_impact._LAST_RESULTS
    from sklearn.metrics import confusion_matrix, roc_auc_score
    final = ev["stages"][-1]["fold_outputs"]
    yt = np.concatenate([f["y_true"] for f in final])
    yp = np.concatenate([f["y_pred"] for f in final])
    pr = np.concatenate([f["y_proba"] for f in final])
    labels = sorted(set(yt))
    RESULTS["final_confusion"] = {"labels": labels,
                                  "matrix": confusion_matrix(yt, yp, labels=labels).tolist()}
    RESULTS["final_pooled_auc"] = float(roc_auc_score(yt == labels[1], pr))
    RESULTS["eval"] = {
        "main_metric": ev["main_metric"],
        "significance": ev["significance"],
        "stages": [
            {k: v for k, v in st.items() if k in ("name", "n_features", "metrics", "delta_main", "error")}
            for st in ev["stages"]
        ],
    }

    # 10. Confusion matrix and ROC / PR
    evaluate_impact.show_confusion_matrices(root)
    yield 800
    w = newest_toplevel("Confusion")
    shot(w, "10a_confusion_matrix")
    w.destroy()
    evaluate_impact.show_roc_pr_curves(root)
    yield 800
    w = newest_toplevel("ROC")
    if w is not None:
        shot(w, "10b_roc_pr")
        w.destroy()

    # 11. Export report
    Main.on_export_report()
    yield 500

    RESULTS["dialogs"] = DIALOG_LOG
    with open(os.path.join(HERE, "gui_results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    print("DONE", flush=True)
    root.after(300, lambda: os._exit(0))  # TF threads keep mainloop alive otherwise


def runner(gen):
    try:
        step = next(gen)
    except StopIteration:
        return
    except Exception:
        traceback.print_exc()
        root.destroy()
        return
    if callable(step):
        def poll():
            if step():
                runner(gen)
            else:
                root.after(300, poll)
        root.after(300, poll)
    else:
        root.after(int(step), lambda: runner(gen))


root.after(1500, lambda: runner(session()))
root.mainloop()
os._exit(0)  # TensorFlow worker threads otherwise keep the process alive
