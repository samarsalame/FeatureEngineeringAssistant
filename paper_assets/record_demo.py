"""
Record a narrated demo video of the Feature Engineering Assistant running the full
pipeline on data.csv (same seeded session as capture_gui.py / the paper).

Usage (from the project root):
    python paper_assets/record_demo.py
Output:
    paper_assets/video/feature_engineering_assistant_demo.mp4

Only the application's screen area is recorded. The app windows are kept on top
while recording; avoid using the computer until the script finishes (~3 min).
"""
import ctypes
import os
import sys
import time
import threading
import traceback

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
APP_DIR = os.path.join(PROJECT, "FinalFeatureEng")
DATA = os.path.join(PROJECT, "data.csv")
OUT_DIR = os.path.join(HERE, "video")
OUT = os.path.join(OUT_DIR, "feature_engineering_assistant_demo.mp4")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

import numpy as np
import imageio.v2 as imageio
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageFont, ImageGrab

FPS = 10
BAR_H = 70
W_APP, H_APP = 1440, 860

# ---------------------------------------------------------------- captions
STATE = {"step": "", "toast": "", "toast_until": 0.0, "stop": False}


def say(text):
    STATE["step"] = text
    print("[step]", text, flush=True)


def _toast(kind):
    def f(title="", message="", **kw):
        STATE["toast"] = f"{title}: {message}".replace("\n", " ")
        STATE["toast_until"] = time.time() + 3.0
        print(f"[{kind}] {title}: {message}", flush=True)
        return True
    return f


for name in ("showinfo", "showwarning", "showerror"):
    setattr(messagebox, name, _toast(name))
filedialog.askopenfilename = lambda **kw: DATA
filedialog.asksaveasfilename = lambda **kw: os.path.join(OUT_DIR, "demo_pipeline_report.docx")

# 100 % UI scale inside the DPI-aware process (crisp, unclipped matplotlib figures)
_orig_tk_init = tk.Tk.__init__
def _tk_init(self, *a, **k):
    _orig_tk_init(self, *a, **k)
    self.tk.call("tk", "scaling", 96 / 72)
tk.Tk.__init__ = _tk_init

import Main
import dataset
import evaluate_impact
import tensorflow as tf
tf.keras.utils.set_random_seed(42)

root = Main.root
root.geometry(f"{W_APP}x{H_APP}+0+0")
root.attributes("-topmost", True)


def font(size, bold=False):
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_STEP, F_TOAST = font(24, True), font(18)
F_TITLE, F_SUB, F_BODY = font(46, True), font(26), font(24)


# ---------------------------------------------------------------- recorder
class RECT(ctypes.Structure):
    _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long), ("r", ctypes.c_long), ("b", ctypes.c_long)]


def window_rect(w):
    hwnd = int(w.wm_frame(), 16)
    rect = RECT()
    if ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)):
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.l, rect.t, rect.r, rect.b


class Recorder(threading.Thread):
    def __init__(self, writer, bbox):
        super().__init__(daemon=True)
        self.writer, self.bbox = writer, bbox
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]
        self.frames = 0

    def compose(self, shot):
        frame = Image.new("RGB", (self.w, self.h + BAR_H), "#0b0d12")
        frame.paste(shot, (0, 0))
        d = ImageDraw.Draw(frame)
        d.rectangle([0, self.h, self.w, self.h + 3], fill="#4C8BF5")
        d.text((20, self.h + 10), STATE["step"], fill="white", font=F_STEP)
        if time.time() < STATE["toast_until"]:
            d.text((20, self.h + 42), "App message — " + STATE["toast"][:150], fill="#4ade80", font=F_TOAST)
        return np.asarray(frame)

    def run(self):
        t0 = time.time()
        last = None
        while not STATE["stop"]:
            try:
                last = self.compose(ImageGrab.grab(bbox=self.bbox, all_screens=True))
            except Exception:
                pass
            due = int((time.time() - t0) * FPS) + 1
            while self.frames < due and last is not None:   # keep real-time pacing
                self.writer.append_data(last)
                self.frames += 1
            time.sleep(max(0.0, (self.frames / FPS) - (time.time() - t0)))


def card(size, title, subtitle, lines, seconds, writer):
    img = Image.new("RGB", size, "#0b0d12")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0], 8], fill="#4C8BF5")
    d.text((80, 120), title, fill="white", font=F_TITLE)
    d.text((80, 190), subtitle, fill="#9da5b4", font=F_SUB)
    y = 290
    for ln in lines:
        d.text((80, y), ln, fill="#d7dae0", font=F_BODY)
        y += 46
    arr = np.asarray(img)
    for _ in range(int(seconds * FPS)):
        writer.append_data(arr)


# ---------------------------------------------------------------- helpers
def newest_toplevel(title_contains=None):
    tops = [c for c in root.winfo_children() if isinstance(c, tk.Toplevel)]
    if title_contains:
        tops = [t for t in tops if title_contains in t.title()]
    return tops[-1] if tops else None


def find_all(parent, cls):
    out = []
    for c in parent.winfo_children():
        if isinstance(c, cls):
            out.append(c)
        out.extend(find_all(c, cls))
    return out


def find_widget(parent, cls, text):
    for c in find_all(parent, cls):
        try:
            if c.cget("text") == text:
                return c
        except tk.TclError:
            pass
    return None


def on_top(w, geometry=None):
    if geometry:
        w.geometry(geometry)
    w.attributes("-topmost", True)
    w.lift()


def typing(entry, text):
    entry.delete(0, "end")
    for ch in text:
        entry.insert("end", ch)
        yield 110


# ---------------------------------------------------------------- session
def session():
    nb = Main.notebook
    yield 800

    say("1 / 9  Load dataset — data.csv (Wisconsin Diagnostic Breast Cancer)")
    yield 1500
    Main.on_load_dataset()
    yield 3500
    Main.preview_text.yview_moveto(0.35)
    yield 2000

    say("2 / 9  Column profiling — numeric feature radius_mean")
    Main.column_profile.show_column_profile(root, "radius_mean")
    yield 200
    w = newest_toplevel("Profile")
    on_top(w, "920x800+260+30")
    yield 2200
    for rb_text in ["KDE", "Box", "Violin", "Histogram"]:
        rb = find_widget(w, tk.Radiobutton, rb_text)
        if rb:
            rb.invoke()
        yield 1500
    w.grab_release(); w.destroy()

    say("2 / 9  Column profiling — categorical target diagnosis (357 benign / 212 malignant)")
    Main.column_profile.show_column_profile(root, "diagnosis")
    yield 200
    w = newest_toplevel("Profile")
    on_top(w, "920x800+260+30")
    yield 2500
    rb = find_widget(w, tk.Radiobutton, "Pie")
    if rb:
        rb.invoke()
    yield 2200
    w.grab_release(); w.destroy()

    say("3 / 9  Data cleaning — removes the empty column and the ID-like 'id' column")
    yield 1200
    Main.on_data_cleaning()
    Main.preview_text.configure(state="normal")
    Main.preview_text.see("end")
    yield 4500

    say("4 / 9  Control settings — target, task type and model")
    nb.select(Main.tab_controls)
    yield 800
    yield from typing(Main.entry_target, "diagnosis")
    yield 400
    Main.task_type_var.set("Classification")
    yield 600
    Main.model_var.set("Random Forest")
    Main.on_apply_settings()
    yield 2500

    say("5 / 9  Feature creation — Deep mode: 5 PCA + 5 ICA + 6 autoencoder features")
    Main.open_feature_creation_popup()
    yield 200
    w = newest_toplevel("Feature Creation")
    on_top(w, "520x560+460+120")
    yield 3500
    w.grab_release(); w.destroy()
    Main.start_feature_creation_with_progress(True, True, True, 5, 5, 6, "MinMaxScaler",
                                              target_column="diagnosis")
    yield 100
    p = newest_toplevel("Creating")
    if p is not None:
        on_top(p, "+530+300")
    yield lambda: dataset.POST_CREATION_DF is not None and newest_toplevel("Creating") is None
    yield 3000

    say("6 / 9  Latent Features Explorer — distributions, correlations, explainability, importance")
    nb.select(Main.latent_tab)
    yield 800
    find_widget(Main.latent_tab, tk.Button, "Refresh list").invoke()
    combos = find_all(Main.latent_tab, ttk.Combobox)
    latent_combo, target_combo = combos[0], combos[1]
    latent_combo.set("latent_pca_1")
    find_widget(Main.latent_tab, tk.Button, "Plot selected latent").invoke()
    yield 3000
    find_widget(Main.latent_tab, tk.Button, "Latent Corr Heatmap").invoke()
    yield 3500
    latent_combo.set("latent_auto_1")
    yield 600
    find_widget(Main.latent_tab, tk.Button, "Explain selected latent").invoke()
    yield 3500
    find_widget(Main.latent_tab, tk.Button, "Refresh targets").invoke()
    target_combo.set("diagnosis")
    yield 600
    find_widget(Main.latent_tab, tk.Button, "Compare importance").invoke()
    yield 4000

    say("7 / 9  Manual pruning panel (no columns removed in this run)")
    Main.on_transformations()
    yield 200
    w = newest_toplevel("Pruning")
    on_top(w, "520x720+460+60")
    yield 3500
    w.destroy()

    nb.select(Main.tab_results)
    Main.results_text.delete("1.0", "end")
    Main.results_text.insert("end", "Running supervised feature selection (MI + Random Forest + RFE, K tuned by CV)...\n")
    say("8 / 9  Supervised feature selection — MI + Random Forest + RFE, K chosen by CV (running…)")
    yield 1500
    Main.on_feature_selection()
    say("8 / 9  Supervised feature selection — K = 25 features retained (7 latent)")
    yield 4500
    Main.results_text.yview_moveto(0.45)
    yield 3000

    say("9 / 9  Evaluate impact — 5-fold CV across pipeline stages (running…)")
    Main.on_eval_impact()
    yield 500
    yield lambda: Main.LAST_EVAL_RESULTS is not None and (
        Main.EVAL_THREAD is None or not Main.EVAL_THREAD.is_alive())
    say("9 / 9  Evaluate impact — F1 0.958 → 0.967 → 0.968 (differences not significant)")
    yield 5500

    say("9 / 9  Final-stage diagnostics — confusion matrix")
    evaluate_impact.show_confusion_matrices(root)
    yield 200
    w = newest_toplevel("Confusion")
    on_top(w, "+420+180")
    yield 3500
    w.destroy()
    say("9 / 9  Final-stage diagnostics — ROC (AUC = 0.994) and precision–recall curves")
    evaluate_impact.show_roc_pr_curves(root)
    yield 200
    w = newest_toplevel("ROC")
    if w is not None:
        on_top(w, "+260+200")
    yield 4000
    if w is not None:
        w.destroy()

    say("Export report — DOCX summary of dataset, features, settings and evaluation")
    Main.on_export_report()
    yield 3500
    STATE["stop"] = True
    yield 600
    root.after(100, root.quit)


def runner(gen):
    try:
        step = next(gen)
    except StopIteration:
        return
    except Exception:
        traceback.print_exc()
        STATE["stop"] = True
        root.after(100, root.quit)
        return
    if callable(step):
        def poll():
            if step():
                runner(gen)
            else:
                root.after(200, poll)
        root.after(200, poll)
    else:
        root.after(int(step), lambda: runner(gen))


# ---------------------------------------------------------------- main
root.update()
time.sleep(0.5)
root.update()
l, t, r, b = window_rect(root)
bbox = (l, t, l + ((r - l) // 2) * 2, t + ((b - t) // 2) * 2)
vid_size = (bbox[2] - bbox[0], bbox[3] - bbox[1] + BAR_H)
if vid_size[1] % 2:
    vid_size = (vid_size[0], vid_size[1] + 1)

writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=1)
card(vid_size, "Feature Engineering Assistant",
     "End-to-end demo on the Wisconsin Diagnostic Breast Cancer dataset",
     ["569 samples · 30 numeric features · binary diagnosis (M / B)",
      "Load → profile → clean → create latent features → explore",
      "→ select features → evaluate impact → export report",
      "",
      "Real-time recording of the desktop application (seed 42)."], 5, writer)

rec = Recorder(writer, bbox)
rec.start()
root.after(300, lambda: runner(session()))
root.mainloop()
STATE["stop"] = True
rec.join(timeout=5)

card(vid_size, "Results on WDBC",
     "Random Forest, 5-fold stratified cross-validation (in-app Evaluate Impact)",
     ["Raw features (32 inputs)            F1 = 0.958",
      "+ 16 latent features (46 inputs)    F1 = 0.967   (p = 0.09)",
      "+ feature selection (25 inputs)     F1 = 0.968   (p = 0.70)",
      "Final stage: sensitivity 95.8 %, specificity 97.5 %, ROC-AUC 0.994",
      "",
      "Leakage-free benchmark (4 models, 30 folds): no significant differences;",
      "see the paper for details."], 7, writer)
writer.close()
print("saved", OUT, f"{rec.frames / FPS:.0f} s of screen recording", flush=True)
os._exit(0)
