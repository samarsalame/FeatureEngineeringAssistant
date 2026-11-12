# datareport.py
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import dataset  # to fetch CURRENT_DF

# Optional libs (wrap with try so app won’t crash)
try:
    from ydata_profiling import ProfileReport
except:
    ProfileReport = None



try:
    import sweetviz as sv
except:
    sv = None


# -----------------------------------------------------------
# Helper: safely get dataframe
# -----------------------------------------------------------
def _get_df():
    df = dataset.get_current_df()
    if df is None or df.empty:
        messagebox.showerror("No Data", "Please load a dataset first.")
        return None
    return df


# -----------------------------------------------------------
# YData Profiling
# -----------------------------------------------------------
def generate_ydata_report():
    df = _get_df()
    if df is None:
        return

    if ProfileReport is None:
        messagebox.showerror("Missing Library",
                             "ydata-profiling is not installed.\nInstall with:\n\npip install ydata-profiling")
        return

    save_path = filedialog.asksaveasfilename(
        title="Save YData Profiling report",
        defaultextension=".html",
        filetypes=[("HTML File", "*.html")]
    )
    if not save_path:
        return

    try:
        profile = ProfileReport(df, title="Dataset Profile Report", explorative=True)
        profile.to_file(save_path)
        messagebox.showinfo("Success", f"Report saved:\n{save_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# -----------------------------------------------------------
# Dataprep EDA
# -----------------------------------------------------------
def generate_dataprep_report():
    df = _get_df()
    if df is None:
        return



    save_path = filedialog.asksaveasfilename(
        title="Save Dataprep EDA report",
        defaultextension=".html",
        filetypes=[("HTML File", "*.html")]
    )
    if not save_path:
        return




# -----------------------------------------------------------
# Sweetviz Report
#----------------------------------------------

def generate_skimpy_png():
    df = _get_df()
    if df is None:
        return

    try:
        from skimpy import skim
    except ImportError:
        messagebox.showerror(
            "Missing Library",
            "skimpy is not installed.\nInstall with:\n\npip install skimpy"
        )
        return

    # Ask user where to save the PNG
    save_path = filedialog.asksaveasfilename(
        title="Save Skimpy Summary as PNG",
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png")]
    )
    if not save_path:
        return

    # --- Clean dataframe a bit (same as before) ---
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")

    # --- STEP 1: Capture EXACT skimpy terminal output ---
    from io import StringIO
    import sys, re

    buffer = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        skim(df)   # this prints the big table to stdout
    finally:
        sys.stdout = old_stdout

    raw_text = buffer.getvalue()

    # If nothing came out, bail
    if not raw_text.strip():
        messagebox.showerror("Skimpy Error", "Skimpy produced no output to capture.")
        return

    # Optional: strip ANSI color codes so the PNG is clean text
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    text = ansi_escape.sub("", raw_text)

    # --- STEP 2: Render text into a PNG using Pillow ---
    from PIL import Image, ImageDraw, ImageFont

    lines = text.splitlines()
    if not lines:
        messagebox.showerror("Skimpy Error", "No lines captured from Skimpy output.")
        return

    # Use a monospaced font
    try:
        # if you have Consolas on Windows
        font = ImageFont.truetype("consola.ttf", 14)
    except Exception:
        # fallback to default bitmap font
        font = ImageFont.load_default()

    # Measure max line width and line height
    def text_size(s: str):
        bbox = font.getbbox(s if s else " ")
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    max_w = 0
    line_h = 0
    for line in lines:
        w, h = text_size(line)
        max_w = max(max_w, w)
        line_h = max(line_h, h)

    margin_x, margin_y = 20, 20
    img_w = max_w + 2 * margin_x
    img_h = line_h * len(lines) + 2 * margin_y

    img = Image.new("RGB", (img_w, img_h), color="black")
    draw = ImageDraw.Draw(img)

    y = margin_y
    for line in lines:
        draw.text((margin_x, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    img.save(save_path)
    messagebox.showinfo("Success", f"PNG saved:\n{save_path}")
