import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import dataset


def export_current_dataset(default_name):
    df = dataset.get_current_df()

    if df is None or df.empty:
        messagebox.showerror("No Data", "No dataset is currently available.")
        return

    save_path = filedialog.asksaveasfilename(
        title="Save Dataset",
        defaultextension=".csv",
        filetypes=[("CSV file", "*.csv")],
        initialfile=default_name,
    )

    if not save_path:
        return

    try:
        df.to_csv(save_path, index=False)
        messagebox.showinfo("Success", f"Dataset successfully saved:\n{save_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Could not save dataset:\n{e}")


def download_cleaned():
    export_current_dataset("dataset_cleaned.csv")


def download_features_added():
    export_current_dataset("dataset_with_features.csv")


def download_features_selected():
    export_current_dataset("dataset_selected_features.csv")
