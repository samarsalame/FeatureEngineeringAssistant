import tkinter as tk
from tkinter import ttk, messagebox
import dataset


def build_pruning_panel(parent, update_preview_callback):
    """
    parent: frame inside Transformations tab
    update_preview_callback: function to refresh dataset preview UI
    """
    frame = tk.Frame(parent, bg="#20232a")
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    title = tk.Label(
        frame,
        text="Manual Feature Pruning",
        bg="#20232a",
        fg="white",
        font=("Segoe UI", 12, "bold"),
    )
    title.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))

    subtitle = tk.Label(
        frame,
        text=(
            "Select features to delete from the dataset. Deleted features\n"
            "remain banned until reset. Click RESET to restore the dataset\n"
            "to the state right after Feature Creation."
        ),
        bg="#20232a",
        fg="#9da5b4",
        font=("Segoe UI", 9),
        justify="left",
    )
    subtitle.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 10))

    # Scrollable list container
    list_container = tk.Frame(frame, bg="#20232a")
    list_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
    frame.grid_rowconfigure(2, weight=1)

    canvas = tk.Canvas(list_container, bg="#20232a", highlightthickness=0)
    scroll_y = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg="#20232a")

    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scroll_y.set)

    canvas.pack(side="left", fill="both", expand=True)
    scroll_y.pack(side="right", fill="y")

    checkbox_vars = {}

    # -----------------------------------------------------------
    # Populate checkboxes for CURRENT_DF
    # -----------------------------------------------------------
    def populate_checkboxes():
        for child in inner.winfo_children():
            child.destroy()

        checkbox_vars.clear()

        df = dataset.CURRENT_DF
        if df is None:
            msg = tk.Label(inner, text="No dataset loaded.", bg="#20232a",
                           fg="#9da5b4", font=("Segoe UI", 9))
            msg.pack(anchor="w", padx=6, pady=4)
            return

        for col in df.columns:
            var = tk.BooleanVar()
            checkbox_vars[col] = var

            cb = tk.Checkbutton(
                inner,
                text=col,
                variable=var,
                bg="#20232a",
                fg="#d7dae0",
                selectcolor="#20232a",
                anchor="w",
                activebackground="#20232a",
                activeforeground="#d7dae0"
            )
            cb.pack(fill="x", padx=8, pady=1)

        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    populate_checkboxes()

    # -----------------------------------------------------------
    # DELETE SELECTED FEATURES
    # -----------------------------------------------------------
    def delete_selected():
        if dataset.CURRENT_DF is None:
            messagebox.showwarning("No dataset", "Load or create features first.")
            return

        to_delete = [col for col, var in checkbox_vars.items() if var.get()]

        if not to_delete:
            messagebox.showinfo("Nothing selected", "Select features to delete.")
            return

        # Add them to banned list
        for col in to_delete:
            if col not in dataset.BANNED_FEATURES:
                dataset.BANNED_FEATURES.append(col)

        # Drop from dataset
        dataset.CURRENT_DF = dataset.CURRENT_DF.drop(columns=to_delete, errors="ignore")

        # Update metadata
        dataset.CURRENT_META = dataset._summarize_dataframe(dataset.CURRENT_DF)

        # Refresh UI
        update_preview_callback()

        # Rebuild list
        populate_checkboxes()

    btn_delete = tk.Button(
        frame,
        text="Delete Selected",
        bg="#FF5EAA",
        fg="white",
        pady=6,
        padx=10,
        font=("Segoe UI", 10, "bold"),
        command=delete_selected
    )
    btn_delete.grid(row=3, column=0, sticky="e", padx=10, pady=(0, 6))

    # -----------------------------------------------------------
    # RESET BUTTON (NEW)
    # -----------------------------------------------------------
    def reset_pruning():
        # Ensure feature creation snapshot exists
        if not hasattr(dataset, "POST_CREATION_DF") or dataset.POST_CREATION_DF is None:
            messagebox.showwarning(
                "Reset Unavailable",
                "Feature Creation has not been run yet.\n\n"
                "Run Feature Creation before using RESET."
            )
            return

        # Restore dataset to POST FEATURE CREATION state
        dataset.CURRENT_DF = dataset.POST_CREATION_DF.copy()

        # Clear banned feature list
        dataset.BANNED_FEATURES.clear()

        # Update metadata
        dataset.CURRENT_META = dataset._summarize_dataframe(dataset.CURRENT_DF)

        # Refresh preview UI
        update_preview_callback()

        # Rebuild checkboxes
        populate_checkboxes()

        messagebox.showinfo("Reset Complete", "Dataset restored to post-Feature-Creation state.")

    btn_reset = tk.Button(
        frame,
        text="RESET",
        bg="#4C8BF5",
        fg="white",
        pady=6,
        padx=10,
        font=("Segoe UI", 10, "bold"),
        command=reset_pruning
    )
    btn_reset.grid(row=4, column=0, sticky="e", padx=10, pady=(0, 10))


    # -----------------------------------------------------------
    # Display banned features list
    # -----------------------------------------------------------
    banned_label = tk.Label(
        frame,
        text="Banned Features:",
        bg="#20232a",
        fg="#9da5b4",
        font=("Segoe UI", 9, "bold"),
    )
    banned_label.grid(row=5, column=0, sticky="w", padx=10, pady=(4, 2))

    banned_text = tk.Text(
        frame,
        height=4,
        bg="#1e2127",
        fg="#d7dae0",
        font=("Consolas", 9),
        relief="flat",
        highlightthickness=1,
        highlightbackground="#2f333d"
    )
    banned_text.grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 10))

    def update_banned_list():
        banned_text.delete("1.0", "end")
        if dataset.BANNED_FEATURES:
            for col in dataset.BANNED_FEATURES:
                banned_text.insert("end", f"- {col}\n")
        else:
            banned_text.insert("end", "– none –")
        frame.after(800, update_banned_list)

    update_banned_list()

    return frame


def open_pruning_window(root, update_preview_callback):
    """Wrapper window for manual pruning popup"""
    win = tk.Toplevel(root)
    win.title("Manual Feature Pruning")
    win.geometry("520x720")
    win.config(bg="#20232a")

    panel = build_pruning_panel(win, update_preview_callback)
    panel.pack(fill="both", expand=True)
