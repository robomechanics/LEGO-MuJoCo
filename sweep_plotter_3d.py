#!/usr/bin/env python3
"""GUI 3D plotter for sweep CSV results."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
if not mpl_config_dir.exists() or not os.access(mpl_config_dir, os.W_OK):
    mpl_config_dir = Path("/tmp/matplotlib")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from sweep_axis_utils import infer_base, percentage_difference


@dataclass(frozen=True)
class MetricOption:
    key: str
    label: str
    candidates: tuple[str, ...]
    percent_axis: str | None = None
    derived: str | None = None


METRIC_OPTIONS: tuple[MetricOption, ...] = (
    MetricOption("curve_x", "Curve X Difference (%)", ("Curve_X", "Mesh_X"), "curve_x"),
    MetricOption("curve_y", "Curve Y Difference (%)", ("Curve_Y", "Mesh_Y"), "curve_y"),
    MetricOption("curve_z", "Curve Z Difference (%)", ("Curve_Z",), "curve_z"),
    MetricOption("box_x", "Box X Difference (%)", ("Box_X",), "box_x"),
    MetricOption("box_y", "Box Y Difference (%)", ("Box_Y",), "box_y"),
    MetricOption("box_z", "Box Z Difference (%)", ("Box_Z",), "box_z"),
    MetricOption("foot_x", "Foot X", ("Foot_X",)),
    MetricOption("foot_y", "Foot Y", ("Foot_Y",)),
    MetricOption("kp", "Kp", ("Kp",)),
    MetricOption("kd", "Kd", ("Kd",)),
    MetricOption("start_amp", "Start Amp Mult", ("Start_Amp_Mult",)),
    MetricOption("start_freq", "Start Freq Mult", ("Start_Freq_Mult",)),
    MetricOption("amp", "Amplitude Deg", ("Amplitude_Deg",)),
    MetricOption("freq", "Frequency Hz", ("Frequency_Hz",)),
    MetricOption("distance", "Distance Traversed (m)", ("Distance_Traversed",)),
    MetricOption("forward_progress", "Instantaneous Forward Progress", ("Instantaneous_Forward_Progress",)),
    MetricOption("forward_absolute", "Instantaneous Forward Absolute", ("Instantaneous_Forward_Absolute",)),
    MetricOption("lateral_progress", "Instantaneous Lateral Progress", ("Instantaneous_Lateral_Progress",)),
    MetricOption("velocity", "Instantaneous Forward Velocity (m/s)", (), derived="velocity"),
    MetricOption("roll", "Average Absolute Roll (deg)", ("Average_Abs_Roll_Deg",)),
    MetricOption("pitch", "Average Absolute Pitch (deg)", ("Average_Abs_Pitch_Deg",)),
    MetricOption("path_length", "Path Length", ("Path_Length",)),
    MetricOption("heading_change", "Heading Change (deg)", ("Heading_Change_Deg",)),
    MetricOption("walk_score", "Walk Score", ("Walk_Score",)),
    MetricOption("cot", "CoT", ("CoT",)),
)


def default_csv_path() -> Path:
    candidates = (
        Path("data/sweeps/cyby0/sweep_results.csv"),
        Path("data/sweeps/cyby/sweep_results_final.csv"),
        Path("data/sweeps/cxbx/sweep_results.csv"),
        Path("sweep_results.csv"),
    )
    for path in candidates:
        if path.exists():
            return path
    return Path("sweep_results.csv")


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def default_success_min_distance() -> float:
    return 0.0


def row_distance(row: dict[str, str]) -> float:
    return safe_float(row.get("Distance_Traversed"))


def row_success(row: dict[str, str], min_distance: float) -> bool:
    distance = row_distance(row)
    if "Gait_Quality_Pass" in row:
        quality_pass = parse_bool(row.get("Gait_Quality_Pass", ""))
    elif "Fell" in row:
        quality_pass = not parse_bool(row.get("Fell", ""))
    else:
        quality_pass = True
    return quality_pass and math.isfinite(distance) and distance >= min_distance


def row_instantaneous_forward_velocity(row: dict[str, str]) -> float:
    progress = safe_float(row.get("Instantaneous_Forward_Progress"))
    motion_time = safe_float(row.get("Motion_Time"))
    if not math.isfinite(progress) or not math.isfinite(motion_time) or motion_time <= 1e-9:
        return float("nan")
    return progress / motion_time


def nice_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if math.isclose(vmin, vmax):
        pad = max(1e-3, abs(vmin) * 0.05, 0.5)
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.05
    return vmin - pad, vmax + pad


def metric_column(fieldnames: list[str], option: MetricOption) -> str | None:
    field_set = set(fieldnames)
    for candidate in option.candidates:
        if candidate in field_set:
            return candidate
    return None


def derived_metric_available(fieldnames: list[str], option: MetricOption) -> bool:
    if option.derived == "velocity":
        return {"Instantaneous_Forward_Progress", "Motion_Time"} <= set(fieldnames)
    return False


class SweepPlotter3DApp:
    def __init__(self, root: tk.Tk, initial_csv: Path) -> None:
        self.root = root
        self.root.title("3D Sweep Plotter")
        self.root.geometry("1300x900")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.csv_path_var = tk.StringVar(value=str(initial_csv))
        self.status_var = tk.StringVar(value="Open a CSV to begin.")
        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        self.z_var = tk.StringVar()
        self.color_var = tk.StringVar()
        self.min_distance_var = tk.StringVar(value=str(default_success_min_distance()))

        self.rows: list[dict[str, str]] = []
        self.fieldnames: list[str] = []
        self.metric_by_label: dict[str, MetricOption] = {}
        self.column_by_label: dict[str, str] = {}
        self.colorbar = None

        self._build_ui()
        if initial_csv.exists():
            self.load_csv(initial_csv)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        for column in (1, 3, 5, 7):
            controls.columnconfigure(column, weight=1)

        ttk.Label(controls, text="CSV").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(controls, textvariable=self.csv_path_var).grid(row=0, column=1, columnspan=5, sticky="ew")
        ttk.Button(controls, text="Open CSV", command=self.browse_csv).grid(row=0, column=6, padx=(8, 0), sticky="ew")
        ttk.Button(controls, text="Reload", command=self.reload_csv).grid(row=0, column=7, padx=(8, 0), sticky="ew")

        ttk.Label(controls, text="X").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.x_combo = ttk.Combobox(controls, textvariable=self.x_var, state="readonly")
        self.x_combo.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        self.x_combo.bind("<<ComboboxSelected>>", self._selection_changed)

        ttk.Label(controls, text="Y").grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(10, 0))
        self.y_combo = ttk.Combobox(controls, textvariable=self.y_var, state="readonly")
        self.y_combo.grid(row=1, column=3, sticky="ew", pady=(10, 0))
        self.y_combo.bind("<<ComboboxSelected>>", self._selection_changed)

        ttk.Label(controls, text="Z").grid(row=1, column=4, sticky="w", padx=(12, 0), pady=(10, 0))
        self.z_combo = ttk.Combobox(controls, textvariable=self.z_var, state="readonly")
        self.z_combo.grid(row=1, column=5, sticky="ew", pady=(10, 0))
        self.z_combo.bind("<<ComboboxSelected>>", self._selection_changed)

        ttk.Label(controls, text="Color").grid(row=1, column=6, sticky="w", padx=(12, 0), pady=(10, 0))
        self.color_combo = ttk.Combobox(controls, textvariable=self.color_var, state="readonly")
        self.color_combo.grid(row=1, column=7, sticky="ew", pady=(10, 0))
        self.color_combo.bind("<<ComboboxSelected>>", self._selection_changed)

        ttk.Label(controls, text="Min Distance").grid(row=2, column=0, sticky="w", pady=(10, 0))
        min_distance_entry = ttk.Entry(controls, textvariable=self.min_distance_var, width=12)
        min_distance_entry.grid(row=2, column=1, sticky="w", pady=(10, 0))
        min_distance_entry.bind("<Return>", self._selection_changed)
        ttk.Button(controls, text="Redraw", command=self.redraw).grid(row=2, column=2, padx=(12, 0), pady=(10, 0))
        ttk.Button(controls, text="Save PNG", command=self.save_png).grid(row=2, column=3, padx=(12, 0), pady=(10, 0))
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=3,
            column=0,
            columnspan=8,
            sticky="w",
            pady=(10, 0),
        )

        plot_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.figure = plt.Figure(figsize=(11, 8), constrained_layout=True)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")

    def browse_csv(self) -> None:
        initial = Path(self.csv_path_var.get()).expanduser()
        selected = filedialog.askopenfilename(
            title="Select sweep CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialdir=str(initial.parent if initial.parent.exists() else Path.cwd()),
        )
        if selected:
            self.csv_path_var.set(selected)
            self.load_csv(Path(selected))

    def reload_csv(self) -> None:
        self.load_csv(Path(self.csv_path_var.get()).expanduser())

    def load_csv(self, csv_path: Path) -> None:
        try:
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.fieldnames = list(reader.fieldnames or [])
                self.rows = list(reader)
        except OSError as exc:
            messagebox.showerror("CSV Error", f"Could not read:\n{csv_path}\n\n{exc}")
            return

        if not self.fieldnames:
            messagebox.showerror("CSV Error", f"{csv_path} has no header row.")
            return
        if not self.rows:
            messagebox.showwarning("CSV Empty", f"{csv_path} has no data rows.")

        self.csv_path_var.set(str(csv_path))
        self.populate_metric_choices()
        self.choose_defaults()
        self.status_var.set(f"Loaded {len(self.rows)} rows from {csv_path.name}.")
        self.redraw()

    def populate_metric_choices(self) -> None:
        self.metric_by_label = {}
        self.column_by_label = {}

        for option in METRIC_OPTIONS:
            column = metric_column(self.fieldnames, option)
            if column is not None or derived_metric_available(self.fieldnames, option):
                self.metric_by_label[option.label] = option
                if column is not None:
                    self.column_by_label[option.label] = column

        known_columns = {
            candidate
            for option in METRIC_OPTIONS
            for candidate in option.candidates
        }
        for fieldname in self.fieldnames:
            if fieldname in known_columns:
                continue
            if any(math.isfinite(safe_float(row.get(fieldname))) for row in self.rows):
                label = fieldname.replace("_", " ").title()
                self.metric_by_label[label] = MetricOption(fieldname, label, (fieldname,))
                self.column_by_label[label] = fieldname

        values = [""] + list(self.metric_by_label)
        for combo in (self.x_combo, self.y_combo, self.z_combo, self.color_combo):
            combo["values"] = values

    def choose_defaults(self) -> None:
        labels = set(self.metric_by_label)
        defaults = (
            (self.x_var, ("Curve X Difference (%)", "Curve Y Difference (%)", "Box X Difference (%)")),
            (self.y_var, ("Box X Difference (%)", "Box Y Difference (%)", "Curve Y Difference (%)")),
            (self.z_var, ("Average Absolute Pitch (deg)", "Average Absolute Roll (deg)", "Distance Traversed (m)")),
            (self.color_var, ("Instantaneous Forward Velocity (m/s)", "Distance Traversed (m)")),
        )
        for variable, candidates in defaults:
            if variable.get() in labels:
                continue
            variable.set("")
            for candidate in candidates:
                if candidate in labels:
                    variable.set(candidate)
                    break

    def metric_values(self, label: str) -> tuple[np.ndarray, str]:
        option = self.metric_by_label[label]
        if option.derived == "velocity":
            values = np.array(
                [row_instantaneous_forward_velocity(row) for row in self.rows],
                dtype=float,
            )
            return values, option.label

        column = self.column_by_label[label]
        values = np.array([safe_float(row.get(column)) for row in self.rows], dtype=float)
        if option.percent_axis is not None:
            finite = values[np.isfinite(values)].tolist()
            if not finite:
                return values, option.label
            base = infer_base(finite, option.percent_axis)
            values = np.array([percentage_difference(value, base) for value in values], dtype=float)
        return values, option.label

    def selected_min_distance(self) -> float:
        try:
            return float(self.min_distance_var.get())
        except ValueError:
            return default_success_min_distance()

    def clear_colorbar(self) -> None:
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None

    def redraw(self, _event: object | None = None) -> None:
        self.clear_colorbar()
        self.ax.clear()

        required = (self.x_var.get(), self.y_var.get(), self.z_var.get())
        if not all(required):
            self.ax.text2D(0.5, 0.5, "Choose X, Y, and Z axes", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        try:
            x_values, x_label = self.metric_values(self.x_var.get())
            y_values, y_label = self.metric_values(self.y_var.get())
            z_values, z_label = self.metric_values(self.z_var.get())
            color_label = self.color_var.get()
            if color_label:
                color_values, colorbar_label = self.metric_values(color_label)
            else:
                color_values = np.full(len(self.rows), np.nan)
                colorbar_label = ""
        except (KeyError, ValueError) as exc:
            self.ax.text2D(0.5, 0.5, str(exc), ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        successes = np.array(
            [row_success(row, self.selected_min_distance()) for row in self.rows],
            dtype=bool,
        )
        finite_xyz = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
        failed_mask = finite_xyz & ~successes
        success_mask = finite_xyz & successes

        self.ax.scatter(
            x_values[failed_mask],
            y_values[failed_mask],
            z_values[failed_mask],
            c="#bdbdbd",
            marker="x",
            s=10,
            alpha=0.25,
            linewidths=0.7,
            label="Failed",
        )

        if color_label:
            color_mask = success_mask & np.isfinite(color_values)
            scatter = self.ax.scatter(
                x_values[color_mask],
                y_values[color_mask],
                z_values[color_mask],
                c=color_values[color_mask],
                cmap="viridis",
                s=14,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.2,
                label="Successful",
            )
            if color_mask.any():
                self.colorbar = self.figure.colorbar(scatter, ax=self.ax, shrink=0.75, pad=0.08)
                self.colorbar.set_label(colorbar_label)
            plotted_successes = int(color_mask.sum())
        else:
            self.ax.scatter(
                x_values[success_mask],
                y_values[success_mask],
                z_values[success_mask],
                c="#1b9e77",
                s=14,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.2,
                label="Successful",
            )
            plotted_successes = int(success_mask.sum())

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_zlabel(z_label)
        self.ax.set_xlim(*nice_limits(x_values[finite_xyz]))
        self.ax.set_ylim(*nice_limits(y_values[finite_xyz]))
        self.ax.set_zlim(*nice_limits(z_values[finite_xyz]))
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        self.ax.set_title(f"{z_label} vs {y_label} vs {x_label} ({plotted_successes}/{len(self.rows)} successful)")
        self.status_var.set(f"Plotted {plotted_successes} successful rows from {len(self.rows)} total rows.")
        self.canvas.draw_idle()

    def save_png(self) -> None:
        suggested = Path(self.csv_path_var.get()).with_suffix(".3d.png")
        selected = filedialog.asksaveasfilename(
            title="Save 3D plot",
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")],
            initialdir=str(suggested.parent if suggested.parent.exists() else Path.cwd()),
            initialfile=suggested.name,
        )
        if not selected:
            return
        try:
            self.figure.savefig(selected, dpi=200)
        except OSError as exc:
            messagebox.showerror("Save Error", f"Could not save:\n{selected}\n\n{exc}")
            return
        self.status_var.set(f"Saved {selected}")

    def _selection_changed(self, _event: object) -> None:
        self.redraw()

    def close(self) -> None:
        try:
            self.clear_colorbar()
            self.canvas.get_tk_widget().destroy()
            plt.close(self.figure)
        finally:
            self.root.quit()
            self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=default_csv_path(), help="Optional CSV to open on startup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    SweepPlotter3DApp(root, args.csv)
    root.mainloop()


if __name__ == "__main__":
    main()
