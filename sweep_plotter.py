#!/usr/bin/env python3
"""GUI plotter for successful sweep trials."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
from matplotlib.figure import Figure


@dataclass(frozen=True)
class MetricSpec:
    label: str
    candidates: tuple[str, ...]


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("Foot X", ("Foot_X",)),
    MetricSpec("Foot Y", ("Foot_Y",)),
    MetricSpec("Kp", ("Kp",)),
    MetricSpec("Kd", ("Kd",)),
    MetricSpec("Start Amp", ("Start_Amp_Mult",)),
    MetricSpec("Start Freq", ("Start_Freq_Mult",)),
    MetricSpec("Amp", ("Amplitude_Deg",)),
    MetricSpec("Freq", ("Frequency_Hz",)),
    MetricSpec("Dist Traversed", ("Distance_Traversed",)),
    MetricSpec("Forward Progress", ("Forward_Progress",)),
    MetricSpec("Lateral Progress", ("Lateral_Drift", "Lateral_Progress")),
    MetricSpec(
        "Instantaneous Forward Progress",
        ("Instantaneous_Forward_Progress",),
    ),
    MetricSpec(
        "Instantaneous Forward Absolute",
        ("Instantaneous_Forward_Absolute",),
    ),
    MetricSpec(
        "Instantaneous Lateral Progress",
        ("Instantaneous_Lateral_Progress",),
    ),
    MetricSpec("Forwards Eff", ("Forward_Efficiency",)),
    MetricSpec("Heading Change", ("Heading_Change_Deg",)),
    MetricSpec("Walk Score", ("Walk_Score",)),
    MetricSpec("CoT", ("CoT",)),
)

PLOT_TYPES = ("Scatter", "Hexbin", "2D Histogram")


def default_csv_path() -> Path:
    candidates = (
        Path("sweep_results_v2_2000.csv"),
        Path("sweep_results.csv"),
        Path("xy_sweep_results.csv"),
    )
    for path in candidates:
        if path.exists():
            return path
    return Path("sweep_results.csv")


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def is_successful_row(row: dict[str, str]) -> bool:
    if "Gait_Quality_Pass" in row:
        return parse_bool(row["Gait_Quality_Pass"])
    if "Passed" in row:
        return parse_bool(row["Passed"])
    if "Successful" in row:
        return parse_bool(row["Successful"])
    if "Fell" in row:
        return not parse_bool(row["Fell"])
    return True


def find_metric_column(fieldnames: list[str], spec: MetricSpec) -> str | None:
    field_set = set(fieldnames)
    for candidate in spec.candidates:
        if candidate in field_set:
            return candidate
    return None


def nice_axis_limits(values: np.ndarray) -> tuple[float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return -1.0, 1.0
    if math.isclose(vmin, vmax):
        pad = max(1e-3, abs(vmin) * 0.05, 0.5)
        return vmin - pad, vmax + pad
    span = vmax - vmin
    pad = span * 0.05
    return vmin - pad, vmax + pad


def format_mesh_value(value: float) -> str:
    return f"{value:.6g}"


class SweepPlotterApp:
    def __init__(self, root: tk.Tk, initial_csv: Path) -> None:
        self.root = root
        self.root.title("Sweep Plotter")
        self.root.geometry("1200x900")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.csv_path_var = tk.StringVar(value=str(initial_csv))
        self.status_var = tk.StringVar(value="Load a CSV to begin.")
        self.plot_type_var = tk.StringVar(value=PLOT_TYPES[0])
        self.x_metric_var = tk.StringVar()
        self.y_metric_var = tk.StringVar()
        self.mesh_color_mode = "x"

        self.rows: list[dict[str, str]] = []
        self.available_metrics: dict[str, str] = {}
        self.mesh_columns: dict[str, str] = {}
        self.colorbar = None
        self.toolbar: NavigationToolbar2Tk | None = None
        self._closed = False
        self._canvas_widget: tk.Widget | None = None
        self.color_mode_button: ttk.Button | None = None

        self._build_ui()
        self.load_csv(initial_csv)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(5, weight=1)

        ttk.Label(controls, text="CSV").grid(row=0, column=0, sticky="w", padx=(0, 8))
        csv_entry = ttk.Entry(controls, textvariable=self.csv_path_var)
        csv_entry.grid(row=0, column=1, columnspan=4, sticky="ew")
        ttk.Button(controls, text="Browse", command=self.browse_csv).grid(row=0, column=5, sticky="ew", padx=(8, 0))
        ttk.Button(controls, text="Reload", command=self.reload_csv).grid(row=0, column=6, sticky="ew", padx=(8, 0))

        ttk.Label(controls, text="Plot").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.plot_type_combo = ttk.Combobox(
            controls,
            textvariable=self.plot_type_var,
            values=PLOT_TYPES,
            state="readonly",
        )
        self.plot_type_combo.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        self.plot_type_combo.bind("<<ComboboxSelected>>", self._on_selection_change)

        ttk.Label(controls, text="X Axis").grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(10, 0))
        self.x_combo = ttk.Combobox(controls, textvariable=self.x_metric_var, state="readonly")
        self.x_combo.grid(row=1, column=3, sticky="ew", pady=(10, 0))
        self.x_combo.bind("<<ComboboxSelected>>", self._on_selection_change)

        ttk.Label(controls, text="Y Axis").grid(row=1, column=4, sticky="w", padx=(12, 0), pady=(10, 0))
        self.y_combo = ttk.Combobox(controls, textvariable=self.y_metric_var, state="readonly")
        self.y_combo.grid(row=1, column=5, sticky="ew", pady=(10, 0))
        self.y_combo.bind("<<ComboboxSelected>>", self._on_selection_change)

        self.color_mode_button = ttk.Button(
            controls,
            text="Color By: Mesh X",
            command=self.toggle_mesh_color_mode,
        )
        self.color_mode_button.grid(row=1, column=6, sticky="ew", padx=(8, 0), pady=(10, 0))

        ttk.Label(controls, textvariable=self.status_var).grid(
            row=2,
            column=0,
            columnspan=7,
            sticky="w",
            pady=(12, 0),
        )

        plot_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self._canvas_widget = self.canvas.get_tk_widget()
        self._canvas_widget.grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")

    def browse_csv(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select sweep CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialdir=str(Path(self.csv_path_var.get()).resolve().parent) if self.csv_path_var.get() else os.getcwd(),
        )
        if selected:
            self.csv_path_var.set(selected)
            self.load_csv(Path(selected))

    def reload_csv(self) -> None:
        self.load_csv(Path(self.csv_path_var.get()))

    def load_csv(self, csv_path: Path) -> None:
        try:
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                raw_rows = list(reader)
        except FileNotFoundError:
            messagebox.showerror("CSV not found", f"Could not find:\n{csv_path}")
            return
        except OSError as exc:
            messagebox.showerror("CSV error", f"Could not read:\n{csv_path}\n\n{exc}")
            return

        if not fieldnames:
            messagebox.showerror("CSV error", f"{csv_path} has no header row.")
            return

        success_rows = [row for row in raw_rows if is_successful_row(row)]
        if not success_rows:
            messagebox.showwarning(
                "No successful trials",
                f"{csv_path} contains no successful rows under the current success columns.",
            )

        self.rows = success_rows
        self.available_metrics = {}
        self.mesh_columns = {}
        for spec in METRIC_SPECS:
            column = find_metric_column(fieldnames, spec)
            if column is not None:
                self.available_metrics[spec.label] = column
        for key, column_name in (("x", "Mesh_X"), ("y", "Mesh_Y")):
            if column_name in fieldnames:
                self.mesh_columns[key] = column_name

        labels = list(self.available_metrics)
        if len(labels) < 2:
            messagebox.showerror(
                "Missing metrics",
                f"{csv_path} does not contain enough supported numeric columns to plot.",
            )
            return

        self.x_combo["values"] = labels
        self.y_combo["values"] = labels

        if self.x_metric_var.get() not in self.available_metrics:
            self.x_metric_var.set("Foot X" if "Foot X" in self.available_metrics else labels[0])
        if self.y_metric_var.get() not in self.available_metrics:
            preferred_y = "Foot Y" if "Foot Y" in self.available_metrics else labels[min(1, len(labels) - 1)]
            self.y_metric_var.set(preferred_y)

        self.update_color_mode_button()
        self.csv_path_var.set(str(csv_path))
        self.status_var.set(
            f"Loaded {len(success_rows)} successful trials from {len(raw_rows)} rows in {csv_path.name}."
        )
        self.redraw_plot()

    def get_metric_arrays(self) -> tuple[str, np.ndarray, str, np.ndarray] | None:
        x_label = self.x_metric_var.get()
        y_label = self.y_metric_var.get()
        x_column = self.available_metrics.get(x_label)
        y_column = self.available_metrics.get(y_label)
        if x_column is None or y_column is None or not self.rows:
            return None

        x_values: list[float] = []
        y_values: list[float] = []
        for row in self.rows:
            x_value = safe_float(row.get(x_column))
            y_value = safe_float(row.get(y_column))
            if math.isfinite(x_value) and math.isfinite(y_value):
                x_values.append(x_value)
                y_values.append(y_value)

        if not x_values:
            return None

        return x_label, np.asarray(x_values, dtype=float), y_label, np.asarray(y_values, dtype=float)

    def update_color_mode_button(self) -> None:
        if self.color_mode_button is None:
            return
        has_mesh_colors = "x" in self.mesh_columns and "y" in self.mesh_columns
        next_label = "Mesh X" if self.mesh_color_mode == "x" else "Mesh Y"
        self.color_mode_button.configure(text=f"Color By: {next_label}")
        self.color_mode_button.state(["!disabled"] if has_mesh_colors else ["disabled"])

    def toggle_mesh_color_mode(self) -> None:
        if "x" not in self.mesh_columns or "y" not in self.mesh_columns:
            return
        self.mesh_color_mode = "y" if self.mesh_color_mode == "x" else "x"
        self.update_color_mode_button()
        self.redraw_plot()

    def get_scatter_plot_data(
        self,
    ) -> tuple[str, np.ndarray, str, np.ndarray, np.ndarray] | None:
        metric_arrays = self.get_metric_arrays()
        mesh_column = self.mesh_columns.get(self.mesh_color_mode)
        if metric_arrays is None or mesh_column is None:
            return None

        x_label, x_values, y_label, y_values = metric_arrays
        filtered_x: list[float] = []
        filtered_y: list[float] = []
        mesh_values: list[float] = []
        x_column = self.available_metrics[x_label]
        y_column = self.available_metrics[y_label]

        for row in self.rows:
            x_value = safe_float(row.get(x_column))
            y_value = safe_float(row.get(y_column))
            mesh_value = safe_float(row.get(mesh_column))
            if math.isfinite(x_value) and math.isfinite(y_value) and math.isfinite(mesh_value):
                filtered_x.append(x_value)
                filtered_y.append(y_value)
                mesh_values.append(mesh_value)

        if not filtered_x:
            return None

        return (
            x_label,
            np.asarray(filtered_x, dtype=float),
            y_label,
            np.asarray(filtered_y, dtype=float),
            np.asarray(mesh_values, dtype=float),
        )

    def draw_mesh_colored_scatter(self) -> bool:
        scatter_data = self.get_scatter_plot_data()
        if scatter_data is None:
            return False

        x_label, x_values, y_label, y_values, mesh_values = scatter_data
        unique_mesh_values = sorted({float(value) for value in mesh_values})
        cmap = colormaps["turbo"]
        mesh_min = unique_mesh_values[0]
        mesh_max = unique_mesh_values[-1]

        if math.isclose(mesh_min, mesh_max):
            mesh_to_color = {mesh_min: cmap(0.5)}
        else:
            mesh_to_color = {
                mesh_value: cmap((mesh_value - mesh_min) / (mesh_max - mesh_min))
                for mesh_value in unique_mesh_values
            }

        colors = [mesh_to_color[float(mesh_value)] for mesh_value in mesh_values]

        self.ax.scatter(
            x_values,
            y_values,
            s=28,
            alpha=0.85,
            c=colors,
            edgecolors="none",
        )

        legend_label = "Mesh X" if self.mesh_color_mode == "x" else "Mesh Y"
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=mesh_to_color[mesh_value],
                markeredgewidth=0,
                label=format_mesh_value(mesh_value),
            )
            for mesh_value in unique_mesh_values
        ]
        if handles:
            self.ax.legend(
                handles=handles,
                title=legend_label,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=True,
            )

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_xlim(*nice_axis_limits(x_values))
        self.ax.set_ylim(*nice_axis_limits(y_values))
        self.ax.grid(True, alpha=0.25)
        self.ax.set_title(f"Scatter: {y_label} vs {x_label} | Colored by {legend_label}")
        return True

    def clear_colorbar(self) -> None:
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

    def redraw_plot(self) -> None:
        metric_arrays = self.get_metric_arrays()
        self.ax.clear()
        self.clear_colorbar()

        if metric_arrays is None:
            self.ax.set_title("No plottable successful trials")
            self.ax.text(0.5, 0.5, "No plottable successful trials", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        x_label, x_values, y_label, y_values = metric_arrays
        plot_type = self.plot_type_var.get()

        if plot_type == "Scatter":
            if not self.draw_mesh_colored_scatter():
                self.ax.set_title("No plottable successful trials")
                self.ax.text(
                    0.5,
                    0.5,
                    "No plottable successful trials",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                )
        elif plot_type == "Hexbin":
            artist = self.ax.hexbin(
                x_values,
                y_values,
                gridsize=30,
                mincnt=1,
                cmap="viridis",
            )
            self.colorbar = self.figure.colorbar(artist, ax=self.ax)
            self.colorbar.set_label("Successful trial count")
        elif plot_type == "2D Histogram":
            hist = self.ax.hist2d(
                x_values,
                y_values,
                bins=30,
                cmap="viridis",
            )
            self.colorbar = self.figure.colorbar(hist[3], ax=self.ax)
            self.colorbar.set_label("Successful trial count")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label)
            self.ax.set_xlim(*nice_axis_limits(x_values))
            self.ax.set_ylim(*nice_axis_limits(y_values))
            self.ax.grid(True, alpha=0.25)
            self.ax.set_title(f"{plot_type}: {y_label} vs {x_label}")
        else:
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label)
            self.ax.set_xlim(*nice_axis_limits(x_values))
            self.ax.set_ylim(*nice_axis_limits(y_values))
            self.ax.grid(True, alpha=0.25)
            self.ax.set_title(f"{plot_type}: {y_label} vs {x_label}")
        self.canvas.draw_idle()

    def _on_selection_change(self, _event: object) -> None:
        self.redraw_plot()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            self.root.withdraw()
        except tk.TclError:
            pass

        try:
            after_ids = list(self.root.tk.call("after", "info"))
        except tk.TclError:
            after_ids = []

        for after_id in after_ids:
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass

        try:
            self.clear_colorbar()
        except Exception:
            pass

        try:
            if hasattr(self, "canvas"):
                self.canvas.draw_idle()
                self.canvas.flush_events()
        except Exception:
            pass

        try:
            if self.toolbar is not None:
                self.toolbar.destroy()
                self.toolbar = None
        except Exception:
            self.toolbar = None

        try:
            if self._canvas_widget is not None:
                self._canvas_widget.destroy()
                self._canvas_widget = None
        except Exception:
            self._canvas_widget = None

        try:
            self.root.quit()
        except tk.TclError:
            pass

        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass

        try:
            self.root.destroy()
        except tk.TclError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv_path(),
        help="CSV file to load. Defaults to an existing sweep results CSV in the repo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = SweepPlotterApp(root, args.csv)
    try:
        root.mainloop()
    finally:
        if not app._closed:
            app.close()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
