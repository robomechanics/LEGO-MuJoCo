#!/usr/bin/env python3
"""Interactive X/Y sweep plot with scroll-wheel min-distance control."""

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def default_input_csv() -> Path:
    xy_csv = Path("xy_sweep_results.csv")
    if xy_csv.exists():
        return xy_csv
    return Path("sweep_results.csv")


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def trunc_sig(value: float, sig_digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    if value == 0:
        return "0"

    sign = -1 if value < 0 else 1
    value_abs = abs(value)
    exponent = math.floor(math.log10(value_abs))
    scale = 10 ** (exponent - sig_digits + 1)
    truncated = sign * math.floor(value_abs / scale) * scale
    return f"{truncated:.{sig_digits}g}"


def finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def percentage_difference(value: float, base: float) -> float:
    return 100.0 * (value / base - 1.0)


def infer_base(values: list[float], config_name: str) -> float:
    try:
        import xy_sweep_config as config

        return float(getattr(config, config_name))
    except (ImportError, AttributeError, TypeError, ValueError):
        return float(np.median(values))


def configured_percent_centers(values: list[float], base_name: str, values_name: str) -> list[float]:
    try:
        import xy_sweep_config as config

        base = float(getattr(config, base_name))
        configured_values = getattr(config, values_name)
        return sorted(round(percentage_difference(float(value), base), 6) for value in configured_values)
    except (ImportError, AttributeError, TypeError, ValueError):
        return values


def cell_edges(centers: list[float]) -> np.ndarray:
    centers_array = np.array(sorted(centers), dtype=float)
    if len(centers_array) == 1:
        half_width = 1.0
        return np.array([centers_array[0] - half_width, centers_array[0] + half_width])

    midpoints = (centers_array[:-1] + centers_array[1:]) / 2.0
    first = centers_array[0] - (midpoints[0] - centers_array[0])
    last = centers_array[-1] + (centers_array[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"Mesh_X", "Mesh_Y", "Gait_Quality_Pass", "Distance_Traversed"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"{csv_path} is missing required column(s): {missing_list}. "
            "Use the combined CSV produced by run_xy_sweep.py."
        )

    if not rows:
        raise ValueError(f"{csv_path} has no data rows.")

    return rows


@dataclass
class GroupedCell:
    row_idx: int
    col_idx: int
    rows: list[dict]


@dataclass
class PlotData:
    x_centers: list[float]
    y_centers: list[float]
    cells: list[GroupedCell]


def prepare_plot_data(rows: list[dict]) -> PlotData:
    mesh_x_values = [float(row["Mesh_X"]) for row in rows]
    mesh_y_values = [float(row["Mesh_Y"]) for row in rows]
    base_x = infer_base(mesh_x_values, "BASE_X")
    base_y = infer_base(mesh_y_values, "BASE_Y")

    grouped = defaultdict(list)
    for row in rows:
        x_pct = round(percentage_difference(float(row["Mesh_X"]), base_x), 6)
        y_pct = round(percentage_difference(float(row["Mesh_Y"]), base_y), 6)
        grouped[(x_pct, y_pct)].append(row)

    x_centers = configured_percent_centers(sorted({key[0] for key in grouped}), "BASE_X", "X_VALUES")
    y_centers = configured_percent_centers(sorted({key[1] for key in grouped}), "BASE_Y", "Y_VALUES")
    x_index = {value: idx for idx, value in enumerate(x_centers)}
    y_index = {value: idx for idx, value in enumerate(y_centers)}

    cells = [
        GroupedCell(
            row_idx=y_index[y_pct],
            col_idx=x_index[x_pct],
            rows=group_rows,
        )
        for (x_pct, y_pct), group_rows in grouped.items()
    ]

    return PlotData(x_centers=x_centers, y_centers=y_centers, cells=cells)


def build_grid(plot_data: PlotData, min_distance: float) -> tuple[np.ndarray, dict[tuple[int, int], str]]:
    success_grid = np.full((len(plot_data.y_centers), len(plot_data.x_centers)), np.nan)
    labels: dict[tuple[int, int], str] = {}

    for cell in plot_data.cells:
        successes = [
            parse_bool(row["Gait_Quality_Pass"]) and float(row["Distance_Traversed"]) >= min_distance
            for row in cell.rows
        ]
        successful_distances = [
            float(row["Distance_Traversed"])
            for row, is_success in zip(cell.rows, successes)
            if is_success
        ]
        success_count = sum(successes)
        total_count = len(cell.rows)
        mean_success_distance = finite_mean(successful_distances)

        if math.isfinite(mean_success_distance):
            success_grid[cell.row_idx, cell.col_idx] = 100.0 * success_count / total_count

        labels[(cell.row_idx, cell.col_idx)] = (
            f"d={trunc_sig(mean_success_distance)}\n"
            f"pass={success_count}/{total_count}"
        )

    return success_grid, labels


class InteractiveSweepPlot:
    def __init__(self, plot_data: PlotData, initial_min_distance: float, scroll_step: float) -> None:
        self.plot_data = plot_data
        self.min_distance = max(0.0, initial_min_distance)
        self.scroll_step = scroll_step

        self.cmap = LinearSegmentedColormap.from_list(
            "success_red_green",
            ["#c62828", "#f9d65c", "#2e7d32"],
        )
        self.cmap.set_bad("#d0d0d0")

        fig_width = max(8, len(plot_data.x_centers) * 0.85)
        fig_height = max(6, len(plot_data.y_centers) * 0.75)
        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

        self.x_edges = cell_edges(plot_data.x_centers)
        self.y_edges = cell_edges(plot_data.y_centers)
        self.mesh = None
        self.texts: dict[tuple[int, int], any] = {}
        self.colorbar = None
        self.instruction_text = self.fig.text(
            0.02,
            0.01,
            f"Scroll wheel changes min-distance by {self.scroll_step:.2f} m",
            ha="left",
            va="bottom",
        )

        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.redraw()

    def redraw(self) -> None:
        success_grid, labels = build_grid(self.plot_data, self.min_distance)
        masked_grid = np.ma.masked_invalid(success_grid)

        self.ax.clear()
        self.mesh = self.ax.pcolormesh(
            self.x_edges,
            self.y_edges,
            masked_grid,
            cmap=self.cmap,
            vmin=0,
            vmax=100,
            edgecolors="white",
            linewidth=1.0,
            shading="flat",
        )

        for row_idx, y_value in enumerate(self.plot_data.y_centers):
            for col_idx, x_value in enumerate(self.plot_data.x_centers):
                label = labels.get((row_idx, col_idx))
                if label is None:
                    continue
                success_rate = success_grid[row_idx, col_idx]
                text_color = "white" if math.isfinite(success_rate) and success_rate < 35 else "black"
                self.ax.text(
                    x_value,
                    y_value,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

        self.ax.set_xlabel("X difference (%)")
        self.ax.set_ylabel("Y difference (%)")
        self.ax.set_xticks(self.plot_data.x_centers)
        self.ax.set_yticks(self.plot_data.y_centers)
        self.ax.set_xticklabels([f"{value:g}" for value in self.plot_data.x_centers])
        self.ax.set_yticklabels([f"{value:g}" for value in self.plot_data.y_centers])
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(f"X/Y Mesh Sweep Success Rate | Min Distance: {self.min_distance:.2f} m")

        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(self.mesh, ax=self.ax)
            self.colorbar.set_label("Successful trials (%)")
        else:
            self.colorbar.update_normal(self.mesh)

        self.fig.canvas.draw_idle()

    def on_scroll(self, event) -> None:
        if event.button == "up":
            self.min_distance += self.scroll_step
        elif event.button == "down":
            self.min_distance = max(0.0, self.min_distance - self.scroll_step)
        else:
            return
        self.redraw()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=default_input_csv(), help="Combined sweep CSV path.")
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.0,
        help="Initial minimum distance for a trial to count as successful.",
    )
    parser.add_argument(
        "--scroll-step",
        type=float,
        default=0.1,
        help="Distance increment in meters for each mouse-wheel step.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    plot_data = prepare_plot_data(rows)
    viewer = InteractiveSweepPlot(plot_data, args.min_distance, args.scroll_step)
    plt.show()


if __name__ == "__main__":
    main()
