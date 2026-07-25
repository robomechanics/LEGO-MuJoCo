#!/usr/bin/env python3
"""Plot X/Y mesh sweep success rates from the combined sweep CSV."""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def default_input_csv() -> Path:
    xy_csv = Path("xy_sweep_results.csv")
    if xy_csv.exists():
        return xy_csv
    return Path("sweep_results.csv")


def default_success_min_distance() -> float:
    return 0.0


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


def load_rows(csv_path: Path, filter_min_distance: float | None = None) -> list[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"Mesh_X", "Mesh_Y", "Fell", "Distance_Traversed"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"{csv_path} is missing required column(s): {missing_list}. "
            "Use the combined CSV produced by run_xy_sweep.py."
        )

    if not rows:
        raise ValueError(f"{csv_path} has no data rows.")

    if filter_min_distance is not None:
        rows = [
            row for row in rows
            if float(row["Distance_Traversed"]) >= filter_min_distance
        ]
        if not rows:
            raise ValueError(
                f"{csv_path} has no rows with Distance_Traversed >= {filter_min_distance}."
            )

    return rows


def finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def build_grid(
    rows: list[dict],
    min_distance: float,
) -> tuple[list[float], list[float], np.ndarray, dict]:
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

    success_grid = np.full((len(y_centers), len(x_centers)), np.nan)
    labels = {}

    for (x_pct, y_pct), group_rows in grouped.items():
        distances = [float(row["Distance_Traversed"]) for row in group_rows]
        successes = [
            (not parse_bool(row["Fell"])) and float(row["Distance_Traversed"]) >= min_distance
            for row in group_rows
        ]
        success_count = sum(successes)
        total_count = len(group_rows)

        row_idx = y_index[y_pct]
        col_idx = x_index[x_pct]
        success_grid[row_idx, col_idx] = 100.0 * success_count / total_count
        labels[(row_idx, col_idx)] = (
            f"d={trunc_sig(finite_mean(distances))}\n"
            f"pass={success_count}/{total_count}"
        )

    return x_centers, y_centers, success_grid, labels


def plot_grid(
    x_centers: list[float],
    y_centers: list[float],
    success_grid: np.ndarray,
    labels: dict,
    output_path: Path,
) -> None:
    x_edges = cell_edges(x_centers)
    y_edges = cell_edges(y_centers)

    cmap = LinearSegmentedColormap.from_list("success_red_green", ["#c62828", "#f9d65c", "#2e7d32"])
    cmap.set_bad("#d0d0d0")

    fig_width = max(8, len(x_centers) * 0.85)
    fig_height = max(6, len(y_centers) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(success_grid),
        cmap=cmap,
        vmin=0,
        vmax=100,
        edgecolors="white",
        linewidth=1.0,
        shading="flat",
    )

    for row_idx, y_value in enumerate(y_centers):
        for col_idx, x_value in enumerate(x_centers):
            label = labels.get((row_idx, col_idx))
            if label is None:
                continue
            success_rate = success_grid[row_idx, col_idx]
            text_color = "white" if math.isfinite(success_rate) and success_rate < 35 else "black"
            ax.text(
                x_value,
                y_value,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    ax.set_xlabel("X difference (%)")
    ax.set_ylabel("Y difference (%)")
    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)
    ax.set_xticklabels([f"{value:g}" for value in x_centers])
    ax.set_yticklabels([f"{value:g}" for value in y_centers])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("X/Y Mesh Sweep Success Rate")

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("Successful trials (%)")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=default_input_csv(), help="Combined sweep CSV path.")
    parser.add_argument("--out", type=Path, default=Path("xy_sweep_results_plot.png"), help="Output image path.")
    parser.add_argument(
        "--min-distance",
        type=float,
        default=default_success_min_distance(),
        help="Minimum distance for a trial to count as successful.",
    )
    parser.add_argument(
        "--filter-min-distance",
        type=float,
        default=None,
        help="Drop CSV rows with Distance_Traversed below this threshold before plotting.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv, args.filter_min_distance)
    x_centers, y_centers, success_grid, labels = build_grid(rows, args.min_distance)
    plot_grid(x_centers, y_centers, success_grid, labels, args.out)
    print(f"Wrote plot to {args.out}")


if __name__ == "__main__":
    main()
