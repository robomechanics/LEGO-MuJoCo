#!/usr/bin/env python3
"""Plot sweep results as a simple 2D or 3D scatter plot."""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from sweep_axis_utils import (
    axis_percent_label,
    axis_short_label,
    infer_base,
    percentage_difference,
    resolve_sweep_axes,
    validate_sweep_columns,
)


def default_input_csv() -> Path:
    xy_csv = Path("xy_sweep_results.csv")
    if xy_csv.exists():
        return xy_csv
    return Path("sweep_results.csv")


def default_success_min_distance() -> float:
    return 0.0


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_rows(csv_path: Path, filter_min_distance: float | None = None) -> list[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    try:
        validate_sweep_columns(fieldnames)
    except ValueError as exc:
        raise ValueError(
            f"{csv_path}: {exc}. "
            "Use the combined CSV produced by run_sweep.py."
        ) from exc

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


def parse_axis_list(values: list[str] | None) -> list[str]:
    axes: list[str] = []
    for value in values or []:
        axes.extend(axis.strip() for axis in value.split(",") if axis.strip())
    return axes


def prompt_for_ignored_axes(axis_names: tuple[str, ...], already_ignored: set[str]) -> set[str]:
    ignored_axes = set(already_ignored)
    while len(axis_names) - len(ignored_axes) > 3:
        remaining = [axis_name for axis_name in axis_names if axis_name not in ignored_axes]
        required = len(remaining) - 3
        print("CSV has more than 3 sweep axes.", file=sys.stderr)
        print("Axes:", ", ".join(axis_names), file=sys.stderr)
        print(
            f"Enter at least {required} axis name(s) to ignore. "
            "Ignored axes are filtered to their default/base value.",
            file=sys.stderr,
        )
        response = input("Ignore axis/axes: ")
        selected = {axis.strip() for axis in response.split(",") if axis.strip()}
        invalid = selected - set(axis_names)
        if invalid:
            print(f"Unknown axis name(s): {', '.join(sorted(invalid))}", file=sys.stderr)
            continue
        ignored_axes.update(selected)
    return ignored_axes


def resolve_active_axes(axis_names: tuple[str, ...], ignored_axes: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ignored_set = set(ignored_axes)
    invalid = ignored_set - set(axis_names)
    if invalid:
        raise ValueError(f"Cannot ignore unknown sweep axis/axes: {', '.join(sorted(invalid))}")

    if len(axis_names) - len(ignored_set) > 3:
        if not sys.stdin.isatty():
            needed = len(axis_names) - 3
            raise ValueError(
                f"CSV has {len(axis_names)} sweep axes. Pass at least {needed} "
                "axis name(s) with --ignore-axis so only 2 or 3 axes remain."
            )
        ignored_set = prompt_for_ignored_axes(axis_names, ignored_set)

    active_axes = tuple(axis_name for axis_name in axis_names if axis_name not in ignored_set)
    if len(active_axes) not in {2, 3}:
        raise ValueError(
            f"Expected 2 or 3 active axes after filtering, got {len(active_axes)}: "
            f"{', '.join(active_axes) or 'none'}"
        )
    return active_axes, tuple(axis_name for axis_name in axis_names if axis_name in ignored_set)


def axis_column(axis_spec, axis_name: str) -> str:
    try:
        return axis_spec.value_columns[axis_spec.axis_names.index(axis_name)]
    except ValueError as exc:
        raise ValueError(f"Unknown sweep axis: {axis_name}") from exc


def numeric_values(rows: list[dict], column: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def filter_default_axis_values(rows: list[dict], axis_spec, ignored_axes: tuple[str, ...]) -> list[dict]:
    filtered_rows = rows
    for axis_name in ignored_axes:
        column = axis_column(axis_spec, axis_name)
        observed_values = numeric_values(filtered_rows, column)
        if not observed_values:
            raise ValueError(f"No finite values found for ignored axis {axis_name}.")

        base = infer_base(observed_values, axis_name)
        nearest_value = min(observed_values, key=lambda value: abs(value - base))
        tolerance = max(1e-9, abs(nearest_value) * 1e-6)
        before_count = len(filtered_rows)
        filtered_rows = [
            row for row in filtered_rows
            if math.isclose(float(row[column]), nearest_value, rel_tol=1e-6, abs_tol=tolerance)
        ]
        print(
            f"Ignored {axis_name}: kept default/base slice {nearest_value:g} "
            f"({before_count} -> {len(filtered_rows)} rows)."
        )

    if not filtered_rows:
        raise ValueError("No rows remain after filtering ignored axes to default/base values.")
    return filtered_rows


def axis_percent_values(rows: list[dict], axis_spec, axis_name: str) -> list[float]:
    column = axis_column(axis_spec, axis_name)
    values = numeric_values(rows, column)
    if not values:
        raise ValueError(f"No finite values found for active axis {axis_name}.")
    base = infer_base(values, axis_name)
    return [percentage_difference(float(row[column]), base) for row in rows]


def row_distance(row: dict) -> float:
    try:
        value = float(row["Distance_Traversed"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def row_success(row: dict, min_distance: float) -> bool:
    distance = row_distance(row)
    return parse_bool(row.get("Gait_Quality_Pass", "")) and math.isfinite(distance) and distance >= min_distance


def plot_scatter(
    rows: list[dict],
    axis_spec,
    active_axes: tuple[str, ...],
    min_distance: float,
    output_path: Path,
) -> None:
    axis_values = [axis_percent_values(rows, axis_spec, axis_name) for axis_name in active_axes]
    distances = np.array([row_distance(row) for row in rows], dtype=float)
    successes = np.array([row_success(row, min_distance) for row in rows], dtype=bool)
    successful_distances = distances[successes]

    if len(active_axes) == 3:
        fig = plt.figure(figsize=(9, 7), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            np.array(axis_values[0])[~successes],
            np.array(axis_values[1])[~successes],
            np.array(axis_values[2])[~successes],
            c="#bdbdbd",
            marker="x",
            s=18,
            alpha=0.35,
            linewidths=0.8,
            label="Failed",
        )
        scatter = ax.scatter(
            np.array(axis_values[0])[successes],
            np.array(axis_values[1])[successes],
            np.array(axis_values[2])[successes],
            c=successful_distances,
            cmap="viridis",
            s=28,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
            label="Successful",
        )
        ax.set_zlabel(axis_percent_label(active_axes[2]))
    else:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.scatter(
            np.array(axis_values[0])[~successes],
            np.array(axis_values[1])[~successes],
            c="#bdbdbd",
            marker="x",
            s=22,
            alpha=0.35,
            linewidths=0.8,
            label="Failed",
        )
        scatter = ax.scatter(
            np.array(axis_values[0])[successes],
            np.array(axis_values[1])[successes],
            c=successful_distances,
            cmap="viridis",
            s=34,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
            label="Successful",
        )
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(axis_percent_label(active_axes[0]))
    ax.set_ylabel(axis_percent_label(active_axes[1]))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(
        f"{' vs '.join(axis_short_label(axis_name) for axis_name in active_axes)} "
        f"Scatter ({successes.sum()}/{len(rows)} successful, min distance {min_distance:g} m)"
    )

    if successful_distances.size:
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Distance Traversed (m)")

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
    parser.add_argument(
        "--ignore-axis",
        action="append",
        default=[],
        help=(
            "Sweep axis to ignore when the CSV has more than 3 axes. "
            "Can be passed multiple times or as comma-separated names. "
            "Ignored axes are filtered to their default/base value."
        ),
    )
    parser.add_argument(
        "--ignore-axes",
        action="append",
        default=[],
        help="Comma-separated alias for --ignore-axis.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv, args.filter_min_distance)
    axis_spec = resolve_sweep_axes(rows, list(rows[0].keys()))
    ignored_axes = parse_axis_list(args.ignore_axis + args.ignore_axes)
    active_axes, ignored_axes_tuple = resolve_active_axes(axis_spec.axis_names, ignored_axes)
    rows = filter_default_axis_values(rows, axis_spec, ignored_axes_tuple)
    plot_scatter(rows, axis_spec, active_axes, args.min_distance, args.out)
    print(f"Wrote plot to {args.out}")


if __name__ == "__main__":
    main()
