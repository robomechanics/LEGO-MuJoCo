#!/usr/bin/env python3
"""Plot ES foot/hip offset exploration from bigfoot_es_history*.csv.

Inspired by plot_xy_sweep_results.py: binned foot_x × foot_y heatmaps plus
scatter of every evaluated individual.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    required = {"foot_x", "foot_y", "fitness", "Forward_Progress", "Fell", "Survival_Time_s"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    return rows


def cell_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        half = max(abs(centers[0]) * 0.05, 0.005)
        return np.array([centers[0] - half, centers[0] + half])
    mid = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def bin_centers(values: np.ndarray, n_bins: int) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    if abs(hi - lo) < 1e-12:
        return np.array([lo])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers


def assign_bin(value: float, centers: np.ndarray) -> int:
    if len(centers) == 1:
        return 0
    return int(np.argmin(np.abs(centers - value)))


def build_offset_grids(
    rows: list[dict],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    fx = np.array([float(r["foot_x"]) for r in rows])
    fy = np.array([float(r["foot_y"]) for r in rows])
    x_centers = bin_centers(fx, n_bins)
    y_centers = bin_centers(fy, n_bins)

    buckets: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        ix = assign_bin(float(row["foot_x"]), x_centers)
        iy = assign_bin(float(row["foot_y"]), y_centers)
        buckets.setdefault((iy, ix), []).append(row)

    shape = (len(y_centers), len(x_centers))
    grids = {
        "mean_fitness": np.full(shape, np.nan),
        "mean_forward": np.full(shape, np.nan),
        "full_surv_pct": np.full(shape, np.nan),
        "count": np.zeros(shape, dtype=float),
    }
    labels: dict[tuple[int, int], str] = {}

    for (iy, ix), group in buckets.items():
        fitness = np.array([float(r["fitness"]) for r in group])
        forward = np.array([float(r["Forward_Progress"]) for r in group])
        full = np.array(
            [
                (not parse_bool(r["Fell"])) and float(r["Survival_Time_s"]) >= 19.9
                for r in group
            ],
            dtype=float,
        )
        grids["mean_fitness"][iy, ix] = float(np.mean(fitness))
        grids["mean_forward"][iy, ix] = float(np.mean(forward))
        grids["full_surv_pct"][iy, ix] = 100.0 * float(np.mean(full))
        grids["count"][iy, ix] = float(len(group))
        labels[(iy, ix)] = (
            f"n={len(group)}\n"
            f"fwd={np.mean(forward):.2f}\n"
            f"fit={np.mean(fitness):.1f}"
        )

    return x_centers, y_centers, grids, labels


def plot_heatmap(
    ax,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    grid: np.ndarray,
    labels: dict,
    title: str,
    cbar_label: str,
    cmap,
    vmin=None,
    vmax=None,
) -> None:
    x_edges = cell_edges(x_centers)
    y_edges = cell_edges(y_centers)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(grid),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidth=0.8,
        shading="flat",
    )
    for (iy, ix), text in labels.items():
        if not math.isfinite(grid[iy, ix]):
            continue
        ax.text(
            x_centers[ix],
            y_centers[iy],
            text,
            ha="center",
            va="center",
            fontsize=6.5,
            color="black",
        )
    ax.set_xlabel("foot_x (m)  [+ inward]")
    ax.set_ylabel("foot_y (m)  [+ forward]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def plot_es_offsets(rows: list[dict], out_path: Path, n_bins: int) -> None:
    x_centers, y_centers, grids, labels = build_offset_grids(rows, n_bins)

    fx = np.array([float(r["foot_x"]) for r in rows])
    fy = np.array([float(r["foot_y"]) for r in rows])
    fitness = np.array([float(r["fitness"]) for r in rows])
    forward = np.array([float(r["Forward_Progress"]) for r in rows])
    gens = np.array([int(float(r.get("Generation", 0))) for r in rows])
    full = np.array(
        [(not parse_bool(r["Fell"])) and float(r["Survival_Time_s"]) >= 19.9 for r in rows],
        dtype=bool,
    )

    cmap_fit = LinearSegmentedColormap.from_list("fit", ["#c62828", "#f9d65c", "#2e7d32"])
    cmap_fwd = LinearSegmentedColormap.from_list("fwd", ["#263238", "#0288d1", "#a5d6a7"])
    cmap_surv = LinearSegmentedColormap.from_list("surv", ["#c62828", "#f9d65c", "#2e7d32"])
    cmap_fit.set_bad("#d0d0d0")
    cmap_fwd.set_bad("#d0d0d0")
    cmap_surv.set_bad("#d0d0d0")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    plot_heatmap(
        axes[0, 0],
        x_centers,
        y_centers,
        grids["mean_forward"],
        labels,
        "Mean forward progress by offset bin",
        "Forward (m)",
        cmap_fwd,
    )
    plot_heatmap(
        axes[0, 1],
        x_centers,
        y_centers,
        grids["mean_fitness"],
        labels,
        "Mean fitness by offset bin",
        "Fitness",
        cmap_fit,
    )
    plot_heatmap(
        axes[1, 0],
        x_centers,
        y_centers,
        grids["full_surv_pct"],
        labels,
        "Full-survival rate by offset bin",
        "Full survivors (%)",
        cmap_surv,
        vmin=0,
        vmax=100,
    )

    ax = axes[1, 1]
    sc = ax.scatter(
        fx[full],
        fy[full],
        c=forward[full],
        s=18 + 4 * (gens[full] / max(gens.max(), 1)),
        cmap=cmap_fwd,
        alpha=0.75,
        edgecolors="none",
        label="full survivor",
    )
    ax.scatter(
        fx[~full],
        fy[~full],
        c="#9e9e9e",
        s=10,
        alpha=0.35,
        edgecolors="none",
        label="fell / short",
    )
    # Mark best-by-fitness and best-by-forward.
    best_fit_i = int(np.argmax(fitness))
    best_fwd_i = int(np.argmax(forward))
    ax.scatter(
        [fx[best_fit_i]],
        [fy[best_fit_i]],
        s=120,
        facecolors="none",
        edgecolors="#c62828",
        linewidths=2.0,
        label=f"best fitness ({fitness[best_fit_i]:.1f})",
    )
    ax.scatter(
        [fx[best_fwd_i]],
        [fy[best_fwd_i]],
        s=120,
        marker="D",
        facecolors="none",
        edgecolors="#1565c0",
        linewidths=2.0,
        label=f"best forward ({forward[best_fwd_i]:.2f} m)",
    )
    ax.axvline(0.0, color="#90a4ae", lw=0.8, ls="--")
    ax.axhline(0.0, color="#90a4ae", lw=0.8, ls="--")
    ax.set_xlabel("foot_x (m)  [+ inward]")
    ax.set_ylabel("foot_y (m)  [+ forward]")
    ax.set_title("All ES individuals (size ~ generation)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Forward (m) [survivors]")

    n = len(rows)
    n_full = int(np.sum(full))
    fig.suptitle(
        f"ES foot/hip offsets  |  N={n}  full_surv={n_full} ({100 * n_full / max(n, 1):.0f}%)  "
        f"gens={gens.min()}–{gens.max()}",
        fontsize=13,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(
        f"Best fitness @ fx={fx[best_fit_i]:.4f} fy={fy[best_fit_i]:.4f} "
        f"fit={fitness[best_fit_i]:.2f} fwd={forward[best_fit_i]:.3f}"
    )
    print(
        f"Best forward @ fx={fx[best_fwd_i]:.4f} fy={fy[best_fwd_i]:.4f} "
        f"fit={fitness[best_fwd_i]:.2f} fwd={forward[best_fwd_i]:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/es/bigfoot_es_history_fwdonly_offsets_long.csv"),
        help="ES history CSV (must include foot_x / foot_y).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/es/es_foot_offsets_plot.png"),
        help="Output figure path.",
    )
    parser.add_argument("--bins", type=int, default=8, help="Bins per offset axis for heatmaps.")
    args = parser.parse_args()

    if not args.csv.exists():
        # Fall back to the shorter offsets run if long file is not ready yet.
        fallback = Path("data/es/bigfoot_es_history_fwdonly_offsets.csv")
        if fallback.exists():
            print(f"{args.csv} missing — using {fallback}")
            args.csv = fallback
        else:
            raise FileNotFoundError(args.csv)

    rows = load_rows(args.csv)
    # Drop instantaneous fails from settle/model bugs (survival ~0) for cleaner maps.
    rows = [r for r in rows if float(r["Survival_Time_s"]) > 0.05]
    if not rows:
        raise ValueError("No usable rows after filtering near-zero survival.")
    plot_es_offsets(rows, args.out, n_bins=max(3, args.bins))


if __name__ == "__main__":
    main()
