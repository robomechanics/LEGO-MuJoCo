"""
Reads sweep_actuation_best_per_bin.csv (from gen_sweep_actuation.py) and
renders a heatmap over the (Frequency_Hz, Amplitude_Deg) grid: cell color =
average velocity of the best startup combo found for that bin (sequential --
light = slow, dark = fast); bins where every trial fell (no trial survived
the full run) are hatched gray, since average velocity isn't meaningful for
a crash.

Kept separate from the sweep script so the plot can be restyled without
re-running the (up to 6+ hour) sweep. Uses pcolormesh-style Rectangle
patches rather than plot_foot_offset_settling.py's per-cell text-box style,
since the actuation grid can run 20x20 to 40x40+ bins -- too dense for
annotated boxes to stay readable.
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

CSV_FILE = "results/sweep_actuation_best_per_bin.csv"
OUT_FILE = "results/actuation_sweep_grid.png"

# Single-hue sequential ramp (light -> dark blue), from the dataviz skill's
# validated default palette -- magnitude encoding, not a rainbow.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)

NO_SURVIVOR_COLOR = "#c9ccd1"


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["Amplitude_Deg"] = float(row["Amplitude_Deg"])
            row["Frequency_Hz"] = float(row["Frequency_Hz"])
            row["Fell"] = row["Fell"] == "True"
            row["Passed"] = row["Passed"] == "True"
            row["Distance_Traversed"] = float(row["Distance_Traversed"])
            row["CoT"] = float(row["CoT"])
            av = row.get("Avg_Velocity", "")
            row["Avg_Velocity"] = float(av) if av not in ("", "None", "nan") else None
            rows.append(row)
    return rows


def main():
    rows = load_rows(CSV_FILE)

    amp_vals  = sorted({r["Amplitude_Deg"] for r in rows})
    freq_vals = sorted({r["Frequency_Hz"] for r in rows})
    nx, ny = len(freq_vals), len(amp_vals)

    grid = {(r["Frequency_Hz"], r["Amplitude_Deg"]): r for r in rows}

    velocities = [r["Avg_Velocity"] for r in rows if r["Avg_Velocity"] is not None]
    vmin, vmax = (0.0, max(velocities)) if velocities else (0.0, 1.0)

    # Cap the grid's own figsize growth for very dense sweeps -- readability
    # comes from the colorbar/gradient at that point, not cell-by-cell detail.
    cell_size = max(0.14, min(0.4, 14.0 / max(nx, ny)))
    fig, ax = plt.subplots(figsize=(cell_size * nx + 2.2, cell_size * ny + 2))

    for xi, freq in enumerate(freq_vals):
        for yi, amp in enumerate(amp_vals):
            r = grid.get((freq, amp))
            if r is None:
                continue

            if r["Avg_Velocity"] is None:
                ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=NO_SURVIVOR_COLOR,
                                        edgecolor="none", hatch="///" if nx <= 25 else None))
                continue

            norm = (r["Avg_Velocity"] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            color = SEQUENTIAL_BLUE(np.clip(norm, 0.0, 1.0))
            ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=color, edgecolor="none"))

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    # Thin the tick labels for dense grids so they don't overlap.
    tick_stride = max(1, nx // 12)
    ax.set_xticks([i + 0.5 for i in range(0, nx, tick_stride)])
    ax.set_xticklabels([f"{v:.2f}" for v in freq_vals[::tick_stride]], rotation=45, ha="right")
    tick_stride_y = max(1, ny // 12)
    ax.set_yticks([i + 0.5 for i in range(0, ny, tick_stride_y)])
    ax.set_yticklabels([f"{v:.1f}" for v in amp_vals[::tick_stride_y]])

    ax.set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Amplitude (deg)", fontsize=11, fontweight="bold")

    total = len(rows)
    survived = sum(1 for r in rows if r["Avg_Velocity"] is not None)
    ax.set_title(
        f"Actuation sweep -- best startup combo per amp/freq bin\n"
        f"({survived}/{total} bins had a surviving trial; gray = all fell)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL_BLUE, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Average velocity (m/s) -- best surviving combo; higher = better", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=200)
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
