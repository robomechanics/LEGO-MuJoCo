"""
Reads sweep_foot_offset_settling.csv (from gen_sweep_foot_offset_settling.py)
and renders a 9x9 grid: x-axis = foot_x, y-axis = foot_y, one annotated box
per (foot_x, foot_y) bin. Box background = Tilt_Geodesic_Deg (full-orientation
deviation from the (0,0) baseline); box text = pitch diff, settle frequency,
settle time, and the raw averaged quaternion (w,x,y,z) from phase 2. Fallen
trials are hatched and labeled distinctly.

Kept separate from the sweep script so the plot can be restyled without
re-running the ~30s x 82-trial sweep.
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from sim_common import quat_to_rpy

CSV_FILE = "results/sweep_foot_offset_settling.csv"
OUT_FILE = "results/foot_offset_settling_grid.png"

# Single-hue sequential ramp (light -> dark blue), from the dataviz skill's
# validated default palette -- magnitude encoding, not a rainbow.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)

FELL_COLOR = "#c9ccd1"


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["Foot_X"] = float(row["Foot_X"])
            row["Foot_Y"] = float(row["Foot_Y"])
            row["Fell"] = row["Fell"] == "True"
            for key in ("Avg_Quat_W", "Avg_Quat_X", "Avg_Quat_Y", "Avg_Quat_Z",
                        "Pitch_Diff_Deg", "Tilt_Geodesic_Deg", "Settle_Freq_Hz", "Settle_Time_S"):
                val = row[key]
                row[key] = float(val) if val not in ("", "nan") else float("nan")
            rows.append(row)
    return rows


def main():
    rows = load_rows(CSV_FILE)

    # Baseline is stored as its own row at (0,0); grid rows are everything else.
    grid_rows = [r for r in rows if not (r["Foot_X"] == 0.0 and r["Foot_Y"] == 0.0)]
    baseline = next(r for r in rows if r["Foot_X"] == 0.0 and r["Foot_Y"] == 0.0)

    foot_x_vals = sorted({r["Foot_X"] for r in grid_rows})
    foot_y_vals = sorted({r["Foot_Y"] for r in grid_rows})
    nx, ny = len(foot_x_vals), len(foot_y_vals)

    grid = {(r["Foot_X"], r["Foot_Y"]): r for r in grid_rows}

    finite_geo = [r["Tilt_Geodesic_Deg"] for r in grid_rows if not r["Fell"]]
    vmin, vmax = 0.0, max(finite_geo) if finite_geo else 1.0

    fig, ax = plt.subplots(figsize=(1.9 * nx + 2, 1.9 * ny + 2))

    for xi, fx in enumerate(foot_x_vals):
        for yi, fy in enumerate(foot_y_vals):
            r = grid[(fx, fy)]

            if r["Fell"]:
                ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=FELL_COLOR,
                                        edgecolor="white", linewidth=2, hatch="///"))
                ax.text(xi + 0.5, yi + 0.5, "FELL", ha="center", va="center",
                         fontsize=10, fontweight="bold", color="#3a3d42")
                continue

            norm = (r["Tilt_Geodesic_Deg"] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            color = SEQUENTIAL_BLUE(norm)
            ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=color,
                                    edgecolor="white", linewidth=2))

            # Text color flips to white on dark (high-magnitude) cells for contrast.
            text_color = "white" if norm > 0.55 else "#1c1f24"
            quat_line = (f"q=({r['Avg_Quat_W']:+.2f},{r['Avg_Quat_X']:+.2f},"
                         f"{r['Avg_Quat_Y']:+.2f},{r['Avg_Quat_Z']:+.2f})")
            label = (f"pitch {r['Pitch_Diff_Deg']:+.1f}°\n"
                     f"{r['Settle_Freq_Hz']:.2f} Hz\n"
                     f"{r['Settle_Time_S']:.1f} s\n"
                     f"{quat_line}")
            ax.text(xi + 0.5, yi + 0.5, label, ha="center", va="center",
                     fontsize=8, color=text_color, linespacing=1.7, family="monospace")

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xticks([i + 0.5 for i in range(nx)])
    ax.set_xticklabels([f"{v:+.4f}" for v in foot_x_vals], rotation=45, ha="right")
    ax.set_yticks([i + 0.5 for i in range(ny)])
    ax.set_yticklabels([f"{v:+.4f}" for v in foot_y_vals])
    ax.set_xlabel("foot_x offset (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("foot_y offset (m)", fontsize=11, fontweight="bold")
    baseline_q = np.array([baseline["Avg_Quat_W"], baseline["Avg_Quat_X"],
                            baseline["Avg_Quat_Y"], baseline["Avg_Quat_Z"]])
    baseline_pitch_deg = float(np.rad2deg(quat_to_rpy(baseline_q)[0][1]))
    ax.set_title(
        "Settling behavior vs. foot placement\n"
        f"(baseline @ (0,0): pitch {baseline_pitch_deg:+.1f}°, "
        f"settle {baseline['Settle_Time_S']:.1f}s, {baseline['Settle_Freq_Hz']:.2f}Hz, "
        f"q=({baseline_q[0]:+.2f},{baseline_q[1]:+.2f},{baseline_q[2]:+.2f},{baseline_q[3]:+.2f}))",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL_BLUE,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Tilt deviation from baseline (deg, geodesic)", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=200)
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
