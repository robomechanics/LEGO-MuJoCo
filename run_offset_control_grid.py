#!/usr/bin/env python3
"""Bilevel offset × controller robustness grid.

Instead of jointly mutating feet and gait (the ES), this evaluates a frozen
catalog of 5 controllers at every (foot_x, foot_y) cell.

Default catalog: diverse 100%-robust full-survival controllers from walkgait v12
(control genes only; every catalog member is evaluated at every grid cell).

Cell score (not best-of-5):
  occupancy        = k / 5 heading-gated full-survivors
  expected_forward = mean over the 5 of (forward_m if success else 0)

A trial counts as success only if it full-survives AND |heading| <= --heading-max-deg.

Presets:
  full  — broad foot bounds
  zoom  — older occupancy ridge around fy≈−37 mm
  v12   — heading-gated fine grid around the v12 foot island

Optional --cross-mm then re-tests the top cells at ±Δ foot noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import test_sim_sweep as sweep
from run_evolutionary_controller import (
    FIXED_FREQ_HZ,
    OCCUPANCY_ISLAND,
    PARAM_BOUNDS,
    V12_FEET,
    controller_stem,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

CONTROL_KEYS = ("amp_deg", "start_amp_mult", "start_freq_mult", "Kp", "Kd")
OFFSET_BOUNDS = {
    "foot_x": PARAM_BOUNDS["foot_x"],
    "foot_y": PARAM_BOUNDS["foot_y"],
}

# Diverse v12 100%-robust full survivors (control genes only).
CATALOG_CONTROLLERS: tuple[dict, ...] = (
    {
        "id": "C0",
        "label": "v12_fit284",
        "amp_deg": 48.940606327751546,
        "start_amp_mult": 1.2840040496029064,
        "start_freq_mult": 1.958686763739469,
        "Kp": 29.207580386062894,
        "Kd": 8.100165117179767,
    },
    {
        "id": "C1",
        "label": "v12_fit156",
        "amp_deg": 46.68309626523455,
        "start_amp_mult": 1.6196743167720136,
        "start_freq_mult": 2.3165318387658003,
        "Kp": 35.191390494642036,
        "Kd": 9.790587413956134,
    },
    {
        "id": "C2",
        "label": "v12_fit106",
        "amp_deg": 41.24612920077805,
        "start_amp_mult": 1.532032489730025,
        "start_freq_mult": 1.8917278389106895,
        "Kp": 33.21474004998967,
        "Kd": 8.45545965842749,
    },
    {
        "id": "C3",
        "label": "v12_fit96",
        "amp_deg": 36.96260007482137,
        "start_amp_mult": 1.938022232864824,
        "start_freq_mult": 1.1496421684300777,
        "Kp": 43.12892786075382,
        "Kd": 7.968509848005885,
    },
    {
        "id": "C4",
        "label": "v12_fit86",
        "amp_deg": 29.489465277495203,
        "start_amp_mult": 2.019484006696985,
        "start_freq_mult": 1.4281913267253838,
        "Kp": 78.2088834794088,
        "Kd": 7.688199970415928,
    },
)

SEED_OFFSET = dict(V12_FEET)
SNAP_OFFSETS = (SEED_OFFSET, OCCUPANCY_ISLAND)

TRIAL_FIELDNAMES = [
    "phase",
    "cell_ix",
    "cell_iy",
    "foot_x",
    "foot_y",
    "controller_id",
    "controller_label",
    "amp_deg",
    "start_amp_mult",
    "start_freq_mult",
    "Kp",
    "Kd",
    "Fell",
    "Survival_Time_s",
    "Distance_Traversed",
    "Forward_Progress",
    "Backward_Progress",
    "Walked_Backward",
    "Heading_Change_Deg",
    "Alternating_Steps",
    "Min_Swing_Clearance",
    "Slip_Ratio",
    "Gait_Quality_Pass",
    "survived_full",
    "heading_ok",
    "full_survivor",
    "expected_forward_m",
]


def control_only(ctrl: dict) -> dict[str, float]:
    return {key: float(ctrl[key]) for key in CONTROL_KEYS}


def build_catalog() -> list[dict]:
    return [dict(ctrl) for ctrl in CATALOG_CONTROLLERS]


def snap_axis_to_values(centers: np.ndarray, values: list[float]) -> np.ndarray:
    """Replace nearest grid nodes with known offsets so they are evaluated exactly."""
    snapped = np.array(centers, dtype=float)
    used: set[int] = set()
    unique: list[float] = []
    for value in values:
        if any(abs(float(value) - existing) < 1e-9 for existing in unique):
            continue
        unique.append(float(value))
    for value in unique:
        diffs = np.abs(snapped - value)
        for idx in np.argsort(diffs):
            i = int(idx)
            if i not in used:
                snapped[i] = value
                used.add(i)
                break
    return snapped


def make_offset_grid(
    nx: int,
    ny: int,
    fx_bounds: tuple[float, float],
    fy_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(fx_bounds[0], fx_bounds[1], nx)
    ys = np.linspace(fy_bounds[0], fy_bounds[1], ny)
    xs = snap_axis_to_values(xs, [off["foot_x"] for off in SNAP_OFFSETS])
    ys = snap_axis_to_values(ys, [off["foot_y"] for off in SNAP_OFFSETS])
    return xs, ys


def is_full_survivor(fell: bool, survival_s: float) -> bool:
    return (not fell) and survival_s >= sweep.FULL_SURVIVAL_S


def heading_within_limit(heading_deg: float, heading_max_deg: float) -> bool:
    if not np.isfinite(heading_deg):
        return False
    return abs(float(heading_deg)) <= float(heading_max_deg)


def is_success(fell: bool, survival_s: float, heading_deg: float, heading_max_deg: float) -> bool:
    return is_full_survivor(fell, survival_s) and heading_within_limit(heading_deg, heading_max_deg)


def expected_forward(success: bool, forward_m: float) -> float:
    if not success:
        return 0.0
    return float(max(0.0, forward_m))


def score_cell(trials: list[dict]) -> dict:
    n = len(trials)
    k = sum(1 for t in trials if t["full_survivor"])
    forwards = [float(t["expected_forward_m"]) for t in trials]
    survivals = [float(t["Survival_Time_s"]) for t in trials]
    survivor_fwd = [float(t["Forward_Progress"]) for t in trials if t["full_survivor"]]
    return {
        "n_controllers": n,
        "k_full": k,
        "occupancy": k / max(n, 1),
        "expected_forward_m": float(np.mean(forwards)) if forwards else 0.0,
        "mean_survivor_forward_m": float(np.mean(survivor_fwd)) if survivor_fwd else 0.0,
        "min_survival_s": float(np.min(survivals)) if survivals else 0.0,
        "mean_survival_s": float(np.mean(survivals)) if survivals else 0.0,
    }


def trial_key(row: dict) -> tuple:
    return (
        str(row["phase"]),
        str(row["cell_ix"]),
        str(row["cell_iy"]),
        f"{float(row['foot_x']):.6f}",
        f"{float(row['foot_y']):.6f}",
        str(row["controller_id"]),
    )


def load_completed(csv_path: Path) -> dict[tuple, dict]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = {}
    for row in rows:
        row["Fell"] = str(row["Fell"]).strip().lower() in {"1", "true", "yes"}
        row["Walked_Backward"] = str(row.get("Walked_Backward", "")).strip().lower() in {"1", "true", "yes"}
        row["Gait_Quality_Pass"] = str(row.get("Gait_Quality_Pass", "")).strip().lower() in {"1", "true", "yes"}
        row["full_survivor"] = str(row["full_survivor"]).strip().lower() in {"1", "true", "yes"}
        for key in (
            "foot_x",
            "foot_y",
            "amp_deg",
            "start_amp_mult",
            "start_freq_mult",
            "Kp",
            "Kd",
            "Survival_Time_s",
            "Distance_Traversed",
            "Forward_Progress",
            "Backward_Progress",
            "Heading_Change_Deg",
            "Min_Swing_Clearance",
            "Slip_Ratio",
            "expected_forward_m",
        ):
            if key in row and row[key] != "":
                row[key] = float(row[key])
        row["Alternating_Steps"] = int(float(row.get("Alternating_Steps", 0) or 0))
        if "survived_full" in row and row["survived_full"] != "":
            row["survived_full"] = str(row["survived_full"]).strip().lower() in {"1", "true", "yes"}
        if "heading_ok" in row and row["heading_ok"] != "":
            row["heading_ok"] = str(row["heading_ok"]).strip().lower() in {"1", "true", "yes"}
        completed[trial_key(row)] = row
    return completed


def append_trial_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in TRIAL_FIELDNAMES})


def result_to_trial_row(
    *,
    phase: str,
    cell_ix: int,
    cell_iy: int,
    foot_x: float,
    foot_y: float,
    controller: dict,
    result: dict,
    heading_max_deg: float,
) -> dict:
    fell = bool(result["Fell"])
    survival = float(result["Survival_Time_s"])
    forward = float(result["Forward_Progress"])
    heading = float(result.get("Heading_Change_Deg", 0.0))
    survived_full = is_full_survivor(fell, survival)
    heading_ok = heading_within_limit(heading, heading_max_deg)
    success = survived_full and heading_ok
    return {
        "phase": phase,
        "cell_ix": cell_ix,
        "cell_iy": cell_iy,
        "foot_x": float(foot_x),
        "foot_y": float(foot_y),
        "controller_id": controller["id"],
        "controller_label": controller["label"],
        "amp_deg": float(controller["amp_deg"]),
        "start_amp_mult": float(controller["start_amp_mult"]),
        "start_freq_mult": float(controller["start_freq_mult"]),
        "Kp": float(controller["Kp"]),
        "Kd": float(controller["Kd"]),
        "Fell": fell,
        "Survival_Time_s": survival,
        "Distance_Traversed": float(result["Distance_Traversed"]),
        "Forward_Progress": forward,
        "Backward_Progress": float(result["Backward_Progress"]),
        "Walked_Backward": bool(result["Walked_Backward"]),
        "Heading_Change_Deg": heading,
        "Alternating_Steps": int(result.get("Alternating_Steps", 0)),
        "Min_Swing_Clearance": float(result.get("Min_Swing_Clearance", 0.0)),
        "Slip_Ratio": float(result.get("Slip_Ratio", 0.0)) if np.isfinite(float(result.get("Slip_Ratio", 0.0))) else 10.0,
        "Gait_Quality_Pass": bool(result.get("Gait_Quality_Pass", False)),
        "survived_full": survived_full,
        "heading_ok": heading_ok,
        "full_survivor": success,
        "expected_forward_m": expected_forward(success, forward),
    }


def run_one_trial(
    ctx: sweep.SimulationContext,
    controller: dict,
    foot_x: float,
    foot_y: float,
) -> dict:
    params = {
        **control_only(controller),
        "foot_x": float(foot_x),
        "foot_y": float(foot_y),
        "freq_hz": FIXED_FREQ_HZ,
    }
    return sweep.run_single_trial(ctx, params, verbose=False)


def evaluate_offset(
    ctx: sweep.SimulationContext,
    catalog: list[dict],
    foot_x: float,
    foot_y: float,
    *,
    phase: str,
    cell_ix: int,
    cell_iy: int,
    completed: dict[tuple, dict],
    csv_path: Path,
    heading_max_deg: float,
) -> list[dict]:
    trials = []
    for controller in catalog:
        key = trial_key(
            {
                "phase": phase,
                "cell_ix": cell_ix,
                "cell_iy": cell_iy,
                "foot_x": foot_x,
                "foot_y": foot_y,
                "controller_id": controller["id"],
            }
        )
        if key in completed:
            row = completed[key]
            # Re-apply heading gate so resumed rows match this run's threshold.
            survived = bool(row.get("survived_full", row.get("full_survivor", False)))
            heading_ok = heading_within_limit(float(row["Heading_Change_Deg"]), heading_max_deg)
            row["survived_full"] = survived
            row["heading_ok"] = heading_ok
            row["full_survivor"] = survived and heading_ok
            row["expected_forward_m"] = expected_forward(row["full_survivor"], float(row["Forward_Progress"]))
            trials.append(row)
            continue
        result = run_one_trial(ctx, controller, foot_x, foot_y)
        row = result_to_trial_row(
            phase=phase,
            cell_ix=cell_ix,
            cell_iy=cell_iy,
            foot_x=foot_x,
            foot_y=foot_y,
            controller=controller,
            result=result,
            heading_max_deg=heading_max_deg,
        )
        append_trial_row(csv_path, row)
        completed[key] = row
        trials.append(row)
        if row["full_survivor"]:
            flag = "OK"
        elif row["survived_full"]:
            flag = "TURN"
        else:
            flag = "FAIL"
        print(
            f"    {controller['id']} {controller['label']:16s} {flag:4s} "
            f"surv={row['Survival_Time_s']:.1f}s "
            f"fwd={row['Forward_Progress']:.3f}m "
            f"head={row['Heading_Change_Deg']:.0f}deg "
            f"gait={row['Gait_Quality_Pass']}",
            flush=True,
        )
    return trials


def cell_payload(ix: int, iy: int, foot_x: float, foot_y: float, trials: list[dict]) -> dict:
    summary = score_cell(trials)
    return {
        "cell_ix": ix,
        "cell_iy": iy,
        "foot_x": float(foot_x),
        "foot_y": float(foot_y),
        **summary,
        "controllers": [
            {
                "id": t["controller_id"],
                "label": t["controller_label"],
                "full_survivor": t["full_survivor"],
                "survival_s": t["Survival_Time_s"],
                "forward_m": t["Forward_Progress"],
                "heading_deg": t["Heading_Change_Deg"],
                "heading_ok": bool(t.get("heading_ok", False)),
                "survived_full": bool(t.get("survived_full", t["full_survivor"])),
                "gait_pass": t["Gait_Quality_Pass"],
            }
            for t in trials
        ],
    }


def rank_cells(cells: list[dict]) -> list[dict]:
    return sorted(
        cells,
        key=lambda c: (c["occupancy"], c["expected_forward_m"], c["min_survival_s"]),
        reverse=True,
    )


def offset_neighbors(foot_x: float, foot_y: float, deltas_m: list[float]) -> list[tuple[float, float, str]]:
    fx_lo, fx_hi = OFFSET_BOUNDS["foot_x"]
    fy_lo, fy_hi = OFFSET_BOUNDS["foot_y"]
    neighbors = []
    seen = {(round(foot_x, 6), round(foot_y, 6))}
    for delta in deltas_m:
        for axis, sign in (("x", 1.0), ("x", -1.0), ("y", 1.0), ("y", -1.0)):
            nx = foot_x + (sign * delta if axis == "x" else 0.0)
            ny = foot_y + (sign * delta if axis == "y" else 0.0)
            nx = float(np.clip(nx, fx_lo, fx_hi))
            ny = float(np.clip(ny, fy_lo, fy_hi))
            tag = f"{axis}{'+' if sign > 0 else '-'}{delta * 1000:.0f}mm"
            key = (round(nx, 6), round(ny, 6))
            if key in seen:
                continue
            seen.add(key)
            neighbors.append((nx, ny, tag))
    return neighbors


def cell_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        half = max(abs(centers[0]) * 0.05, 0.005)
        return np.array([centers[0] - half, centers[0] + half])
    mid = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def plot_grid(cells: list[dict], xs: np.ndarray, ys: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    occ = np.full((len(ys), len(xs)), np.nan)
    fwd = np.full((len(ys), len(xs)), np.nan)
    lookup = {(c["cell_ix"], c["cell_iy"]): c for c in cells}
    for iy in range(len(ys)):
        for ix in range(len(xs)):
            cell = lookup.get((ix, iy))
            if cell is None:
                continue
            occ[iy, ix] = cell["occupancy"]
            fwd[iy, ix] = cell["expected_forward_m"]

    x_edges = cell_edges(xs) * 1000.0
    y_edges = cell_edges(ys) * 1000.0
    cmap_occ = LinearSegmentedColormap.from_list("occ", ["#eceff1", "#1565c0"])
    cmap_fwd = LinearSegmentedColormap.from_list("fwd", ["#fff8e1", "#ef6c00"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    mesh0 = axes[0].pcolormesh(x_edges, y_edges, occ, cmap=cmap_occ, vmin=0.0, vmax=1.0)
    mesh1 = axes[1].pcolormesh(x_edges, y_edges, fwd, cmap=cmap_fwd, vmin=0.0)
    fig.colorbar(mesh0, ax=axes[0], fraction=0.046, pad=0.04, label="k / 5 full survivors")
    fig.colorbar(mesh1, ax=axes[1], fraction=0.046, pad=0.04, label="Expected forward (m)")

    seed_x = SEED_OFFSET["foot_x"] * 1000.0
    seed_y = SEED_OFFSET["foot_y"] * 1000.0
    occ_x = OCCUPANCY_ISLAND["foot_x"] * 1000.0
    occ_y = OCCUPANCY_ISLAND["foot_y"] * 1000.0
    ranked = rank_cells(cells)
    best = ranked[0] if ranked else None
    for ax, title in (
        (axes[0], "Basin occupancy (k/5, heading-gated)"),
        (axes[1], "Expected forward (0 if not success)"),
    ):
        ax.axvline(0.0, color="#90a4ae", lw=0.8, ls="--")
        ax.axhline(0.0, color="#90a4ae", lw=0.8, ls="--")
        ax.scatter([seed_x], [seed_y], s=90, facecolors="none", edgecolors="#c62828", linewidths=1.8, label="v10 seed offset")
        ax.scatter([occ_x], [occ_y], s=70, marker="s", facecolors="none", edgecolors="#6a1b9a", linewidths=1.8, label="5/5 island")
        if best is not None:
            ax.scatter(
                [best["foot_x"] * 1000.0],
                [best["foot_y"] * 1000.0],
                s=70,
                marker="D",
                facecolors="none",
                edgecolors="#2e7d32",
                linewidths=1.8,
                label="best cell",
            )
        ax.set_xlabel("foot_x (mm)")
        ax.set_ylabel("foot_y (mm)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8)

    n_cells = len(cells)
    wide = sum(1 for c in cells if c["occupancy"] >= 0.8)
    fig.suptitle(
        f"Offset × 5-controller grid  |  {n_cells} cells  "
        f"{wide} with ≥4/5 heading-gated survivors  |  freq={FIXED_FREQ_HZ:.3f} Hz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_catalog(catalog: list[dict]) -> None:
    print("Frozen controller catalog (same 5 at every offset):")
    for ctrl in catalog:
        print(
            f"  {ctrl['id']} {ctrl['label']:16s}  "
            f"amp={ctrl['amp_deg']:.2f} sam={ctrl['start_amp_mult']:.3f} "
            f"sfm={ctrl['start_freq_mult']:.3f} Kp={ctrl['Kp']:.1f} Kd={ctrl['Kd']:.1f}"
        )


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_cross_mm(raw: str | None) -> list[float]:
    if not raw:
        return []
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def parse_mm_pair(raw: str) -> tuple[float, float]:
    lo_s, hi_s = raw.split(",", 1)
    lo_m = float(lo_s) / 1000.0
    hi_m = float(hi_s) / 1000.0
    return (min(lo_m, hi_m), max(lo_m, hi_m))


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    zoom = args.preset == "zoom"
    v12 = args.preset == "v12"
    fine = zoom or v12
    if args.nx is None:
        args.nx = 9 if v12 else 8
    if args.ny is None:
        args.ny = 9 if v12 else (7 if zoom else 8)
    if args.fx_mm:
        args.fx_bounds = parse_mm_pair(args.fx_mm)
    elif v12:
        # v12 promising foot region (mm): roughly [-10, 30] × [-45, -5]
        args.fx_bounds = (-0.010, 0.030)
    elif zoom:
        args.fx_bounds = (-0.015, 0.020)
    else:
        args.fx_bounds = OFFSET_BOUNDS["foot_x"]
    if args.fy_mm:
        args.fy_bounds = parse_mm_pair(args.fy_mm)
    elif v12:
        args.fy_bounds = (-0.045, -0.005)
    elif zoom:
        args.fy_bounds = (-0.050, -0.020)
    else:
        args.fy_bounds = OFFSET_BOUNDS["foot_y"]
    if args.heading_max_deg is None:
        args.heading_max_deg = 60.0 if fine else 180.0
    if fine and not args.cross_mm:
        args.cross_mm = "0.003"
    if fine and args.top_k == 8:
        args.top_k = 5 if v12 else 3
    if zoom and args.trials_csv == Path("data/es/offset_control_grid_trials.csv"):
        args.trials_csv = Path("data/es/offset_control_grid_zoom_trials.csv")
    if zoom and args.summary_json == Path("data/es/offset_control_grid_summary.json"):
        args.summary_json = Path("data/es/offset_control_grid_zoom_summary.json")
    if zoom and args.plot_png == Path("data/es/offset_control_grid.png"):
        args.plot_png = Path("data/es/offset_control_grid_zoom.png")
    if v12 and args.trials_csv == Path("data/es/offset_control_grid_trials.csv"):
        args.trials_csv = Path("data/es/offset_control_grid_v12_trials.csv")
    if v12 and args.summary_json == Path("data/es/offset_control_grid_summary.json"):
        args.summary_json = Path("data/es/offset_control_grid_v12_summary.json")
    if v12 and args.plot_png == Path("data/es/offset_control_grid.png"):
        args.plot_png = Path("data/es/offset_control_grid_v12.png")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("full", "zoom", "v12"),
        default="full",
        help="zoom: old occupancy ridge; v12: walkgait-v12 foot island.",
    )
    parser.add_argument("--nx", type=int, default=None, help="Grid cells along foot_x.")
    parser.add_argument("--ny", type=int, default=None, help="Grid cells along foot_y.")
    parser.add_argument("--fx-mm", default="", help="foot_x range in mm, e.g. -15,20.")
    parser.add_argument("--fy-mm", default="", help="foot_y range in mm, e.g. -50,-20.")
    parser.add_argument(
        "--heading-max-deg",
        type=float,
        default=None,
        help="Success requires |heading| <= this (full default 180, zoom 60).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Kept for resume metadata.")
    parser.add_argument("--top-k", type=int, default=8, help="Cells to print / send to --cross-mm.")
    parser.add_argument(
        "--cross-mm",
        default="",
        help="Optional Phase B: comma-separated foot perturbations in meters, e.g. 0.003",
    )
    parser.add_argument("--model-xml", default=os.environ.get("MODEL_XML_PATH", "Bigfoot/scene.xml"))
    parser.add_argument("--trials-csv", type=Path, default=Path("data/es/offset_control_grid_trials.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/es/offset_control_grid_summary.json"))
    parser.add_argument("--plot-png", type=Path, default=Path("data/es/offset_control_grid.png"))
    parser.add_argument("--dry-run", action="store_true", help="Print catalog + grid, do not simulate.")
    parser.add_argument("--plot-only", action="store_true", help="Rebuild the heatmap from an existing summary JSON.")
    return apply_preset(parser.parse_args())


def main() -> None:
    args = parse_args()
    catalog = build_catalog()
    xs, ys = make_offset_grid(args.nx, args.ny, args.fx_bounds, args.fy_bounds)
    n_grid = args.nx * args.ny * len(catalog)
    cross_deltas = parse_cross_mm(args.cross_mm)
    heading_max = float(args.heading_max_deg)

    print_catalog(catalog)
    print(
        f"\nGrid: {args.nx}×{args.ny} = {args.nx * args.ny} offsets  "
        f"× {len(catalog)} controllers = {n_grid} trials"
    )
    print(f"  preset={args.preset}  heading_max={heading_max:.0f}deg")
    print(f"  fx (mm): {np.round(xs * 1000, 1).tolist()}")
    print(f"  fy (mm): {np.round(ys * 1000, 1).tolist()}")
    seed_ix = int(np.argmin(np.abs(xs - SEED_OFFSET["foot_x"])))
    seed_iy = int(np.argmin(np.abs(ys - SEED_OFFSET["foot_y"])))
    occ_ix = int(np.argmin(np.abs(xs - OCCUPANCY_ISLAND["foot_x"])))
    occ_iy = int(np.argmin(np.abs(ys - OCCUPANCY_ISLAND["foot_y"])))
    print(
        f"  seed snapped to ({seed_ix},{seed_iy}) "
        f"fx={xs[seed_ix]*1000:.1f}mm fy={ys[seed_iy]*1000:.1f}mm"
    )
    print(
        f"  occupancy island snapped to ({occ_ix},{occ_iy}) "
        f"fx={xs[occ_ix]*1000:.1f}mm fy={ys[occ_iy]*1000:.1f}mm"
    )
    if cross_deltas:
        print(f"  Phase B cross-mm: {[d * 1000 for d in cross_deltas]} mm on top-{args.top_k} cells")

    if args.dry_run:
        return

    if args.plot_only:
        payload = json.loads(args.summary_json.read_text())
        cells = payload["cells"]
        xs = np.array(payload["grid_foot_x"], dtype=float)
        ys = np.array(payload["grid_foot_y"], dtype=float)
        plot_grid(cells, xs, ys, args.plot_png)
        print(f"Wrote {args.plot_png}")
        return

    completed = load_completed(args.trials_csv)
    if completed:
        print(f"Resuming: {len(completed)} trials already in {args.trials_csv}")

    ctx = sweep.load_simulation(args.model_xml)
    cells: list[dict] = []
    total_grid = args.nx * args.ny

    for iy, fy in enumerate(ys):
        for ix, fx in enumerate(xs):
            cell_n = iy * args.nx + ix + 1
            print(
                f"\n[{cell_n}/{total_grid}] offset fx={fx*1000:.1f}mm fy={fy*1000:.1f}mm",
                flush=True,
            )
            trials = evaluate_offset(
                ctx,
                catalog,
                fx,
                fy,
                phase="grid",
                cell_ix=ix,
                cell_iy=iy,
                completed=completed,
                csv_path=args.trials_csv,
                heading_max_deg=heading_max,
            )
            cell = cell_payload(ix, iy, fx, fy, trials)
            cells.append(cell)
            print(
                f"  occupancy={cell['k_full']}/{cell['n_controllers']} "
                f"({cell['occupancy']:.0%}) heading<={heading_max:.0f}deg  "
                f"E[fwd]={cell['expected_forward_m']:.3f}m  "
                f"min_surv={cell['min_survival_s']:.1f}s",
                flush=True,
            )
            payload = {
                "fixed_frequency_hz": FIXED_FREQ_HZ,
                "seed": args.seed,
                "heading_max_deg": heading_max,
                "catalog": catalog,
                "grid_foot_x": xs.tolist(),
                "grid_foot_y": ys.tolist(),
                "n_cells": len(cells),
                "cells": cells,
                "ranked": rank_cells(cells)[: args.top_k],
            }
            write_summary(args.summary_json, payload)

    ranked = rank_cells(cells)
    print("\n=== Top offset cells (heading-gated occupancy, then expected forward) ===")
    for rank, cell in enumerate(ranked[: args.top_k], 1):
        print(
            f"  #{rank:02d} fx={cell['foot_x']*1000:+6.1f}mm fy={cell['foot_y']*1000:+6.1f}mm  "
            f"k={cell['k_full']}/{cell['n_controllers']}  "
            f"E[fwd]={cell['expected_forward_m']:.3f}m  "
            f"min_surv={cell['min_survival_s']:.1f}s"
        )

    cross_cells: list[dict] = []
    if cross_deltas:
        print(
            f"\nPhase B: ±foot noise {[d * 1000 for d in cross_deltas]} mm "
            f"on top {args.top_k} cells"
        )
        for parent in ranked[: args.top_k]:
            neighbors = offset_neighbors(parent["foot_x"], parent["foot_y"], cross_deltas)
            for nx, ny, tag in neighbors:
                print(
                    f"\n  cross {tag} from "
                    f"({parent['foot_x']*1000:.1f},{parent['foot_y']*1000:.1f}) mm",
                    flush=True,
                )
                trials = evaluate_offset(
                    ctx,
                    catalog,
                    nx,
                    ny,
                    phase="cross",
                    cell_ix=parent["cell_ix"],
                    cell_iy=parent["cell_iy"],
                    completed=completed,
                    csv_path=args.trials_csv,
                    heading_max_deg=heading_max,
                )
                entry = cell_payload(parent["cell_ix"], parent["cell_iy"], nx, ny, trials)
                entry["parent_foot_x"] = parent["foot_x"]
                entry["parent_foot_y"] = parent["foot_y"]
                entry["perturbation"] = tag
                cross_cells.append(entry)
                print(
                    f"    occupancy={entry['k_full']}/{entry['n_controllers']} "
                    f"E[fwd]={entry['expected_forward_m']:.3f}m",
                    flush=True,
                )

    plot_grid(cells, xs, ys, args.plot_png)

    # Archive catalog controllers under the fixed naming scheme.
    best_dir = args.summary_json.parent / f"{args.summary_json.stem}_controllers"
    best_dir.mkdir(parents=True, exist_ok=True)
    named: list[str] = []
    for ctrl in catalog:
        stem = controller_stem(ctrl)
        path = best_dir / f"{stem}.json"
        path.write_text(
            json.dumps(
                {
                    "label": stem,
                    "catalog_id": ctrl["id"],
                    "catalog_label": ctrl["label"],
                    "freq_hz": FIXED_FREQ_HZ,
                    "genes": {**control_only(ctrl), **dict(SEED_OFFSET)},
                    "note": "Control genes from catalog; feet are the v12 seed snap point.",
                },
                indent=2,
            )
        )
        named.append(str(path))
        print(f"  saved catalog controller {path.name}")

    payload = {
        "fixed_frequency_hz": FIXED_FREQ_HZ,
        "seed": args.seed,
        "heading_max_deg": heading_max,
        "catalog": catalog,
        "grid_foot_x": xs.tolist(),
        "grid_foot_y": ys.tolist(),
        "n_grid_trials": n_grid,
        "n_cells": len(cells),
        "cells": cells,
        "ranked": ranked[: args.top_k],
        "cross_mm": cross_deltas,
        "cross_cells": cross_cells,
        "named_controllers": named,
        "scoring": {
            "occupancy": (
                f"k/5 success: full-survive AND |heading| <= {heading_max:.0f} deg"
            ),
            "expected_forward_m": "mean over catalog of (forward if success else 0)",
        },
    }
    write_summary(args.summary_json, payload)
    print(f"\nSaved trials to {args.trials_csv}")
    print(f"Saved summary to {args.summary_json}")
    print(f"Saved plot to {args.plot_png}")


if __name__ == "__main__":
    main()
