"""
xy_scale_sweep.py

Grid sweep over (x_scale, y_scale) pairs.
At each pair, a 2-D CMA-ES independently optimizes (hip_omega, leg_amp_deg).
z_scale is held fixed throughout.

Outputs:
  xy_scale_sweep_results.pkl   -- full data
  xy_scale_sweep_heatmap.png   -- best speed on the x/y grid
  xy_scale_sweep_bar.png       -- ranked bar chart of best speed per pair,
                                  annotated with winning omega and amp
"""

import copy
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from src.sim import ProgressCallback
from src.rduplo import Duplo
from src.footscalesweep import CMAES

# ─── Outer grid ───────────────────────────────────────────────────────────────
SWEEP_X = [0.8, 1.0, 1.3, 1.7, 2.0]   # x_scale values
SWEEP_Y = [0.8, 0.9, 1.0, 1.2, 1.5]   # y_scale values
Z_SCALE = 1.0                           # held fixed

# ─── Inner CMA-ES (2-D: omega, amp) ──────────────────────────────────────────
HZ_TO_RAD    = 2.0 * np.pi
INNER_LOWS   = np.array([0.40 * HZ_TO_RAD, 25.0])
INNER_HIGHS  = np.array([0.80 * HZ_TO_RAD, 45.0])
INNER_X0     = np.array([0.60 * HZ_TO_RAD, 32.0])   # neutral mid-range start
INNER_SIGMA0 = np.array([0.35, 3.0])
INNER_POP    = 8
INNER_GENS   = 5

# ─── Objective weights ────────────────────────────────────────────────────────
W_SPEED      = 1.00
W_YAW_DRIFT  = 0.012
W_YAW_RATE   = 0.020
PENALTY_FALL = 1.00
PENALTY_SWAY = 0.05

# ─── I/O ──────────────────────────────────────────────────────────────────────
RESULTS_PKL = "xy_scale_sweep_results.pkl"
OUT_PREFIX  = "xy_scale_sweep"

_DEFAULT_X_MM = 14.3
_DEFAULT_Y_MM = 9.7


# ─── Sim helpers ──────────────────────────────────────────────────────────────

def make_args_base() -> dict:
    x = _DEFAULT_X_MM * 1e-3
    y = _DEFAULT_Y_MM * 1e-3
    return {
        "name": "xy-scale-sweep",
        "robot_dir": "./robots",
        "sim_time": 20.0,
        "video_dir": "./videos",
        "video_fps": 30,
        "gui": False,
        "record": False,
        "ctrl_dict": {
            "Kp": 20,
            "Kd": 12,
            "leg_amp_deg": 30.0,
            "hip_omega": 0.60 * HZ_TO_RAD,
        },
        "design_params": {
            "geom_pos_offset": {
                "RightFoot": [ x, -y, 0.0],
                "LeftFoot":  [-x, -y, 0.0],
                "hip_rod_1": [0, 0, 0], "hip_rod_2": [0, 0, 0],
                "leg_rod_1": [0, 0, 0], "leg_rod_2": [0, 0, 0],
                "motor_part1_1": [0, 0, 0], "motor_part2_1": [0, 0, 0],
                "motor_part3_1": [0, 0, 0],
                "arm_rod_1": [0, 0, 0], "arm_rod_2": [0, 0, 0],
                "battery_1": [0, 0, 0], "battery_2": [0, 0, 0],
            },
            "mesh_scale": {
                "RightFoot": [1.0, 1.0, 1.0],
                "LeftFoot":  [1.0, 1.0, 1.0],
                "leg_rod":   [1.0, 1.0, 1.0],
            },
            "body_quat": {
                "motor": [0.98074, -0.19443, 0.00070, -0.01876],
            },
        },
    }


def apply_scales(args: dict, sx: float, sy: float, sz: float) -> None:
    s = [float(sx), float(sy), float(sz)]
    args["design_params"]["mesh_scale"]["RightFoot"] = s[:]
    args["design_params"]["mesh_scale"]["LeftFoot"]  = s[:]


def correct_args_quats(args: dict, avg_quat: np.ndarray) -> None:
    for part in args["design_params"]["body_quat"]:
        args["design_params"]["body_quat"][part] = list(avg_quat)


def stabilise(base_args: dict) -> tuple[bool, np.ndarray]:
    args = copy.deepcopy(base_args)
    args["ctrl_dict"]["leg_amp_deg"] = 0.0
    args["sim_time"] = 8.0
    robot = Duplo(args)
    robot.run_sim(callbacks={"progress_bar": ProgressCallback(args["sim_time"]).update})
    swayed = bool(robot.sway)
    q = robot.mean_quat.copy() / np.linalg.norm(robot.mean_quat)
    robot.close()
    return swayed, q


def run_walking_sim(args: dict) -> dict:
    robot = Duplo(args)
    robot.run_sim(callbacks={"progress_bar": ProgressCallback(args["sim_time"]).update})
    metrics = copy.deepcopy(robot.walk_metrics)
    robot.close()
    return metrics


def safe_run(args: dict) -> dict:
    try:
        return run_walking_sim(args)
    except Exception as exc:
        print(f"  [SIM ERROR] {exc}")
        return {"fwd_speed_ms": 0.0, "yaw_drift_deg": 0.0,
                "yaw_rate_deg_per_s": 0.0, "fell": True}


def objective(metrics: dict, swayed: bool) -> float:
    return (
        W_SPEED      * float(metrics.get("fwd_speed_ms", 0.0))
        - W_YAW_DRIFT  * abs(float(metrics.get("yaw_drift_deg", 0.0)))
        - W_YAW_RATE   * abs(float(metrics.get("yaw_rate_deg_per_s", 0.0)))
        - PENALTY_FALL * float(metrics.get("fell", True))
        - PENALTY_SWAY * float(swayed)
    )


# ─── Inner optimisation ───────────────────────────────────────────────────────

def evaluate_inner(theta2d: np.ndarray, sx: float, sy: float, sz: float) -> dict:
    """Evaluate one (omega, amp) candidate with fixed foot scales."""
    theta2d = np.clip(theta2d.astype(float), INNER_LOWS, INNER_HIGHS)
    args = make_args_base()
    args["ctrl_dict"]["hip_omega"]   = float(theta2d[0])
    args["ctrl_dict"]["leg_amp_deg"] = float(theta2d[1])
    apply_scales(args, sx, sy, sz)

    swayed, q = stabilise(args)
    if swayed:
        correct_args_quats(args, q)

    metrics = safe_run(copy.deepcopy(args))
    score   = objective(metrics, swayed)
    return {
        "theta": theta2d.copy(),
        "hip_omega":   float(theta2d[0]),
        "leg_amp_deg": float(theta2d[1]),
        "score":       score,
        "fwd_speed_ms":       float(metrics.get("fwd_speed_ms", 0.0)),
        "yaw_drift_deg":      float(metrics.get("yaw_drift_deg", 0.0)),
        "yaw_rate_deg_per_s": float(metrics.get("yaw_rate_deg_per_s", 0.0)),
        "fell":        bool(metrics.get("fell", True)),
        "swayed":      bool(swayed),
    }


def optimise_actuation(sx: float, sy: float, sz: float) -> dict:
    """Run the inner CMA-ES for a fixed (sx, sy, sz) and return the best record."""
    print(f"\n  [inner CMA-ES] x={sx:.2f}  y={sy:.2f}  z={sz:.2f}")
    es = CMAES(x0=INNER_X0, sigma0=INNER_SIGMA0,
               lows=INNER_LOWS, highs=INNER_HIGHS, popsize=INNER_POP)

    # baseline at X0
    base = evaluate_inner(INNER_X0, sx, sy, sz)
    best = base
    print(f"    base  score={base['score']:.4f}  speed={base['fwd_speed_ms']:.4f} m/s")

    for gen in range(INNER_GENS):
        xs, _ = es.ask()
        recs, fitness = [], []
        for x in xs:
            r = evaluate_inner(x, sx, sy, sz)
            recs.append(r)
            fitness.append(-r["score"])
            if r["score"] > best["score"]:
                best = r
        es.tell(np.array([r["theta"] for r in recs], dtype=float),
                np.array(fitness, dtype=float))
        gen_best = max(recs, key=lambda r: r["score"])
        print(f"    gen {gen+1}/{INNER_GENS}  best_score={gen_best['score']:.4f}"
              f"  speed={gen_best['fwd_speed_ms']:.4f} m/s"
              f"  omega={gen_best['hip_omega']:.3f}  amp={gen_best['leg_amp_deg']:.1f}")

    print(f"  --> best: score={best['score']:.4f}  speed={best['fwd_speed_ms']:.4f} m/s"
          f"  omega={best['hip_omega']:.3f}  amp={best['leg_amp_deg']:.1f}")
    return best


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_heatmap(grid_results: dict, xs: list, ys: list, prefix: str) -> None:
    """2-D heatmap: x_scale × y_scale, colour = best forward speed."""
    speed_grid = np.full((len(ys), len(xs)), np.nan)
    for (sx, sy), res in grid_results.items():
        xi = xs.index(sx)
        yi = ys.index(sy)
        speed_grid[yi, xi] = res["fwd_speed_ms"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(speed_grid, origin="lower", aspect="auto",
                   cmap="RdYlGn",
                   extent=[min(xs)-0.05, max(xs)+0.05,
                           min(ys)-0.05, max(ys)+0.05])
    plt.colorbar(im, ax=ax, label="Best forward speed (m/s)")

    # Annotate each cell with speed + winning (ω, amp)
    for (sx, sy), res in grid_results.items():
        xi = xs.index(sx)
        yi = ys.index(sy)
        # map grid index to image coordinate
        cx = min(xs) + xi * (max(xs) - min(xs)) / max(len(xs) - 1, 1)
        cy = min(ys) + yi * (max(ys) - min(ys)) / max(len(ys) - 1, 1)
        ax.text(cx, cy,
                f"{res['fwd_speed_ms']:.3f}\nω={res['hip_omega']:.2f}\na={res['leg_amp_deg']:.0f}°",
                ha="center", va="center", fontsize=7,
                color="black" if res["fwd_speed_ms"] > 0.01 else "gray")

    ax.set_xticks(xs); ax.set_xticklabels([f"{v:.2f}" for v in xs])
    ax.set_yticks(ys); ax.set_yticklabels([f"{v:.2f}" for v in ys])
    ax.set_xlabel("x_scale")
    ax.set_ylabel("y_scale")
    ax.set_title(f"Best forward speed per (x_scale, y_scale)  [z={Z_SCALE}]")
    plt.tight_layout()
    path = f"{prefix}_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_bar(grid_results: dict, prefix: str) -> None:
    """Ranked bar chart of best speed per (x_scale, y_scale) pair."""
    pairs = sorted(grid_results.items(), key=lambda kv: kv[1]["fwd_speed_ms"], reverse=True)

    labels = [f"x={sx:.1f}\ny={sy:.1f}" for (sx, sy), _ in pairs]
    speeds = [res["fwd_speed_ms"]        for _, res in pairs]
    omegas = [res["hip_omega"]            for _, res in pairs]
    amps   = [res["leg_amp_deg"]          for _, res in pairs]
    fells  = [res["fell"]                 for _, res in pairs]

    cmap  = plt.get_cmap("RdYlGn")
    s_max = max(speeds) if max(speeds) > 0 else 1.0
    colors = [cmap(v / s_max) for v in speeds]

    fig, ax = plt.subplots(figsize=(max(10, len(pairs) * 0.7), 5))
    bars = ax.bar(range(len(pairs)), speeds, color=colors, edgecolor="k", linewidth=0.6)

    # Mark pairs where the best candidate still fell
    for i, (bar, fell) in enumerate(zip(bars, fells)):
        if fell:
            ax.text(i, bar.get_height() + 0.001, "✗", ha="center",
                    va="bottom", color="crimson", fontsize=10, fontweight="bold")

    # Annotate with winning omega and amp
    for i, (bar, w, a) in enumerate(zip(bars, omegas, amps)):
        ax.text(i, bar.get_height() / 2,
                f"ω={w:.2f}\na={a:.0f}°",
                ha="center", va="center", fontsize=7, color="black")

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Best forward speed (m/s)")
    ax.set_title(f"Best speed per (x_scale, y_scale) pair  [z={Z_SCALE}]"
                 f"\n✗ = best candidate still fell")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{prefix}_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    pairs = [(sx, sy) for sy in SWEEP_Y for sx in SWEEP_X]
    total = len(pairs)
    print(f"XY scale sweep: {total} (x,y) pairs  ×  {INNER_GENS} inner CMA-ES gens  ×  pop={INNER_POP}")
    print(f"x_scale values: {SWEEP_X}")
    print(f"y_scale values: {SWEEP_Y}")
    print(f"z_scale fixed:  {Z_SCALE}\n")

    grid_results: dict[tuple[float, float], dict] = {}

    for idx, (sx, sy) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Pair {idx+1}/{total}  x_scale={sx:.2f}  y_scale={sy:.2f}")
        print("="*60)
        best = optimise_actuation(sx, sy, Z_SCALE)
        best["x_scale"] = sx
        best["y_scale"] = sy
        best["z_scale"] = Z_SCALE
        grid_results[(sx, sy)] = best

    # Save
    payload = {
        "sweep_x": SWEEP_X, "sweep_y": SWEEP_Y, "z_scale": Z_SCALE,
        "inner_gens": INNER_GENS, "inner_pop": INNER_POP,
        "inner_lows": INNER_LOWS.tolist(), "inner_highs": INNER_HIGHS.tolist(),
        "grid_results": {f"{sx},{sy}": v for (sx, sy), v in grid_results.items()},
    }
    with open(RESULTS_PKL, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"\nSaved results -> {RESULTS_PKL}")

    # Report
    best_pair = max(grid_results.items(), key=lambda kv: kv[1]["fwd_speed_ms"])
    (bx, by), br = best_pair
    print(f"\nBest pair: x={bx:.2f}  y={by:.2f}  speed={br['fwd_speed_ms']:.4f} m/s"
          f"  omega={br['hip_omega']:.3f}  amp={br['leg_amp_deg']:.1f}")

    # Plots
    plot_heatmap(grid_results, SWEEP_X, SWEEP_Y, OUT_PREFIX)
    plot_bar(grid_results, OUT_PREFIX)


if __name__ == "__main__":
    main()
