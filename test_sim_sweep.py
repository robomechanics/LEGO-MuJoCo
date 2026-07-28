import csv
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.stats import qmc


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


VERBOSE = _env_flag("TEST_SIM_SWEEP_VERBOSE", True)


def vprint(*args, verbose: bool = True, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


JOINT_NAME = "hip"
TORSO_BODY_NAME = "motor"
TORQUE_LIMIT = 25.0
T_WAIT = 3.0
USE_RAMP = False
RAMP_TIME = 1.0
CMD_DELAY_STEPS = 1
ITERATION_DURATION = 20.0

RANGES = {
    "foot_x": (-0.07, 0.025),
    "foot_y": (-0.02, 0.01),
    "Kp": (20.0, 45.0),
    "Kd": (2.0, 15.0),
    "start_amp_mult": (0.8, 1.8),
    "start_freq_mult": (0.7, 1.4),
    "amp_deg": (20.0, 40.0),
    "freq_hz": (0.4, 0.8),
}

FOOT_GEOM_NAMES = [
    "right_foot_1",
    "right_foot_2",
    "right_foot_3",
    "left_foot_1",
    "left_foot_2",
    "left_foot_3",
    "right_foot_1_col",
    "right_foot_2_col",
    "right_foot_3_col",
    "left_foot_1_col",
    "left_foot_2_col",
    "left_foot_3_col",
]


@dataclass
class SimulationContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    total_mass: float
    gravity: float
    torso_body_id: int
    qpos_idx: int
    qvel_idx: int
    original_geom_pos: dict[str, np.ndarray]
    foot_parent_body_ids: tuple[int, ...]
    original_body_ipos: dict[int, np.ndarray]
    debug_geom_id: int


def generate_lhs_samples(n_trials: int) -> list[dict]:
    """Latin Hypercube sampler for space coverage."""
    sampler = qmc.LatinHypercube(d=8)
    samples = sampler.random(n=n_trials)

    keys = sorted(list(RANGES.keys()))
    bounds_low = [RANGES[k][0] for k in keys]
    bounds_high = [RANGES[k][1] for k in keys]
    scaled = qmc.scale(samples, bounds_low, bounds_high)

    return [dict(zip(keys, row)) for row in scaled]


def load_simulation(model_xml_path: str | Path) -> SimulationContext:
    model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    data = mujoco.MjData(model)
    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    gravity = 9.81

    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)
    if joint_id == -1 or torso_body_id == -1:
        raise ValueError(f"Could not find joint '{JOINT_NAME}' or body '{TORSO_BODY_NAME}'.")

    qpos_idx = model.jnt_qposadr[joint_id]
    qvel_idx = model.jnt_dofadr[joint_id]

    original_geom_pos = {}
    foot_parent_body_ids = set()
    for name in FOOT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id != -1:
            original_geom_pos[name] = model.geom_pos[geom_id].copy()
            foot_parent_body_ids.add(model.geom_bodyid[geom_id])

    original_body_ipos = {bid: model.body_ipos[bid].copy() for bid in foot_parent_body_ids}
    debug_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_3_col")

    return SimulationContext(
        model=model,
        data=data,
        total_mass=total_mass,
        gravity=gravity,
        torso_body_id=torso_body_id,
        qpos_idx=qpos_idx,
        qvel_idx=qvel_idx,
        original_geom_pos=original_geom_pos,
        foot_parent_body_ids=tuple(sorted(foot_parent_body_ids)),
        original_body_ipos=original_body_ipos,
        debug_geom_id=debug_geom_id,
    )


def apply_foot_offsets(ctx: SimulationContext, foot_x: float, foot_y: float) -> None:
    """
    Applies foot offsets by correcting geoms and body centers of mass.
    """
    delta_right_geom = np.array([foot_x, foot_y, 0.0])
    delta_left_geom = np.array([0.0, -foot_y, foot_x])

    delta_right_body = np.array([foot_x, foot_y, 0.0])
    delta_left_body = np.array([-foot_x, foot_y, 0.0])

    for bid in ctx.foot_parent_body_ids:
        body_name = mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if body_name == "motor":
            ctx.model.body_ipos[bid] = ctx.original_body_ipos[bid] + delta_right_body
        elif body_name == "simplified_motor___arm_rod":
            ctx.model.body_ipos[bid] = ctx.original_body_ipos[bid] + delta_left_body

    for name in FOOT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            continue
        if "right" in name.lower():
            ctx.model.geom_pos[geom_id] = ctx.original_geom_pos[name] + delta_right_geom
        else:
            ctx.model.geom_pos[geom_id] = ctx.original_geom_pos[name] + delta_left_geom

    mujoco.mj_setConst(ctx.model, ctx.data)
    mujoco.mj_forward(ctx.model, ctx.data)


def check_has_fallen(ctx: SimulationContext, height_threshold: float = 0.5, angle_threshold_deg: float = 45.0) -> bool:
    if ctx.data.xpos[ctx.torso_body_id][2] < height_threshold:
        return True

    torso_up_z = ctx.data.xmat[ctx.torso_body_id][8]
    tilt_angle_deg = np.rad2deg(np.arccos(np.clip(torso_up_z, -1.0, 1.0)))
    return tilt_angle_deg > angle_threshold_deg


def calculate_sine_reference(t: float, hip_omega: float, leg_amp_rad: float, start_amp_mult: float, start_freq_mult: float) -> tuple[float, float]:
    w1 = hip_omega * start_freq_mult
    w2 = hip_omega
    t0 = T_WAIT
    at = start_amp_mult * leg_amp_rad
    a_steady = leg_amp_rad
    t_transition = t0 + np.pi / w1

    if t <= t0:
        return 0.0, 0.0
    if t < t_transition:
        phase = w1 * (t - t0)
        return at * np.sin(phase), at * w1 * np.cos(phase)

    phase = w2 * (t - t_transition)
    return -a_steady * np.sin(phase), -a_steady * w2 * np.cos(phase)


def run_single_trial(ctx: SimulationContext, params: dict, verbose: bool = True) -> dict:
    mujoco.mj_resetData(ctx.model, ctx.data)

    apply_foot_offsets(ctx, params["foot_x"], params["foot_y"])
    mujoco.mj_forward(ctx.model, ctx.data)

    if ctx.debug_geom_id != -1:
        vprint(f"Global Position: {ctx.data.geom_xpos[ctx.debug_geom_id]}", verbose=verbose)

    start_x = ctx.data.xpos[ctx.torso_body_id][0]
    start_y = ctx.data.xpos[ctx.torso_body_id][1]
    hip_omega = params["freq_hz"] * 2 * np.pi
    leg_amp_rad = np.deg2rad(params["amp_deg"])
    cmd_buffer = [0.0] * CMD_DELAY_STEPS
    fell = False
    max_steps = int(ITERATION_DURATION / ctx.model.opt.timestep)
    energy_used = 0.0

    for _ in range(max_steps):
        t = ctx.data.time
        target_pos_rad, target_vel_rad = calculate_sine_reference(
            t,
            hip_omega,
            leg_amp_rad,
            params["start_amp_mult"],
            params["start_freq_mult"],
        )

        current_pos = ctx.data.qpos[ctx.qpos_idx]
        current_vel = ctx.data.qvel[ctx.qvel_idx]

        ramp = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
        tau = (
            params["Kp"] * ramp * (target_pos_rad - current_pos)
            + params["Kd"] * ramp * (target_vel_rad - current_vel)
        )
        tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        energy_used += abs(tau * current_vel) * ctx.model.opt.timestep

        cmd_buffer.append(tau)
        ctx.data.ctrl[0] = cmd_buffer.pop(0)

        mujoco.mj_step(ctx.model, ctx.data)

        if check_has_fallen(ctx):
            vprint(
                f"   [DEBUG] Fell at t={ctx.data.time:.3f}s. Height={ctx.data.xpos[ctx.torso_body_id][2]:.2f}m",
                verbose=verbose,
            )
            fell = True
            break

    final_x = ctx.data.xpos[ctx.torso_body_id][0]
    final_y = ctx.data.xpos[ctx.torso_body_id][1]
    distance = float(np.sqrt((final_x - start_x) ** 2 + (final_y - start_y) ** 2))
    cot = (energy_used / (ctx.total_mass * ctx.gravity * distance)) if distance > 1e-6 else float("inf")

    return {
        "Foot_X": round(params["foot_x"], 4),
        "Foot_Y": round(params["foot_y"], 4),
        "Kp": round(params["Kp"], 2),
        "Kd": round(params["Kd"], 2),
        "Start_Amp_Mult": round(params["start_amp_mult"], 3),
        "Start_Freq_Mult": round(params["start_freq_mult"], 3),
        "Amplitude_Deg": round(params["amp_deg"], 2),
        "Frequency_Hz": round(params["freq_hz"], 3),
        "Fell": fell,
        "Distance_Traversed": round(distance, 4),
        "CoT": round(cot, 4),
    }


def should_save_result(result_row: dict, min_distance: float, save_all_results: bool) -> bool:
    passed = (not result_row["Fell"]) and result_row["Distance_Traversed"] >= min_distance
    return save_all_results or passed or ((not passed) and result_row["Distance_Traversed"] >= min_distance)


def attach_metadata(rows: list[dict], metadata: dict | None) -> list[dict]:
    if not metadata:
        return rows
    clean_metadata = {key: value for key, value in metadata.items() if value is not None}
    if not clean_metadata:
        return rows
    return [{**clean_metadata, **row} for row in rows]


def run_parameter_chunk(
    model_xml_path: str | Path,
    parameter_rows: list[dict],
    min_distance: float,
    save_all_results: bool,
    metadata: dict | None = None,
    verbose: bool = True,
    progress_every: int | None = None,
    progress_offset: int = 0,
    total_trials_override: int | None = None,
) -> list[dict]:
    ctx = load_simulation(model_xml_path)
    results = []

    total_trials = len(parameter_rows)
    vprint(
        f"Starting random sweep: {total_trials} trials. Saving results with no fall and >{min_distance}m distance.",
        verbose=verbose,
    )

    for trial_idx, params in enumerate(parameter_rows, 1):
        result_row = run_single_trial(ctx, params, verbose=verbose)
        passed = (not result_row["Fell"]) and result_row["Distance_Traversed"] >= min_distance
        vprint(
            f"[{trial_idx:>5}/{total_trials}] "
            f"freq={params['freq_hz']:.2f}Hz amp={params['amp_deg']:.1f}° "
            f"Kp={params['Kp']:.1f} Kd={params['Kd']:.1f} "
            f"fx={params['foot_x']:.3f} fy={params['foot_y']:.3f} | "
            f"fell={result_row['Fell']} dist={result_row['Distance_Traversed']:.2f}m "
            f"CoT={result_row['CoT']:.3f}| {'SAVED' if passed else 'skipped'}",
            verbose=verbose,
        )

        if should_save_result(result_row, min_distance, save_all_results):
            results.append(result_row)

        if progress_every:
            global_trial_idx = progress_offset + trial_idx
            total_for_progress = total_trials_override or total_trials
            if global_trial_idx % progress_every == 0:
                pair_label = metadata.get("Pair_Index") if metadata else "?"
                run_label = metadata.get("Run_Index") if metadata else "?"
                print(
                    f"Pair {pair_label} run {run_label}: completed "
                    f"{global_trial_idx}/{total_for_progress} trials",
                    flush=True,
                )

    vprint(f"Save rate: {len(results)}/{total_trials} = {len(results)/total_trials*100:.1f}%", verbose=verbose)
    return attach_metadata(results, metadata)


def write_results(rows: list[dict], csv_file: str | Path, append: bool = False, verbose: bool = True) -> None:
    if not rows:
        vprint("\nSweep complete. No trials met the criteria.", verbose=verbose)
        return

    csv_file = str(csv_file)
    write_header = True
    if append and os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        write_header = False

    mode = "a" if append else "w"
    with open(csv_file, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    vprint(f"\nSweep complete. {len(rows)} rows saved to '{csv_file}'.", verbose=verbose)


def main() -> None:
    min_distance = float(os.environ.get("MIN_DISTANCE", "2.0"))
    num_trials = int(os.environ.get("NUM_TRIALS", "500"))
    model_xml_path = os.environ.get("MODEL_XML_PATH", "modified_model.xml")
    results_csv = os.environ.get("SWEEP_RESULTS_CSV", "sweep_results.csv")
    append_results = os.environ.get("SWEEP_APPEND_RESULTS", "0") == "1"
    save_all_results = os.environ.get("SWEEP_SAVE_ALL_RESULTS", "0") == "1"
    metadata = {
        "Mesh_X": os.environ.get("SWEEP_MESH_X"),
        "Mesh_Y": os.environ.get("SWEEP_MESH_Y"),
        "Mesh_X_Index": os.environ.get("SWEEP_MESH_X_INDEX"),
        "Mesh_Y_Index": os.environ.get("SWEEP_MESH_Y_INDEX"),
        "Pair_Index": os.environ.get("SWEEP_PAIR_INDEX"),
        "Run_Index": os.environ.get("SWEEP_RUN_INDEX"),
    }

    parameter_rows = generate_lhs_samples(num_trials)
    results = run_parameter_chunk(
        model_xml_path=model_xml_path,
        parameter_rows=parameter_rows,
        min_distance=min_distance,
        save_all_results=save_all_results,
        metadata=metadata,
        verbose=VERBOSE,
    )
    write_results(results, results_csv, append=append_results, verbose=VERBOSE)


if __name__ == "__main__":
    main()
