import csv
import json
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
MIN_SWING_CLEARANCE = 0.02
MIN_ALTERNATING_STEPS = 1
ANALYSIS_STRIDE = 2

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

RIGHT_COLLISION_GEOM_NAMES = [
    "right_foot_1_col",
    "right_foot_2_col",
    "right_foot_3_col",
]

LEFT_COLLISION_GEOM_NAMES = [
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
    foot_collision_geom_ids: dict[str, tuple[int, ...]]


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
    foot_collision_geom_ids = {
        "right": tuple(
            geom_id
            for geom_id in (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in RIGHT_COLLISION_GEOM_NAMES
            )
            if geom_id != -1
        ),
        "left": tuple(
            geom_id
            for geom_id in (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in LEFT_COLLISION_GEOM_NAMES
            )
            if geom_id != -1
        ),
    }

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
        foot_collision_geom_ids=foot_collision_geom_ids,
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


def normalize_xy(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    xy = np.asarray(vec[:2], dtype=float)
    norm = np.linalg.norm(xy)
    if norm < 1e-9:
        return fallback.copy()
    return xy / norm


def get_heading_axes(xmat_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(xmat_flat, dtype=float).reshape(3, 3)
    forward_xy = normalize_xy(rot[:, 0], fallback=np.array([1.0, 0.0]))
    left_xy = np.array([-forward_xy[1], forward_xy[0]])
    return forward_xy, left_xy


def collect_active_contact_geom_ids(data: mujoco.MjData) -> set[int]:
    active_geom_ids = set()
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        active_geom_ids.add(int(contact.geom1))
        active_geom_ids.add(int(contact.geom2))
    return active_geom_ids


def collect_foot_contact_flags(
    data: mujoco.MjData,
    right_geom_ids: tuple[int, ...],
    left_geom_ids: tuple[int, ...],
) -> tuple[bool, bool]:
    right_in_contact = False
    left_in_contact = False
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if not right_in_contact and (geom1 in right_geom_ids or geom2 in right_geom_ids):
            right_in_contact = True
        if not left_in_contact and (geom1 in left_geom_ids or geom2 in left_geom_ids):
            left_in_contact = True
        if right_in_contact and left_in_contact:
            break
    return right_in_contact, left_in_contact


def get_mean_geom_position(data: mujoco.MjData, geom_ids: tuple[int, ...]) -> np.ndarray:
    if not geom_ids:
        return np.zeros(3, dtype=float)
    total = np.zeros(3, dtype=float)
    for geom_id in geom_ids:
        total += data.geom_xpos[geom_id]
    return total / len(geom_ids)


def compute_walk_score(
    fell: bool,
    forward_progress: float,
    lateral_drift: float,
    heading_change_rad: float,
    slip_ratio: float,
    alternating_steps: int,
    min_swing_clearance: float,
) -> float:
    bounded_slip_ratio = min(slip_ratio, 10.0)
    score = forward_progress - 0.5 * lateral_drift - 0.25 * heading_change_rad - 0.1 * bounded_slip_ratio
    if fell:
        score -= 2.0
    if alternating_steps < MIN_ALTERNATING_STEPS:
        score -= 0.5
    if min_swing_clearance < MIN_SWING_CLEARANCE:
        score -= 0.5
    return score


def build_replay_params(params: dict) -> dict[str, float]:
    return {
        "foot_x": float(params["foot_x"]),
        "foot_y": float(params["foot_y"]),
        "Kp": float(params["Kp"]),
        "Kd": float(params["Kd"]),
        "start_amp_mult": float(params["start_amp_mult"]),
        "start_freq_mult": float(params["start_freq_mult"]),
        "amp_deg": float(params["amp_deg"]),
        "freq_hz": float(params["freq_hz"]),
    }


def run_single_trial(ctx: SimulationContext, params: dict, verbose: bool = True) -> dict:
    mujoco.mj_resetData(ctx.model, ctx.data)

    replay_params = build_replay_params(params)

    apply_foot_offsets(ctx, replay_params["foot_x"], replay_params["foot_y"])
    mujoco.mj_forward(ctx.model, ctx.data)

    if ctx.debug_geom_id != -1:
        vprint(f"Global Position: {ctx.data.geom_xpos[ctx.debug_geom_id]}", verbose=verbose)

    start_torso_xy = ctx.data.xpos[ctx.torso_body_id][:2].copy()
    start_forward_xy, start_left_xy = get_heading_axes(ctx.data.xmat[ctx.torso_body_id])
    hip_omega = replay_params["freq_hz"] * 2 * np.pi
    leg_amp_rad = np.deg2rad(replay_params["amp_deg"])
    cmd_buffer = [0.0] * CMD_DELAY_STEPS
    fell = False
    max_steps = int(ITERATION_DURATION / ctx.model.opt.timestep)
    energy_used = 0.0
    path_length = 0.0
    prev_torso_xy = start_torso_xy.copy()
    instantaneous_forward_progress = 0.0
    instantaneous_forward_absolute = 0.0
    instantaneous_lateral_progress = 0.0
    right_geom_ids = ctx.foot_collision_geom_ids["right"]
    left_geom_ids = ctx.foot_collision_geom_ids["left"]
    right_in_contact, left_in_contact = collect_foot_contact_flags(ctx.data, right_geom_ids, left_geom_ids)
    foot_metrics = {}
    landing_sequence: list[str] = []
    for side, geom_ids in (("right", right_geom_ids), ("left", left_geom_ids)):
        foot_center = get_mean_geom_position(ctx.data, geom_ids)
        foot_metrics[side] = {
            "geom_ids": geom_ids,
            "baseline_z": float(foot_center[2]),
            "prev_xy": foot_center[:2].copy(),
            "prev_contact": right_in_contact if side == "right" else left_in_contact,
            "current_air_max_clearance": 0.0,
            "max_clearance": 0.0,
            "contact_slip": 0.0,
            "landings": 0,
        }

    def update_gait_metrics() -> None:
        nonlocal path_length
        nonlocal instantaneous_forward_progress
        nonlocal instantaneous_forward_absolute
        nonlocal instantaneous_lateral_progress
        nonlocal prev_torso_xy

        torso_xy = ctx.data.xpos[ctx.torso_body_id][:2].copy()
        torso_delta_xy = torso_xy - prev_torso_xy
        path_length += float(np.linalg.norm(torso_delta_xy))
        current_forward_xy, current_left_xy = get_heading_axes(ctx.data.xmat[ctx.torso_body_id])
        instantaneous_forward_step = float(np.dot(torso_delta_xy, current_forward_xy))
        instantaneous_lateral_step = float(np.dot(torso_delta_xy, current_left_xy))
        instantaneous_forward_progress += instantaneous_forward_step
        instantaneous_forward_absolute += abs(instantaneous_forward_step)
        instantaneous_lateral_progress += abs(instantaneous_lateral_step)
        prev_torso_xy = torso_xy

        right_contact_now, left_contact_now = collect_foot_contact_flags(ctx.data, right_geom_ids, left_geom_ids)
        contact_flags = {"right": right_contact_now, "left": left_contact_now}

        for side, metrics in foot_metrics.items():
            foot_center = get_mean_geom_position(ctx.data, metrics["geom_ids"])
            current_xy = foot_center[:2]
            if metrics["prev_contact"]:
                metrics["contact_slip"] += float(np.linalg.norm(current_xy - metrics["prev_xy"]))
            metrics["prev_xy"] = current_xy.copy()

            clearance = float(foot_center[2] - metrics["baseline_z"])
            metrics["max_clearance"] = max(metrics["max_clearance"], clearance)

            in_contact = contact_flags[side]
            if not in_contact:
                metrics["current_air_max_clearance"] = max(metrics["current_air_max_clearance"], clearance)
            elif not metrics["prev_contact"]:
                if metrics["current_air_max_clearance"] >= MIN_SWING_CLEARANCE:
                    metrics["landings"] += 1
                    landing_sequence.append(side)
                metrics["current_air_max_clearance"] = 0.0

            metrics["prev_contact"] = in_contact

    last_analysis_step = 0

    for step_idx in range(1, max_steps + 1):
        t = ctx.data.time
        target_pos_rad, target_vel_rad = calculate_sine_reference(
            t,
            hip_omega,
            leg_amp_rad,
            replay_params["start_amp_mult"],
            replay_params["start_freq_mult"],
        )

        current_pos = ctx.data.qpos[ctx.qpos_idx]
        current_vel = ctx.data.qvel[ctx.qvel_idx]

        ramp = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
        tau = (
            replay_params["Kp"] * ramp * (target_pos_rad - current_pos)
            + replay_params["Kd"] * ramp * (target_vel_rad - current_vel)
        )
        tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        energy_used += abs(tau * current_vel) * ctx.model.opt.timestep

        cmd_buffer.append(tau)
        ctx.data.ctrl[0] = cmd_buffer.pop(0)

        mujoco.mj_step(ctx.model, ctx.data)

        analysis_due = (step_idx % ANALYSIS_STRIDE) == 0
        if analysis_due:
            update_gait_metrics()
            last_analysis_step = step_idx

        if check_has_fallen(ctx):
            if not analysis_due:
                update_gait_metrics()
                last_analysis_step = step_idx
            vprint(
                f"   [DEBUG] Fell at t={ctx.data.time:.3f}s. Height={ctx.data.xpos[ctx.torso_body_id][2]:.2f}m",
                verbose=verbose,
            )
            fell = True
            break

    if not fell and last_analysis_step != max_steps:
        update_gait_metrics()

    final_torso_xy = ctx.data.xpos[ctx.torso_body_id][:2].copy()
    displacement_xy = final_torso_xy - start_torso_xy
    distance = float(np.linalg.norm(displacement_xy))
    forward_progress = float(np.dot(displacement_xy, start_forward_xy))
    lateral_drift = float(abs(np.dot(displacement_xy, start_left_xy)))
    final_forward_xy, _ = get_heading_axes(ctx.data.xmat[ctx.torso_body_id])
    heading_change_rad = float(
        np.arccos(np.clip(np.dot(start_forward_xy, final_forward_xy), -1.0, 1.0))
    )
    forward_efficiency = forward_progress / path_length if path_length > 1e-6 else 0.0
    total_contact_slip = float(sum(metrics["contact_slip"] for metrics in foot_metrics.values()))
    if forward_progress > 1e-6:
        slip_ratio = total_contact_slip / forward_progress
    else:
        slip_ratio = 999.0 if total_contact_slip > 0 else 0.0
    alternating_steps = sum(
        1 for previous, current in zip(landing_sequence, landing_sequence[1:]) if current != previous
    )
    landing_count = len(landing_sequence)
    min_swing_clearance = float(
        min(metrics["max_clearance"] for metrics in foot_metrics.values()) if foot_metrics else 0.0
    )
    gait_quality_pass = (
        (not fell)
        and instantaneous_forward_absolute > instantaneous_lateral_progress
        and alternating_steps >= MIN_ALTERNATING_STEPS
        and min_swing_clearance >= MIN_SWING_CLEARANCE
    )
    walk_score = compute_walk_score(
        fell=fell,
        forward_progress=forward_progress,
        lateral_drift=lateral_drift,
        heading_change_rad=heading_change_rad,
        slip_ratio=slip_ratio,
        alternating_steps=alternating_steps,
        min_swing_clearance=min_swing_clearance,
    )
    cot = (energy_used / (ctx.total_mass * ctx.gravity * distance)) if distance > 1e-6 else float("inf")

    return {
        "Foot_X": replay_params["foot_x"],
        "Foot_Y": replay_params["foot_y"],
        "Kp": replay_params["Kp"],
        "Kd": replay_params["Kd"],
        "Start_Amp_Mult": replay_params["start_amp_mult"],
        "Start_Freq_Mult": replay_params["start_freq_mult"],
        "Amplitude_Deg": replay_params["amp_deg"],
        "Frequency_Hz": replay_params["freq_hz"],
        "Replay_Params_JSON": json.dumps(replay_params, sort_keys=True),
        "Fell": fell,
        "Distance_Traversed": distance,
        "Forward_Progress": forward_progress,
        "Lateral_Drift": lateral_drift,
        "Instantaneous_Forward_Progress": instantaneous_forward_progress,
        "Instantaneous_Forward_Absolute": instantaneous_forward_absolute,
        "Instantaneous_Lateral_Progress": instantaneous_lateral_progress,
        "Path_Length": path_length,
        "Forward_Efficiency": forward_efficiency,
        "Heading_Change_Deg": float(np.rad2deg(heading_change_rad)),
        "Right_Landings": foot_metrics["right"]["landings"],
        "Left_Landings": foot_metrics["left"]["landings"],
        "Landing_Count": landing_count,
        "Alternating_Steps": alternating_steps,
        "Right_Max_Swing_Clearance": float(foot_metrics["right"]["max_clearance"]),
        "Left_Max_Swing_Clearance": float(foot_metrics["left"]["max_clearance"]),
        "Min_Swing_Clearance": min_swing_clearance,
        "Right_Contact_Slip": float(foot_metrics["right"]["contact_slip"]),
        "Left_Contact_Slip": float(foot_metrics["left"]["contact_slip"]),
        "Total_Contact_Slip": total_contact_slip,
        "Slip_Ratio": slip_ratio,
        "Gait_Quality_Pass": gait_quality_pass,
        "Walk_Score": walk_score,
        "CoT": cot,
    }


def should_save_result(result_row: dict, save_all_results: bool) -> bool:
    passed = result_row["Gait_Quality_Pass"]
    return save_all_results or passed


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
        (
            f"Starting random sweep: {total_trials} trials. "
            "Saving results with gait-quality pass."
        ),
        verbose=verbose,
    )

    for trial_idx, params in enumerate(parameter_rows, 1):
        result_row = run_single_trial(ctx, params, verbose=verbose)
        passed = result_row["Gait_Quality_Pass"]
        vprint(
            f"[{trial_idx:>5}/{total_trials}] "
            f"freq={params['freq_hz']:.2f}Hz amp={params['amp_deg']:.1f}° "
            f"Kp={params['Kp']:.1f} Kd={params['Kd']:.1f} "
            f"fx={params['foot_x']:.3f} fy={params['foot_y']:.3f} | "
            f"fell={result_row['Fell']} dist={result_row['Distance_Traversed']:.2f}m "
            f"forward={result_row['Forward_Progress']:.2f}m drift={result_row['Lateral_Drift']:.2f}m "
            f"inst_fwd={result_row['Instantaneous_Forward_Progress']:.2f}m "
            f"inst_fwd_abs={result_row['Instantaneous_Forward_Absolute']:.2f}m "
            f"inst_lat={result_row['Instantaneous_Lateral_Progress']:.2f}m "
            f"steps={result_row['Alternating_Steps']} clearance={result_row['Min_Swing_Clearance']:.3f}m "
            f"score={result_row['Walk_Score']:.3f} CoT={result_row['CoT']:.3f}| "
            f"{'SAVED' if passed else 'skipped'}",
            verbose=verbose,
        )

        if should_save_result(result_row, save_all_results):
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
        "Mesh_Generator": os.environ.get("SWEEP_MESH_GENERATOR"),
        "Mesh_Generator_Entry_XML": os.environ.get("SWEEP_MESH_GENERATOR_ENTRY_XML"),
        "Mesh_Generator_SCAD": os.environ.get("SWEEP_MESH_GENERATOR_SCAD"),
    }

    parameter_rows = generate_lhs_samples(num_trials)
    results = run_parameter_chunk(
        model_xml_path=model_xml_path,
        parameter_rows=parameter_rows,
        save_all_results=save_all_results,
        metadata=metadata,
        verbose=VERBOSE,
    )
    write_results(results, results_csv, append=append_results, verbose=VERBOSE)


if __name__ == "__main__":
    main()
