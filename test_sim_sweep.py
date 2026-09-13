import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from coordinate_frame import mujoco_heading_axes
from control_waveform import (
    DEFAULT_KP,
    DEFAULT_KD,
    DEFAULT_TORQUE_LIMIT,
    DEFAULT_GAIN_DISTRIBUTIONS,
    DEFAULT_HIP_FREQ_HZ,
    DEFAULT_LEG_AMP_DEG,
    DEFAULT_START_AMP_MULT,
    DEFAULT_START_FREQ_MULT,
    DEFAULT_STARTUP_RAMP_TIME,
    DEFAULT_T_WAIT,
    startup_sine_reference,
)
from settle_utils import startup_settle_orientation


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
USE_RAMP = False
RAMP_TIME = 1.0
CMD_DELAY_STEPS = 1
ITERATION_DURATION = 20.0
MIN_SWING_CLEARANCE = 0.02
MIN_ALTERNATING_STEPS = 1
ANALYSIS_STRIDE = 2
STARTUP_SETTLE_S = 8.0
STARTUP_SETTLE_AVG_S = 2.0

DEFAULT_FIXED_PARAMS = {
    "foot_x": 0.0,
    "foot_y": 0.0,
    "torque_limit": DEFAULT_TORQUE_LIMIT,
    "Kp": DEFAULT_KP,
    "Kd": DEFAULT_KD,
    "start_amp_mult": DEFAULT_START_AMP_MULT,
    "start_freq_mult": DEFAULT_START_FREQ_MULT,
    "ramp_time": float(os.environ.get("TEST_SIM_STARTUP_RAMP_TIME", str(DEFAULT_STARTUP_RAMP_TIME))),
    "amp_deg": DEFAULT_LEG_AMP_DEG,
    "freq_hz": DEFAULT_HIP_FREQ_HZ,
}

DEFAULT_NORMAL_DISTRIBUTIONS = {
    **{name: dict(spec) for name, spec in DEFAULT_GAIN_DISTRIBUTIONS.items()},
    "start_amp_mult": {"mean": DEFAULT_START_AMP_MULT, "std": 0.2, "min": 0.8, "max": 1.8},
    "start_freq_mult": {"mean": DEFAULT_START_FREQ_MULT, "std": 0.15, "min": 0.7, "max": 1.4},
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


def _json_env(name: str, default: dict | None = None) -> dict | None:
    value = os.environ.get(name)
    if value is None:
        return default
    return json.loads(value)


def _sample_clipped_normal(rng: np.random.Generator, spec: dict, size: int) -> np.ndarray:
    mean = float(spec["mean"])
    std = float(spec["std"])
    low = float(spec.get("min", -np.inf))
    high = float(spec.get("max", np.inf))
    if std < 0:
        raise ValueError(f"Normal distribution std must be non-negative, got {std}.")
    if std == 0:
        samples = np.full(size, mean, dtype=float)
    else:
        samples = rng.normal(loc=mean, scale=std, size=size)
    return np.clip(samples, low, high)


def generate_parameter_samples(
    n_trials: int,
    fixed_params: dict | None = None,
    normal_distributions: dict | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    if n_trials < 1:
        return []

    rng = np.random.default_rng() if rng is None else rng
    fixed = {**DEFAULT_FIXED_PARAMS, **(fixed_params or {})}
    sampled = dict(DEFAULT_NORMAL_DISTRIBUTIONS if normal_distributions is None else normal_distributions)

    rows = [dict(fixed) for _ in range(n_trials)]
    for key, spec in sampled.items():
        values = _sample_clipped_normal(rng, spec, n_trials)
        for row, value in zip(rows, values):
            row[key] = float(value)

    return rows


def generate_lhs_samples(n_trials: int) -> list[dict]:
    """Backward-compatible alias for the new parameter sampler."""
    return generate_parameter_samples(n_trials)


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
    Applies public foot offsets to geoms and body centers of mass.

    Matches closed-loop FOOT_X/FOOT_Y: positive foot_x moves both feet inward;
    positive foot_y moves both forward. Values are metres. Each foot and its
    approximate body CoM receive the same displacement in their parent frame.
    """
    body_offsets = {
        "motor": np.array([foot_x, foot_y, 0.0]),
        "simplified_motor___arm_rod": np.array([0.0, -foot_y, foot_x]),
    }

    for bid in ctx.foot_parent_body_ids:
        body_name = mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if body_name in {"motor", "simplified_motor___arm_rod"}:
            ctx.model.body_ipos[bid] = ctx.original_body_ipos[bid] + body_offsets[body_name]

    for name in FOOT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            continue
        body_name = ctx.model.body(int(ctx.model.geom_bodyid[geom_id])).name
        ctx.model.geom_pos[geom_id] = ctx.original_geom_pos[name] + body_offsets[body_name]

    mujoco.mj_setConst(ctx.model, ctx.data)
    mujoco.mj_forward(ctx.model, ctx.data)


def check_has_fallen(ctx: SimulationContext, height_threshold: float = 0.5, angle_threshold_deg: float = 45.0) -> bool:
    if ctx.data.xpos[ctx.torso_body_id][2] < height_threshold:
        return True

    torso_up_z = ctx.data.xmat[ctx.torso_body_id][8]
    tilt_angle_deg = np.rad2deg(np.arccos(np.clip(torso_up_z, -1.0, 1.0)))
    return tilt_angle_deg > angle_threshold_deg


def calculate_sine_reference(
    t: float,
    hip_omega: float,
    leg_amp_rad: float,
    start_amp_mult: float,
    start_freq_mult: float,
    ramp_time: float = 0.0,
) -> tuple[float, float]:
    return startup_sine_reference(
        t=t,
        hip_omega=hip_omega,
        leg_amp_rad=leg_amp_rad,
        t_wait=DEFAULT_T_WAIT,
        start_amp_mult=start_amp_mult,
        start_freq_mult=start_freq_mult,
        ramp_time=ramp_time,
    )


def normalize_xy(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    xy = np.asarray(vec[:2], dtype=float)
    norm = np.linalg.norm(xy)
    if norm < 1e-9:
        return fallback.copy()
    return xy / norm


def get_heading_axes(xmat_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return mujoco_heading_axes(xmat_flat)


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
    merged = {**DEFAULT_FIXED_PARAMS, **params}
    return {
        "foot_x": float(merged["foot_x"]),
        "foot_y": float(merged["foot_y"]),
        "torque_limit": float(merged.get("torque_limit", DEFAULT_TORQUE_LIMIT)),
        "Kp": float(merged["Kp"]),
        "Kd": float(merged["Kd"]),
        "start_amp_mult": float(merged["start_amp_mult"]),
        "start_freq_mult": float(merged["start_freq_mult"]),
        "ramp_time": float(merged.get("ramp_time", DEFAULT_STARTUP_RAMP_TIME)),
        "amp_deg": float(merged["amp_deg"]),
        "freq_hz": float(merged["freq_hz"]),
    }


def run_single_trial(
    ctx: SimulationContext,
    params: dict,
    verbose: bool = True,
    viewer=None,
    iteration_duration: float | None = None,
    stop_on_fall: bool = True,
    frame_callback=None,
) -> dict:
    mujoco.mj_resetData(ctx.model, ctx.data)

    replay_params = build_replay_params(params)

    apply_foot_offsets(ctx, replay_params["foot_x"], replay_params["foot_y"])
    mujoco.mj_forward(ctx.model, ctx.data)

    if STARTUP_SETTLE_S > 0:
        settle_print = (
            (lambda *args, **kwargs: vprint(*args, verbose=verbose, **kwargs))
            if verbose
            else None
        )
        startup_settle_orientation(
            model=ctx.model,
            data=ctx.data,
            hip_qpos_adr=ctx.qpos_idx,
            hip_qvel_adr=ctx.qvel_idx,
            torque_limit=replay_params["torque_limit"],
            settle_s=STARTUP_SETTLE_S,
            avg_s=STARTUP_SETTLE_AVG_S,
            settle_kp=replay_params["Kp"],
            settle_kd=replay_params["Kd"],
            print_fn=settle_print,
        )

    if ctx.debug_geom_id != -1:
        vprint(f"Global Position: {ctx.data.geom_xpos[ctx.debug_geom_id]}", verbose=verbose)

    start_torso_xy = ctx.data.xpos[ctx.torso_body_id][:2].copy()
    start_forward_xy, start_left_xy = get_heading_axes(ctx.data.xmat[ctx.torso_body_id])
    hip_omega = replay_params["freq_hz"] * 2 * np.pi
    leg_amp_rad = np.deg2rad(replay_params["amp_deg"])
    cmd_buffer = [float(ctx.data.ctrl[0])] * CMD_DELAY_STEPS
    fell = False
    duration = ITERATION_DURATION if iteration_duration is None else float(iteration_duration)
    max_steps = int(duration / ctx.model.opt.timestep)
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

    if frame_callback is not None:
        frame_callback(ctx)
    for step_idx in range(1, max_steps + 1):
        t = ctx.data.time

        target_pos_rad, target_vel_rad = calculate_sine_reference(
            t,
            hip_omega,
            leg_amp_rad,
            replay_params["start_amp_mult"],
            replay_params["start_freq_mult"],
            replay_params["ramp_time"],
        )

        current_pos = ctx.data.qpos[ctx.qpos_idx]
        current_vel = ctx.data.qvel[ctx.qvel_idx]

        ramp = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
        tau = (
            replay_params["Kp"] * ramp * (target_pos_rad - current_pos)
            + replay_params["Kd"] * ramp * (target_vel_rad - current_vel)
        )
        tau = np.clip(tau, -replay_params["torque_limit"], replay_params["torque_limit"])
        energy_used += abs(tau * current_vel) * ctx.model.opt.timestep

        cmd_buffer.append(tau)
        ctx.data.ctrl[0] = cmd_buffer.pop(0)

        mujoco.mj_step(ctx.model, ctx.data)

        if frame_callback is not None:
            frame_callback(ctx)

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(ctx.model.opt.timestep)

        analysis_due = (step_idx % ANALYSIS_STRIDE) == 0
        if analysis_due:
            update_gait_metrics()
            last_analysis_step = step_idx

        if stop_on_fall and check_has_fallen(ctx):
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
        "Torque_Limit": replay_params["torque_limit"],
        "Kp": replay_params["Kp"],
        "Kd": replay_params["Kd"],
        "Start_Amp_Mult": replay_params["start_amp_mult"],
        "Start_Freq_Mult": replay_params["start_freq_mult"],
        "Ramp_Time": replay_params["ramp_time"],
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
            f"tau_lim={result_row['Torque_Limit']:.1f} "
            f"Kp={params['Kp']:.1f} Kd={params['Kd']:.1f} "
            f"ramp={result_row['Ramp_Time']:.2f}s "
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
    fixed_params = _json_env("SWEEP_FIXED_PARAMS_JSON", DEFAULT_FIXED_PARAMS)
    normal_distributions = _json_env("SWEEP_NORMAL_DISTRIBUTIONS_JSON", DEFAULT_NORMAL_DISTRIBUTIONS)
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

    parameter_rows = generate_parameter_samples(
        num_trials,
        fixed_params=fixed_params,
        normal_distributions=normal_distributions,
    )
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
