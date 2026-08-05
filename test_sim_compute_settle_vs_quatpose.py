"""
Compute-time comparison: dynamic pitch-settle wait vs a per-trial
"average quaternion" pose.

Method A ("settle"): the same as test_sim_sweep_hip_offset_settle.py's
wait=True path -- place on ground, then physically simulate holding the hip
rigid until wait_until_pitch_settled() detects convergence (a peak-to-peak +
dwell check), before starting the gait. Settle time is variable and can run
all the way to the 25s cap.

Method B ("quatpose"): for EACH trial, with that trial's own (foot_x,
foot_y, Kp, Kd), run LEG_AMP_DEG=0 (hip held rigid, no gait) for a FIXED
QUAT_MEASURE_S window -- no convergence check -- and average the torso
quaternion over that window (Markley's method). Averaging over a window that
spans at least one oscillation period cancels out the residual rocking even
before the robot has technically "settled" by the peak-to-peak criterion, so
it can be a faster way to get a good starting pose than waiting for formal
convergence. The trial is then reset, its free-joint quaternion is
teleported directly to that trial-specific average, and the gait starts at
t=0.

Both methods run the identical TRIALS LHS-sampled (foot_x, foot_y, Kp, Kd,
gait-shape) parameter sets, so the trial-to-trial workload is apples-to-apples
-- what's being timed is the wall-clock cost of dynamically waiting for
convergence (A) vs a fixed-duration averaging pass (B). This is primarily a
*compute-time* comparison; fall rate/distance are reported as a side-effect,
not the point.
"""
import time as walltime

import mujoco
import numpy as np
from scipy.stats import qmc

from sim_common import (calculate_sine_reference, pd_torque, apply_foot_offsets,
                         place_on_ground, wait_until_pitch_settled, average_quaternion)

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

JOINT_NAME         = "hip"
TORSO_BODY_NAME    = "motor"
TORQUE_LIMIT       = 25.0
CMD_DELAY_STEPS    = 1
ITERATION_DURATION = 20.0
TRIALS             = 250    # LHS samples (x2 methods = 500 total walking runs)

MAX_SETTLE_WAIT_S         = 25.0
PITCH_SETTLE_WINDOW_S     = 1.0
PITCH_SETTLE_PP_THRESH_DEG = 2.0
PITCH_SETTLE_DWELL_S       = 2.0

QUAT_MEASURE_S = 5.0   # fixed hip-rigid window (using the trial's own Kp/Kd/foot offset)
                        # averaged into that trial's own quaternion -- no convergence check

ENTRY_XML = "bigfoot/scene.xml"

# ── Randomisation ranges [min, max] -- same as test_sim_sweep_hip_offset_settle.py ─
RANGES = {
    'foot_x':          (-0.07,  0.025),
    'foot_y':          (-0.02,  0.01),
    'Kp':              ( 20.0,  45.0 ),
    'Kd':              ( 2.0,  15.0),
    'start_amp_mult':  ( 0.8,   1.8),
    'start_freq_mult': ( 0.7,   1.4),
    'amp_deg':         (20.0,  40.0),
    'freq_hz':         ( 0.4,   0.8),
}

FOOT_GEOM_NAMES = [
    "right_foot_1",     "right_foot_2",     "right_foot_3",
    "left_foot_1",      "left_foot_2",      "left_foot_3",
    "right_foot_1_col", "right_foot_2_col", "right_foot_3_col",
    "left_foot_1_col",  "left_foot_2_col",  "left_foot_3_col",
]

FOOT_COLLISION_GEOMS = [
    "right_foot_1_col", "right_foot_2_col", "right_foot_3_col",
    "left_foot_1_col",  "left_foot_2_col",  "left_foot_3_col",
]


def generate_lhs_samples(n_trials, seed):
    sampler = qmc.LatinHypercube(d=len(RANGES), seed=seed)
    samples = sampler.random(n=n_trials)

    keys        = sorted(RANGES.keys())
    bounds_low  = [RANGES[k][0] for k in keys]
    bounds_high = [RANGES[k][1] for k in keys]
    scaled      = qmc.scale(samples, bounds_low, bounds_high)

    return [dict(zip(keys, row)) for row in scaled]


def check_has_fallen(data, body_id, height_threshold=0.5, angle_threshold_deg=45.0):
    if data.xpos[body_id][2] < height_threshold:
        return True
    torso_up_z     = data.xmat[body_id][8]
    tilt_angle_deg = np.rad2deg(np.arccos(np.clip(torso_up_z, -1.0, 1.0)))
    return tilt_angle_deg > angle_threshold_deg


def compute_avg_quat_for_trial(model, data, hip_qpos_idx, hip_qvel_idx, torso_body_id, free_z_qpos_idx,
                                foot_parent_body_ids, original_geom_pos, original_body_ipos, params):
    """LEG_AMP_DEG=0 (hip held rigid, no gait) for a fixed QUAT_MEASURE_S
    window, using this trial's own foot offset and gains -- no convergence
    check, just average whatever the pitch does over that window."""
    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, params['foot_x'], params['foot_y'], FOOT_GEOM_NAMES,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos)
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)

    quats = []
    max_steps = int(QUAT_MEASURE_S / model.opt.timestep)
    for _ in range(max_steps):
        current_pos = data.qpos[hip_qpos_idx]
        current_vel = data.qvel[hip_qvel_idx]
        data.ctrl[0] = pd_torque(0.0, 0.0, current_pos, current_vel, params['Kp'], params['Kd'], TORQUE_LIMIT)
        mujoco.mj_step(model, data)
        quats.append(data.xquat[torso_body_id].copy())

    return average_quaternion(quats)


def run_gait(model, data, torso_body_id, qpos_idx, qvel_idx, t_wait, params):
    """Runs the ITERATION_DURATION gait from wherever data is currently
    posed, starting the trajectory at t_wait. Shared by both methods."""
    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    gravity    = 9.81

    start_x    = data.xpos[torso_body_id][0]
    start_y    = data.xpos[torso_body_id][1]
    hip_omega  = params['freq_hz'] * 2 * np.pi
    leg_amp_rad = np.deg2rad(params['amp_deg'])
    cmd_buffer = [0.0] * CMD_DELAY_STEPS
    fell       = False
    max_steps  = int(ITERATION_DURATION / model.opt.timestep)
    energy_used = 0.0

    for step in range(max_steps):
        t = data.time

        target_pos_rad, target_vel_rad = calculate_sine_reference(
            t, hip_omega, leg_amp_rad, params['start_amp_mult'], params['start_freq_mult'], t_wait=t_wait
        )

        current_pos = data.qpos[qpos_idx]
        current_vel = data.qvel[qvel_idx]

        tau = pd_torque(target_pos_rad, target_vel_rad, current_pos, current_vel,
                         params['Kp'], params['Kd'], TORQUE_LIMIT)
        energy_used += abs(tau * current_vel) * model.opt.timestep

        cmd_buffer.append(tau)
        data.ctrl[0] = cmd_buffer.pop(0)

        mujoco.mj_step(model, data)

        if t > t_wait and check_has_fallen(data, torso_body_id):
            fell = True
            break

    final_x  = data.xpos[torso_body_id][0]
    final_y  = data.xpos[torso_body_id][1]
    distance = float(np.sqrt((final_x - start_x) ** 2 + (final_y - start_y) ** 2))
    cot = (energy_used / (total_mass * gravity * distance)) if distance > 1e-6 else float('inf')

    # Actual time spent walking -- data.time at exit minus t_wait, not the nominal
    # ITERATION_DURATION, since a fall ends the loop early. Guards against
    # over-crediting velocity to trials that barely got moving before falling.
    walk_time = max(0.0, data.time - t_wait)
    avg_velocity = distance / walk_time if walk_time > 1e-6 else None

    return fell, distance, cot, avg_velocity


def run_trial_settle(model, data, torso_body_id, qpos_idx, qvel_idx, free_z_qpos_idx,
                      foot_parent_body_ids, original_geom_pos, original_body_ipos, params):
    """Method A: dynamic per-trial settle wait."""
    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, params['foot_x'], params['foot_y'], FOOT_GEOM_NAMES,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos)
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)

    t_wait = wait_until_pitch_settled(model, data, qpos_idx, qvel_idx, torso_body_id,
                                       params['Kp'], params['Kd'], TORQUE_LIMIT,
                                       max_wait_s=MAX_SETTLE_WAIT_S, window_s=PITCH_SETTLE_WINDOW_S,
                                       pp_thresh_deg=PITCH_SETTLE_PP_THRESH_DEG, dwell_s=PITCH_SETTLE_DWELL_S)

    fell, distance, cot, avg_velocity = run_gait(model, data, torso_body_id, qpos_idx, qvel_idx, t_wait, params)
    return {"Fell": fell, "Distance_Traversed": round(distance, 4), "CoT": round(cot, 4),
            "Settle_Time_S": round(t_wait, 4), "Avg_Velocity": avg_velocity}


def run_trial_quatpose(model, data, torso_body_id, qpos_idx, qvel_idx, free_z_qpos_idx, free_quat_idx,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos, params):
    """Method B: compute this trial's own average quaternion over a fixed
    QUAT_MEASURE_S window (own foot offset/gains), then teleport to it --
    no convergence check, no dynamic wait."""
    avg_quat = compute_avg_quat_for_trial(model, data, qpos_idx, qvel_idx, torso_body_id, free_z_qpos_idx,
                                           foot_parent_body_ids, original_geom_pos, original_body_ipos, params)

    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, params['foot_x'], params['foot_y'], FOOT_GEOM_NAMES,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos)
    data.qpos[free_quat_idx:free_quat_idx + 4] = avg_quat
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)

    t_wait = 0.0
    fell, distance, cot, avg_velocity = run_gait(model, data, torso_body_id, qpos_idx, qvel_idx, t_wait, params)
    return {"Fell": fell, "Distance_Traversed": round(distance, 4), "CoT": round(cot, 4),
            "Settle_Time_S": round(t_wait, 4), "Quat_Measure_S": QUAT_MEASURE_S, "Avg_Velocity": avg_velocity}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    model = mujoco.MjModel.from_xml_path(ENTRY_XML)
    data  = mujoco.MjData(model)

    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
    joint_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)
    if joint_id == -1 or torso_body_id == -1:
        raise ValueError(f"Could not find joint '{JOINT_NAME}' or body '{TORSO_BODY_NAME}' in {ENTRY_XML}.")

    qpos_idx = model.jnt_qposadr[joint_id]
    qvel_idx = model.jnt_dofadr[joint_id]

    free_joint_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "motor_freejoint")
    free_qpos_start = model.jnt_qposadr[free_joint_id]
    free_z_qpos_idx = free_qpos_start + 2
    free_quat_idx   = free_qpos_start + 3

    original_geom_pos = {}
    for name in FOOT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id != -1:
            original_geom_pos[name] = model.geom_pos[geom_id].copy()

    foot_parent_body_ids = set()
    for name in FOOT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id != -1:
            foot_parent_body_ids.add(model.geom_bodyid[geom_id])
    original_body_ipos = {bid: model.body_ipos[bid].copy() for bid in foot_parent_body_ids}

    all_params = generate_lhs_samples(TRIALS, seed=42)

    # ── Method A: dynamic settle, every trial ───────────────────────────────────
    settle_results = []
    t0 = walltime.perf_counter()
    for i, params in enumerate(all_params, 1):
        r = run_trial_settle(model, data, torso_body_id, qpos_idx, qvel_idx, free_z_qpos_idx,
                              foot_parent_body_ids, original_geom_pos, original_body_ipos, params)
        settle_results.append(r)
        if i % 25 == 0 or i == TRIALS:
            print(f"[settle  {i:>3}/{TRIALS}] fell={r['Fell']} dist={r['Distance_Traversed']:.2f}m settle_t={r['Settle_Time_S']:.2f}s")
    settle_wall_s = walltime.perf_counter() - t0

    # ── Method B: per-trial average-quaternion pose (own foot offset/gains) ─────
    quatpose_results = []
    t0 = walltime.perf_counter()
    for i, params in enumerate(all_params, 1):
        r = run_trial_quatpose(model, data, torso_body_id, qpos_idx, qvel_idx, free_z_qpos_idx, free_quat_idx,
                                foot_parent_body_ids, original_geom_pos, original_body_ipos, params)
        quatpose_results.append(r)
        if i % 25 == 0 or i == TRIALS:
            print(f"[quatpose {i:>3}/{TRIALS}] fell={r['Fell']} dist={r['Distance_Traversed']:.2f}m")
    quatpose_wall_s = walltime.perf_counter() - t0

    # ═══════════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    settle_falls = sum(1 for r in settle_results if r["Fell"])
    quatpose_falls = sum(1 for r in quatpose_results if r["Fell"])
    settle_dist = sum(r["Distance_Traversed"] for r in settle_results) / TRIALS
    quatpose_dist = sum(r["Distance_Traversed"] for r in quatpose_results) / TRIALS

    print("\n" + "=" * 78)
    print(f"COMPUTE TIME -- {TRIALS} trials each")
    print("=" * 78)
    print(f"Method A (dynamic settle each trial):        {settle_wall_s:.2f}s wall  "
          f"({settle_wall_s / TRIALS * 1000:.1f} ms/trial)")
    print(f"Method B (per-trial avg-quat pose, {QUAT_MEASURE_S:.0f}s fixed window): {quatpose_wall_s:.2f}s wall  "
          f"({quatpose_wall_s / TRIALS * 1000:.1f} ms/trial)")
    speedup = settle_wall_s / quatpose_wall_s if quatpose_wall_s > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    print()
    print(f"Fall rate  -- settle: {settle_falls}/{TRIALS} ({settle_falls/TRIALS*100:.1f}%)   "
          f"quatpose: {quatpose_falls}/{TRIALS} ({quatpose_falls/TRIALS*100:.1f}%)")
    print(f"Mean dist  -- settle: {settle_dist:.3f}m   quatpose: {quatpose_dist:.3f}m")

    settle_vels   = [r["Avg_Velocity"] for r in settle_results   if r["Avg_Velocity"] is not None]
    quatpose_vels = [r["Avg_Velocity"] for r in quatpose_results if r["Avg_Velocity"] is not None]
    settle_mean_v   = sum(settle_vels) / len(settle_vels) if settle_vels else float('nan')
    quatpose_mean_v = sum(quatpose_vels) / len(quatpose_vels) if quatpose_vels else float('nan')
    print(f"Avg velocity -- settle: {settle_mean_v:.4f} m/s (n={len(settle_vels)}/{TRIALS})   "
          f"quatpose: {quatpose_mean_v:.4f} m/s (n={len(quatpose_vels)}/{TRIALS})")


if __name__ == "__main__":
    main()
