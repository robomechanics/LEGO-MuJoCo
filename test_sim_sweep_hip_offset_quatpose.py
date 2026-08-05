"""
Hip-offset randomization sweep, pose-initialized via the per-trial average
quaternion method (see test_sim_compute_settle_vs_quatpose.py for the
compute-time/accuracy comparison against a dynamic pitch-settle wait --
quatpose is ~2x faster and at least as good on distance/velocity, so it's
now the standard pose-init method for the sweep scripts).

For each LHS-sampled (foot_x, foot_y, Kp, Kd, gait-shape) trial: place on
the ground, hold the hip rigid for a fixed QUAT_MEASURE_S window and average
the torso quaternion over it (measure_avg_quaternion_pose), then reset,
teleport the free joint straight to that average quaternion, and start the
gait immediately (t_wait=0).
"""
import csv
import os
import mujoco
import numpy as np
from scipy.stats import qmc

from sim_common import (calculate_sine_reference, pd_torque, apply_foot_offsets,
                         place_on_ground, measure_avg_quaternion_pose)

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

JOINT_NAME         = "hip"
TORSO_BODY_NAME    = "motor"
TORQUE_LIMIT       = 25.0
USE_RAMP           = False
RAMP_TIME          = 1.0
CMD_DELAY_STEPS    = 1
ITERATION_DURATION = 20.0
MIN_DISTANCE       = 2.0    # metres -- minimum to save a result
TRIALS             = 500
QUAT_MEASURE_S      = 5.0   # fixed hip-rigid window (own Kp/Kd/foot offset) averaged into the pose

ENTRY_XML = "bigfoot/scene.xml"

# ── Randomisation ranges [min, max] ───────────────────────────────────────────
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


def run_trial(model, data, torso_body_id, qpos_idx, qvel_idx, free_z_qpos_idx, free_quat_idx,
              foot_parent_body_ids, original_geom_pos, original_body_ipos, params):
    # Pass 1: measure this trial's own average quaternion (own foot offset/gains).
    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, params['foot_x'], params['foot_y'], FOOT_GEOM_NAMES,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos)
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)
    avg_quat = measure_avg_quaternion_pose(model, data, qpos_idx, qvel_idx, torso_body_id,
                                            params['Kp'], params['Kd'], TORQUE_LIMIT,
                                            measure_s=QUAT_MEASURE_S)

    # Pass 2: fresh reset, teleport to that quaternion, walk from t=0.
    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, params['foot_x'], params['foot_y'], FOOT_GEOM_NAMES,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos)
    data.qpos[free_quat_idx:free_quat_idx + 4] = avg_quat
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)

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
            t, hip_omega, leg_amp_rad, params['start_amp_mult'], params['start_freq_mult'], t_wait=0.0
        )

        current_pos = data.qpos[qpos_idx]
        current_vel = data.qvel[qvel_idx]

        ramp = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
        tau  = pd_torque(target_pos_rad, target_vel_rad, current_pos, current_vel,
                         params['Kp'], params['Kd'], TORQUE_LIMIT, ramp=ramp)
        energy_used += abs(tau * current_vel) * model.opt.timestep

        cmd_buffer.append(tau)
        data.ctrl[0] = cmd_buffer.pop(0)

        mujoco.mj_step(model, data)

        if check_has_fallen(data, torso_body_id):
            fell = True
            break

    final_x  = data.xpos[torso_body_id][0]
    final_y  = data.xpos[torso_body_id][1]
    distance = float(np.sqrt((final_x - start_x) ** 2 + (final_y - start_y) ** 2))
    cot = (energy_used / (total_mass * gravity * distance)) if distance > 1e-6 else float('inf')

    walk_time = data.time
    avg_velocity = distance / walk_time if walk_time > 1e-6 else None

    return {
        "Fell": fell,
        "Distance_Traversed": round(distance, 4),
        "CoT": round(cot, 4),
        "Avg_Velocity": avg_velocity,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    results = []

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

    for i, params in enumerate(all_params, 1):
        trial = run_trial(model, data, torso_body_id, qpos_idx, qvel_idx, free_z_qpos_idx, free_quat_idx,
                           foot_parent_body_ids, original_geom_pos, original_body_ipos, params)

        passed = not trial["Fell"] and trial["Distance_Traversed"] >= MIN_DISTANCE

        vel_str = f"{trial['Avg_Velocity']:.4f}m/s" if trial['Avg_Velocity'] is not None else "n/a"
        print(f"[{i:>3}/{TRIALS}] freq={params['freq_hz']:.2f}Hz amp={params['amp_deg']:.1f}deg "
              f"Kp={params['Kp']:.1f} Kd={params['Kd']:.1f} "
              f"fx={params['foot_x']:.3f} fy={params['foot_y']:.3f} | "
              f"fell={trial['Fell']} dist={trial['Distance_Traversed']:.2f}m "
              f"CoT={trial['CoT']:.3f} vel={vel_str} | "
              f"{'SAVED' if passed else 'skipped'}")

        results.append({
            "Foot_X":             round(params['foot_x'], 4),
            "Foot_Y":             round(params['foot_y'], 4),
            "Kp":                 round(params['Kp'], 2),
            "Kd":                 round(params['Kd'], 2),
            "Start_Amp_Mult":     round(params['start_amp_mult'], 3),
            "Start_Freq_Mult":    round(params['start_freq_mult'], 3),
            "Amplitude_Deg":      round(params['amp_deg'], 2),
            "Frequency_Hz":       round(params['freq_hz'], 3),
            "Fell":               trial["Fell"],
            "Distance_Traversed": trial["Distance_Traversed"],
            "CoT":                trial["CoT"],
            "Avg_Velocity":       trial["Avg_Velocity"],
            "Passed":             passed,
        })

    os.makedirs("results", exist_ok=True)
    with open("results/sweep_hip_offset_quatpose.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    saved = sum(1 for r in results if r["Passed"])
    print(f"\nSweep complete. {saved}/{len(results)} trials passed (no fall + >{MIN_DISTANCE}m). "
          f"Wrote {len(results)} rows to 'sweep_hip_offset_quatpose.csv'.")


if __name__ == "__main__":
    main()
