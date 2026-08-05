import csv
import os
import mujoco
import numpy as np
from scipy.stats import qmc

from sim_common import (calculate_sine_reference, pd_torque, apply_foot_offsets,
                         place_on_ground, measure_avg_quaternion_pose)

def generate_lhs_samples(n_trials):
    """
    Latin Hypercube sampler for space coverage.
    """
    sampler   = qmc.LatinHypercube(d=8, seed=42)
    samples   = sampler.random(n=n_trials)

    keys        = sorted(list(RANGES.keys()))

    # Now the columns will map to the exact same parameters every time
    bounds_low  = [RANGES[k][0] for k in keys]
    bounds_high = [RANGES[k][1] for k in keys]
    scaled      = qmc.scale(samples, bounds_low, bounds_high)

    return [dict(zip(keys, row)) for row in scaled]

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
MIN_DISTANCE       = 2.0    # metres — minimum to save a result
NUM_TRIALS         = 500  # ← change this to run more or fewer trials
QUAT_MEASURE_S     = 5.0    # fixed hip-rigid window (own Kp/Kd/foot offset) averaged into the pose

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

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

model = mujoco.MjModel.from_xml_path("modified_model_y10pct.xml")
data  = mujoco.MjData(model)
TOTAL_MASS = sum(model.body_mass[i] for i in range(model.nbody))
GRAVITY    = 9.81

torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
joint_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)

if joint_id == -1 or torso_body_id == -1:
    raise ValueError(f"Could not find joint '{JOINT_NAME}' or body '{TORSO_BODY_NAME}'.")

qpos_idx = model.jnt_qposadr[joint_id]
qvel_idx = model.jnt_dofadr[joint_id]

free_joint_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "motor_freejoint")
free_qpos_start = model.jnt_qposadr[free_joint_id]
free_z_qpos_idx = free_qpos_start + 2
free_quat_idx   = free_qpos_start + 3

FOOT_COLLISION_GEOMS = [
    "right_foot_1_col", "right_foot_2_col", "right_foot_3_col",
    "left_foot_1_col",  "left_foot_2_col",  "left_foot_3_col",
]

# ── Snapshot original geom positions BEFORE any modification ─────────────────
original_geom_pos = {}
for name in FOOT_GEOM_NAMES:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id != -1:
        original_geom_pos[name] = model.geom_pos[geom_id].copy()

# Find the unique parent body IDs for the feet geoms
foot_parent_body_ids = set()
for name in FOOT_GEOM_NAMES:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id != -1:
        foot_parent_body_ids.add(model.geom_bodyid[geom_id])

# Store a clean baseline copy of the parent body inertial positions
original_body_ipos = {bid: model.body_ipos[bid].copy() for bid in foot_parent_body_ids}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def check_has_fallen(body_id, height_threshold=0.5, angle_threshold_deg=45.0):
    if data.xpos[body_id][2] < height_threshold:
        return True

    # Extract the world-Z component from the flat 1D matrix view
    # Index 8 is the local Z-axis projecting onto world Z
    torso_up_z     = data.xmat[body_id][8]
    # NOTE: If your CAD model was built with the Y-axis pointing up, change index to 5
    # NOTE: If your CAD model was built with the X-axis pointing up, change index to 2

    tilt_angle_deg = np.rad2deg(np.arccos(np.clip(torso_up_z, -1.0, 1.0)))
    return tilt_angle_deg > angle_threshold_deg


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

results = []
all_params = generate_lhs_samples(NUM_TRIALS)
print(f"Starting random sweep: {NUM_TRIALS} trials. Saving results with no fall and >{MIN_DISTANCE}m distance.")

for trial_idx, params in enumerate(all_params, 1):

    # Pass 1: measure this trial's own average quaternion (own foot offset/gains) --
    # hold the hip rigid over a fixed window instead of free-falling from the
    # XML's guessed 1.2m spawn height or waiting for a dynamic convergence check.
    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, params['foot_x'], params['foot_y'], FOOT_GEOM_NAMES,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos)
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)
    avg_quat = measure_avg_quaternion_pose(model, data, qpos_idx, qvel_idx, torso_body_id,
                                            params['Kp'], params['Kd'], TORQUE_LIMIT,
                                            measure_s=QUAT_MEASURE_S)

    # Pass 2: fresh reset, teleport straight to that quaternion, walk from t=0.
    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, params['foot_x'], params['foot_y'], FOOT_GEOM_NAMES,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos)
    data.qpos[free_quat_idx:free_quat_idx + 4] = avg_quat
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)
    t_wait = 0.0

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_3_col")
    print(f"Global Position: {data.geom_xpos[geom_id]}")

    start_x    = data.xpos[torso_body_id][0]
    start_y    = data.xpos[torso_body_id][1]
    hip_omega  = params['freq_hz'] * 2 * np.pi
    leg_amp_rad = np.deg2rad(params['amp_deg'])
    cmd_buffer = [0.0] * CMD_DELAY_STEPS
    fell       = False
    max_steps  = int(ITERATION_DURATION / model.opt.timestep)
    energy_used = 0.0   # accumulated mechanical work, J

    for step in range(max_steps):
        t = data.time

        target_pos_rad, target_vel_rad = calculate_sine_reference(
            t, hip_omega, leg_amp_rad, params['start_amp_mult'], params['start_freq_mult'], t_wait=t_wait
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

        if t > t_wait and check_has_fallen(torso_body_id):
            print(f"   [DEBUG] Fell at t={data.time:.3f}s. Height={data.xpos[torso_body_id][2]:.2f}m")
            fell = True
            break

    final_x  = data.xpos[torso_body_id][0]
    final_y  = data.xpos[torso_body_id][1]
    distance = float(np.sqrt((final_x - start_x)**2 + (final_y - start_y)**2))
    cot = (energy_used / (TOTAL_MASS * GRAVITY * distance)) if distance > 1e-6 else float('inf')
    passed   = not fell and distance >= MIN_DISTANCE

    print(f"[{trial_idx:>5}/{NUM_TRIALS}] "
          f"freq={params['freq_hz']:.2f}Hz amp={params['amp_deg']:.1f}° "
          f"Kp={params['Kp']:.1f} Kd={params['Kd']:.1f} "
          f"fx={params['foot_x']:.3f} fy={params['foot_y']:.3f} | "
          f"fell={fell} dist={distance:.2f}m CoT={cot:.3f}| {'SAVED' if passed else 'skipped'}")

    if passed or (not passed and distance >= MIN_DISTANCE):
        results.append({
            "Foot_X":            round(params['foot_x'],          4),
            "Foot_Y":            round(params['foot_y'],          4),
            "Kp":                round(params['Kp'],              2),
            "Kd":                round(params['Kd'],              2),
            "Start_Amp_Mult":    round(params['start_amp_mult'],  3),
            "Start_Freq_Mult":   round(params['start_freq_mult'], 3),
            "Amplitude_Deg":     round(params['amp_deg'],         2),
            "Frequency_Hz":      round(params['freq_hz'],         3),
            "Fell":              fell,
            "Distance_Traversed": round(distance,                 4),
            "CoT": round(cot, 4),
        })

print(f"Save rate: {len(results)}/{NUM_TRIALS} = {len(results)/NUM_TRIALS*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

if results:
    os.makedirs("results", exist_ok=True)
    csv_file = "results/sweep_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSweep complete. {len(results)}/{NUM_TRIALS} trials saved to '{csv_file}'.")
else:
    print(f"\nSweep complete. No trials met the criteria (no fall + >{MIN_DISTANCE}m).")
    print("Consider reducing MIN_DISTANCE or increasing NUM_TRIALS.")