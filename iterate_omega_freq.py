import csv
import json
import mujoco
import numpy as np

from coordinate_frame import public_to_mujoco_vec
from control_waveform import DEFAULT_KP as KP, DEFAULT_KD as KD, DEFAULT_TORQUE_LIMIT as TORQUE_LIMIT
from control_waveform import startup_sine_reference

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION & CONSTANTS — Matched to single simulation / hardware
# ═══════════════════════════════════════════════════════════════════════════════
# Hardware/XML Mapping Names
JOINT_NAME      = "hip"  # <--- Replace with your actual XML joint name
TORSO_BODY_NAME = "motor"               # Main root body for fall detection

# Motor / control (Direct SI units)

# Trajectory Timing / Tweaks
T_WAIT          = 1.0
START_FREQ_MULT = 1.5
START_AMP_MULT  = 1.4
STARTUP_RAMP_TIME = 0.0

USE_RAMP           = False
RAMP_TIME          = 1.0
CMD_DELAY_STEPS    = 1
ITERATION_DURATION = 20.0  # Seconds per test run

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT GRID GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
frequencies = np.arange(0.35, 0.75 + 0.01, 0.02)       
amplitudes  = np.arange(5, 38 + 1, 1)                 
results     = []

# Foot Offsets
# Public frame: +x forward, +y robot-left, +z down.
# MuJoCo frame: +x forward, +y robot-left, +z up.
right_foot_offset_public = np.array([0.0, -0.01, 0.0])
left_foot_offset_public = np.array([0.0, 0.01, 0.0])
foot_position_deltaRight = public_to_mujoco_vec(right_foot_offset_public)
foot_position_deltaLeft = public_to_mujoco_vec(left_foot_offset_public)
foot_geom_offsets = {
    "right_foot_1": foot_position_deltaRight, "right_foot_2": foot_position_deltaRight, "right_foot_3": foot_position_deltaRight,
    "left_foot_1": foot_position_deltaLeft,   "left_foot_2": foot_position_deltaLeft,   "left_foot_3": foot_position_deltaLeft,
    "right_foot_1_col": foot_position_deltaRight, "right_foot_2_col": foot_position_deltaRight, "right_foot_3_col": foot_position_deltaRight,
    "left_foot_1_col": foot_position_deltaLeft,   "left_foot_2_col": foot_position_deltaLeft,   "left_foot_3_col": foot_position_deltaLeft,
}

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE MODEL & IDENTIFY INDICES
# ═══════════════════════════════════════════════════════════════════════════════
model = mujoco.MjModel.from_xml_path("bigfoot/scene.xml")
data  = mujoco.MjData(model)

# Dynamic ID Lookups to pull state from the correct addresses
torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
joint_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)

if joint_id == -1 or torso_body_id == -1:
    raise ValueError(f"Could not find joint '{JOINT_NAME}' or body '{TORSO_BODY_NAME}' in XML. Verify names.")

qpos_idx = model.jnt_qposadr[joint_id]
qvel_idx = model.jnt_dofadr[joint_id]
'''
# Apply geometry offsets
for name, delta in foot_geom_offsets.items():
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id != -1:
        model.geom_pos[geom_id] += delta

mujoco.mj_setConst(model, data)
'''
for name, delta in foot_geom_offsets.items():
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id != -1:
        # 1. Shift the collision shape
        model.geom_pos[geom_id] += delta
        
        # 2. Find the parent body holding the mass for this geom
        parent_body_id = model.geom_bodyid[geom_id]
        
        # 3. Shift the heavy inertial frame by the exact same amount
        model.body_ipos[parent_body_id] += delta

# Recompute the system constants after shifting masses
mujoco.mj_setConst(model, data)
#mujoco.mj_forward(model, data)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_sine_reference(t, hip_omega, leg_amp_rad):
    return startup_sine_reference(
        t=t,
        hip_omega=hip_omega,
        leg_amp_rad=leg_amp_rad,
        t_wait=T_WAIT,
        start_amp_mult=START_AMP_MULT,
        start_freq_mult=START_FREQ_MULT,
        ramp_time=STARTUP_RAMP_TIME,
    )

def check_has_fallen(model, data, body_id, height_threshold=0.3, angle_threshold_deg=45.0):
    """Evaluates torso spatial metrics to classify falls."""
    # 1. Height Drop Check
    torso_z = data.xpos[body_id][2]
    if torso_z < height_threshold:
        return True

    # 2. Tilt Axis Angle Check
    torso_up_z = data.xmat[body_id][8]
    tilt_angle_deg = np.rad2deg(np.arccos(np.clip(torso_up_z, -1.0, 1.0)))
    if tilt_angle_deg > angle_threshold_deg:
        return True

    return False


print(f"Starting parameter sweep: {len(frequencies)} Frequencies x {len(amplitudes)} Amplitudes = {len(frequencies)*len(amplitudes)} total runs.")

run_idx = 1
for freq_hz in frequencies:
    for amp_deg in amplitudes:
        
        # Reset simulation state completely for the next run
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 1.0  # drop height initialization
        mujoco.mj_forward(model, data)
        
        # --- ADD THIS: Record the starting coordinates ---
        start_x = data.xpos[torso_body_id][0]
        start_y = data.xpos[torso_body_id][1]
        distance_traversed = 0.0

        # Run parameters mapping
        hip_omega   = freq_hz * 2 * np.pi
        leg_amp_rad = np.deg2rad(amp_deg)
        cmd_buffer  = [0.0] * CMD_DELAY_STEPS
        
        steps_taken = 0
        fell        = False
        max_steps   = int(ITERATION_DURATION / model.opt.timestep)
        
        # Physics stepping loop
        for step in range(max_steps):
            t = data.time
            
            # Radians-based target generations
            target_pos_rad, target_vel_rad = calculate_sine_reference(t, hip_omega, leg_amp_rad)
            
            # Raw state readings in radians and rad/s
            current_pos = data.qpos[qpos_idx]
            current_vel = data.qvel[qvel_idx]
            
            # Ramping calculation
            ramp      = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
            Kp_ramped = KP * ramp
            Kd_ramped = KD * ramp
            
            # PD Loop matching your hardware specifications
            tau = (Kp_ramped * (target_pos_rad - current_pos) +
                   Kd_ramped * (target_vel_rad - current_vel))
            
            # Bounding output to actual torque capacity
            tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
            
            # Manage CAN latency representation
            cmd_buffer.append(tau)
            data.ctrl[0] = cmd_buffer.pop(0)
            
            # Evaluate simulation change
            mujoco.mj_step(model, data)
            steps_taken += 1
            
            # Assess fall termination condition
            if check_has_fallen(model, data, torso_body_id, height_threshold=0.3, angle_threshold_deg=45.0):
                fell = True
                break
            # Physics stepping loop ends here
        
            # --- ADD THIS: Calculate total linear distance traversed ---
            if not fell:
                final_x = data.xpos[torso_body_id][0]
                final_y = data.xpos[torso_body_id][1]
                distance_traversed = float(np.sqrt((final_x - start_x)**2 + (final_y - start_y)**2))
            else:
                distance_traversed = 0.0 # Penalize distance if it fell over
                
        # Aggregate performance metadata
        result_entry = {
            "Frequency_Hz": round(freq_hz, 2),
            "Amplitude_Deg": int(amp_deg),
            "Fell": fell,
            "Steps_Taken": steps_taken,
            "Distance_Traversed": round(distance_traversed, 4),
        }
        results.append(result_entry)
        
        print(f"Run {run_idx}/{len(frequencies)*len(amplitudes)} | Freq: {freq_hz:.2f}Hz, Amp: {amp_deg}° | Fell: {fell} | Steps: {steps_taken} | Dist: {distance_traversed:.2f}m")
        run_idx += 1

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT DATA
# ═══════════════════════════════════════════════════════════════════════════════
csv_file = "simulation_results.csv"
keys     = results[0].keys()

with open(csv_file, 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(results)

print(f"\nSweep completed! Results saved to '{csv_file}'.")
