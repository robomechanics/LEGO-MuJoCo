import mujoco
import mujoco.viewer
import numpy as np
import time
import pickle as pkl

# ═══════════════════════════════════════════════════════════════════════════════
# USER PARAMETERS — match these directly to motorwave.py for hardware replication
# ═══════════════════════════════════════════════════════════════════════════════

# ── Motor / control ───────────────────────────────────────────────────────────
KP           = 42.0  
KD           = 6.7
TORQUE_LIMIT = 25.0       # Nm — matches MIT_Params T_max and gear in XML

# ── Trajectory ────────────────────────────────────────────────────────────────
HIP_OMEGA       = 0.57 * 2 * np.pi #natural freq should be 0.52 Hz
LEG_AMP_DEG     = 37.5
T_WAIT          = 3.0
START_FREQ_MULT = 0.9
START_AMP_MULT  = 1.3

# FOOT OFFSETS
#If Right = [a, b, c], then left = [-c, -b, a]
# for right: [Pos = shift left (inward), Pos = shift forward, pos = shift up]
# For left: [Pos = down, Pos = shift backward, pos = shift right (inward)]
#Notes: Position of Y shift @0.0 reflects the second slot pretty well, but not perfectly 
foot_position_deltaRight = np.array([0.0, 0.0, 0.0])
foot_position_deltaLeft  = np.array([0.0, -foot_position_deltaRight[1], 0.0])
# ═══════════════════════════════════════════════════════════════════════════════

# True spatial mirroring: invert X (lateral), keep Y (forward) identical, keep Z (vertical) 0
body_deltaRight = np.array([foot_position_deltaRight[0], foot_position_deltaRight[1], foot_position_deltaRight[2]])
body_deltaLeft  = np.array([-foot_position_deltaRight[0], foot_position_deltaRight[1], foot_position_deltaRight[2]]) # Symmetrical world shift

# ── Gain ramp ─────────────────────────────────────────────────────────────────
USE_RAMP  = False
RAMP_TIME = 2.0

# ── CAN latency simulation ────────────────────────────────────────────────────
CMD_DELAY_STEPS = 1

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

model = mujoco.MjModel.from_xml_path("bigfoot/scene.xml")
data  = mujoco.MjData(model)

motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "motor")
arm_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "simplified_motor___arm_rod")

#=========

# 1. Store clean baseline arrays
original_geom_pos = np.array(model.geom_pos)
original_body_ipos = np.array(model.body_ipos)
##"Mirrored" foot is the right foot
# Map visual geoms to their rotated offsets
geom_offsets = {
    "right_foot_1":     foot_position_deltaRight, "right_foot_2":     foot_position_deltaRight, "right_foot_3":     foot_position_deltaRight,
    "left_foot_1":      foot_position_deltaLeft,  "left_foot_2":      foot_position_deltaLeft,  "left_foot_3":      foot_position_deltaLeft,
    "right_foot_1_col": foot_position_deltaRight, "right_foot_2_col": foot_position_deltaRight, "right_foot_3_col": foot_position_deltaRight,
    "left_foot_1_col":  foot_position_deltaLeft,  "left_foot_2_col":  foot_position_deltaLeft,  "left_foot_3_col":  foot_position_deltaLeft,
}

# 2. Shift the visual/collision boxes using your custom mesh rotations
for name, delta in geom_offsets.items():
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id != -1:
        model.geom_pos[geom_id] = original_geom_pos[geom_id] + delta

# 3. Shift the unique parent bodies using clean, non-rotated world coordinates
unique_body_shifts = {}
for name in geom_offsets.keys():
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id != -1:
        parent_bid = model.geom_bodyid[geom_id]
        # Assign world-aligned body offsets depending on if it's a left or right link
        if "right" in name:
            unique_body_shifts[parent_bid] = body_deltaRight
        else:
            unique_body_shifts[parent_bid] = body_deltaLeft

for bid, delta in unique_body_shifts.items():
    model.body_ipos[bid] = original_body_ipos[bid] + delta



print(f"Whole-robot CoM: {data.subtree_com[motor_id].round(4)}")
print(f"Total mass:      {sum(model.body_mass[i] for i in range(model.nbody)):.3f} kg")

# ── Derived constants ─────────────────────────────────────────────────────────
leg_amp_rad = np.deg2rad(LEG_AMP_DEG)
cmd_buffer  = [0.0] * CMD_DELAY_STEPS


def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """Vectorized (N,4) quat[w,x,y,z] -> (N,3) [roll, pitch, yaw] in radians."""
    q = np.atleast_2d(quat_wxyz)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
 
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
 
    # Pitch (y-axis rotation)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
 
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
 
    return np.column_stack([roll, pitch, yaw])

# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY — direct port of motorwave.py calculate_sine_reference
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sine_reference(t):
    w1           = HIP_OMEGA * START_FREQ_MULT
    w2           = HIP_OMEGA
    t0           = T_WAIT
    At           = START_AMP_MULT * leg_amp_rad
    As           = leg_amp_rad
    t_transition = t0 + np.pi / w1

    if t <= t0:
        position, velocity = 0.0, 0.0

    elif t < t_transition:
        phase    = w1 * (t - t0)
        position = At * np.sin(phase)
        velocity = At * w1 * np.cos(phase)      # true derivative

    else:
        phase    = w2 * (t - t_transition)
        position = -As * np.sin(phase)
        velocity = -As * w2 * np.cos(phase)     # true derivative

    return position, velocity

for i in range(model.njnt):
    name = model.joint(i).name
    adr  = model.jnt_qposadr[i]
    dof  = model.jnt_dofadr[i]
    print(f"Joint '{name}': qpos[{adr}], qvel[{dof}]")

hip_joint = model.joint("hip")
hip_qpos_adr = hip_joint.qposadr[0]
hip_qvel_adr = hip_joint.dofadr[0]


# ===========
# PLOTTING INFORMATION
# ===========
time_history = []
torqueActual_history = []
torqueCommand_history = []
positionActual_history = []
velocityActual_history = []
targetPosition_history = []
targetVelocity_history = []
quat_history = []

##FOOT CONTACT

def init_geom_to_body_tracking(model, geom_names: list):
    """Initializes tracking containers for specific geoms."""
    contact_geoms = {}
    con_dict = {}
    for name in geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            print(f"Warning: Geom '{name}' not found.")
            continue
        contact_geoms[name] = {'geom_id': geom_id}
        con_dict[name] = {'t_coords': []}
    return contact_geoms, con_dict

def record_contacts_in_body_frame(model, data, contact_geoms, con_dict):
    """Tracks specific geoms but logs points in the parent body's frame."""
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        
        for name, v in contact_geoms.items():
            target_id = v['geom_id']
            
            if g1 == target_id or g2 == target_id:
                pos_world = contact.pos
                
                # 1. Look up the parent body of this specific geom
                parent_body_id = model.geom_bodyid[target_id]
                
                # 2. Get the parent body's live world position and orientation matrix
                body_pos = data.body(parent_body_id).xpos
                body_mat = data.body(parent_body_id).xmat.reshape(3, 3)
                
                # 3. Transform world contact to the shared parent body frame
                p_body = body_mat.T @ (pos_world - body_pos)
                
                # 4. Log [time, x, y, z]
                timed_p_body = np.hstack([data.time, p_body])
                con_dict[name]['t_coords'].append(timed_p_body)

my_collision_geoms = ['right_foot_1_col', 'right_foot_2_col', 'right_foot_3_col']
contact_geoms, con_dict = init_geom_to_body_tracking(model, my_collision_geoms)

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════
# ========

mujoco.mj_setConst(model, data)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    #data.qpos[2] = 1.2

    while viewer.is_running():
        t = data.time

        # radians — same units as motorwave.py MIT_controller calls
        target_pos_rad, target_vel_rad = calculate_sine_reference(t)

        current_pos = data.qpos[hip_qpos_adr]
        current_vel = data.qvel[hip_qvel_adr]

        #Keep Track of Data for Plotting
        time_history.append(t)
        torqueActual_history.append(data.ctrl[0])
        positionActual_history.append(current_pos)
        velocityActual_history.append(current_vel)
        targetPosition_history.append(target_pos_rad)
        targetVelocity_history.append(target_vel_rad)
        quat_history.append(data.xquat[motor_id].copy())
        #torqueCommand_history.append(data.qfrc_actuator[0])
        if t > T_WAIT:
            record_contacts_in_body_frame(model, data, contact_geoms, con_dict)


        ramp      = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
        Kp_ramped = KP * ramp
        Kd_ramped = KD * ramp

        # PD in radians — identical to motorwave.py
        tau = (Kp_ramped * (target_pos_rad - current_pos) +
               Kd_ramped * (target_vel_rad - current_vel))

        # clamp to motor torque limit
        tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

        cmd_buffer.append(tau)
        data.ctrl[0] = cmd_buffer.pop(0)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

# ===========
# PLOTTING
# ===========
print("Viewer closed. Generating Matplotlib plots...")

import matplotlib
matplotlib.use('agg') # Force a non-interactive background backend
import matplotlib.pyplot as plt

time_history = np.array(time_history)
torqueActual_history = np.array(torqueActual_history)
torqueCommand_history = np.array(torqueCommand_history)
positionActual_history = np.array(positionActual_history)
velocityActual_history = np.array(velocityActual_history)
targetPosition_history = np.array(targetPosition_history)
targetVelocity_history = np.array(targetVelocity_history)
quat_history = np.array(quat_history)                 # (N,4) [w,x,y,z]
rpy_history = np.rad2deg(quat_to_rpy(quat_history))    # (N,3) degrees
roll_history, pitch_history, yaw_history = rpy_history[:, 0], rpy_history[:, 1], rpy_history[:, 2]

# Note: If you add a positionCommand_history later, convert it here too:
# positionCommand_history = np.array(positionCommand_history)

# Create a single plot instead of subplots
fig, ax1 = plt.subplots(figsize=(11, 7))

# --- Primary Y-Axis: Radians (Position & Velocity) ---
line1 = ax1.plot(time_history, positionActual_history, color='#1f77b4', linewidth=2, label='Actual Joint Position')
line2 = ax1.plot(time_history, velocityActual_history, color='#1f77b4', linewidth=2, label='Actual Joint Velocity', linestyle='--', alpha=0.7)
line3 = ax1.plot(time_history, targetPosition_history, color='#2ca02c', linewidth=2, label='Target Joint Position', linestyle='-.', alpha=0.9)
line4 = ax1.plot(time_history, targetVelocity_history, color='#2ca02c', linewidth=2, label='Target Joint Velocity', linestyle='--', alpha=0.7)

ax1.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Position / Velocity (rad, rad/s)', fontsize=11, fontweight='bold', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4') # Matches axis numbers to the data color
ax1.set_title('Joint Control Performance & Telemetry', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, linestyle=':', alpha=0.6)

# --- Secondary Y-Axis: Torque ---
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

# If you track commanded torque, uncomment the line below to overlay it:
# line4 = ax2.plot(time_history, torqueCommand_history, color='#d62728', linewidth=2, label='Commanded Torque (PD)')
line5 = ax2.plot(time_history, torqueActual_history, color='#ff7f0e', linewidth=1.5, alpha=0.9, label='Actual Torque')

ax2.set_ylabel('Torque (Nm)', fontsize=11, fontweight='bold', color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e') # Matches axis numbers to the data color
# Grid is turned off for ax2 to prevent conflicting gridlines with ax1
ax2.grid(False) 

# --- Combined Legend ---
# This grabs the labels from both axes so they can be displayed in a single legend box
lines = line1 + line2 + line5 + line3 + line4 # Add line3 or line4 here if you uncommented them
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

# 3. Clean layout adjustments and render
plt.tight_layout()
plt.savefig('joint_telemetry_plot.png', dpi=300)
print("Plot successfully saved as 'joint_telemetry_plot.png'")

# ── Roll / Pitch / Yaw plot (motor body), overwritten fresh each run ──────────
fig2, ax3 = plt.subplots(figsize=(11, 7))
#Roll and Pitch Swapped to match true orientation in viewer
ax3.plot(time_history, roll_history,  color='#d62728', linewidth=2, label='Pitch (motor body)')
ax3.plot(time_history, pitch_history, color='#9467bd', linewidth=2, label='Roll (motor body)')
ax3.plot(time_history, yaw_history,   color='#17becf', linewidth=2, label='Yaw (motor body)')
 
ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Angle (degrees)', fontsize=11, fontweight='bold')
ax3.set_title('Motor Body Orientation — Roll / Pitch / Yaw', fontsize=14, fontweight='bold', pad=15)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right')
 
plt.tight_layout()
plt.savefig('orientation_telemetry_plot.png', dpi=300)
plt.close(fig2)
print("Plot successfully saved as 'orientation_telemetry_plot.png'")

#Foot Contact Plotting Save
for name in con_dict.keys():
    if len(con_dict[name]['t_coords']) > 0:
        con_dict[name]['t_coords'] = np.array(con_dict[name]['t_coords'])
    else:
        # If a geom never touched the ground, give it a clean empty shape
        con_dict[name]['t_coords'] = np.empty((0, 4))

# Save the dictionary to disk so your PyVista script can read it
output_filename = "contact_dict.pkl"
with open(output_filename, "wb") as f:
    pkl.dump(con_dict, f)

print(f"Simulation finished! Contact points saved to {output_filename}")