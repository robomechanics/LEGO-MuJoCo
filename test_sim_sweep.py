import csv
import mujoco
import numpy as np
from scipy.stats import qmc

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
T_WAIT             = 3.0
USE_RAMP           = False
RAMP_TIME          = 1.0
CMD_DELAY_STEPS    = 1
ITERATION_DURATION = 20.0
MIN_DISTANCE       = 2.0    # metres — minimum to save a result
NUM_TRIALS         = 20000  # ← change this to run more or fewer trials

# ── Randomisation ranges [min, max] ───────────────────────────────────────────
RANGES = {
    'foot_x':          (-0.07,  0.025), #Free Var
    'foot_y':          (-0.02,  0.01), #Free
    'Kp':              ( 35.0,  45.0 ), #Static (ish)
    'Kd':              ( 7.5,  9.0 ), #Static (ish)
    'start_amp_mult':  ( 1.0,   1.5), #static
    'start_freq_mult': ( 0.7,   1.4), #S
    'amp_deg':         (33.0,  38.0), #S
    'freq_hz':         ( 0.5,   0.65), #S
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

model = mujoco.MjModel.from_xml_path("bigfoot/scene.xml")
data  = mujoco.MjData(model)
TOTAL_MASS = sum(model.body_mass[i] for i in range(model.nbody))
GRAVITY    = 9.81

torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
joint_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)

if joint_id == -1 or torso_body_id == -1:
    raise ValueError(f"Could not find joint '{JOINT_NAME}' or body '{TORSO_BODY_NAME}'.")

qpos_idx = model.jnt_qposadr[joint_id]
qvel_idx = model.jnt_dofadr[joint_id]

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

def apply_foot_offsets(model, data, foot_x, foot_y):
    """
    Applies foot offsets by correcting geoms and body centers of mass
    (matches the head-on):

      - GEOM offsets use the mesh-local rotated mapping:
            right: [x,  y, 0]
            left:  [0, -y, x]

      - BODY (mass/inertia) offsets use the Global mirrored mapping:
            right: [ x, y, 0]
            left:  [-x, y, 0]
        Shifts gloablly.
    """
    # Geom-space deltas (mesh-local, rotated for the left leg)
    delta_right_geom = np.array([foot_x,  foot_y, 0.0])
    delta_left_geom  = np.array([0.0,    -foot_y, foot_x])

    # Body-space deltas (world-aligned, true mirror — matches head-on script)
    delta_right_body = np.array([foot_x, foot_y, 0.0])
    delta_left_body  = np.array([-foot_x, foot_y, 0.0])

    # --- STEP 1: Update Body Centers of Mass (ipos) ---
    for bid in foot_parent_body_ids:
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)

        if "motor" == body_name:  # Right side root body
            model.body_ipos[bid] = original_body_ipos[bid] + delta_right_body

        elif "simplified_motor___arm_rod" == body_name:  # Left side sub-body
            model.body_ipos[bid] = original_body_ipos[bid] + delta_left_body

    # --- STEP 2: Update Individual Geoms ---
    for name in FOOT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id != -1:
            if "right" in name.lower():
                model.geom_pos[geom_id] = original_geom_pos[name] + delta_right_geom
            else:
                model.geom_pos[geom_id] = original_geom_pos[name] + delta_left_geom

    # --- STEP 3: Bake changes into MuJoCo's solver engine ---
    mujoco.mj_setConst(model, data)  # Recalculates mass distribution matrices
    mujoco.mj_forward(model, data)   # Re-evaluates global geometry transformations


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


def calculate_sine_reference(t, hip_omega, leg_amp_rad, start_amp_mult, start_freq_mult):
    w1           = hip_omega * start_freq_mult
    w2           = hip_omega
    t0           = T_WAIT
    At           = start_amp_mult * leg_amp_rad
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


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

results = []
all_params = generate_lhs_samples(NUM_TRIALS)
print(f"Starting random sweep: {NUM_TRIALS} trials. Saving results with no fall and >{MIN_DISTANCE}m distance.")

for trial_idx, params in enumerate(all_params, 1):

    # full sim reset
    mujoco.mj_resetData(model, data)

    # apply foot offsets for this trial
    apply_foot_offsets(model, data, params['foot_x'], params['foot_y'])
    mujoco.mj_forward(model, data)

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
            t, hip_omega, leg_amp_rad, params['start_amp_mult'], params['start_freq_mult']
        )

        current_pos = data.qpos[qpos_idx]
        current_vel = data.qvel[qvel_idx]

        ramp      = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
        tau       = (params['Kp'] * ramp * (target_pos_rad - current_pos) +
                     params['Kd'] * ramp * (target_vel_rad - current_vel))
        tau       = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        energy_used += abs(tau * current_vel) * model.opt.timestep


        cmd_buffer.append(tau)
        data.ctrl[0] = cmd_buffer.pop(0)

        mujoco.mj_step(model, data)

        if t > T_WAIT and check_has_fallen(torso_body_id):
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
    csv_file = "sweep_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSweep complete. {len(results)}/{NUM_TRIALS} trials saved to '{csv_file}'.")
else:
    print(f"\nSweep complete. No trials met the criteria (no fall + >{MIN_DISTANCE}m).")
    print("Consider reducing MIN_DISTANCE or increasing NUM_TRIALS.")