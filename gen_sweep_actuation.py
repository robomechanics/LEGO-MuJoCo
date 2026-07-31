"""
Actuation sweep: grid over amplitude/frequency (10x10, +/-20% around the
hardware-validated defaults from copy_motorwave.py), and for each bin a
5x5 grid over the startup multipliers (start_amp_mult, start_freq_mult) to
find which startup profile lets that amplitude/frequency combo walk best.

Runs against the original (unmodified) robot model straight from
bigfoot/scene.xml, with a fixed foot offset (same values as
test_sim_usefeet.py's LEFT_OFFSET/RIGHT_OFFSET) applied to the baked-in
foot geoms -- no OpenSCAD foot-geometry injection, no offset sweep.
"""
import csv
import numpy as np
import mujoco

from sim_common import calculate_sine_reference, pd_torque

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ENTRY_XML          = "bigfoot/scene.xml"
JOINT_NAME         = "hip"
TORSO_BODY_NAME    = "motor"
TORQUE_LIMIT       = 25.0
T_WAIT             = 3.0
USE_RAMP           = False
RAMP_TIME          = 1.0
CMD_DELAY_STEPS    = 1
ITERATION_DURATION = 20.0
MIN_DISTANCE       = 2.0    # metres -- minimum to count as a walking result

# ── Fixed gains (hardware-validated defaults, from copy_motorwave.py) ────────
Kp = 35.5
Kd = 6.5

# ── Fixed foot offset (same values as test_sim_usefeet.py) ───────────────────
LEFT_OFFSET  = np.array([0.0, 0.0105, 0.07])
RIGHT_OFFSET = np.array([0.07, -0.0105, 0.0])

FOOT_GEOM_NAMES = [
    "right_foot_1",     "right_foot_2",     "right_foot_3",
    "left_foot_1",      "left_foot_2",      "left_foot_3",
    "right_foot_1_col", "right_foot_2_col", "right_foot_3_col",
    "left_foot_1_col",  "left_foot_2_col",  "left_foot_3_col",
]

# ── Amplitude / frequency grid: +/-20% around the hardware defaults ─────────
AMP_DEG_CENTER  = 37.5
FREQ_HZ_CENTER  = 0.55
GRID_BINS       = 20
AMP_DEG_VALUES  = np.linspace(AMP_DEG_CENTER * 0.8, AMP_DEG_CENTER * 1.2, GRID_BINS)
FREQ_HZ_VALUES  = np.linspace(FREQ_HZ_CENTER * 0.8, FREQ_HZ_CENTER * 1.2, GRID_BINS)

# ── Startup multiplier grid, per amplitude/frequency bin ─────────────────────
START_AMP_MULT_VALUES  = np.linspace(0.8, 1.8, 10)
START_FREQ_MULT_VALUES = np.linspace(0.7, 1.4, 10)

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

model = mujoco.MjModel.from_xml_path(ENTRY_XML)
data  = mujoco.MjData(model)

# ── Apply the fixed foot offset to the baked-in geoms/bodies ─────────────────
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

for name in FOOT_GEOM_NAMES:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id != -1:
        delta = RIGHT_OFFSET if "right" in name else LEFT_OFFSET
        model.geom_pos[geom_id] = original_geom_pos[name] + delta

for bid in foot_parent_body_ids:
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    if body_name == "motor":  # right-side root body
        model.body_ipos[bid] = original_body_ipos[bid] + RIGHT_OFFSET
    elif body_name == "simplified_motor___arm_rod":  # left-side sub-body
        model.body_ipos[bid] = original_body_ipos[bid] + LEFT_OFFSET

mujoco.mj_setConst(model, data)
mujoco.mj_forward(model, data)

TOTAL_MASS = sum(model.body_mass[i] for i in range(model.nbody))
GRAVITY    = 9.81

torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
joint_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)

if joint_id == -1 or torso_body_id == -1:
    raise ValueError(f"Could not find joint '{JOINT_NAME}' or body '{TORSO_BODY_NAME}'.")

qpos_idx = model.jnt_qposadr[joint_id]
qvel_idx = model.jnt_dofadr[joint_id]

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def check_has_fallen(body_id, height_threshold=0.5, angle_threshold_deg=45.0):
    if data.xpos[body_id][2] < height_threshold:
        return True
    torso_up_z     = data.xmat[body_id][8]
    tilt_angle_deg = np.rad2deg(np.arccos(np.clip(torso_up_z, -1.0, 1.0)))
    return tilt_angle_deg > angle_threshold_deg


def run_trial(amp_deg, freq_hz, start_amp_mult, start_freq_mult):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    start_x    = data.xpos[torso_body_id][0]
    start_y    = data.xpos[torso_body_id][1]
    hip_omega  = freq_hz * 2 * np.pi
    leg_amp_rad = np.deg2rad(amp_deg)
    cmd_buffer = [0.0] * CMD_DELAY_STEPS
    fell       = False
    max_steps  = int(ITERATION_DURATION / model.opt.timestep)
    energy_used = 0.0

    for step in range(max_steps):
        t = data.time

        target_pos_rad, target_vel_rad = calculate_sine_reference(
            t, hip_omega, leg_amp_rad, start_amp_mult, start_freq_mult, t_wait=T_WAIT
        )

        current_pos = data.qpos[qpos_idx]
        current_vel = data.qvel[qvel_idx]

        ramp = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
        tau  = pd_torque(target_pos_rad, target_vel_rad, current_pos, current_vel,
                         Kp, Kd, TORQUE_LIMIT, ramp=ramp)
        energy_used += abs(tau * current_vel) * model.opt.timestep

        cmd_buffer.append(tau)
        data.ctrl[0] = cmd_buffer.pop(0)

        mujoco.mj_step(model, data)

        if t > T_WAIT and check_has_fallen(torso_body_id):
            fell = True
            break

    final_x  = data.xpos[torso_body_id][0]
    final_y  = data.xpos[torso_body_id][1]
    distance = float(np.sqrt((final_x - start_x) ** 2 + (final_y - start_y) ** 2))
    cot = (energy_used / (TOTAL_MASS * GRAVITY * distance)) if distance > 1e-6 else float('inf')
    passed = not fell and distance >= MIN_DISTANCE

    return fell, distance, cot, passed


# ═══════════════════════════════════════════════════════════════════════════════
# GRID SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    all_results = []
    best_per_bin = []

    total_bins  = len(AMP_DEG_VALUES) * len(FREQ_HZ_VALUES)
    total_trials = total_bins * len(START_AMP_MULT_VALUES) * len(START_FREQ_MULT_VALUES)
    print(f"Starting actuation sweep: {total_bins} amp/freq bins x "
          f"{len(START_AMP_MULT_VALUES) * len(START_FREQ_MULT_VALUES)} startup combos "
          f"= {total_trials} trials.")

    bin_idx = 0
    for amp_deg in AMP_DEG_VALUES:
        for freq_hz in FREQ_HZ_VALUES:
            bin_idx += 1
            bin_results = []

            for start_amp_mult in START_AMP_MULT_VALUES:
                for start_freq_mult in START_FREQ_MULT_VALUES:
                    fell, distance, cot, passed = run_trial(
                        amp_deg, freq_hz, start_amp_mult, start_freq_mult
                    )

                    row = {
                        # Full precision, not rounded: this system is sensitive
                        # enough near these operating points that a few
                        # thousandths of a degree/Hz can flip walk vs. fall,
                        # so rounded values won't reproduce in the GUI viewer.
                        "Amplitude_Deg":   amp_deg,
                        "Frequency_Hz":    freq_hz,
                        "Start_Amp_Mult":  start_amp_mult,
                        "Start_Freq_Mult": start_freq_mult,
                        "Fell":            fell,
                        "Distance_Traversed": round(distance, 4),
                        "CoT":             round(cot, 4),
                        "Passed":          passed,
                    }
                    all_results.append(row)
                    bin_results.append(row)

            print(f"[bin {bin_idx:>3}/{total_bins}] amp={amp_deg:.2f} deg freq={freq_hz:.3f} Hz "
                  f"| passed={sum(r['Passed'] for r in bin_results)}/{len(bin_results)}")

            passing = [r for r in bin_results if r["Passed"]]
            if passing:
                best = min(passing, key=lambda r: r["CoT"])
            else:
                best = max(bin_results, key=lambda r: r["Distance_Traversed"])
            best_per_bin.append(best)

    print(f"\nSweep complete. {sum(r['Passed'] for r in all_results)}/{total_trials} "
          f"trials passed (no fall + >{MIN_DISTANCE}m).")

    # ═══════════════════════════════════════════════════════════════════════
    # EXPORT
    # ═══════════════════════════════════════════════════════════════════════

    with open("sweep_actuation_all.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print("Wrote all trials to 'sweep_actuation_all.csv'.")

    with open("sweep_actuation_best_per_bin.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_per_bin[0].keys())
        writer.writeheader()
        writer.writerows(best_per_bin)
    print("Wrote best startup combo per amp/freq bin to 'sweep_actuation_best_per_bin.csv'.")


if __name__ == "__main__":
    main()
