"""
Foot-offset settling sweep: for a 9x9 grid of (foot_x, foot_y) placement
offsets, plus an explicit (0,0) baseline, runs two phases per trial with
the hip held rigid at 0 deg (no gait):

  Phase 1 ("find the starting quaternion"): place the feet touching the
  ground (not a guessed spawn height) and step until it settles (or a cap
  is hit) -- this is where each offset naturally lands, which may not be a
  clean equilibrium (some offsets rock/oscillate indefinitely). Also used
  to compute the settle time/frequency of the pitch transient.

  Phase 2 (measurement): from wherever phase 1 ended, run a fresh 20s
  holding the hip at 0, recording the torso quaternion throughout, and
  average it (Markley's method) -- this is the steady-state quaternion
  characterization, not contaminated by the initial drop transient.

Runs against the plain, unmodified bigfoot/scene.xml (no fixed foot offset
stacked in, unlike gen_sweep_actuation.py) so the swept (foot_x, foot_y)
is the only source of asymmetry.
"""
import csv
import os
import numpy as np
import mujoco

from sim_common import pd_torque, quat_to_rpy, apply_foot_offsets, place_on_ground

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ENTRY_XML       = "bigfoot/scene.xml"
JOINT_NAME      = "hip"
TORSO_BODY_NAME = "motor"
TORQUE_LIMIT    = 25.0
PHASE1_MAX_S    = 10.0    # cap on the settle/"find starting quaternion" phase
PHASE2_DURATION_S = 20.0  # fresh measurement window, averaged for the reported quaternion

# ── Fixed gains (hardware-validated defaults) ─────────────────────────────────
Kp = 35.5
Kd = 6.5

# ── Foot offset grid (same bounds as test_sim_sweep.py's RANGES) ─────────────
FOOT_X_RANGE = (-0.07, 0.025)
FOOT_Y_RANGE = (-0.02, 0.01)
GRID_BINS    = 9
FOOT_X_VALUES = np.linspace(*FOOT_X_RANGE, GRID_BINS)
FOOT_Y_VALUES = np.linspace(*FOOT_Y_RANGE, GRID_BINS)

# ── Settle-time tolerance band ────────────────────────────────────────────────
SETTLE_TOLERANCE_DEG = 1.0
STEADY_STATE_WINDOW_S = 2.0   # tail window used to estimate the final pitch
FREQ_WINDOW_S = 5.0           # leading window used for the FFT (excludes the settled tail)

FOOT_GEOM_NAMES = [
    "right_foot_1",     "right_foot_2",     "right_foot_3",
    "left_foot_1",      "left_foot_2",      "left_foot_3",
    "right_foot_1_col", "right_foot_2_col", "right_foot_3_col",
    "left_foot_1_col",  "left_foot_2_col",  "left_foot_3_col",
]

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

model = mujoco.MjModel.from_xml_path(ENTRY_XML)
data  = mujoco.MjData(model)

torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
joint_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)

if joint_id == -1 or torso_body_id == -1:
    raise ValueError(f"Could not find joint '{JOINT_NAME}' or body '{TORSO_BODY_NAME}'.")

qpos_idx = model.jnt_qposadr[joint_id]
qvel_idx = model.jnt_dofadr[joint_id]

free_joint_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "motor_freejoint")
free_z_qpos_idx = model.jnt_qposadr[free_joint_id] + 2

FOOT_COLLISION_GEOMS = [
    "right_foot_1_col", "right_foot_2_col", "right_foot_3_col",
    "left_foot_1_col",  "left_foot_2_col",  "left_foot_3_col",
]

# ── Snapshot original geom/body positions before any offset is applied ───────
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

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def check_has_fallen(body_id, height_threshold=0.5, angle_threshold_deg=45.0):
    if data.xpos[body_id][2] < height_threshold:
        return True
    torso_up_z     = data.xmat[body_id][8]
    tilt_angle_deg = np.rad2deg(np.arccos(np.clip(torso_up_z, -1.0, 1.0)))
    return tilt_angle_deg > angle_threshold_deg


def average_quaternion(quats):
    """Markley's method: eigenvector of the largest eigenvalue of sum(q q^T).
    Handles the q/-q sign ambiguity that breaks a naive component-wise mean."""
    M = np.zeros((4, 4))
    for q in quats:
        M += np.outer(q, q)
    M /= len(quats)
    eigvals, eigvecs = np.linalg.eigh(M)
    avg_q = eigvecs[:, np.argmax(eigvals)]
    if avg_q[0] < 0:
        avg_q = -avg_q
    return avg_q


def settle_time_and_freq(time_arr, pitch_deg):
    """Settle time: earliest time after which pitch stays within
    SETTLE_TOLERANCE_DEG of its steady-state value through the end of the
    run. Settle frequency: peak-magnitude frequency (0.05-20Hz band) of the
    detrended pitch signal's FFT."""
    n_tail = max(1, int(STEADY_STATE_WINDOW_S / (time_arr[1] - time_arr[0])))
    pitch_final = np.mean(pitch_deg[-n_tail:])

    within_band = np.abs(pitch_deg - pitch_final) <= SETTLE_TOLERANCE_DEG
    settle_time = float("nan")
    for i in range(len(within_band) - 1, -1, -1):
        if not within_band[i]:
            if i + 1 < len(time_arr):
                settle_time = float(time_arr[i + 1])
            break
    else:
        settle_time = float(time_arr[0])

    # FFT only over the transient window (not the full run): once settled,
    # the long flat tail dilutes the spectrum toward near-zero frequencies
    # and swamps the actual oscillation frequency we want.
    dt = time_arr[1] - time_arr[0]
    transient_window_s = min(FREQ_WINDOW_S, time_arr[-1] - time_arr[0])
    n_transient = max(8, int(transient_window_s / dt))
    detrended = pitch_deg[:n_transient] - pitch_final
    fft_vals = np.fft.rfft(detrended)
    fft_freqs = np.fft.rfftfreq(len(detrended), d=dt)
    band = (fft_freqs >= 0.05) & (fft_freqs <= 20.0)
    if np.any(band):
        peak_idx = np.argmax(np.abs(fft_vals[band]))
        settle_freq = float(fft_freqs[band][peak_idx])
    else:
        settle_freq = float("nan")

    return settle_time, settle_freq


def _step_holding_hip(duration_s, record=True):
    """Steps physics for duration_s (or until a fall), holding the hip at
    0 target. Returns (times, quats, pitches_deg, fell)."""
    max_steps = int(duration_s / model.opt.timestep)
    times, quats, pitches_deg = [], [], []
    fell = False

    for step in range(max_steps):
        t = data.time
        current_pos = data.qpos[qpos_idx]
        current_vel = data.qvel[qvel_idx]

        tau = pd_torque(0.0, 0.0, current_pos, current_vel, Kp, Kd, TORQUE_LIMIT)
        data.ctrl[0] = tau

        mujoco.mj_step(model, data)

        if record:
            q = data.xquat[torso_body_id].copy()
            rpy = quat_to_rpy(q)[0]
            times.append(t)
            quats.append(q)
            pitches_deg.append(np.rad2deg(rpy[1]))

        if t > 0.5 and check_has_fallen(torso_body_id):
            fell = True
            break

    return np.array(times), quats, np.array(pitches_deg), fell


def run_trial(foot_x, foot_y):
    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, foot_x, foot_y, FOOT_GEOM_NAMES, foot_parent_body_ids,
                        original_geom_pos, original_body_ipos)
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)

    # Phase 1: find the starting quaternion (settle, or hit the cap trying).
    p1_times, _, p1_pitches_deg, fell = _step_holding_hip(PHASE1_MAX_S)

    if fell:
        return {
            "Fell": True,
            "Avg_Quat_W": float("nan"), "Avg_Quat_X": float("nan"),
            "Avg_Quat_Y": float("nan"), "Avg_Quat_Z": float("nan"),
            "Pitch_Diff_Deg": float("nan"), "Tilt_Geodesic_Deg": float("nan"),
            "Settle_Freq_Hz": float("nan"), "Settle_Time_S": float("nan"),
        }

    settle_time, settle_freq = settle_time_and_freq(p1_times, p1_pitches_deg)

    # Phase 2: fresh 20s measurement from wherever phase 1 ended, averaged.
    _, p2_quats, _, fell = _step_holding_hip(PHASE2_DURATION_S)

    if fell:
        return {
            "Fell": True,
            "Avg_Quat_W": float("nan"), "Avg_Quat_X": float("nan"),
            "Avg_Quat_Y": float("nan"), "Avg_Quat_Z": float("nan"),
            "Pitch_Diff_Deg": float("nan"), "Tilt_Geodesic_Deg": float("nan"),
            "Settle_Freq_Hz": float("nan"), "Settle_Time_S": float("nan"),
        }

    avg_q = average_quaternion(p2_quats)
    avg_pitch_deg = float(np.rad2deg(quat_to_rpy(avg_q)[0][1]))

    return {
        "Fell": False,
        "Avg_Quat_W": float(avg_q[0]), "Avg_Quat_X": float(avg_q[1]),
        "Avg_Quat_Y": float(avg_q[2]), "Avg_Quat_Z": float(avg_q[3]),
        "_avg_pitch_deg": avg_pitch_deg,
        "Settle_Freq_Hz": settle_freq, "Settle_Time_S": settle_time,
    }


def geodesic_deg(q1, q2):
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return float(np.rad2deg(2.0 * np.arccos(dot)))


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Running (0, 0) baseline trial...")
    baseline = run_trial(0.0, 0.0)
    baseline_q = np.array([baseline["Avg_Quat_W"], baseline["Avg_Quat_X"],
                            baseline["Avg_Quat_Y"], baseline["Avg_Quat_Z"]])
    baseline_pitch_deg = baseline["_avg_pitch_deg"]
    print(f"Baseline: pitch={baseline_pitch_deg:.3f} deg, "
          f"settle_time={baseline['Settle_Time_S']:.2f}s, "
          f"settle_freq={baseline['Settle_Freq_Hz']:.3f}Hz")

    results = []
    baseline_row = {
        "Foot_X": 0.0, "Foot_Y": 0.0,
        "Fell": baseline["Fell"],
        "Avg_Quat_W": baseline["Avg_Quat_W"], "Avg_Quat_X": baseline["Avg_Quat_X"],
        "Avg_Quat_Y": baseline["Avg_Quat_Y"], "Avg_Quat_Z": baseline["Avg_Quat_Z"],
        "Pitch_Diff_Deg": 0.0, "Tilt_Geodesic_Deg": 0.0,
        "Settle_Freq_Hz": baseline["Settle_Freq_Hz"], "Settle_Time_S": baseline["Settle_Time_S"],
    }
    results.append(baseline_row)

    total = len(FOOT_X_VALUES) * len(FOOT_Y_VALUES)
    idx = 0
    for foot_x in FOOT_X_VALUES:
        for foot_y in FOOT_Y_VALUES:
            idx += 1
            trial = run_trial(foot_x, foot_y)

            if trial["Fell"]:
                row = {
                    "Foot_X": foot_x, "Foot_Y": foot_y,
                    "Fell": True,
                    "Avg_Quat_W": float("nan"), "Avg_Quat_X": float("nan"),
                    "Avg_Quat_Y": float("nan"), "Avg_Quat_Z": float("nan"),
                    "Pitch_Diff_Deg": float("nan"), "Tilt_Geodesic_Deg": float("nan"),
                    "Settle_Freq_Hz": float("nan"), "Settle_Time_S": float("nan"),
                }
                print(f"[{idx:>3}/{total}] foot_x={foot_x:+.4f} foot_y={foot_y:+.4f} | FELL")
            else:
                q = np.array([trial["Avg_Quat_W"], trial["Avg_Quat_X"],
                              trial["Avg_Quat_Y"], trial["Avg_Quat_Z"]])
                tilt_geo = geodesic_deg(q, baseline_q)
                pitch_diff = trial["_avg_pitch_deg"] - baseline_pitch_deg
                row = {
                    "Foot_X": foot_x, "Foot_Y": foot_y,
                    "Fell": False,
                    "Avg_Quat_W": trial["Avg_Quat_W"], "Avg_Quat_X": trial["Avg_Quat_X"],
                    "Avg_Quat_Y": trial["Avg_Quat_Y"], "Avg_Quat_Z": trial["Avg_Quat_Z"],
                    "Pitch_Diff_Deg": pitch_diff, "Tilt_Geodesic_Deg": tilt_geo,
                    "Settle_Freq_Hz": trial["Settle_Freq_Hz"], "Settle_Time_S": trial["Settle_Time_S"],
                }
                print(f"[{idx:>3}/{total}] foot_x={foot_x:+.4f} foot_y={foot_y:+.4f} | "
                      f"pitch_diff={pitch_diff:+.2f}deg tilt_geo={tilt_geo:.2f}deg "
                      f"settle_t={trial['Settle_Time_S']:.2f}s freq={trial['Settle_Freq_Hz']:.2f}Hz")

            results.append(row)

    os.makedirs("results", exist_ok=True)
    with open("results/sweep_foot_offset_settling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to 'results/sweep_foot_offset_settling.csv'.")


if __name__ == "__main__":
    main()
