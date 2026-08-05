"""
Runs one trial end to end, in two simulations:

  1. Headless, no GUI: measure this configuration's average settled
     quaternion (hip held rigid over a fixed QUAT_MEASURE_S window, own
     Kp/Kd/foot offset) via measure_avg_quaternion_pose -- the same
     pose-init method now standard across the sweep scripts (see
     test_sim_compute_settle_vs_quatpose.py for why: ~2x faster than a
     dynamic settle wait, at least as good on distance/velocity).

  2. The actual gait, teleported straight to that quaternion (t_wait=0),
     camera tracking the motor body. Either shown live in the interactive
     MuJoCo viewer (default), or -- with -record -- rendered offscreen for
     a fixed ITERATION_DURATION and saved as an mp4.

Usage:
    python run_trial.py                       # live interactive viewer
    python run_trial.py -record -n "title"     # saves results/videos/title.mp4
"""
import argparse
import os
import sys

# Must happen before `import mujoco`: offscreen recording needs the EGL
# backend selected before mujoco loads its GL bindings, while the
# interactive viewer needs the default (GLFW, needs a display) backend --
# so this only kicks in when -record is actually present, and only if the
# user hasn't already set MUJOCO_GL themselves.
if "-record" in sys.argv and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import time

from sim_common import (calculate_sine_reference, pd_torque, apply_foot_offsets,
                         place_on_ground, measure_avg_quaternion_pose)

# ═══════════════════════════════════════════════════════════════════════════════
# USER PARAMETERS — pulled from results/sweep_actuation_all.csv's best row.
# NOTE: dropping the body_ipos offset (geom-only now, see
# sim_common.apply_foot_offsets) changes the physics, so these values need to
# be re-pulled from a fresh actuation sweep run under the new offset scheme.
# ═══════════════════════════════════════════════════════════════════════════════

ENTRY_XML       = "bigfoot/scene.xml"
JOINT_NAME      = "hip"
TORSO_BODY_NAME = "motor"

# ── Motor / control ───────────────────────────────────────────────────────────
KP           = 35.5
KD           = 6.5
TORQUE_LIMIT = 25.0

# ── Trajectory -- best-surviving row from the 100-trial geom-only sweep
# (results/sweep_actuation_100_geomonly.csv): dist=0.474m, Fell=False ────────
HIP_OMEGA       = 0.55 * 2 * np.pi
LEG_AMP_DEG     = 35
START_FREQ_MULT = 1.1425
START_AMP_MULT  = 1.14

# ── Foot offset (same convention as the sweep scripts: apply_foot_offsets) ────
FOOT_X = -0.07
FOOT_Y = -0.0105

# ── Pose-init (see module docstring) ──────────────────────────────────────────
QUAT_MEASURE_S = 30

# ── Gain ramp ─────────────────────────────────────────────────────────────────
USE_RAMP  = False
RAMP_TIME = 2.0

# ── CAN latency simulation ────────────────────────────────────────────────────
CMD_DELAY_STEPS = 1

# ── Recording ──────────────────────────────────────────────────────────────────
PRE_ROLL_S          = 2.0   # robot holds its teleported pose for this long before the gait starts --
                             # gives the viewer/recording a settled beat to look at before it moves
ITERATION_DURATION  = 20.0  # length of the gait itself, on top of PRE_ROLL_S (live viewer runs until closed)
VIDEO_FPS    = 30
VIDEO_WIDTH  = 640   # capped by bigfoot/scene.xml's default offscreen framebuffer size --
VIDEO_HEIGHT = 480   # bump both here AND the model's <visual><global offwidth=".../> to go higher
CAM_DISTANCE  = 1.4         # centered, tracking camera on the motor body
CAM_AZIMUTH   = 90.0
CAM_ELEVATION = -15.0

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


def check_has_fallen(data, body_id, height_threshold=0.5, angle_threshold_deg=45.0):
    if data.xpos[body_id][2] < height_threshold:
        return True
    torso_up_z     = data.xmat[body_id][8]
    tilt_angle_deg = np.rad2deg(np.arccos(np.clip(torso_up_z, -1.0, 1.0)))
    return tilt_angle_deg > angle_threshold_deg


def make_tracking_camera(torso_body_id):
    cam = mujoco.MjvCamera()
    cam.type       = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso_body_id
    cam.distance   = CAM_DISTANCE
    cam.azimuth    = CAM_AZIMUTH
    cam.elevation  = CAM_ELEVATION
    return cam


def main():
    parser = argparse.ArgumentParser(description="Run one gait trial: measure the settled pose, then view or record it.")
    parser.add_argument("-record", action="store_true", help="record an offscreen video instead of the live viewer")
    parser.add_argument("-n", "--name", type=str, default="trial", help="video title; saved to results/videos/<name>.mp4")
    args = parser.parse_args()

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

    # ═════════════════════════════════════════════════════════════════════════
    # SIMULATION 1 -- headless: measure the average settled quaternion
    # ═════════════════════════════════════════════════════════════════════════
    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, FOOT_X, FOOT_Y, FOOT_GEOM_NAMES,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos)
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)
    avg_quat = measure_avg_quaternion_pose(model, data, qpos_idx, qvel_idx, torso_body_id,
                                            KP, KD, TORQUE_LIMIT, measure_s=QUAT_MEASURE_S)
    print(f"[1/2] Measured average settled quaternion: {avg_quat.round(4)}")

    # ═════════════════════════════════════════════════════════════════════════
    # SIMULATION 2 -- teleport to that pose, run the actual gait
    # ═════════════════════════════════════════════════════════════════════════
    mujoco.mj_resetData(model, data)
    apply_foot_offsets(model, data, FOOT_X, FOOT_Y, FOOT_GEOM_NAMES,
                        foot_parent_body_ids, original_geom_pos, original_body_ipos)
    data.qpos[free_quat_idx:free_quat_idx + 4] = avg_quat
    place_on_ground(model, data, FOOT_COLLISION_GEOMS, free_z_qpos_idx)

    leg_amp_rad = np.deg2rad(LEG_AMP_DEG)
    cmd_buffer  = [0.0] * CMD_DELAY_STEPS
    cam = make_tracking_camera(torso_body_id)

    def step_physics():
        t = data.time
        target_pos_rad, target_vel_rad = calculate_sine_reference(
            t, HIP_OMEGA, leg_amp_rad, START_AMP_MULT, START_FREQ_MULT, t_wait=PRE_ROLL_S
        )
        current_pos = data.qpos[qpos_idx]
        current_vel = data.qvel[qvel_idx]
        ramp = min(1.0, t / RAMP_TIME) if USE_RAMP and RAMP_TIME > 0 else 1.0
        tau = pd_torque(target_pos_rad, target_vel_rad, current_pos, current_vel,
                         KP, KD, TORQUE_LIMIT, ramp=ramp)
        cmd_buffer.append(tau)
        data.ctrl[0] = cmd_buffer.pop(0)
        mujoco.mj_step(model, data)

    if args.record:
        total_s = PRE_ROLL_S + ITERATION_DURATION
        print(f"[2/2] Recording {PRE_ROLL_S:.0f}s pre-roll + {ITERATION_DURATION:.0f}s gait "
              f"to results/videos/{args.name}.mp4 ...")
        os.makedirs("results/videos", exist_ok=True)
        video_path = f"results/videos/{args.name}.mp4"

        renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
        max_steps = int(total_s / model.opt.timestep)
        steps_per_frame = max(1, int(round(1.0 / (VIDEO_FPS * model.opt.timestep))))

        with imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264", quality=8) as writer:
            for step in range(max_steps):
                step_physics()
                if step % steps_per_frame == 0:
                    renderer.update_scene(data, camera=cam)
                    writer.append_data(renderer.render())
                if check_has_fallen(data, torso_body_id):
                    print(f"   Fell at t={data.time:.2f}s -- stopping recording early.")
                    break
        renderer.close()
        print(f"[2/2] Wrote {video_path}")
    else:
        print("[2/2] Opening interactive viewer (close the window to end the trial)...")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type        = cam.type
            viewer.cam.trackbodyid = cam.trackbodyid
            viewer.cam.distance    = cam.distance
            viewer.cam.azimuth     = cam.azimuth
            viewer.cam.elevation   = cam.elevation

            while viewer.is_running():
                step_physics()
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
