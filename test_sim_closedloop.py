import argparse
import subprocess
import tempfile
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import time
import pickle as pkl

from control_waveform import DEFAULT_WAVEFORM, startup_sine_reference
from settle_utils import (
    print_settle_quaternion as _print_settle_quaternion,
    quat_to_rpy,
    startup_settle_orientation as _startup_settle_orientation,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CLI — video recording
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bigfoot closed-loop sim (optional video record)")
    parser.add_argument("-r", "--record", action="store_true", help="Record an offscreen tracking video")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="closedloop_sim",
        help="Video title / filename stem (saved under data/videos/)",
    )
    parser.add_argument("-vdir", "--video_dir", type=str, default="data/videos", help="Directory for videos")
    parser.add_argument("-vfps", "--video_fps", type=int, default=30, help="Recorded video FPS")
    parser.add_argument("--width", type=int, default=1280, help="Recorded video width")
    parser.add_argument("--height", type=int, default=720, help="Recorded video height")
    parser.add_argument(
        "-settle",
        "--settle",
        action="store_true",
        help="On viewer close, print averaged torso quaternion from the last settle window",
    )
    parser.add_argument(
        "--settle-window",
        type=float,
        default=2.0,
        help="Seconds of trailing quaternion samples to average when using -settle",
    )
    return parser.parse_args()


ARGS = parse_args()
RECORD = bool(ARGS.record)
SETTLE = bool(ARGS.settle)
SETTLE_WINDOW_S = float(ARGS.settle_window)
VIDEO_NAME = ARGS.name
VIDEO_DIR = Path(ARGS.video_dir)
VIDEO_FPS = int(ARGS.video_fps)
VIDEO_WIDTH = int(ARGS.width)
VIDEO_HEIGHT = int(ARGS.height)

# ═══════════════════════════════════════════════════════════════════════════════
# USER PARAMETERS — match these directly to motorwave.py for hardware replication
# ═══════════════════════════════════════════════════════════════════════════════

# ── Motor / control ───────────────────────────────────────────────────────────
KP           = 29.1
KD           = 8.2
TORQUE_LIMIT = 25.0       # Nm — matches MIT_Params T_max and gear in XML

# ── Trajectory ────────────────────────────────────────────────────────────────
WAVEFORM = DEFAULT_WAVEFORM

# FOOT OFFSETS — Bigfoot single-piece STLs (right_foot_1 / left_foot_1)
# Same mapping as test_sim_sweep.apply_foot_offsets.
# CAD foot placement is already baked into robot.xml; leave these at 0 unless
# you intentionally shift stance.
#
# FOOT_X : positive = shift feet inward (toward robot midline)
# FOOT_Y : positive = shift feet forward
# FOOT_Z : positive = shift feet up (right parent frame +Z)
#
# Right foot lives on body "motor"; left on "simplified_motor___arm_rod".
#   right geom delta = [ FOOT_X,  FOOT_Y,  FOOT_Z ]
#   left  geom delta = [ -FOOT_Z, -FOOT_Y,  FOOT_X ]  # == sweep [0,-fy,fx] when Z=0
FOOT_X = 0.004
FOOT_Y = -0.023
FOOT_Z = 0.0
# ═══════════════════════════════════════════════════════════════════════════════

# ── Gain ramp ─────────────────────────────────────────────────────────────────
USE_RAMP  = False
RAMP_TIME = 2.0

# ── CAN latency simulation ────────────────────────────────────────────────────
CMD_DELAY_STEPS = 1

# Prepare and validate a supported rest state with the same PD as the gait.
STARTUP_SETTLE_S = 8.0
STARTUP_SETTLE_AVG_S = 2.0
# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

model = mujoco.MjModel.from_xml_path("Bigfoot/scene.xml")
model = mujoco.MjModel.from_xml_path("modified_model.xml")
data  = mujoco.MjData(model)

motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "motor")
arm_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "simplified_motor___arm_rod")

# Bigfoot CAD export: one visual + one collision geom per foot.
RIGHT_FOOT_GEOMS = ("right_foot_1", "right_foot_1_col")
LEFT_FOOT_GEOMS  = ("left_foot_1", "left_foot_1_col")

delta_right_geom = np.array([FOOT_X, FOOT_Y, FOOT_Z], dtype=float)
delta_left_geom  = np.array([-FOOT_Z, -FOOT_Y, FOOT_X], dtype=float)
# Parent-body CoM shifts (world-ish on motor; mirrored lateral on arm link).
delta_right_body = np.array([0,0,0], dtype=float)
delta_left_body  = np.array([0,0,0], dtype=float)

original_geom_pos = {
    name: model.geom_pos[gid].copy()
    for name in (*RIGHT_FOOT_GEOMS, *LEFT_FOOT_GEOMS)
    if (gid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) != -1
}
original_body_ipos = {
    bid: model.body_ipos[bid].copy()
    for bid in (motor_id, arm_id)
    if bid != -1
}

for name in RIGHT_FOOT_GEOMS:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id != -1:
        model.geom_pos[geom_id] = original_geom_pos[name] + delta_right_geom

for name in LEFT_FOOT_GEOMS:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id != -1:
        model.geom_pos[geom_id] = original_geom_pos[name] + delta_left_geom

if motor_id != -1:
    model.body_ipos[motor_id] = original_body_ipos[motor_id] + delta_right_body
if arm_id != -1:
    model.body_ipos[arm_id] = original_body_ipos[arm_id] + delta_left_body

mujoco.mj_setConst(model, data)
mujoco.mj_forward(model, data)

print(
    f"Closed-loop trial  amp={WAVEFORM.leg_amp_deg:.1f}° "
    f"sam={WAVEFORM.start_amp_mult:.2f} sfm={WAVEFORM.start_freq_mult:.2f} "
    f"Kp={KP:.1f} Kd={KD:.1f} freq={WAVEFORM.hip_freq_hz:.3f} Hz"
)
print(
    f"Foot offsets: FOOT_X={FOOT_X:.4f} FOOT_Y={FOOT_Y:.4f} FOOT_Z={FOOT_Z:.4f} | "
    f"right_geom={delta_right_geom} left_geom={delta_left_geom}"
)
print(f"Whole-robot CoM: {data.subtree_com[motor_id].round(4)}")
print(f"Total mass:      {sum(model.body_mass[i] for i in range(model.nbody)):.3f} kg")
# ── Derived constants ─────────────────────────────────────────────────────────
cmd_buffer  = [0.0] * CMD_DELAY_STEPS


def startup_settle_orientation() -> np.ndarray:
    """Prepare a supported stance and verify that it stays still before walking."""
    return _startup_settle_orientation(
        model=model,
        data=data,
        hip_qpos_adr=hip_qpos_adr,
        hip_qvel_adr=hip_qvel_adr,
        torque_limit=TORQUE_LIMIT,
        settle_s=STARTUP_SETTLE_S,
        avg_s=STARTUP_SETTLE_AVG_S,
        settle_kp=KP,
        settle_kd=KD,
        print_fn=print,
    )


def print_settle_quaternion(times: np.ndarray, body_quats: np.ndarray, window_s: float) -> None:
    """Average trailing body quaternions and print XML-ready settle pose."""
    _print_settle_quaternion(times, body_quats, window_s, data=data, print_fn=print)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY — direct port of motorwave.py calculate_sine_reference
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_sine_reference(t):
    return startup_sine_reference(
        t=t,
        hip_omega=WAVEFORM.hip_omega,
        leg_amp_rad=WAVEFORM.leg_amp_rad,
        t_wait=WAVEFORM.t_wait,
        start_amp_mult=WAVEFORM.start_amp_mult,
        start_freq_mult=WAVEFORM.start_freq_mult,
        ramp_time=WAVEFORM.startup_ramp_time,
    )

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

my_collision_geoms = ["right_foot_1_col", "left_foot_1_col"]
contact_geoms, con_dict = init_geom_to_body_tracking(model, my_collision_geoms)

def build_tracking_camera(
    distance: float = 2.4,
    azimuth: float = 145.0,
    elevation: float = -18.0,
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = motor_id
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def save_video_h264(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Write RGB frames to an H.264 mp4 (Cursor/browser-friendly)."""
    if not frames:
        print("No frames captured; skipping video write.")
        return

    output_path = output_path if output_path.suffix else output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]

    with tempfile.TemporaryDirectory(prefix="closedloop_vid_") as tmp:
        tmp_dir = Path(tmp)
        for i, frame in enumerate(frames):
            # PPM is trivial to write without extra image deps.
            path = tmp_dir / f"frame_{i:06d}.ppm"
            rgb = np.asarray(frame, dtype=np.uint8)
            header = f"P6\n{width} {height}\n255\n".encode("ascii")
            path.write_bytes(header + rgb.tobytes())

        pattern = str(tmp_dir / "frame_%06d.ppm")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)

    print(f"Saved video ({len(frames)} frames @ {fps} fps) to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════
# ========

mujoco.mj_setConst(model, data)
mujoco.mj_forward(model, data)

video_frames: list[np.ndarray] = []
next_frame_time = 0.0
frame_interval = 1.0 / VIDEO_FPS if RECORD else None
renderer = None
record_camera = None

if RECORD:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), VIDEO_WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), VIDEO_HEIGHT)
    renderer = mujoco.Renderer(model, VIDEO_HEIGHT, VIDEO_WIDTH)
    record_camera = build_tracking_camera()
    video_out = VIDEO_DIR / VIDEO_NAME
    if video_out.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
        video_out = video_out.with_suffix(".mp4")
    print(f"Recording enabled → {video_out} ({VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS} fps)")

if SETTLE:
    print(f"Settle mode on: will average last {SETTLE_WINDOW_S:.1f}s of body quaternion when viewer closes")

if STARTUP_SETTLE_S > 0:
    startup_settle_orientation()
    cmd_buffer[:] = [float(data.ctrl[0])] * CMD_DELAY_STEPS

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
        if t > WAVEFORM.t_wait:
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

        if RECORD and renderer is not None and record_camera is not None and frame_interval is not None:
            while data.time >= next_frame_time:
                renderer.update_scene(data, camera=record_camera)
                video_frames.append(renderer.render().copy())
                next_frame_time += frame_interval

        viewer.sync()
        time.sleep(model.opt.timestep)

if renderer is not None:
    renderer.close()

if RECORD:
    video_out = VIDEO_DIR / VIDEO_NAME
    if video_out.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
        video_out = video_out.with_suffix(".mp4")
    try:
        save_video_h264(video_frames, video_out, VIDEO_FPS)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Failed to encode video with ffmpeg: {exc}")

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

if SETTLE:
    print_settle_quaternion(time_history, quat_history, SETTLE_WINDOW_S)

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
