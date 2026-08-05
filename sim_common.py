"""
Shared trajectory, PD control law, foot-offset, and ground-placement/settle
logic used by the sweep and viewer scripts (gen_sweep_actuation.py,
test_sim_closedloop.py, test_sim_sweep.py). Keeping this in one place means
a gain/trajectory/placement change only has to happen once, instead of each
caller silently drifting apart.
"""
from collections import deque

import mujoco
import numpy as np


def calculate_sine_reference(t, hip_omega, leg_amp_rad, start_amp_mult, start_freq_mult, t_wait=3.0):
    """Direct port of motorwave.py's calculate_sine_reference."""
    w1           = hip_omega * start_freq_mult
    w2           = hip_omega
    t0           = t_wait
    At           = start_amp_mult * leg_amp_rad
    As           = leg_amp_rad
    t_transition = t0 + np.pi / w1

    if t <= t0:
        position, velocity = 0.0, 0.0
    elif t < t_transition:
        phase    = w1 * (t - t0)
        position = At * np.sin(phase)
        velocity = At * w1 * np.cos(phase)
    else:
        phase    = w2 * (t - t_transition)
        position = -As * np.sin(phase)
        velocity = -As * w2 * np.cos(phase)

    return position, velocity


def pd_torque(target_pos, target_vel, current_pos, current_vel, Kp, Kd, torque_limit, ramp=1.0):
    """PD law in radians, clamped to the motor torque limit."""
    tau = Kp * ramp * (target_pos - current_pos) + Kd * ramp * (target_vel - current_vel)
    return float(np.clip(tau, -torque_limit, torque_limit))


def quat_to_rpy(quat_wxyz):
    """Vectorized (N,4) quat[w,x,y,z] -> (N,3) [roll, pitch, yaw] in radians."""
    q = np.atleast_2d(quat_wxyz)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.column_stack([roll, pitch, yaw])


def apply_foot_offsets(model, data, foot_x, foot_y, foot_geom_names, foot_parent_body_ids,
                        original_geom_pos, original_body_ipos):
    """
    Shifts the foot GEOMS only (visual/collision mesh position) by
    (foot_x, foot_y), using the mesh-local rotated mapping:
        right: [x,  y, 0]
        left:  [0, -y, x]

    Deliberately does NOT touch body_ipos (center of mass): confirmed against
    the original rduplo.py/MJCFHandler framework (utils/xml_handler.py) that
    moving a foot's geometry there never shifts the parent body's mass --
    only geom_pos_offset is used for feet, body_pos_offset is a separate,
    unused-for-feet mechanism, and it moves kinematic body placement, not
    mass, anyway. An earlier version of this function also shifted
    model.body_ipos to approximate "the leg's mass moves with the foot", but
    that had no basis in the original design and used a formula that didn't
    even agree with gen_sweep_actuation.py's own (different) body-shift
    formula -- see the run_trial.py debugging session that uncovered this.
    foot_parent_body_ids/original_body_ipos are accepted but unused, kept for
    call-site compatibility.
    """
    delta_right_geom = np.array([foot_x,  foot_y, 0.0])
    delta_left_geom  = np.array([0.0,    -foot_y, foot_x])

    for name in foot_geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id != -1:
            if "right" in name.lower():
                model.geom_pos[geom_id] = original_geom_pos[name] + delta_right_geom
            else:
                model.geom_pos[geom_id] = original_geom_pos[name] + delta_left_geom

    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)


def min_foot_z(model, data, foot_geom_names):
    """Lowest world-Z vertex across the given (mesh) foot geoms, at the
    current pose. Used instead of guessing a spawn height in the XML --
    stays correct across foot scale/curvature/offset changes since it's
    computed from whatever mesh is actually loaded."""
    zmin = np.inf
    for name in foot_geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid == -1:
            continue
        dataid = model.geom_dataid[gid]
        if dataid == -1:
            continue  # not a mesh geom
        vadr = model.mesh_vertadr[dataid]
        vnum = model.mesh_vertnum[dataid]
        verts_local = model.mesh_vert[vadr:vadr + vnum]
        xpos = data.geom_xpos[gid]
        xmat = data.geom_xmat[gid].reshape(3, 3)
        verts_world = verts_local @ xmat.T + xpos
        zmin = min(zmin, verts_world[:, 2].min())
    return zmin


def place_on_ground(model, data, foot_geom_names, z_qpos_idx, floor_z=0.0, clearance=0.001):
    """Shifts the free-joint's z so the lowest foot-mesh vertex just touches
    the floor (+clearance), instead of guessing a fixed spawn height. Call
    after mj_resetData and after any foot-offset/geometry change, before
    stepping."""
    mujoco.mj_forward(model, data)
    zmin = min_foot_z(model, data, foot_geom_names)
    data.qpos[z_qpos_idx] += (floor_z + clearance - zmin)
    mujoco.mj_forward(model, data)


def wait_until_settled(model, data, hip_qpos_idx, hip_qvel_idx, free_qvel_adr,
                        Kp, Kd, torque_limit, max_wait_s=2.0,
                        lin_vel_thresh=0.05, ang_vel_thresh=0.05, dwell_s=0.1):
    """Steps physics with the hip held at 0 target, until the torso's
    linear+angular velocity both stay below threshold continuously for
    dwell_s (settled) or max_wait_s elapses -- a verified condition instead
    of a fixed guessed wait time. A single below-threshold sample isn't
    enough on its own: velocity is exactly 0 at t=0, and a body that's about
    to start tipping can still have a momentarily small velocity right
    after the first step. Returns the elapsed settle time (== data.time at
    exit, assuming data.time was 0 at the start of this call)."""
    max_steps = int(max_wait_s / model.opt.timestep)
    dwell_steps = max(1, int(dwell_s / model.opt.timestep))
    below_thresh_streak = 0

    for _ in range(max_steps):
        current_pos = data.qpos[hip_qpos_idx]
        current_vel = data.qvel[hip_qvel_idx]
        data.ctrl[0] = pd_torque(0.0, 0.0, current_pos, current_vel, Kp, Kd, torque_limit)
        mujoco.mj_step(model, data)

        lin_vel = np.linalg.norm(data.qvel[free_qvel_adr:free_qvel_adr + 3])
        ang_vel = np.linalg.norm(data.qvel[free_qvel_adr + 3:free_qvel_adr + 6])
        if lin_vel < lin_vel_thresh and ang_vel < ang_vel_thresh:
            below_thresh_streak += 1
            if below_thresh_streak >= dwell_steps:
                break
        else:
            below_thresh_streak = 0

    return data.time


def wait_until_pitch_settled(model, data, hip_qpos_idx, hip_qvel_idx, torso_body_id,
                              Kp, Kd, torque_limit, max_wait_s=2.0,
                              window_s=0.3, pp_thresh_deg=0.5, dwell_s=None):
    """Steps physics with the hip held at 0 target, until the torso pitch's
    peak-to-peak amplitude over a trailing window_s window stays below
    pp_thresh_deg continuously for dwell_s (settled) or max_wait_s elapses.

    A single below-threshold window isn't enough on its own: a lightly-damped
    oscillation can produce a low peak-to-peak reading purely from where the
    trailing window happens to land relative to the swing (e.g. a window that
    spans a small piece near one peak and a small piece near the next trough
    can read a small p2p even though the swing between the actual extrema is
    still large) -- that's a coincidence of phase alignment, not convergence,
    and the window slides past it within a cycle or two. Requiring the
    below-threshold condition to hold for a further dwell_s (default:
    window_s, so ~2 window-widths of sustained flatness) rejects that false
    trigger the same way wait_until_settled's dwell_s guards against a lone
    below-threshold velocity sample. Returns the elapsed settle time
    (== data.time at exit, assuming data.time was 0 at the start of this
    call).

    Uses quat_to_rpy(...)[0][0] (the component quat_to_rpy calls "roll"),
    not [0][1] -- for this robot's body frame, index 0 is the physically
    meaningful forward/back tip (what test_sim_closedloop.py's orientation
    plot calls "Pitch (motor body)"), while index 1 stays under ~1 deg
    regardless of how much the robot is actually tipping."""
    if dwell_s is None:
        dwell_s = window_s
    max_steps = int(max_wait_s / model.opt.timestep)
    window_steps = max(1, int(window_s / model.opt.timestep))
    dwell_steps = max(1, int(dwell_s / model.opt.timestep))
    pitch_window = deque(maxlen=window_steps)
    below_thresh_streak = 0

    for _ in range(max_steps):
        current_pos = data.qpos[hip_qpos_idx]
        current_vel = data.qvel[hip_qvel_idx]
        data.ctrl[0] = pd_torque(0.0, 0.0, current_pos, current_vel, Kp, Kd, torque_limit)
        mujoco.mj_step(model, data)

        pitch_deg = float(np.rad2deg(quat_to_rpy(data.xquat[torso_body_id])[0][0]))
        pitch_window.append(pitch_deg)

        if len(pitch_window) == window_steps and (max(pitch_window) - min(pitch_window)) < pp_thresh_deg:
            below_thresh_streak += 1
            if below_thresh_streak >= dwell_steps:
                break
        else:
            below_thresh_streak = 0

    return data.time


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


def measure_avg_quaternion_pose(model, data, hip_qpos_idx, hip_qvel_idx, torso_body_id,
                                 Kp, Kd, torque_limit, measure_s=5.0):
    """Steps physics with the hip held at 0 target for a FIXED measure_s
    window (no convergence check) and returns the average torso quaternion
    over that window (Markley's method via average_quaternion).

    Faster substitute for wait_until_pitch_settled(): averaging over a
    window spanning at least one oscillation period cancels out residual
    rocking even before the robot has technically settled by a
    peak-to-peak criterion, so it reaches a usable "settled" pose estimate
    in a fixed ~5s instead of a variable, sometimes 25s, dynamic wait.
    Assumes data is already reset/positioned (apply_foot_offsets +
    place_on_ground) before this is called; the caller is responsible for
    resetting and re-placing before applying the returned quaternion and
    starting the actual gait (this function's own stepping consumes sim
    time/state that the real trial shouldn't start from)."""
    quats = []
    max_steps = int(measure_s / model.opt.timestep)
    for _ in range(max_steps):
        current_pos = data.qpos[hip_qpos_idx]
        current_vel = data.qvel[hip_qvel_idx]
        data.ctrl[0] = pd_torque(0.0, 0.0, current_pos, current_vel, Kp, Kd, torque_limit)
        mujoco.mj_step(model, data)
        quats.append(data.xquat[torso_body_id].copy())

    return average_quaternion(quats)
