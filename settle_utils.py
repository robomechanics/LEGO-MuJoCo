"""Shared startup-settle helpers for Bigfoot simulations."""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np


def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
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


def average_quaternions(quats: np.ndarray) -> np.ndarray:
    """Component-wise mean of (N,4) wxyz quats, flipped to one hemisphere, then renormalized."""
    q = np.asarray(quats, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or len(q) == 0:
        raise ValueError(f"Expected non-empty (N,4) quats, got {getattr(q, 'shape', None)}")
    aligned = q.copy()
    dots = aligned @ aligned[0]
    aligned[dots < 0.0] *= -1.0
    mean = aligned.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        raise RuntimeError("Quaternion average collapsed to zero.")
    return mean / norm


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def lock_quat_yaw(settled_quat: np.ndarray, yaw_reference_quat: np.ndarray) -> np.ndarray:
    settled_rpy = np.ravel(quat_to_rpy(settled_quat))
    ref_rpy = np.ravel(quat_to_rpy(yaw_reference_quat))
    return rpy_to_quat(float(settled_rpy[0]), float(settled_rpy[1]), float(ref_rpy[2]))


def startup_settle_orientation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hip_qpos_adr: int,
    hip_qvel_adr: int,
    torque_limit: float,
    settle_s: float,
    avg_s: float,
    settle_kp: float,
    settle_kd: float,
    print_fn: Callable[..., None] | None = print,
) -> np.ndarray:
    """Find a supported rest state, then verify it with only the gait PD.

    Temporary viscous forces accelerate offline relaxation; they are removed
    for every release test and before returning. Never transplant an averaged
    orientation into a different height/hip configuration. This initializer
    assumes the robot's first joint is free and the ground is a horizontal plane.
    """
    if settle_s <= 0 or avg_s <= 0:
        raise ValueError("Settling and observation durations must be positive.")
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Startup settling requires a leading free joint.")
    spawn_quat = data.qpos[3:7].copy()
    original_forces = data.qfrc_applied.copy()
    dt = float(model.opt.timestep)
    window_s = max(5.0, avg_s)
    block_steps = int(np.ceil(window_s / dt))
    max_steps = int(np.ceil(max(600.0, settle_s) / dt))
    damping = np.array([30.0, 30.0, 30.0, 10.0, 10.0, 10.0])
    trial = mujoco.MjData(model)
    elapsed_steps = 0
    accepted = False
    span = np.full(3, np.inf)

    def command(state: mujoco.MjData) -> float:
        return float(np.clip(
            -settle_kp * state.qpos[hip_qpos_adr]
            - settle_kd * state.qvel[hip_qvel_adr],
            -torque_limit, torque_limit,
        ))

    if print_fn is not None:
        print_fn("Preparing supported stance; checking unassisted yaw/rocking before release...")
    try:
        while elapsed_steps < max_steps:
            for _ in range(min(block_steps, max_steps - elapsed_steps)):
                data.ctrl[0] = command(data)
                data.qfrc_applied[:6] = original_forces[:6] - damping * data.qvel[:6]
                mujoco.mj_step(model, data)
                elapsed_steps += 1
            if elapsed_steps * dt < settle_s:
                continue

            mujoco.mj_copyData(trial, model, data)
            trial.qfrc_applied[:] = original_forces
            # Restore heading once, before validation, as a rigid rotation of
            # the whole stance. On a horizontal plane this preserves support.
            yaw = float(quat_to_rpy(trial.qpos[3:7])[0, 2])
            ref_yaw = float(quat_to_rpy(spawn_quat)[0, 2])
            angle = ref_yaw - yaw
            c, s = np.cos(angle), np.sin(angle)
            trial.qpos[3:7] = lock_quat_yaw(trial.qpos[3:7], spawn_quat)
            trial.qvel[:2] = np.array([[c, -s], [s, c]]) @ trial.qvel[:2]
            trial.qacc_warmstart[:] = 0.0
            mujoco.mj_forward(model, trial)
            rpy_samples = [quat_to_rpy(trial.qpos[3:7])[0]]
            max_speed = float(np.max(np.abs(trial.qvel)))
            # Match the one-step command delay used by both simulation loops.
            delayed_command = float(trial.ctrl[0])
            for _ in range(block_steps):
                next_command = command(trial)
                trial.ctrl[0] = delayed_command
                delayed_command = next_command
                mujoco.mj_step(model, trial)
                rpy_samples.append(quat_to_rpy(trial.qpos[3:7])[0])
                max_speed = max(max_speed, float(np.max(np.abs(trial.qvel))))
            span = np.rad2deg(np.ptp(np.unwrap(rpy_samples, axis=0), axis=0))
            # Both feet must actually be supported, not merely motionless.
            supported = set()
            for contact in trial.contact:
                if contact.efc_address < 0:
                    continue
                names = [model.geom(int(g)).name for g in (contact.geom1, contact.geom2)]
                if "floor" in names:
                    supported.update(name for name in names if "foot" in name)
            both_feet = any("left" in n for n in supported) and any("right" in n for n in supported)
            if both_feet and span[2] < 0.02 and max(span[:2]) < 0.05 and max_speed < 0.005:
                mujoco.mj_copyData(data, model, trial)
                accepted = True
                break
    finally:
        data.qfrc_applied[:] = original_forces

    if not accepted:
        raise RuntimeError(
            f"No stable supported stance after {elapsed_steps * dt:.1f}s; "
            f"release RPY span (deg)={span}. Refusing to start from a rocking stance."
        )
    # Restart the gait clock only; preserve height, hip, velocities and holding
    # torque from the successfully validated state.
    data.time = 0.0
    mujoco.mj_forward(model, data)
    if print_fn is not None:
        print_fn(
            f"Stance ready after {elapsed_steps * dt:.1f}s simulated relaxation: "
            f"{window_s:.1f}s release RPY span (deg)={span.round(5)}, "
            f"hip={np.rad2deg(data.qpos[hip_qpos_adr]):.4f} deg, "
            f"holding torque={data.ctrl[0]:.5f} Nm."
        )
    return data.qpos[3:7].copy()


def print_settle_quaternion(
    times: np.ndarray,
    body_quats: np.ndarray,
    window_s: float,
    data: mujoco.MjData,
    print_fn: Callable[..., None] = print,
) -> None:
    """Average trailing body quaternions and print XML-ready settle pose."""
    if len(times) == 0 or len(body_quats) == 0:
        print_fn("Settle requested, but no quaternion samples were recorded.")
        return

    t_end = float(times[-1])
    t_start = max(float(times[0]), t_end - window_s)
    mask = times >= t_start
    samples = body_quats[mask]
    if len(samples) == 0:
        print_fn("Settle requested, but no samples fell inside the settle window.")
        return

    mean_body = average_quaternions(samples)
    rpy_body = np.rad2deg(quat_to_rpy(mean_body))
    free_q = np.asarray(data.qpos[3:7], dtype=float)
    free_q = free_q / max(np.linalg.norm(free_q), 1e-12)
    rpy_free = np.rad2deg(quat_to_rpy(free_q))

    print_fn("\n=== Settled average quaternion (-settle) ===")
    print_fn(f"Samples: {len(samples)}  (t = [{t_start:.2f}, {t_end:.2f}] s)")
    print_fn(
        "Body xquat (w x y z): "
        f"{mean_body[0]:.8f} {mean_body[1]:.8f} {mean_body[2]:.8f} {mean_body[3]:.8f}"
    )
    print_fn(
        "Freejoint qpos[3:7] (final): "
        f"{free_q[0]:.8f} {free_q[1]:.8f} {free_q[2]:.8f} {free_q[3]:.8f}"
    )
    print_fn(f"Body RPY (deg):      roll={rpy_body[0]:.3f}  pitch={rpy_body[1]:.3f}  yaw={rpy_body[2]:.3f}")
    print_fn(f"Freejoint RPY (deg): roll={rpy_free[0]:.3f}  pitch={rpy_free[1]:.3f}  yaw={rpy_free[2]:.3f}")
    print_fn("\nXML-ready body quat attribute:")
    print_fn(f'quat="{mean_body[0]:.6f} {mean_body[1]:.6f} {mean_body[2]:.6f} {mean_body[3]:.6f}"')
