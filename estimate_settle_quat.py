#!/usr/bin/env python3
"""Hold zero leg amplitude, settle under gravity, report average torso quaternion.

Useful for estimating the rest lean / spawn quat for Bigfoot without gait motion.
When changing foot/hip offsets, re-run this and bake the quat into robot.xml (or
pass --foot-x/--foot-y so the settle matches the offset you will use).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# USER PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_XML = os.environ.get("MODEL_XML_PATH", "Bigfoot/scene.xml")
BODY_NAME = "motor"
JOINT_NAME = "hip"

KP = 45.0
KD = 7.0
TORQUE_LIMIT = 25.0

SETTLE_S = 100.0
AVG_WINDOW_S = 4.0  # only average the last N seconds (true rest pose)
LEG_AMP_DEG = 0.0

# Same mapping as test_sim_sweep.apply_foot_offsets / closedloop FOOT_X/Y
FOOT_X = 0.024
FOOT_Y = -0.025

RIGHT_FOOT_GEOMS = ("right_foot_1", "right_foot_1_col", "right_foot_2", "right_foot_2_col", "right_foot_3", "right_foot_3_col")
LEFT_FOOT_GEOMS = ("left_foot_1", "left_foot_1_col", "left_foot_2", "left_foot_2_col", "left_foot_3", "left_foot_3_col")

# ═══════════════════════════════════════════════════════════════════════════════


def apply_foot_offsets(model: mujoco.MjModel, foot_x: float, foot_y: float) -> None:
    """Shift named foot geoms + parent body CoMs (sweep / closedloop convention)."""
    delta_right_geom = np.array([foot_x, foot_y, 0.0], dtype=float)
    delta_left_geom = np.array([0.0, -foot_y, foot_x], dtype=float)
    delta_right_body = np.array([foot_x, foot_y, 0.0], dtype=float)
    delta_left_body = np.array([-foot_x, foot_y, 0.0], dtype=float)

    motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "motor")
    arm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "simplified_motor___arm_rod")
    if motor_id != -1:
        model.body_ipos[motor_id] = model.body_ipos[motor_id] + delta_right_body
    if arm_id != -1:
        model.body_ipos[arm_id] = model.body_ipos[arm_id] + delta_left_body

    for name in RIGHT_FOOT_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            model.geom_pos[gid] = model.geom_pos[gid] + delta_right_geom
    for name in LEFT_FOOT_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            model.geom_pos[gid] = model.geom_pos[gid] + delta_left_geom


def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(np.asarray(quat_wxyz, dtype=float))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = np.column_stack([roll, pitch, yaw])
    return out[0] if np.ndim(quat_wxyz) == 1 else out


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
    settled_rpy = quat_to_rpy(settled_quat)
    ref_rpy = quat_to_rpy(yaw_reference_quat)
    return rpy_to_quat(float(settled_rpy[0]), float(settled_rpy[1]), float(ref_rpy[2]))


def average_quaternions(quats: np.ndarray) -> np.ndarray:
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


def estimate_settle_quats(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    foot_x: float = 0.0,
    foot_y: float = 0.0,
    settle_s: float = SETTLE_S,
    avg_window_s: float = AVG_WINDOW_S,
    kp: float = KP,
    kd: float = KD,
    body_name: str = BODY_NAME,
    joint_name: str = JOINT_NAME,
    viewer: bool = False,
) -> dict[str, np.ndarray]:
    """Zero-amp settle with given foot offsets; return mean body + freejoint quats."""
    if avg_window_s <= 0 or avg_window_s > settle_s:
        raise ValueError("avg_window_s must be > 0 and <= settle_s")

    apply_foot_offsets(model, foot_x, foot_y)
    mujoco.mj_setConst(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found")
    if joint_id < 0:
        raise ValueError(f"Joint '{joint_name}' not found")

    qpos_idx = int(model.jnt_qposadr[joint_id])
    qvel_idx = int(model.jnt_dofadr[joint_id])

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    spawn_xyz = data.qpos[0:3].copy()
    spawn_quat = data.qpos[3:7].copy()
    dt = float(model.opt.timestep)
    n_steps = int(np.ceil(settle_s / dt))
    avg_start_t = settle_s - avg_window_s

    body_quats: list[np.ndarray] = []
    free_quats: list[np.ndarray] = []

    def step_once() -> None:
        pos = data.qpos[qpos_idx]
        vel = data.qvel[qvel_idx]
        tau = kp * (0.0 - pos) + kd * (0.0 - vel)
        data.ctrl[0] = float(np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT))
        mujoco.mj_step(model, data)
        # Only keep the trailing rest window — averaging t=0..T mixes in the fall-in lean.
        if data.time >= avg_start_t:
            body_quats.append(data.xquat[body_id].copy())
            free_quats.append(data.qpos[3:7].copy())

    if viewer:
        with mujoco.viewer.launch_passive(model, data) as v:
            for _ in range(n_steps):
                if not v.is_running():
                    break
                step_once()
                v.sync()
    else:
        for _ in range(n_steps):
            step_once()

    if not body_quats:
        raise RuntimeError("No quaternion samples collected — settle/window too short?")

    # Lock yaw to spawn heading so repeated settles don't spin the freejoint.
    mean_body = lock_quat_yaw(average_quaternions(np.asarray(body_quats)), spawn_quat)
    mean_free = lock_quat_yaw(average_quaternions(np.asarray(free_quats)), spawn_quat)

    return {
        "body": mean_body,
        "freejoint": mean_free,
        "spawn_xyz": spawn_xyz,
    }


def patch_robot_body_quat(robot_xml: Path, quat_wxyz: np.ndarray) -> None:
    """Rewrite the motor body quat= attribute in robot.xml."""
    text = robot_xml.read_text()
    q = " ".join(f"{float(v):.8f}" for v in quat_wxyz)
    pattern = re.compile(
        r'(<body\s+name="motor"[^>]*\bquat=")([^"]+)(")',
        re.MULTILINE,
    )
    new_text, n = pattern.subn(rf"\g<1>{q}\g<3>", text, count=1)
    if n != 1:
        raise RuntimeError(f"Could not patch motor quat in {robot_xml}")
    robot_xml.write_text(new_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", default=MODEL_XML)
    parser.add_argument("--settle-s", type=float, default=SETTLE_S)
    parser.add_argument("--avg-window-s", type=float, default=AVG_WINDOW_S)
    parser.add_argument("--kp", type=float, default=KP)
    parser.add_argument("--kd", type=float, default=KD)
    parser.add_argument("--foot-x", type=float, default=FOOT_X)
    parser.add_argument("--foot-y", type=float, default=FOOT_Y)
    parser.add_argument(
        "--write-robot-xml",
        type=str,
        default="",
        help="If set, patch this robot.xml motor body quat (e.g. Bigfoot/robot.xml)",
    )
    parser.add_argument("--viewer", action="store_true", help="Open passive viewer while settling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.model_xml)
    data = mujoco.MjData(model)

    print(f"Model: {args.model_xml}")
    print(f"Foot offsets: foot_x={args.foot_x:.4f}, foot_y={args.foot_y:.4f}")
    print(f"Zero-amp settle: {args.settle_s:.1f}s, average last {args.avg_window_s:.1f}s")
    print(f"Holding hip at 0 with Kp={args.kp:.1f}, Kd={args.kd:.1f}")

    result = estimate_settle_quats(
        model,
        data,
        foot_x=args.foot_x,
        foot_y=args.foot_y,
        settle_s=args.settle_s,
        avg_window_s=args.avg_window_s,
        kp=args.kp,
        kd=args.kd,
        viewer=args.viewer,
    )
    # Freejoint quat is what MuJoCo stores in the body quat= attribute for a freejoint root.
    mean_body = result["body"]
    mean_free = result["freejoint"]
    rpy_body = np.rad2deg(quat_to_rpy(mean_body))
    rpy_free = np.rad2deg(quat_to_rpy(mean_free))

    print("\n=== Settled average quaternion ===")
    print(
        "Body xquat (w x y z): "
        f"{mean_body[0]:.8f} {mean_body[1]:.8f} {mean_body[2]:.8f} {mean_body[3]:.8f}"
    )
    print(
        "Freejoint qpos[3:7]:  "
        f"{mean_free[0]:.8f} {mean_free[1]:.8f} {mean_free[2]:.8f} {mean_free[3]:.8f}"
    )
    print(f"Body RPY (deg):      roll={rpy_body[0]:.3f}  pitch={rpy_body[1]:.3f}  yaw={rpy_body[2]:.3f}")
    print(f"Freejoint RPY (deg): roll={rpy_free[0]:.3f}  pitch={rpy_free[1]:.3f}  yaw={rpy_free[2]:.3f}")
    print("\nXML-ready body quat attribute (use freejoint):")
    print(f'quat="{mean_free[0]:.6f} {mean_free[1]:.6f} {mean_free[2]:.6f} {mean_free[3]:.6f}"')

    if args.write_robot_xml:
        path = Path(args.write_robot_xml)
        patch_robot_body_quat(path, mean_free)
        print(f"\nPatched motor quat in {path}")


if __name__ == "__main__":
    main()
