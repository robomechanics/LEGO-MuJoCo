"""Shared coordinate-frame conventions.

Public repo convention:
    +x = forward walking direction
    +y = robot-left
    +z = downward into the ground

MuJoCo model convention remains z-up internally:
    +x = forward walking direction
    +y = robot-left
    +z = upward from the ground
"""

from __future__ import annotations

import numpy as np

PUBLIC_FRAME_DESCRIPTION = "+x forward, +y robot-left, +z down"
MUJOCO_FRAME_DESCRIPTION = "+x forward, +y robot-left, +z up"


def public_to_mujoco_vec(vec_xyz: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    """Convert a public-frame vector/offset to MuJoCo's internal z-up frame."""
    vec = np.asarray(vec_xyz, dtype=float)
    return np.array([vec[0], vec[1], -vec[2]], dtype=float)


def mujoco_to_public_vec(vec_xyz: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    """Convert a MuJoCo z-up vector/offset to the repo's public z-down frame."""
    vec = np.asarray(vec_xyz, dtype=float)
    return np.array([vec[0], vec[1], -vec[2]], dtype=float)


def public_xy_to_mujoco_xy(x_forward: float, y_left: float) -> np.ndarray:
    """Horizontal x/y axes are identical between public and MuJoCo frames."""
    return np.array([x_forward, y_left], dtype=float)


def mujoco_heading_axes(xmat_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal forward and left unit vectors from a MuJoCo body xmat."""
    rot = np.asarray(xmat_flat, dtype=float).reshape(3, 3)
    forward_xy = rot[:, 0][:2]
    norm = np.linalg.norm(forward_xy)
    if norm < 1e-9:
        forward_xy = np.array([1.0, 0.0], dtype=float)
    else:
        forward_xy = forward_xy / norm
    left_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=float)
    return forward_xy, left_xy
