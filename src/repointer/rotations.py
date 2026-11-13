"""Shortest-hemisphere quaternion nlerp used by frozen M1, without SciPy."""
from __future__ import annotations
import numpy as np


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    scale = np.divide(np.sin(angle / 2), angle, out=np.full_like(angle, .5), where=angle > 1e-8)
    return np.concatenate([np.cos(angle / 2), aa * scale], axis=-1)


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    scale = np.divide(2 * np.arctan2(s, q[..., :1]), s,
                      out=np.full_like(s, 2.), where=s > 1e-8)
    return q[..., 1:] * scale


def nlerp(q0: np.ndarray, q1: np.ndarray, weight: np.ndarray) -> np.ndarray:
    q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -q1, q1)
    q = (1 - weight) * q0 + weight * q1
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def rotation_matrix(aa: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues rotation, stable at zero."""
    q = axis_angle_to_quat(aa)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(q.shape[:-1] + (3, 3))
