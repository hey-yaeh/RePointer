"""Apex-preserving temporal warp and active-arm tapered hold (M1)."""
from __future__ import annotations
import numpy as np
from .contracts import ACTIVE_JOINTS, Motion, finite_array
from .rotations import axis_angle_to_quat, quat_to_axis_angle, nlerp


def piecewise_positions(src_len: int, src_peak: int, dst_len: int, dst_peak: int) -> np.ndarray:
    for length, peak in ((src_len, src_peak), (dst_len, dst_peak)):
        if type(length) is not int or length < 3 or type(peak) is not int or not 1 <= peak < length - 1:
            raise ValueError("warp lengths >=3 and interior integer apex frames required")
    pos = np.empty(dst_len, dtype=np.float64)
    pos[:dst_peak + 1] = np.linspace(0., src_peak, dst_peak + 1)
    pos[dst_peak:] = np.linspace(src_peak, src_len - 1, dst_len - dst_peak)
    return pos


def warp_clip(motion: Motion, src_peak: int, dst_len: int, dst_peak: int) -> Motion:
    pos = piecewise_positions(motion.frames, src_peak, dst_len, dst_peak)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, motion.frames - 1)
    q = axis_angle_to_quat(motion.poses.reshape(-1, 55, 3))
    poses = quat_to_axis_angle(nlerp(q[lo], q[hi], (pos-lo)[:, None, None])).reshape(dst_len, 165)
    trans = np.stack([np.interp(pos, np.arange(motion.frames), motion.trans[:, k]) for k in range(3)], axis=1)
    return Motion(poses, trans - trans[dst_peak])


def hold_weights(frames: int, peak: int, plateau: int = 5, taper: int = 8) -> np.ndarray:
    if (any(type(value) is not int for value in (frames, peak, plateau, taper))
            or frames < 1 or not 0 <= peak < frames or plateau < 0 or taper < 0):
        raise ValueError("invalid hold length, peak, plateau or taper")
    distance = np.abs(np.arange(frames) - peak)
    half = plateau // 2
    weights = np.where(distance <= half, 1., 0.)
    if taper:
        mask = (distance > half) & (distance <= half + taper)
        weights[mask] = .5 * (1 + np.cos(np.pi * (distance[mask] - half) / taper))
    return weights


def apply_hold(motion: Motion, peak: int, optimized_apex: np.ndarray, hand: int,
               plateau: int = 5, taper: int = 8) -> Motion:
    if type(hand) is not int or hand not in ACTIVE_JOINTS:
        raise ValueError("invalid active hand or apex pose")
    optimized_apex = finite_array(optimized_apex, (165,), "apex pose")
    out = motion.poses.copy().reshape(-1, 55, 3)
    weights = hold_weights(motion.frames, peak, plateau, taper)[:, None]
    target = np.asarray(optimized_apex).reshape(55, 3)
    for joint in ACTIVE_JOINTS[hand]:
        out[:, joint] = quat_to_axis_angle(nlerp(axis_angle_to_quat(out[:, joint]),
            axis_angle_to_quat(target[joint]), weights))
    return Motion(out, motion.trans)
