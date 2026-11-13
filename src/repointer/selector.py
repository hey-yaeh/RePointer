"""Detector-robust spatial diagnostics and unchanged whole-candidate selection.

Utility weights and all ten hard guards follow frozen P2C2. Input score tracks
must be supplied by the geometry/detector backend; no official scorer is implied.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, TypeVar
import numpy as np

CandidateT = TypeVar('CandidateT')
P2C2Error = ValueError

def fnum(value: Any, default: float = float("nan")) -> float:
    if is_blank(value):
        return default
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def clipped_window(center: int, radius: int, length: int) -> list[int]:
    center = int(np.clip(center, 0, max(0, length - 1)))
    return list(range(max(0, center - radius), min(length, center + radius + 1)))


def angular_dispersion_deg(vectors: np.ndarray) -> float:
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(vectors) <= 1:
        return 0.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-12)
    mean = np.sum(unit, axis=0)
    mean /= max(float(np.linalg.norm(mean)), 1e-12)
    dots = np.clip(unit @ mean, -1.0, 1.0)
    return float(np.max(np.degrees(np.arccos(dots))))


def score_summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise P2C2Error("Invalid robust score array")
    return {
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }


def choose_candidate(a: Mapping[str, Any], b: Mapping[str, Any], threshold: float = 0.001) -> dict[str, Any]:
    # Candidate A is the explicit safe fallback. Candidate B replaces it only when
    # the detector-robust spatial utility improves materially and hard guards pass.
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('selector threshold must be finite and nonnegative')
    required = (
        'detector_floor_score1b', 'window_q10_score1b', 'planned_floor_score1b',
        'world_bbox_window_q10_score1b', 'world_point_window_q10_score1b',
        'forearm_direction_dispersion_deg', 'detector_spread_frames',
        'generic_joint_jerk', 'generic_foot_sliding', 'joint_limit_fraction',
        'robust_spatial_utility',
    )
    for metrics in (a, b):
        if any(key not in metrics or not np.isfinite(fnum(metrics[key])) for key in required):
            raise ValueError('selector metrics must include all finite guard and utility values')
    guards = {
        "detector_floor": fnum(b["detector_floor_score1b"]) >= fnum(a["detector_floor_score1b"]) - 0.002,
        "window_q10": fnum(b["window_q10_score1b"]) >= fnum(a["window_q10_score1b"]) - 0.002,
        "planned_floor": fnum(b["planned_floor_score1b"]) >= fnum(a["planned_floor_score1b"]) - 0.003,
        "world_bbox_q10": fnum(b["world_bbox_window_q10_score1b"]) >= fnum(a["world_bbox_window_q10_score1b"]) - 0.003,
        "world_point_q10": fnum(b["world_point_window_q10_score1b"]) >= fnum(a["world_point_window_q10_score1b"]) - 0.004,
        "direction_dispersion": fnum(b["forearm_direction_dispersion_deg"]) <= max(
            fnum(a["forearm_direction_dispersion_deg"]) + 3.0, 6.0
        ),
        "detector_spread": fnum(b["detector_spread_frames"]) <= fnum(a["detector_spread_frames"]) + 5.0,
        "joint_jerk": fnum(b["generic_joint_jerk"]) <= max(
            fnum(a["generic_joint_jerk"]) * 1.35,
            fnum(a["generic_joint_jerk"]) + 5e-5,
        ),
        "foot_sliding": fnum(b["generic_foot_sliding"]) <= fnum(a["generic_foot_sliding"]) + 5e-4,
        "joint_limits": fnum(b["joint_limit_fraction"]) <= max(fnum(a["joint_limit_fraction"]), 0.005),
    }
    guard_pass = all(guards.values())
    utility_gain = fnum(b["robust_spatial_utility"]) - fnum(a["robust_spatial_utility"])
    detector_floor_gain = fnum(b["detector_floor_score1b"]) - fnum(a["detector_floor_score1b"])
    window_q10_gain = fnum(b["window_q10_score1b"]) - fnum(a["window_q10_score1b"])
    # A material replacement requires a direct gain in the frozen robust utility.
    # The component gains are reported for diagnosis but do not bypass this gate.
    material_gain = utility_gain >= threshold
    choose_b = bool(guard_pass and material_gain)
    failed = [name for name, passed in guards.items() if not passed]
    return {
        "choice": "B" if choose_b else "A",
        "guard_pass": guard_pass,
        "failed_guards": failed,
        "utility_gain_b_minus_a": utility_gain,
        "detector_floor_gain_b_minus_a": detector_floor_gain,
        "window_q10_gain_b_minus_a": window_q10_gain,
        "material_gain": material_gain,
    }


def robust_metrics(
    relative_scores: np.ndarray,
    world_bbox_scores: np.ndarray,
    world_point_scores: np.ndarray,
    planned_frame: int,
    detector_frames: list[int] | np.ndarray,
    detector_agreement: float,
    direction_dispersion_deg: float,
    quality: Mapping[str, float],
    planned_radius: int = 2,
    detector_radius: int = 2,
    point_inside_bbox: bool = False,
) -> dict[str, float | int]:
    """Summarize spatial scores over planned and detector windows as frozen P2C2.

    Score tracks have shape (T,) in [0,1]; detector frames are ordered source,
    refined, heuristic, ensemble. Direction dispersion is measured over the same
    union of windows. Quality uses the historical local metric scale.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in (relative_scores, world_bbox_scores, world_point_scores)]
    if arrays[0].ndim != 1 or not len(arrays[0]):
        raise ValueError('score tracks must be nonempty vectors')
    T = len(arrays[0])
    if any(v.shape != (T,) or not np.isfinite(v).all() or np.any((v < 0) | (v > 1)) for v in arrays):
        raise ValueError('score tracks must have matching shape and finite scores within [0,1]')
    detector = np.asarray(detector_frames)
    if detector.shape != (4,) or not np.issubdtype(detector.dtype, np.integer) or np.any((detector < 0) | (detector >= T)):
        raise ValueError('four integer detector frames within the clip are required')
    if not isinstance(planned_frame, (int, np.integer)) or not 0 <= planned_frame < T:
        raise ValueError('planned frame outside clip')
    if any(not isinstance(r, (int, np.integer)) or r < 0 for r in (planned_radius, detector_radius)):
        raise ValueError('window radii must be nonnegative integers')
    if not np.isfinite(detector_agreement) or not 0 <= detector_agreement <= 1:
        raise ValueError('detector agreement must be within [0,1]')
    if not np.isfinite(direction_dispersion_deg) or direction_dispersion_deg < 0:
        raise ValueError('direction dispersion must be nonnegative and finite')
    qkeys = ('generic_joint_jerk', 'generic_foot_sliding', 'joint_limit_fraction')
    if any(k not in quality or not np.isfinite(quality[k]) or quality[k] < 0 for k in qkeys):
        raise ValueError('finite nonnegative motion quality metrics are required')
    if quality['joint_limit_fraction'] > 1:
        raise ValueError('joint limit fraction must be within [0,1]')
    planned = clipped_window(planned_frame, planned_radius, T)
    window = set(planned)
    for frame in detector:
        window.update(clipped_window(int(frame), detector_radius, T))
    window = sorted(window)
    combined = np.minimum(arrays[0], arrays[1])
    if not point_inside_bbox:
        combined = np.minimum(combined, arrays[2])
    det_summary = score_summary(combined[detector])
    win_summary = score_summary(combined[window])
    plan_summary = score_summary(combined[planned])
    stability = float(math.exp(-direction_dispersion_deg ** 2 / (2 * 6 ** 2)))
    utility = (
        .34 * det_summary['min'] + .26 * win_summary['q10']
        + .18 * plan_summary['min'] + .10 * det_summary['mean']
        + .06 * win_summary['mean'] + .04 * detector_agreement + .02 * stability
    )
    return {
        'planned_frame': int(planned_frame),
        'det_source': int(detector[0]), 'det_refined': int(detector[1]),
        'det_heuristic': int(detector[2]), 'det_ensemble': int(detector[3]),
        'detector_agreement': float(detector_agreement),
        'detector_spread_frames': float(np.std(detector)),
        'ensemble_robust_score1b': float(combined[detector[3]]),
        'detector_floor_score1b': det_summary['min'],
        'detector_mean_score1b': det_summary['mean'],
        'planned_floor_score1b': plan_summary['min'],
        'planned_mean_score1b': plan_summary['mean'],
        'window_floor_score1b': win_summary['min'],
        'window_q10_score1b': win_summary['q10'],
        'window_mean_score1b': win_summary['mean'],
        'relative_window_q10_score1b': float(np.quantile(arrays[0][window], .1)),
        'world_bbox_window_q10_score1b': float(np.quantile(arrays[1][window], .1)),
        'world_point_window_q10_score1b': float(np.quantile(arrays[2][window], .1)),
        'forearm_direction_dispersion_deg': float(direction_dispersion_deg),
        'direction_stability': stability, 'robust_spatial_utility': float(utility),
        'robust_frame_count': len(window),
        **{k: float(quality[k]) for k in qkeys},
    }


def select_candidate(
    a: CandidateT, b: CandidateT,
    a_metrics: Mapping[str, Any], b_metrics: Mapping[str, Any],
    threshold: float = .001,
) -> tuple[CandidateT, dict[str, Any]]:
    """Return the original complete A or B object, with no blending or mutation."""
    decision = choose_candidate(a_metrics, b_metrics, threshold)
    return (b if decision['choice'] == 'B' else a), decision
