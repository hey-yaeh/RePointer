"""Conservative timing routes and group-disjoint candidate ranking.

Scientific feature transforms and selection guards follow the frozen P2A/P2A-MH
source. Model fitting validates splits explicitly; no checkpoint is bundled.
The caller must generate all training features without held-out labels or donors.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

SMOOTH_KERNEL = np.asarray([1., 2., 3., 2., 1.]) / 9.
RouterError = ValueError
MHError = ValueError

def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def inum(x: Any, default: int = -1) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def build_router_features(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[str], list[str]]:
    """Build an explicitly allow-listed, inference-safe feature matrix."""
    names = [
        "domain_natural", "domain_synthetic", "hand_left", "hand_right",
        "clip_len_norm", "log_target_distance", "sin_target_azimuth",
        "cos_target_azimuth", "sin_target_elevation", "cos_target_elevation",
        "median_peak_ratio", "wordphase_peak_ratio", "peak_diff_norm",
        "peak_abs_diff_norm",
    ]
    base_candidate_names = [
        "detector_agreement", "planned_score1b", "self_score1b", "self_angle_norm",
        "self_plan_offset", "self_plan_abs", "det_spread", "det_mean_offset",
        "joint_jerk_log", "foot_sliding_log", "pose_deviation_log", "max_reach",
        "donor_rank_norm",
    ]
    for method in ("median", "wordphase"):
        names.extend([f"{method}_{x}" for x in base_candidate_names])
    names.extend([f"diff_wordphase_minus_median_{x}" for x in base_candidate_names])

    source_columns = [
        "domain", "hand", "clip_length", "target_distance", "target_azimuth_deg",
        "target_elevation_deg", "median_peak", "wordphase_peak",
    ]
    for method in ("median", "wordphase"):
        source_columns.extend([
            f"{method}_planned_frame", f"{method}_self_frame",
            f"{method}_det_source", f"{method}_det_refined", f"{method}_det_heuristic",
            f"{method}_detector_agreement", f"{method}_planned_score1b_proxy20",
            f"{method}_self_score1b_proxy20", f"{method}_self_angle_deg",
            f"{method}_joint_jerk", f"{method}_foot_sliding",
            f"{method}_pose_deviation", f"{method}_max_reach", f"{method}_donor_rank",
        ])

    prohibited_tokens = (
        "gt_peak", "hold_start", "hold_end", "hold_width", "alignment",
        "oracle", "distance_to_gt_hold", "timing_temporal", "temporal_proxy",
        "released_peak",
    )
    offenders = sorted({c for c in source_columns if any(t in c for t in prohibited_tokens)})
    if offenders:
        raise RouterError(f"Prohibited inference feature columns were selected: {offenders}")

    data: list[list[float]] = []
    for r in rows:
        T = max(1.0, fnum(r.get("clip_length"), 120.0) - 1.0)
        domain = str(r.get("domain", ""))
        hand = inum(r.get("hand"), -1)
        td = max(fnum(r.get("target_distance"), 1.0), 1e-4)
        az = math.radians(fnum(r.get("target_azimuth_deg"), 0.0))
        el = math.radians(fnum(r.get("target_elevation_deg"), 0.0))
        med_peak = fnum(r.get("median_peak"), 0.5 * T)
        wp_peak = fnum(r.get("wordphase_peak"), 0.5 * T)
        vec: list[float] = [
            float(domain == "natural"), float(domain == "synthetic"),
            float(hand == 0), float(hand == 1),
            (T + 1.0) / 180.0, math.log(td), math.sin(az), math.cos(az),
            math.sin(el), math.cos(el), med_peak / T, wp_peak / T,
            (wp_peak - med_peak) / T, abs(wp_peak - med_peak) / T,
        ]
        by_method: dict[str, list[float]] = {}
        for method in ("median", "wordphase"):
            planned = fnum(r.get(f"{method}_planned_frame"), 0.0)
            self_frame = fnum(r.get(f"{method}_self_frame"), planned)
            det = np.asarray([
                fnum(r.get(f"{method}_det_source"), planned),
                fnum(r.get(f"{method}_det_refined"), planned),
                fnum(r.get(f"{method}_det_heuristic"), planned),
            ], dtype=np.float64)
            vals = [
                fnum(r.get(f"{method}_detector_agreement"), 0.0),
                fnum(r.get(f"{method}_planned_score1b_proxy20"), 0.0),
                fnum(r.get(f"{method}_self_score1b_proxy20"), 0.0),
                fnum(r.get(f"{method}_self_angle_deg"), 180.0) / 180.0,
                (self_frame - planned) / T,
                abs(self_frame - planned) / T,
                float(np.max(det) - np.min(det)) / T,
                (float(np.mean(det)) - planned) / T,
                math.log1p(max(0.0, fnum(r.get(f"{method}_joint_jerk"), 0.0)) * 1e5),
                math.log1p(max(0.0, fnum(r.get(f"{method}_foot_sliding"), 0.0)) * 1e4),
                math.log1p(max(0.0, fnum(r.get(f"{method}_pose_deviation"), 0.0)) * 1e4),
                fnum(r.get(f"{method}_max_reach"), 0.0),
                fnum(r.get(f"{method}_donor_rank"), 1.0) / 8.0,
            ]
            by_method[method] = vals
            vec.extend(vals)
        vec.extend([a - b for a, b in zip(by_method["wordphase"], by_method["median"])])
        data.append(vec)

    X = np.asarray(data, dtype=np.float64).reshape(len(data), len(names))
    if X.shape[1] != len(names):
        raise RouterError(f"Feature dimension mismatch: {X.shape[1]} != {len(names)}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names, source_columns


def smooth_prob(p: np.ndarray) -> np.ndarray:
    pad = len(SMOOTH_KERNEL) // 2
    x = np.pad(np.asarray(p, dtype=np.float64), (pad, pad), mode="edge")
    s = np.convolve(x, SMOOTH_KERNEL, mode="valid")
    return s / max(float(s.sum()), 1e-12)


def frame_from_ratio(r: float, T: int) -> int:
    return int(np.clip(round(float(r) * max(1, T - 1)), 2, max(2, T - 3)))


def ratio_from_frame(frame: int, T: int) -> float:
    return float(frame) / max(1, T - 1)


def probability_at_ratio(prob: np.ndarray, ratio: float) -> float:
    p = np.asarray(prob, dtype=np.float64)
    x = float(np.clip(ratio, 0.0, 1.0)) * max(1, len(p) - 1)
    lo = int(math.floor(x))
    hi = min(len(p) - 1, lo + 1)
    a = x - lo
    return float((1.0 - a) * p[lo] + a * p[hi])


def cdf_at_ratio(prob: np.ndarray, ratio: float) -> float:
    p = np.asarray(prob, dtype=np.float64)
    idx = int(np.clip(round(float(ratio) * max(1, len(p) - 1)), 0, len(p) - 1))
    return float(np.sum(p[: idx + 1]))


def quantile_ratio(prob: np.ndarray, q: float) -> float:
    p = np.asarray(prob, dtype=np.float64)
    c = np.cumsum(p / max(float(p.sum()), 1e-12))
    idx = int(np.searchsorted(c, q, side="left"))
    return float(np.clip(idx / max(1, len(p) - 1), 0.0, 1.0))


def local_mode_ratios(prob: np.ndarray) -> list[tuple[float, float, int]]:
    s = smooth_prob(prob)
    ids: list[int] = []
    for i in range(len(s)):
        left = s[i - 1] if i > 0 else -np.inf
        right = s[i + 1] if i + 1 < len(s) else -np.inf
        if s[i] >= left and s[i] >= right:
            ids.append(i)
    if not ids:
        ids = [int(np.argmax(s))]
    ids.sort(key=lambda i: (-float(s[i]), i))
    return [(i / max(1, len(s) - 1), float(s[i]), i) for i in ids]


def hypothesis_type_vector(name: str) -> list[float]:
    return [
        float(name == "median"),
        float(name == "wp_expected" or name == "wordphase"),
        float(name.startswith("wp_mode")),
        float(name.startswith("wp_q")),
        float("std" in name),
        float(name.startswith("wp_bin")),
    ]


def ranker_feature_names() -> list[str]:
    names = [
        "domain_natural", "domain_synthetic", "hand_left", "hand_right",
        "clip_len_norm", "log_target_distance", "sin_target_azimuth", "cos_target_azimuth",
        "sin_target_elevation", "cos_target_elevation",
        "planned_ratio", "delta_to_median_norm", "abs_delta_to_median_norm",
        "delta_to_expected_norm", "abs_delta_to_expected_norm", "delta_to_mode1_norm",
        "wordphase_probability", "wordphase_log_probability", "wordphase_cdf",
        "wordphase_entropy_norm", "wordphase_max_prob", "wordphase_top_bin_margin",
        "wordphase_std_ratio",
        "hyp_median", "hyp_expected", "hyp_mode", "hyp_quantile", "hyp_uncertainty_offset", "hyp_other",
        "donor_rank_norm", "log1p_donor_distance",
        "detector_agreement", "planned_score1b", "self_score1b", "self_angle_norm",
        "self_plan_offset", "self_plan_abs", "det_spread", "det_mean_offset",
        "joint_jerk_log", "foot_sliding_log", "pose_deviation_log", "max_reach",
        "joint_limit_fraction",
        "self_score_x_agreement", "planned_x_self_score", "prob_x_self_score",
        "uncertainty_x_self_plan_abs",
    ]
    return names


def build_ranker_feature(
    query: Mapping[str, Any], timing: Mapping[str, Any], candidate: Mapping[str, Any],
) -> np.ndarray:
    T = max(1.0, fnum(query.get("clip_length"), 120.0) - 1.0)
    domain = str(query.get("domain", ""))
    hand = inum(query.get("hand"), -1)
    td = max(fnum(query.get("target_distance"), 1.0), 1e-4)
    az = math.radians(fnum(query.get("target_azimuth_deg"), 0.0))
    el = math.radians(fnum(query.get("target_elevation_deg"), 0.0))
    planned = fnum(candidate.get("planned_frame"), 0.0)
    self_frame = fnum(candidate.get("self_frame"), planned)
    det = np.asarray([
        fnum(candidate.get("det_source"), planned),
        fnum(candidate.get("det_refined"), planned),
        fnum(candidate.get("det_heuristic"), planned),
    ], dtype=np.float64)
    agreement = fnum(candidate.get("detector_agreement"), 0.0)
    planned_score = fnum(candidate.get("planned_score1b_proxy20"), 0.0)
    self_score = fnum(candidate.get("self_score1b_proxy20"), 0.0)
    self_plan_abs = abs(self_frame - planned) / T
    entropy = fnum(timing.get("wordphase_entropy_norm"), 0.0)
    hyp = str(timing.get("hypothesis", candidate.get("method", "other")))
    vec = [
        float(domain == "natural"), float(domain == "synthetic"),
        float(hand == 0), float(hand == 1),
        (T + 1.0) / 180.0, math.log(td), math.sin(az), math.cos(az), math.sin(el), math.cos(el),
        planned / T,
        fnum(timing.get("delta_to_median_norm"), 0.0), abs(fnum(timing.get("delta_to_median_norm"), 0.0)),
        fnum(timing.get("delta_to_expected_norm"), 0.0), abs(fnum(timing.get("delta_to_expected_norm"), 0.0)),
        fnum(timing.get("delta_to_mode1_norm"), 0.0),
        fnum(timing.get("wordphase_probability"), 0.0),
        fnum(timing.get("wordphase_log_probability"), math.log(1e-12)),
        fnum(timing.get("wordphase_cdf"), 0.5), entropy,
        fnum(timing.get("wordphase_max_prob"), 0.0),
        fnum(timing.get("wordphase_top_bin_margin"), 0.0),
        fnum(timing.get("wordphase_std_ratio"), 0.0),
        *hypothesis_type_vector(hyp),
        fnum(candidate.get("donor_rank"), 1.0) / 8.0,
        math.log1p(max(0.0, fnum(candidate.get("donor_distance"), 0.0))),
        agreement, planned_score, self_score,
        fnum(candidate.get("self_angle_deg"), 180.0) / 180.0,
        (self_frame - planned) / T, self_plan_abs,
        float(np.max(det) - np.min(det)) / T,
        (float(np.mean(det)) - planned) / T,
        math.log1p(max(0.0, fnum(candidate.get("joint_jerk"), 0.0)) * 1e5),
        math.log1p(max(0.0, fnum(candidate.get("foot_sliding"), 0.0)) * 1e4),
        math.log1p(max(0.0, fnum(candidate.get("pose_deviation"), 0.0)) * 1e4),
        fnum(candidate.get("max_reach"), 0.0),
        fnum(candidate.get("joint_limit_fraction"), 0.0),
        self_score * agreement,
        planned_score * self_score,
        fnum(timing.get("wordphase_probability"), 0.0) * self_score,
        entropy * self_plan_abs,
    ]
    arr = np.asarray(vec, dtype=np.float64)
    names = ranker_feature_names()
    if len(arr) != len(names):
        raise MHError(f"Ranker feature dimension mismatch: {len(arr)} != {len(names)}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def candidate_guard(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> bool:
    cand_self = fnum(candidate.get("self_score1b_proxy20"), -1.0)
    base_self = fnum(base.get("self_score1b_proxy20"), -1.0)
    cand_plan = fnum(candidate.get("planned_score1b_proxy20"), -1.0)
    base_plan = fnum(base.get("planned_score1b_proxy20"), -1.0)
    if cand_self < base_self - 0.05:
        return False
    if cand_plan < min(0.90, base_plan - 0.04):
        return False
    if fnum(candidate.get("joint_limit_fraction"), 1.0) > max(0.005, fnum(base.get("joint_limit_fraction"), 0.0) + 0.002):
        return False
    if fnum(candidate.get("foot_sliding"), 1e9) > max(1.5 * fnum(base.get("foot_sliding"), 0.0), fnum(base.get("foot_sliding"), 0.0) + 0.001):
        return False
    if fnum(candidate.get("pose_deviation"), 1e9) > max(1.5 * fnum(base.get("pose_deviation"), 0.0), fnum(base.get("pose_deviation"), 0.0) + 0.003):
        return False
    return True


def select_with_threshold(
    rows: Sequence[Mapping[str, Any]], predicted: np.ndarray, base_index: int, threshold: float,
) -> int:
    predicted = np.asarray(predicted, dtype=np.float64)
    if not rows or predicted.shape != (len(rows),) or not np.isfinite(predicted).all():
        raise ValueError('ranker predictions must be finite and match the candidate pool')
    if not isinstance(base_index, (int, np.integer)) or not 0 <= base_index < len(rows):
        raise ValueError('base candidate index outside pool')
    if not np.isfinite(threshold):
        raise ValueError('ranker replacement threshold must be finite')
    base = rows[base_index]
    eligible = [i for i, r in enumerate(rows) if i != base_index and candidate_guard(r, base)]
    if not eligible:
        return base_index
    best = max(eligible, key=lambda i: (float(predicted[i]), -inum(rows[i].get("donor_rank"), 999), -i))
    return best if float(predicted[best] - predicted[base_index]) >= threshold else base_index


def _build_hypotheses(
    dist: Mapping[str, Any], median_frame: int, count: int, min_sep_frames: int,
) -> list[dict[str, Any]]:
    T = int(dist["clip_length"])
    prob = np.asarray(dist["prob"], dtype=np.float64)
    mean = float(dist["expected_ratio"])
    std = float(dist["std_ratio"])
    proposals: list[tuple[str, int, int]] = []
    proposals.append(("median", int(np.clip(median_frame, 2, max(2, T - 3))), 0))
    proposals.append(("wp_expected", frame_from_ratio(mean, T), 1))
    for j, (r, _p, _idx) in enumerate(local_mode_ratios(prob)[:6], 1):
        proposals.append((f"wp_mode{j}", frame_from_ratio(r, T), 1 + j))
    proposals.extend([
        ("wp_q25", frame_from_ratio(quantile_ratio(prob, 0.25), T), 20),
        ("wp_q75", frame_from_ratio(quantile_ratio(prob, 0.75), T), 21),
        ("wp_mean_minus_std", frame_from_ratio(mean - std, T), 22),
        ("wp_mean_plus_std", frame_from_ratio(mean + std, T), 23),
    ])
    selected: list[tuple[str, int, int]] = []
    used: set[int] = set()
    # Exact duplicates are removed. Median and expected are preferred, but if they
    # collapse to one frame the freed slot is filled by a genuinely different mode.
    for name, frame, priority in proposals:
        if frame in used:
            continue
        if len(selected) >= 2 and any(abs(frame - f) < min_sep_frames for _, f, _ in selected):
            continue
        selected.append((name, frame, priority))
        used.add(frame)
        if len(selected) >= count:
            break
    if len(selected) < count:
        # Deterministic fill from highest-probability bins while retaining uniqueness.
        for idx in np.argsort(prob)[::-1].tolist():
            frame = frame_from_ratio(idx / max(1, len(prob) - 1), T)
            if frame in used:
                continue
            selected.append((f"wp_bin{idx}", frame, 100 + int(idx)))
            used.add(frame)
            if len(selected) >= count:
                break
    selected.sort(key=lambda z: z[2])
    out: list[dict[str, Any]] = []
    mode1_frame = frame_from_ratio(local_mode_ratios(prob)[0][0], T)
    for rank, (name, frame, _priority) in enumerate(selected[:count], 1):
        ratio = ratio_from_frame(frame, T)
        out.append({
            "hypothesis": name,
            "hypothesis_rank": rank,
            "planned_frame": frame,
            "planned_ratio": ratio,
            "wordphase_probability": probability_at_ratio(prob, ratio),
            "wordphase_log_probability": math.log(max(probability_at_ratio(prob, ratio), 1e-12)),
            "wordphase_cdf": cdf_at_ratio(prob, ratio),
            "wordphase_entropy_norm": float(dist["entropy_norm"]),
            "wordphase_max_prob": float(dist["max_prob"]),
            "wordphase_top_bin_margin": float(dist["top_bin_margin"]),
            "wordphase_std_ratio": float(dist["std_ratio"]),
            "delta_to_median_norm": (frame - median_frame) / max(1, T - 1),
            "delta_to_expected_norm": (frame - int(dist["expected_frame"])) / max(1, T - 1),
            "delta_to_mode1_norm": (frame - mode1_frame) / max(1, T - 1),
            "is_legacy_median": float(name == "median"),
            "is_legacy_expected": float(name == "wp_expected"),
            "is_mode": float(name.startswith("wp_mode")),
            "is_quantile": float(name.startswith("wp_q")),
            "is_uncertainty_offset": float("std" in name),
        })
    return out


def fit_domain_hand_medians(
    domains: Sequence[str], hands: Sequence[int], apex_ratios: Sequence[float],
) -> dict[tuple[str, int], float]:
    """Fit normalized medians on the training partition only; no implicit fallback."""
    domains, hands = list(domains), list(hands)
    ratios = np.asarray(apex_ratios, dtype=np.float64)
    if not domains or not (len(domains) == len(hands) == len(ratios)):
        raise ValueError('median training arrays must have equal nonzero length')
    if ratios.ndim != 1 or not np.isfinite(ratios).all() or np.any((ratios < 0) | (ratios > 1)):
        raise ValueError('apex ratios must be finite and within [0, 1]')
    grouped: dict[tuple[str, int], list[float]] = {}
    for domain, hand, ratio in zip(domains, hands, ratios):
        if domain not in {'natural', 'synthetic'} or hand not in (0, 1):
            raise ValueError('domain must be natural/synthetic and hand must be 0/1')
        grouped.setdefault((domain, int(hand)), []).append(float(ratio))
    return {key: float(np.median(values)) for key, values in grouped.items()}


def median_frame(medians: Mapping[tuple[str, int], float], domain: str, hand: int, frames: int) -> int:
    if frames < 5:
        raise ValueError('timing hypotheses require at least five frames')
    ratio = float(medians[(domain, hand)])
    if not np.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('invalid domain/hand median ratio')
    return frame_from_ratio(ratio, frames)


def phase_route(predicted_delta: float) -> str:
    """Choose learned timing at the frozen nonnegative predicted STG-delta boundary."""
    if not np.isfinite(predicted_delta):
        raise ValueError('router prediction must be finite')
    return 'wordphase' if predicted_delta >= 0 else 'median'


def build_hypotheses(
    prob: np.ndarray, frames: int, median_frame: int, count: int = 4,
    min_sep_frames: int = 6,
) -> list[dict[str, Any]]:
    """Frozen median/expectation/mode/quantile proposal order, with unique fill.

    Four hypotheses are the submitted setting. Small clips may expose fewer than
    four interior frames; asking for an impossible budget fails explicitly.
    """
    p = np.asarray(prob, dtype=np.float64)
    if p.shape != (64,) or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError('WordPhase distribution must have 64 finite nonnegative bins with positive mass')
    if not isinstance(frames, (int, np.integer)) or frames < 5:
        raise ValueError('at least five frames required')
    if not isinstance(count, (int, np.integer)) or not isinstance(min_sep_frames, (int, np.integer)) or count < 1 or count > frames - 4 or min_sep_frames < 0:
        raise ValueError('hypothesis count exceeds available interior frames or separation is negative')
    if not 0 <= median_frame < frames:
        raise ValueError('median frame outside clip')
    p = p / p.sum()
    centers = np.linspace(0., 1., len(p))
    mean = float(p @ centers)
    ordered = np.sort(p)
    dist = {
        'clip_length': frames, 'prob': p, 'expected_ratio': mean,
        'std_ratio': float(np.sqrt(p @ (centers - mean) ** 2)),
        'expected_frame': frame_from_ratio(mean, frames),
        'entropy_norm': float(-(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(len(p))),
        'max_prob': float(ordered[-1]), 'top_bin_margin': float(ordered[-1] - ordered[-2]),
    }
    result = _build_hypotheses(dist, median_frame, count, min_sep_frames)
    # Very long clips can expose fewer unique frame locations than requested via
    # 64 probability bins. Never silently return a reduced search budget.
    if len(result) != count:
        raise ValueError('distribution bins cannot supply requested number of distinct hypotheses')
    return result


def _matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError('features must be a nonempty finite (N,D) matrix')
    return X


@dataclass(frozen=True)
class RidgeModel:
    """Standardized weighted ridge, using only training-partition statistics."""
    beta: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        beta = np.asarray(self.beta, dtype=np.float64)
        if mean.ndim != 1 or not len(mean) or std.shape != mean.shape or beta.shape != (len(mean) + 1,):
            raise ValueError('ridge parameter dimensions do not agree')
        if not all(np.isfinite(v).all() for v in (beta, mean, std)) or np.any(std <= 0):
            raise ValueError('ridge parameters must be finite with positive standard deviations')
        object.__setattr__(self, 'mean', mean.copy())
        object.__setattr__(self, 'std', std.copy())
        object.__setattr__(self, 'beta', beta.copy())

    @classmethod
    def fit(
        cls, X: np.ndarray, y: np.ndarray, groups: Sequence[str], lam: float = .5,
        query_ids: Sequence[str] | None = None,
    ) -> RidgeModel:
        X = _matrix(X)
        y = np.asarray(y, dtype=np.float64)
        group = np.asarray(groups, dtype=str)
        if y.shape != (len(X),) or not np.isfinite(y).all() or group.shape != y.shape or np.any(group == ''):
            raise ValueError('targets and nonempty leakage groups must match feature rows')
        if not np.isfinite(lam) or lam < 0:
            raise ValueError('ridge regularization must be nonnegative and finite')
        counts = Counter(group.tolist())
        w = np.asarray([1 / counts[g] for g in group])
        if query_ids is not None:
            query = np.asarray(query_ids, dtype=str)
            if query.shape != y.shape or np.any(query == ''):
                raise ValueError('query identifiers must match feature rows')
            for q in np.unique(query):
                if len(np.unique(group[query == q])) != 1:
                    raise ValueError('a query must belong to one leakage group')
            qcounts = Counter(query.tolist())
            w /= np.asarray([qcounts[q] for q in query])
        w /= w.mean()
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.
        Z = np.column_stack([np.ones(len(X)), (X - mean) / std])
        gram = (Z.T * w) @ Z / w.sum()
        rhs = Z.T @ (w * y) / w.sum()
        regularizer = np.eye(Z.shape[1]) * lam
        regularizer[0, 0] = 0.
        try:
            beta = np.linalg.solve(gram + regularizer, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram + regularizer) @ rhs
        return cls(beta, mean, std)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _matrix(X)
        if X.shape[1] != len(self.mean):
            raise ValueError('ridge prediction feature dimension mismatch')
        return np.column_stack([np.ones(len(X)), (X - self.mean) / self.std]) @ self.beta

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': 1, 'beta': self.beta.tolist(), 'mean': self.mean.tolist(), 'std': self.std.tolist()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RidgeModel:
        if data.get('schema_version') != 1:
            raise ValueError('unsupported ridge schema version')
        return cls(np.asarray(data['beta']), np.asarray(data['mean']), np.asarray(data['std']))

    def save(self, path: str | Path) -> None:
        """Store numeric parameters only; existing checkpoints are not overwritten."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            np.savez(handle, beta=self.beta, feature_mean=self.mean, feature_std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> RidgeModel:
        with np.load(path, allow_pickle=False) as data:
            return cls(data['beta'], data['feature_mean'], data['feature_std'])


@dataclass(frozen=True)
class CrossFitResult:
    predictions: np.ndarray
    models: dict[int, RidgeModel]


def cross_fit_ridge(
    X: np.ndarray, y: np.ndarray, groups: Sequence[str], folds: Sequence[int],
    query_ids: Sequence[str] | None = None, lam: float = .5,
) -> CrossFitResult:
    """Fit one model per held-out fold; reject group/query leakage before fitting.

    Lambda must be chosen independently of outer validation labels. For threshold
    or lambda tuning use nested cross-fitting on each outer training partition.
    """
    X = _matrix(X)
    y, group, folds = np.asarray(y), np.asarray(groups, dtype=str), np.asarray(folds)
    if y.shape != (len(X),) or group.shape != y.shape or folds.shape != y.shape:
        raise ValueError('cross-fit arrays must match feature rows')
    if not np.issubdtype(folds.dtype, np.integer) or len(np.unique(folds)) < 2 or np.any(folds < 0):
        raise ValueError('at least two nonnegative integer folds are required')
    query = None if query_ids is None else np.asarray(query_ids, dtype=str)
    if query is not None and query.shape != y.shape:
        raise ValueError('query identifiers must match feature rows')
    for partition in (group,) if query is None else (group, query):
        for identifier in np.unique(partition):
            if len(np.unique(folds[partition == identifier])) != 1:
                raise ValueError('leakage group or query spans multiple folds')
    predictions = np.full(len(X), np.nan)
    models = {}
    for fold in np.unique(folds):
        train = folds != fold
        model = RidgeModel.fit(X[train], y[train], group[train], lam, None if query is None else query[train])
        predictions[~train] = model.predict(X[~train])
        models[int(fold)] = model
    return CrossFitResult(predictions, models)


@dataclass(frozen=True)
class CalibratedRanker:
    model: RidgeModel
    ridge_lambda: float
    replacement_threshold: float
    calibration: list[dict[str, float | int]]


def nested_ranker_calibration(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    folds: Sequence[int],
    query_ids: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
    base_indices: Mapping[str, int],
    lambdas: Sequence[float] = (.5,),
    thresholds: Sequence[float] = (-.02, 0., .01, .02, .05, .10),
) -> dict[int, CalibratedRanker]:
    """Tune each outer ranker on inner held-out queries, following frozen P2A-MH.

    Candidate rows contain inference-safe generation diagnostics. `y` contains
    post-generation training utilities. Base indices are global row indices for
    the unchanged Candidate A fallback. No outer-fold labels enter calibration.
    """
    X = _matrix(X)
    y, group = np.asarray(y, dtype=np.float64), np.asarray(groups, dtype=str)
    fold, query = np.asarray(folds), np.asarray(query_ids, dtype=str)
    if len(candidates) != len(X) or y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError('candidate rows and finite targets must match features')
    if len(np.unique(fold)) < 3:
        raise ValueError('nested calibration requires at least three folds')
    if group.shape != y.shape or fold.shape != y.shape or query.shape != y.shape:
        raise ValueError('groups, folds, and query identifiers must match feature rows')
    if not np.issubdtype(fold.dtype, np.integer) or np.any(fold < 0):
        raise ValueError('nonnegative integer folds required')
    if np.any(group == '') or np.any(query == ''):
        raise ValueError('nonempty leakage groups and query identifiers required')
    # Validate disjointness without fitting any model on outer-fold targets.
    for partition in (group, query):
        for identifier in np.unique(partition):
            if len(np.unique(fold[partition == identifier])) != 1:
                raise ValueError('leakage group or query spans multiple folds')
    for q in np.unique(query):
        if len(np.unique(group[query == q])) != 1:
            raise ValueError('a query must belong to one leakage group')
    lambdas, thresholds = tuple(lambdas), tuple(thresholds)
    if not lambdas or not thresholds or any(not np.isfinite(v) or v < 0 for v in lambdas) or any(not np.isfinite(v) for v in thresholds):
        raise ValueError('finite nonempty regularization and threshold grids required')
    by_query = {q: np.flatnonzero(query == q) for q in np.unique(query)}
    if set(base_indices) != set(by_query):
        raise ValueError('one fallback index per query is required')
    for q, ids in by_query.items():
        if base_indices[q] not in ids:
            raise ValueError('fallback candidate must belong to its query')
    result = {}
    for outer in np.unique(fold):
        train = fold != outer
        audit = []
        best = None
        for lam in lambdas:
            inner = cross_fit_ridge(X[train], y[train], group[train], fold[train], query[train], lam)
            predicted = np.full(len(X), np.nan)
            predicted[train] = inner.predictions
            for threshold in thresholds:
                selected_utility, base_utility = [], []
                for q, ids in by_query.items():
                    if not train[ids[0]]:
                        continue
                    base = int(base_indices[q])
                    base_local = int(np.flatnonzero(ids == base)[0])
                    selected_local = select_with_threshold(
                        [candidates[i] for i in ids], predicted[ids], base_local, threshold,
                    )
                    selected_utility.append(float(y[ids[selected_local]]))
                    base_utility.append(float(y[base]))
                gain = float(np.mean(selected_utility) - np.mean(base_utility))
                audit.append({
                    'outer_fold': int(outer), 'ridge_lambda': float(lam),
                    'replacement_threshold': float(threshold),
                    'inner_query_count': len(selected_utility), 'inner_gain': gain,
                })
                # Frozen tie order: larger gain, smaller absolute threshold,
                # then smaller regularization. Grid order resolves remaining ties.
                key = (gain, -abs(threshold), -lam)
                if best is None or key > best[0]:
                    best = (key, float(lam), float(threshold))
        _, lam, threshold = best
        model = RidgeModel.fit(X[train], y[train], group[train], lam, query[train])
        result[int(outer)] = CalibratedRanker(model, lam, threshold, audit)
    return result
