#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import socket
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

VERSION = "0.1.0"
SCHEMA_VERSION = "1.0.0"
EXPECTED_FOLDS = (0, 1, 2, 3, 4)


class RouterError(RuntimeError):
    pass


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


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


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build_features(rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, list[str], list[str]]:
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

    X = np.asarray(data, dtype=np.float64)
    if X.shape[1] != len(names):
        raise RouterError(f"Feature dimension mismatch: {X.shape[1]} != {len(names)}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names, source_columns


def fit_weighted_ridge(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd
    A = np.concatenate([np.ones((len(Z), 1), dtype=np.float64), Z], axis=1)
    counts = Counter(groups.tolist())
    w = np.asarray([1.0 / max(1, counts[g]) for g in groups], dtype=np.float64)
    w /= max(np.mean(w), 1e-12)
    mass = max(float(np.sum(w)), 1e-12)
    gram = (A.T * w) @ A / mass
    rhs = A.T @ (w * y) / mass
    reg = np.eye(A.shape[1], dtype=np.float64) * float(lam)
    reg[0, 0] = 0.0
    try:
        beta = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram + reg) @ rhs
    return beta, mu, sd


def ridge_predict(X: np.ndarray, beta: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sd
    A = np.concatenate([np.ones((len(Z), 1), dtype=np.float64), Z], axis=1)
    return A @ beta


def sample_bootstrap_ci(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        vals[i] = float(np.mean(diff[rng.integers(0, n, size=n)]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def cluster_bootstrap_ci(
    diff: np.ndarray, groups: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[g] for g in chosen])
        vals[i] = float(np.mean(diff[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def metric_array(rows: Sequence[Mapping[str, str]], method: str, metric: str) -> np.ndarray:
    return np.asarray([fnum(r.get(f"{method}_{metric}")) for r in rows], dtype=np.float64)


def selected_array(
    rows: Sequence[Mapping[str, str]], choose_wp: np.ndarray, metric: str
) -> np.ndarray:
    wp = metric_array(rows, "wordphase", metric)
    med = metric_array(rows, "median", metric)
    return np.where(choose_wp, wp, med)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-full-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ridge-lambda", type=float, default=0.5)
    ap.add_argument("--sample-bootstrap", type=int, default=2000)
    ap.add_argument("--cluster-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accept = json_load(args.p1_full_dir / "P1_FULL_ACCEPTANCE.json")
    coverage = json_load(args.p1_full_dir / "coverage_audit.json")
    parity = json_load(args.p1_full_dir / "method_parity_audit.json")
    gt_audit = json_load(args.p1_full_dir / "validation_gt_usage_audit.json")
    if accept.get("status") != "P1_WORDPHASE_FULL_OOF_ACCEPTED" or not accept.get("p1_frozen") or not accept.get("p2_allowed"):
        raise RouterError(f"P1 gate is not open: {accept}")
    if coverage.get("status") != "PASS" or inum(coverage.get("schema_error_count"), -1) != 0:
        raise RouterError(f"P1 coverage gate is not PASS: {coverage}")
    if parity.get("status") != "PASS" or gt_audit.get("status") != "PASS":
        raise RouterError("P1 method-parity or validation-GT audit is not PASS")

    rows = read_csv(args.p1_full_dir / "per_sample_comparison.csv")
    expected_n = inum(coverage.get("expected_query_count"), 2525)
    if len(rows) != expected_n:
        raise RouterError(f"Unexpected row count: {len(rows)} != {expected_n}")
    sample_ids = [str(r.get("sample_id", "")) for r in rows]
    if len(set(sample_ids)) != len(sample_ids) or any(not x for x in sample_ids):
        raise RouterError("sample_id uniqueness gate failed")
    folds = np.asarray([inum(r.get("fold")) for r in rows], dtype=np.int16)
    if tuple(sorted(set(folds.tolist()))) != EXPECTED_FOLDS:
        raise RouterError(f"Unexpected fold IDs: {sorted(set(folds.tolist()))}")
    groups = np.asarray([str(r.get("leakage_group_id", "")) for r in rows], dtype="U128")
    if np.any(groups == ""):
        raise RouterError("Missing leakage_group_id")

    X, feature_names, source_columns = build_features(rows)
    wp_stg = metric_array(rows, "wordphase", "self_stg_proxy_sigma5")
    med_stg = metric_array(rows, "median", "self_stg_proxy_sigma5")
    target_delta = wp_stg - med_stg
    if not np.all(np.isfinite(target_delta)):
        raise RouterError("Non-finite STG targets")

    oof_pred = np.full(len(rows), np.nan, dtype=np.float64)
    model_rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        tr = folds != fold
        va = folds == fold
        beta, mu, sd = fit_weighted_ridge(X[tr], target_delta[tr], groups[tr], args.ridge_lambda)
        oof_pred[va] = ridge_predict(X[va], beta, mu, sd)
        np.savez(
            args.output_dir / f"phase_router_fold{fold}.npz",
            beta=beta, feature_mean=mu, feature_std=sd,
            feature_names=np.asarray(feature_names, dtype="U128"),
            ridge_lambda=np.asarray([args.ridge_lambda], dtype=np.float64),
            fold=np.asarray([fold], dtype=np.int16),
            decision=np.asarray(["wordphase_if_predicted_delta_ge_0_else_median"], dtype="U96"),
        )
        model_rows.append({
            "fold": fold, "train_rows": int(np.sum(tr)), "validation_rows": int(np.sum(va)),
            "train_groups": int(len(np.unique(groups[tr]))),
            "validation_groups": int(len(np.unique(groups[va]))),
        })
    if not np.all(np.isfinite(oof_pred)):
        raise RouterError("OOF router prediction coverage failed")

    beta_all, mu_all, sd_all = fit_weighted_ridge(X, target_delta, groups, args.ridge_lambda)
    np.savez(
        args.output_dir / "phase_router_full_model.npz",
        beta=beta_all, feature_mean=mu_all, feature_std=sd_all,
        feature_names=np.asarray(feature_names, dtype="U128"),
        ridge_lambda=np.asarray([args.ridge_lambda], dtype=np.float64),
        decision=np.asarray(["wordphase_if_predicted_delta_ge_0_else_median"], dtype="U96"),
    )
    write_csv(args.output_dir / "fold_training_manifest.csv", model_rows)

    choose_wp = oof_pred >= 0.0
    oracle_choose_wp = wp_stg >= med_stg
    self_score_choose_wp = (
        metric_array(rows, "wordphase", "self_score1b_proxy20") >=
        metric_array(rows, "median", "self_score1b_proxy20")
    )

    eval_metrics = [
        "self_stg_proxy_sigma5", "self_score1b_proxy20", "temporal_proxy_sigma5",
        "self_angle_deg", "joint_jerk", "foot_sliding", "joint_limit_fraction",
        "pose_deviation", "detector_agreement",
    ]
    selected = {m: selected_array(rows, choose_wp, m) for m in eval_metrics}
    oracle_selected = {m: selected_array(rows, oracle_choose_wp, m) for m in eval_metrics}
    score_router_selected = {m: selected_array(rows, self_score_choose_wp, m) for m in eval_metrics}

    pred_rows: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        rec: dict[str, Any] = {
            "sample_id": r.get("sample_id"), "recording_id": r.get("recording_id"),
            "leakage_group_id": r.get("leakage_group_id"), "fold": inum(r.get("fold")),
            "domain": r.get("domain"), "hand": inum(r.get("hand")),
            "predicted_wordphase_minus_median_stg": float(oof_pred[i]),
            "router_choice": "wordphase" if choose_wp[i] else "median",
            "oracle_choice_diagnostic_only": "wordphase" if oracle_choose_wp[i] else "median",
            "wordphase_actual_self_stg": float(wp_stg[i]),
            "median_actual_self_stg": float(med_stg[i]),
            "router_actual_self_stg": float(selected["self_stg_proxy_sigma5"][i]),
        }
        for m in eval_metrics[1:]:
            rec[f"router_{m}"] = float(selected[m][i])
        pred_rows.append(rec)
    write_csv(args.output_dir / "oof_phase_router_predictions.csv", pred_rows)

    def indices_for(group_name: str, group_value: str) -> np.ndarray:
        if group_name == "overall":
            return np.arange(len(rows))
        if group_name == "fold":
            return np.flatnonzero(folds == int(group_value))
        if group_name == "domain":
            return np.flatnonzero(np.asarray([str(r.get("domain")) for r in rows]) == group_value)
        if group_name == "hand":
            return np.flatnonzero(np.asarray([str(inum(r.get("hand"))) for r in rows]) == group_value)
        if group_name == "domain_hand":
            domain, hand = group_value.rsplit("_hand", 1)
            return np.flatnonzero(
                (np.asarray([str(r.get("domain")) for r in rows]) == domain) &
                (np.asarray([str(inum(r.get("hand"))) for r in rows]) == hand)
            )
        raise RouterError(f"Unknown group: {group_name}")

    groups_to_report: list[tuple[str, str]] = [("overall", "overall")]
    groups_to_report += [("fold", str(f)) for f in EXPECTED_FOLDS]
    groups_to_report += [("domain", d) for d in ("natural", "synthetic")]
    groups_to_report += [("hand", h) for h in ("0", "1")]
    groups_to_report += [("domain_hand", f"{d}_hand{h}") for d in ("natural", "synthetic") for h in ("0", "1")]

    router_metric_rows: list[dict[str, Any]] = []
    variants = {
        "wordphase": {m: metric_array(rows, "wordphase", m) for m in eval_metrics},
        "median": {m: metric_array(rows, "median", m) for m in eval_metrics},
        "self_score_rule": score_router_selected,
        "phase_router": selected,
        "oracle_upper_bound_diagnostic_only": oracle_selected,
    }
    for group_name, group_value in groups_to_report:
        idx = indices_for(group_name, group_value)
        for variant, metrics in variants.items():
            rec: dict[str, Any] = {
                "variant": variant, "group_name": group_name, "group_value": group_value,
                "n": len(idx), "wordphase_choice_fraction": (
                    float(np.mean(choose_wp[idx])) if variant == "phase_router" else ""
                ),
            }
            for m in eval_metrics:
                rec[m] = float(np.mean(metrics[m][idx]))
            router_metric_rows.append(rec)
    write_csv(args.output_dir / "router_metrics.csv", router_metric_rows)

    fold_rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        idx = np.flatnonzero(folds == fold)
        fold_rows.append({
            "fold": fold, "n": len(idx),
            "wordphase_self_stg": float(np.mean(wp_stg[idx])),
            "median_self_stg": float(np.mean(med_stg[idx])),
            "router_self_stg": float(np.mean(selected["self_stg_proxy_sigma5"][idx])),
            "router_minus_wordphase_stg": float(np.mean(selected["self_stg_proxy_sigma5"][idx] - wp_stg[idx])),
            "router_minus_median_stg": float(np.mean(selected["self_stg_proxy_sigma5"][idx] - med_stg[idx])),
            "router_wordphase_choice_fraction": float(np.mean(choose_wp[idx])),
        })
    write_csv(args.output_dir / "fold_router_results.csv", fold_rows)

    primary_diff = selected["self_stg_proxy_sigma5"] - wp_stg
    median_diff = selected["self_stg_proxy_sigma5"] - med_stg
    sample_ci = sample_bootstrap_ci(primary_diff, args.sample_bootstrap, args.seed + 11)
    cluster_ci = cluster_bootstrap_ci(primary_diff, groups, args.cluster_bootstrap, args.seed + 29)
    sample_ci_med = sample_bootstrap_ci(median_diff, args.sample_bootstrap, args.seed + 31)
    cluster_ci_med = cluster_bootstrap_ci(median_diff, groups, args.cluster_bootstrap, args.seed + 47)

    fold_positive = sum(r["router_minus_wordphase_stg"] > 0 for r in fold_rows)
    dh_positive = 0
    for d in ("natural", "synthetic"):
        for h in (0, 1):
            idx = indices_for("domain_hand", f"{d}_hand{h}")
            dh_positive += int(float(np.mean(primary_diff[idx])) > 0)

    wp_score = metric_array(rows, "wordphase", "self_score1b_proxy20")
    router_score = selected["self_score1b_proxy20"]
    score_diff = float(np.mean(router_score - wp_score))
    spatial_guard = score_diff >= -0.005

    quality_limits = {
        "joint_jerk": float(np.mean(metric_array(rows, "wordphase", "joint_jerk")) + 5e-5),
        "foot_sliding": float(np.mean(metric_array(rows, "wordphase", "foot_sliding")) + 2e-4),
        "joint_limit_fraction": float(np.mean(metric_array(rows, "wordphase", "joint_limit_fraction")) + 0.005),
        "pose_deviation": float(np.mean(metric_array(rows, "wordphase", "pose_deviation")) + 5e-4),
    }
    quality_components: dict[str, Any] = {}
    for m, limit in quality_limits.items():
        mean_v = float(np.mean(selected[m]))
        quality_components[m] = {
            "router_mean": mean_v,
            "wordphase_mean": float(np.mean(metric_array(rows, "wordphase", m))),
            "allowed_max": limit,
            "pass": mean_v <= limit,
        }
    quality_gate = all(x["pass"] for x in quality_components.values())

    engineering_gate = (
        len(rows) == expected_n and np.all(np.isfinite(oof_pred)) and
        tuple(sorted(set(folds.tolist()))) == EXPECTED_FOLDS
    )
    primary_gate = (
        float(np.mean(primary_diff)) > 0 and sample_ci[0] > 0 and cluster_ci[0] > 0 and
        fold_positive >= 4 and dh_positive >= 3
    )
    if engineering_gate and primary_gate and spatial_guard and quality_gate:
        status = "P2A_PHASE_ROUTER_ACCEPTED"
        p2b_allowed = True
    else:
        status = "P2A_PHASE_ROUTER_COMPLETED_NEEDS_REFINEMENT"
        p2b_allowed = False

    oracle_mean = float(np.mean(oracle_selected["self_stg_proxy_sigma5"]))
    router_mean = float(np.mean(selected["self_stg_proxy_sigma5"]))
    wp_mean = float(np.mean(wp_stg))
    med_mean = float(np.mean(med_stg))
    oracle_headroom = {
        "wordphase_mean_self_stg": wp_mean,
        "median_mean_self_stg": med_mean,
        "phase_router_mean_self_stg": router_mean,
        "oracle_wordphase_or_median_mean_self_stg_diagnostic_only": oracle_mean,
        "router_gain_over_wordphase": router_mean - wp_mean,
        "remaining_oracle_headroom_over_router": oracle_mean - router_mean,
        "median_better_than_wordphase_fraction": float(np.mean(med_stg > wp_stg)),
        "wordphase_better_than_median_fraction": float(np.mean(wp_stg > med_stg)),
        "tie_fraction": float(np.mean(wp_stg == med_stg)),
        "warning": "Oracle uses released post-generation labels and is diagnostic only; it is not deployable.",
    }
    json_dump(args.output_dir / "oracle_headroom.json", oracle_headroom)

    failures: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        loss = float(primary_diff[i])
        if loss < -0.05:
            failures.append({
                "sample_id": r.get("sample_id"), "fold": inum(r.get("fold")),
                "domain": r.get("domain"), "hand": inum(r.get("hand")),
                "router_choice": "wordphase" if choose_wp[i] else "median",
                "router_predicted_delta": float(oof_pred[i]),
                "router_minus_wordphase_stg": loss,
                "wordphase_self_stg": float(wp_stg[i]),
                "median_self_stg": float(med_stg[i]),
                "wordphase_peak": inum(r.get("wordphase_peak")),
                "median_peak": inum(r.get("median_peak")),
            })
    failures.sort(key=lambda r: (r["router_minus_wordphase_stg"], str(r["sample_id"])))
    write_csv(args.output_dir / "router_failure_cases.csv", failures)

    feature_contract = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "status": "FROZEN_FOR_P2A_OOF",
        "model": "inverse-leakage-group-weighted ridge regression on WordPhase-minus-median self-STG delta",
        "ridge_lambda": args.ridge_lambda,
        "decision": "choose WordPhase when predicted delta >= 0, otherwise choose OOF-median timing",
        "feature_names": feature_names,
        "source_columns": source_columns,
        "explicitly_excluded": [
            "released/GT peak", "hold start/end/width", "alignment labels",
            "distance-to-GT-hold", "temporal proxy labels", "oracle frames/scores",
            "scene graph", "listener state",
        ],
        "inference_requirements": [
            "WordPhase and median planned peaks", "both generated candidate pools",
            "frozen local detector diagnostics", "frozen spatial proxy", "query context",
        ],
        "guardrail": "The router is optimized against transparent OOF local proxy labels and is not represented as an organizer model or private scorer.",
    }
    json_dump(args.output_dir / "phase_router_contract_v0_1_0.json", feature_contract)

    acceptance = {
        "schema_version": SCHEMA_VERSION, "version": VERSION,
        "created_utc": now_utc(), "status": status,
        "query_count": len(rows), "fold_count": len(EXPECTED_FOLDS),
        "leakage_group_count": int(len(np.unique(groups))),
        "engineering_gate": engineering_gate, "primary_gate": primary_gate,
        "spatial_guard": spatial_guard, "quality_gate": quality_gate,
        "positive_folds_vs_wordphase_stg": fold_positive,
        "positive_domain_hand_slices_vs_wordphase_stg": dh_positive,
        "required_positive_folds": 4, "required_positive_domain_hand_slices": 3,
        "router_wordphase_choice_fraction": float(np.mean(choose_wp)),
        "primary": {
            "wordphase_mean_self_stg": wp_mean,
            "median_mean_self_stg": med_mean,
            "router_mean_self_stg": router_mean,
            "router_minus_wordphase_self_stg": float(np.mean(primary_diff)),
            "sample_bootstrap95": list(sample_ci),
            "cluster_bootstrap95": list(cluster_ci),
            "router_minus_median_self_stg": float(np.mean(median_diff)),
            "router_vs_median_sample_bootstrap95": list(sample_ci_med),
            "router_vs_median_cluster_bootstrap95": list(cluster_ci_med),
            "wordphase_mean_self_score1b": float(np.mean(wp_score)),
            "router_mean_self_score1b": float(np.mean(router_score)),
            "router_minus_wordphase_self_score1b": score_diff,
            "wordphase_mean_temporal_proxy_sigma5": float(np.mean(metric_array(rows, "wordphase", "temporal_proxy_sigma5"))),
            "router_mean_temporal_proxy_sigma5": float(np.mean(selected["temporal_proxy_sigma5"])),
        },
        "quality_components": quality_components,
        "oracle_upper_bound_diagnostic": oracle_headroom,
        "p2b_allowed": p2b_allowed,
        "next_action": (
            "Freeze the phase router and run a bounded residual-motion refinement pilot with P1 timing, retrieval, IK and router kept frozen."
            if p2b_allowed else
            "Keep P1 WordPhaseNet as the primary method; inspect router failures before any further timing-mixture work."
        ),
        "guardrails": [
            "Outer held-out fold is never used to fit its router.",
            "Inverse leakage-group weighting prevents duplicated rows from dominating training.",
            "Inference features exclude released peak, hold, alignment, temporal-label and oracle fields.",
            "Local proxy results do not establish organizer-server performance.",
            "No residual flow, scene graph or listener state is introduced in P2A.",
        ],
    }
    json_dump(args.output_dir / "P2A_ACCEPTANCE.json", acceptance)

    source_binding = []
    for name in [
        "P1_FULL_ACCEPTANCE.json", "coverage_audit.json", "method_parity_audit.json",
        "validation_gt_usage_audit.json", "per_sample_comparison.csv",
    ]:
        p = args.p1_full_dir / name
        source_binding.append({
            "name": name, "path": str(p), "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    json_dump(args.output_dir / "source_binding.json", source_binding)

    env = {
        "created_utc": now_utc(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": sys.version,
        "python_executable": sys.executable, "cwd": os.getcwd(),
    }
    json_dump(args.output_dir / "environment.json", env)

    summary = [
        "REPOINTER P2A — UNCERTAINTY-AWARE PHASE ROUTER, FIVE-FOLD OOF",
        "=" * 78,
        f"version={VERSION}", f"created_utc={now_utc()}",
        f"p1_full_dir={args.p1_full_dir}", f"output_dir={args.output_dir}",
        "dataset_writes=0", "downloads=0", "motion_generation=0",
        "smplx_forward_kinematics=0", "ik=0", "official_test_inference=0",
        "git_writes=0", "training=weighted_ridge_sample_level_router_only", "",
        "[COVERAGE]",
        f"queries={len(rows)}", f"folds={len(EXPECTED_FOLDS)}",
        f"leakage_groups={len(np.unique(groups))}", f"feature_dim={X.shape[1]}", "",
        "[PRIMARY]",
        f"wordphase_self_stg={wp_mean}", f"median_self_stg={med_mean}",
        f"router_self_stg={router_mean}",
        f"router_minus_wordphase_self_stg={float(np.mean(primary_diff))}",
        f"router_vs_wordphase_sample_ci95={list(sample_ci)}",
        f"router_vs_wordphase_cluster_ci95={list(cluster_ci)}",
        f"positive_folds_vs_wordphase={fold_positive}/5",
        f"positive_domain_hand_slices_vs_wordphase={dh_positive}/4",
        f"wordphase_self_score1b={float(np.mean(wp_score))}",
        f"router_self_score1b={float(np.mean(router_score))}",
        f"router_minus_wordphase_self_score1b={score_diff}",
        f"oracle_upper_bound_diagnostic={oracle_mean}",
        f"remaining_oracle_headroom={oracle_mean - router_mean}", "",
        "[ACCEPTANCE]",
        f"status={status}", f"engineering_gate={engineering_gate}",
        f"primary_gate={primary_gate}", f"spatial_guard={spatial_guard}",
        f"quality_gate={quality_gate}", f"p2b_allowed={p2b_allowed}",
        f"next_action={acceptance['next_action']}", "",
        "[PRIMARY_ARTIFACTS]",
        "SUMMARY.txt", "P2A_ACCEPTANCE.json", "phase_router_contract_v0_1_0.json",
        "oof_phase_router_predictions.csv", "router_metrics.csv", "fold_router_results.csv",
        "oracle_headroom.json", "router_failure_cases.csv",
        "phase_router_fold0.npz ... phase_router_fold4.npz", "phase_router_full_model.npz",
    ]
    (args.output_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest = []
    for p in sorted(args.output_dir.rglob("*")):
        if p.is_file() and p.name != "run_file_manifest.json":
            manifest.append({
                "relative_path": str(p.relative_to(args.output_dir)),
                "size_bytes": p.stat().st_size, "sha256": sha256_file(p),
            })
    json_dump(args.output_dir / "run_file_manifest.json", manifest)
    print("\n".join(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
