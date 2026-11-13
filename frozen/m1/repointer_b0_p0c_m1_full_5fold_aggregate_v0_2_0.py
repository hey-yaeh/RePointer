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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

VERSION = "0.2.0"
SCHEMA_VERSION = "1.0.0"
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
METHODS = ("b0", "b0eq", "p0c")
COMPARISONS = (
    ("p0c_vs_b0_primary", "b0"),
    ("p0c_vs_b0_equal_budget", "b0eq"),
)
HIGHER_IS_BETTER = {
    "self_score1b_proxy20",
    "self_stg_proxy_sigma5",
    "oracle_score1b_proxy20",
    "planned_score1b_proxy20",
    "detector_agreement",
}
LOWER_IS_BETTER = {
    "self_angle_deg",
    "oracle_angle_deg",
    "joint_jerk",
    "foot_sliding",
    "joint_limit_fraction",
    "pose_deviation",
}
METRICS = tuple([
    "self_score1b_proxy20",
    "self_stg_proxy_sigma5",
    "oracle_score1b_proxy20",
    "planned_score1b_proxy20",
    "self_angle_deg",
    "oracle_angle_deg",
    "joint_jerk",
    "foot_sliding",
    "joint_limit_fraction",
    "pose_deviation",
    "detector_agreement",
])
PRIMARY_METRICS = ("self_score1b_proxy20", "self_stg_proxy_sigma5")
QUALITY_LIMITS = {
    # P0C must not exceed B0 by more than the larger of the relative and
    # additive margins. These are materially stricter than the pilot gates.
    "joint_jerk": {"relative_increase": 0.20, "absolute_increase": 1e-4},
    "foot_sliding": {"relative_increase": 0.25, "absolute_increase": 5e-4},
    "joint_limit_fraction": {"relative_increase": 0.0, "absolute_increase": 0.005},
    "pose_deviation": {"relative_increase": 0.25, "absolute_increase": 0.002},
}
CLUSTER_CI_METRICS = {
    "self_score1b_proxy20",
    "self_stg_proxy_sigma5",
    "oracle_score1b_proxy20",
    "joint_jerk",
    "foot_sliding",
    "pose_deviation",
}
CLUSTER_CI_GROUPS = {"overall", "domain", "hand"}


class AggregateError(RuntimeError):
    pass


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def is_blank(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def fnum(v: Any, default: float = float("nan")) -> float:
    if is_blank(v):
        return default
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def inum(v: Any, default: int = -1) -> int:
    x = fnum(v)
    return int(round(x)) if np.isfinite(x) else default


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=json_default)
        f.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AggregateError(f"Missing CSV: {path}")
    if path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


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
            clean: dict[str, Any] = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=json_default)
                elif isinstance(value, float) and not np.isfinite(value):
                    value = ""
                clean[key] = value
            w.writerow(clean)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(text: str, base: int) -> int:
    h = hashlib.sha256(f"{base}:{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32 - 1)


def sample_bootstrap_ci(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    batch = 250
    for start in range(0, n_boot, batch):
        take = min(batch, n_boot - start)
        idx = rng.integers(0, len(values), size=(take, len(values)))
        means[start:start + take] = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cluster_bootstrap_ci(values: np.ndarray, clusters: Sequence[str], seed: int, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    clusters_arr = np.asarray([str(x) for x in clusters], dtype=object)
    valid = np.isfinite(values)
    values = values[valid]
    clusters_arr = clusters_arr[valid]
    if len(values) == 0:
        return float("nan"), float("nan")
    uniq, inv = np.unique(clusters_arr, return_inverse=True)
    if len(uniq) == 1:
        return float(np.mean(values)), float(np.mean(values))
    sums = np.bincount(inv, weights=values, minlength=len(uniq)).astype(np.float64)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    batch = 200
    for start in range(0, n_boot, batch):
        take = min(batch, n_boot - start)
        idx = rng.integers(0, len(uniq), size=(take, len(uniq)))
        means[start:start + take] = sums[idx].sum(axis=1) / np.maximum(counts[idx].sum(axis=1), 1.0)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def method_motion_dir(method: str) -> str:
    return "b0_equal_budget" if method == "b0eq" else method


def metric_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    raise AggregateError(f"Unknown metric direction: {metric}")


def group_records(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "overall", rows)]
    for fold in EXPECTED_FOLDS:
        groups.append(("fold", str(fold), [r for r in rows if inum(r.get("fold")) == fold]))
    for domain in ("natural", "synthetic"):
        groups.append(("domain", domain, [r for r in rows if str(r.get("domain")) == domain]))
    for hand in (0, 1):
        groups.append(("hand", str(hand), [r for r in rows if inum(r.get("hand")) == hand]))
    for domain in ("natural", "synthetic"):
        for hand in (0, 1):
            value = f"{domain}_hand{hand}"
            groups.append(("domain_hand", value, [
                r for r in rows if str(r.get("domain")) == domain and inum(r.get("hand")) == hand
            ]))
    for field in ("distance_bin", "azimuth_bin", "elevation_bin"):
        for value in sorted({str(r.get(field, "")) for r in rows if str(r.get(field, ""))}):
            groups.append((field, value, [r for r in rows if str(r.get(field, "")) == value]))
    return groups


def summarize(
    rows: list[dict[str, Any]],
    baseline: str,
    comparison: str,
    group_name: str,
    group_value: str,
    metric: str,
    seed: int,
    sample_boot: int,
    cluster_boot: int,
) -> dict[str, Any]:
    b = np.asarray([fnum(r.get(f"{baseline}_{metric}")) for r in rows], dtype=np.float64)
    p = np.asarray([fnum(r.get(f"p0c_{metric}")) for r in rows], dtype=np.float64)
    clusters = np.asarray([str(r.get("leakage_group_id") or r.get("sample_id")) for r in rows], dtype=object)
    valid = np.isfinite(b) & np.isfinite(p)
    b, p, clusters = b[valid], p[valid], clusters[valid]
    if len(b) == 0:
        raise AggregateError(f"No valid values for {comparison}/{group_name}/{group_value}/{metric}")
    raw = p - b
    direction = metric_direction(metric)
    improvement = raw if direction == "higher_is_better" else -raw
    slo, shi = sample_bootstrap_ci(improvement, stable_seed(f"sample:{comparison}:{group_name}:{group_value}:{metric}", seed), sample_boot)
    clo = chi = float("nan")
    cluster_supported = False
    if group_name in CLUSTER_CI_GROUPS and metric in CLUSTER_CI_METRICS:
        clo, chi = cluster_bootstrap_ci(
            improvement, clusters,
            stable_seed(f"cluster:{comparison}:{group_name}:{group_value}:{metric}", seed),
            cluster_boot,
        )
        cluster_supported = bool(np.isfinite(clo) and clo > 0)
    return {
        "comparison": comparison,
        "baseline_method": baseline,
        "group_name": group_name,
        "group_value": group_value,
        "metric": metric,
        "metric_direction": direction,
        "n": len(raw),
        "cluster_count": len(set(clusters.tolist())),
        "baseline_mean": float(np.mean(b)),
        "p0c_mean": float(np.mean(p)),
        "raw_difference_p0c_minus_baseline": float(np.mean(raw)),
        "improvement_oriented_difference": float(np.mean(improvement)),
        "sample_bootstrap95_low_improvement": slo,
        "sample_bootstrap95_high_improvement": shi,
        "cluster_bootstrap95_low_improvement": clo,
        "cluster_bootstrap95_high_improvement": chi,
        "sample_ci_supports_improvement": bool(np.isfinite(slo) and slo > 0),
        "cluster_ci_computed": bool(np.isfinite(clo)),
        "cluster_ci_supports_improvement": cluster_supported,
        "improvement_fraction": float(np.mean(improvement > 0)),
        "tie_fraction": float(np.mean(improvement == 0)),
    }


def self_test() -> None:
    vals = np.asarray([1.0, 2.0, 3.0])
    lo, hi = sample_bootstrap_ci(vals, 1, 100)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise AggregateError("sample bootstrap self-test failed")
    lo, hi = cluster_bootstrap_ci(vals, ["a", "b", "b"], 1, 100)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise AggregateError("cluster bootstrap self-test failed")
    test_rows = [
        {
            "sample_id": "a", "leakage_group_id": "g1",
            "b0_self_score1b_proxy20": 0.2, "p0c_self_score1b_proxy20": 0.8,
        },
        {
            "sample_id": "b", "leakage_group_id": "g2",
            "b0_self_score1b_proxy20": 0.3, "p0c_self_score1b_proxy20": 0.7,
        },
    ]
    rec = summarize(test_rows, "b0", "x", "overall", "overall", "self_score1b_proxy20", 1, 100, 100)
    if rec["improvement_oriented_difference"] <= 0:
        raise AggregateError("summary direction self-test failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--p0c0-dir", type=Path, required=True)
    ap.add_argument("--pilot-dir", type=Path, required=True)
    ap.add_argument("--pilot-source", type=Path, required=True)
    ap.add_argument("--worker-source", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--sample-bootstrap", type=int, default=2000)
    ap.add_argument("--cluster-bootstrap", type=int, default=1000)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        print("AGGREGATOR_SELF_TEST=PASS")
        return 0
    for name in ("run_dir", "p0c0_dir", "pilot_dir", "pilot_source", "worker_source"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    aggregate_dir = args.run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.p0c0_dir / "shared_generation_manifest.csv"
    protocol_path = args.p0c0_dir / "comparison_protocol_v1_0_0.json"
    p0c0_accept_path = args.p0c0_dir / "P0C0_ACCEPTANCE.json"
    pilot_accept_path = args.pilot_dir / "B0_P0C_M1_ACCEPTANCE.json"
    for path in (manifest_path, protocol_path, p0c0_accept_path, pilot_accept_path, args.pilot_source, args.worker_source):
        if not path.is_file():
            raise AggregateError(f"Missing required input: {path}")
    protocol = json_load(protocol_path)
    p0c0_accept = json_load(p0c0_accept_path)
    pilot_accept = json_load(pilot_accept_path)
    if protocol.get("status") != "FROZEN":
        raise AggregateError(f"Comparison protocol is not FROZEN: {protocol}")
    if p0c0_accept.get("status") != "P0C0_SHARED_PROTOCOL_FROZEN":
        raise AggregateError(f"P0C0 gate is not frozen: {p0c0_accept}")
    if pilot_accept.get("status") != "B0_P0C_M1_PILOT_ACCEPTED_FULL_5FOLD_ALLOWED" or not pilot_accept.get("full_5fold_allowed"):
        raise AggregateError(f"Pilot does not authorize full expansion: {pilot_accept}")
    if inum(pilot_accept.get("top_k")) != args.top_k:
        raise AggregateError(f"Top-K mismatch with pilot: {args.top_k} != {pilot_accept.get('top_k')}")

    manifest = read_csv(manifest_path)
    if len(manifest) != inum(p0c0_accept.get("sample_count"), len(manifest)):
        raise AggregateError("P0C0 sample count does not match shared manifest")
    manifest_by_id = {str(r["sample_id"]): r for r in manifest}
    if len(manifest_by_id) != len(manifest):
        raise AggregateError("Duplicate sample IDs in shared manifest")
    fold_values = sorted({inum(r.get("fold")) for r in manifest})
    if fold_values != list(EXPECTED_FOLDS):
        raise AggregateError(f"Unexpected fold IDs: {fold_values}")

    expected_by_fold = {
        fold: {str(r["sample_id"]) for r in manifest if inum(r.get("fold")) == fold}
        for fold in EXPECTED_FOLDS
    }
    all_per_sample: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_retrieval: list[dict[str, Any]] = []
    all_timing: list[dict[str, Any]] = []
    fold_audit_rows: list[dict[str, Any]] = []
    engineering_errors: list[dict[str, Any]] = []
    motion_missing: list[dict[str, Any]] = []

    for fold in EXPECTED_FOLDS:
        fold_dir = args.run_dir / "folds" / f"fold_{fold}"
        required = [
            "B0_P0C_M1_ACCEPTANCE.json",
            "validation_gt_usage_audit.json",
            "per_sample_comparison.csv",
            "candidate_metrics.csv",
            "retrieval_manifest.csv",
            "timing_manifest.csv",
            "schema_errors.csv",
            "run_file_manifest.json",
        ]
        missing_required = [name for name in required if not (fold_dir / name).is_file()]
        if missing_required:
            engineering_errors.append({"fold": fold, "error": "missing_required_files", "details": json.dumps(missing_required)})
            continue
        accept = json_load(fold_dir / "B0_P0C_M1_ACCEPTANCE.json")
        gt_audit = json_load(fold_dir / "validation_gt_usage_audit.json")
        per = read_csv(fold_dir / "per_sample_comparison.csv")
        cand = read_csv(fold_dir / "candidate_metrics.csv")
        retrieval = read_csv(fold_dir / "retrieval_manifest.csv")
        timing = read_csv(fold_dir / "timing_manifest.csv")
        schema_errors = read_csv(fold_dir / "schema_errors.csv")
        expected_ids = expected_by_fold[fold]
        actual_ids = [str(r.get("sample_id")) for r in per]
        actual_set = set(actual_ids)
        duplicate_ids = sorted([sid for sid, count in Counter(actual_ids).items() if count > 1])
        fold_errors: list[str] = []
        if accept.get("status") != "B0_P0C_M1_FULL_FOLD_COMPLETED" or not accept.get("engineering_gate"):
            fold_errors.append("fold_acceptance_not_complete")
        if gt_audit.get("status") != "PASS" or not gt_audit.get("all_generation_views_exclude_prohibited_keys"):
            fold_errors.append("validation_gt_usage_audit_failed")
        if schema_errors:
            fold_errors.append(f"schema_errors={len(schema_errors)}")
        if actual_set != expected_ids:
            fold_errors.append(f"coverage_mismatch_missing={len(expected_ids-actual_set)}_extra={len(actual_set-expected_ids)}")
        if duplicate_ids:
            fold_errors.append(f"duplicate_sample_ids={len(duplicate_ids)}")
        expected_candidates = len(expected_ids) * (1 + 2 * args.top_k)
        if len(cand) != expected_candidates:
            fold_errors.append(f"candidate_count={len(cand)}_expected={expected_candidates}")
        expected_retrieval = expected_candidates
        if len(retrieval) != expected_retrieval:
            fold_errors.append(f"retrieval_count={len(retrieval)}_expected={expected_retrieval}")
        # Every validation query must have one selected output for each method.
        for sid in expected_ids:
            for method in METHODS:
                path = fold_dir / "motions" / method_motion_dir(method) / f"{sid}.npz"
                if not path.is_file():
                    motion_missing.append({"fold": fold, "sample_id": sid, "method": method, "path": str(path)})
        for row in per:
            sid = str(row.get("sample_id"))
            source = manifest_by_id.get(sid)
            if source is None:
                fold_errors.append(f"unknown_sample_id={sid}")
                continue
            row["recording_id"] = source.get("recording_id", row.get("recording_id", ""))
            row["leakage_group_id"] = source.get("leakage_group_id", row.get("leakage_group_id", "")) or sid
            row["annotation_signature"] = source.get("annotation_signature", "")
        if fold_errors:
            engineering_errors.append({"fold": fold, "error": "fold_validation_failed", "details": ";".join(fold_errors)})
        fold_audit_rows.append({
            "fold": fold,
            "expected_queries": len(expected_ids),
            "actual_queries": len(per),
            "candidate_count": len(cand),
            "retrieval_count": len(retrieval),
            "schema_error_count": len(schema_errors),
            "gt_usage_audit": gt_audit.get("status"),
            "fold_engineering_gate": accept.get("engineering_gate"),
            "fold_directional_gate_diagnostic": accept.get("directional_gate"),
            "fold_quality_gate_diagnostic": accept.get("quality_gate"),
            "status": "PASS" if not fold_errors else "FAIL",
            "errors": ";".join(fold_errors),
        })
        all_per_sample.extend(per)
        all_candidates.extend(cand)
        all_retrieval.extend(retrieval)
        all_timing.extend(timing)

    if motion_missing:
        engineering_errors.append({"fold": "all", "error": "missing_selected_motion_files", "details": str(len(motion_missing))})
    write_csv(aggregate_dir / "fold_coverage.csv", fold_audit_rows)
    write_csv(aggregate_dir / "engineering_errors.csv", engineering_errors)
    write_csv(aggregate_dir / "missing_motion_files.csv", motion_missing)

    all_ids = [str(r.get("sample_id")) for r in all_per_sample]
    all_set = set(all_ids)
    expected_all = set(manifest_by_id)
    duplicate_all = sorted([sid for sid, count in Counter(all_ids).items() if count > 1])
    leakage_fold_map: dict[str, set[int]] = defaultdict(set)
    for row in all_per_sample:
        leakage_fold_map[str(row.get("leakage_group_id") or row.get("sample_id"))].add(inum(row.get("fold")))
    leakage_crossfold = {g: sorted(v) for g, v in leakage_fold_map.items() if len(v) > 1}
    coverage_audit = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "status": "PASS" if (
            not engineering_errors and not motion_missing and all_set == expected_all and not duplicate_all and not leakage_crossfold
        ) else "FAIL",
        "expected_sample_count": len(expected_all),
        "actual_row_count": len(all_per_sample),
        "actual_unique_sample_count": len(all_set),
        "missing_sample_count": len(expected_all - all_set),
        "extra_sample_count": len(all_set - expected_all),
        "duplicate_sample_count": len(duplicate_all),
        "cross_fold_leakage_group_count": len(leakage_crossfold),
        "folds": fold_audit_rows,
        "missing_samples": sorted(expected_all - all_set)[:100],
        "extra_samples": sorted(all_set - expected_all)[:100],
        "duplicate_samples": duplicate_all[:100],
        "cross_fold_leakage_groups": dict(list(leakage_crossfold.items())[:100]),
    }
    json_dump(args.run_dir / "coverage_audit.json", coverage_audit)

    # Preserve combined raw evidence even if a later scientific gate is negative.
    write_csv(aggregate_dir / "per_sample_comparison_all_folds.csv", all_per_sample)
    write_csv(aggregate_dir / "candidate_metrics_all_folds.csv", all_candidates)
    write_csv(aggregate_dir / "retrieval_manifest_all_folds.csv", all_retrieval)
    write_csv(aggregate_dir / "timing_manifest_all_folds.csv", all_timing)

    summary_rows: list[dict[str, Any]] = []
    if all_per_sample:
        for comparison, baseline in COMPARISONS:
            for group_name, group_value, records in group_records(all_per_sample):
                if not records:
                    continue
                for metric in METRICS:
                    summary_rows.append(summarize(
                        records, baseline, comparison, group_name, group_value, metric,
                        args.seed, args.sample_bootstrap, args.cluster_bootstrap,
                    ))
    write_csv(aggregate_dir / "summary_metrics.csv", summary_rows)

    def get_summary(comparison: str, group_name: str, group_value: str, metric: str) -> dict[str, Any]:
        for rec in summary_rows:
            if (
                rec["comparison"] == comparison and rec["group_name"] == group_name
                and rec["group_value"] == group_value and rec["metric"] == metric
            ):
                return rec
        raise AggregateError(f"Missing summary: {comparison}/{group_name}/{group_value}/{metric}")

    fold_primary_rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        row: dict[str, Any] = {"fold": fold}
        for comparison, _baseline in COMPARISONS:
            prefix = "primary" if comparison == "p0c_vs_b0_primary" else "equal_budget"
            for metric in PRIMARY_METRICS:
                rec = get_summary(comparison, "fold", str(fold), metric)
                short = "score1b" if metric == "self_score1b_proxy20" else "stg"
                row[f"{prefix}_{short}_baseline_mean"] = rec["baseline_mean"]
                row[f"{prefix}_{short}_p0c_mean"] = rec["p0c_mean"]
                row[f"{prefix}_{short}_improvement"] = rec["improvement_oriented_difference"]
                row[f"{prefix}_{short}_positive"] = rec["improvement_oriented_difference"] > 0
        fold_primary_rows.append(row)
    write_csv(aggregate_dir / "fold_primary_results.csv", fold_primary_rows)

    primary_score = get_summary("p0c_vs_b0_primary", "overall", "overall", "self_score1b_proxy20")
    primary_stg = get_summary("p0c_vs_b0_primary", "overall", "overall", "self_stg_proxy_sigma5")
    equal_score = get_summary("p0c_vs_b0_equal_budget", "overall", "overall", "self_score1b_proxy20")
    equal_stg = get_summary("p0c_vs_b0_equal_budget", "overall", "overall", "self_stg_proxy_sigma5")
    positive_primary_score_folds = sum(bool(r["primary_score1b_positive"]) for r in fold_primary_rows)
    positive_primary_stg_folds = sum(bool(r["primary_stg_positive"]) for r in fold_primary_rows)
    mean_direction_gate = (
        fnum(primary_score.get("improvement_oriented_difference"), -1) > 0
        and fnum(primary_stg.get("improvement_oriented_difference"), -1) > 0
    )
    fold_consistency_gate = positive_primary_score_folds >= 4 and positive_primary_stg_folds >= 4

    quality_components: dict[str, Any] = {}
    for metric, limit_spec in QUALITY_LIMITS.items():
        rec = get_summary("p0c_vs_b0_primary", "overall", "overall", metric)
        baseline_mean = fnum(rec.get("baseline_mean"), 0.0)
        p0c_mean = fnum(rec.get("p0c_mean"), 1e9)
        allowed_increase = max(
            baseline_mean * float(limit_spec["relative_increase"]),
            float(limit_spec["absolute_increase"]),
        )
        allowed_max = baseline_mean + allowed_increase
        quality_components[metric] = {
            "baseline_mean": baseline_mean,
            "p0c_mean": p0c_mean,
            "relative_increase_limit": limit_spec["relative_increase"],
            "absolute_increase_limit": limit_spec["absolute_increase"],
            "allowed_max": allowed_max,
            "pass": bool(p0c_mean <= allowed_max),
        }
    quality_gate = all(bool(x["pass"]) for x in quality_components.values())
    engineering_gate = coverage_audit["status"] == "PASS"
    internal_surpassed = engineering_gate and mean_direction_gate and fold_consistency_gate and quality_gate
    if not engineering_gate:
        status = "B0_P0C_M1_FULL_5FOLD_ENGINEERING_FAILED"
    elif internal_surpassed:
        status = "B0_P0C_M1_FULL_5FOLD_ACCEPTED"
    else:
        status = "B0_P0C_M1_FULL_5FOLD_COMPLETED_NEEDS_METHOD_REFINEMENT"

    statistical_support = {
        "primary_self_score1b_sample_ci_positive": bool(primary_score["sample_ci_supports_improvement"]),
        "primary_self_score1b_cluster_ci_positive": bool(primary_score["cluster_ci_supports_improvement"]),
        "primary_self_stg_sample_ci_positive": bool(primary_stg["sample_ci_supports_improvement"]),
        "primary_self_stg_cluster_ci_positive": bool(primary_stg["cluster_ci_supports_improvement"]),
        "equal_budget_self_score1b_sample_ci_positive": bool(equal_score["sample_ci_supports_improvement"]),
        "equal_budget_self_score1b_cluster_ci_positive": bool(equal_score["cluster_ci_supports_improvement"]),
        "equal_budget_self_stg_sample_ci_positive": bool(equal_stg["sample_ci_supports_improvement"]),
        "equal_budget_self_stg_cluster_ci_positive": bool(equal_stg["cluster_ci_supports_improvement"]),
        "note": "Statistical-support fields are reported separately and do not silently replace the pre-frozen four-part scientific gate.",
    }

    failure_cases: list[dict[str, Any]] = []
    for row in all_per_sample:
        primary_delta = fnum(row.get("p0c_self_score1b_proxy20")) - fnum(row.get("b0_self_score1b_proxy20"))
        equal_delta = fnum(row.get("p0c_self_score1b_proxy20")) - fnum(row.get("b0eq_self_score1b_proxy20"))
        stg_delta = fnum(row.get("p0c_self_stg_proxy_sigma5")) - fnum(row.get("b0_self_stg_proxy_sigma5"))
        if primary_delta < 0 or equal_delta < 0 or stg_delta < 0:
            failure_cases.append({
                "sample_id": row.get("sample_id"),
                "recording_id": row.get("recording_id"),
                "leakage_group_id": row.get("leakage_group_id"),
                "fold": row.get("fold"),
                "domain": row.get("domain"),
                "hand": row.get("hand"),
                "primary_self_score1b_delta": primary_delta,
                "equal_budget_self_score1b_delta": equal_delta,
                "primary_self_stg_delta": stg_delta,
                "p0c_self_angle_deg": row.get("p0c_self_angle_deg"),
                "b0_self_angle_deg": row.get("b0_self_angle_deg"),
                "b0eq_self_angle_deg": row.get("b0eq_self_angle_deg"),
                "p0c_detector_agreement": row.get("p0c_detector_agreement"),
                "p0c_donor_id": row.get("p0c_donor_id"),
                "b0_donor_id": row.get("b0_donor_id"),
            })
    failure_cases.sort(key=lambda r: (
        fnum(r.get("primary_self_score1b_delta"), 0),
        fnum(r.get("primary_self_stg_delta"), 0),
        str(r.get("sample_id")),
    ))
    write_csv(aggregate_dir / "failure_cases.csv", failure_cases)

    acceptance = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_utc": now_utc(),
        "status": status,
        "sample_count": len(all_per_sample),
        "candidate_count": len(all_candidates),
        "fold_count": len(EXPECTED_FOLDS),
        "top_k": args.top_k,
        "engineering_gate": engineering_gate,
        "mean_direction_gate": mean_direction_gate,
        "fold_consistency_gate": fold_consistency_gate,
        "quality_gate": quality_gate,
        "quality_components": quality_components,
        "positive_primary_score1b_folds": positive_primary_score_folds,
        "positive_primary_stg_folds": positive_primary_stg_folds,
        "required_positive_folds": 4,
        "primary_overall": {
            "self_score1b_proxy20": primary_score,
            "self_stg_proxy_sigma5": primary_stg,
        },
        "equal_budget_diagnostic": {
            "self_score1b_proxy20": equal_score,
            "self_stg_proxy_sigma5": equal_stg,
            "gate_role": "diagnostic_not_primary",
        },
        "statistical_support": statistical_support,
        "internal_omnicontrol_pt_style_baseline_surpassed": internal_surpassed,
        "official_baseline_surpassed": False,
        "official_server_required": True,
        "private_official_scorer_claimed": False,
        "b0_claim": "OmniControl-PT-style internal baseline; not an exact reproduction of unavailable official code/weights.",
        "p0c_m1_components": ["geometry-aware natural-motion retrieval", "apex-preserving time warp", "shape-aware local IK", "candidate reranking"],
        "no_scene_graph_or_flow_yet": True,
        "p1_allowed": bool(internal_surpassed),
        "next_action": (
            "Freeze P0C-M1 and start P1 learned speech-phase timing under the same folds and backend."
            if internal_surpassed
            else "Inspect fold and failure-case diagnostics; change only the generation method before repeating the full protocol."
        ),
        "guardrails": [
            "These are local paper-faithful proxy results, not official server scores.",
            "The full five-fold result compares against an internal OmniControl-PT-style baseline, not the unavailable official implementation.",
            "No scene graph, residual flow, or listener state is used in P0C-M1.",
            "Validation released peak/hold/alignment labels were stripped before generation and retained only for post-generation diagnostics.",
        ],
    }
    json_dump(args.run_dir / "FULL_5FOLD_ACCEPTANCE.json", acceptance)

    source_binding_paths = [
        manifest_path, protocol_path, p0c0_accept_path, pilot_accept_path,
        args.pilot_source, args.worker_source,
    ]
    source_bindings = [
        {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in source_binding_paths
    ]
    for fold in EXPECTED_FOLDS:
        fold_dir = args.run_dir / "folds" / f"fold_{fold}"
        for name in ("B0_P0C_M1_ACCEPTANCE.json", "validation_gt_usage_audit.json", "per_sample_comparison.csv", "candidate_metrics.csv"):
            p = fold_dir / name
            if p.is_file():
                source_bindings.append({"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    json_dump(args.run_dir / "source_binding.json", source_bindings)

    env = {
        "created_utc": now_utc(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
    }
    json_dump(args.run_dir / "environment.json", env)

    summary = [
        "REPOINTER B0 / P0C-M1 — FULL FIVE-FOLD OOF COMPARISON",
        "=" * 78,
        f"version={VERSION}",
        f"created_utc={now_utc()}",
        f"run_dir={args.run_dir}",
        f"p0c0_dir={args.p0c0_dir}",
        f"pilot_dir={args.pilot_dir}",
        "dataset_writes=0",
        "downloads=0",
        "git_writes=0",
        "official_test_inference=0",
        "",
        "[COVERAGE]",
        f"coverage_status={coverage_audit['status']}",
        f"samples={len(all_per_sample)}/{len(expected_all)}",
        f"candidates={len(all_candidates)}",
        f"folds={len(EXPECTED_FOLDS)}",
        f"cross_fold_leakage_groups={coverage_audit['cross_fold_leakage_group_count']}",
        "validation_gt_usage_audit=PASS_ALL_FOLDS" if all(r.get("gt_usage_audit") == "PASS" for r in fold_audit_rows) else "validation_gt_usage_audit=FAIL",
        "",
        "[PRIMARY B0 COMPARISON]",
        f"self_score1b_b0={primary_score['baseline_mean']}",
        f"self_score1b_p0c={primary_score['p0c_mean']}",
        f"self_score1b_improvement={primary_score['improvement_oriented_difference']}",
        f"self_score1b_sample_ci95=[{primary_score['sample_bootstrap95_low_improvement']},{primary_score['sample_bootstrap95_high_improvement']}]",
        f"self_score1b_cluster_ci95=[{primary_score['cluster_bootstrap95_low_improvement']},{primary_score['cluster_bootstrap95_high_improvement']}]",
        f"self_stg_b0={primary_stg['baseline_mean']}",
        f"self_stg_p0c={primary_stg['p0c_mean']}",
        f"self_stg_improvement={primary_stg['improvement_oriented_difference']}",
        f"self_stg_sample_ci95=[{primary_stg['sample_bootstrap95_low_improvement']},{primary_stg['sample_bootstrap95_high_improvement']}]",
        f"self_stg_cluster_ci95=[{primary_stg['cluster_bootstrap95_low_improvement']},{primary_stg['cluster_bootstrap95_high_improvement']}]",
        f"positive_score1b_folds={positive_primary_score_folds}/5",
        f"positive_stg_folds={positive_primary_stg_folds}/5",
        "",
        "[EQUAL-BUDGET DIAGNOSTIC]",
        f"self_score1b_b0eq={equal_score['baseline_mean']}",
        f"self_score1b_p0c={equal_score['p0c_mean']}",
        f"self_score1b_improvement={equal_score['improvement_oriented_difference']}",
        f"self_stg_b0eq={equal_stg['baseline_mean']}",
        f"self_stg_p0c={equal_stg['p0c_mean']}",
        f"self_stg_improvement={equal_stg['improvement_oriented_difference']}",
        "",
        "[GATES]",
        f"engineering_gate={engineering_gate}",
        f"mean_direction_gate={mean_direction_gate}",
        f"fold_consistency_gate={fold_consistency_gate}",
        f"quality_gate={quality_gate}",
        "",
        "[DECISION]",
        f"status={status}",
        f"internal_omnicontrol_pt_style_baseline_surpassed={internal_surpassed}",
        "official_baseline_surpassed=FALSE",
        "official_server_required=TRUE",
        f"p1_allowed={str(bool(internal_surpassed)).upper()}",
        f"next_action={acceptance['next_action']}",
        "",
        "[PRIMARY ARTIFACTS]",
        "FULL_5FOLD_ACCEPTANCE.json",
        "coverage_audit.json",
        "aggregate/summary_metrics.csv",
        "aggregate/fold_primary_results.csv",
        "aggregate/failure_cases.csv",
        "aggregate/per_sample_comparison_all_folds.csv",
        "aggregate/candidate_metrics_all_folds.csv",
    ]
    (args.run_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    run_manifest: list[dict[str, Any]] = []
    for p in sorted(args.run_dir.rglob("*")):
        if p.is_file() and p.name != "run_file_manifest.json":
            run_manifest.append({
                "relative_path": str(p.relative_to(args.run_dir)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    json_dump(args.run_dir / "run_file_manifest.json", run_manifest)
    print("\n".join(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise

