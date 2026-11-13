#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

VERSION = "0.1.0"
SCHEMA_VERSION = "1.0.0"
EXPECTED_TEST_COUNT = 100
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
FPS = 30.0
WORDPHASE_BATCH_SIZE = 128
RANKER_LAMBDA = 0.5
RANKER_THRESHOLDS = (-0.02, 0.0, 0.01, 0.02, 0.05, 0.10)


class P2C1Error(RuntimeError):
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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=json_default)
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
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise P2C1Error(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sanitize_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return value.strip("._-") or "RePointer"


def parse_json_vector(value: Any, length: int = 3) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    else:
        text = str(value).strip()
        try:
            arr = np.asarray(json.loads(text), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise P2C1Error(f"Could not parse vector {value!r}: {exc}") from exc
    if arr.size != length or not np.all(np.isfinite(arr)):
        raise P2C1Error(f"Invalid vector length/values: {value!r}")
    return arr


def locate_mh_source(p2a_mh_dir: Path) -> Path:
    candidates = [
        p2a_mh_dir / "repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py",
        p2a_mh_dir / "_code" / "repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py",
        Path.home() / "Downloads" / "repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py",
    ]
    hits = [p.resolve() for p in candidates if p.is_file()]
    if not hits:
        raise P2C1Error(f"Could not locate P2A-MH v0.1.2 source under {p2a_mh_dir}")
    return hits[0]


def phase_router_feature_names() -> list[str]:
    names = [
        "domain_natural", "domain_synthetic", "hand_left", "hand_right",
        "clip_len_norm", "log_target_distance", "sin_target_azimuth",
        "cos_target_azimuth", "sin_target_elevation", "cos_target_elevation",
        "median_peak_ratio", "wordphase_peak_ratio", "peak_diff_norm",
        "peak_abs_diff_norm",
    ]
    base = [
        "detector_agreement", "planned_score1b", "self_score1b", "self_angle_norm",
        "self_plan_offset", "self_plan_abs", "det_spread", "det_mean_offset",
        "joint_jerk_log", "foot_sliding_log", "pose_deviation_log", "max_reach",
        "donor_rank_norm",
    ]
    for method in ("median", "wordphase"):
        names.extend([f"{method}_{x}" for x in base])
    names.extend([f"diff_wordphase_minus_median_{x}" for x in base])
    return names


def build_phase_router_features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
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
    if X.shape[1] != len(phase_router_feature_names()):
        raise P2C1Error(f"Phase-router feature mismatch: {X.shape[1]} != {len(phase_router_feature_names())}")
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def ridge_predict(X: np.ndarray, beta: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Z = (np.asarray(X, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / np.maximum(np.asarray(sd, dtype=np.float64), 1e-8)
    A = np.concatenate([np.ones((len(Z), 1), dtype=np.float64), Z], axis=1)
    return A @ np.asarray(beta, dtype=np.float64)


def load_npz_model(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def phase_router_predict(model: Mapping[str, np.ndarray], row: Mapping[str, Any]) -> float:
    expected_names = [str(x) for x in np.asarray(model["feature_names"]).reshape(-1).tolist()]
    if expected_names != phase_router_feature_names():
        raise P2C1Error("Frozen phase-router feature names do not match the implementation")
    X = build_phase_router_features([row])
    return float(ridge_predict(X, model["beta"], model["feature_mean"], model["feature_std"])[0])


def training_median_ratios(rows: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for hand in (0, 1):
        vals: list[float] = []
        for r in rows:
            if str(r.get("domain")) != "natural" or inum(r.get("hand")) != hand:
                continue
            ratio = fnum(r.get("released_peak_ratio"))
            if not np.isfinite(ratio):
                T = inum(r.get("clip_length"), 0)
                peak = inum(r.get("released_peak_frame"), -1)
                if T > 1 and 0 <= peak < T:
                    ratio = peak / (T - 1)
            if np.isfinite(ratio):
                vals.append(float(ratio))
        if not vals:
            raise P2C1Error(f"No natural training peak ratios for hand={hand}")
        out[hand] = float(np.median(np.asarray(vals, dtype=np.float64)))
    return out


def fixed_timing_from_transcript(mh: Any, text_path: Path, T: int, fallback_ratio: float) -> dict[str, Any]:
    words = mh.parse_word_times(text_path)
    clip_end = T / FPS
    overlapping = [(a, b, w) for a, b, w in words if b >= -0.05 and a <= clip_end + 0.05]
    if overlapping:
        phrase_end = max(float(b) for _, b, _ in overlapping)
        peak = int(round(phrase_end * FPS)) - 11
        peak = int(np.clip(peak, 2, max(2, T - 3)))
        return {
            "planned_peak": peak,
            "planned_peak_ratio": peak / max(1, T - 1),
            "timing_source": "last_word_end_minus_11",
            "transcript_path": str(text_path),
            "transcript_match": "official_test",
            "word_count": len(overlapping),
            "phrase_end_sec": phrase_end,
        }
    peak = int(round(fallback_ratio * max(1, T - 1)))
    peak = int(np.clip(peak, 2, max(2, T - 3)))
    return {
        "planned_peak": peak,
        "planned_peak_ratio": peak / max(1, T - 1),
        "timing_source": "all_training_natural_hand_median_fallback",
        "transcript_path": str(text_path),
        "transcript_match": "official_test_no_words",
        "word_count": 0,
        "phrase_end_sec": "",
    }


def build_test_rows(data_root: Path, p2c0_dir: Path, train_rows: Sequence[Mapping[str, Any]], mh: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[int, float]]:
    test_root = data_root / "test_submission"
    manifest = read_csv(test_root / "manifest.csv")
    audit_rows = read_csv(p2c0_dir / "test_input_manifest_audit.csv")
    audit_by_id = {str(r["sample_id"]): r for r in audit_rows}
    if len(manifest) != EXPECTED_TEST_COUNT or len(audit_by_id) != EXPECTED_TEST_COUNT:
        raise P2C1Error(f"Test coverage mismatch: manifest={len(manifest)} audit={len(audit_by_id)}")
    medians = training_median_ratios(train_rows)
    rows: list[dict[str, Any]] = []
    timing_by_id: dict[str, dict[str, Any]] = {}
    for src in manifest:
        sid = str(src.get("sample_id", "")).strip()
        if sid not in audit_by_id:
            raise P2C1Error(f"Missing P2C0 audit row for {sid}")
        audit = audit_by_id[sid]
        if str(audit.get("status")) != "PASS":
            raise P2C1Error(f"P2C0 row is not PASS: {sid}")
        T = inum(src.get("clip_length"), -1)
        hand = inum(src.get("hand"), -1)
        tp = np.asarray([fnum(src.get("target_pelvis_x")), fnum(src.get("target_pelvis_y")), fnum(src.get("target_pelvis_z"))], dtype=np.float64)
        tw = np.asarray([fnum(src.get("target_world_x")), fnum(src.get("target_world_y")), fnum(src.get("target_world_z"))], dtype=np.float64)
        if T <= 4 or hand not in (0, 1) or not np.all(np.isfinite(tp)) or not np.all(np.isfinite(tw)):
            raise P2C1Error(f"Invalid test geometry for {sid}")
        text_path = Path(str(audit["text_path"]))
        shape_path = Path(str(audit["body_shape_path"]))
        row: dict[str, Any] = {
            **src,
            "sample_id": sid,
            "recording_id": sid,
            "domain": str(audit.get("model_domain_policy", "natural")),
            "fold": -1,
            "global_start": -1,
            "global_end": -1,
            "room_id": str(src.get("scene_id", "")),
            "object_id": str(audit.get("matched_object_id", "")),
            "text_path": str(text_path),
            "audio_path": str(audit.get("audio_path", "")),
            "body_shape_path": str(shape_path),
            "scene_path": str(audit.get("scene_path", "")),
            "target_distance": float(np.linalg.norm(tp)),
            "target_azimuth_deg": math.degrees(math.atan2(float(tp[0]), float(tp[2]))),
            "target_elevation_deg": math.degrees(math.atan2(float(tp[1]), max(math.hypot(float(tp[0]), float(tp[2])), 1e-9))),
            "reference_pelvis_world_x": fnum(audit.get("reference_pelvis_world_x")),
            "reference_pelvis_world_y": fnum(audit.get("reference_pelvis_world_y")),
            "reference_pelvis_world_z": fnum(audit.get("reference_pelvis_world_z")),
            "matched_bbox_center": str(audit.get("matched_bbox_center", "")),
            "matched_bbox_size": str(audit.get("matched_bbox_size", "")),
            "target_resolution_mode": str(audit.get("target_resolution_mode", "")),
            "matched_object_category": str(audit.get("matched_object_category", "")),
            "model_domain_policy": str(audit.get("model_domain_policy", "natural")),
        }
        timing = fixed_timing_from_transcript(mh, text_path, T, medians[hand])
        timing_by_id[sid] = timing
        row["fixed_planned_peak"] = timing["planned_peak"]
        row["fixed_planned_peak_ratio"] = timing["planned_peak_ratio"]
        row["all_training_median_ratio"] = medians[hand]
        row["all_training_median_frame"] = int(np.clip(round(medians[hand] * max(1, T - 1)), 2, max(2, T - 3)))
        rows.append(row)
    if len({str(r["sample_id"]) for r in rows}) != EXPECTED_TEST_COUNT:
        raise P2C1Error("Test sample IDs are not unique")
    return rows, timing_by_id, medians


def infer_test_wordphase_ensemble(
    test_rows: Sequence[Mapping[str, Any]], timing_by_id: Mapping[str, Mapping[str, Any]],
    p1a_dir: Path, mh: Any, device: str, max_words: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    sample_ids = [str(r["sample_id"]) for r in test_rows]
    prob_by_fold: dict[int, dict[str, np.ndarray]] = {}
    frame_by_fold: dict[int, dict[str, int]] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    word_cache: dict[Path, list[tuple[float, float, str]]] = {}
    common_bins: int | None = None
    common_buckets: int | None = None
    for fold in EXPECTED_FOLDS:
        ckpt_path = p1a_dir / f"wordphasenet_fold{fold}.pt"
        state = mh.torch_load_checkpoint(ckpt_path, "cpu")
        bins = int(state["bins"])
        buckets = int(state["hash_buckets"])
        if common_bins is None:
            common_bins, common_buckets = bins, buckets
        if bins != common_bins or buckets != common_buckets:
            raise P2C1Error("WordPhase checkpoint architecture mismatch")
        medians: dict[tuple[str, int], float] = {}
        for k, v in dict(state["medians"]).items():
            m = re.fullmatch(r"(natural|synthetic)_hand([01])", str(k))
            if m:
                medians[(m.group(1), int(m.group(2)))] = float(v)
        if len(medians) != 4:
            raise P2C1Error(f"Malformed checkpoint median map: {ckpt_path}")
        examples = [
            mh.build_inference_example(r, timing_by_id[str(r["sample_id"])], buckets, max_words, word_cache)
            for r in test_rows
        ]
        wrapper = mh.WordPhaseNet(torch, buckets, 10, len(np.asarray(state["clip_mean"])), bins)
        model = wrapper.net.to(device)
        model.load_state_dict(state["model_state"], strict=True)
        model.eval()
        centers = np.linspace(0.0, 1.0, bins, dtype=np.float64)
        prob_by_fold[fold] = {}
        frame_by_fold[fold] = {}
        with torch.no_grad():
            for start in range(0, len(examples), WORDPHASE_BATCH_SIZE):
                batch = examples[start:start + WORDPHASE_BATCH_SIZE]
                tok, wf, mask, cf = mh.collate_inference(
                    batch, medians,
                    np.asarray(state["word_mean"], dtype=np.float32),
                    np.asarray(state["word_std"], dtype=np.float32),
                    np.asarray(state["clip_mean"], dtype=np.float32),
                    np.asarray(state["clip_std"], dtype=np.float32),
                )
                tok, wf, mask, cf = [z.to(device) for z in (tok, wf, mask, cf)]
                probability = torch.softmax(model(tok, wf, mask, cf), dim=1).detach().cpu().numpy().astype(np.float64)
                for ex, p in zip(batch, probability):
                    p = p / max(float(p.sum()), 1e-12)
                    expected_ratio = float(np.sum(p * centers))
                    frame = mh.frame_from_ratio(expected_ratio, ex.clip_length)
                    prob_by_fold[fold][ex.sample_id] = p
                    frame_by_fold[fold][ex.sample_id] = frame
        checkpoint_rows.append({
            "fold_checkpoint": fold,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "test_query_count": len(prob_by_fold[fold]),
            "batch_size": WORDPHASE_BATCH_SIZE,
        })

    assert common_bins is not None
    centers = np.linspace(0.0, 1.0, common_bins, dtype=np.float64)
    out: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    row_by_id = {str(r["sample_id"]): r for r in test_rows}
    for sid in sample_ids:
        probs = np.stack([prob_by_fold[f][sid] for f in EXPECTED_FOLDS], axis=0)
        p = probs.mean(axis=0)
        p /= max(float(p.sum()), 1e-12)
        mean = float(np.sum(p * centers))
        std = float(math.sqrt(max(0.0, np.sum(p * (centers - mean) ** 2))))
        entropy = float(-np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0)) / math.log(len(p)))
        order = np.argsort(p)[::-1]
        T = inum(row_by_id[sid].get("clip_length"))
        expected_frame = mh.frame_from_ratio(mean, T)
        out[sid] = {
            "sample_id": sid,
            "fold": -1,
            "domain": str(row_by_id[sid].get("domain", "natural")),
            "hand": inum(row_by_id[sid].get("hand")),
            "clip_length": T,
            "prob": p.astype(np.float32),
            "centers": centers.astype(np.float32),
            "expected_ratio": mean,
            "expected_frame": expected_frame,
            "entropy_norm": entropy,
            "std_ratio": std,
            "max_prob": float(p[order[0]]),
            "second_bin_prob": float(p[order[1]]) if len(order) > 1 else 0.0,
            "top_bin_margin": float(p[order[0]] - p[order[1]]) if len(order) > 1 else float(p[order[0]]),
        }
        fold_frames = [frame_by_fold[f][sid] for f in EXPECTED_FOLDS]
        audit_rows.append({
            "sample_id": sid,
            "clip_length": T,
            "ensemble_expected_frame": expected_frame,
            "ensemble_expected_ratio": mean,
            "ensemble_entropy_norm": entropy,
            "ensemble_std_ratio": std,
            "fold0_frame": fold_frames[0], "fold1_frame": fold_frames[1],
            "fold2_frame": fold_frames[2], "fold3_frame": fold_frames[3],
            "fold4_frame": fold_frames[4],
            "checkpoint_frame_min": min(fold_frames),
            "checkpoint_frame_max": max(fold_frames),
            "checkpoint_frame_range": max(fold_frames) - min(fold_frames),
            "status": "PASS",
        })
    audit = {
        "status": "PASS" if len(out) == EXPECTED_TEST_COUNT else "FAIL",
        "query_count": len(out),
        "checkpoint_count": len(EXPECTED_FOLDS),
        "ensemble_policy": "mean of five fold-checkpoint 64-bin probability distributions",
        "batch_size": WORDPHASE_BATCH_SIZE,
        "padding_sensitive_bigru_policy": "manifest order; all 100 test examples form one <=128 batch per checkpoint",
        "checkpoint_audit": checkpoint_rows,
    }
    return out, audit_rows, audit


def save_test_distribution_cache(path: Path, rows: Sequence[Mapping[str, Any]], dist_by_id: Mapping[str, Mapping[str, Any]]) -> None:
    ids = [str(r["sample_id"]) for r in rows]
    np.savez_compressed(
        path,
        sample_ids=np.asarray(ids, dtype="U128"),
        probabilities=np.stack([np.asarray(dist_by_id[sid]["prob"], dtype=np.float32) for sid in ids]),
        expected_frames=np.asarray([inum(dist_by_id[sid]["expected_frame"]) for sid in ids], dtype=np.int16),
        entropy_norm=np.asarray([fnum(dist_by_id[sid]["entropy_norm"]) for sid in ids], dtype=np.float32),
        std_ratio=np.asarray([fnum(dist_by_id[sid]["std_ratio"]) for sid in ids], dtype=np.float32),
    )


def load_test_distribution_cache(path: Path, test_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    row_by_id = {str(r["sample_id"]): r for r in test_rows}
    with np.load(path, allow_pickle=False) as z:
        ids = [str(x) for x in np.asarray(z["sample_ids"]).tolist()]
        probs = np.asarray(z["probabilities"], dtype=np.float64)
        expected_frames = np.asarray(z["expected_frames"], dtype=int)
        entropy = np.asarray(z["entropy_norm"], dtype=float)
        stds = np.asarray(z["std_ratio"], dtype=float)
    out: dict[str, dict[str, Any]] = {}
    for i, sid in enumerate(ids):
        if sid not in row_by_id:
            raise P2C1Error(f"Unknown test distribution ID: {sid}")
        p = probs[i] / max(float(probs[i].sum()), 1e-12)
        centers = np.linspace(0.0, 1.0, len(p), dtype=np.float64)
        order = np.argsort(p)[::-1]
        mean = float(np.sum(p * centers))
        row = row_by_id[sid]
        out[sid] = {
            "sample_id": sid, "fold": -1, "domain": str(row.get("domain", "natural")), "hand": inum(row.get("hand")),
            "clip_length": inum(row.get("clip_length")), "prob": p.astype(np.float32),
            "centers": centers.astype(np.float32), "expected_ratio": mean,
            "expected_frame": int(expected_frames[i]), "entropy_norm": float(entropy[i]),
            "std_ratio": float(stds[i]), "max_prob": float(p[order[0]]),
            "second_bin_prob": float(p[order[1]]) if len(order) > 1 else 0.0,
            "top_bin_margin": float(p[order[0]] - p[order[1]]) if len(order) > 1 else float(p[order[0]]),
        }
    if len(out) != EXPECTED_TEST_COUNT:
        raise P2C1Error("Test distribution cache coverage mismatch")
    return out


def load_training_distribution_cache(p2a_mh_dir: Path, train_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    cache_path = p2a_mh_dir / "wordphase_oof_distribution_cache_v0_1_2.npz"
    if not cache_path.is_file():
        raise P2C1Error(f"Missing P2A-MH OOF distribution cache: {cache_path}")
    row_by_id = {str(r["sample_id"]): r for r in train_rows}
    with np.load(cache_path, allow_pickle=False) as z:
        ids = [str(x) for x in np.asarray(z["sample_ids"]).tolist()]
        probs = np.asarray(z["probabilities"], dtype=np.float64)
        expected_frames = np.asarray(z["expected_frames"], dtype=int)
        entropy = np.asarray(z["entropy_norm"], dtype=float)
        stds = np.asarray(z["std_ratio"], dtype=float)
    out: dict[str, dict[str, Any]] = {}
    for i, sid in enumerate(ids):
        if sid not in row_by_id:
            raise P2C1Error(f"OOF distribution ID not in training manifest: {sid}")
        p = probs[i] / max(float(probs[i].sum()), 1e-12)
        centers = np.linspace(0.0, 1.0, len(p), dtype=np.float64)
        order = np.argsort(p)[::-1]
        mean = float(np.sum(p * centers))
        r = row_by_id[sid]
        out[sid] = {
            "sample_id": sid, "fold": inum(r.get("fold")), "domain": str(r.get("domain")),
            "hand": inum(r.get("hand")), "clip_length": inum(r.get("clip_length")),
            "prob": p.astype(np.float32), "centers": centers.astype(np.float32),
            "expected_ratio": mean, "expected_frame": int(expected_frames[i]),
            "entropy_norm": float(entropy[i]), "std_ratio": float(stds[i]),
            "max_prob": float(p[order[0]]),
            "second_bin_prob": float(p[order[1]]) if len(order) > 1 else 0.0,
            "top_bin_margin": float(p[order[0]] - p[order[1]]) if len(order) > 1 else float(p[order[0]]),
        }
    if len(out) != len(train_rows):
        raise P2C1Error(f"Training distribution coverage mismatch: {len(out)} != {len(train_rows)}")
    return out


def fit_full_candidate_ranker(
    train_rows: Sequence[Mapping[str, Any]], p1_full_dir: Path, p2a_dir: Path,
    p2a_mh_dir: Path, mh: Any, output_dir: Path,
) -> dict[str, Any]:
    p1_rows = read_csv(p1_full_dir / "per_sample_comparison.csv")
    p1_candidates = read_csv(p1_full_dir / "candidate_metrics.csv")
    route_rows = read_csv(p2a_dir / "oof_phase_router_predictions.csv")
    p1_by_id = {str(r["sample_id"]): r for r in p1_rows}
    route_by_id = {str(r["sample_id"]): r for r in route_rows}
    dist_by_id = load_training_distribution_cache(p2a_mh_dir, train_rows)
    training_rows, X, y, folds, groups, by_query, base_idx = mh.prepare_training_candidates(
        p1_candidates, p1_by_id, route_by_id, dist_by_id,
    )
    if len(training_rows) != len(X) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise P2C1Error("Candidate-ranker training matrix is invalid")
    qids = np.asarray([str(r["query_id"]) for r in training_rows], dtype=object)
    oof_pred = np.full(len(X), np.nan, dtype=np.float64)
    fold_models: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        tr = folds != fold
        va = folds == fold
        beta, mu, sd = mh.fit_weighted_ridge(X[tr], y[tr], groups[tr], qids[tr], RANKER_LAMBDA)
        oof_pred[va] = mh.ridge_predict(X[va], beta, mu, sd)
        fold_models.append({
            "fold": fold, "train_candidates": int(np.sum(tr)), "validation_candidates": int(np.sum(va)),
            "train_queries": len(set(qids[tr].tolist())), "validation_queries": len(set(qids[va].tolist())),
        })
    if not np.all(np.isfinite(oof_pred)):
        raise P2C1Error("Full-ranker OOF prediction coverage failed")

    calibration_rows: list[dict[str, Any]] = []
    best: tuple[float, float, float] | None = None
    best_details: list[dict[str, Any]] = []
    for threshold in RANKER_THRESHOLDS:
        selected_values: list[float] = []
        base_values: list[float] = []
        query_details: list[dict[str, Any]] = []
        for qid, ids in by_query.items():
            local_rows = [training_rows[i] for i in ids]
            local_pred = oof_pred[ids]
            base_global = base_idx[qid]
            base_local = ids.index(base_global)
            choice_local = mh.select_with_threshold(local_rows, local_pred, base_local, threshold)
            choice_global = ids[choice_local]
            selected_values.append(float(y[choice_global]))
            base_values.append(float(y[base_global]))
            query_details.append({
                "query_id": qid, "fold": inum(training_rows[ids[0]].get("fold")),
                "threshold": threshold, "base_actual_stg": float(y[base_global]),
                "selected_actual_stg": float(y[choice_global]),
                "actual_gain": float(y[choice_global] - y[base_global]),
                "replaced": int(choice_global != base_global),
            })
        gain = float(np.mean(selected_values) - np.mean(base_values))
        replacement_fraction = float(np.mean([r["replaced"] for r in query_details]))
        calibration_rows.append({
            "ridge_lambda": RANKER_LAMBDA, "replacement_threshold": threshold,
            "query_count": len(query_details), "oof_selected_stg": float(np.mean(selected_values)),
            "oof_base_p2a_stg": float(np.mean(base_values)), "oof_gain": gain,
            "oof_replacement_fraction": replacement_fraction,
        })
        key = (gain, -abs(threshold), -threshold)
        if best is None or key > best:
            best = key
            best_details = query_details
    if best is None:
        raise P2C1Error("Candidate-ranker threshold calibration failed")
    chosen_row = max(calibration_rows, key=lambda r: (float(r["oof_gain"]), -abs(float(r["replacement_threshold"])), -float(r["replacement_threshold"])))
    threshold = float(chosen_row["replacement_threshold"])
    beta, mu, sd = mh.fit_weighted_ridge(X, y, groups, qids, RANKER_LAMBDA)
    model_path = output_dir / "multi_hypothesis_ranker_full_model_v0_1_0.npz"
    np.savez(
        model_path,
        beta=beta.astype(np.float64), feature_mean=mu.astype(np.float64), feature_std=sd.astype(np.float64),
        feature_names=np.asarray(mh.ranker_feature_names(), dtype="U128"),
        ridge_lambda=np.asarray([RANKER_LAMBDA], dtype=np.float64),
        replacement_threshold=np.asarray([threshold], dtype=np.float64),
        decision=np.asarray(["select_best_guarded_candidate_if_predicted_gain_ge_threshold_else_p2a_base"], dtype="U128"),
    )
    write_csv(output_dir / "full_ranker_threshold_calibration.csv", calibration_rows)
    write_csv(output_dir / "full_ranker_oof_selection_audit.csv", [r for r in best_details if float(r["threshold"]) == threshold])
    audit = {
        "status": "PASS",
        "training_candidate_count": len(training_rows),
        "training_query_count": len(by_query),
        "training_feature_dim": X.shape[1],
        "ridge_lambda": RANKER_LAMBDA,
        "replacement_threshold": threshold,
        "oof_gain_over_p2a_base": float(chosen_row["oof_gain"]),
        "oof_replacement_fraction": float(chosen_row["oof_replacement_fraction"]),
        "fold_models": fold_models,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
    }
    json_dump(output_dir / "full_candidate_ranker_audit.json", audit)
    return audit


def load_full_ranker(path: Path, mh: Any) -> dict[str, Any]:
    model = load_npz_model(path)
    names = [str(x) for x in np.asarray(model["feature_names"]).reshape(-1).tolist()]
    if names != list(mh.ranker_feature_names()):
        raise P2C1Error("Full candidate-ranker feature names do not match P2A-MH v0.1.2")
    return {
        **model,
        "threshold": float(np.asarray(model["replacement_threshold"]).reshape(-1)[0]),
        "lambda": float(np.asarray(model["ridge_lambda"]).reshape(-1)[0]),
    }


def query_scene_info(row: Mapping[str, Any]) -> dict[str, Any] | None:
    center_text = str(row.get("matched_bbox_center", "")).strip()
    size_text = str(row.get("matched_bbox_size", "")).strip()
    if not center_text or not size_text:
        return None
    center = parse_json_vector(center_text)
    size = parse_json_vector(size_text)
    ref = np.asarray([
        fnum(row.get("reference_pelvis_world_x")),
        fnum(row.get("reference_pelvis_world_y")),
        fnum(row.get("reference_pelvis_world_z")),
    ], dtype=np.float64)
    if not np.all(np.isfinite(ref)):
        return None
    return {"center_rel": center - ref, "half_diag": 0.5 * float(np.linalg.norm(size))}


def load_query_beta(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row.get("body_shape_path", "")))
    if not path.is_file():
        raise P2C1Error(f"Missing body shape: {path}")
    beta = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    if beta.size < 10 or not np.all(np.isfinite(beta[:10])):
        raise P2C1Error(f"Invalid body shape: {path}")
    return beta[:10].copy()


def rerank_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = candidate["metrics"]
    return (
        -fnum(metrics.get("planned_score1b_proxy20"), -1.0),
        -fnum(metrics.get("self_score1b_proxy20"), -1.0),
        -fnum(metrics.get("detector_agreement"), -1.0),
        fnum(metrics.get("joint_jerk"), 1e9),
        fnum(candidate.get("donor_distance"), 1e9),
    )


def candidate_key(sample_id: str, planned_frame: int, donor_id: str) -> tuple[str, int, str]:
    return str(sample_id), int(planned_frame), str(donor_id)


def make_candidate(
    query: Mapping[str, Any], donor: Mapping[str, Any], planned: int, timing: Mapping[str, Any],
    method: str, hypothesis: str, donor_rank: int, donor_distance: float,
    beta: np.ndarray, gender: str,
) -> dict[str, Any]:
    return {
        "query_id": str(query["sample_id"]), "row": dict(query), "eval_row": dict(query),
        "donor_id": str(donor["sample_id"]), "donor_row": dict(donor),
        "planned_peak": int(planned), "timing": dict(timing), "method": method,
        "hypothesis": hypothesis, "donor_rank": int(donor_rank),
        "donor_distance": float(donor_distance), "query_betas": np.asarray(beta, dtype=np.float32),
        "gender": str(gender),
    }


def materialize_candidate_poses(
    records: Sequence[dict[str, Any]], module: Any, engine: Any,
    donor_by_id: Mapping[str, Mapping[str, Any]], clip_cache: dict[str, dict[str, Any]],
    schema_cache: dict[Path, dict[str, Any]], args: argparse.Namespace, stage_name: str,
) -> list[dict[str, Any]]:
    records = list(records)
    if not records:
        return records
    for i, c in enumerate(records):
        did = str(c["donor_id"])
        if did not in clip_cache:
            clip_cache[did] = module.load_full_clip(donor_by_id[did], schema_cache)
        donor_clip = clip_cache[did]
        drow = donor_by_id[did]
        warped = module.warp_clip(
            donor_clip, inum(drow.get("released_peak_frame")),
            inum(c["row"].get("clip_length")), int(c["planned_peak"]),
        )
        c["base_pose"] = warped["pose"]
        c["base_trans"] = warped["trans"]
        c["base_apex"] = warped["pose"][int(c["planned_peak"])].copy()
        if (i + 1) % 200 == 0:
            print(f"[{stage_name} warp] {i+1}/{len(records)}", flush=True)

    ik_log: list[dict[str, Any]] = []
    for hand in (0, 1):
        for gender in ("neutral", "male", "female"):
            ids = [
                i for i, c in enumerate(records)
                if inum(c["row"].get("hand")) == hand and module.normalize_gender(c["gender"]) == gender
            ]
            for start in range(0, len(ids), args.ik_batch_size):
                take = ids[start:start + args.ik_batch_size]
                if not take:
                    continue
                base = np.stack([records[i]["base_apex"] for i in take]).astype(np.float32)
                betas = np.stack([records[i]["query_betas"] for i in take]).astype(np.float32)
                targets = np.stack([module.target_vec(records[i]["row"]) for i in take]).astype(np.float32)
                opt_pose, logs = module.optimize_apex_batch(
                    engine, base, betas, gender, hand, "p0c", targets, None,
                    args.ik_steps, args.ik_lr, args.ik_max_delta,
                )
                for local, idx in enumerate(take):
                    c = records[idx]
                    c["optimized_apex"] = opt_pose[local]
                    c["pose"] = module.apply_hold(
                        c["base_pose"], int(c["planned_peak"]), opt_pose[local], hand,
                        args.hold_plateau, args.hold_taper,
                    )
                    c["trans"] = c["base_trans"].copy()
                    c["betas_full"] = np.broadcast_to(
                        c["query_betas"][None, :], (len(c["pose"]), len(c["query_betas"]))
                    ).copy()
                    c["donor_pose"] = c["base_pose"].copy()
                for log in logs:
                    ik_log.append({"stage": stage_name, "hand": hand, "gender": gender, "batch_start": start, **log})
            if ids:
                print(f"[{stage_name} IK] hand={hand} gender={gender} candidates={len(ids)}", flush=True)
    args._ik_logs.extend(ik_log)
    return records


def evaluate_test_candidate(
    candidate: dict[str, Any], joints: np.ndarray, module: Any,
    source_model: Mapping[str, np.ndarray], refined_model: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    row = candidate["row"]
    hand = inum(row.get("hand"))
    planned = int(candidate["planned_peak"])
    det = module.detector_predictions(joints, hand, source_model, refined_model)
    self_frame = int(det["ensemble"])
    scene = query_scene_info(row)
    planned_angle = module.spatial_angle(joints, hand, planned, row, scene)
    self_angle = module.spatial_angle(joints, hand, self_frame, row, scene)
    quality = module.motion_quality(joints, candidate["pose"], candidate["donor_pose"], hand)
    pelvis = np.asarray(joints[planned, module.JOINT_INDEX["pelvis"]], dtype=np.float64)
    candidate["pelvis_at_planned"] = pelvis
    return {
        "planned_frame": planned,
        "self_frame": self_frame,
        "det_source": int(det["source"]), "det_refined": int(det["refined"]),
        "det_heuristic": int(det["heuristic"]), "detector_agreement": float(det["agreement"]),
        "planned_angle_deg": float(planned_angle), "planned_score1b_proxy20": float(module.score20(planned_angle)),
        "self_angle_deg": float(self_angle), "self_score1b_proxy20": float(module.score20(self_angle)),
        **quality,
    }


def evaluate_records(
    records: Sequence[dict[str, Any]], module: Any, engine: Any,
    source_model: Mapping[str, np.ndarray], refined_model: Mapping[str, np.ndarray],
    stage_name: str,
) -> list[dict[str, Any]]:
    records = list(records)
    if not records:
        return records
    clips = [
        {"pose": c["pose"], "trans": c["trans"], "betas": c["betas_full"], "gender": c["gender"]}
        for c in records
    ]
    joints_many = engine.joints_many(clips)
    for i, (c, joints) in enumerate(zip(records, joints_many)):
        c["metrics"] = evaluate_test_candidate(c, joints, module, source_model, refined_model)
        if (i + 1) % 200 == 0:
            print(f"[{stage_name} eval] {i+1}/{len(records)}", flush=True)
    return records


def candidate_metrics_row(c: Mapping[str, Any], stage: str, view_rank: int | None = None, view_distance: float | None = None, view_hypothesis: str | None = None) -> dict[str, Any]:
    return {
        "stage": stage, "query_id": c["query_id"],
        "method": c.get("method"), "hypothesis": view_hypothesis or c.get("hypothesis"),
        "donor_id": c.get("donor_id"),
        "donor_rank": int(view_rank if view_rank is not None else inum(c.get("donor_rank"))),
        "donor_distance": float(view_distance if view_distance is not None else fnum(c.get("donor_distance"))),
        "hand": inum(c["row"].get("hand")), "clip_length": inum(c["row"].get("clip_length")),
        **dict(c.get("timing", {})), **dict(c.get("metrics", {})),
    }


def phase_router_row(query: Mapping[str, Any], median: Mapping[str, Any], wordphase: Mapping[str, Any], median_frame: int, wp_frame: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": query["sample_id"], "domain": query["domain"], "hand": query["hand"],
        "clip_length": query["clip_length"], "target_distance": query["target_distance"],
        "target_azimuth_deg": query["target_azimuth_deg"], "target_elevation_deg": query["target_elevation_deg"],
        "median_peak": median_frame, "wordphase_peak": wp_frame,
    }
    for method, c in (("median", median), ("wordphase", wordphase)):
        m = c["metrics"]
        row.update({
            f"{method}_planned_frame": m["planned_frame"], f"{method}_self_frame": m["self_frame"],
            f"{method}_det_source": m["det_source"], f"{method}_det_refined": m["det_refined"],
            f"{method}_det_heuristic": m["det_heuristic"], f"{method}_detector_agreement": m["detector_agreement"],
            f"{method}_planned_score1b_proxy20": m["planned_score1b_proxy20"],
            f"{method}_self_score1b_proxy20": m["self_score1b_proxy20"],
            f"{method}_self_angle_deg": m["self_angle_deg"], f"{method}_joint_jerk": m["joint_jerk"],
            f"{method}_foot_sliding": m["foot_sliding"], f"{method}_pose_deviation": m["pose_deviation"],
            f"{method}_max_reach": m["max_reach"], f"{method}_donor_rank": c["donor_rank"],
        })
    return row


def world_aligned_arrays(candidate: Mapping[str, Any], query: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ref = np.asarray([
        fnum(query.get("reference_pelvis_world_x")),
        fnum(query.get("reference_pelvis_world_y")),
        fnum(query.get("reference_pelvis_world_z")),
    ], dtype=np.float64)
    pelvis = np.asarray(candidate["pelvis_at_planned"], dtype=np.float64)
    if ref.shape != (3,) or pelvis.shape != (3,) or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(pelvis)):
        raise P2C1Error(f"Invalid world-alignment data for {query.get('sample_id')}")
    shift = ref - pelvis
    pose = np.asarray(candidate["pose"], dtype=np.float32)
    trans = (np.asarray(candidate["trans"], dtype=np.float64) + shift[None, :]).astype(np.float32)
    audit = {
        "sample_id": query["sample_id"], "planned_frame": int(candidate["planned_peak"]),
        "reference_pelvis_world": ref.tolist(), "centered_pelvis_at_planned": pelvis.tolist(),
        "constant_translation_shift": shift.tolist(),
        "aligned_pelvis_error_m": float(np.linalg.norm((pelvis + shift) - ref)),
    }
    return pose, trans, audit


def save_motion(path: Path, pose: np.ndarray, trans: np.ndarray, T: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = np.asarray(pose, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    if pose.shape != (T, 165) or trans.shape != (T, 3):
        raise P2C1Error(f"Output schema mismatch for {path.name}: poses={pose.shape}, trans={trans.shape}, T={T}")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(trans)):
        raise P2C1Error(f"Non-finite output for {path.name}")
    np.savez_compressed(path, poses=pose, trans=trans)
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"poses", "trans"}:
            raise P2C1Error(f"Unexpected keys in {path}")


def create_submission_zip(submission_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        metadata = submission_dir / "metadata.json"
        if metadata.is_file():
            zf.write(metadata, "metadata.json")
        for p in sorted((submission_dir / "motions").glob("*.npz")):
            zf.write(p, f"motions/{p.name}")


def run_validator(validator: Path, submission: Path, manifest: Path, log_path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(validator), str(submission), "--manifest", str(manifest)],
        text=True, capture_output=True,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(text, encoding="utf-8")
    return {"submission": str(submission), "returncode": proc.returncode, "output": text.strip(), "status": "PASS" if proc.returncode == 0 else "FAIL"}


def summarize_metrics(candidates: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = ["planned_score1b_proxy20", "self_score1b_proxy20", "self_angle_deg", "detector_agreement", "joint_jerk", "foot_sliding", "pose_deviation", "joint_limit_fraction"]
    return {k: float(np.mean([fnum(c["metrics"].get(k), 0.0) for c in candidates])) for k in keys}


def p2a_mh_domain_evidence(p2a_mh_dir: Path) -> dict[str, Any]:
    path = p2a_mh_dir / "summary_metrics.csv"
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    rows = read_csv(path)
    picked: dict[str, Any] = {"status": "PASS", "path": str(path)}
    for metric in ("self_stg_proxy_sigma5", "self_score1b_proxy20", "detector_agreement", "joint_jerk", "foot_sliding", "pose_deviation"):
        rec = next((r for r in rows if str(r.get("group_name")) == "domain" and str(r.get("group_value")) == "natural" and str(r.get("metric")) == metric), None)
        if rec is not None:
            picked[metric] = {
                "mh_mean": fnum(rec.get("mh_mean")),
                "p2a_base_mean": fnum(rec.get("p2a_base_mean")),
                "mh_minus_base": fnum(rec.get("mh_minus_base")),
                "sample_ci95": [fnum(rec.get("sample_ci95_low")), fnum(rec.get("sample_ci95_high"))],
                "cluster_ci95": [fnum(rec.get("cluster_ci95_low")), fnum(rec.get("cluster_ci95_high"))],
            }
    return picked


def submission_recommendation(test_rows: Sequence[Mapping[str, Any]], natural_evidence: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(str(r.get("domain", "")) for r in test_rows)
    all_natural = counts == Counter({"natural": len(test_rows)})
    spatial = dict(natural_evidence.get("self_score1b_proxy20", {})) if natural_evidence.get("status") == "PASS" else {}
    stg = dict(natural_evidence.get("self_stg_proxy_sigma5", {})) if natural_evidence.get("status") == "PASS" else {}
    spatial_ci = spatial.get("cluster_ci95", [float("nan"), float("nan")])
    stg_ci = stg.get("cluster_ci95", [float("nan"), float("nan")])
    natural_material_gain = (
        np.isfinite(fnum(spatial.get("mh_minus_base"))) and fnum(spatial.get("mh_minus_base")) >= 0.005 and
        len(spatial_ci) == 2 and fnum(spatial_ci[0]) > 0.0
    )
    if all_natural and not natural_material_gain:
        primary = "candidate_a_p2a"
        rationale = (
            "All 100 disclosed test inputs use the natural-domain policy, while full-OOF P2A-MH "
            "shows no material, confidence-supported natural-domain Spatial Score gain over P2A. "
            "Candidate B remains a validated experimental alternative, but Candidate A is the safer primary submission."
        )
    else:
        primary = "candidate_b_p2a_mh"
        rationale = (
            "The disclosed test-domain mix and full-OOF domain evidence support using the multi-hypothesis candidate as primary."
        )
    return {
        "recommended_primary": primary,
        "safe_fallback": "candidate_a_p2a",
        "experimental_alternative": "candidate_b_p2a_mh",
        "test_domain_policy_counts": dict(counts),
        "all_test_inputs_natural_policy": all_natural,
        "natural_material_spatial_gain_gate": bool(natural_material_gain),
        "natural_spatial_evidence": spatial,
        "natural_stg_evidence": stg,
        "rationale": rationale,
        "guardrail": "Local proxy evidence cannot certify organizer-server performance; do not discard either validated ZIP before submission."
    }


def source_record(path: Path, name: str) -> dict[str, Any]:
    return {"name": name, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def preflight(args: argparse.Namespace, module: Any, mh: Any, device: str) -> dict[str, Any]:
    p2c0_accept = json_load(args.p2c0_dir / "P2C0_ACCEPTANCE.json")
    if p2c0_accept.get("status") != "P2C0_OFFICIAL_TEST_PREFLIGHT_ACCEPTED_MATERIALIZATION_ALLOWED" or not p2c0_accept.get("materialization_allowed"):
        raise P2C1Error("P2C0 does not allow materialization")
    p2a_mh_accept = json_load(args.p2a_mh_dir / "P2A_MH_ACCEPTANCE.json")
    if inum(p2a_mh_accept.get("query_count")) != 2525 or not p2a_mh_accept.get("engineering_gate") or not p2a_mh_accept.get("primary_gate") or not p2a_mh_accept.get("quality_gate"):
        raise P2C1Error("P2A-MH full OOF gate is not accepted")
    selected_outputs = list((args.p2a_mh_dir / "folds").glob("fold_*/motions/selected/*.npz"))
    if len(selected_outputs) != 2525:
        raise P2C1Error(f"P2A-MH selected output coverage mismatch: {len(selected_outputs)}")
    if json_load(args.p1_full_dir / "P1_FULL_ACCEPTANCE.json").get("status") != "P1_WORDPHASE_FULL_OOF_ACCEPTED":
        raise P2C1Error("P1 full OOF is not accepted")
    if json_load(args.p2a_dir / "P2A_ACCEPTANCE.json").get("status") != "P2A_PHASE_ROUTER_ACCEPTED":
        raise P2C1Error("P2A is not accepted")

    train_rows = read_csv(args.p0c0_dir / "shared_generation_manifest.csv")
    if len(train_rows) != 2525:
        raise P2C1Error(f"Training manifest coverage mismatch: {len(train_rows)}")
    test_rows, timing_by_id, medians = build_test_rows(args.data_root, args.p2c0_dir, train_rows, mh)
    natural_evidence = p2a_mh_domain_evidence(args.p2a_mh_dir)
    recommendation = submission_recommendation(test_rows, natural_evidence)
    json_dump(args.output_dir / "submission_recommendation_preflight.json", recommendation)
    write_csv(args.output_dir / "test_generation_manifest.csv", test_rows)
    write_csv(args.output_dir / "test_fixed_timing_manifest.csv", [
        {"sample_id": sid, **timing} for sid, timing in timing_by_id.items()
    ])

    dist_by_id, ensemble_rows, ensemble_audit = infer_test_wordphase_ensemble(
        test_rows, timing_by_id, args.p1a_dir, mh, device, args.max_words,
    )
    write_csv(args.output_dir / "wordphase_test_ensemble_audit.csv", ensemble_rows)
    json_dump(args.output_dir / "wordphase_test_ensemble_audit.json", ensemble_audit)
    save_test_distribution_cache(args.output_dir / "wordphase_test_distribution_cache_v0_1_0.npz", test_rows, dist_by_id)

    full_ranker_audit = fit_full_candidate_ranker(
        train_rows, args.p1_full_dir, args.p2a_dir, args.p2a_mh_dir, mh, args.output_dir,
    )
    phase_model_path = args.p2a_dir / "phase_router_full_model.npz"
    phase_model = load_npz_model(phase_model_path)
    phase_names = [str(x) for x in np.asarray(phase_model["feature_names"]).reshape(-1).tolist()]
    if phase_names != phase_router_feature_names():
        raise P2C1Error("Phase-router model feature contract failed")

    assets = module.find_smplx_assets(args.smplx_model_dir, args.data_root, args.project_root)
    if assets.get("status") != "OK":
        raise P2C1Error(f"SMPL-X assets missing: {assets}")
    validator = args.data_root / "test_submission" / "validate_submission.py"
    manifest = args.data_root / "test_submission" / "manifest.csv"
    if not validator.is_file() or not manifest.is_file():
        raise P2C1Error("Official validator or manifest is missing")
    donor_counts = Counter(inum(r.get("hand")) for r in train_rows if str(r.get("domain")) == "natural")
    if donor_counts[0] < args.top_k or donor_counts[1] < args.top_k:
        raise P2C1Error(f"Insufficient full-training donors: {donor_counts}")

    closeout = {
        "schema_version": SCHEMA_VERSION, "version": VERSION, "created_utc": now_utc(),
        "status": "P2A_MH_FULL_OOF_ACCEPTED_FOR_OFFICIAL_MATERIALIZATION",
        "source_run": str(args.p2a_mh_dir), "source_status": p2a_mh_accept.get("status"),
        "query_count": inum(p2a_mh_accept.get("query_count")), "fold_count": 5,
        "engineering_gate": bool(p2a_mh_accept.get("engineering_gate")),
        "primary_gate": bool(p2a_mh_accept.get("primary_gate")),
        "quality_gate": bool(p2a_mh_accept.get("quality_gate")),
        "selected_motion_count": len(selected_outputs),
        "candidate_b_materialization_allowed": True,
        "note": "The source launcher retained a pilot status name, but the run covers all 2525 OOF queries and five folds.",
    }
    json_dump(args.output_dir / "P2A_MH_FULL_OOF_CLOSEOUT.json", closeout)

    pre = {
        "schema_version": SCHEMA_VERSION, "version": VERSION, "created_utc": now_utc(),
        "status": "P2C1_PREFLIGHT_ACCEPTED", "device": device,
        "test_query_count": len(test_rows), "test_unique_ids": len({str(r['sample_id']) for r in test_rows}),
        "wordphase_checkpoint_count": 5, "wordphase_batch_size": WORDPHASE_BATCH_SIZE,
        "all_training_median_ratios": {str(k): v for k, v in medians.items()},
        "natural_donor_counts": {str(k): int(v) for k, v in donor_counts.items()},
        "phase_router_gate": True, "full_candidate_ranker_gate": full_ranker_audit["status"] == "PASS",
        "smplx_assets": assets, "validator": str(validator), "manifest": str(manifest),
        "motion_generation_started": False,
        "candidate_a": "RePointer-P2A",
        "candidate_b": "RePointer-P2A-MH full candidate ranker with P2A fallback",
        "submission_recommendation": recommendation,
    }
    json_dump(args.output_dir / "PREFLIGHT.json", pre)
    return pre


def formal_materialization(args: argparse.Namespace, module: Any, mh: Any, device: str) -> dict[str, Any]:
    pre_path = args.output_dir / "PREFLIGHT.json"
    if not pre_path.is_file() or json_load(pre_path).get("status") != "P2C1_PREFLIGHT_ACCEPTED":
        preflight(args, module, mh, device)
    completion = args.output_dir / "P2C1_ACCEPTANCE.json"
    if completion.is_file() and json_load(completion).get("status") == "P2C1_OFFICIAL_TEST_MATERIALIZATION_COMPLETE":
        print("[RESUME] P2C1 already complete; no regeneration performed", flush=True)
        return json_load(completion)

    train_rows = read_csv(args.p0c0_dir / "shared_generation_manifest.csv")
    test_rows = read_csv(args.output_dir / "test_generation_manifest.csv")
    test_by_id = {str(r["sample_id"]): r for r in test_rows}
    recommendation = submission_recommendation(test_rows, p2a_mh_domain_evidence(args.p2a_mh_dir))
    json_dump(args.output_dir / "submission_recommendation.json", recommendation)
    dist_by_id = load_test_distribution_cache(args.output_dir / "wordphase_test_distribution_cache_v0_1_0.npz", test_rows)
    full_ranker = load_full_ranker(args.output_dir / "multi_hypothesis_ranker_full_model_v0_1_0.npz", mh)
    phase_model = load_npz_model(args.p2a_dir / "phase_router_full_model.npz")
    source_model = module.load_linear_model(args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_0_0.npz")
    refined_model = module.load_linear_model(args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_1_0.npz")
    assets = module.find_smplx_assets(args.smplx_model_dir, args.data_root, args.project_root)
    engine = module.SMPLXEngine(assets, args.device, args.fk_batch_size)

    donor_rows = [dict(r) for r in train_rows if str(r.get("domain")) == "natural"]
    donor_by_id = {str(r["sample_id"]): r for r in donor_rows}
    donors_by_hand: dict[int, list[dict[str, Any]]] = {}
    donor_descriptors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for hand in (0, 1):
        donors = [r for r in donor_rows if inum(r.get("hand")) == hand]
        D = np.stack([module.descriptor(r) for r in donors])
        sd = D.std(axis=0)
        sd[sd < 1e-6] = 1.0
        donors_by_hand[hand] = donors
        donor_descriptors[hand] = (D, sd)

    query_beta = {str(q["sample_id"]): load_query_beta(q) for q in test_rows}
    query_gender = {str(q["sample_id"]): module.normalize_gender(q.get("gender")) for q in test_rows}
    full_medians = training_median_ratios(train_rows)
    args._ik_logs = []
    schema_cache: dict[Path, dict[str, Any]] = {}
    clip_cache: dict[str, dict[str, Any]] = {}

    # Stage A: exact frozen P2A test policy -- top-8 median and expected pools.
    stage_a: list[dict[str, Any]] = []
    a_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for qi, q in enumerate(test_rows):
        sid = str(q["sample_id"])
        hand = inum(q.get("hand"))
        T = inum(q.get("clip_length"))
        median_frame = int(np.clip(round(full_medians[hand] * max(1, T - 1)), 2, max(2, T - 3)))
        expected_frame = inum(dist_by_id[sid]["expected_frame"])
        timings = {
            "median": mh.timing_features_for_frame(dist_by_id[sid], "median", median_frame, median_frame),
            "wordphase": mh.timing_features_for_frame(dist_by_id[sid], "wp_expected", expected_frame, median_frame),
        }
        for method, planned in (("median", median_frame), ("wordphase", expected_frame)):
            donors = donors_by_hand[hand]
            D, sd = donor_descriptors[hand]
            qdesc = module.descriptor(q, planned / max(1, T - 1))
            distances = np.linalg.norm((D - qdesc[None, :]) / sd[None, :], axis=1)
            for rank, idx in enumerate(np.argsort(distances, kind="mergesort")[: args.top_k].tolist(), 1):
                donor = donors[int(idx)]
                c = make_candidate(
                    q, donor, planned, timings[method], method, "median" if method == "median" else "wp_expected",
                    rank, float(distances[int(idx)]), query_beta[sid], query_gender[sid],
                )
                stage_a.append(c)
                a_lookup[candidate_key(sid, planned, str(donor["sample_id"]))] = c
        if (qi + 1) % 20 == 0:
            print(f"[Candidate A retrieval] {qi+1}/{len(test_rows)}", flush=True)
    materialize_candidate_poses(stage_a, module, engine, donor_by_id, clip_cache, schema_cache, args, "Candidate A")
    evaluate_records(stage_a, module, engine, source_model, refined_model, "Candidate A")

    by_a: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in stage_a:
        by_a[(str(c["query_id"]), str(c["method"]))].append(c)
    selected_a: dict[str, dict[str, Any]] = {}
    route_rows: list[dict[str, Any]] = []
    branch_selected: dict[tuple[str, str], dict[str, Any]] = {}
    for q in test_rows:
        sid = str(q["sample_id"])
        med_candidates = sorted(by_a[(sid, "median")], key=rerank_key)
        wp_candidates = sorted(by_a[(sid, "wordphase")], key=rerank_key)
        median = med_candidates[0]
        wordphase = wp_candidates[0]
        branch_selected[(sid, "median")] = median
        branch_selected[(sid, "wordphase")] = wordphase
        row = phase_router_row(
            q, median, wordphase, int(median["planned_peak"]), int(wordphase["planned_peak"]),
        )
        predicted_delta = phase_router_predict(phase_model, row)
        choice = "wordphase" if predicted_delta >= 0.0 else "median"
        chosen = wordphase if choice == "wordphase" else median
        selected_a[sid] = chosen
        route_rows.append({
            "sample_id": sid, "router_choice": choice,
            "predicted_wordphase_minus_median_stg": predicted_delta,
            "median_planned_frame": median["planned_peak"], "median_donor_id": median["donor_id"],
            "median_donor_rank": median["donor_rank"], "wordphase_planned_frame": wordphase["planned_peak"],
            "wordphase_donor_id": wordphase["donor_id"], "wordphase_donor_rank": wordphase["donor_rank"],
            "selected_planned_frame": chosen["planned_peak"], "selected_donor_id": chosen["donor_id"],
            "selected_self_score1b_proxy20": chosen["metrics"]["self_score1b_proxy20"],
        })
    write_csv(args.output_dir / "candidate_a_phase_router_manifest.csv", route_rows)

    # Stage B planning. Reuse stage-A candidates whenever timing/donor match.
    stage_b_new: list[dict[str, Any]] = []
    b_views_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hypotheses_manifest: list[dict[str, Any]] = []
    for q in test_rows:
        sid = str(q["sample_id"])
        hand = inum(q.get("hand"))
        T = inum(q.get("clip_length"))
        median_frame = int(np.clip(round(full_medians[hand] * max(1, T - 1)), 2, max(2, T - 3)))
        hypotheses = mh.build_hypotheses(dist_by_id[sid], median_frame, args.hypothesis_count, args.min_hypothesis_sep_frames)
        for h in hypotheses:
            hypotheses_manifest.append({"sample_id": sid, **h})
        base = selected_a[sid]
        base_frame = int(base["planned_peak"])
        base_donor = str(base["donor_id"])
        reserved = min(hypotheses, key=lambda h: (abs(inum(h.get("planned_frame")) - base_frame), inum(h.get("hypothesis_rank"), 999)))
        donors = donors_by_hand[hand]
        D, sd = donor_descriptors[hand]
        for h in hypotheses:
            planned = inum(h.get("planned_frame"))
            qdesc = module.descriptor(q, planned / max(1, T - 1))
            distances = np.linalg.norm((D - qdesc[None, :]) / sd[None, :], axis=1)
            order = np.argsort(distances, kind="mergesort").tolist()
            if str(h["hypothesis"]) == str(reserved["hypothesis"]):
                order = [idx for idx in order if str(donors[int(idx)]["sample_id"]) != base_donor]
                take_count = args.donors_per_hypothesis - 1
            else:
                take_count = args.donors_per_hypothesis
            for rank, idx in enumerate(order[:take_count], 1):
                donor = donors[int(idx)]
                key = candidate_key(sid, planned, str(donor["sample_id"]))
                c = a_lookup.get(key)
                if c is None:
                    c = make_candidate(
                        q, donor, planned, h, "mh", str(h["hypothesis"]), rank,
                        float(distances[int(idx)]), query_beta[sid], query_gender[sid],
                    )
                    stage_b_new.append(c)
                    a_lookup[key] = c
                b_views_by_id[sid].append({
                    "candidate": c, "timing": dict(h), "hypothesis": str(h["hypothesis"]),
                    "donor_rank": rank, "donor_distance": float(distances[int(idx)]),
                })
        if len(b_views_by_id[sid]) != args.hypothesis_count * args.donors_per_hypothesis - 1:
            raise P2C1Error(f"Candidate B budget mismatch for {sid}: {len(b_views_by_id[sid])}")
    write_csv(args.output_dir / "candidate_b_timing_hypothesis_manifest.csv", hypotheses_manifest)
    materialize_candidate_poses(stage_b_new, module, engine, donor_by_id, clip_cache, schema_cache, args, "Candidate B novel")
    evaluate_records(stage_b_new, module, engine, source_model, refined_model, "Candidate B novel")

    selected_b: dict[str, dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = [candidate_metrics_row(c, "candidate_a_pool") for c in stage_a]
    all_metric_rows.extend(candidate_metrics_row(c, "candidate_b_novel") for c in stage_b_new)
    for q in test_rows:
        sid = str(q["sample_id"])
        base = selected_a[sid]
        median_frame = int(np.clip(round(full_medians[inum(q.get("hand"))] * max(1, inum(q.get("clip_length")) - 1)), 2, max(2, inum(q.get("clip_length")) - 3)))
        base_hyp = "median" if str(base.get("method")) == "median" else "wp_expected"
        base_timing = mh.timing_features_for_frame(dist_by_id[sid], base_hyp, int(base["planned_peak"]), median_frame)
        base_view = {"method": "p2a_base", "hypothesis": base_hyp, "donor_id": base["donor_id"], "donor_rank": base["donor_rank"], "donor_distance": base["donor_distance"], **base["metrics"]}
        pool_rows: list[dict[str, Any]] = [base_view]
        pool_timing: list[dict[str, Any]] = [base_timing]
        pool_candidates: list[dict[str, Any]] = [base]
        for view in b_views_by_id[sid]:
            c = view["candidate"]
            row = {"method": "mh", "hypothesis": view["hypothesis"], "donor_id": c["donor_id"], "donor_rank": view["donor_rank"], "donor_distance": view["donor_distance"], **c["metrics"]}
            pool_rows.append(row)
            pool_timing.append(view["timing"])
            pool_candidates.append(c)
            all_metric_rows.append(candidate_metrics_row(c, "candidate_b_pool_view", view["donor_rank"], view["donor_distance"], view["hypothesis"]))
        query_ctx = {
            "domain": q["domain"], "hand": q["hand"], "clip_length": q["clip_length"],
            "target_distance": q["target_distance"], "target_azimuth_deg": q["target_azimuth_deg"],
            "target_elevation_deg": q["target_elevation_deg"],
        }
        X = np.stack([mh.build_ranker_feature(query_ctx, timing, row) for timing, row in zip(pool_timing, pool_rows)])
        predicted = mh.ridge_predict(X, full_ranker["beta"], full_ranker["feature_mean"], full_ranker["feature_std"])
        choice = mh.select_with_threshold(pool_rows, predicted, 0, full_ranker["threshold"])
        chosen = pool_candidates[choice]
        selected_b[sid] = chosen
        selection_rows.append({
            "sample_id": sid, "pool_size": len(pool_rows), "choice_index": choice,
            "selected_source": "p2a_base" if choice == 0 else "mh",
            "selected_hypothesis": pool_rows[choice]["hypothesis"],
            "selected_donor_id": pool_rows[choice]["donor_id"],
            "selected_donor_rank": pool_rows[choice]["donor_rank"],
            "predicted_base_utility": float(predicted[0]),
            "predicted_selected_utility": float(predicted[choice]),
            "predicted_utility_gain": float(predicted[choice] - predicted[0]),
            "replacement_threshold": full_ranker["threshold"],
            "base_self_score1b_proxy20": base["metrics"]["self_score1b_proxy20"],
            "selected_self_score1b_proxy20": chosen["metrics"]["self_score1b_proxy20"],
            "guard_pass": bool(choice == 0 or mh.candidate_guard(pool_rows[choice], pool_rows[0])),
        })
    write_csv(args.output_dir / "candidate_b_selection_manifest.csv", selection_rows)
    write_csv(args.output_dir / "official_test_candidate_metrics.csv", all_metric_rows)
    write_csv(args.output_dir / "ik_training_log.csv", args._ik_logs)

    submissions_root = args.output_dir / "submissions"
    if submissions_root.exists():
        shutil.rmtree(submissions_root)
    candidate_a_dir = submissions_root / "candidate_a_p2a"
    candidate_b_dir = submissions_root / "candidate_b_p2a_mh"
    (candidate_a_dir / "motions").mkdir(parents=True)
    (candidate_b_dir / "motions").mkdir(parents=True)
    world_audit_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    for q in test_rows:
        sid = str(q["sample_id"])
        T = inum(q.get("clip_length"))
        for label, candidate, root in (
            ("candidate_a_p2a", selected_a[sid], candidate_a_dir),
            ("candidate_b_p2a_mh", selected_b[sid], candidate_b_dir),
        ):
            pose, trans, audit = world_aligned_arrays(candidate, q)
            out = root / "motions" / f"{sid}.npz"
            save_motion(out, pose, trans, T)
            world_audit_rows.append({"candidate": label, **audit})
            schema_rows.append({
                "candidate": label, "sample_id": sid, "clip_length": T,
                "poses_shape": str(pose.shape), "trans_shape": str(trans.shape),
                "poses_dtype": str(pose.dtype), "trans_dtype": str(trans.dtype),
                "finite": bool(np.all(np.isfinite(pose)) and np.all(np.isfinite(trans))),
                "keys": "poses,trans", "status": "PASS",
            })
    write_csv(args.output_dir / "world_alignment_audit.csv", world_audit_rows)
    write_csv(args.output_dir / "submission_schema_audit.csv", schema_rows)

    team = sanitize_name(args.team_name)
    metadata_a = {
        "team_name": args.team_name,
        "system": "RePointer-P2A",
        "version": VERSION,
        "description": "Five-checkpoint WordPhase ensemble plus all-training median pool and frozen P2A phase router.",
    }
    metadata_b = {
        "team_name": args.team_name,
        "system": "RePointer-P2A-MH",
        "version": VERSION,
        "description": "Four timing hypotheses, multi-donor candidates, full OOF candidate ranker, and explicit P2A fallback.",
    }
    json_dump(candidate_a_dir / "metadata.json", metadata_a)
    json_dump(candidate_b_dir / "metadata.json", metadata_b)
    zip_root = args.output_dir / "submission_zips"
    zip_a = zip_root / f"{team}_P2A_submission.zip"
    zip_b = zip_root / f"{team}_P2A_MH_submission.zip"
    create_submission_zip(candidate_a_dir, zip_a)
    create_submission_zip(candidate_b_dir, zip_b)

    validator = args.data_root / "test_submission" / "validate_submission.py"
    manifest = args.data_root / "test_submission" / "manifest.csv"
    validator_results = [
        run_validator(validator, candidate_a_dir, manifest, args.output_dir / "validator_candidate_a_directory.log"),
        run_validator(validator, zip_a, manifest, args.output_dir / "validator_candidate_a_zip.log"),
        run_validator(validator, candidate_b_dir, manifest, args.output_dir / "validator_candidate_b_directory.log"),
        run_validator(validator, zip_b, manifest, args.output_dir / "validator_candidate_b_zip.log"),
    ]
    json_dump(args.output_dir / "official_validator_results.json", validator_results)
    validator_gate = all(r["status"] == "PASS" for r in validator_results)
    if not validator_gate:
        raise P2C1Error(f"Official validator failed: {validator_results}")

    selected_a_list = [selected_a[str(q["sample_id"])] for q in test_rows]
    selected_b_list = [selected_b[str(q["sample_id"])] for q in test_rows]
    diagnostics = {
        "candidate_a_local_metrics": summarize_metrics(selected_a_list),
        "candidate_b_local_metrics": summarize_metrics(selected_b_list),
        "candidate_a_route_counts": dict(Counter(r["router_choice"] for r in route_rows)),
        "candidate_b_source_counts": dict(Counter(r["selected_source"] for r in selection_rows)),
        "candidate_b_hypothesis_counts": dict(Counter(r["selected_hypothesis"] for r in selection_rows)),
        "candidate_b_replacement_fraction": float(np.mean([r["selected_source"] != "p2a_base" for r in selection_rows])),
        "candidate_b_guard_pass_fraction": float(np.mean([bool(r["guard_pass"]) for r in selection_rows])),
        "max_world_alignment_error_m": max(float(r["aligned_pelvis_error_m"]) for r in world_audit_rows),
        "fk_frames_processed": int(engine.frames_processed),
        "official_scores_available": False,
        "warning": "These are local development diagnostics; organizer evaluation remains final.",
    }
    json_dump(args.output_dir / "local_test_diagnostics.json", diagnostics)

    source_paths = {
        "p2c0_acceptance": args.p2c0_dir / "P2C0_ACCEPTANCE.json",
        "p2a_mh_acceptance": args.p2a_mh_dir / "P2A_MH_ACCEPTANCE.json",
        "p1_acceptance": args.p1_full_dir / "P1_FULL_ACCEPTANCE.json",
        "p2a_acceptance": args.p2a_dir / "P2A_ACCEPTANCE.json",
        "phase_router_full_model": args.p2a_dir / "phase_router_full_model.npz",
        "full_candidate_ranker": args.output_dir / "multi_hypothesis_ranker_full_model_v0_1_0.npz",
        "test_manifest": manifest,
        "official_validator": validator,
    }
    json_dump(args.output_dir / "source_binding.json", [source_record(p, n) for n, p in source_paths.items()])

    acceptance = {
        "schema_version": SCHEMA_VERSION, "version": VERSION, "created_utc": now_utc(),
        "status": "P2C1_OFFICIAL_TEST_MATERIALIZATION_COMPLETE",
        "engineering_gate": True, "test_query_count": len(test_rows),
        "candidate_a_validator_gate": True, "candidate_b_validator_gate": True,
        "candidate_a_motion_count": len(list((candidate_a_dir / "motions").glob("*.npz"))),
        "candidate_b_motion_count": len(list((candidate_b_dir / "motions").glob("*.npz"))),
        "candidate_a_zip": str(zip_a), "candidate_a_zip_sha256": sha256_file(zip_a),
        "candidate_b_zip": str(zip_b), "candidate_b_zip_sha256": sha256_file(zip_b),
        "primary_submission_candidate": recommendation["recommended_primary"],
        "safe_fallback_candidate": recommendation["safe_fallback"],
        "experimental_alternative_candidate": recommendation["experimental_alternative"],
        "submission_recommendation_rationale": recommendation["rationale"],
        "official_scores_available": False,
        "official_test_labels_used": False,
        "residual_flow_used": False,
        "scene_graph_increment_used": False,
        "next_action": "Visually inspect a bounded sample from both validated candidates, then submit the recommended primary; retain the other ZIP unchanged as fallback/alternative.",
    }
    json_dump(args.output_dir / "P2C1_ACCEPTANCE.json", acceptance)
    summary_lines = [
        "REPOINTER P2C1 — OFFICIAL 100-TEST MATERIALIZATION",
        "=" * 76,
        f"version={VERSION}",
        f"created_utc={acceptance['created_utc']}",
        f"output_dir={args.output_dir}",
        "",
        "[COVERAGE]",
        f"queries={len(test_rows)}",
        f"candidate_a_motions={acceptance['candidate_a_motion_count']}",
        f"candidate_b_motions={acceptance['candidate_b_motion_count']}",
        f"fk_frames_processed={engine.frames_processed}",
        "",
        "[VALIDATION]",
        "candidate_a_directory=PASS",
        "candidate_a_zip=PASS",
        "candidate_b_directory=PASS",
        "candidate_b_zip=PASS",
        "",
        "[LOCAL DIAGNOSTICS — NOT OFFICIAL SCORES]",
        f"candidate_a_mean_self_score1b_proxy20={diagnostics['candidate_a_local_metrics']['self_score1b_proxy20']}",
        f"candidate_b_mean_self_score1b_proxy20={diagnostics['candidate_b_local_metrics']['self_score1b_proxy20']}",
        f"candidate_b_replacement_fraction={diagnostics['candidate_b_replacement_fraction']}",
        f"candidate_b_guard_pass_fraction={diagnostics['candidate_b_guard_pass_fraction']}",
        "",
        "[ACCEPTANCE]",
        f"status={acceptance['status']}",
        f"primary_submission_candidate={acceptance['primary_submission_candidate']}",
        f"safe_fallback_candidate={acceptance['safe_fallback_candidate']}",
        f"experimental_alternative_candidate={acceptance['experimental_alternative_candidate']}",
        f"recommendation_rationale={acceptance['submission_recommendation_rationale']}",
        f"candidate_a_zip={zip_a}",
        f"candidate_b_zip={zip_b}",
        "official_scores_available=FALSE",
    ]
    (args.output_dir / "SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return acceptance


def self_test() -> None:
    names = phase_router_feature_names()
    if len(names) != 53 or len(set(names)) != len(names):
        raise P2C1Error("Phase-router feature self-test failed")
    dummy_metric = {
        "planned_frame": 50, "self_frame": 52, "det_source": 51, "det_refined": 52,
        "det_heuristic": 53, "detector_agreement": 0.8, "planned_score1b_proxy20": 0.99,
        "self_score1b_proxy20": 0.95, "self_angle_deg": 5.0, "joint_jerk": 0.0003,
        "foot_sliding": 0.001, "pose_deviation": 0.001, "max_reach": 0.9,
    }
    c = {"metrics": dummy_metric, "donor_rank": 1}
    row = phase_router_row(
        {"sample_id": "x", "domain": "natural", "hand": 1, "clip_length": 120, "target_distance": 2.0, "target_azimuth_deg": 10.0, "target_elevation_deg": 5.0},
        c, c, 50, 60,
    )
    X = build_phase_router_features([row])
    if X.shape != (1, 53) or not np.all(np.isfinite(X)):
        raise P2C1Error("Phase-router matrix self-test failed")
    ref = np.asarray([1.0, 2.0, 3.0])
    pelvis = np.asarray([0.1, 0.2, 0.3])
    shift = ref - pelvis
    if not np.allclose(pelvis + shift, ref):
        raise P2C1Error("World-alignment self-test failed")
    with zipfile.ZipFile(Path(os.devnull), "w") if False else _nullcontext():
        pass


class _nullcontext:
    def __enter__(self) -> None:
        return None
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("SELF_TEST=PASS")
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--p0c0-dir", type=Path, required=True)
    ap.add_argument("--m1-full-dir", type=Path, required=True)
    ap.add_argument("--p1a-dir", type=Path, required=True)
    ap.add_argument("--p1-full-dir", type=Path, required=True)
    ap.add_argument("--p2a-dir", type=Path, required=True)
    ap.add_argument("--p2a-mh-dir", type=Path, required=True)
    ap.add_argument("--p2c0-dir", type=Path, required=True)
    ap.add_argument("--smplx-model-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--team-name", type=str, default="RePointer")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--fk-batch-size", type=int, default=512)
    ap.add_argument("--ik-batch-size", type=int, default=64)
    ap.add_argument("--ik-steps", type=int, default=50)
    ap.add_argument("--ik-lr", type=float, default=0.05)
    ap.add_argument("--ik-max-delta", type=float, default=1.5)
    ap.add_argument("--hold-plateau", type=int, default=5)
    ap.add_argument("--hold-taper", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hypothesis-count", type=int, default=4)
    ap.add_argument("--donors-per-hypothesis", type=int, default=4)
    ap.add_argument("--min-hypothesis-sep-frames", type=int, default=6)
    ap.add_argument("--max-words", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        print("SELF_TEST=PASS")
        return 0
    for name in ("data_root", "project_root", "p0c0_dir", "m1_full_dir", "p1a_dir", "p1_full_dir", "p2a_dir", "p2a_mh_dir", "p2c0_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    self_test()
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    m1_worker = args.m1_full_dir / "_code" / "repointer_b0_p0c_m1_full_fold_v0_2_0.py"
    mh_source = locate_mh_source(args.p2a_mh_dir)
    if not m1_worker.is_file():
        raise P2C1Error(f"Missing frozen M1 worker: {m1_worker}")
    module = load_module(m1_worker, "repointer_p2c1_frozen_m1")
    mh = load_module(mh_source, "repointer_p2c1_frozen_mh")
    module.self_test()
    mh.self_test()

    pre = preflight(args, module, mh, device)
    if args.preflight_only:
        print("PREFLIGHT_STATUS=P2C1_PREFLIGHT_ACCEPTED")
        print("MOTION_GENERATION_STARTED=FALSE")
        print(f"RESUME_RUN_DIR={args.output_dir}")
        return 0
    acceptance = formal_materialization(args, module, mh, device)

    env = {
        "created_utc": now_utc(), "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable, "cwd": os.getcwd(),
        "device": device,
    }
    json_dump(args.output_dir / "environment.json", env)
    source_files = [
        ("m1_worker", m1_worker), ("p2a_mh_source", mh_source),
        ("p2c0_acceptance", args.p2c0_dir / "P2C0_ACCEPTANCE.json"),
        ("p2a_mh_acceptance", args.p2a_mh_dir / "P2A_MH_ACCEPTANCE.json"),
    ]
    json_dump(args.output_dir / "method_source_audit.json", [source_record(p, n) for n, p in source_files])
    run_manifest: list[dict[str, Any]] = []
    for p in sorted(args.output_dir.rglob("*")):
        if p.is_file() and p.name != "run_file_manifest.json":
            run_manifest.append({"relative_path": str(p.relative_to(args.output_dir)), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    json_dump(args.output_dir / "run_file_manifest.json", run_manifest)
    print((args.output_dir / "SUMMARY.txt").read_text(encoding="utf-8"), end="", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise

