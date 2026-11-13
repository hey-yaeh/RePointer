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
import re
import shutil
import socket
import subprocess
import sys
import traceback
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

VERSION = "0.1.0"
SCHEMA_VERSION = "1.0.0"
EXPECTED_TEST_COUNT = 100
FPS = 30.0


class P2C2Error(RuntimeError):
    pass


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def fnum(value: Any, default: float = float("nan")) -> float:
    if is_blank(value):
        return default
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def inum(value: Any, default: int = -1) -> int:
    out = fnum(value)
    return int(round(out)) if np.isfinite(out) else default


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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        handle.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise P2C2Error(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sanitize_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return value.strip("._-") or "RePointer"


def parse_vector(value: Any, length: int = 3) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    else:
        text = str(value).strip()
        try:
            arr = np.asarray(json.loads(text), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise P2C2Error(f"Could not parse vector {value!r}: {exc}") from exc
    if arr.size != length or not np.all(np.isfinite(arr)):
        raise P2C2Error(f"Invalid vector: {value!r}")
    return arr


def locate_m1_worker(m1_full_dir: Path) -> Path:
    candidates = [
        m1_full_dir / "_code" / "repointer_b0_p0c_m1_full_fold_v0_2_0.py",
        m1_full_dir / "repointer_b0_p0c_m1_full_fold_v0_2_0.py",
        Path.home() / "Downloads" / "repointer_b0_p0c_m1_full_fold_v0_2_0.py",
    ]
    hits = [path.resolve() for path in candidates if path.is_file()]
    if not hits:
        raise P2C2Error(f"Could not locate frozen M1 worker under {m1_full_dir}")
    return hits[0]


def load_query_beta(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row.get("body_shape_path", "")))
    if not path.is_file():
        raise P2C2Error(f"Missing body shape: {path}")
    beta = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    if beta.size < 10 or not np.all(np.isfinite(beta[:10])):
        raise P2C2Error(f"Invalid body shape: {path}")
    return beta[:10].copy()


def load_motion(path: Path, expected_length: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise P2C2Error(f"Missing motion: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"poses", "trans"}:
            raise P2C2Error(f"Unexpected NPZ keys in {path}: {archive.files}")
        poses = np.asarray(archive["poses"], dtype=np.float32)
        trans = np.asarray(archive["trans"], dtype=np.float32)
    if poses.shape != (expected_length, 165) or trans.shape != (expected_length, 3):
        raise P2C2Error(f"Motion shape mismatch for {path.name}: poses={poses.shape} trans={trans.shape} expected={expected_length}")
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(trans)):
        raise P2C2Error(f"Non-finite motion values: {path}")
    return poses, trans


def query_scene_info(row: Mapping[str, Any]) -> dict[str, Any] | None:
    center_text = str(row.get("matched_bbox_center", "")).strip()
    size_text = str(row.get("matched_bbox_size", "")).strip()
    if not center_text or not size_text:
        return None
    center = parse_vector(center_text)
    size = parse_vector(size_text)
    reference = np.asarray([
        fnum(row.get("reference_pelvis_world_x")),
        fnum(row.get("reference_pelvis_world_y")),
        fnum(row.get("reference_pelvis_world_z")),
    ], dtype=np.float64)
    if not np.all(np.isfinite(reference)):
        return None
    return {
        "center_rel": center - reference,
        "center_world": center,
        "half_diag": 0.5 * float(np.linalg.norm(size)),
    }


def target_world(row: Mapping[str, Any]) -> np.ndarray:
    arr = np.asarray([
        fnum(row.get("target_world_x")),
        fnum(row.get("target_world_y")),
        fnum(row.get("target_world_z")),
    ], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise P2C2Error(f"Invalid target_world for {row.get('sample_id')}")
    return arr


def angle_to_fixed_target(
    joints: np.ndarray,
    hand: int,
    frame: int,
    module: Any,
    center_world: np.ndarray,
    half_diag: float,
) -> float:
    frame = int(np.clip(frame, 0, len(joints) - 1))
    hand_name = "left" if hand == 0 else "right"
    elbow = np.asarray(joints[frame, module.JOINT_INDEX[f"{hand_name}_elbow"]], dtype=np.float64)
    wrist = np.asarray(joints[frame, module.JOINT_INDEX[f"{hand_name}_wrist"]], dtype=np.float64)
    forearm = wrist - elbow
    target_vector = np.asarray(center_world, dtype=np.float64) - elbow
    denominator = max(float(np.linalg.norm(forearm) * np.linalg.norm(target_vector)), 1e-8)
    theta = math.degrees(math.acos(float(np.clip(np.dot(forearm, target_vector) / denominator, -1.0, 1.0))))
    margin = 0.0
    if half_diag > 0:
        margin = math.degrees(math.asin(np.clip(float(half_diag) / max(float(np.linalg.norm(target_vector)), 1e-8), 0.0, 1.0)))
    return max(0.0, theta - margin)


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


def central_gradient(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values)
    return np.gradient(values, axis=0, edge_order=1)


def generic_motion_quality(joints: np.ndarray, poses: np.ndarray, module: Any) -> dict[str, float]:
    velocity = central_gradient(joints) / FPS
    acceleration = central_gradient(velocity) / FPS
    jerk = central_gradient(acceleration) / FPS
    joint_jerk = float(np.mean(np.linalg.norm(jerk[:, :22], axis=2)))
    feet = np.asarray(joints[:, [module.JOINT_INDEX["left_foot"], module.JOINT_INDEX["right_foot"]]], dtype=np.float64)
    foot_velocity = central_gradient(feet) / FPS
    min_y = np.min(feet[:, :, 1], axis=0)
    contact = feet[:, :, 1] <= (min_y[None, :] + 0.05)
    horizontal = np.linalg.norm(foot_velocity[:, :, [0, 2]], axis=2)
    foot_sliding = float(np.sum(horizontal * contact) / max(1, int(np.sum(contact))))
    aa = np.asarray(poses[:, 3:66], dtype=np.float64).reshape(len(poses), -1, 3)
    joint_limit_fraction = float(np.mean(np.linalg.norm(aa, axis=2) > math.pi))
    return {
        "generic_joint_jerk": joint_jerk,
        "generic_foot_sliding": foot_sliding,
        "joint_limit_fraction": joint_limit_fraction,
    }


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


def evaluate_motion(
    label: str,
    row: Mapping[str, Any],
    poses: np.ndarray,
    joints: np.ndarray,
    planned_frame: int,
    module: Any,
    source_model: Mapping[str, np.ndarray],
    refined_model: Mapping[str, np.ndarray],
    planned_radius: int,
    detector_radius: int,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    hand = inum(row.get("hand"))
    T = inum(row.get("clip_length"))
    planned_frame = int(np.clip(planned_frame, 0, T - 1))
    detector = module.detector_predictions(joints, hand, source_model, refined_model)
    detector_frames = [int(detector[key]) for key in ("source", "refined", "heuristic", "ensemble")]
    planned_frames = clipped_window(planned_frame, planned_radius, T)
    robust_frames = set(planned_frames)
    for frame in detector_frames:
        robust_frames.update(clipped_window(frame, detector_radius, T))
    robust_frames_sorted = sorted(robust_frames)
    scene = query_scene_info(row)
    world_target = target_world(row)
    resolution_mode = str(row.get("target_resolution_mode", ""))

    frame_rows: list[dict[str, Any]] = []
    combined_scores: list[float] = []
    relative_scores: list[float] = []
    world_scores: list[float] = []
    point_world_scores: list[float] = []
    hand_name = "left" if hand == 0 else "right"
    forearm_vectors: list[np.ndarray] = []
    for frame in robust_frames_sorted:
        relative_angle = module.spatial_angle(joints, hand, frame, row, scene)
        relative_score = float(module.score20(relative_angle))
        if scene is not None:
            world_angle = angle_to_fixed_target(joints, hand, frame, module, scene["center_world"], float(scene["half_diag"]))
        else:
            world_angle = angle_to_fixed_target(joints, hand, frame, module, world_target, 0.0)
        world_score = float(module.score20(world_angle))
        point_world_angle = angle_to_fixed_target(joints, hand, frame, module, world_target, 0.0)
        point_world_score = float(module.score20(point_world_angle))
        if resolution_mode == "POINT_INSIDE_AABB" and scene is not None:
            combined = min(relative_score, world_score)
        else:
            combined = min(relative_score, world_score, point_world_score)
        relative_scores.append(relative_score)
        world_scores.append(world_score)
        point_world_scores.append(point_world_score)
        combined_scores.append(combined)
        elbow = np.asarray(joints[frame, module.JOINT_INDEX[f"{hand_name}_elbow"]], dtype=np.float64)
        wrist = np.asarray(joints[frame, module.JOINT_INDEX[f"{hand_name}_wrist"]], dtype=np.float64)
        forearm_vectors.append(wrist - elbow)
        frame_rows.append({
            "sample_id": sample_id,
            "candidate": label,
            "frame": frame,
            "is_planned_window": frame in planned_frames,
            "is_detector_frame": frame in detector_frames,
            "relative_score1b": relative_score,
            "world_bbox_score1b": world_score,
            "world_point_score1b": point_world_score,
            "combined_robust_score1b": combined,
        })

    frame_to_score = {int(r["frame"]): float(r["combined_robust_score1b"]) for r in frame_rows}
    detector_scores = [frame_to_score[int(np.clip(frame, 0, T - 1))] for frame in detector_frames]
    planned_scores = [frame_to_score[frame] for frame in planned_frames]
    combined_summary = score_summary(combined_scores)
    detector_summary = score_summary(detector_scores)
    planned_summary = score_summary(planned_scores)
    relative_summary = score_summary(relative_scores)
    world_summary = score_summary(world_scores)
    point_summary = score_summary(point_world_scores)
    direction_dispersion = angular_dispersion_deg(np.stack(forearm_vectors))
    detector_spread = float(np.std(np.asarray(detector_frames, dtype=np.float64)))
    quality = generic_motion_quality(joints, poses, module)

    # Spatial-first utility. It intentionally emphasizes worst-case behaviour over
    # multiple plausible apex frames rather than one locally optimal frame.
    direction_stability = math.exp(-(direction_dispersion ** 2) / (2.0 * 6.0 ** 2))
    frame_agreement = float(detector["agreement"])
    robust_utility = (
        0.34 * detector_summary["min"]
        + 0.26 * combined_summary["q10"]
        + 0.18 * planned_summary["min"]
        + 0.10 * detector_summary["mean"]
        + 0.06 * combined_summary["mean"]
        + 0.04 * frame_agreement
        + 0.02 * direction_stability
    )

    ensemble_frame = int(detector["ensemble"])
    ensemble_score = frame_to_score.get(ensemble_frame)
    if ensemble_score is None:
        relative_angle = module.spatial_angle(joints, hand, ensemble_frame, row, scene)
        ensemble_score = float(module.score20(relative_angle))

    metrics: dict[str, Any] = {
        "sample_id": sample_id,
        "candidate": label,
        "clip_length": T,
        "hand": hand,
        "planned_frame": planned_frame,
        "det_source": detector_frames[0],
        "det_refined": detector_frames[1],
        "det_heuristic": detector_frames[2],
        "det_ensemble": detector_frames[3],
        "detector_agreement": frame_agreement,
        "detector_spread_frames": detector_spread,
        "ensemble_robust_score1b": float(ensemble_score),
        "detector_floor_score1b": detector_summary["min"],
        "detector_mean_score1b": detector_summary["mean"],
        "planned_floor_score1b": planned_summary["min"],
        "planned_mean_score1b": planned_summary["mean"],
        "window_floor_score1b": combined_summary["min"],
        "window_q10_score1b": combined_summary["q10"],
        "window_mean_score1b": combined_summary["mean"],
        "relative_window_q10_score1b": relative_summary["q10"],
        "world_bbox_window_q10_score1b": world_summary["q10"],
        "world_point_window_q10_score1b": point_summary["q10"],
        "forearm_direction_dispersion_deg": direction_dispersion,
        "direction_stability": direction_stability,
        "robust_spatial_utility": float(robust_utility),
        "target_resolution_mode": resolution_mode,
        "robust_frame_count": len(robust_frames_sorted),
        **quality,
    }
    return {"metrics": metrics, "frame_rows": frame_rows}


def choose_candidate(a: Mapping[str, Any], b: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    # Candidate A is the explicit safe fallback. Candidate B replaces it only when
    # the detector-robust spatial utility improves materially and hard guards pass.
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


def create_submission_zip(submission_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(submission_dir / "metadata.json", arcname="metadata.json")
        for path in sorted((submission_dir / "motions").glob("*.npz")):
            archive.write(path, arcname=f"motions/{path.name}")


def run_validator(validator: Path, submission: Path, manifest: Path, log_path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(validator), str(submission), "--manifest", str(manifest)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    text = process.stdout or ""
    log_path.write_text(text, encoding="utf-8")
    return {
        "submission": str(submission),
        "returncode": process.returncode,
        "output": text.strip(),
        "status": "PASS" if process.returncode == 0 else "FAIL",
    }


def source_record(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise P2C2Error(f"Missing source artifact {name}: {path}")
    return {"name": name, "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def summarize(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, float]:
    keys = [
        "ensemble_robust_score1b",
        "detector_floor_score1b",
        "window_q10_score1b",
        "planned_floor_score1b",
        "robust_spatial_utility",
        "detector_agreement",
        "forearm_direction_dispersion_deg",
        "generic_joint_jerk",
        "generic_foot_sliding",
    ]
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([fnum(row.get(key)) for row in rows], dtype=np.float64)
        if np.all(np.isfinite(values)):
            out[f"{prefix}_{key}"] = float(np.mean(values))
    return out


def preflight(args: argparse.Namespace, module: Any) -> dict[str, Any]:
    acceptance_path = args.p2c1_dir / "P2C1_ACCEPTANCE.json"
    acceptance = json_load(acceptance_path)
    if acceptance.get("status") != "P2C1_OFFICIAL_TEST_MATERIALIZATION_COMPLETE":
        raise P2C2Error(f"P2C1 is not complete: {acceptance.get('status')}")
    test_rows = read_csv(args.p2c1_dir / "test_generation_manifest.csv")
    if len(test_rows) != EXPECTED_TEST_COUNT or len({str(row["sample_id"]) for row in test_rows}) != EXPECTED_TEST_COUNT:
        raise P2C2Error("P2C1 test manifest coverage mismatch")
    candidate_a_dir = args.p2c1_dir / "submissions" / "candidate_a_p2a"
    candidate_b_dir = args.p2c1_dir / "submissions" / "candidate_b_p2a_mh"
    expected_ids = {str(row["sample_id"]) for row in test_rows}
    for label, root in (("A", candidate_a_dir), ("B", candidate_b_dir)):
        ids = {path.stem for path in (root / "motions").glob("*.npz")}
        if ids != expected_ids:
            raise P2C2Error(f"Candidate {label} coverage mismatch: {len(ids)}")
    world_rows = read_csv(args.p2c1_dir / "world_alignment_audit.csv")
    plans = {(str(row.get("candidate")), str(row.get("sample_id"))): inum(row.get("planned_frame")) for row in world_rows}
    for sid in expected_ids:
        if ("candidate_a_p2a", sid) not in plans or ("candidate_b_p2a_mh", sid) not in plans:
            raise P2C2Error(f"Missing planned-frame audit for {sid}")
    validator = args.data_root / "test_submission" / "validate_submission.py"
    manifest = args.data_root / "test_submission" / "manifest.csv"
    if not validator.is_file() or not manifest.is_file():
        raise P2C2Error("Official validator or manifest is missing")
    assets = module.find_smplx_assets(args.smplx_model_dir, args.data_root, args.project_root)
    if assets.get("status") != "OK":
        raise P2C2Error(f"SMPL-X assets missing: {assets}")
    source_model = args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_0_0.npz"
    refined_model = args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_1_0.npz"
    if not source_model.is_file() or not refined_model.is_file():
        raise P2C2Error("Frozen detector models are missing")
    pre = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_utc": now_utc(),
        "status": "P2C2_PREFLIGHT_ACCEPTED",
        "p2c1_dir": str(args.p2c1_dir),
        "query_count": len(test_rows),
        "candidate_a_motion_count": len(list((candidate_a_dir / "motions").glob("*.npz"))),
        "candidate_b_motion_count": len(list((candidate_b_dir / "motions").glob("*.npz"))),
        "selection_policy": "A-safe fallback; B only on material detector-robust spatial gain under hard guards",
        "planned_radius": args.planned_radius,
        "detector_radius": args.detector_radius,
        "replacement_threshold": args.replacement_threshold,
        "smplx_assets": assets,
        "motion_selection_started": False,
        "official_test_labels_used": False,
    }
    json_dump(args.output_dir / "PREFLIGHT.json", pre)
    return pre


def formal_run(args: argparse.Namespace, module: Any) -> dict[str, Any]:
    completion_path = args.output_dir / "P2C2_ACCEPTANCE.json"
    if completion_path.is_file():
        completion = json_load(completion_path)
        if completion.get("status") == "P2C2_CANDIDATE_C_MATERIALIZATION_COMPLETE":
            print("[RESUME] Candidate C already complete; no recomputation performed", flush=True)
            return completion
    pre_path = args.output_dir / "PREFLIGHT.json"
    if not pre_path.is_file() or json_load(pre_path).get("status") != "P2C2_PREFLIGHT_ACCEPTED":
        preflight(args, module)

    test_rows = read_csv(args.p2c1_dir / "test_generation_manifest.csv")
    test_by_id = {str(row["sample_id"]): row for row in test_rows}
    world_rows = read_csv(args.p2c1_dir / "world_alignment_audit.csv")
    planned = {(str(row.get("candidate")), str(row.get("sample_id"))): inum(row.get("planned_frame")) for row in world_rows}
    source_model = module.load_linear_model(args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_0_0.npz")
    refined_model = module.load_linear_model(args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_1_0.npz")
    assets = module.find_smplx_assets(args.smplx_model_dir, args.data_root, args.project_root)
    engine = module.SMPLXEngine(assets, args.device, args.fk_batch_size)

    labels_and_roots = [
        ("A", "candidate_a_p2a", args.p2c1_dir / "submissions" / "candidate_a_p2a" / "motions"),
        ("B", "candidate_b_p2a_mh", args.p2c1_dir / "submissions" / "candidate_b_p2a_mh" / "motions"),
    ]
    motion_data: dict[tuple[str, str], dict[str, Any]] = {}
    clips: list[dict[str, Any]] = []
    clip_keys: list[tuple[str, str]] = []
    for row in test_rows:
        sid = str(row["sample_id"])
        T = inum(row.get("clip_length"))
        beta = load_query_beta(row)
        gender = module.normalize_gender(row.get("gender"))
        for short_label, long_label, root in labels_and_roots:
            poses, trans = load_motion(root / f"{sid}.npz", T)
            motion_data[(short_label, sid)] = {"poses": poses, "trans": trans, "gender": gender, "beta": beta, "long_label": long_label}
            clips.append({
                "pose": poses,
                "trans": trans,
                "betas": np.broadcast_to(beta[None, :], (T, len(beta))).copy(),
                "gender": gender,
            })
            clip_keys.append((short_label, sid))
    joints_many = engine.joints_many(clips)
    joints_by_key = {key: joints for key, joints in zip(clip_keys, joints_many)}

    pair_metric_rows: list[dict[str, Any]] = []
    frame_metric_rows: list[dict[str, Any]] = []
    metrics_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(test_rows):
        sid = str(row["sample_id"])
        for short_label, long_label, _ in labels_and_roots:
            data = motion_data[(short_label, sid)]
            result = evaluate_motion(
                short_label,
                row,
                data["poses"],
                joints_by_key[(short_label, sid)],
                planned[(long_label, sid)],
                module,
                source_model,
                refined_model,
                args.planned_radius,
                args.detector_radius,
            )
            metrics_by_key[(short_label, sid)] = result["metrics"]
            pair_metric_rows.append(result["metrics"])
            frame_metric_rows.extend(result["frame_rows"])
        if (index + 1) % 20 == 0:
            print(f"[Candidate C robust evaluation] {index+1}/{len(test_rows)}", flush=True)

    selection_rows: list[dict[str, Any]] = []
    selected_source: dict[str, str] = {}
    for row in test_rows:
        sid = str(row["sample_id"])
        a = metrics_by_key[("A", sid)]
        b = metrics_by_key[("B", sid)]
        decision = choose_candidate(a, b, args.replacement_threshold)
        choice = str(decision["choice"])
        selected_source[sid] = choice
        chosen = b if choice == "B" else a
        selection_rows.append({
            "sample_id": sid,
            "selected_source": choice,
            "guard_pass_for_b": decision["guard_pass"],
            "failed_b_guards": json.dumps(decision["failed_guards"], ensure_ascii=False),
            "material_gain_for_b": decision["material_gain"],
            "replacement_threshold": args.replacement_threshold,
            "utility_gain_b_minus_a": decision["utility_gain_b_minus_a"],
            "detector_floor_gain_b_minus_a": decision["detector_floor_gain_b_minus_a"],
            "window_q10_gain_b_minus_a": decision["window_q10_gain_b_minus_a"],
            "a_robust_spatial_utility": a["robust_spatial_utility"],
            "b_robust_spatial_utility": b["robust_spatial_utility"],
            "c_robust_spatial_utility": chosen["robust_spatial_utility"],
            "a_detector_floor_score1b": a["detector_floor_score1b"],
            "b_detector_floor_score1b": b["detector_floor_score1b"],
            "c_detector_floor_score1b": chosen["detector_floor_score1b"],
            "a_window_q10_score1b": a["window_q10_score1b"],
            "b_window_q10_score1b": b["window_q10_score1b"],
            "c_window_q10_score1b": chosen["window_q10_score1b"],
            "a_ensemble_score1b": a["ensemble_robust_score1b"],
            "b_ensemble_score1b": b["ensemble_robust_score1b"],
            "c_ensemble_score1b": chosen["ensemble_robust_score1b"],
            "target_resolution_mode": row.get("target_resolution_mode", ""),
        })

    submission_dir = args.output_dir / "submissions" / "candidate_c_spatial_first_detector_robust"
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    motions_dir = submission_dir / "motions"
    motions_dir.mkdir(parents=True)
    copy_rows: list[dict[str, Any]] = []
    for row in test_rows:
        sid = str(row["sample_id"])
        choice = selected_source[sid]
        source_dir = args.p2c1_dir / "submissions" / ("candidate_b_p2a_mh" if choice == "B" else "candidate_a_p2a") / "motions"
        source = source_dir / f"{sid}.npz"
        destination = motions_dir / source.name
        shutil.copy2(source, destination)
        copy_rows.append({
            "sample_id": sid,
            "selected_source": choice,
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "output_path": str(destination),
            "output_sha256": sha256_file(destination),
            "status": "PASS" if sha256_file(source) == sha256_file(destination) else "FAIL",
        })
    if any(row["status"] != "PASS" for row in copy_rows):
        raise P2C2Error("Candidate C copy integrity failed")

    metadata = {
        "team_name": args.team_name,
        "system": "RePointer-P2A-MH-SFDR",
        "version": VERSION,
        "description": "Spatial-first detector-robust A/B hybrid. Candidate A is the safe fallback; Candidate B is selected only under material multi-frame spatial gain and hard motion/detector guards.",
        "official_test_labels_used": False,
    }
    json_dump(submission_dir / "metadata.json", metadata)
    zip_root = args.output_dir / "submission_zips"
    team = sanitize_name(args.team_name)
    zip_path = zip_root / f"{team}_P2A_MH_SFDR_Candidate_C_submission.zip"
    create_submission_zip(submission_dir, zip_path)

    validator = args.data_root / "test_submission" / "validate_submission.py"
    manifest = args.data_root / "test_submission" / "manifest.csv"
    validator_results = [
        run_validator(validator, submission_dir, manifest, args.output_dir / "validator_candidate_c_directory.log"),
        run_validator(validator, zip_path, manifest, args.output_dir / "validator_candidate_c_zip.log"),
    ]
    json_dump(args.output_dir / "official_validator_results.json", validator_results)
    validator_gate = all(row["status"] == "PASS" for row in validator_results)
    if not validator_gate:
        raise P2C2Error(f"Candidate C validator failed: {validator_results}")

    write_csv(args.output_dir / "candidate_c_pair_metrics.csv", pair_metric_rows)
    write_csv(args.output_dir / "candidate_c_frame_metrics.csv", frame_metric_rows)
    write_csv(args.output_dir / "candidate_c_selection_manifest.csv", selection_rows)
    write_csv(args.output_dir / "candidate_c_copy_integrity.csv", copy_rows)

    a_rows = [metrics_by_key[("A", str(row["sample_id"]))] for row in test_rows]
    b_rows = [metrics_by_key[("B", str(row["sample_id"]))] for row in test_rows]
    c_rows = [metrics_by_key[(selected_source[str(row["sample_id"])], str(row["sample_id"]))] for row in test_rows]
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_utc": now_utc(),
        "candidate_a": summarize(a_rows, "a"),
        "candidate_b": summarize(b_rows, "b"),
        "candidate_c": summarize(c_rows, "c"),
        "candidate_c_source_counts": dict(Counter(selected_source.values())),
        "candidate_c_b_fraction": float(np.mean([value == "B" for value in selected_source.values()])),
        "guard_pass_fraction_for_b": float(np.mean([bool(row["guard_pass_for_b"]) for row in selection_rows])),
        "mean_utility_gain_c_minus_a": float(np.mean([
            fnum(row["c_robust_spatial_utility"]) - fnum(row["a_robust_spatial_utility"]) for row in selection_rows
        ])),
        "mean_utility_gain_c_minus_b": float(np.mean([
            fnum(row["c_robust_spatial_utility"]) - fnum(row["b_robust_spatial_utility"]) for row in selection_rows
        ])),
        "fk_frames_processed": int(engine.frames_processed),
        "official_scores_available": False,
        "warning": "All quantities are local detector/spatial robustness diagnostics. Organizer evaluation remains final.",
    }
    json_dump(args.output_dir / "local_robust_diagnostics.json", diagnostics)

    source_files = [
        source_record(args.p2c1_dir / "P2C1_ACCEPTANCE.json", "P2C1_ACCEPTANCE"),
        source_record(args.p2c1_dir / "submissions" / "candidate_a_p2a" / "metadata.json", "candidate_a_metadata"),
        source_record(args.p2c1_dir / "submissions" / "candidate_b_p2a_mh" / "metadata.json", "candidate_b_metadata"),
        source_record(args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_0_0.npz", "source_apex_detector"),
        source_record(args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_1_0.npz", "refined_apex_detector"),
        source_record(args.m1_full_dir / "_code" / "repointer_b0_p0c_m1_full_fold_v0_2_0.py", "frozen_m1_worker"),
        source_record(manifest, "official_test_manifest"),
        source_record(validator, "official_validator"),
    ]
    json_dump(args.output_dir / "source_binding.json", source_files)

    c_utility = diagnostics["candidate_c"].get("c_robust_spatial_utility", float("nan"))
    a_utility = diagnostics["candidate_a"].get("a_robust_spatial_utility", float("nan"))
    c_vs_a_gate = bool(np.isfinite(c_utility) and np.isfinite(a_utility) and c_utility >= a_utility - 1e-12)
    acceptance = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_utc": now_utc(),
        "status": "P2C2_CANDIDATE_C_MATERIALIZATION_COMPLETE",
        "engineering_gate": True,
        "validator_gate": validator_gate,
        "query_count": len(test_rows),
        "candidate_c_motion_count": len(list(motions_dir.glob("*.npz"))),
        "candidate_c_b_fraction": diagnostics["candidate_c_b_fraction"],
        "robust_utility_not_below_a_gate": c_vs_a_gate,
        "candidate_c_zip": str(zip_path),
        "candidate_c_zip_sha256": sha256_file(zip_path),
        "official_scores_available": False,
        "official_test_labels_used": False,
        "recommended_use": "Experimental Candidate C; retain validated A and B unchanged until organizer scores are available.",
        "next_action": "Freeze Candidate C ZIP, run the official validator once more on the frozen copy, and submit according to the organizer's allowed submission count.",
    }
    json_dump(args.output_dir / "P2C2_ACCEPTANCE.json", acceptance)
    summary = [
        "REPOINTER P2C2 — SPATIAL-FIRST DETECTOR-ROBUST CANDIDATE C",
        "=" * 78,
        f"version={VERSION}",
        f"created_utc={acceptance['created_utc']}",
        f"output_dir={args.output_dir}",
        "",
        "[COVERAGE]",
        f"queries={len(test_rows)}",
        f"candidate_c_motions={acceptance['candidate_c_motion_count']}",
        f"fk_frames_processed={diagnostics['fk_frames_processed']}",
        "",
        "[SELECTION]",
        f"selected_from_a={diagnostics['candidate_c_source_counts'].get('A', 0)}",
        f"selected_from_b={diagnostics['candidate_c_source_counts'].get('B', 0)}",
        f"candidate_c_b_fraction={diagnostics['candidate_c_b_fraction']}",
        f"guard_pass_fraction_for_b={diagnostics['guard_pass_fraction_for_b']}",
        "",
        "[LOCAL ROBUST DIAGNOSTICS — NOT OFFICIAL SCORES]",
        f"candidate_a_robust_utility={a_utility}",
        f"candidate_b_robust_utility={diagnostics['candidate_b'].get('b_robust_spatial_utility')}",
        f"candidate_c_robust_utility={c_utility}",
        f"candidate_c_minus_a_robust_utility={diagnostics['mean_utility_gain_c_minus_a']}",
        f"candidate_a_detector_floor={diagnostics['candidate_a'].get('a_detector_floor_score1b')}",
        f"candidate_b_detector_floor={diagnostics['candidate_b'].get('b_detector_floor_score1b')}",
        f"candidate_c_detector_floor={diagnostics['candidate_c'].get('c_detector_floor_score1b')}",
        "",
        "[VALIDATION]",
        "candidate_c_directory=PASS",
        "candidate_c_zip=PASS",
        "",
        "[ACCEPTANCE]",
        f"status={acceptance['status']}",
        f"robust_utility_not_below_a_gate={c_vs_a_gate}",
        f"candidate_c_zip={zip_path}",
        "official_scores_available=FALSE",
    ]
    (args.output_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return acceptance


def self_test() -> None:
    base = {
        "detector_floor_score1b": 0.90,
        "window_q10_score1b": 0.91,
        "planned_floor_score1b": 0.95,
        "world_bbox_window_q10_score1b": 0.92,
        "world_point_window_q10_score1b": 0.90,
        "forearm_direction_dispersion_deg": 2.0,
        "detector_spread_frames": 2.0,
        "generic_joint_jerk": 0.0003,
        "generic_foot_sliding": 0.001,
        "joint_limit_fraction": 0.0,
        "robust_spatial_utility": 0.92,
    }
    better = dict(base)
    better.update({"detector_floor_score1b": 0.94, "window_q10_score1b": 0.95, "robust_spatial_utility": 0.95})
    decision = choose_candidate(base, better, 0.001)
    if decision["choice"] != "B" or not decision["guard_pass"]:
        raise P2C2Error("Candidate-selection positive self-test failed")
    unsafe = dict(better)
    unsafe["planned_floor_score1b"] = 0.7
    decision = choose_candidate(base, unsafe, 0.001)
    if decision["choice"] != "A" or decision["guard_pass"]:
        raise P2C2Error("Candidate-selection guard self-test failed")
    vectors = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    if abs(angular_dispersion_deg(vectors)) > 1e-8:
        raise P2C2Error("Direction-dispersion self-test failed")
    scores = score_summary([0.8, 0.9, 1.0])
    if not (0.8 <= scores["q10"] <= 0.9):
        raise P2C2Error("Score-summary self-test failed")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("SELF_TEST=PASS")
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--p0c0-dir", type=Path, required=True)
    parser.add_argument("--m1-full-dir", type=Path, required=True)
    parser.add_argument("--p2c1-dir", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--team-name", type=str, default="RePointer")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fk-batch-size", type=int, default=512)
    parser.add_argument("--planned-radius", type=int, default=2)
    parser.add_argument("--detector-radius", type=int, default=2)
    parser.add_argument("--replacement-threshold", type=float, default=0.001)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.project_root = args.project_root.resolve()
    args.p0c0_dir = args.p0c0_dir.resolve()
    args.m1_full_dir = args.m1_full_dir.resolve()
    args.p2c1_dir = args.p2c1_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.planned_radius < 0 or args.detector_radius < 0:
        raise P2C2Error("Frame radii must be non-negative")
    m1_source = locate_m1_worker(args.m1_full_dir)
    module = load_module(m1_source, "repointer_p2c2_frozen_m1")
    if args.preflight_only:
        preflight(args, module)
        print("PREFLIGHT_STATUS=P2C2_PREFLIGHT_ACCEPTED")
        print("MOTION_SELECTION_STARTED=FALSE")
        return 0
    formal_run(args, module)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
