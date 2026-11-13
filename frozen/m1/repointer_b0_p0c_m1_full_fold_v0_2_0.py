#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

VERSION = "0.2.0"
SCHEMA_VERSION = "1.0.0"
FPS = 30.0
POSE_KEYS = ("poses", "pose", "smplx_pose", "pose_body", "rotations")
TRANS_KEYS = ("trans", "transl", "translation", "translations", "root_trans")
BETAS_KEYS = ("betas", "beta", "shape", "shape_params")
GENDER_KEYS = ("gender", "sex")
JOINT_KEYS = ("joints", "joints3d", "joints_3d", "joint_positions", "smplx_joints")
SMOOTHING_KERNEL = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
SMOOTHING_KERNEL /= SMOOTHING_KERNEL.sum()

JOINT_INDEX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_index3": 27,
    "right_index3": 42,
}


EVAL_ONLY_QUERY_KEYS = {
    "released_peak_frame", "released_peak_ratio",
    "hold_start", "hold_end", "hold_width",
    "alignment_angle_deg", "alignment_score",
}


def generation_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the query fields allowed to influence generation.

    Released apex/hold/alignment labels remain available only in the separate
    evaluation row. This makes the held-out-label boundary executable rather
    than relying solely on code review.
    """
    return {k: v for k, v in row.items() if k not in EVAL_ONLY_QUERY_KEYS}


class PilotError(RuntimeError):
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
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False, default=json_default)
        f.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
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
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    v = json.dumps(v, ensure_ascii=False, separators=(",", ":"), default=json_default)
                elif isinstance(v, float) and not np.isfinite(v):
                    v = ""
                clean[k] = v
            w.writerow(clean)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_u64(text: str, seed: int = 0) -> int:
    h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def parse_serialized(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    s = str(value).strip()
    if not s:
        return None
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(s)
        except Exception:
            pass
    return s


def normalize_gender(v: Any) -> str:
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return "neutral"
        v = v.reshape(-1)[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    s = str(v).strip().lower()
    if s in {"m", "male", "man", "0"}:
        return "male"
    if s in {"f", "female", "woman", "1"}:
        return "female"
    return "neutral"


def choose_gender(source_value: Any, annotation_value: Any) -> str:
    """Prefer an explicit source gender, but do not let a missing-source neutral mask the annotation."""
    source = normalize_gender(source_value)
    annotation = normalize_gender(annotation_value)
    if source in {"male", "female"}:
        return source
    if annotation in {"male", "female"}:
        return annotation
    return "neutral"


def choose_key(files: Sequence[str], preferred: Sequence[str], patterns: Sequence[str]) -> str | None:
    lower = {x.lower(): x for x in files}
    for k in preferred:
        if k.lower() in lower:
            return lower[k.lower()]
    for x in files:
        lx = x.lower()
        if any(p in lx for p in patterns):
            return x
    return None


def inspect_npz(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    try:
        with np.load(path, allow_pickle=False) as z:
            files = list(z.files)
            pose_key = choose_key(files, POSE_KEYS, ("pose", "rot"))
            trans_key = choose_key(files, TRANS_KEYS, ("trans", "root_pos"))
            betas_key = choose_key(files, BETAS_KEYS, ("beta", "shape"))
            gender_key = choose_key(files, GENDER_KEYS, ("gender", "sex"))
            joints_key = choose_key(files, JOINT_KEYS, ("joint",))
            n = None
            for k in (pose_key, joints_key, trans_key):
                if k:
                    a = z[k]
                    if a.ndim >= 1:
                        n = int(a.shape[0])
                        break
            return {
                "status": "OK", "path": str(path), "files": files,
                "pose_key": pose_key, "trans_key": trans_key,
                "betas_key": betas_key, "gender_key": gender_key,
                "joints_key": joints_key, "frame_count": n,
            }
    except Exception as e:
        return {"status": f"ERROR:{type(e).__name__}", "path": str(path), "error": str(e)}


def safe_metadata(path: Path, key: str | None) -> Any:
    if not key:
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            return z[key]
    except Exception:
        return None


def flatten_pose_frames(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim < 2:
        raise PilotError(f"Pose array must have time dimension, got {x.shape}")
    if x.ndim == 3 and x.shape[1:] == (55, 3):
        x = x.reshape(x.shape[0], 165)
    else:
        x = x.reshape(x.shape[0], -1)
    if x.shape[1] < 165:
        x = np.pad(x, ((0, 0), (0, 165 - x.shape[1])))
    elif x.shape[1] > 165:
        x = x[:, :165]
    return x.astype(np.float32, copy=False)


def framewise_vec3(a: np.ndarray | None, n: int) -> np.ndarray:
    if a is None:
        return np.zeros((n, 3), dtype=np.float32)
    x = np.asarray(a)
    if x.ndim == 1:
        x = np.broadcast_to(x.reshape(1, -1), (n, x.size))
    else:
        x = x.reshape(x.shape[0], -1)
    if x.shape[0] != n:
        if x.shape[0] == 1:
            x = np.broadcast_to(x, (n, x.shape[1]))
        else:
            raise PilotError(f"Translation length {x.shape[0]} != {n}")
    if x.shape[1] < 3:
        x = np.pad(x, ((0, 0), (0, 3 - x.shape[1])))
    return x[:, :3].astype(np.float32, copy=False)


def framewise_betas(a: np.ndarray | None, n: int, width: int = 10) -> np.ndarray:
    if a is None:
        return np.zeros((n, width), dtype=np.float32)
    x = np.asarray(a)
    if x.ndim == 1:
        x = np.broadcast_to(x.reshape(1, -1), (n, x.size))
    else:
        x = x.reshape(x.shape[0], -1)
        if x.shape[0] != n:
            if x.shape[0] == 1:
                x = np.broadcast_to(x, (n, x.shape[1]))
            else:
                flat = np.asarray(a).reshape(-1)
                x = np.broadcast_to(flat.reshape(1, -1), (n, flat.size))
    if x.shape[1] < width:
        x = np.pad(x, ((0, 0), (0, width - x.shape[1])))
    return x[:, :width].astype(np.float32, copy=False)


def load_npz_slice(path: Path, start: int, end: int, schema_cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    schema = schema_cache.setdefault(path, inspect_npz(path))
    if schema.get("status") != "OK":
        raise PilotError(f"NPZ schema failure for {path}: {schema}")
    n_total = int(schema.get("frame_count") or 0)
    if not (0 <= start < end <= n_total):
        raise PilotError(f"Invalid slice [{start},{end}) for {path}, n={n_total}")
    with np.load(path, allow_pickle=False) as z:
        pose = flatten_pose_frames(z[schema["pose_key"]][start:end]) if schema.get("pose_key") else None
        trans = framewise_vec3(z[schema["trans_key"]][start:end] if schema.get("trans_key") else None, end - start)
        betas_arr = z[schema["betas_key"]] if schema.get("betas_key") else None
        if betas_arr is not None and np.asarray(betas_arr).ndim >= 2 and np.asarray(betas_arr).shape[0] == n_total:
            betas_arr = betas_arr[start:end]
        betas = framewise_betas(betas_arr, end - start, 10)
        joints = None
        if schema.get("joints_key"):
            j = np.asarray(z[schema["joints_key"]][start:end], dtype=np.float32)
            if j.ndim == 3 and j.shape[1] >= 55 and j.shape[2] == 3:
                joints = j
    gender = normalize_gender(safe_metadata(path, schema.get("gender_key")))
    return {"pose": pose, "trans": trans, "betas": betas, "joints": joints, "gender": gender}


def concat_optional(arrays: Sequence[np.ndarray | None]) -> np.ndarray | None:
    if not arrays or any(a is None for a in arrays):
        return None
    return np.concatenate([np.asarray(a) for a in arrays if a is not None], axis=0)


def load_full_clip(row: Mapping[str, Any], schema_cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    clip_len = inum(row.get("clip_length"), -1)
    domain = str(row.get("domain", ""))
    source_files = parse_serialized(row.get("source_files")) or []
    segments = parse_serialized(row.get("clip_segments")) or []
    parts: list[dict[str, Any]] = []
    paths: list[str] = []
    if domain == "natural":
        for seg in segments:
            p = Path(str(seg["path"]))
            parts.append(load_npz_slice(p, int(seg["local_start"]), int(seg["local_end"]), schema_cache))
            paths.append(str(p))
    else:
        if not source_files:
            raise PilotError(f"Synthetic sample {row.get('sample_id')} has no source file")
        p = Path(str(source_files[0]))
        schema = schema_cache.setdefault(p, inspect_npz(p))
        n = int(schema.get("frame_count") or clip_len)
        parts.append(load_npz_slice(p, 0, min(n, clip_len), schema_cache))
        paths.append(str(p))
    if not parts:
        raise PilotError(f"No source parts for {row.get('sample_id')}")
    pose = concat_optional([p["pose"] for p in parts])
    if pose is None:
        raise PilotError(f"No pose array for {row.get('sample_id')}")
    trans = np.concatenate([p["trans"] for p in parts], axis=0)
    betas = np.concatenate([p["betas"] for p in parts], axis=0)
    joints = concat_optional([p["joints"] for p in parts])
    gender = Counter(normalize_gender(p.get("gender")) for p in parts).most_common(1)[0][0]
    if len(pose) != clip_len or len(trans) != clip_len or len(betas) != clip_len:
        raise PilotError(f"Loaded length mismatch for {row.get('sample_id')}: pose={len(pose)} expected={clip_len}")
    return {"pose": pose, "trans": trans, "betas": betas, "joints": joints, "gender": gender, "source_paths": paths}


def load_clip_frame(row: Mapping[str, Any], frame: int, schema_cache: dict[Path, dict[str, Any]]) -> dict[str, Any]:
    clip_len = inum(row.get("clip_length"), -1)
    if not (0 <= frame < clip_len):
        raise PilotError(f"Frame {frame} outside {clip_len} for {row.get('sample_id')}")
    domain = str(row.get("domain", ""))
    source_files = parse_serialized(row.get("source_files")) or []
    segments = parse_serialized(row.get("clip_segments")) or []
    if domain == "natural":
        cursor = 0
        for seg in segments:
            length = int(seg["local_end"]) - int(seg["local_start"])
            if cursor <= frame < cursor + length:
                local = int(seg["local_start"]) + (frame - cursor)
                p = Path(str(seg["path"]))
                part = load_npz_slice(p, local, local + 1, schema_cache)
                return {k: (v[0] if isinstance(v, np.ndarray) and v.shape[0] == 1 else v) for k, v in part.items()}
            cursor += length
        raise PilotError(f"Frame mapping failed for {row.get('sample_id')}")
    if not source_files:
        raise PilotError(f"No source file for {row.get('sample_id')}")
    p = Path(str(source_files[0]))
    part = load_npz_slice(p, frame, frame + 1, schema_cache)
    return {k: (v[0] if isinstance(v, np.ndarray) and v.shape[0] == 1 else v) for k, v in part.items()}


def find_smplx_assets(explicit: str | None, data_root: Path, project_root: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    for x in (explicit, os.getenv("SMPLX_MODEL_DIR"), os.getenv("BODY_MODEL_DIR"), os.getenv("SMPLX_PATH")):
        if x:
            candidates.append(Path(x).expanduser())
    home = Path.home()
    candidates.extend([
        project_root / "models", project_root / "body_models",
        data_root / "models", data_root / "body_models", data_root.parent / "body_models",
        home / "body_models", home / "models", home / "smplx",
        home / "Projects" / "GraphARDM" / "body_models",
    ])
    names = {
        "neutral": ("SMPLX_NEUTRAL.npz", "SMPLX_NEUTRAL.pkl"),
        "male": ("SMPLX_MALE.npz", "SMPLX_MALE.pkl"),
        "female": ("SMPLX_FEMALE.npz", "SMPLX_FEMALE.pkl"),
    }
    found: dict[str, str] = {}
    checked: list[str] = []
    seen: set[str] = set()
    for root in candidates:
        root = root.resolve() if root.exists() else root
        if str(root) in seen:
            continue
        seen.add(str(root))
        for sub in (root, root / "smplx", root / "models" / "smplx"):
            checked.append(str(sub))
            for gender, opts in names.items():
                if gender in found:
                    continue
                for name in opts:
                    p = sub / name
                    if p.is_file():
                        found[gender] = str(p.resolve())
                        break
    return {"found": found, "checked": checked, "status": "OK" if found else "MISSING"}


def create_smplx_model(smplx_module: Any, model_file: Path, gender: str, device: str) -> Any:
    attempts: list[tuple[str, str]] = []
    kwargs_variants = [
        dict(model_type="smplx", gender=gender, use_pca=False, flat_hand_mean=True, num_betas=10, num_expression_coeffs=10),
        dict(model_type="smplx", gender=gender, use_pca=False, flat_hand_mean=True, num_betas=10),
        dict(model_type="smplx", gender=gender, use_pca=False, flat_hand_mean=True),
    ]
    candidates = [model_file]
    if model_file.parent.name.lower() == "smplx":
        candidates.append(model_file.parent.parent)
    candidates.append(model_file.parent)
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        for kwargs in kwargs_variants:
            sig = (str(candidate), json.dumps(kwargs, sort_keys=True))
            if sig in seen:
                continue
            seen.add(sig)
            try:
                model = smplx_module.create(str(candidate), **kwargs).to(device)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                return model
            except Exception as e:
                attempts.append((str(candidate), f"{type(e).__name__}: {e}"))
    raise PilotError(f"Could not instantiate SMPL-X {gender}; attempts={attempts}")


class SMPLXEngine:
    def __init__(self, assets: Mapping[str, Any], device: str, batch_size: int):
        import torch
        import smplx
        self.torch = torch
        self.smplx = smplx
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.batch_size = int(batch_size)
        self.found = {k: Path(v) for k, v in assets.get("found", {}).items()}
        if not self.found:
            raise PilotError("No SMPL-X assets found")
        self.models: dict[str, Any] = {}
        self.frames_processed = 0

    def model(self, gender: str) -> tuple[Any, str]:
        g = normalize_gender(gender)
        model_gender = g if g in {"male", "female"} and g in self.found else "neutral"
        model_file = self.found.get(model_gender) or self.found.get("neutral") or next(iter(self.found.values()))
        key = f"{model_gender}:{model_file}"
        if key not in self.models:
            self.models[key] = create_smplx_model(self.smplx, model_file, model_gender, self.device)
        return self.models[key], model_gender

    def forward_torch(self, pose: Any, trans: Any, betas: Any, gender: str) -> Any:
        torch = self.torch
        model, _ = self.model(gender)
        num_betas = int(getattr(model, "num_betas", 10))
        num_expr = int(getattr(model, "num_expression_coeffs", 10))
        if betas.shape[1] < num_betas:
            betas = torch.nn.functional.pad(betas, (0, num_betas - betas.shape[1]))
        betas = betas[:, :num_betas]
        out = model(
            betas=betas,
            global_orient=pose[:, 0:3],
            body_pose=pose[:, 3:66],
            jaw_pose=pose[:, 66:69],
            leye_pose=pose[:, 69:72],
            reye_pose=pose[:, 72:75],
            left_hand_pose=pose[:, 75:120],
            right_hand_pose=pose[:, 120:165],
            expression=torch.zeros((pose.shape[0], num_expr), dtype=pose.dtype, device=pose.device),
            transl=trans,
            return_verts=False,
        )
        return out.joints

    def joints(self, pose: np.ndarray, trans: np.ndarray, betas: np.ndarray, gender: str) -> np.ndarray:
        torch = self.torch
        n = len(pose)
        chunks: list[np.ndarray] = []
        model, mg = self.model(gender)
        del model
        for start in range(0, n, self.batch_size):
            end = min(n, start + self.batch_size)
            p = torch.as_tensor(pose[start:end], dtype=torch.float32, device=self.device)
            tr = torch.as_tensor(trans[start:end], dtype=torch.float32, device=self.device)
            b = torch.as_tensor(betas[start:end], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                j = self.forward_torch(p, tr, b, mg)
            chunks.append(j.detach().cpu().numpy().astype(np.float32, copy=False))
        self.frames_processed += n
        return np.concatenate(chunks, axis=0)

    def joints_many(self, clips: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
        out: list[np.ndarray | None] = [None] * len(clips)
        groups: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(clips):
            _, mg = self.model(str(c.get("gender", "neutral")))
            groups[mg].append(i)
        for mg, indices in groups.items():
            offsets = [0]
            poses: list[np.ndarray] = []
            trans: list[np.ndarray] = []
            betas: list[np.ndarray] = []
            for i in indices:
                c = clips[i]
                poses.append(np.asarray(c["pose"], dtype=np.float32))
                trans.append(np.asarray(c["trans"], dtype=np.float32))
                b = np.asarray(c["betas"], dtype=np.float32)
                if b.ndim == 1:
                    b = np.broadcast_to(b[None, :], (len(poses[-1]), b.size))
                betas.append(b)
                offsets.append(offsets[-1] + len(poses[-1]))
            all_pose = np.concatenate(poses, axis=0)
            all_trans = np.concatenate(trans, axis=0)
            all_betas = np.concatenate(betas, axis=0)
            all_j = self.joints(all_pose, all_trans, all_betas, mg)
            for local, i in enumerate(indices):
                out[i] = all_j[offsets[local]:offsets[local + 1]]
        return [np.asarray(x) for x in out]


def unit_rows(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1)
    return v / np.maximum(n[..., None], 1e-8), n


def central_gradient(x: np.ndarray, fps: float = FPS) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros_like(x, dtype=np.float64)
    return np.gradient(x.astype(np.float64), axis=0) * fps


def angular_speed(unit_vec: np.ndarray, fps: float = FPS) -> np.ndarray:
    n = len(unit_vec)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    dots = np.sum(unit_vec[1:] * unit_vec[:-1], axis=1)
    ang = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))) * fps
    out = np.empty(n, dtype=np.float64)
    out[0] = ang[0]
    out[1:] = ang
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64, copy=True)
    left = window // 2
    right = window - 1 - left
    p = np.pad(x.astype(np.float64), ((left, right), (0, 0)), mode="edge")
    c = np.concatenate([np.zeros((1, p.shape[1]), dtype=np.float64), np.cumsum(p, axis=0)], axis=0)
    return (c[window:] - c[:-window]) / float(window)


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    m = rolling_mean(x, window)
    m2 = rolling_mean(np.square(x.astype(np.float64)), window)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def smooth_1d(x: np.ndarray, kernel: np.ndarray = SMOOTHING_KERNEL) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    pad = len(kernel) // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid")


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(1.4826 * mad, float(np.std(x)), 1e-6)
    return (x - med) / scale


def angle_between_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    au, an = unit_rows(a)
    bu, bn = unit_rows(b)
    out = np.degrees(np.arccos(np.clip(np.sum(au * bu, axis=1), -1.0, 1.0)))
    out[(an < 1e-8) | (bn < 1e-8)] = 0.0
    return out


def kinematic_features(joints: np.ndarray, hand: int) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    hand_name = "left" if int(hand) == 0 else "right"
    inactive = "right" if hand_name == "left" else "left"
    pelvis = joints[:, JOINT_INDEX["pelvis"]].astype(np.float64)
    neck = joints[:, JOINT_INDEX["neck"]].astype(np.float64)
    lsho = joints[:, JOINT_INDEX["left_shoulder"]].astype(np.float64)
    rsho = joints[:, JOINT_INDEX["right_shoulder"]].astype(np.float64)
    shoulder = joints[:, JOINT_INDEX[f"{hand_name}_shoulder"]].astype(np.float64)
    elbow = joints[:, JOINT_INDEX[f"{hand_name}_elbow"]].astype(np.float64)
    wrist = joints[:, JOINT_INDEX[f"{hand_name}_wrist"]].astype(np.float64)
    index3 = joints[:, JOINT_INDEX[f"{hand_name}_index3"]].astype(np.float64)
    inactive_wrist = joints[:, JOINT_INDEX[f"{inactive}_wrist"]].astype(np.float64)
    upper = elbow - shoulder
    fore = wrist - elbow
    full = wrist - shoulder
    fore_u, fore_len_frame = unit_rows(fore)
    upper_u, upper_len_frame = unit_rows(upper)
    arm_len = float(np.median(upper_len_frame + fore_len_frame))
    if not np.isfinite(arm_len) or arm_len < 1e-4:
        arm_len = 1.0
    reach_ratio = np.linalg.norm(full, axis=1) / arm_len
    elbow_angle = angle_between_rows(shoulder - elbow, wrist - elbow) / 180.0
    lateral, _ = unit_rows(rsho - lsho)
    up0, _ = unit_rows(neck - pelvis)
    forward, _ = unit_rows(np.cross(lateral, up0))
    up, _ = unit_rows(np.cross(forward, lateral))
    hand_sign = -1.0 if hand_name == "left" else 1.0
    wrist_sh = (wrist - shoulder) / arm_len
    wrist_pelvis = (wrist - pelvis) / arm_len
    elbow_pelvis = (elbow - pelvis) / arm_len
    shoulder_pelvis = (shoulder - pelvis) / arm_len
    fore_lat = np.sum(fore_u * lateral, axis=1) * hand_sign
    fore_up = np.sum(fore_u * up, axis=1)
    fore_fwd = np.sum(fore_u * forward, axis=1)
    wrist_lat = np.sum(wrist_sh * lateral, axis=1) * hand_sign
    wrist_up = np.sum(wrist_sh * up, axis=1)
    wrist_fwd = np.sum(wrist_sh * forward, axis=1)
    wrist_vel = central_gradient(wrist) / arm_len
    elbow_vel = central_gradient(elbow) / arm_len
    shoulder_vel = central_gradient(shoulder) / arm_len
    index_vel = central_gradient(index3) / arm_len
    inactive_vel = central_gradient(inactive_wrist) / arm_len
    pelvis_vel = central_gradient(pelvis) / arm_len
    wrist_acc = central_gradient(wrist_vel) / FPS
    index_acc = central_gradient(index_vel) / FPS
    wrist_jerk = central_gradient(wrist_acc) / FPS
    wrist_speed = np.linalg.norm(wrist_vel, axis=1)
    elbow_speed = np.linalg.norm(elbow_vel, axis=1)
    shoulder_speed = np.linalg.norm(shoulder_vel, axis=1)
    index_speed = np.linalg.norm(index_vel, axis=1)
    inactive_speed = np.linalg.norm(inactive_vel, axis=1)
    pelvis_speed = np.linalg.norm(pelvis_vel, axis=1)
    wrist_acc_mag = np.linalg.norm(wrist_acc, axis=1)
    index_acc_mag = np.linalg.norm(index_acc, axis=1)
    wrist_jerk_mag = np.linalg.norm(wrist_jerk, axis=1)
    fore_ang = angular_speed(fore_u) / 180.0
    upper_ang = angular_speed(upper_u) / 180.0
    lateral_ang = angular_speed(lateral) / 180.0
    reach_vel = central_gradient(reach_ratio[:, None])[:, 0] / FPS
    reach_acc = central_gradient(reach_vel[:, None])[:, 0] / FPS
    selected = np.stack([pelvis, neck, lsho, rsho, elbow, wrist, inactive_wrist], axis=1)
    selected_vel = central_gradient(selected) / arm_len
    selected_speed = np.linalg.norm(selected_vel, axis=2)
    body_mean_speed = np.mean(selected_speed, axis=1)
    body_max_speed = np.max(selected_speed, axis=1)
    speed_logratio = np.log((wrist_speed + 1e-3) / (inactive_speed + 1e-3))
    index_fore_alignment = np.sum(fore_u * unit_rows(index3 - wrist)[0], axis=1)
    raw_cols = [
        reach_ratio, elbow_angle, np.linalg.norm(wrist_pelvis, axis=1),
        wrist_lat, wrist_up, wrist_fwd, fore_lat, fore_up, fore_fwd,
        wrist_speed, elbow_speed, shoulder_speed, index_speed, inactive_speed, pelvis_speed,
        wrist_acc_mag, index_acc_mag, fore_ang, upper_ang, reach_vel, reach_acc,
        body_mean_speed, body_max_speed, speed_logratio,
        wrist_pelvis[:, 1], elbow_pelvis[:, 1], shoulder_pelvis[:, 1],
        index_fore_alignment, lateral_ang, wrist_jerk_mag,
    ]
    names = [
        "reach_ratio", "elbow_angle_norm", "wrist_pelvis_distance_norm",
        "wrist_outward_norm", "wrist_up_from_shoulder_norm", "wrist_forward_norm",
        "forearm_outward", "forearm_up", "forearm_forward",
        "wrist_speed_norm", "elbow_speed_norm", "shoulder_speed_norm", "index_speed_norm",
        "inactive_wrist_speed_norm", "pelvis_speed_norm", "wrist_accel_norm", "index_accel_norm",
        "forearm_angular_speed_norm", "upperarm_angular_speed_norm", "reach_velocity", "reach_acceleration",
        "body_mean_speed_norm", "body_max_speed_norm", "active_inactive_speed_logratio",
        "wrist_height_pelvis_norm", "elbow_height_pelvis_norm", "shoulder_height_pelvis_norm",
        "index_forearm_alignment", "torso_lateral_angular_speed_norm", "wrist_jerk_norm",
    ]
    raw = np.nan_to_num(np.stack(raw_cols, axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pooled_parts = [raw]
    pooled_names = list(names)
    for w in (3, 7, 15):
        pooled_parts.append(rolling_mean(raw, w).astype(np.float32))
        pooled_names.extend([f"mean{w}:{n}" for n in names])
    for w in (7, 15):
        pooled_parts.append(rolling_std(raw, w).astype(np.float32))
        pooled_names.extend([f"std{w}:{n}" for n in names])
    pooled = np.nan_to_num(np.concatenate(pooled_parts, axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return pooled, pooled_names, {
        "raw": raw, "raw_names": names, "reach_ratio": reach_ratio.astype(np.float32),
        "wrist_speed": wrist_speed.astype(np.float32), "forearm_angular_speed": fore_ang.astype(np.float32),
        "forearm_ray": fore.astype(np.float32), "elbow": elbow.astype(np.float32),
        "wrist": wrist.astype(np.float32), "pelvis": pelvis.astype(np.float32),
    }


def rolling_mean_causal(x: np.ndarray, w: int) -> np.ndarray:
    n, d = x.shape
    cs = np.vstack([np.zeros((1, d), dtype=np.float64), np.cumsum(x, axis=0, dtype=np.float64)])
    out = np.empty((n, d), dtype=np.float64)
    for t in range(n):
        a = max(0, t - w + 1)
        out[t] = (cs[t + 1] - cs[a]) / float(t + 1 - a)
    return out


def rolling_mean_future(x: np.ndarray, w: int) -> np.ndarray:
    return rolling_mean_causal(x[::-1], w)[::-1]


def rolling_mean_centered_1d(x: np.ndarray, w: int) -> np.ndarray:
    x2 = np.asarray(x, dtype=np.float64)[:, None]
    half = w // 2
    pad = np.pad(x2, ((half, half), (0, 0)), mode="edge")
    cs = np.vstack([np.zeros((1, 1), dtype=np.float64), np.cumsum(pad, axis=0)])
    return ((cs[w:] - cs[:-w]) / float(w))[:, 0]


def percentile_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n <= 1:
        return np.full(n, 0.5, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / float(n - 1)


def phase_features(raw: np.ndarray, raw_names: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    idx = {name: i for i, name in enumerate(raw_names)}
    wanted = [
        "reach_ratio", "elbow_angle_norm", "wrist_outward_norm", "wrist_up_from_shoulder_norm",
        "wrist_forward_norm", "forearm_outward", "forearm_up", "forearm_forward",
        "wrist_speed_norm", "index_speed_norm", "forearm_angular_speed_norm", "reach_velocity",
        "reach_acceleration", "body_mean_speed_norm", "active_inactive_speed_logratio",
        "wrist_height_pelvis_norm", "index_forearm_alignment", "wrist_jerk_norm",
    ]
    missing = [n for n in wanted if n not in idx]
    if missing:
        raise PilotError(f"Missing phase features: {missing}")
    sel = [idx[n] for n in wanted]
    clip = np.asarray(raw, dtype=np.float64)
    mu = clip.mean(axis=0)
    sd = clip.std(axis=0)
    sd[sd < 1e-6] = 1.0
    z = np.clip((clip - mu) / sd, -8.0, 8.0)
    xs = clip[:, sel]
    phase_parts: list[np.ndarray] = []
    for w in (3, 7, 15):
        past = rolling_mean_causal(xs, w)
        future = rolling_mean_future(xs, w)
        phase_parts.extend([past, future, future - past])
    reach = clip[:, idx["reach_ratio"]]
    speed = clip[:, idx["wrist_speed_norm"]]
    fore_ang = clip[:, idx["forearm_angular_speed_norm"]]
    outward = clip[:, idx["wrist_outward_norm"]]
    up = clip[:, idx["wrist_up_from_shoulder_norm"]]
    jerk = clip[:, idx["wrist_jerk_norm"]]
    rr = percentile_rank(reach)
    sr = percentile_rank(-(speed + 0.5 * fore_ang))
    local7 = rolling_mean_centered_1d(reach, 7)
    local15 = rolling_mean_centered_1d(reach, 15)
    speed7 = rolling_mean_centered_1d(speed, 7)
    ang7 = rolling_mean_centered_1d(fore_ang, 7)
    n = len(clip)
    edge = np.minimum(np.arange(n), np.arange(n)[::-1]) / max(1.0, (n - 1) / 2.0)
    extra = np.stack([
        rr, sr, percentile_rank(outward), percentile_rank(up), reach - local7, reach - local15,
        -speed7, -ang7, rr * sr, rr * percentile_rank(outward), -percentile_rank(jerk), edge,
    ], axis=1)
    X = np.concatenate([clip, z[:, sel], *phase_parts, extra], axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, {"reach_rank": rr, "stability_rank": sr, "edge": edge}


def valid_mask(n: int) -> np.ndarray:
    m = np.ones(n, dtype=bool)
    if n >= 7:
        m[:2] = False
        m[-2:] = False
    return m


def contiguous_components(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(mask) - 1):
            end = i if v and i == len(mask) - 1 else i - 1
            out.append((start, end))
            start = None
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def select_component(logit: np.ndarray, reach_rank: np.ndarray, stability_rank: np.ndarray, edge: np.ndarray) -> int:
    s = smooth_1d(logit)
    p = sigmoid(s)
    mvalid = valid_mask(len(p))
    finite = p[mvalid]
    if finite.size == 0:
        return int(np.argmax(p))
    thr = max(float(np.quantile(finite, 0.75)), 0.55 * float(np.max(finite)))
    comps = contiguous_components((p >= thr) & mvalid)
    if not comps:
        score = robust_z(s) + 0.45 * reach_rank + 0.25 * stability_rank + 0.10 * edge
        score = np.where(mvalid, score, -np.inf)
        return int(np.argmax(score))
    best: tuple[float, int, int] | None = None
    for a, b in comps:
        q = np.arange(a, b + 1)
        mass = float(np.sum(p[q] * (0.55 + 0.30 * reach_rank[q] + 0.15 * stability_rank[q])))
        mass *= 1.0 + 0.05 * math.log1p(len(q))
        cand = (mass, a, b)
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None
    _, a, b = best
    q = np.arange(a, b + 1)
    local = p[q] + 0.20 * reach_rank[q] + 0.10 * stability_rank[q] + 0.05 * edge[q]
    return int(q[int(np.argmax(local))])


def load_linear_model(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def model_logits(model: Mapping[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    coef = np.asarray(model["coef"], dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(model["intercept"]).reshape(-1)[0])
    mean = np.asarray(model["feature_mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(model["feature_std"], dtype=np.float64).reshape(-1)
    if X.shape[1] != len(coef):
        raise PilotError(f"Detector feature dimension mismatch {X.shape[1]} != {len(coef)}")
    return ((X.astype(np.float64) - mean) / np.maximum(std, 1e-8)) @ coef + intercept


def detector_predictions(joints: np.ndarray, hand: int, source_model: Mapping[str, np.ndarray], refined_model: Mapping[str, np.ndarray]) -> dict[str, Any]:
    pooled, _, aux = kinematic_features(joints, hand)
    raw = np.asarray(aux["raw"], dtype=np.float32)
    raw_names = list(aux["raw_names"])
    src_variant = str(np.asarray(source_model.get("model_variant", np.asarray(["logreg_raw"]))).reshape(-1)[0])
    src_X = raw if src_variant == "logreg_raw" else pooled
    src_logit = model_logits(source_model, src_X)
    src_score = smooth_1d(src_logit)
    src_score = np.where(valid_mask(len(src_score)), src_score, -np.inf)
    src_pred = int(np.argmax(src_score))
    phase_X, ph_aux = phase_features(raw, raw_names)
    ref_logit = model_logits(refined_model, phase_X)
    ref_pred = select_component(ref_logit, ph_aux["reach_rank"], ph_aux["stability_rank"], ph_aux["edge"])
    ix = {n: i for i, n in enumerate(raw_names)}
    reach = raw[:, ix["reach_ratio"]].astype(np.float64)
    elbow = raw[:, ix["elbow_angle_norm"]].astype(np.float64)
    speed = raw[:, ix["wrist_speed_norm"]].astype(np.float64)
    ang = raw[:, ix["forearm_angular_speed_norm"]].astype(np.float64)
    reach_v = np.abs(raw[:, ix["reach_velocity"]].astype(np.float64))
    heur_score = 1.4 * robust_z(reach) + 0.4 * robust_z(elbow) - 0.9 * robust_z(speed) - 0.5 * robust_z(ang) - 0.25 * robust_z(reach_v)
    heur_score = np.where(valid_mask(len(heur_score)), heur_score, -np.inf)
    heur_pred = int(np.argmax(heur_score))
    preds = np.asarray([src_pred, ref_pred, heur_pred], dtype=int)
    ensemble = int(np.median(preds))
    agreement = float(math.exp(-(float(np.std(preds)) ** 2) / (2.0 * 5.0 ** 2)))
    return {"source": src_pred, "refined": ref_pred, "heuristic": heur_pred, "ensemble": ensemble, "agreement": agreement}


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = 0.5 * angle
    axis = aa / np.maximum(angle, 1e-12)
    xyz = axis * np.sin(half)
    w = np.cos(half)
    q = np.concatenate([w, xyz], axis=-1)
    small = angle[..., 0] < 1e-8
    if np.any(small):
        q[small, 0] = 1.0
        q[small, 1:] = 0.5 * aa[small]
        q[small] /= np.maximum(np.linalg.norm(q[small], axis=-1, keepdims=True), 1e-12)
    return q


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q = np.where(q[..., :1] < 0, -q, q)
    w = np.clip(q[..., 0:1], -1.0, 1.0)
    xyz = q[..., 1:]
    s = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, w)
    axis = xyz / np.maximum(s, 1e-12)
    aa = axis * angle
    small = s[..., 0] < 1e-8
    aa[small] = 2.0 * xyz[small]
    return aa


def resample_axis_angle(pose: np.ndarray, sample_positions: np.ndarray) -> np.ndarray:
    T, D = pose.shape
    J = D // 3
    aa = pose.reshape(T, J, 3).astype(np.float64)
    q = axis_angle_to_quat(aa)
    out_q = np.empty((len(sample_positions), J, 4), dtype=np.float64)
    for i, pos in enumerate(sample_positions):
        lo = int(math.floor(float(pos)))
        hi = min(T - 1, lo + 1)
        a = float(pos - lo)
        q0 = q[lo]
        q1 = q[hi]
        q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -q1, q1)
        qi = (1.0 - a) * q0 + a * q1
        qi /= np.maximum(np.linalg.norm(qi, axis=-1, keepdims=True), 1e-12)
        out_q[i] = qi
    return quat_to_axis_angle(out_q).reshape(len(sample_positions), D).astype(np.float32)


def piecewise_positions(src_len: int, src_peak: int, dst_len: int, dst_peak: int) -> np.ndarray:
    if src_len < 2 or dst_len < 2:
        return np.zeros(dst_len, dtype=np.float64)
    src_peak = int(np.clip(src_peak, 1, src_len - 2))
    dst_peak = int(np.clip(dst_peak, 1, dst_len - 2))
    pos = np.empty(dst_len, dtype=np.float64)
    if dst_peak > 0:
        pos[:dst_peak + 1] = np.linspace(0.0, float(src_peak), dst_peak + 1)
    if dst_peak < dst_len - 1:
        pos[dst_peak:] = np.linspace(float(src_peak), float(src_len - 1), dst_len - dst_peak)
    return pos


def warp_clip(clip: Mapping[str, Any], src_peak: int, dst_len: int, dst_peak: int) -> dict[str, Any]:
    pose = np.asarray(clip["pose"], dtype=np.float32)
    trans = np.asarray(clip["trans"], dtype=np.float32)
    pos = piecewise_positions(len(pose), src_peak, dst_len, dst_peak)
    pose_out = resample_axis_angle(pose, pos)
    trans_out = np.empty((dst_len, 3), dtype=np.float32)
    for k in range(3):
        trans_out[:, k] = np.interp(pos, np.arange(len(trans)), trans[:, k])
    trans_out = trans_out - trans_out[dst_peak:dst_peak + 1]
    return {"pose": pose_out, "trans": trans_out}


def target_vec(row: Mapping[str, Any]) -> np.ndarray:
    v = np.asarray([fnum(row.get("target_pelvis_x")), fnum(row.get("target_pelvis_y")), fnum(row.get("target_pelvis_z"))], dtype=np.float64)
    if not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-6:
        raise PilotError(f"Invalid target vector for {row.get('sample_id')}: {v}")
    return v


def descriptor(row: Mapping[str, Any], peak_ratio: float | None = None) -> np.ndarray:
    v = target_vec(row)
    d = float(np.linalg.norm(v))
    az = math.atan2(float(v[0]), float(v[2])) / math.pi
    el = math.atan2(float(v[1]), max(math.hypot(float(v[0]), float(v[2])), 1e-8)) / (math.pi / 2.0)
    pr = fnum(row.get("released_peak_ratio"), 0.5) if peak_ratio is None else float(peak_ratio)
    return np.asarray([az, el, math.log(max(d, 1e-4)), inum(row.get("clip_length"), 120) / 180.0, pr], dtype=np.float64)


def transcript_index(text_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(text_root.rglob("*.tsv")):
        out.setdefault(p.stem, p)
    return out


def resolve_transcript(row: Mapping[str, Any], index: Mapping[str, Path]) -> tuple[Path | None, str]:
    rid = str(row.get("recording_id", ""))
    sid = str(row.get("sample_id", ""))
    for key in (rid, sid):
        if key in index:
            return index[key], "exact"
    variants = [rid, rid.removesuffix("_main"), sid]
    matches: list[Path] = []
    for stem, p in index.items():
        if any(v and (stem == v or stem.startswith(v) or v.startswith(stem)) for v in variants):
            matches.append(p)
    if len(matches) == 1:
        return matches[0], "fuzzy_unique"
    return None, "missing"


def read_word_times(path: Path) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    pat = re.compile(r"^\s*(?:\|\s*)?([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+(.*?)(?:\s*)$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line)
        if not m:
            continue
        rows.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
    return rows


def timing_plan(row: Mapping[str, Any], idx: Mapping[str, Path], fallback_ratio: float, cache: dict[Path, list[tuple[float, float, str]]]) -> dict[str, Any]:
    T = inum(row.get("clip_length"), 120)
    path, status = resolve_transcript(row, idx)
    start_frame = inum(row.get("global_start"), -1)
    domain = str(row.get("domain"))
    clip_start = start_frame / FPS if domain == "natural" and start_frame >= 0 else 0.0
    clip_end = clip_start + T / FPS
    if path is not None:
        words = cache.setdefault(path, read_word_times(path))
        overlapping = [(a, b, w) for a, b, w in words if b >= clip_start - 0.05 and a <= clip_end + 0.05]
        if overlapping:
            phrase_end = max(b for _, b, _ in overlapping)
            p = int(round((phrase_end - clip_start) * FPS)) - 11
            p = int(np.clip(p, 2, max(2, T - 3)))
            return {
                "planned_peak": p, "planned_peak_ratio": p / max(1, T - 1),
                "timing_source": "last_word_end_minus_11", "transcript_path": str(path),
                "transcript_match": status, "word_count": len(overlapping), "phrase_end_sec": phrase_end,
            }
    p = int(round(float(fallback_ratio) * max(1, T - 1)))
    p = int(np.clip(p, 2, max(2, T - 3)))
    return {
        "planned_peak": p, "planned_peak_ratio": p / max(1, T - 1),
        "timing_source": "training_median_peak_ratio_fallback", "transcript_path": str(path or ""),
        "transcript_match": status, "word_count": 0, "phrase_end_sec": "",
    }


def b0_feature(row: Mapping[str, Any], beta: np.ndarray, planned_ratio: float) -> np.ndarray:
    v = target_vec(row)
    d = float(np.linalg.norm(v))
    u = v / max(d, 1e-8)
    domain_flag = 1.0 if str(row.get("domain")) == "natural" else 0.0
    return np.concatenate([
        v.astype(np.float64), u.astype(np.float64), np.asarray([math.log(max(d, 1e-4))]),
        np.asarray(beta[:10], dtype=np.float64),
        np.asarray([planned_ratio, inum(row.get("clip_length"), 120) / 180.0, domain_flag], dtype=np.float64),
    ])


@dataclass
class B0Model:
    hand: int
    state: dict[str, Any]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def predict(self, x: np.ndarray, device: str) -> np.ndarray:
        import torch
        net = make_mlp(len(self.x_mean), 6).to(device)
        net.load_state_dict(self.state)
        net.eval()
        xx = ((x - self.x_mean) / np.maximum(self.x_std, 1e-8)).astype(np.float32)
        with torch.no_grad():
            yy = net(torch.as_tensor(xx[None, :], dtype=torch.float32, device=device))[0].cpu().numpy()
        return yy * self.y_std + self.y_mean


def make_mlp(in_dim: int, out_dim: int) -> Any:
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, out_dim),
    )


def train_b0_models(
    rows: Sequence[Mapping[str, Any]], timing: Mapping[str, Mapping[str, Any]],
    frame_data: Mapping[str, Mapping[str, Any]], joints: Mapping[str, np.ndarray],
    device: str, epochs: int, seed: int, output_dir: Path,
) -> tuple[dict[int, B0Model], list[dict[str, Any]]]:
    import torch
    models: dict[int, B0Model] = {}
    history_rows: list[dict[str, Any]] = []
    for hand in (0, 1):
        selected = [r for r in rows if inum(r.get("hand")) == hand]
        X: list[np.ndarray] = []
        Y: list[np.ndarray] = []
        for r in selected:
            sid = str(r["sample_id"])
            fd = frame_data[sid]
            j = joints[sid]
            hn = "left" if hand == 0 else "right"
            pelvis = j[JOINT_INDEX["pelvis"]]
            elbow = j[JOINT_INDEX[f"{hn}_elbow"]] - pelvis
            wrist = j[JOINT_INDEX[f"{hn}_wrist"]] - pelvis
            X.append(b0_feature(r, np.asarray(fd["betas"], dtype=np.float32), float(timing[sid]["planned_peak_ratio"])))
            Y.append(np.concatenate([elbow, wrist]).astype(np.float64))
        x = np.stack(X).astype(np.float64)
        y = np.stack(Y).astype(np.float64)
        xm, xs = x.mean(axis=0), x.std(axis=0)
        ym, ys = y.mean(axis=0), y.std(axis=0)
        xs[xs < 1e-6] = 1.0
        ys[ys < 1e-6] = 1.0
        xn = ((x - xm) / xs).astype(np.float32)
        yn = ((y - ym) / ys).astype(np.float32)
        torch.manual_seed(seed + 1000 * hand)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed + 1000 * hand)
        net = make_mlp(x.shape[1], y.shape[1]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-5)
        rng = np.random.default_rng(seed + hand)
        batch = 256
        for epoch in range(epochs):
            perm = rng.permutation(len(xn))
            total = 0.0
            count = 0
            net.train()
            for a in range(0, len(perm), batch):
                ids = perm[a:a + batch]
                tx = torch.as_tensor(xn[ids], dtype=torch.float32, device=device)
                ty = torch.as_tensor(yn[ids], dtype=torch.float32, device=device)
                pred = net(tx)
                loss = torch.mean((pred - ty) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu()) * len(ids)
                count += len(ids)
            sched.step()
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                history_rows.append({"hand": hand, "epoch": epoch + 1, "mse_standardized": total / max(1, count), "lr": sched.get_last_lr()[0]})
        state_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        torch.save({
            "state_dict": state_cpu, "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
            "hand": hand, "version": VERSION,
        }, output_dir / f"b0_controller_hand{hand}.pt")
        models[hand] = B0Model(hand, state_cpu, xm, xs, ym, ys)
    return models, history_rows


def active_joint_ids(hand: int) -> list[int]:
    hn = "left" if hand == 0 else "right"
    return [JOINT_INDEX[f"{hn}_collar"], JOINT_INDEX[f"{hn}_shoulder"], JOINT_INDEX[f"{hn}_elbow"], JOINT_INDEX[f"{hn}_wrist"]]


def optimize_apex_batch(
    engine: SMPLXEngine, base_pose: np.ndarray, betas: np.ndarray, gender: str, hand: int,
    mode: str, targets: np.ndarray | None, anchors: np.ndarray | None,
    steps: int, lr: float, max_delta: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    torch = engine.torch
    device = engine.device
    B = len(base_pose)
    joint_ids = active_joint_ids(hand)
    base = torch.as_tensor(base_pose, dtype=torch.float32, device=device)
    b = torch.as_tensor(betas, dtype=torch.float32, device=device)
    tr = torch.zeros((B, 3), dtype=torch.float32, device=device)
    delta = torch.zeros((B, len(joint_ids), 3), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    with torch.no_grad():
        j0 = engine.forward_torch(base, tr, b, gender)
        hn = "left" if hand == 0 else "right"
        s0 = j0[:, JOINT_INDEX[f"{hn}_shoulder"]]
        e0 = j0[:, JOINT_INDEX[f"{hn}_elbow"]]
        w0 = j0[:, JOINT_INDEX[f"{hn}_wrist"]]
        desired_reach = torch.maximum(torch.linalg.norm(w0 - s0, dim=1) / torch.clamp(torch.linalg.norm(e0 - s0, dim=1) + torch.linalg.norm(w0 - e0, dim=1), min=1e-6), torch.tensor(0.88, device=device))
        base_elbow_cos = torch.sum((s0 - e0) * (w0 - e0), dim=1) / torch.clamp(torch.linalg.norm(s0 - e0, dim=1) * torch.linalg.norm(w0 - e0, dim=1), min=1e-6)
    target_t = torch.as_tensor(targets, dtype=torch.float32, device=device) if targets is not None else None
    anchor_t = torch.as_tensor(anchors, dtype=torch.float32, device=device) if anchors is not None else None
    logs: list[dict[str, float]] = []
    for step in range(steps):
        pose = base.clone()
        for k, jid in enumerate(joint_ids):
            pose[:, jid * 3:(jid + 1) * 3] = base[:, jid * 3:(jid + 1) * 3] + delta[:, k]
        j = engine.forward_torch(pose, tr, b, gender)
        hn = "left" if hand == 0 else "right"
        pelvis = j[:, JOINT_INDEX["pelvis"]]
        shoulder = j[:, JOINT_INDEX[f"{hn}_shoulder"]]
        elbow = j[:, JOINT_INDEX[f"{hn}_elbow"]]
        wrist = j[:, JOINT_INDEX[f"{hn}_wrist"]]
        if mode == "b0":
            assert anchor_t is not None
            er = elbow - pelvis
            wr = wrist - pelvis
            loss_primary = torch.mean(torch.sum((er - anchor_t[:, 0:3]) ** 2, dim=1) + torch.sum((wr - anchor_t[:, 3:6]) ** 2, dim=1))
        elif mode == "p0c":
            assert target_t is not None
            target_world = pelvis + target_t
            fore = wrist - elbow
            to_target = target_world - elbow
            dot = torch.sum(fore * to_target, dim=1) / torch.clamp(torch.linalg.norm(fore, dim=1) * torch.linalg.norm(to_target, dim=1), min=1e-6)
            ray_loss = torch.mean(1.0 - torch.clamp(dot, -1.0, 1.0))
            reach = torch.linalg.norm(wrist - shoulder, dim=1) / torch.clamp(torch.linalg.norm(elbow - shoulder, dim=1) + torch.linalg.norm(wrist - elbow, dim=1), min=1e-6)
            reach_loss = torch.mean((reach - desired_reach) ** 2)
            elbow_cos = torch.sum((shoulder - elbow) * (wrist - elbow), dim=1) / torch.clamp(torch.linalg.norm(shoulder - elbow, dim=1) * torch.linalg.norm(wrist - elbow, dim=1), min=1e-6)
            bend_loss = torch.mean((elbow_cos - base_elbow_cos) ** 2)
            loss_primary = 8.0 * ray_loss + 0.8 * reach_loss + 0.15 * bend_loss
        else:
            raise PilotError(f"Unknown IK mode {mode}")
        reg = 0.015 * torch.mean(delta ** 2)
        large = 0.02 * torch.mean(torch.relu(torch.linalg.norm(delta, dim=2) - 1.0) ** 2)
        loss = loss_primary + reg + large
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-max_delta, max_delta)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            logs.append({"step": float(step + 1), "loss": float(loss.detach().cpu()), "primary": float(loss_primary.detach().cpu()), "reg": float(reg.detach().cpu())})
    out = base_pose.copy()
    d = delta.detach().cpu().numpy()
    for k, jid in enumerate(joint_ids):
        out[:, jid * 3:(jid + 1) * 3] += d[:, k]
    return out, logs


def hold_weights(T: int, peak: int, plateau: int, taper: int) -> np.ndarray:
    half = max(0, plateau // 2)
    w = np.zeros(T, dtype=np.float64)
    for t in range(T):
        d = abs(t - peak)
        if d <= half:
            w[t] = 1.0
        elif d <= half + taper:
            x = (d - half) / max(1, taper)
            w[t] = 0.5 * (1.0 + math.cos(math.pi * x))
    return w


def apply_hold(base_pose: np.ndarray, peak: int, optimized_apex: np.ndarray, hand: int, plateau: int, taper: int) -> np.ndarray:
    """Blend active-arm rotations to the optimized apex on SO(3) via quaternion nlerp."""
    out = np.asarray(base_pose, dtype=np.float32).copy()
    w = hold_weights(len(out), peak, plateau, taper).astype(np.float64)
    for jid in active_joint_ids(hand):
        sl = slice(jid * 3, (jid + 1) * 3)
        q0 = axis_angle_to_quat(out[:, sl].astype(np.float64))
        qt = axis_angle_to_quat(np.asarray(optimized_apex[sl], dtype=np.float64)[None, :])[0]
        qt_all = np.broadcast_to(qt[None, :], q0.shape).copy()
        qt_all = np.where(np.sum(q0 * qt_all, axis=1, keepdims=True) < 0.0, -qt_all, qt_all)
        qb = (1.0 - w[:, None]) * q0 + w[:, None] * qt_all
        qb /= np.maximum(np.linalg.norm(qb, axis=1, keepdims=True), 1e-12)
        out[:, sl] = quat_to_axis_angle(qb).astype(np.float32)
    return out


def resolve_scene(data_root: Path, row: Mapping[str, Any], cache: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("domain")) != "natural":
        return None
    room = str(row.get("room_id", "")).strip()
    obj = str(row.get("object_id", "")).strip()
    if not room or not obj:
        return None
    if room not in cache:
        candidates = [data_root / "data" / "scenes" / f"{room}.json"]
        candidates.extend(sorted((data_root / "data" / "scenes").glob(f"*{room}*.json")))
        path = next((p for p in candidates if p.is_file()), None)
        cache[room] = json_load(path) if path else None
    scene = cache[room]
    if not isinstance(scene, dict):
        return None
    rec = scene.get(obj)
    if not isinstance(rec, dict):
        return None
    center = np.asarray(rec.get("bbox_center", []), dtype=np.float64)
    size = np.asarray(rec.get("bbox_size", []), dtype=np.float64)
    if center.shape != (3,) or size.shape != (3,) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(size)):
        return None
    tw = np.asarray([fnum(row.get("target_world_x")), fnum(row.get("target_world_y")), fnum(row.get("target_world_z"))], dtype=np.float64)
    tp = target_vec(row)
    if not np.all(np.isfinite(tw)):
        return None
    ref_pelvis = tw - tp
    return {"center_rel": center - ref_pelvis, "half_diag": 0.5 * float(np.linalg.norm(size))}


def spatial_angle(joints: np.ndarray, hand: int, frame: int, row: Mapping[str, Any], scene_info: Mapping[str, Any] | None) -> float:
    frame = int(np.clip(frame, 0, len(joints) - 1))
    hn = "left" if hand == 0 else "right"
    pelvis = joints[frame, JOINT_INDEX["pelvis"]].astype(np.float64)
    elbow = joints[frame, JOINT_INDEX[f"{hn}_elbow"]].astype(np.float64)
    wrist = joints[frame, JOINT_INDEX[f"{hn}_wrist"]].astype(np.float64)
    fore = wrist - elbow
    if scene_info is not None:
        center = pelvis + np.asarray(scene_info["center_rel"], dtype=np.float64)
        margin = math.degrees(math.asin(np.clip(float(scene_info["half_diag"]) / max(float(np.linalg.norm(center - elbow)), 1e-8), 0.0, 1.0)))
    else:
        center = pelvis + target_vec(row)
        margin = 0.0
    denom = max(float(np.linalg.norm(fore) * np.linalg.norm(center - elbow)), 1e-8)
    theta = math.degrees(math.acos(float(np.clip(np.dot(fore, center - elbow) / denom, -1.0, 1.0))))
    return max(0.0, theta - margin)


def score20(angle: float) -> float:
    return float(math.exp(-(angle ** 2) / (2.0 * 20.0 ** 2)))


def distance_to_hold(frame: int, start: int, end: int) -> int:
    if frame < start:
        return start - frame
    if frame > end:
        return frame - end
    return 0


def temporal_score(distance: int, sigma: float = 5.0) -> float:
    return 1.0 if distance <= 0 else float(math.exp(-(distance ** 2) / (2.0 * sigma ** 2)))


def motion_quality(joints: np.ndarray, pose: np.ndarray, donor_pose: np.ndarray, hand: int) -> dict[str, float]:
    vel = central_gradient(joints) / FPS
    acc = central_gradient(vel) / FPS
    jerk = central_gradient(acc) / FPS
    joint_jerk = float(np.mean(np.linalg.norm(jerk[:, :22], axis=2)))
    feet = joints[:, [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]]].astype(np.float64)
    fvel = central_gradient(feet) / FPS
    min_y = np.min(feet[:, :, 1], axis=0)
    contact = feet[:, :, 1] <= (min_y[None, :] + 0.05)
    horiz = np.linalg.norm(fvel[:, :, [0, 2]], axis=2)
    foot_slide = float(np.sum(horiz * contact) / max(1, int(np.sum(contact))))
    aa = pose[:, 3:66].reshape(len(pose), -1, 3)
    limit_frac = float(np.mean(np.linalg.norm(aa, axis=2) > math.pi))
    pose_dev = float(np.mean(np.linalg.norm((pose - donor_pose).reshape(len(pose), -1, 3), axis=2)))
    hn = "left" if hand == 0 else "right"
    reach = np.linalg.norm(joints[:, JOINT_INDEX[f"{hn}_wrist"]] - joints[:, JOINT_INDEX[f"{hn}_shoulder"]], axis=1)
    return {"joint_jerk": joint_jerk, "foot_sliding": foot_slide, "joint_limit_fraction": limit_frac, "pose_deviation": pose_dev, "max_reach": float(np.max(reach))}


def evaluate_candidate(
    cand: Mapping[str, Any], joints: np.ndarray, row: Mapping[str, Any], scene_info: Mapping[str, Any] | None,
    source_model: Mapping[str, np.ndarray], refined_model: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    hand = inum(row.get("hand"))
    planned = inum(cand.get("planned_peak"))
    oracle = inum(row.get("released_peak_frame"))
    det = detector_predictions(joints, hand, source_model, refined_model)
    self_frame = int(det["ensemble"])
    hold_start = inum(row.get("hold_start"))
    hold_end = inum(row.get("hold_end"))
    ang_planned = spatial_angle(joints, hand, planned, row, scene_info)
    ang_self = spatial_angle(joints, hand, self_frame, row, scene_info)
    ang_oracle = spatial_angle(joints, hand, oracle, row, scene_info)
    dhold = distance_to_hold(self_frame, hold_start, hold_end)
    tscore = temporal_score(dhold, 5.0)
    q = motion_quality(joints, np.asarray(cand["pose"]), np.asarray(cand["donor_pose"]), hand)
    return {
        "planned_frame": planned, "self_frame": self_frame, "oracle_frame": oracle,
        "det_source": det["source"], "det_refined": det["refined"], "det_heuristic": det["heuristic"],
        "detector_agreement": det["agreement"], "distance_to_gt_hold": dhold,
        "temporal_proxy_sigma5": tscore,
        "planned_angle_deg": ang_planned, "planned_score1b_proxy20": score20(ang_planned),
        "self_angle_deg": ang_self, "self_score1b_proxy20": score20(ang_self),
        "self_stg_proxy_sigma5": tscore * score20(ang_self),
        "oracle_angle_deg": ang_oracle, "oracle_score1b_proxy20": score20(ang_oracle),
        **q,
    }


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def self_test() -> None:
    aa = np.asarray([[[0.1, 0.2, -0.1]], [[0.2, -0.1, 0.05]]], dtype=np.float64)
    q = axis_angle_to_quat(aa)
    back = quat_to_axis_angle(q)
    if not np.allclose(aa, back, atol=1e-6):
        raise PilotError("axis-angle quaternion roundtrip failed")
    pos = piecewise_positions(10, 4, 12, 6)
    if len(pos) != 12 or abs(pos[6] - 4) > 1e-9:
        raise PilotError("piecewise timing warp failed")
    w = hold_weights(21, 10, 5, 8)
    if not (w[10] == 1.0 and w[0] == 0.0 and np.all(w >= 0) and np.all(w <= 1)):
        raise PilotError("hold weights failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--p0c0-dir", type=Path, required=True)
    ap.add_argument("--smplx-model-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument(
        "--queries-per-slice", type=int, default=0,
        help="0 means all validation queries in every domain-hand slice",
    )
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--fk-batch-size", type=int, default=512)
    ap.add_argument("--ik-batch-size", type=int, default=64)
    ap.add_argument("--ik-steps", type=int, default=50)
    ap.add_argument("--ik-lr", type=float, default=0.05)
    ap.add_argument("--ik-max-delta", type=float, default=1.5)
    ap.add_argument("--hold-plateau", type=int, default=5)
    ap.add_argument("--hold-taper", type=int, default=8)
    ap.add_argument("--mlp-epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()
    for name in ("data_root", "project_root", "p0c0_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    self_test()
    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = args.p0c0_dir / "shared_generation_manifest.csv"
    accept_path = args.p0c0_dir / "P0C0_ACCEPTANCE.json"
    comparison_path = args.p0c0_dir / "comparison_protocol_v1_0_0.json"
    source_model_path = args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_0_0.npz"
    refined_model_path = args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_1_0.npz"
    for p in (manifest_path, accept_path, comparison_path, source_model_path, refined_model_path):
        if not p.is_file():
            raise PilotError(f"Missing frozen P0C0 input: {p}")
    accept = json_load(accept_path)
    if accept.get("status") != "P0C0_SHARED_PROTOCOL_FROZEN" or not accept.get("b0_allowed") or not accept.get("p0c_m1_allowed"):
        raise PilotError(f"P0C0 gate is not open: {accept}")
    rows = read_csv(manifest_path)
    if not rows:
        raise PilotError("Shared manifest is empty")
    row_by_id = {str(r["sample_id"]): r for r in rows}
    val_all = [r for r in rows if inum(r.get("fold")) == args.fold]
    train_rows = [r for r in rows if inum(r.get("fold")) != args.fold]
    if not val_all or not train_rows:
        raise PilotError(f"Fold {args.fold} has no train/validation rows")

    pilot_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    for domain in ("natural", "synthetic"):
        for hand in (0, 1):
            candidates = [r for r in val_all if str(r.get("domain")) == domain and inum(r.get("hand")) == hand]
            candidates = sorted(candidates, key=lambda r: stable_u64(str(r["sample_id"]), args.seed))
            take = candidates if args.queries_per_slice <= 0 else candidates[:args.queries_per_slice]
            if not take:
                raise PilotError(f"Empty validation slice domain={domain} hand={hand}")
            pilot_rows.extend(take)
            slice_rows.append({"domain": domain, "hand": hand, "available": len(candidates), "selected": len(take)})
    pilot_rows = sorted(pilot_rows, key=lambda r: (str(r.get("domain")), inum(r.get("hand")), str(r.get("sample_id"))))
    if args.queries_per_slice <= 0 and len(pilot_rows) != len(val_all):
        raise PilotError(f"Full-fold selection mismatch: selected={len(pilot_rows)} expected={len(val_all)}")
    write_csv(args.output_dir / "pilot_slice_manifest.csv", slice_rows)
    write_csv(args.output_dir / "pilot_queries.csv", pilot_rows)

    validation_generation_views: dict[str, dict[str, Any]] = {
        str(r["sample_id"]): generation_view(r) for r in pilot_rows
    }
    gt_usage_audit = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "status": "PASS",
        "fold": args.fold,
        "validation_query_count": len(pilot_rows),
        "prohibited_generation_keys": sorted(EVAL_ONLY_QUERY_KEYS),
        "all_generation_views_exclude_prohibited_keys": all(
            all(k not in validation_generation_views[str(r["sample_id"])] for k in EVAL_ONLY_QUERY_KEYS)
            for r in pilot_rows
        ),
        "released_labels_retained_only_for_post_generation_evaluation": True,
        "training_donor_peak_metadata_allowed": True,
        "note": "Validation peak/hold/alignment labels are stripped before timing, retrieval, controller input, warping destination, and IK. They are used only after generation for oracle/diagnostic evaluation.",
    }
    if not gt_usage_audit["all_generation_views_exclude_prohibited_keys"]:
        raise PilotError("Validation GT usage audit failed")
    json_dump(args.output_dir / "validation_gt_usage_audit.json", gt_usage_audit)

    assets = find_smplx_assets(args.smplx_model_dir, args.data_root, args.project_root)
    if assets.get("status") != "OK":
        raise PilotError(f"SMPL-X assets missing: {assets}")
    engine = SMPLXEngine(assets, args.device, args.fk_batch_size)
    device = engine.device
    schema_cache: dict[Path, dict[str, Any]] = {}
    clip_cache: dict[str, dict[str, Any]] = {}

    medians: dict[tuple[str, int], float] = {}
    for domain in ("natural", "synthetic"):
        for hand in (0, 1):
            vals = [fnum(r.get("released_peak_ratio"), 0.5) for r in train_rows if str(r.get("domain")) == domain and inum(r.get("hand")) == hand]
            medians[(domain, hand)] = float(np.median(vals)) if vals else 0.5
    text_idx = transcript_index(args.data_root / "data" / "text")
    text_cache: dict[Path, list[tuple[float, float, str]]] = {}
    timing: dict[str, dict[str, Any]] = {}
    timing_rows: list[dict[str, Any]] = []
    for r in train_rows:
        sid = str(r["sample_id"])
        key = (str(r.get("domain")), inum(r.get("hand")))
        rec = timing_plan(r, text_idx, medians[key], text_cache)
        timing[sid] = rec
        timing_rows.append({"sample_id": sid, "fold": inum(r.get("fold")), "domain": r.get("domain"), "hand": inum(r.get("hand")), "role": "training", **rec})
    for r_eval in pilot_rows:
        sid = str(r_eval["sample_id"])
        r_gen = validation_generation_views[sid]
        key = (str(r_gen.get("domain")), inum(r_gen.get("hand")))
        rec = timing_plan(r_gen, text_idx, medians[key], text_cache)
        timing[sid] = rec
        timing_rows.append({"sample_id": sid, "fold": inum(r_gen.get("fold")), "domain": r_gen.get("domain"), "hand": inum(r_gen.get("hand")), "role": "validation_generation", **rec})
    write_csv(args.output_dir / "timing_manifest.csv", timing_rows)

    # Extract training apex frames and FK labels for the B0 controller.
    frame_data: dict[str, dict[str, Any]] = {}
    frame_records: list[dict[str, Any]] = []
    for i, r in enumerate(train_rows):
        sid = str(r["sample_id"])
        fr = load_clip_frame(r, inum(r.get("released_peak_frame")), schema_cache)
        frame_data[sid] = fr
        frame_records.append({"sid": sid, "pose": np.asarray(fr["pose"], dtype=np.float32), "trans": np.asarray(fr["trans"], dtype=np.float32), "betas": np.asarray(fr["betas"], dtype=np.float32), "gender": choose_gender(fr.get("gender"), r.get("gender"))})
        if (i + 1) % 250 == 0:
            print(f"[B0 labels] loaded {i+1}/{len(train_rows)} peak frames", flush=True)
    peak_clips = [{"pose": x["pose"][None, :], "trans": x["trans"][None, :], "betas": x["betas"][None, :], "gender": x["gender"]} for x in frame_records]
    peak_joint_list = engine.joints_many(peak_clips)
    peak_joints = {frame_records[i]["sid"]: peak_joint_list[i][0] for i in range(len(frame_records))}
    b0_models, b0_history = train_b0_models(train_rows, timing, frame_data, peak_joints, device, args.mlp_epochs, args.seed, args.output_dir)
    write_csv(args.output_dir / "b0_training_history.csv", b0_history)

    source_model = load_linear_model(source_model_path)
    refined_model = load_linear_model(refined_model_path)
    scene_cache: dict[str, Any] = {}

    donors_by_hand: dict[int, list[dict[str, Any]]] = {}
    donor_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for hand in (0, 1):
        donors = [r for r in train_rows if inum(r.get("hand")) == hand and str(r.get("domain")) == "natural"]
        if len(donors) < args.top_k:
            raise PilotError(f"Not enough natural donors for hand {hand}")
        D = np.stack([descriptor(r) for r in donors])
        mu, sd = D.mean(axis=0), D.std(axis=0)
        sd[sd < 1e-6] = 1.0
        donors_by_hand[hand] = donors
        donor_stats[hand] = (mu, sd)

    candidate_records: list[dict[str, Any]] = []
    query_context: dict[str, dict[str, Any]] = {}
    retrieval_rows: list[dict[str, Any]] = []
    for qi, q_eval in enumerate(pilot_rows):
        sid = str(q_eval["sample_id"])
        q = validation_generation_views[sid]
        hand = inum(q.get("hand"))
        qframe = load_clip_frame(q, 0, schema_cache)
        beta = np.asarray(qframe["betas"], dtype=np.float32)[:10]
        gender = choose_gender(qframe.get("gender"), q.get("gender"))
        plan = timing[sid]
        planned_peak = int(plan["planned_peak"])
        qdesc = descriptor(q, float(plan["planned_peak_ratio"]))
        donors = donors_by_hand[hand]
        mu, sd = donor_stats[hand]
        D = np.stack([descriptor(r) for r in donors])
        d_full = np.linalg.norm((D - qdesc[None, :]) / sd[None, :], axis=1)
        top_ids = np.argsort(d_full, kind="mergesort")[:args.top_k]
        # B0 template selection only matches duration and planned peak ratio, not target geometry.
        # The primary baseline uses one deterministic donor; an equal-budget diagnostic uses top-K.
        d_timing = np.linalg.norm(((D[:, 3:5] - qdesc[None, 3:5]) / sd[None, 3:5]), axis=1)
        top_timing_ids = np.argsort(d_timing, kind="mergesort")[:args.top_k]
        b0_idx = int(top_timing_ids[0])
        b0_donor = donors[b0_idx]
        x = b0_feature(q, beta, float(plan["planned_peak_ratio"]))
        anchors = b0_models[hand].predict(x, device).astype(np.float32)
        query_context[sid] = {"beta": beta, "gender": gender, "planned_peak": planned_peak, "anchors": anchors, "row": q, "eval_row": q_eval}
        candidate_records.append({"method": "b0", "query_id": sid, "donor_id": str(b0_donor["sample_id"]), "donor_rank": 1, "donor_distance": float(d_timing[b0_idx]), "row": q, "eval_row": q_eval, "donor_row": b0_donor, "beta": beta, "gender": gender, "planned_peak": planned_peak, "anchors": anchors})
        retrieval_rows.append({"method": "b0", "query_id": sid, "donor_id": b0_donor["sample_id"], "rank": 1, "distance": float(d_timing[b0_idx]), "uses_target_geometry_for_retrieval": 0, "candidate_budget": 1})
        for rank, idx in enumerate(top_timing_ids, 1):
            donor = donors[int(idx)]
            candidate_records.append({"method": "b0eq", "query_id": sid, "donor_id": str(donor["sample_id"]), "donor_rank": rank, "donor_distance": float(d_timing[int(idx)]), "row": q, "eval_row": q_eval, "donor_row": donor, "beta": beta, "gender": gender, "planned_peak": planned_peak, "anchors": anchors})
            retrieval_rows.append({"method": "b0eq", "query_id": sid, "donor_id": donor["sample_id"], "rank": rank, "distance": float(d_timing[int(idx)]), "uses_target_geometry_for_retrieval": 0, "candidate_budget": args.top_k})
        for rank, idx in enumerate(top_ids, 1):
            donor = donors[int(idx)]
            candidate_records.append({"method": "p0c", "query_id": sid, "donor_id": str(donor["sample_id"]), "donor_rank": rank, "donor_distance": float(d_full[int(idx)]), "row": q, "eval_row": q_eval, "donor_row": donor, "beta": beta, "gender": gender, "planned_peak": planned_peak, "anchors": None})
            retrieval_rows.append({"method": "p0c", "query_id": sid, "donor_id": donor["sample_id"], "rank": rank, "distance": float(d_full[int(idx)]), "uses_target_geometry_for_retrieval": 1, "candidate_budget": args.top_k})
        if (qi + 1) % 20 == 0:
            print(f"[retrieval] prepared {qi+1}/{len(pilot_rows)} queries", flush=True)
    write_csv(args.output_dir / "retrieval_manifest.csv", retrieval_rows)

    # Load and warp donor motions.
    for i, c in enumerate(candidate_records):
        did = str(c["donor_id"])
        if did not in clip_cache:
            clip_cache[did] = load_full_clip(row_by_id[did], schema_cache)
        donor_clip = clip_cache[did]
        drow = c["donor_row"]
        warped = warp_clip(donor_clip, inum(drow.get("released_peak_frame")), inum(c["row"].get("clip_length")), int(c["planned_peak"]))
        c["base_pose"] = warped["pose"]
        c["base_trans"] = warped["trans"]
        c["base_apex"] = warped["pose"][int(c["planned_peak"])].copy()
        c["query_betas"] = np.asarray(c["beta"], dtype=np.float32)
        if (i + 1) % 100 == 0:
            print(f"[warp] prepared {i+1}/{len(candidate_records)} candidates", flush=True)

    # Batched differentiable apex optimization.
    ik_logs: list[dict[str, Any]] = []
    for method in ("b0", "b0eq", "p0c"):
        for hand in (0, 1):
            for gender in ("neutral", "male", "female"):
                ids = [i for i, c in enumerate(candidate_records) if c["method"] == method and inum(c["row"].get("hand")) == hand and normalize_gender(c["gender"]) == gender]
                if not ids:
                    continue
                for a in range(0, len(ids), args.ik_batch_size):
                    take = ids[a:a + args.ik_batch_size]
                    base = np.stack([candidate_records[i]["base_apex"] for i in take]).astype(np.float32)
                    betas = np.stack([candidate_records[i]["query_betas"] for i in take]).astype(np.float32)
                    ik_mode = "p0c" if method == "p0c" else "b0"
                    targets = np.stack([target_vec(candidate_records[i]["row"]) for i in take]).astype(np.float32) if ik_mode == "p0c" else None
                    anchors = np.stack([candidate_records[i]["anchors"] for i in take]).astype(np.float32) if ik_mode == "b0" else None
                    opt_pose, logs = optimize_apex_batch(engine, base, betas, gender, hand, ik_mode, targets, anchors, args.ik_steps, args.ik_lr, args.ik_max_delta)
                    for local, idx in enumerate(take):
                        c = candidate_records[idx]
                        c["optimized_apex"] = opt_pose[local]
                        c["pose"] = apply_hold(c["base_pose"], int(c["planned_peak"]), opt_pose[local], hand, args.hold_plateau, args.hold_taper)
                        c["trans"] = c["base_trans"].copy()
                        c["betas_full"] = np.broadcast_to(c["query_betas"][None, :], (len(c["pose"]), len(c["query_betas"]))).copy()
                        c["donor_pose"] = c["base_pose"].copy()
                    for rec in logs:
                        ik_logs.append({"method": method, "hand": hand, "gender": gender, "batch_start": a, **rec})
                print(f"[IK] method={method} hand={hand} gender={gender} candidates={len(ids)}", flush=True)
    write_csv(args.output_dir / "ik_training_log.csv", ik_logs)

    eval_clips = [{"pose": c["pose"], "trans": c["trans"], "betas": c["betas_full"], "gender": c["gender"]} for c in candidate_records]
    eval_joints = engine.joints_many(eval_clips)
    candidate_metrics: list[dict[str, Any]] = []
    for i, c in enumerate(candidate_records):
        q = c["eval_row"]
        scene = resolve_scene(args.data_root, q, scene_cache)
        m = evaluate_candidate(c, eval_joints[i], q, scene, source_model, refined_model)
        c["metrics"] = m
        candidate_metrics.append({
            "method": c["method"], "query_id": c["query_id"], "donor_id": c["donor_id"],
            "donor_rank": c["donor_rank"], "donor_distance": c["donor_distance"],
            "fold": inum(q.get("fold")), "domain": q.get("domain"), "hand": inum(q.get("hand")),
            **m,
        })
    write_csv(args.output_dir / "candidate_metrics.csv", candidate_metrics)

    def rerank_key(c: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            -fnum(c["metrics"].get("planned_score1b_proxy20"), -1),
            -fnum(c["metrics"].get("self_score1b_proxy20"), -1),
            -fnum(c["metrics"].get("detector_agreement"), -1),
            fnum(c["metrics"].get("joint_jerk"), 1e9),
            fnum(c.get("donor_distance"), 1e9),
        )

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for q in pilot_rows:
        sid = str(q["sample_id"])
        b0 = next(c for c in candidate_records if c["method"] == "b0" and c["query_id"] == sid)
        b0eqs = [c for c in candidate_records if c["method"] == "b0eq" and c["query_id"] == sid]
        p0cs = [c for c in candidate_records if c["method"] == "p0c" and c["query_id"] == sid]
        b0eqs.sort(key=rerank_key)
        p0cs.sort(key=rerank_key)
        selected[(sid, "b0")] = b0
        selected[(sid, "b0eq")] = b0eqs[0]
        selected[(sid, "p0c")] = p0cs[0]

    motion_root = args.output_dir / "motions"
    (motion_root / "b0").mkdir(parents=True, exist_ok=True)
    (motion_root / "b0_equal_budget").mkdir(parents=True, exist_ok=True)
    (motion_root / "p0c").mkdir(parents=True, exist_ok=True)
    per_sample: list[dict[str, Any]] = []
    schema_errors: list[dict[str, Any]] = []
    for q in pilot_rows:
        sid = str(q["sample_id"])
        tv = target_vec(q)
        td = float(np.linalg.norm(tv))
        taz = math.degrees(math.atan2(float(tv[0]), float(tv[2])))
        tel = math.degrees(math.atan2(float(tv[1]), max(math.hypot(float(tv[0]), float(tv[2])), 1e-8)))
        distance_bin = "near_lt2" if td < 2.0 else ("mid_2_3p5" if td <= 3.5 else "far_gt3p5")
        azimuth_bin = "front_abslt30" if abs(taz) < 30.0 else ("side_30_60" if abs(taz) <= 60.0 else "extreme_absgt60")
        elevation_bin = "low_ltminus15" if tel < -15.0 else ("mid_minus15_15" if tel <= 15.0 else "high_gt15")
        row_out: dict[str, Any] = {
            "sample_id": sid, "recording_id": q.get("recording_id", ""),
            "leakage_group_id": q.get("leakage_group_id", q.get("group_id", "")),
            "fold": inum(q.get("fold")), "domain": q.get("domain"),
            "hand": inum(q.get("hand")), "clip_length": inum(q.get("clip_length")),
            "timing_source": timing[sid]["timing_source"], "planned_peak": timing[sid]["planned_peak"],
            "target_distance": td, "target_azimuth_deg": taz, "target_elevation_deg": tel,
            "distance_bin": distance_bin, "azimuth_bin": azimuth_bin, "elevation_bin": elevation_bin,
        }
        for method in ("b0", "b0eq", "p0c"):
            c = selected[(sid, method)]
            method_dir = "b0_equal_budget" if method == "b0eq" else method
            path = motion_root / method_dir / f"{sid}.npz"
            pose = np.asarray(c["pose"], dtype=np.float32)
            trans = np.asarray(c["trans"], dtype=np.float32)
            np.savez_compressed(path, poses=pose, trans=trans)
            valid = pose.shape == (inum(q.get("clip_length")), 165) and trans.shape == (inum(q.get("clip_length")), 3) and np.all(np.isfinite(pose)) and np.all(np.isfinite(trans))
            if not valid:
                schema_errors.append({"sample_id": sid, "method": method, "pose_shape": str(pose.shape), "trans_shape": str(trans.shape)})
            row_out[f"{method}_donor_id"] = c["donor_id"]
            row_out[f"{method}_donor_rank"] = c["donor_rank"]
            row_out[f"{method}_output"] = str(path)
            for k, v in c["metrics"].items():
                row_out[f"{method}_{k}"] = v
        per_sample.append(row_out)
    write_csv(args.output_dir / "per_sample_comparison.csv", per_sample)
    write_csv(args.output_dir / "schema_errors.csv", schema_errors)

    metric_names = [
        "self_score1b_proxy20", "self_stg_proxy_sigma5", "oracle_score1b_proxy20",
        "planned_score1b_proxy20", "self_angle_deg", "oracle_angle_deg", "joint_jerk",
        "foot_sliding", "joint_limit_fraction", "pose_deviation", "detector_agreement",
    ]
    summary_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "overall", per_sample)]
    for domain in ("natural", "synthetic"):
        groups.append(("domain", domain, [r for r in per_sample if str(r.get("domain")) == domain]))
    for hand in (0, 1):
        groups.append(("hand", str(hand), [r for r in per_sample if inum(r.get("hand")) == hand]))
    for domain in ("natural", "synthetic"):
        for hand in (0, 1):
            groups.append(("domain_hand", f"{domain}_hand{hand}", [
                r for r in per_sample if str(r.get("domain")) == domain and inum(r.get("hand")) == hand
            ]))
    for field in ("distance_bin", "azimuth_bin", "elevation_bin"):
        for value in sorted({str(r.get(field)) for r in per_sample}):
            groups.append((field, value, [r for r in per_sample if str(r.get(field)) == value]))

    comparisons = [
        ("p0c_vs_b0_primary", "b0"),
        ("p0c_vs_b0_equal_budget", "b0eq"),
    ]
    for comparison, baseline in comparisons:
        for gname, gvalue, records in groups:
            if not records:
                continue
            for metric in metric_names:
                b = np.asarray([fnum(r.get(f"{baseline}_{metric}")) for r in records], dtype=np.float64)
                p = np.asarray([fnum(r.get(f"p0c_{metric}")) for r in records], dtype=np.float64)
                valid = np.isfinite(b) & np.isfinite(p)
                b, p = b[valid], p[valid]
                if len(b) == 0:
                    continue
                diff = p - b
                lo, hi = bootstrap_ci(
                    diff, args.seed + stable_u64(f"{comparison}:{gname}:{gvalue}:{metric}") % 100000
                )
                summary_rows.append({
                    "comparison": comparison, "baseline_method": baseline,
                    "group_name": gname, "group_value": gvalue, "metric": metric, "n": len(diff),
                    "baseline_mean": float(np.mean(b)), "p0c_mean": float(np.mean(p)),
                    "paired_difference_p0c_minus_baseline": float(np.mean(diff)),
                    "bootstrap95_low": lo, "bootstrap95_high": hi,
                    "positive_fraction": float(np.mean(diff > 0)),
                })
    write_csv(args.output_dir / "summary_metrics.csv", summary_rows)

    def get_summary(comparison: str, group_name: str, group_value: str, metric: str) -> dict[str, Any]:
        return next(
            r for r in summary_rows
            if r["comparison"] == comparison and r["group_name"] == group_name
            and r["group_value"] == group_value and r["metric"] == metric
        )

    primary_comparison = "p0c_vs_b0_primary"
    equal_comparison = "p0c_vs_b0_equal_budget"
    overall_s1b = get_summary(primary_comparison, "overall", "overall", "self_score1b_proxy20")
    overall_stg = get_summary(primary_comparison, "overall", "overall", "self_stg_proxy_sigma5")
    overall_jerk = get_summary(primary_comparison, "overall", "overall", "joint_jerk")
    equal_s1b = get_summary(equal_comparison, "overall", "overall", "self_score1b_proxy20")
    equal_stg = get_summary(equal_comparison, "overall", "overall", "self_stg_proxy_sigma5")

    slice_positive = 0
    slice_total = 0
    for domain in ("natural", "synthetic"):
        for hand in (0, 1):
            records = [r for r in per_sample if str(r.get("domain")) == domain and inum(r.get("hand")) == hand]
            if records:
                d = np.mean([fnum(r["p0c_self_score1b_proxy20"]) - fnum(r["b0_self_score1b_proxy20"]) for r in records])
                slice_total += 1
                slice_positive += int(d > 0)

    engineering_gate = (
        not schema_errors
        and len(per_sample) == len(pilot_rows)
        and len({str(r["sample_id"]) for r in per_sample}) == len(pilot_rows)
        and (args.queries_per_slice > 0 or len(pilot_rows) == len(val_all))
        and gt_usage_audit["status"] == "PASS"
    )
    directional_gate = (
        fnum(overall_s1b["paired_difference_p0c_minus_baseline"], -1) > 0
        and fnum(overall_stg["paired_difference_p0c_minus_baseline"], -1) > 0
        and slice_positive >= 3
    )
    quality_limits = {
        "joint_jerk": (1.20, 0.0001),
        "foot_sliding": (1.25, 0.0005),
        "joint_limit_fraction": (1.00, 0.005),
        "pose_deviation": (1.25, 0.002),
    }
    quality_components: dict[str, Any] = {}
    for metric, (ratio, additive) in quality_limits.items():
        rec = get_summary(primary_comparison, "overall", "overall", metric)
        baseline_mean = fnum(rec.get("baseline_mean"), 0.0)
        p0c_mean = fnum(rec.get("p0c_mean"), 1e9)
        limit = max(ratio * baseline_mean, baseline_mean + additive)
        quality_components[metric] = {
            "baseline_mean": baseline_mean, "p0c_mean": p0c_mean,
            "allowed_max": limit, "pass": bool(p0c_mean <= limit),
        }
    quality_gate = all(bool(v["pass"]) for v in quality_components.values())
    if engineering_gate:
        status = "B0_P0C_M1_FULL_FOLD_COMPLETED"
    else:
        status = "B0_P0C_M1_FULL_FOLD_ENGINEERING_FAILED"
    full_allowed = False
    acceptance = {
        "schema_version": SCHEMA_VERSION, "version": VERSION, "created_utc": now_utc(),
        "status": status, "run_mode": "full_validation_fold" if args.queries_per_slice <= 0 else "bounded_subset",
        "fold": args.fold, "validation_query_count": len(pilot_rows), "pilot_query_count": len(pilot_rows),
        "candidate_count": len(candidate_records), "top_k": args.top_k,
        "engineering_gate": engineering_gate, "directional_gate": directional_gate,
        "quality_gate": quality_gate, "quality_components": quality_components,
        "positive_domain_hand_slices": slice_positive,
        "domain_hand_slice_count": slice_total, "full_5fold_allowed": full_allowed,
        "validation_gt_usage_audit": gt_usage_audit,
        "primary_differences": {
            "self_score1b_proxy20": overall_s1b,
            "self_stg_proxy_sigma5": overall_stg,
            "joint_jerk": overall_jerk,
        },
        "equal_budget_diagnostic": {
            "self_score1b_proxy20": equal_s1b,
            "self_stg_proxy_sigma5": equal_stg,
            "note": "Diagnostic only: both B0 and P0C receive top-K donor candidates and the same reranking signals.",
        },
        "guardrails": [
            "This is one held-out fold of the full five-fold OOF experiment; only the aggregate can satisfy the frozen scientific gate.",
            "B0 is OmniControl-PT-style, not an exact reproduction of unavailable official code or weights.",
            "P0B detector ensemble is advisory and is not represented as the private organizer detector.",
            "No scene graph, residual flow, or listener state is used in P0C-M1.",
        ],
        "next_action": "Aggregate all five completed folds under the frozen comparison protocol.",
    }
    json_dump(args.output_dir / "B0_P0C_M1_ACCEPTANCE.json", acceptance)

    source_bindings = []
    for p in (manifest_path, accept_path, comparison_path, source_model_path, refined_model_path):
        source_bindings.append({"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    json_dump(args.output_dir / "source_binding.json", source_bindings)
    env = {
        "created_utc": now_utc(), "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable, "device": device,
        "smplx_assets": assets, "fk_frames_processed": engine.frames_processed,
    }
    json_dump(args.output_dir / "environment.json", env)

    summary = [
        "REPOINTER B0 / P0C-M1 — FULL HELD-OUT FOLD GENERATION",
        "=" * 78,
        f"version={VERSION}", f"created_utc={now_utc()}",
        f"data_root={args.data_root}", f"p0c0_dir={args.p0c0_dir}", f"output_dir={args.output_dir}",
        "dataset_writes=0", "downloads=0", "git_writes=0", "official_test_inference=0", "",
        "[FOLD]", f"fold={args.fold}", f"queries={len(pilot_rows)}", f"all_fold_queries={args.queries_per_slice <= 0}", f"top_k={args.top_k}",
        f"candidate_count={len(candidate_records)}", f"device={device}", f"fk_frames_processed={engine.frames_processed}", "",
        "[B0]", "name=OmniControl-PT-style modular target-control internal baseline",
        "timing=last transcript word end minus 11 frames, fold-training median fallback",
        "controller=small MLP predicts pelvis-relative active elbow/wrist anchors",
        "backend=timing-matched natural donor + shared differentiable SMPL-X IK/hold synthesis",
        f"candidate_budget_primary=1 diagnostic_equal_budget={args.top_k}", "",
        "[P0C-M1]", "name=RePointer-v0 retrieval + shape-aware IK",
        "retrieval=top-K natural same-hand donors from training folds only",
        "ik=target-ray alignment with query SMPL-X shape and clear plateau hold",
        "excluded=scene graph,residual flow,listener state", "",
        "[RESULT]", f"status={status}", f"engineering_gate={engineering_gate}",
        f"directional_gate={directional_gate}", f"quality_gate={quality_gate}",
        f"self_score1b_difference={overall_s1b['paired_difference_p0c_minus_baseline']}",
        f"self_stg_difference={overall_stg['paired_difference_p0c_minus_baseline']}",
        f"equal_budget_self_score1b_difference={equal_s1b['paired_difference_p0c_minus_baseline']}",
        f"equal_budget_self_stg_difference={equal_stg['paired_difference_p0c_minus_baseline']}",
        f"positive_domain_hand_slices={slice_positive}/{slice_total}",
        "aggregate_decision_required=TRUE", "",
        "[PRIMARY_ARTIFACTS]", "B0_P0C_M1_ACCEPTANCE.json", "validation_gt_usage_audit.json", "per_sample_comparison.csv",
        "summary_metrics.csv", "candidate_metrics.csv", "retrieval_manifest.csv", "timing_manifest.csv",
        "motions/b0/*.npz", "motions/b0_equal_budget/*.npz", "motions/p0c/*.npz", "b0_controller_hand0.pt", "b0_controller_hand1.pt",
    ]
    (args.output_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    run_manifest: list[dict[str, Any]] = []
    for p in sorted(args.output_dir.rglob("*")):
        if p.is_file() and p.name != "run_file_manifest.json":
            run_manifest.append({"relative_path": str(p.relative_to(args.output_dir)), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    json_dump(args.output_dir / "run_file_manifest.json", run_manifest)
    print("\n".join(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise

