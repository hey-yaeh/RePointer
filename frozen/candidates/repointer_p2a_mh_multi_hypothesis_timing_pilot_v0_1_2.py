#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

VERSION = "0.1.2"
SCHEMA_VERSION = "1.0.0"
FPS = 30.0
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
# Frozen replay detail: accepted P1A evaluated the un-packed bidirectional
# GRU in batches of 128. Right-padding affects the backward GRU state, so this
# replay batch size is part of the frozen computational contract.
FROZEN_P1A_REPLAY_BATCH_SIZE = 128
DEICTIC = {"this", "that", "these", "those", "here", "there", "it", "one"}
SPATIAL = {"left", "right", "front", "behind", "back", "above", "below", "under", "over", "next", "near", "beside", "between", "on", "in", "at", "corner", "side"}
FUNCTION = {"the", "a", "an", "to", "of", "is", "are", "was", "were", "and", "or", "with", "you", "can", "could", "please"}
SMOOTH_KERNEL = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
SMOOTH_KERNEL /= SMOOTH_KERNEL.sum()
PROHIBITED_INFERENCE_TOKENS = (
    "gt_peak", "hold_start", "hold_end", "hold_width", "alignment",
    "oracle", "distance_to_gt_hold", "timing_temporal", "temporal_proxy",
    "released_peak",
)


class MHError(RuntimeError):
    pass


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def is_blank(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def fnum(x: Any, default: float = float("nan")) -> float:
    if is_blank(x):
        return default
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def inum(x: Any, default: int = -1) -> int:
    v = fnum(x)
    return int(round(v)) if np.isfinite(v) else default


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
        for key in row:
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


def stable_seed(text: str, base: int) -> int:
    h = hashlib.sha256(f"{base}:{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32 - 1)


def load_module(path: Path, name: str = "repointer_m1_frozen_worker") -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MHError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def function_ast_hash(path: Path, function_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function_name]
    if len(matches) != 1:
        raise MHError(f"Expected exactly one {function_name} in {path}, found {len(matches)}")
    return hashlib.sha256(ast.dump(matches[0], include_attributes=False).encode()).hexdigest()


def token_normalize(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"^[^\w']+|[^\w']+$", "", x, flags=re.UNICODE)
    return x or "<empty>"


def token_id(token: str, buckets: int) -> int:
    if token == "<no_word>":
        return 1
    h = hashlib.blake2b(token.encode("utf-8", errors="replace"), digest_size=8).digest()
    return 2 + (int.from_bytes(h, "little") % buckets)


def parse_word_times(path: Path) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip().lstrip("|").strip()
        if not line:
            continue
        parts = re.split(r"\s+|\t+", line, maxsplit=2)
        if len(parts) < 2:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
            token = parts[2].strip() if len(parts) >= 3 else ""
        except Exception:
            nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", line)
            if len(nums) < 2:
                continue
            a, b = float(nums[0]), float(nums[1])
            after = line.find(nums[1]) + len(nums[1])
            token = line[after:].strip(" \t|")
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if b < a:
            a, b = b, a
        out.append((a, b, token_normalize(token)))
    out.sort(key=lambda z: (z[0], z[1], z[2]))
    return out


@dataclass
class InferenceExample:
    sample_id: str
    fold: int
    domain: str
    hand: int
    clip_length: int
    token_ids: np.ndarray
    word_features: np.ndarray
    base_clip_features: np.ndarray
    transcript_path: str
    transcript_status: str
    word_count: int


class WordPhaseNet:
    def __init__(self, torch_mod: Any, buckets: int, word_dim: int, clip_dim: int, bins: int):
        torch = torch_mod
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(buckets + 2, 24, padding_idx=0)
                self.word_proj = nn.Sequential(nn.Linear(24 + word_dim, 64), nn.SiLU(), nn.LayerNorm(64))
                self.gru = nn.GRU(64, 64, num_layers=1, batch_first=True, bidirectional=True)
                self.att = nn.Linear(128, 1)
                self.clip_proj = nn.Sequential(nn.Linear(clip_dim, 64), nn.SiLU(), nn.LayerNorm(64))
                self.head = nn.Sequential(nn.Linear(192, 128), nn.SiLU(), nn.Dropout(0.10), nn.Linear(128, bins))

            def forward(self, tok: Any, wf: Any, mask: Any, cf: Any) -> Any:
                x = self.word_proj(torch.cat([self.emb(tok), wf], dim=-1))
                x, _ = self.gru(x)
                a = self.att(x).squeeze(-1)
                a = a.masked_fill(~mask, -1e4)
                a = torch.softmax(a, dim=1)
                pooled = torch.sum(x * a[:, :, None], dim=1)
                return self.head(torch.cat([pooled, self.clip_proj(cf)], dim=-1))

        self.net = Net()


def torch_load_checkpoint(path: Path, device: str = "cpu") -> Mapping[str, Any]:
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_inference_example(
    rec: Mapping[str, Any], timing_rec: Mapping[str, Any], buckets: int, max_words: int,
    word_cache: dict[Path, list[tuple[float, float, str]]],
) -> InferenceExample:
    sid = str(rec["sample_id"])
    domain = str(rec.get("domain", ""))
    T = inum(rec.get("clip_length"))
    fixed = inum(timing_rec.get("planned_peak"))
    if T <= 4 or not (0 <= fixed < T):
        raise MHError(f"Invalid inference timing geometry for {sid}: T={T}, fixed={fixed}")
    stated = str(timing_rec.get("transcript_path", "")).strip()
    path = Path(stated) if stated else Path("/nonexistent")
    words = word_cache.setdefault(path, parse_word_times(path)) if path.is_file() else []
    gstart = inum(rec.get("global_start"), -1)
    clip_start = (gstart / FPS) if domain == "natural" and gstart >= 0 else 0.0
    clip_end = clip_start + T / FPS
    selected: list[tuple[float, float, str]] = []
    for a, b, w in words:
        if b >= clip_start - 0.05 and a <= clip_end + 0.05:
            ar = max(0.0, a - clip_start)
            br = min(T / FPS, b - clip_start)
            if br >= -0.05 and ar <= T / FPS + 0.05:
                selected.append((max(0.0, ar), max(0.0, min(T / FPS, br)), w))
    selected.sort(key=lambda z: (z[0], z[1]))
    if len(selected) > max_words:
        ids = np.linspace(0, len(selected) - 1, max_words).round().astype(int)
        selected = [selected[int(i)] for i in ids]

    clip_sec = max(T / FPS, 1e-6)
    toks: list[int] = []
    feats: list[list[float]] = []
    pauses: list[float] = []
    durations: list[float] = []
    n = len(selected)
    for i, (a, b, w) in enumerate(selected):
        prev_end = selected[i - 1][1] if i > 0 else 0.0
        next_start = selected[i + 1][0] if i + 1 < n else clip_sec
        pause_before = max(0.0, a - prev_end)
        pause_after = max(0.0, next_start - b)
        dur = max(0.0, b - a)
        pauses.extend([pause_before, pause_after])
        durations.append(dur)
        toks.append(token_id(w, buckets))
        feats.append([
            a / clip_sec, b / clip_sec, dur / clip_sec,
            pause_before / clip_sec, pause_after / clip_sec,
            i / max(1, n - 1), min(len(w), 24) / 24.0,
            float(w in DEICTIC), float(w in SPATIAL), float(w in FUNCTION),
        ])
    if not toks:
        toks = [token_id("<no_word>", buckets)]
        feats = [[0.0] * 10]

    v = np.asarray([
        fnum(rec.get("target_pelvis_x"), 0.0),
        fnum(rec.get("target_pelvis_y"), 0.0),
        fnum(rec.get("target_pelvis_z"), 1.0),
    ], dtype=np.float64)
    dist = float(np.linalg.norm(v))
    u = v / max(dist, 1e-8)
    first_start = selected[0][0] / clip_sec if selected else 0.0
    last_end = selected[-1][1] / clip_sec if selected else 0.0
    mean_dur = float(np.mean(durations) / clip_sec) if durations else 0.0
    mean_pause = float(np.mean(pauses) / clip_sec) if pauses else 0.0
    max_pause = float(np.max(pauses) / clip_sec) if pauses else 0.0
    fixed_ratio = fixed / max(1, T - 1)
    base_clip = np.asarray([
        1.0 if domain == "natural" else 0.0,
        float(inum(rec.get("hand"))), T / 180.0, math.log(max(dist, 1e-4)),
        float(u[0]), float(u[1]), float(u[2]),
        min(n, max_words) / max(1, max_words),
        first_start, last_end, max(0.0, last_end - first_start),
        mean_dur, mean_pause, max_pause, fixed_ratio,
    ], dtype=np.float32)
    return InferenceExample(
        sample_id=sid, fold=inum(rec.get("fold")), domain=domain,
        hand=inum(rec.get("hand")), clip_length=T,
        token_ids=np.asarray(toks, dtype=np.int64),
        word_features=np.asarray(feats, dtype=np.float32),
        base_clip_features=base_clip,
        transcript_path=str(path) if path.is_file() else "",
        transcript_status=str(timing_rec.get("transcript_match", "missing")),
        word_count=n,
    )


def collate_inference(
    batch: Sequence[InferenceExample], medians: Mapping[tuple[str, int], float],
    word_mean: np.ndarray, word_std: np.ndarray, clip_mean: np.ndarray, clip_std: np.ndarray,
) -> tuple[Any, Any, Any, Any]:
    import torch
    B = len(batch)
    L = max(len(x.token_ids) for x in batch)
    D = batch[0].word_features.shape[1]
    tok = np.zeros((B, L), dtype=np.int64)
    wf = np.zeros((B, L, D), dtype=np.float32)
    mask = np.zeros((B, L), dtype=bool)
    cf = np.zeros((B, len(batch[0].base_clip_features) + 1), dtype=np.float32)
    for i, ex in enumerate(batch):
        n = len(ex.token_ids)
        tok[i, :n] = ex.token_ids
        wf[i, :n] = (ex.word_features - word_mean) / word_std
        mask[i, :n] = True
        raw_cf = np.concatenate([ex.base_clip_features, np.asarray([medians[(ex.domain, ex.hand)]], dtype=np.float32)])
        cf[i] = (raw_cf - clip_mean) / clip_std
    return torch.from_numpy(tok), torch.from_numpy(wf), torch.from_numpy(mask), torch.from_numpy(cf)


def infer_wordphase_distributions(
    rows: Sequence[Mapping[str, Any]], timing_by_id: Mapping[str, Mapping[str, Any]],
    pred_by_id: Mapping[str, Mapping[str, Any]], p1a_dir: Path,
    device: str, batch_size: int, max_words: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    import torch
    word_cache: dict[Path, list[tuple[float, float, str]]] = {}
    out: dict[str, dict[str, Any]] = {}
    regression_diffs: list[int] = []
    regression_rows: list[dict[str, Any]] = []
    transcript_counts: Counter[str] = Counter()
    for fold in EXPECTED_FOLDS:
        ckpt_path = p1a_dir / f"wordphasenet_fold{fold}.pt"
        state = torch_load_checkpoint(ckpt_path, "cpu")
        bins = int(state["bins"])
        buckets = int(state["hash_buckets"])
        medians: dict[tuple[str, int], float] = {}
        for k, v in dict(state["medians"]).items():
            m = re.fullmatch(r"(natural|synthetic)_hand([01])", str(k))
            if m:
                medians[(m.group(1), int(m.group(2)))] = float(v)
        if len(medians) != 4:
            raise MHError(f"Checkpoint median map malformed: {ckpt_path}")
        examples = [
            build_inference_example(r, timing_by_id[str(r["sample_id"])], buckets, max_words, word_cache)
            for r in rows if inum(r.get("fold")) == fold
        ]
        wrapper = WordPhaseNet(torch, buckets, 10, len(np.asarray(state["clip_mean"])), bins)
        model = wrapper.net.to(device)
        model.load_state_dict(state["model_state"], strict=True)
        model.eval()
        centers = np.linspace(0.0, 1.0, bins, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                batch = examples[start:start + batch_size]
                tok, wf, mask, cf = collate_inference(
                    batch, medians,
                    np.asarray(state["word_mean"], dtype=np.float32),
                    np.asarray(state["word_std"], dtype=np.float32),
                    np.asarray(state["clip_mean"], dtype=np.float32),
                    np.asarray(state["clip_std"], dtype=np.float32),
                )
                tok, wf, mask, cf = [z.to(device) for z in (tok, wf, mask, cf)]
                logits = model(tok, wf, mask, cf)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
                for ex, p in zip(batch, prob):
                    p = p / max(float(p.sum()), 1e-12)
                    expected_ratio = float(np.sum(p * centers))
                    expected_frame = int(np.clip(round(expected_ratio * max(1, ex.clip_length - 1)), 2, max(2, ex.clip_length - 3)))
                    frozen_frame = inum(pred_by_id[ex.sample_id].get("wordphasenet_oof_pred"))
                    abs_diff = abs(expected_frame - frozen_frame)
                    regression_diffs.append(abs_diff)
                    regression_rows.append({
                        "sample_id": ex.sample_id, "fold": fold, "domain": ex.domain,
                        "hand": ex.hand, "clip_length": ex.clip_length,
                        "replayed_expected_frame": expected_frame,
                        "frozen_oof_expected_frame": frozen_frame,
                        "abs_diff_frames": abs_diff, "word_count": ex.word_count,
                        "transcript_status": ex.transcript_status,
                        "transcript_path": ex.transcript_path,
                    })
                    entropy = float(-np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0)) / math.log(len(p)))
                    mean = expected_ratio
                    std = float(math.sqrt(max(0.0, np.sum(p * (centers - mean) ** 2))))
                    order = np.argsort(p)[::-1]
                    out[ex.sample_id] = {
                        "sample_id": ex.sample_id,
                        "fold": fold,
                        "domain": ex.domain,
                        "hand": ex.hand,
                        "clip_length": ex.clip_length,
                        "prob": p.astype(np.float32),
                        "centers": centers.astype(np.float32),
                        "expected_ratio": expected_ratio,
                        "expected_frame": expected_frame,
                        "frozen_expected_frame": frozen_frame,
                        "expected_regression_abs_diff": abs(expected_frame - frozen_frame),
                        "entropy_norm": entropy,
                        "std_ratio": std,
                        "max_prob": float(p[order[0]]),
                        "second_bin_prob": float(p[order[1]]) if len(order) > 1 else 0.0,
                        "top_bin_margin": float(p[order[0]] - p[order[1]]) if len(order) > 1 else float(p[order[0]]),
                        "word_count": ex.word_count,
                        "transcript_status": ex.transcript_status,
                        "transcript_path": ex.transcript_path,
                    }
                    transcript_counts[ex.transcript_status] += 1
    if len(out) != len(rows):
        raise MHError(f"WordPhase distribution coverage mismatch: {len(out)} != {len(rows)}")
    max_diff = max(regression_diffs) if regression_diffs else 999
    audit = {
        "status": "PASS" if max_diff <= 1 else "FAIL",
        "query_count": len(out),
        "max_expected_frame_abs_diff_vs_frozen_oof": max_diff,
        "mean_expected_frame_abs_diff_vs_frozen_oof": float(np.mean(regression_diffs)),
        "exact_match_fraction": float(np.mean(np.asarray(regression_diffs) == 0)),
        "allowed_max_abs_diff_frames": 1,
        "frozen_p1a_replay_batch_size": FROZEN_P1A_REPLAY_BATCH_SIZE,
        "observed_replay_batch_size": int(batch_size),
        "batch_size_contract_gate": bool(batch_size == FROZEN_P1A_REPLAY_BATCH_SIZE),
        "padding_sensitive_bidirectional_gru": True,
        "transcript_path_source": "frozen M1 timing_manifest_all_folds.csv",
        "transcript_status_counts": dict(transcript_counts),
        "note": (
            "The accepted P1A model used an un-packed bidirectional GRU. Its backward state is padding-sensitive, "
            "so the original 128-example inference batching is part of the frozen replay contract. GT peak/hold "
            "fields are not used to construct model inputs."
        ),
    }
    regression_rows.sort(key=lambda r: (-int(r["abs_diff_frames"]), str(r["sample_id"])))
    return out, audit, regression_rows


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


def build_hypotheses(
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


def distance_to_hold(frame: int, start: int, end: int) -> int:
    if frame < start:
        return start - frame
    if frame > end:
        return frame - end
    return 0


def temporal_score(distance: int, sigma: float = 5.0) -> float:
    if distance <= 0:
        return 1.0
    return float(math.exp(-(float(distance) ** 2) / (2.0 * sigma ** 2)))


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise MHError("Empty numeric grid")
    return vals


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


def fit_weighted_ridge(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, query_ids: np.ndarray, lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd
    A = np.concatenate([np.ones((len(Z), 1), dtype=np.float64), Z], axis=1)
    group_counts = Counter(groups.tolist())
    query_counts = Counter(query_ids.tolist())
    w = np.asarray([
        1.0 / max(1, group_counts[g]) / max(1, query_counts[q])
        for g, q in zip(groups.tolist(), query_ids.tolist())
    ], dtype=np.float64)
    w /= max(float(np.mean(w)), 1e-12)
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


def identify_base_candidate_index(
    rows: Sequence[Mapping[str, Any]], p1_row: Mapping[str, Any], route_choice: str,
) -> int:
    donor_id = str(p1_row.get(f"{route_choice}_donor_id", ""))
    donor_rank = inum(p1_row.get(f"{route_choice}_donor_rank"), -1)
    matches = [
        i for i, r in enumerate(rows)
        if str(r.get("method")) == route_choice
        and str(r.get("donor_id")) == donor_id
        and inum(r.get("donor_rank")) == donor_rank
    ]
    if len(matches) == 1:
        return matches[0]
    matches = [i for i, r in enumerate(rows) if str(r.get("method")) == route_choice and str(r.get("donor_id")) == donor_id]
    if len(matches) == 1:
        return matches[0]
    raise MHError(f"Could not identify base P2A candidate method={route_choice} donor={donor_id} rank={donor_rank}")


def select_with_threshold(
    rows: Sequence[Mapping[str, Any]], predicted: np.ndarray, base_index: int, threshold: float,
) -> int:
    base = rows[base_index]
    eligible = [i for i, r in enumerate(rows) if i != base_index and candidate_guard(r, base)]
    if not eligible:
        return base_index
    best = max(eligible, key=lambda i: (float(predicted[i]), -inum(rows[i].get("donor_rank"), 999), -i))
    return best if float(predicted[best] - predicted[base_index]) >= threshold else base_index


def prepare_training_candidates(
    p1_candidates: Sequence[Mapping[str, Any]], p1_by_id: Mapping[str, Mapping[str, Any]],
    route_by_id: Mapping[str, Mapping[str, Any]], dist_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]], dict[str, int]]:
    rows_out: list[dict[str, Any]] = []
    feats: list[np.ndarray] = []
    labels: list[float] = []
    folds: list[int] = []
    groups: list[str] = []
    qids: list[str] = []
    by_query: dict[str, list[int]] = defaultdict(list)
    for r0 in p1_candidates:
        r = dict(r0)
        qid = str(r["query_id"])
        q = p1_by_id[qid]
        dist = dist_by_id[qid]
        method = str(r.get("method"))
        planned = inum(r.get("planned_frame"))
        median_frame = inum(q.get("median_peak"))
        expected_frame = inum(q.get("wordphase_peak"))
        mode1_frame = frame_from_ratio(local_mode_ratios(np.asarray(dist["prob"]))[0][0], inum(q.get("clip_length")))
        timing = {
            "hypothesis": "median" if method == "median" else "wp_expected",
            "delta_to_median_norm": (planned - median_frame) / max(1, inum(q.get("clip_length")) - 1),
            "delta_to_expected_norm": (planned - expected_frame) / max(1, inum(q.get("clip_length")) - 1),
            "delta_to_mode1_norm": (planned - mode1_frame) / max(1, inum(q.get("clip_length")) - 1),
            "wordphase_probability": probability_at_ratio(np.asarray(dist["prob"]), ratio_from_frame(planned, inum(q.get("clip_length")))),
            "wordphase_log_probability": math.log(max(probability_at_ratio(np.asarray(dist["prob"]), ratio_from_frame(planned, inum(q.get("clip_length")))), 1e-12)),
            "wordphase_cdf": cdf_at_ratio(np.asarray(dist["prob"]), ratio_from_frame(planned, inum(q.get("clip_length")))),
            "wordphase_entropy_norm": dist["entropy_norm"],
            "wordphase_max_prob": dist["max_prob"],
            "wordphase_top_bin_margin": dist["top_bin_margin"],
            "wordphase_std_ratio": dist["std_ratio"],
        }
        query = {
            "domain": q.get("domain"), "hand": q.get("hand"), "clip_length": q.get("clip_length"),
            "target_distance": q.get("target_distance"), "target_azimuth_deg": q.get("target_azimuth_deg"),
            "target_elevation_deg": q.get("target_elevation_deg"),
        }
        feats.append(build_ranker_feature(query, timing, r))
        labels.append(fnum(r.get("self_stg_proxy_sigma5"), 0.0))
        folds.append(inum(r.get("fold")))
        groups.append(str(q.get("leakage_group_id") or qid))
        qids.append(qid)
        idx = len(rows_out)
        rows_out.append({**r, "_hypothesis": timing["hypothesis"], "_query": query, "_timing": timing})
        by_query[qid].append(idx)
    base_index_by_query: dict[str, int] = {}
    for qid, ids in by_query.items():
        route = str(route_by_id[qid].get("router_choice"))
        local_rows = [rows_out[i] for i in ids]
        local_idx = identify_base_candidate_index(local_rows, p1_by_id[qid], route)
        base_index_by_query[qid] = ids[local_idx]
    return (
        rows_out, np.stack(feats), np.asarray(labels, dtype=np.float64),
        np.asarray(folds, dtype=np.int16), np.asarray(groups, dtype=object),
        by_query, base_index_by_query,
    )


def nested_calibrate_ranker(
    outer_fold: int, train_rows: Sequence[Mapping[str, Any]], X: np.ndarray, y: np.ndarray,
    folds: np.ndarray, groups: np.ndarray, by_query: Mapping[str, list[int]],
    base_idx: Mapping[str, int], lambdas: Sequence[float], thresholds: Sequence[float],
) -> tuple[float, float, list[dict[str, Any]]]:
    train_fold_set = [f for f in EXPECTED_FOLDS if f != outer_fold]
    audit: list[dict[str, Any]] = []
    best: tuple[float, float, float] | None = None
    for lam in lambdas:
        pred_all = np.full(len(X), np.nan, dtype=np.float64)
        for inner in train_fold_set:
            tr = (folds != outer_fold) & (folds != inner)
            va = folds == inner
            beta, mu, sd = fit_weighted_ridge(
                X[tr], y[tr], groups[tr], np.asarray([str(train_rows[i]["query_id"]) for i in np.flatnonzero(tr)], dtype=object), lam,
            )
            pred_all[va] = ridge_predict(X[va], beta, mu, sd)
        valid_qids = [q for q in by_query if inum(train_rows[by_query[q][0]].get("fold")) != outer_fold]
        for threshold in thresholds:
            selected_actual: list[float] = []
            base_actual: list[float] = []
            replace_count = 0
            for qid in valid_qids:
                ids = by_query[qid]
                local_pred = pred_all[ids]
                if not np.all(np.isfinite(local_pred)):
                    raise MHError(f"Inner OOF ranker coverage missing for {qid}")
                bglobal = base_idx[qid]
                blocal = ids.index(bglobal)
                choice_local = select_with_threshold([train_rows[i] for i in ids], local_pred, blocal, threshold)
                selected_actual.append(float(y[ids[choice_local]]))
                base_actual.append(float(y[bglobal]))
                replace_count += int(ids[choice_local] != bglobal)
            gain = float(np.mean(selected_actual) - np.mean(base_actual))
            audit.append({
                "outer_fold": outer_fold, "ridge_lambda": lam, "replacement_threshold": threshold,
                "inner_query_count": len(valid_qids), "inner_selected_stg": float(np.mean(selected_actual)),
                "inner_base_p2a_stg": float(np.mean(base_actual)), "inner_gain": gain,
                "inner_replacement_fraction": replace_count / max(1, len(valid_qids)),
            })
            key = (gain, -abs(threshold), -lam)
            if best is None or key > (best[0], -abs(best[2]), -best[1]):
                best = (gain, lam, threshold)
    if best is None:
        raise MHError("Ranker calibration produced no result")
    return float(best[1]), float(best[2]), audit


def fit_outer_ranker(
    outer_fold: int, train_rows: Sequence[Mapping[str, Any]], X: np.ndarray, y: np.ndarray,
    folds: np.ndarray, groups: np.ndarray, lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = folds != outer_fold
    qids = np.asarray([str(train_rows[i]["query_id"]) for i in np.flatnonzero(tr)], dtype=object)
    return fit_weighted_ridge(X[tr], y[tr], groups[tr], qids, lam)


def paired_bootstrap(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cluster_bootstrap(values: np.ndarray, clusters: Sequence[str], seed: int, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    clusters_arr = np.asarray([str(x) for x in clusters], dtype=object)
    valid = np.isfinite(values)
    values = values[valid]
    clusters_arr = clusters_arr[valid]
    if len(values) == 0:
        return float("nan"), float("nan")
    uniq, inv = np.unique(clusters_arr, return_inverse=True)
    if len(uniq) == 1:
        m = float(np.mean(values))
        return m, m
    sums = np.bincount(inv, weights=values, minlength=len(uniq)).astype(np.float64)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(uniq), size=len(uniq))
        means[i] = float(sums[idx].sum() / max(counts[idx].sum(), 1.0))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def archive_existing(path: Path, archive_root: Path) -> None:
    if not path.exists():
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / path.name
    i = 1
    while target.exists():
        target = archive_root / f"{path.name}.{i}"
        i += 1
    shutil.move(str(path), str(target))


def candidate_from_p1_selected(
    p1_row: Mapping[str, Any], route_choice: str, p1_candidate_lookup: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    qid = str(p1_row["sample_id"])
    donor_id = str(p1_row.get(f"{route_choice}_donor_id", ""))
    donor_rank = inum(p1_row.get(f"{route_choice}_donor_rank"), -1)
    key = (qid, route_choice, donor_id, donor_rank)
    src = dict(p1_candidate_lookup.get(key, {}))
    if not src:
        matches = [v for k, v in p1_candidate_lookup.items() if k[0] == qid and k[1] == route_choice and k[2] == donor_id]
        if len(matches) != 1:
            raise MHError(f"Cannot resolve base candidate row for {qid} {route_choice} {donor_id} rank={donor_rank}")
        src = dict(matches[0])
    src["method"] = "p2a_base"
    src["hypothesis"] = "median" if route_choice == "median" else "wp_expected"
    src["output"] = str(p1_row.get(f"{route_choice}_output", ""))
    return src


def make_query_context(p1_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "domain": p1_row.get("domain"), "hand": p1_row.get("hand"), "clip_length": p1_row.get("clip_length"),
        "target_distance": p1_row.get("target_distance"), "target_azimuth_deg": p1_row.get("target_azimuth_deg"),
        "target_elevation_deg": p1_row.get("target_elevation_deg"),
    }


def timing_features_for_frame(
    dist: Mapping[str, Any], hypothesis: str, frame: int, median_frame: int,
) -> dict[str, Any]:
    T = int(dist["clip_length"])
    mode1 = frame_from_ratio(local_mode_ratios(np.asarray(dist["prob"]))[0][0], T)
    ratio = ratio_from_frame(frame, T)
    p = probability_at_ratio(np.asarray(dist["prob"]), ratio)
    return {
        "hypothesis": hypothesis,
        "planned_frame": frame,
        "planned_ratio": ratio,
        "wordphase_probability": p,
        "wordphase_log_probability": math.log(max(p, 1e-12)),
        "wordphase_cdf": cdf_at_ratio(np.asarray(dist["prob"]), ratio),
        "wordphase_entropy_norm": dist["entropy_norm"],
        "wordphase_max_prob": dist["max_prob"],
        "wordphase_top_bin_margin": dist["top_bin_margin"],
        "wordphase_std_ratio": dist["std_ratio"],
        "delta_to_median_norm": (frame - median_frame) / max(1, T - 1),
        "delta_to_expected_norm": (frame - int(dist["expected_frame"])) / max(1, T - 1),
        "delta_to_mode1_norm": (frame - mode1) / max(1, T - 1),
    }


def validate_complete_fold(fold_dir: Path, expected: int) -> bool:
    marker = fold_dir / "FOLD_COMPLETE.json"
    per_path = fold_dir / "per_sample_comparison.csv"
    if not (marker.is_file() and per_path.is_file()):
        return False
    try:
        obj = json_load(marker)
        rows = read_csv(per_path)
        return obj.get("status") == "FOLD_COMPLETE" and int(obj.get("query_count", -1)) == expected and len(rows) == expected
    except Exception:
        return False


def select_pilot_rows(
    rows: Sequence[Mapping[str, Any]], p2b_dir: Path, queries_per_slice: int, seed: int,
) -> list[dict[str, Any]]:
    by_id = {str(r["sample_id"]): dict(r) for r in rows}
    p2b_path = p2b_dir / "per_sample_comparison.csv"
    if queries_per_slice == 10 and p2b_path.is_file():
        ids = [str(r["sample_id"]) for r in read_csv(p2b_path)]
        selected = [by_id[sid] for sid in ids if sid in by_id]
        counts = Counter((inum(r.get("fold")), str(r.get("domain")), inum(r.get("hand"))) for r in selected)
        if len(selected) == 200 and all(counts[(f, d, h)] == 10 for f in EXPECTED_FOLDS for d in ("natural", "synthetic") for h in (0, 1)):
            return sorted(selected, key=lambda r: (inum(r.get("fold")), str(r.get("domain")), inum(r.get("hand")), str(r.get("sample_id"))))
    selected: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        for domain in ("natural", "synthetic"):
            for hand in (0, 1):
                pool = [dict(r) for r in rows if inum(r.get("fold")) == fold and str(r.get("domain")) == domain and inum(r.get("hand")) == hand]
                pool.sort(key=lambda r: stable_u64(str(r["sample_id"]), seed))
                take = pool if queries_per_slice <= 0 else pool[:queries_per_slice]
                if not take:
                    raise MHError(f"Empty pilot slice fold={fold} domain={domain} hand={hand}")
                selected.extend(take)
    return sorted(selected, key=lambda r: (inum(r.get("fold")), str(r.get("domain")), inum(r.get("hand")), str(r.get("sample_id"))))


def run_fold(
    fold: int, fold_dir: Path, selected_rows: Sequence[Mapping[str, Any]], all_rows: Sequence[Mapping[str, Any]],
    p1_by_id: Mapping[str, Mapping[str, Any]], route_by_id: Mapping[str, Mapping[str, Any]],
    dist_by_id: Mapping[str, Mapping[str, Any]], hypotheses_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    module: Any, engine: Any, source_model: Mapping[str, Any], refined_model: Mapping[str, Any],
    p1_candidate_lookup: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    ranker_beta: np.ndarray, ranker_mu: np.ndarray, ranker_sd: np.ndarray,
    ranker_lambda: float, replacement_threshold: float, args: argparse.Namespace,
) -> None:
    fold_rows = [dict(r) for r in selected_rows if inum(r.get("fold")) == fold]
    train_rows = [dict(r) for r in all_rows if inum(r.get("fold")) != fold]
    if not fold_rows:
        raise MHError(f"No pilot rows for fold {fold}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_csv(fold_dir / "pilot_queries.csv", fold_rows)
    generation_views = {str(r["sample_id"]): module.generation_view(r) for r in fold_rows}
    prohibited = sorted(getattr(module, "EVAL_ONLY_QUERY_KEYS", []))
    audit = {
        "status": "PASS",
        "fold": fold,
        "validation_query_count": len(fold_rows),
        "all_generation_views_exclude_prohibited_keys": all(all(k not in generation_views[str(r["sample_id"])] for k in prohibited) for r in fold_rows),
        "prohibited_generation_keys": prohibited,
        "timing_inputs": ["frozen WordPhaseNet OOF checkpoint distribution", "OOF domain-hand median"],
        "ranker_inputs_exclude_gt_fields": True,
        "post_generation_only": ["released peak", "hold start/end", "alignment labels", "temporal/STG proxy"],
    }
    if not audit["all_generation_views_exclude_prohibited_keys"]:
        audit["status"] = "FAIL"
        raise MHError("Validation GT usage audit failed")
    json_dump(fold_dir / "validation_gt_usage_audit.json", audit)

    schema_cache: dict[Path, dict[str, Any]] = {}
    clip_cache: dict[str, dict[str, Any]] = {}
    row_by_id = {str(r["sample_id"]): dict(r) for r in all_rows}
    donors_by_hand: dict[int, list[dict[str, Any]]] = {}
    donor_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for hand in (0, 1):
        donors = [r for r in train_rows if inum(r.get("hand")) == hand and str(r.get("domain")) == "natural"]
        if len(donors) < args.donors_per_hypothesis:
            raise MHError(f"Fold {fold}: insufficient natural hand-{hand} donors")
        D = np.stack([module.descriptor(r) for r in donors])
        sd = D.std(axis=0)
        sd[sd < 1e-6] = 1.0
        donors_by_hand[hand] = donors
        donor_stats[hand] = (D, sd)

    candidate_records: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for qi, q_eval in enumerate(fold_rows):
        sid = str(q_eval["sample_id"])
        q = generation_views[sid]
        hand = inum(q.get("hand"))
        qframe = module.load_clip_frame(q, 0, schema_cache)
        beta = np.asarray(qframe["betas"], dtype=np.float32)[:10]
        gender = module.choose_gender(qframe.get("gender"), q.get("gender"))
        hypotheses = list(hypotheses_by_id[sid])
        for h in hypotheses:
            timing_rows.append({
                "sample_id": sid, "fold": fold, "domain": q.get("domain"), "hand": hand,
                "clip_length": inum(q.get("clip_length")), **h,
            })
        donors = donors_by_hand[hand]
        D, sd = donor_stats[hand]
        p1_base = p1_by_id[sid]
        route_choice = str(route_by_id[sid].get("router_choice"))
        base_planned_frame = inum(p1_base.get(f"{route_choice}_planned_frame"))
        reserved_hypothesis = min(
            hypotheses,
            key=lambda hh: (abs(inum(hh.get("planned_frame")) - base_planned_frame), inum(hh.get("hypothesis_rank"), 999)),
        )["hypothesis"]
        base_donor_id = str(p1_base.get(f"{route_choice}_donor_id", ""))
        for h in hypotheses:
            planned = inum(h.get("planned_frame"))
            qdesc = module.descriptor(q, planned / max(1, inum(q.get("clip_length")) - 1))
            distances = np.linalg.norm((D - qdesc[None, :]) / sd[None, :], axis=1)
            ordered = np.argsort(distances, kind="mergesort").tolist()
            if str(h.get("hypothesis")) == str(reserved_hypothesis):
                ordered = [idx for idx in ordered if str(donors[int(idx)].get("sample_id")) != base_donor_id]
                take_count = max(0, args.donors_per_hypothesis - 1)
            else:
                take_count = args.donors_per_hypothesis
            top_ids = ordered[:take_count]
            for rank, idx in enumerate(top_ids, 1):
                donor = donors[int(idx)]
                candidate_records.append({
                    "method": "mh", "hypothesis": h["hypothesis"], "timing": dict(h),
                    "query_id": sid, "donor_id": str(donor["sample_id"]),
                    "donor_rank": rank, "donor_distance": float(distances[int(idx)]),
                    "row": q, "eval_row": q_eval, "donor_row": donor,
                    "beta": beta, "gender": gender, "planned_peak": planned,
                })
                retrieval_rows.append({
                    "query_id": sid, "fold": fold, "hypothesis": h["hypothesis"],
                    "planned_frame": planned, "donor_id": donor["sample_id"],
                    "donor_rank": rank, "donor_distance": float(distances[int(idx)]),
                    "candidate_budget_total_including_frozen_p2a": args.hypothesis_count * args.donors_per_hypothesis,
                    "donor_pool": "frozen natural same-hand train-fold only",
                })
        if (qi + 1) % 20 == 0:
            print(f"[fold {fold} retrieval] {qi+1}/{len(fold_rows)}", flush=True)
    write_csv(fold_dir / "timing_hypothesis_manifest.csv", timing_rows)
    write_csv(fold_dir / "retrieval_manifest.csv", retrieval_rows)

    for i, c in enumerate(candidate_records):
        did = str(c["donor_id"])
        if did not in clip_cache:
            clip_cache[did] = module.load_full_clip(row_by_id[did], schema_cache)
        donor_clip = clip_cache[did]
        drow = c["donor_row"]
        warped = module.warp_clip(
            donor_clip, inum(drow.get("released_peak_frame")),
            inum(c["row"].get("clip_length")), int(c["planned_peak"]),
        )
        c["base_pose"] = warped["pose"]
        c["base_trans"] = warped["trans"]
        c["base_apex"] = warped["pose"][int(c["planned_peak"])].copy()
        c["query_betas"] = np.asarray(c["beta"], dtype=np.float32)
        if (i + 1) % 200 == 0:
            print(f"[fold {fold} warp] {i+1}/{len(candidate_records)}", flush=True)

    ik_logs: list[dict[str, Any]] = []
    for hand in (0, 1):
        for gender in ("neutral", "male", "female"):
            ids = [
                i for i, c in enumerate(candidate_records)
                if inum(c["row"].get("hand")) == hand and module.normalize_gender(c["gender"]) == gender
            ]
            for a in range(0, len(ids), args.ik_batch_size):
                take = ids[a:a + args.ik_batch_size]
                if not take:
                    continue
                base = np.stack([candidate_records[i]["base_apex"] for i in take]).astype(np.float32)
                betas = np.stack([candidate_records[i]["query_betas"] for i in take]).astype(np.float32)
                targets = np.stack([module.target_vec(candidate_records[i]["row"]) for i in take]).astype(np.float32)
                opt_pose, logs = module.optimize_apex_batch(
                    engine, base, betas, gender, hand, "p0c", targets, None,
                    args.ik_steps, args.ik_lr, args.ik_max_delta,
                )
                for local, idx in enumerate(take):
                    c = candidate_records[idx]
                    c["optimized_apex"] = opt_pose[local]
                    c["pose"] = module.apply_hold(
                        c["base_pose"], int(c["planned_peak"]), opt_pose[local], hand,
                        args.hold_plateau, args.hold_taper,
                    )
                    c["trans"] = c["base_trans"].copy()
                    c["betas_full"] = np.broadcast_to(c["query_betas"][None, :], (len(c["pose"]), len(c["query_betas"]))).copy()
                    c["donor_pose"] = c["base_pose"].copy()
                for rec in logs:
                    ik_logs.append({"hand": hand, "gender": gender, "batch_start": a, **rec})
            if ids:
                print(f"[fold {fold} IK] hand={hand} gender={gender} candidates={len(ids)}", flush=True)
    write_csv(fold_dir / "ik_training_log.csv", ik_logs)

    eval_clips = [{"pose": c["pose"], "trans": c["trans"], "betas": c["betas_full"], "gender": c["gender"]} for c in candidate_records]
    eval_joints = engine.joints_many(eval_clips)
    scene_cache: dict[str, Any] = {}
    candidate_metrics: list[dict[str, Any]] = []
    for i, c in enumerate(candidate_records):
        q_eval = c["eval_row"]
        scene = module.resolve_scene(args.data_root, q_eval, scene_cache)
        metrics = module.evaluate_candidate(c, eval_joints[i], q_eval, scene, source_model, refined_model)
        c["metrics"] = metrics
        candidate_metrics.append({
            "method": "mh", "hypothesis": c["hypothesis"], "query_id": c["query_id"],
            "donor_id": c["donor_id"], "donor_rank": c["donor_rank"],
            "donor_distance": c["donor_distance"], "fold": fold,
            "domain": q_eval.get("domain"), "hand": inum(q_eval.get("hand")),
            **c["timing"], **metrics,
        })
    write_csv(fold_dir / "candidate_metrics.csv", candidate_metrics)

    by_query_new: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(candidate_records):
        by_query_new[str(c["query_id"])].append(i)
    expected_new_per_query = args.hypothesis_count * args.donors_per_hypothesis - 1
    for q_eval in fold_rows:
        sid = str(q_eval["sample_id"])
        if len(by_query_new[sid]) != expected_new_per_query:
            raise MHError(f"Candidate budget mismatch for {sid}: {len(by_query_new[sid])} != {expected_new_per_query}")
    motion_root = fold_dir / "motions" / "selected"
    motion_root.mkdir(parents=True, exist_ok=True)
    per_sample: list[dict[str, Any]] = []
    schema_errors: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for q_eval in fold_rows:
        sid = str(q_eval["sample_id"])
        p1 = p1_by_id[sid]
        route_choice = str(route_by_id[sid].get("router_choice"))
        base = candidate_from_p1_selected(p1, route_choice, p1_candidate_lookup)
        base_timing = timing_features_for_frame(
            dist_by_id[sid], "median" if route_choice == "median" else "wp_expected",
            inum(base.get("planned_frame")), inum(p1.get("median_peak")),
        )
        query_ctx = make_query_context(p1)
        new_ids = by_query_new[sid]
        pool_rows: list[dict[str, Any]] = [base]
        pool_timing: list[dict[str, Any]] = [base_timing]
        pool_refs: list[tuple[str, int | None]] = [("base", None)]
        for idx in new_ids:
            c = candidate_records[idx]
            pool_rows.append({
                "method": "mh", "hypothesis": c["hypothesis"], "donor_id": c["donor_id"],
                "donor_rank": c["donor_rank"], "donor_distance": c["donor_distance"],
                **c["metrics"],
            })
            pool_timing.append(c["timing"])
            pool_refs.append(("new", idx))
        Xpool = np.stack([build_ranker_feature(query_ctx, t, r) for t, r in zip(pool_timing, pool_rows)])
        pred = ridge_predict(Xpool, ranker_beta, ranker_mu, ranker_sd)
        choice = select_with_threshold(pool_rows, pred, 0, replacement_threshold)
        source, ref = pool_refs[choice]
        if source == "base":
            src_path = Path(str(base["output"]))
            if not src_path.is_file():
                raise MHError(f"Missing base P2A output: {src_path}")
            with np.load(src_path, allow_pickle=False) as z:
                pose = np.asarray(z["poses"], dtype=np.float32)
                trans = np.asarray(z["trans"], dtype=np.float32)
            selected_metrics = base
            selected_hypothesis = "p2a_base_" + route_choice
            selected_donor_id = str(base.get("donor_id", ""))
            selected_donor_rank = inum(base.get("donor_rank"))
        else:
            assert ref is not None
            c = candidate_records[int(ref)]
            pose = np.asarray(c["pose"], dtype=np.float32)
            trans = np.asarray(c["trans"], dtype=np.float32)
            selected_metrics = c["metrics"]
            selected_hypothesis = str(c["hypothesis"])
            selected_donor_id = str(c["donor_id"])
            selected_donor_rank = inum(c["donor_rank"])
        out_path = motion_root / f"{sid}.npz"
        np.savez_compressed(out_path, poses=pose.astype(np.float32), trans=trans.astype(np.float32))
        T = inum(q_eval.get("clip_length"))
        if pose.shape != (T, 165) or trans.shape != (T, 3) or not np.all(np.isfinite(pose)) or not np.all(np.isfinite(trans)):
            schema_errors.append({"sample_id": sid, "pose_shape": str(pose.shape), "trans_shape": str(trans.shape)})
        hs = inum(q_eval.get("hold_start")); he = inum(q_eval.get("hold_end"))
        planned = inum(selected_metrics.get("planned_frame"))
        row_out: dict[str, Any] = {
            "sample_id": sid, "recording_id": q_eval.get("recording_id", ""),
            "leakage_group_id": q_eval.get("leakage_group_id", ""), "fold": fold,
            "domain": q_eval.get("domain"), "hand": inum(q_eval.get("hand")), "clip_length": T,
            "gt_peak": inum(q_eval.get("released_peak_frame")), "hold_start": hs, "hold_end": he,
            "base_route_choice": route_choice, "base_output": base.get("output"),
            "mh_selected_source": source, "mh_selected_hypothesis": selected_hypothesis,
            "mh_selected_donor_id": selected_donor_id, "mh_selected_donor_rank": selected_donor_rank,
            "mh_output": str(out_path), "ridge_lambda": ranker_lambda,
            "replacement_threshold": replacement_threshold,
            "predicted_base_utility": float(pred[0]), "predicted_selected_utility": float(pred[choice]),
            "predicted_utility_gain": float(pred[choice] - pred[0]),
            "base_timing_distance_to_hold": distance_to_hold(inum(base.get("planned_frame")), hs, he),
            "mh_timing_distance_to_hold": distance_to_hold(planned, hs, he),
            "base_timing_temporal_sigma5": temporal_score(distance_to_hold(inum(base.get("planned_frame")), hs, he)),
            "mh_timing_temporal_sigma5": temporal_score(distance_to_hold(planned, hs, he)),
        }
        for k, v in base.items():
            if k in {
                "planned_frame", "self_frame", "det_source", "det_refined", "det_heuristic", "detector_agreement",
                "distance_to_gt_hold", "temporal_proxy_sigma5", "planned_angle_deg", "planned_score1b_proxy20",
                "self_angle_deg", "self_score1b_proxy20", "self_stg_proxy_sigma5", "oracle_angle_deg",
                "oracle_score1b_proxy20", "joint_jerk", "foot_sliding", "joint_limit_fraction", "pose_deviation", "max_reach",
            }:
                row_out[f"base_{k}"] = v
        for k, v in selected_metrics.items():
            if k in {
                "planned_frame", "self_frame", "det_source", "det_refined", "det_heuristic", "detector_agreement",
                "distance_to_gt_hold", "temporal_proxy_sigma5", "planned_angle_deg", "planned_score1b_proxy20",
                "self_angle_deg", "self_score1b_proxy20", "self_stg_proxy_sigma5", "oracle_angle_deg",
                "oracle_score1b_proxy20", "joint_jerk", "foot_sliding", "joint_limit_fraction", "pose_deviation", "max_reach",
            }:
                row_out[f"mh_{k}"] = v
        per_sample.append(row_out)
        selection_rows.append({
            "sample_id": sid, "fold": fold, "choice_index": choice, "selected_source": source,
            "selected_hypothesis": selected_hypothesis, "pool_size": len(pool_rows),
            "predicted_base_utility": float(pred[0]), "predicted_selected_utility": float(pred[choice]),
            "predicted_gain": float(pred[choice] - pred[0]),
        })
    write_csv(fold_dir / "per_sample_comparison.csv", per_sample)
    write_csv(fold_dir / "selection_manifest.csv", selection_rows)
    write_csv(fold_dir / "schema_errors.csv", schema_errors)
    np.savez(
        fold_dir / f"multi_hypothesis_ranker_fold{fold}.npz",
        beta=ranker_beta, feature_mean=ranker_mu, feature_std=ranker_sd,
        feature_names=np.asarray(ranker_feature_names(), dtype="U128"),
        ridge_lambda=np.asarray([ranker_lambda], dtype=np.float64),
        replacement_threshold=np.asarray([replacement_threshold], dtype=np.float64),
    )
    marker = {
        "status": "FOLD_COMPLETE" if not schema_errors else "FOLD_ENGINEERING_FAILED",
        "fold": fold, "query_count": len(per_sample), "candidate_count": len(candidate_records),
        "schema_error_count": len(schema_errors), "ranker_lambda": ranker_lambda,
        "replacement_threshold": replacement_threshold,
    }
    json_dump(fold_dir / "FOLD_COMPLETE.json", marker)
    if schema_errors:
        raise MHError(f"Fold {fold} schema errors: {len(schema_errors)}")


def compare_metric(
    rows: Sequence[Mapping[str, Any]], metric: str, group_name: str, group_value: str,
    seed: int, n_boot: int,
) -> dict[str, Any]:
    if group_name == "overall":
        group = list(rows)
    elif group_name == "fold":
        group = [r for r in rows if str(r.get("fold")) == str(group_value)]
    elif group_name == "domain":
        group = [r for r in rows if str(r.get("domain")) == str(group_value)]
    elif group_name == "hand":
        group = [r for r in rows if str(r.get("hand")) == str(group_value)]
    elif group_name == "domain_hand":
        d, h = group_value.rsplit("_hand", 1)
        group = [r for r in rows if str(r.get("domain")) == d and str(r.get("hand")) == h]
    else:
        group = []
    left = np.asarray([fnum(r.get(f"mh_{metric}")) for r in group], dtype=np.float64)
    right = np.asarray([fnum(r.get(f"base_{metric}")) for r in group], dtype=np.float64)
    clusters = np.asarray([str(r.get("leakage_group_id") or r.get("sample_id")) for r in group], dtype=object)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right, clusters = left[valid], right[valid], clusters[valid]
    diff = left - right
    slo, shi = paired_bootstrap(diff, stable_seed(f"sample:{metric}:{group_name}:{group_value}", seed), n_boot)
    clo, chi = cluster_bootstrap(diff, clusters, stable_seed(f"cluster:{metric}:{group_name}:{group_value}", seed), n_boot)
    return {
        "group_name": group_name, "group_value": group_value, "metric": metric, "n": len(diff),
        "cluster_count": len(set(clusters.tolist())), "mh_mean": float(np.mean(left)),
        "p2a_base_mean": float(np.mean(right)), "mh_minus_base": float(np.mean(diff)),
        "sample_ci95_low": slo, "sample_ci95_high": shi,
        "cluster_ci95_low": clo, "cluster_ci95_high": chi,
        "mh_better_fraction": float(np.mean(diff > 0)), "tie_fraction": float(np.mean(diff == 0)),
    }


def aggregate(
    args: argparse.Namespace, all_per: Sequence[Mapping[str, Any]], timing_audit_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    metric_names = [
        "self_stg_proxy_sigma5", "self_score1b_proxy20", "temporal_proxy_sigma5",
        "planned_score1b_proxy20", "self_angle_deg", "joint_jerk", "foot_sliding",
        "joint_limit_fraction", "pose_deviation", "detector_agreement",
    ]
    groups: list[tuple[str, str]] = [("overall", "overall")]
    groups += [("fold", str(f)) for f in EXPECTED_FOLDS]
    groups += [("domain", d) for d in ("natural", "synthetic")]
    groups += [("hand", str(h)) for h in (0, 1)]
    groups += [("domain_hand", f"{d}_hand{h}") for d in ("natural", "synthetic") for h in (0, 1)]
    metrics: list[dict[str, Any]] = []
    for gname, gvalue in groups:
        for metric in metric_names:
            metrics.append(compare_metric(all_per, metric, gname, gvalue, args.seed, args.bootstrap))
    write_csv(output_dir / "summary_metrics.csv", metrics)

    def rec(metric: str, gname: str = "overall", gvalue: str = "overall") -> Mapping[str, Any]:
        return next(r for r in metrics if r["metric"] == metric and r["group_name"] == gname and r["group_value"] == gvalue)

    stg = rec("self_stg_proxy_sigma5")
    score = rec("self_score1b_proxy20")
    timing = rec("temporal_proxy_sigma5")
    fold_positive = sum(fnum(rec("self_stg_proxy_sigma5", "fold", str(f)).get("mh_minus_base"), -1) > 0 for f in EXPECTED_FOLDS)
    slice_positive = sum(fnum(rec("self_stg_proxy_sigma5", "domain_hand", f"{d}_hand{h}").get("mh_minus_base"), -1) > 0 for d in ("natural", "synthetic") for h in (0, 1))
    replacement_fraction = float(np.mean([str(r.get("mh_selected_source")) == "new" for r in all_per]))
    novel_fraction = float(np.mean([
        str(r.get("mh_selected_source")) == "new" and str(r.get("mh_selected_hypothesis")) not in {"median", "wp_expected"}
        for r in all_per
    ]))
    engineering_gate = len(all_per) == args.expected_queries and len({str(r.get("sample_id")) for r in all_per}) == args.expected_queries
    primary_gate = (
        fnum(stg.get("mh_minus_base"), -1) > 0
        and fnum(stg.get("sample_ci95_low"), -1) > 0
        and fnum(stg.get("cluster_ci95_low"), -1) > 0
        and fold_positive >= 4 and slice_positive >= 3
    )
    spatial_guard = fnum(score.get("mh_minus_base"), -1) >= -0.01
    timing_guard = fnum(timing.get("mh_minus_base"), -1) >= -0.01
    quality_limits = {
        "joint_jerk": (1.20, 0.0001), "foot_sliding": (1.25, 0.0005),
        "joint_limit_fraction": (1.0, 0.005), "pose_deviation": (1.25, 0.002),
    }
    quality: dict[str, Any] = {}
    for metric, (ratio, additive) in quality_limits.items():
        r = rec(metric)
        base = fnum(r.get("p2a_base_mean"), 0.0)
        mh = fnum(r.get("mh_mean"), 1e9)
        limit = max(ratio * base, base + additive)
        quality[metric] = {"base_mean": base, "mh_mean": mh, "allowed_max": limit, "pass": mh <= limit}
    quality_gate = all(bool(x["pass"]) for x in quality.values())
    utilization_gate = replacement_fraction >= 0.10 and novel_fraction >= 0.03

    if not engineering_gate:
        status = "P2A_MH_PILOT_ENGINEERING_FAILED"
        full_allowed = False
    elif primary_gate and spatial_guard and timing_guard and quality_gate and utilization_gate:
        status = "P2A_MH_PILOT_ACCEPTED_FULL_OOF_ALLOWED"
        full_allowed = True
    elif fnum(stg.get("mh_minus_base"), -1) > 0 and spatial_guard and quality_gate:
        status = "P2A_MH_PILOT_CONDITIONALLY_POSITIVE_NEEDS_MORE_EVIDENCE"
        full_allowed = False
    else:
        status = "P2A_MH_PILOT_NO_MATERIAL_GAIN_KEEP_P2A"
        full_allowed = False

    timing_oracle_gain = float(np.mean([
        fnum(r.get("hypothesis_oracle_temporal_sigma5"), 0.0) - fnum(r.get("p2a_route_timing_temporal_sigma5"), 0.0)
        for r in timing_audit_rows
    ])) if timing_audit_rows else float("nan")
    acceptance = {
        "schema_version": SCHEMA_VERSION, "version": VERSION, "created_utc": now_utc(),
        "status": status, "query_count": len(all_per),
        "selection_pool_per_query": args.hypothesis_count * args.donors_per_hypothesis,
        "new_generated_candidates_per_query": args.hypothesis_count * args.donors_per_hypothesis - 1,
        "p2a_fallback_is_always_eligible": True,
        "engineering_gate": engineering_gate, "primary_gate": primary_gate,
        "spatial_guard": spatial_guard, "timing_guard": timing_guard,
        "quality_gate": quality_gate, "utilization_gate": utilization_gate,
        "full_oof_allowed": full_allowed,
        "primary": {"self_stg": stg, "self_score1b": score, "temporal": timing},
        "positive_folds_stg": fold_positive, "positive_domain_hand_slices_stg": slice_positive,
        "replacement_fraction": replacement_fraction, "novel_timing_selection_fraction": novel_fraction,
        "timing_hypothesis_oracle_gain_over_p2a_route": timing_oracle_gain,
        "quality_components": quality,
        "guardrails": [
            "P2A is retained as an explicit fallback candidate for every query.",
            "WordPhase timing hypotheses are inferred from the held-out fold checkpoint; released peak/hold fields are post-generation evaluation only.",
            "The candidate ranker is nested-calibrated on existing P1 OOF candidate metrics from non-held-out folds.",
            "Retrieval donor pool, time warp, shape-aware IK, hold synthesis, detector proxies and spatial proxy remain frozen.",
            "This pilot does not add scene graph, listener state, residual flow or synthetic-specific donors.",
        ],
        "next_action": (
            "Run full 2525-query P2A-MH OOF generation, then materialize official Candidate B."
            if full_allowed else "Keep P2A as Candidate A; inspect timing/ranker failures before any full expansion."
        ),
    }
    json_dump(output_dir / "P2A_MH_ACCEPTANCE.json", acceptance)

    fold_rows: list[dict[str, Any]] = []
    for f in EXPECTED_FOLDS:
        fold_rows.append({
            "fold": f,
            "query_count": sum(1 for r in all_per if inum(r.get("fold")) == f),
            "self_stg_delta": fnum(rec("self_stg_proxy_sigma5", "fold", str(f)).get("mh_minus_base")),
            "self_score1b_delta": fnum(rec("self_score1b_proxy20", "fold", str(f)).get("mh_minus_base")),
            "temporal_delta": fnum(rec("temporal_proxy_sigma5", "fold", str(f)).get("mh_minus_base")),
            "replacement_fraction": float(np.mean([str(r.get("mh_selected_source")) == "new" for r in all_per if inum(r.get("fold")) == f])),
        })
    write_csv(output_dir / "fold_results.csv", fold_rows)

    summary = [
        "REPOINTER P2A-MH — MULTI-HYPOTHESIS TIMING + CROSS-FITTED CANDIDATE RANKER PILOT",
        "=" * 88,
        f"version={VERSION}", f"created_utc={now_utc()}",
        f"queries={len(all_per)}", f"folds=5",
        f"hypotheses_per_query={args.hypothesis_count}",
        f"donors_per_hypothesis={args.donors_per_hypothesis}",
        f"new_generated_candidates={args.hypothesis_count * args.donors_per_hypothesis - 1}",
        f"selection_pool_including_p2a={args.hypothesis_count * args.donors_per_hypothesis}",
        "base_fallback=P2A routed motion occupies one frozen budget slot", "",
        "[PRIMARY]",
        f"p2a_base_self_stg={stg['p2a_base_mean']}", f"mh_self_stg={stg['mh_mean']}", f"mh_minus_base_self_stg={stg['mh_minus_base']}",
        f"stg_sample_ci95=[{stg['sample_ci95_low']}, {stg['sample_ci95_high']}]",
        f"stg_cluster_ci95=[{stg['cluster_ci95_low']}, {stg['cluster_ci95_high']}]",
        f"p2a_base_self_score1b={score['p2a_base_mean']}", f"mh_self_score1b={score['mh_mean']}", f"mh_minus_base_self_score1b={score['mh_minus_base']}",
        f"p2a_base_temporal={timing['p2a_base_mean']}", f"mh_temporal={timing['mh_mean']}", f"mh_minus_base_temporal={timing['mh_minus_base']}",
        f"positive_folds_stg={fold_positive}/5", f"positive_domain_hand_slices_stg={slice_positive}/4",
        f"replacement_fraction={replacement_fraction}", f"novel_timing_selection_fraction={novel_fraction}",
        f"timing_hypothesis_oracle_gain_over_p2a_route={timing_oracle_gain}", "",
        "[ACCEPTANCE]",
        f"status={status}", f"engineering_gate={engineering_gate}", f"primary_gate={primary_gate}",
        f"spatial_guard={spatial_guard}", f"timing_guard={timing_guard}", f"quality_gate={quality_gate}",
        f"utilization_gate={utilization_gate}", f"full_oof_allowed={full_allowed}", "",
        "[PRIMARY_ARTIFACTS]",
        "P2A_MH_ACCEPTANCE.json", "summary_metrics.csv", "per_sample_comparison.csv",
        "candidate_metrics.csv", "timing_hypothesis_manifest.csv", "timing_hypothesis_audit.csv",
        "ranker_calibration_audit.csv", "wordphase_distribution_audit.json", "fold_results.csv",
        "motions/selected/*.npz",
    ]
    (output_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")



def normalize_cli_argv(argv: Sequence[str]) -> list[str]:
    """Normalize comma-separated options whose value may begin with ``-``.

    ``argparse`` treats a separate token such as ``-0.02,0.0,0.01`` as a new
    option rather than as the string value of ``--replacement-thresholds``.
    Combining the option and value with ``=`` keeps the scientific defaults
    unchanged while making direct Python invocation robust as well as the
    shell launcher.
    """
    out: list[str] = []
    i = 0
    csv_options = {"--replacement-thresholds", "--ridge-lambdas"}
    while i < len(argv):
        token = str(argv[i])
        if token in csv_options and i + 1 < len(argv):
            value = str(argv[i + 1])
            if value.startswith("-"):
                out.append(f"{token}={value}")
                i += 2
                continue
        out.append(token)
        i += 1
    return out

def self_test() -> None:
    raw_cli = ["--replacement-thresholds", "-0.02,0.0,0.01"]
    normalized_cli = normalize_cli_argv(raw_cli)
    if normalized_cli != ["--replacement-thresholds=-0.02,0.0,0.01"]:
        raise MHError("Negative CSV CLI normalization self-test failed")
    p = np.asarray([0.02, 0.08, 0.30, 0.10, 0.05, 0.35, 0.08, 0.02], dtype=np.float64)
    p /= p.sum()
    dist = {
        "clip_length": 120, "prob": p, "expected_ratio": float(np.sum(p * np.linspace(0, 1, len(p)))),
        "expected_frame": 60, "entropy_norm": 0.8, "max_prob": float(p.max()),
        "top_bin_margin": 0.05, "std_ratio": 0.2,
    }
    h = build_hypotheses(dist, 70, 4, 5)
    if len(h) != 4 or len({x["planned_frame"] for x in h}) != 4:
        raise MHError("Hypothesis construction self-test failed")
    names = ranker_feature_names()
    q = {"domain": "natural", "hand": 0, "clip_length": 120, "target_distance": 2.0, "target_azimuth_deg": 30, "target_elevation_deg": 5}
    c = {
        "planned_frame": 60, "self_frame": 61, "det_source": 60, "det_refined": 61, "det_heuristic": 59,
        "detector_agreement": 0.5, "planned_score1b_proxy20": 0.99, "self_score1b_proxy20": 0.95,
        "self_angle_deg": 5, "joint_jerk": 0.0003, "foot_sliding": 0.001, "pose_deviation": 0.001,
        "max_reach": 0.5, "joint_limit_fraction": 0, "donor_rank": 1, "donor_distance": 0.3,
    }
    x = build_ranker_feature(q, h[0], c)
    if len(x) != len(names) or not np.all(np.isfinite(x)):
        raise MHError("Ranker feature self-test failed")
    X = np.stack([x, x + 0.01, x - 0.01, x + 0.02])
    y = np.asarray([0.4, 0.5, 0.3, 0.6])
    beta, mu, sd = fit_weighted_ridge(X, y, np.asarray(["a", "b", "c", "d"], dtype=object), np.asarray(["q1", "q2", "q3", "q4"], dtype=object), 0.5)
    pred = ridge_predict(X, beta, mu, sd)
    if pred.shape != (4,) or not np.all(np.isfinite(pred)):
        raise MHError("Ridge self-test failed")


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
    ap.add_argument("--p2b-dir", type=Path, required=True)
    ap.add_argument("--smplx-model-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--queries-per-slice", type=int, default=10)
    ap.add_argument("--hypothesis-count", type=int, default=4)
    ap.add_argument("--donors-per-hypothesis", type=int, default=4)
    ap.add_argument("--min-hypothesis-sep-frames", type=int, default=6)
    ap.add_argument("--wordphase-batch-size", type=int, default=128)
    ap.add_argument("--max-words", type=int, default=64)
    ap.add_argument("--fk-batch-size", type=int, default=512)
    ap.add_argument("--ik-batch-size", type=int, default=64)
    ap.add_argument("--ik-steps", type=int, default=50)
    ap.add_argument("--ik-lr", type=float, default=0.05)
    ap.add_argument("--ik-max-delta", type=float, default=1.5)
    ap.add_argument("--hold-plateau", type=int, default=5)
    ap.add_argument("--hold-taper", type=int, default=8)
    ap.add_argument("--ridge-lambdas", type=str, default="0.5")
    ap.add_argument("--replacement-thresholds", type=str, default="-0.02,0.0,0.01,0.02,0.05,0.10")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(normalize_cli_argv(sys.argv[1:]))
    if args.self_test:
        self_test()
        print("SELF_TEST=PASS")
        return 0
    if args.wordphase_batch_size != FROZEN_P1A_REPLAY_BATCH_SIZE:
        raise MHError(
            f"WordPhase OOF replay requires batch_size={FROZEN_P1A_REPLAY_BATCH_SIZE}; "
            f"received {args.wordphase_batch_size}. The accepted P1A BiGRU was evaluated without packed "
            "sequences, so changing batch padding changes the backward GRU state."
        )
    for name in ("data_root", "project_root", "p0c0_dir", "m1_full_dir", "p1a_dir", "p1_full_dir", "p2a_dir", "p2b_dir", "output_dir"):
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

    paths = {
        "manifest": args.p0c0_dir / "shared_generation_manifest.csv",
        "p0c0_accept": args.p0c0_dir / "P0C0_ACCEPTANCE.json",
        "m1_timing": args.m1_full_dir / "aggregate" / "timing_manifest_all_folds.csv",
        "m1_worker": args.m1_full_dir / "_code" / "repointer_b0_p0c_m1_full_fold_v0_2_0.py",
        "source_detector": args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_0_0.npz",
        "refined_detector": args.p0c0_dir / "frozen_inputs" / "apex_detector_model_v1_1_0.npz",
        "p1a_accept": args.p1a_dir / "P1A_ACCEPTANCE.json",
        "p1a_pred": args.p1a_dir / "oof_timing_predictions.csv",
        "p1_accept": args.p1_full_dir / "P1_FULL_ACCEPTANCE.json",
        "p1_per": args.p1_full_dir / "per_sample_comparison.csv",
        "p1_candidates": args.p1_full_dir / "candidate_metrics.csv",
        "p2a_accept": args.p2a_dir / "P2A_ACCEPTANCE.json",
        "p2a_routes": args.p2a_dir / "oof_phase_router_predictions.csv",
        "p2b_accept": args.p2b_dir / "P2B_ACCEPTANCE.json",
    }
    for f in EXPECTED_FOLDS:
        paths[f"wordphase_fold{f}"] = args.p1a_dir / f"wordphasenet_fold{f}.pt"
    for name, p in paths.items():
        if not p.is_file():
            raise MHError(f"Missing required input {name}: {p}")
    if json_load(paths["p0c0_accept"]).get("status") != "P0C0_SHARED_PROTOCOL_FROZEN":
        raise MHError("P0C0 is not frozen")
    if json_load(paths["p1a_accept"]).get("status") != "P1A_WORDPHASENET_ACCEPTED_P1_GENERATION_ALLOWED":
        raise MHError("P1A is not accepted")
    if json_load(paths["p1_accept"]).get("status") != "P1_WORDPHASE_FULL_OOF_ACCEPTED":
        raise MHError("P1 full OOF is not accepted")
    if json_load(paths["p2a_accept"]).get("status") != "P2A_PHASE_ROUTER_ACCEPTED":
        raise MHError("P2A is not accepted")
    p2b_accept = json_load(paths["p2b_accept"])
    if p2b_accept.get("status") != "P2B_RESIDUAL_PROJECTION_NO_MATERIAL_HEADROOM":
        raise MHError("P2B closeout state is unexpected")

    module = load_module(paths["m1_worker"])
    module.self_test()
    parity = {
        "status": "PASS",
        "frozen_m1_worker": str(paths["m1_worker"]),
        "frozen_m1_worker_sha256": sha256_file(paths["m1_worker"]),
        "imported_functions": [
            "generation_view", "load_clip_frame", "load_full_clip", "descriptor", "warp_clip",
            "target_vec", "optimize_apex_batch", "apply_hold", "resolve_scene", "evaluate_candidate",
        ],
        "evaluate_candidate_ast_hash": function_ast_hash(paths["m1_worker"], "evaluate_candidate"),
        "motion_quality_ast_hash": function_ast_hash(paths["m1_worker"], "motion_quality"),
    }
    json_dump(args.output_dir / "method_parity_audit.json", parity)

    rows = read_csv(paths["manifest"])
    timing_rows = read_csv(paths["m1_timing"])
    validation_timing = [r for r in timing_rows if str(r.get("role")) in {"validation", "validation_generation"}]
    p1a_preds = read_csv(paths["p1a_pred"])
    p1_rows = read_csv(paths["p1_per"])
    p1_candidates = read_csv(paths["p1_candidates"])
    route_rows = read_csv(paths["p2a_routes"])
    timing_by_id = {str(r["sample_id"]): r for r in validation_timing}
    pred_by_id = {str(r["sample_id"]): r for r in p1a_preds}
    p1_by_id = {str(r["sample_id"]): r for r in p1_rows}
    route_by_id = {str(r["sample_id"]): r for r in route_rows}
    if not (len(rows) == len(timing_by_id) == len(pred_by_id) == len(p1_by_id) == len(route_by_id) == 2525):
        raise MHError(f"Coverage mismatch: manifest={len(rows)} timing={len(timing_by_id)} pred={len(pred_by_id)} p1={len(p1_by_id)} route={len(route_by_id)}")

    dist_cache_path = args.output_dir / "wordphase_oof_distribution_cache_v0_1_2.npz"
    dist_by_id, dist_audit, dist_regression_rows = infer_wordphase_distributions(
        rows, timing_by_id, pred_by_id, args.p1a_dir, device, args.wordphase_batch_size, args.max_words,
    )
    json_dump(args.output_dir / "wordphase_distribution_audit.json", dist_audit)
    write_csv(args.output_dir / "wordphase_distribution_regression_rows.csv", dist_regression_rows)
    if dist_audit["status"] != "PASS":
        raise MHError(f"WordPhase checkpoint regression failed after frozen replay: {dist_audit}")
    sample_order = [str(r["sample_id"]) for r in rows]
    np.savez_compressed(
        dist_cache_path,
        sample_ids=np.asarray(sample_order, dtype="U128"),
        probabilities=np.stack([np.asarray(dist_by_id[sid]["prob"], dtype=np.float32) for sid in sample_order]),
        expected_frames=np.asarray([int(dist_by_id[sid]["expected_frame"]) for sid in sample_order], dtype=np.int16),
        entropy_norm=np.asarray([float(dist_by_id[sid]["entropy_norm"]) for sid in sample_order], dtype=np.float32),
        std_ratio=np.asarray([float(dist_by_id[sid]["std_ratio"]) for sid in sample_order], dtype=np.float32),
    )

    selected_rows = select_pilot_rows(rows, args.p2b_dir, args.queries_per_slice, args.seed)
    args.expected_queries = len(selected_rows)
    write_csv(args.output_dir / "pilot_query_manifest.csv", selected_rows)
    hypotheses_by_id: dict[str, list[dict[str, Any]]] = {}
    timing_audit_rows: list[dict[str, Any]] = []
    for r in selected_rows:
        sid = str(r["sample_id"])
        med = inum(pred_by_id[sid].get("oof_domain_hand_median_pred"))
        hyps = build_hypotheses(dist_by_id[sid], med, args.hypothesis_count, args.min_hypothesis_sep_frames)
        hypotheses_by_id[sid] = hyps
        hs = inum(r.get("hold_start")); he = inum(r.get("hold_end"))
        route_choice = str(route_by_id[sid].get("router_choice"))
        route_frame = inum(pred_by_id[sid].get("wordphasenet_oof_pred" if route_choice == "wordphase" else "oof_domain_hand_median_pred"))
        base_d = distance_to_hold(route_frame, hs, he)
        ds = [distance_to_hold(inum(h["planned_frame"]), hs, he) for h in hyps]
        timing_audit_rows.append({
            "sample_id": sid, "fold": inum(r.get("fold")), "domain": r.get("domain"), "hand": inum(r.get("hand")),
            "p2a_route_choice": route_choice, "p2a_route_frame": route_frame,
            "p2a_route_distance_to_hold": base_d, "p2a_route_timing_temporal_sigma5": temporal_score(base_d),
            "hypothesis_count": len(hyps), "hypothesis_frames": [h["planned_frame"] for h in hyps],
            "hypothesis_names": [h["hypothesis"] for h in hyps],
            "hypothesis_oracle_distance_to_hold": min(ds),
            "hypothesis_oracle_temporal_sigma5": max(temporal_score(d) for d in ds),
        })
    write_csv(args.output_dir / "timing_hypothesis_audit.csv", timing_audit_rows)
    write_csv(args.output_dir / "timing_hypothesis_manifest.csv", [
        {"sample_id": sid, **h} for sid, hs in hypotheses_by_id.items() for h in hs
    ])

    training_rows, Xtrain, ytrain, candidate_folds, candidate_groups, by_query, base_idx = prepare_training_candidates(
        p1_candidates, p1_by_id, route_by_id, dist_by_id,
    )
    lambdas = parse_float_list(args.ridge_lambdas)
    thresholds = parse_float_list(args.replacement_thresholds)
    calibration_rows: list[dict[str, Any]] = []
    ranker_params: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
    for fold in EXPECTED_FOLDS:
        lam, threshold, audit_rows = nested_calibrate_ranker(
            fold, training_rows, Xtrain, ytrain, candidate_folds, candidate_groups,
            by_query, base_idx, lambdas, thresholds,
        )
        calibration_rows.extend(audit_rows)
        beta, mu, sd = fit_outer_ranker(fold, training_rows, Xtrain, ytrain, candidate_folds, candidate_groups, lam)
        ranker_params[fold] = (beta, mu, sd, lam, threshold)
    write_csv(args.output_dir / "ranker_calibration_audit.csv", calibration_rows)

    assets = module.find_smplx_assets(args.smplx_model_dir, args.data_root, args.project_root)
    if assets.get("status") != "OK":
        raise MHError(f"SMPL-X assets missing: {assets}")
    preflight = {
        "status": "P2A_MH_PREFLIGHT_ACCEPTED" if dist_audit["status"] == "PASS" else "P2A_MH_PREFLIGHT_FAILED",
        "device": device, "query_count": len(selected_rows),
        "hypothesis_count": args.hypothesis_count, "donors_per_hypothesis": args.donors_per_hypothesis,
        "new_generated_candidates": args.hypothesis_count * args.donors_per_hypothesis - 1,
        "selection_pool_including_p2a": args.hypothesis_count * args.donors_per_hypothesis,
        "wordphase_regression_gate": dist_audit["status"],
        "wordphase_replay_batch_size": args.wordphase_batch_size,
        "wordphase_padding_sensitive_replay_contract": True,
        "smplx_assets": assets, "motion_generation_started": False,
        "ranker_feature_dim": len(ranker_feature_names()),
        "ranker_calibration_complete": True,
    }
    json_dump(args.output_dir / "PREFLIGHT.json", preflight)
    if args.preflight_only:
        print("PREFLIGHT_STATUS=P2A_MH_PREFLIGHT_ACCEPTED")
        print("MOTION_GENERATION_STARTED=FALSE")
        print(f"RESUME_RUN_DIR={args.output_dir}")
        return 0

    engine = module.SMPLXEngine(assets, args.device, args.fk_batch_size)
    source_model = module.load_linear_model(paths["source_detector"])
    refined_model = module.load_linear_model(paths["refined_detector"])
    p1_candidate_lookup: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for r in p1_candidates:
        p1_candidate_lookup[(str(r["query_id"]), str(r["method"]), str(r["donor_id"]), inum(r.get("donor_rank")))] = r

    fold_root = args.output_dir / "folds"
    incomplete_root = args.output_dir / "_incomplete"
    for fold in EXPECTED_FOLDS:
        expected = sum(1 for r in selected_rows if inum(r.get("fold")) == fold)
        fold_dir = fold_root / f"fold_{fold}"
        if validate_complete_fold(fold_dir, expected):
            print(f"[RESUME] fold={fold} complete; skipping", flush=True)
            continue
        if fold_dir.exists():
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_existing(fold_dir, incomplete_root / f"fold_{fold}_{stamp}")
        beta, mu, sd, lam, threshold = ranker_params[fold]
        print(f"[P2A-MH] starting fold={fold} queries={expected} lambda={lam} threshold={threshold}", flush=True)
        run_fold(
            fold, fold_dir, selected_rows, rows, p1_by_id, route_by_id, dist_by_id,
            hypotheses_by_id, module, engine, source_model, refined_model,
            p1_candidate_lookup, beta, mu, sd, lam, threshold, args,
        )

    all_per: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_timing: list[dict[str, Any]] = []
    all_selection: list[dict[str, Any]] = []
    all_audits: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        fold_dir = fold_root / f"fold_{fold}"
        expected = sum(1 for r in selected_rows if inum(r.get("fold")) == fold)
        if not validate_complete_fold(fold_dir, expected):
            raise MHError(f"Fold {fold} incomplete")
        all_per.extend(read_csv(fold_dir / "per_sample_comparison.csv"))
        all_candidates.extend(read_csv(fold_dir / "candidate_metrics.csv"))
        all_timing.extend(read_csv(fold_dir / "timing_hypothesis_manifest.csv"))
        all_selection.extend(read_csv(fold_dir / "selection_manifest.csv"))
        all_audits.append(json_load(fold_dir / "validation_gt_usage_audit.json"))
    write_csv(args.output_dir / "per_sample_comparison.csv", all_per)
    write_csv(args.output_dir / "candidate_metrics.csv", all_candidates)
    write_csv(args.output_dir / "timing_hypothesis_manifest_generated.csv", all_timing)
    write_csv(args.output_dir / "selection_manifest.csv", all_selection)
    json_dump(args.output_dir / "validation_gt_usage_audit.json", {
        "status": "PASS" if all(a.get("status") == "PASS" for a in all_audits) else "FAIL",
        "fold_audits": all_audits,
    })
    aggregate(args, all_per, timing_audit_rows, args.output_dir)

    source_binding = []
    for name, p in paths.items():
        source_binding.append({"name": name, "path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    json_dump(args.output_dir / "source_binding.json", source_binding)
    env = {
        "created_utc": now_utc(), "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable, "cwd": os.getcwd(),
        "device": engine.device,
    }
    json_dump(args.output_dir / "environment.json", env)
    run_manifest = []
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
