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
import random
import re
import socket
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

VERSION = "0.1.0"
SCHEMA_VERSION = "1.0.0"
FPS = 30.0
DEICTIC = {"this", "that", "these", "those", "here", "there", "it", "one"}
SPATIAL = {"left", "right", "front", "behind", "back", "above", "below", "under", "over", "next", "near", "beside", "between", "on", "in", "at", "corner", "side"}
FUNCTION = {"the", "a", "an", "to", "of", "is", "are", "was", "were", "and", "or", "with", "you", "can", "could", "please"}


class P1AError(RuntimeError):
    pass


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def inum(x: Any, default: int = -1) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


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


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def token_normalize(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"^[^\w']+|[^\w']+$", "", x, flags=re.UNICODE)
    return x or "<empty>"


def token_id(token: str, buckets: int) -> int:
    # 0 is PAD; 1 is NO_WORD; hashed words are 2..buckets+1.
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
        except Exception:
            # More permissive fallback for pipe/tab formatted rows.
            nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+)", line)
            if len(nums) < 2:
                continue
            a, b = float(nums[0]), float(nums[1])
            after = line.find(nums[1]) + len(nums[1])
            token = line[after:].strip(" \t|")
        else:
            token = parts[2].strip() if len(parts) >= 3 else ""
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if b < a:
            a, b = b, a
        out.append((a, b, token_normalize(token)))
    out.sort(key=lambda z: (z[0], z[1], z[2]))
    return out


def transcript_index(root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = defaultdict(list)
    for ext in ("*.tsv", "*.txt"):
        for p in sorted(root.rglob(ext)):
            stem = p.stem
            keys = {stem, stem.removesuffix("_main"), stem.removesuffix("_words"), stem.removesuffix("_transcript")}
            for k in keys:
                if k:
                    idx[k].append(p)
    return idx


def resolve_transcript(rec: Mapping[str, Any], timing_rec: Mapping[str, Any], idx: Mapping[str, list[Path]]) -> tuple[Path | None, str]:
    stated = str(timing_rec.get("transcript_path", "")).strip()
    if stated:
        p = Path(stated)
        if p.is_file():
            return p, "frozen_timing_manifest"
    for key in (str(rec.get("sample_id", "")), str(rec.get("recording_id", ""))):
        if key in idx and len(idx[key]) == 1:
            return idx[key][0], "exact_index"
        k2 = key.removesuffix("_main")
        if k2 in idx and len(idx[k2]) == 1:
            return idx[k2][0], "normalized_index"
    rid = str(rec.get("recording_id", ""))
    sid = str(rec.get("sample_id", ""))
    candidates: list[Path] = []
    for key, paths in idx.items():
        if (rid and (key.startswith(rid) or rid.startswith(key))) or (sid and (key.startswith(sid) or sid.startswith(key))):
            candidates.extend(paths)
    uniq = sorted(set(candidates))
    return (uniq[0], "fuzzy_unique") if len(uniq) == 1 else (None, "missing_or_ambiguous")


@dataclass
class Example:
    sample_id: str
    recording_id: str
    leakage_group_id: str
    fold: int
    domain: str
    hand: int
    clip_length: int
    gt_peak: int
    hold_start: int
    hold_end: int
    fixed_peak: int
    token_ids: np.ndarray
    word_features: np.ndarray
    base_clip_features: np.ndarray
    transcript_status: str
    transcript_path: str
    word_count: int


def make_example(
    rec: Mapping[str, Any], timing_rec: Mapping[str, Any], idx: Mapping[str, list[Path]],
    word_cache: dict[Path, list[tuple[float, float, str]]], buckets: int, max_words: int,
) -> Example:
    sid = str(rec["sample_id"])
    domain = str(rec.get("domain"))
    T = inum(rec.get("clip_length"))
    gt = inum(rec.get("released_peak_frame"))
    hs = inum(rec.get("hold_start"))
    he = inum(rec.get("hold_end"))
    fixed = inum(timing_rec.get("planned_peak"))
    if not (T > 4 and 0 <= gt < T and 0 <= hs <= he < T and 0 <= fixed < T):
        raise P1AError(f"Invalid timing geometry for {sid}: T={T}, gt={gt}, hold={hs}:{he}, fixed={fixed}")

    path, tstatus = resolve_transcript(rec, timing_rec, idx)
    all_words = word_cache.setdefault(path, parse_word_times(path)) if path is not None else []
    gstart = inum(rec.get("global_start"), -1)
    clip_start = (gstart / FPS) if domain == "natural" and gstart >= 0 else 0.0
    clip_end = clip_start + T / FPS
    selected: list[tuple[float, float, str]] = []
    for a, b, w in all_words:
        if b >= clip_start - 0.05 and a <= clip_end + 0.05:
            ar = max(0.0, a - clip_start)
            br = min(T / FPS, b - clip_start)
            if br >= -0.05 and ar <= T / FPS + 0.05:
                selected.append((max(0.0, ar), max(0.0, min(T / FPS, br)), w))
    selected.sort(key=lambda z: (z[0], z[1]))
    if len(selected) > max_words:
        # Retain broad temporal coverage deterministically.
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
            a / clip_sec,
            b / clip_sec,
            dur / clip_sec,
            pause_before / clip_sec,
            pause_after / clip_sec,
            i / max(1, n - 1),
            min(len(w), 24) / 24.0,
            float(w in DEICTIC),
            float(w in SPATIAL),
            float(w in FUNCTION),
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
        float(inum(rec.get("hand"))),
        T / 180.0,
        math.log(max(dist, 1e-4)),
        float(u[0]), float(u[1]), float(u[2]),
        min(n, max_words) / max(1, max_words),
        first_start, last_end, max(0.0, last_end - first_start),
        mean_dur, mean_pause, max_pause,
        fixed_ratio,
    ], dtype=np.float32)
    return Example(
        sample_id=sid,
        recording_id=str(rec.get("recording_id", "")),
        leakage_group_id=str(rec.get("leakage_group_id", "")),
        fold=inum(rec.get("fold")),
        domain=domain,
        hand=inum(rec.get("hand")),
        clip_length=T,
        gt_peak=gt,
        hold_start=hs,
        hold_end=he,
        fixed_peak=fixed,
        token_ids=np.asarray(toks, dtype=np.int64),
        word_features=np.asarray(feats, dtype=np.float32),
        base_clip_features=base_clip,
        transcript_status=tstatus,
        transcript_path=str(path or ""),
        word_count=n,
    )


def distance_to_hold(pred: int, start: int, end: int) -> int:
    return start - pred if pred < start else (pred - end if pred > end else 0)


def temporal_score(d: int, sigma: float) -> float:
    return 1.0 if d <= 0 else float(math.exp(-(float(d) ** 2) / (2.0 * sigma ** 2)))


def metric_row(examples: Sequence[Example], preds: Mapping[str, int], variant: str, group_name: str, group_value: str) -> dict[str, Any]:
    selected: list[Example] = []
    for ex in examples:
        if group_name == "overall":
            ok = True
        elif group_name == "fold":
            ok = str(ex.fold) == str(group_value)
        elif group_name == "domain":
            ok = ex.domain == str(group_value)
        elif group_name == "hand":
            ok = str(ex.hand) == str(group_value)
        else:
            ok = False
        if ok:
            selected.append(ex)
    if not selected:
        return {"variant": variant, "group_name": group_name, "group_value": group_value, "n": 0}
    signed = np.asarray([int(preds[ex.sample_id]) - ex.gt_peak for ex in selected], dtype=np.float64)
    ae = np.abs(signed)
    dh = np.asarray([distance_to_hold(int(preds[ex.sample_id]), ex.hold_start, ex.hold_end) for ex in selected], dtype=np.float64)
    t5 = np.asarray([temporal_score(int(x), 5.0) for x in dh], dtype=np.float64)
    t10 = np.asarray([temporal_score(int(x), 10.0) for x in dh], dtype=np.float64)
    return {
        "variant": variant,
        "group_name": group_name,
        "group_value": group_value,
        "n": len(selected),
        "mean_signed_error_frames": float(np.mean(signed)),
        "mean_abs_error_frames": float(np.mean(ae)),
        "median_abs_error_frames": float(np.median(ae)),
        "p90_abs_error_frames": float(np.quantile(ae, 0.90)),
        "within_5_frames": float(np.mean(ae <= 5)),
        "within_10_frames": float(np.mean(ae <= 10)),
        "hold_hit_rate": float(np.mean(dh == 0)),
        "mean_distance_to_hold": float(np.mean(dh)),
        "mean_temporal_proxy_sigma5": float(np.mean(t5)),
        "mean_temporal_proxy_sigma10": float(np.mean(t10)),
        "early_fraction": float(np.mean(signed < 0)),
        "late_fraction": float(np.mean(signed > 0)),
    }


def make_soft_targets(ratios: "Any", bins: int, device: "Any") -> "Any":
    import torch
    centers = torch.linspace(0.0, 1.0, bins, device=device)[None, :]
    sigma = 1.5 / max(1, bins - 1)
    y = torch.exp(-0.5 * ((centers - ratios[:, None]) / sigma) ** 2)
    return y / torch.clamp(y.sum(dim=1, keepdim=True), min=1e-12)


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


def collate(batch: Sequence[Example], medians: Mapping[tuple[str, int], float], word_mean: np.ndarray, word_std: np.ndarray, clip_mean: np.ndarray, clip_std: np.ndarray):
    import torch
    B = len(batch)
    L = max(len(x.token_ids) for x in batch)
    D = batch[0].word_features.shape[1]
    tok = np.zeros((B, L), dtype=np.int64)
    wf = np.zeros((B, L, D), dtype=np.float32)
    mask = np.zeros((B, L), dtype=bool)
    cf = np.zeros((B, len(batch[0].base_clip_features) + 1), dtype=np.float32)
    ratio = np.zeros(B, dtype=np.float32)
    for i, ex in enumerate(batch):
        n = len(ex.token_ids)
        tok[i, :n] = ex.token_ids
        wf[i, :n] = (ex.word_features - word_mean) / word_std
        mask[i, :n] = True
        raw_cf = np.concatenate([ex.base_clip_features, np.asarray([medians[(ex.domain, ex.hand)]], dtype=np.float32)])
        cf[i] = (raw_cf - clip_mean) / clip_std
        ratio[i] = ex.gt_peak / max(1, ex.clip_length - 1)
    return (
        torch.from_numpy(tok), torch.from_numpy(wf), torch.from_numpy(mask),
        torch.from_numpy(cf), torch.from_numpy(ratio),
    )


def training_stats(train: Sequence[Example], medians: Mapping[tuple[str, int], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    word = np.concatenate([x.word_features for x in train], axis=0).astype(np.float64)
    wm, ws = word.mean(axis=0), word.std(axis=0)
    ws[ws < 1e-6] = 1.0
    clip = np.stack([
        np.concatenate([x.base_clip_features, np.asarray([medians[(x.domain, x.hand)]], dtype=np.float32)])
        for x in train
    ]).astype(np.float64)
    cm, cs = clip.mean(axis=0), clip.std(axis=0)
    cs[cs < 1e-6] = 1.0
    return wm.astype(np.float32), ws.astype(np.float32), cm.astype(np.float32), cs.astype(np.float32)


def iter_batches(items: Sequence[Example], batch_size: int, rng: np.random.Generator) -> list[list[Example]]:
    ids = rng.permutation(len(items))
    return [[items[int(j)] for j in ids[i:i + batch_size]] for i in range(0, len(ids), batch_size)]


def train_one_fold(
    train: Sequence[Example], val: Sequence[Example], fold: int, args: Any, device: str, output_dir: Path,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    import torch
    torch.manual_seed(args.seed + 1000 * fold)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed + 1000 * fold)
    medians: dict[tuple[str, int], float] = {}
    for domain in ("natural", "synthetic"):
        for hand in (0, 1):
            vals = [x.gt_peak / max(1, x.clip_length - 1) for x in train if x.domain == domain and x.hand == hand]
            medians[(domain, hand)] = float(np.median(vals)) if vals else 0.5
    wm, ws, cm, cs = training_stats(train, medians)
    wrapper = WordPhaseNet(torch, args.hash_buckets, train[0].word_features.shape[1], len(cm), args.bins)
    model = wrapper.net.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.learning_rate * 0.05)
    rng = np.random.default_rng(args.seed + fold * 10007)
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in iter_batches(train, args.batch_size, rng):
            tok, wf, mask, cf, ratio = collate(batch, medians, wm, ws, cm, cs)
            tok, wf, mask, cf, ratio = [z.to(device) for z in (tok, wf, mask, cf, ratio)]
            logits = model(tok, wf, mask, cf)
            soft = make_soft_targets(ratio, args.bins, device)
            logp = torch.log_softmax(logits, dim=1)
            ce = -torch.sum(soft * logp, dim=1).mean()
            centers = torch.linspace(0.0, 1.0, args.bins, device=device)
            expected = torch.sum(torch.softmax(logits, dim=1) * centers[None, :], dim=1)
            reg = torch.nn.functional.smooth_l1_loss(expected, ratio, beta=0.05)
            loss = ce + 0.50 * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            loss_sum += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        sched.step()
        history.append({"fold": fold, "epoch": epoch + 1, "train_loss": loss_sum / max(1, count), "lr": sched.get_last_lr()[0]})

    model.eval()
    preds: dict[str, int] = {}
    with torch.no_grad():
        for start in range(0, len(val), args.batch_size):
            batch = list(val[start:start + args.batch_size])
            tok, wf, mask, cf, _ratio = collate(batch, medians, wm, ws, cm, cs)
            tok, wf, mask, cf = [z.to(device) for z in (tok, wf, mask, cf)]
            logits = model(tok, wf, mask, cf)
            prob = torch.softmax(logits, dim=1)
            centers = torch.linspace(0.0, 1.0, args.bins, device=device)
            expected = torch.sum(prob * centers[None, :], dim=1).cpu().numpy()
            for ex, r in zip(batch, expected.tolist()):
                frame = int(round(float(r) * max(1, ex.clip_length - 1)))
                preds[ex.sample_id] = int(np.clip(frame, 2, max(2, ex.clip_length - 3)))

    state = {
        "model_state": model.state_dict(),
        "word_mean": wm,
        "word_std": ws,
        "clip_mean": cm,
        "clip_std": cs,
        "medians": {f"{k[0]}_hand{k[1]}": v for k, v in medians.items()},
        "bins": args.bins,
        "hash_buckets": args.hash_buckets,
        "version": VERSION,
        "fold": fold,
    }
    torch.save(state, output_dir / f"wordphasenet_fold{fold}.pt")
    meta = {"fold": fold, "train_n": len(train), "val_n": len(val), "final_train_loss": history[-1]["train_loss"], "medians": state["medians"]}
    return preds, history, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--p0c0-dir", type=Path, required=True)
    ap.add_argument("--m1-full-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=0.002)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--hash-buckets", type=int, default=2048)
    ap.add_argument("--max-words", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    manifest_path = args.p0c0_dir / "shared_generation_manifest.csv"
    timing_path = args.m1_full_dir / "aggregate" / "timing_manifest_all_folds.csv"
    per_path = args.m1_full_dir / "aggregate" / "per_sample_comparison_all_folds.csv"
    coverage_path = args.m1_full_dir / "aggregate" / "fold_coverage.csv"
    p0c0_accept = json_load(args.p0c0_dir / "P0C0_ACCEPTANCE.json")
    if p0c0_accept.get("status") != "P0C0_SHARED_PROTOCOL_FROZEN":
        raise P1AError("P0C0 protocol is not frozen")
    coverage = read_csv(coverage_path)
    if len(coverage) != 5 or any(r.get("status") != "PASS" or r.get("gt_usage_audit") != "PASS" for r in coverage):
        raise P1AError(f"M1 fold coverage/GT audit is not PASS: {coverage}")

    manifest = read_csv(manifest_path)
    per_rows = read_csv(per_path)
    timing_rows = read_csv(timing_path)
    validation_timing = [r for r in timing_rows if str(r.get("role")) in {"validation", "validation_generation"}]
    timing_by_id = {str(r["sample_id"]): r for r in validation_timing}
    per_by_id = {str(r["sample_id"]): r for r in per_rows}
    if len(manifest) != 2525 or len(per_by_id) != len(manifest) or len(timing_by_id) != len(manifest):
        raise P1AError(f"Coverage mismatch manifest={len(manifest)} per={len(per_by_id)} timing={len(timing_by_id)}")

    # Optional deterministic smoke subset preserves all folds/domains/hands but is not formal evidence.
    if args.max_rows > 0 and args.max_rows < len(manifest):
        ordered = sorted(manifest, key=lambda r: hashlib.sha256((str(r["sample_id"]) + str(args.seed)).encode()).hexdigest())
        manifest = ordered[:args.max_rows]

    idx = transcript_index(args.data_root / "data" / "text")
    word_cache: dict[Path, list[tuple[float, float, str]]] = {}
    examples: list[Example] = []
    feature_rows: list[dict[str, Any]] = []
    for i, rec in enumerate(manifest):
        sid = str(rec["sample_id"])
        ex = make_example(rec, timing_by_id[sid], idx, word_cache, args.hash_buckets, args.max_words)
        examples.append(ex)
        feature_rows.append({
            "sample_id": sid, "fold": ex.fold, "domain": ex.domain, "hand": ex.hand,
            "clip_length": ex.clip_length, "word_count": ex.word_count,
            "transcript_status": ex.transcript_status, "transcript_path": ex.transcript_path,
            "fixed_peak": ex.fixed_peak, "gt_peak": ex.gt_peak,
            "fixed_signed_error": ex.fixed_peak - ex.gt_peak,
        })
        if (i + 1) % 250 == 0:
            print(f"[features] {i+1}/{len(manifest)}", flush=True)
    write_csv(args.output_dir / "speech_feature_manifest.csv", feature_rows)

    folds = sorted(set(x.fold for x in examples))
    if folds != [0, 1, 2, 3, 4] and args.max_rows == 0:
        raise P1AError(f"Unexpected folds: {folds}")

    # Baselines: frozen M1 timing and an honest fold-training domain/hand median.
    fixed_preds = {x.sample_id: x.fixed_peak for x in examples}
    median_preds: dict[str, int] = {}
    for fold in folds:
        train = [x for x in examples if x.fold != fold]
        val = [x for x in examples if x.fold == fold]
        med: dict[tuple[str, int], float] = {}
        for domain in ("natural", "synthetic"):
            for hand in (0, 1):
                vals = [x.gt_peak / max(1, x.clip_length - 1) for x in train if x.domain == domain and x.hand == hand]
                med[(domain, hand)] = float(np.median(vals)) if vals else 0.5
        for ex in val:
            p = int(round(med[(ex.domain, ex.hand)] * max(1, ex.clip_length - 1)))
            median_preds[ex.sample_id] = int(np.clip(p, 2, max(2, ex.clip_length - 3)))

    # OOF WordPhaseNet.
    learned_preds: dict[str, int] = {}
    all_history: list[dict[str, Any]] = []
    fold_training: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = [x for x in examples if x.fold != fold]
        val = [x for x in examples if x.fold == fold]
        train_groups = {x.leakage_group_id for x in train}
        val_groups = {x.leakage_group_id for x in val}
        overlap = sorted(g for g in train_groups & val_groups if g)
        leakage_rows.append({"fold": fold, "train_n": len(train), "val_n": len(val), "leakage_group_overlap_count": len(overlap)})
        if overlap:
            raise P1AError(f"Leakage groups cross fold {fold}: {overlap[:5]}")
        print(f"[fold {fold}] train={len(train)} val={len(val)} device={device}", flush=True)
        pred, hist, meta = train_one_fold(train, val, fold, args, device, args.output_dir)
        learned_preds.update(pred)
        all_history.extend(hist)
        fold_training.append(meta)
    write_csv(args.output_dir / "training_history.csv", all_history)
    write_csv(args.output_dir / "fold_training_summary.csv", fold_training)
    write_csv(args.output_dir / "leakage_audit.csv", leakage_rows)
    if len(learned_preds) != len(examples):
        raise P1AError(f"OOF prediction coverage failure: {len(learned_preds)}/{len(examples)}")

    variants = {
        "m1_last_word_minus_11": fixed_preds,
        "oof_domain_hand_median": median_preds,
        "wordphasenet_oof": learned_preds,
    }
    pred_rows: list[dict[str, Any]] = []
    for ex in examples:
        row: dict[str, Any] = {
            "sample_id": ex.sample_id, "recording_id": ex.recording_id,
            "leakage_group_id": ex.leakage_group_id, "fold": ex.fold,
            "domain": ex.domain, "hand": ex.hand, "clip_length": ex.clip_length,
            "gt_peak": ex.gt_peak, "hold_start": ex.hold_start, "hold_end": ex.hold_end,
            "word_count": ex.word_count, "transcript_status": ex.transcript_status,
        }
        for name, preds in variants.items():
            p = int(preds[ex.sample_id])
            signed = p - ex.gt_peak
            dh = distance_to_hold(p, ex.hold_start, ex.hold_end)
            row[f"{name}_pred"] = p
            row[f"{name}_signed_error"] = signed
            row[f"{name}_abs_error"] = abs(signed)
            row[f"{name}_distance_to_hold"] = dh
            row[f"{name}_temporal_sigma5"] = temporal_score(dh, 5.0)
            row[f"{name}_temporal_sigma10"] = temporal_score(dh, 10.0)
        pred_rows.append(row)
    write_csv(args.output_dir / "oof_timing_predictions.csv", pred_rows)

    metrics: list[dict[str, Any]] = []
    groups: list[tuple[str, str]] = [("overall", "overall")]
    groups.extend(("fold", str(x)) for x in folds)
    groups.extend(("domain", x) for x in ("natural", "synthetic"))
    groups.extend(("hand", str(x)) for x in (0, 1))
    for name, preds in variants.items():
        for gn, gv in groups:
            metrics.append(metric_row(examples, preds, name, gn, gv))
    write_csv(args.output_dir / "timing_metrics.csv", metrics)

    def get(variant: str, gn: str, gv: str) -> dict[str, Any]:
        return next(r for r in metrics if r["variant"] == variant and r["group_name"] == gn and r["group_value"] == gv)

    fixed_overall = get("m1_last_word_minus_11", "overall", "overall")
    median_overall = get("oof_domain_hand_median", "overall", "overall")
    learned_overall = get("wordphasenet_oof", "overall", "overall")
    fold_rows: list[dict[str, Any]] = []
    learned_better_fixed_folds = 0
    learned_better_median_folds = 0
    for fold in folds:
        f = get("m1_last_word_minus_11", "fold", str(fold))
        m = get("oof_domain_hand_median", "fold", str(fold))
        l = get("wordphasenet_oof", "fold", str(fold))
        better_f = l["mean_abs_error_frames"] < f["mean_abs_error_frames"] and l["mean_temporal_proxy_sigma5"] > f["mean_temporal_proxy_sigma5"]
        better_m = l["mean_abs_error_frames"] < m["mean_abs_error_frames"] and l["mean_temporal_proxy_sigma5"] > m["mean_temporal_proxy_sigma5"]
        learned_better_fixed_folds += int(better_f)
        learned_better_median_folds += int(better_m)
        fold_rows.append({
            "fold": fold,
            "fixed_mae": f["mean_abs_error_frames"], "median_mae": m["mean_abs_error_frames"], "learned_mae": l["mean_abs_error_frames"],
            "fixed_temporal5": f["mean_temporal_proxy_sigma5"], "median_temporal5": m["mean_temporal_proxy_sigma5"], "learned_temporal5": l["mean_temporal_proxy_sigma5"],
            "learned_beats_fixed": better_f, "learned_beats_median": better_m,
        })
    write_csv(args.output_dir / "fold_timing_results.csv", fold_rows)

    engineering_gate = len(examples) == (args.max_rows if args.max_rows > 0 else 2525) and len(learned_preds) == len(examples) and all(r["leakage_group_overlap_count"] == 0 for r in leakage_rows)
    beats_fixed = (
        learned_overall["mean_abs_error_frames"] < fixed_overall["mean_abs_error_frames"]
        and learned_overall["mean_temporal_proxy_sigma5"] > fixed_overall["mean_temporal_proxy_sigma5"]
        and learned_better_fixed_folds >= max(1, len(folds) - 1)
    )
    beats_median = (
        learned_overall["mean_abs_error_frames"] < median_overall["mean_abs_error_frames"]
        and learned_overall["mean_temporal_proxy_sigma5"] > median_overall["mean_temporal_proxy_sigma5"]
        and learned_better_median_folds >= max(1, len(folds) - 1)
    )
    if not engineering_gate:
        status = "P1A_ENGINEERING_FAILED"
        p1_generation_allowed = False
    elif beats_fixed and beats_median and args.max_rows == 0:
        status = "P1A_WORDPHASENET_ACCEPTED_P1_GENERATION_ALLOWED"
        p1_generation_allowed = True
    elif beats_fixed:
        status = "P1A_WORDPHASENET_BEATS_FIXED_BUT_NOT_MEDIAN"
        p1_generation_allowed = False
    else:
        status = "P1A_WORDPHASENET_NEEDS_REFINEMENT"
        p1_generation_allowed = False

    failures = []
    for ex in examples:
        lp = learned_preds[ex.sample_id]
        mp = median_preds[ex.sample_id]
        fp = fixed_preds[ex.sample_id]
        la = abs(lp - ex.gt_peak)
        ma = abs(mp - ex.gt_peak)
        fa = abs(fp - ex.gt_peak)
        if la > ma or la > 15:
            failures.append({
                "sample_id": ex.sample_id, "fold": ex.fold, "domain": ex.domain, "hand": ex.hand,
                "clip_length": ex.clip_length, "gt_peak": ex.gt_peak,
                "fixed_peak": fp, "median_peak": mp, "learned_peak": lp,
                "fixed_abs_error": fa, "median_abs_error": ma, "learned_abs_error": la,
                "word_count": ex.word_count, "transcript_status": ex.transcript_status,
            })
    failures.sort(key=lambda r: (-r["learned_abs_error"], r["sample_id"]))
    write_csv(args.output_dir / "timing_failure_cases.csv", failures)

    acceptance = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_utc": now_utc(),
        "status": status,
        "sample_count": len(examples),
        "fold_count": len(folds),
        "engineering_gate": engineering_gate,
        "beats_frozen_m1_fixed_timing": beats_fixed,
        "beats_oof_domain_hand_median": beats_median,
        "positive_folds_vs_fixed": learned_better_fixed_folds,
        "positive_folds_vs_median": learned_better_median_folds,
        "required_positive_folds": max(1, len(folds) - 1),
        "p1_generation_allowed": p1_generation_allowed,
        "fixed_overall": fixed_overall,
        "median_overall": median_overall,
        "wordphasenet_overall": learned_overall,
        "guardrails": [
            "Timing model inputs exclude motion, released peak, hold labels, and alignment labels at inference.",
            "Five-fold OOF uses the frozen group-disjoint fold assignment.",
            "This stage does not generate motion or claim the private organizer timing detector.",
            "Official server scores remain the final benchmark source of truth.",
        ],
        "next_action": (
            "Run a bounded P1 generation pilot replacing only the timing planner; keep retrieval, IK, reranking, and folds frozen."
            if p1_generation_allowed else
            "Do not regenerate motion yet. Inspect timing failures; add audio-prosody features or use the honest OOF median fallback before P1 generation."
        ),
    }
    json_dump(args.output_dir / "P1A_ACCEPTANCE.json", acceptance)

    contract = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "status": "OOF_CANDIDATE" if not p1_generation_allowed else "FROZEN_FOR_P1_PILOT",
        "name": "WordPhaseNet-v0",
        "purpose": "Predict a query gesture apex from timed words and disclosed query context before motion generation.",
        "inputs": [
            "clip-relative timed-word sequence", "hashed token identity", "word durations and pauses",
            "clip length", "domain", "active hand", "target pelvis vector", "fold-training domain/hand median prior",
        ],
        "excluded_at_inference": ["motion", "released peak_frame", "hold labels", "alignment labels", "scene graph", "listener state"],
        "architecture": "hashed token embedding + bidirectional GRU + attention pooling + clip MLP + 64-bin apex-ratio distribution",
        "oof_protocol": "five frozen group-disjoint folds",
        "model_files": [f"wordphasenet_fold{x}.pt" for x in folds],
    }
    json_dump(args.output_dir / "wordphase_contract_v0_1_0.json", contract)

    env = {
        "created_utc": now_utc(), "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable, "cwd": os.getcwd(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device": device,
    }
    json_dump(args.output_dir / "environment.json", env)
    source_paths = [manifest_path, timing_path, per_path, coverage_path, args.p0c0_dir / "P0C0_ACCEPTANCE.json"]
    json_dump(args.output_dir / "source_binding.json", [
        {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in source_paths
    ])

    summary = [
        "REPOINTER P1A — WORDPHASENET GROUP-DISJOINT TIMING OOF",
        "=" * 78,
        f"version={VERSION}", f"created_utc={now_utc()}",
        f"data_root={args.data_root}", f"p0c0_dir={args.p0c0_dir}", f"m1_full_dir={args.m1_full_dir}",
        f"output_dir={args.output_dir}", "dataset_writes=0", "downloads=0", "motion_generation=0",
        "smplx_forward_kinematics=0", "git_writes=0", "training=word_sequence_timing_model_only", "",
        "[COVERAGE]", f"samples={len(examples)}", f"folds={len(folds)}", f"device={device}",
        f"transcript_status_counts={dict(Counter(x.transcript_status for x in examples))}", "",
        "[FROZEN M1 TIMING]",
        f"mean_abs_error_frames={fixed_overall['mean_abs_error_frames']}",
        f"median_abs_error_frames={fixed_overall['median_abs_error_frames']}",
        f"within_10_frames={fixed_overall['within_10_frames']}",
        f"hold_hit_rate={fixed_overall['hold_hit_rate']}",
        f"temporal_proxy_sigma5={fixed_overall['mean_temporal_proxy_sigma5']}", "",
        "[OOF MEDIAN BASELINE]",
        f"mean_abs_error_frames={median_overall['mean_abs_error_frames']}",
        f"median_abs_error_frames={median_overall['median_abs_error_frames']}",
        f"within_10_frames={median_overall['within_10_frames']}",
        f"hold_hit_rate={median_overall['hold_hit_rate']}",
        f"temporal_proxy_sigma5={median_overall['mean_temporal_proxy_sigma5']}", "",
        "[WORDPHASENET OOF]",
        f"mean_abs_error_frames={learned_overall['mean_abs_error_frames']}",
        f"median_abs_error_frames={learned_overall['median_abs_error_frames']}",
        f"within_10_frames={learned_overall['within_10_frames']}",
        f"hold_hit_rate={learned_overall['hold_hit_rate']}",
        f"temporal_proxy_sigma5={learned_overall['mean_temporal_proxy_sigma5']}", "",
        "[ACCEPTANCE]", f"status={status}", f"engineering_gate={engineering_gate}",
        f"beats_frozen_m1_fixed_timing={beats_fixed}", f"beats_oof_domain_hand_median={beats_median}",
        f"positive_folds_vs_fixed={learned_better_fixed_folds}/{len(folds)}",
        f"positive_folds_vs_median={learned_better_median_folds}/{len(folds)}",
        f"p1_generation_allowed={p1_generation_allowed}", f"next_action={acceptance['next_action']}", "",
        "[PRIMARY_ARTIFACTS]", "SUMMARY.txt", "P1A_ACCEPTANCE.json", "timing_metrics.csv",
        "oof_timing_predictions.csv", "fold_timing_results.csv", "timing_failure_cases.csv",
        "wordphase_contract_v0_1_0.json", "wordphasenet_fold0.pt ... wordphasenet_fold4.pt",
    ]
    (args.output_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest_out = []
    for p in sorted(args.output_dir.iterdir()):
        if p.is_file() and p.name != "run_file_manifest.json":
            manifest_out.append({"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    json_dump(args.output_dir / "run_file_manifest.json", manifest_out)
    print("\n".join(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
