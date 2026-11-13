#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"


class ProtocolError(RuntimeError):
    pass


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


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
        for key in row.keys():
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
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def inum(x: Any, default: int = -1) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def parse_serialized_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text in {"[]", "None", "null"}:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            out = parser(text)
            if isinstance(out, list):
                return out
        except Exception:
            pass
    return [text]


def required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ProtocolError(f"Missing {label}: {path}")
    return path


def stable_signature(row: Mapping[str, str], columns: Sequence[str]) -> str:
    text = "\x1f".join(str(row.get(c, "")).strip() for c in columns)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def copy_bound_file(src: Path, dst: Path) -> dict[str, Any]:
    shutil.copy2(src, dst)
    return {
        "name": dst.name,
        "source_path": str(src),
        "frozen_path": str(dst),
        "size_bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--p0a-proxy-dir", type=Path, required=True)
    ap.add_argument("--source-p0b-dir", type=Path, required=True)
    ap.add_argument("--p0b-r1-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--min-natural-donors-per-hand", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    for name in ("data_root", "project_root", "p0a_proxy_dir", "source_p0b_dir", "p0b_r1_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    annotation_path = required_file(args.data_root / "data" / "annotations" / "gesture_train_split.csv", "training annotations")
    p0a_closeout_path = required_file(args.p0a_proxy_dir / "P0A_CLOSEOUT.json", "P0A closeout")
    spatial_contract_path = required_file(args.p0a_proxy_dir / "spatial_proxy_contract_v1_0_0.json", "spatial proxy contract")
    p0b_split_path = required_file(args.source_p0b_dir / "split_integrity.json", "P0B split integrity")
    fold_path = required_file(args.source_p0b_dir / "fold_assignments.csv", "P0B fold assignments")
    hold_path = required_file(args.source_p0b_dir / "hold_region_manifest.csv", "P0B hold manifest")
    p0b_accept_path = required_file(args.source_p0b_dir / "P0B_ACCEPTANCE.json", "P0B acceptance")
    p0b_r1_accept_path = required_file(args.p0b_r1_dir / "P0B_R1_ACCEPTANCE.json", "P0B-R1 acceptance")
    temporal_contract_path = required_file(args.p0b_r1_dir / "temporal_proxy_contract_v1_1_0.json", "P0B-R1 temporal proxy contract")
    source_model_path = required_file(args.source_p0b_dir / "apex_detector_model_v1_0_0.npz", "source apex model")
    refined_model_path = required_file(args.p0b_r1_dir / "apex_detector_model_v1_1_0.npz", "refined apex model")

    p0a_closeout = json_load(p0a_closeout_path)
    spatial_contract = json_load(spatial_contract_path)
    split = json_load(p0b_split_path)
    p0b_accept = json_load(p0b_accept_path)
    p0b_r1_accept = json_load(p0b_r1_accept_path)
    temporal_contract = json_load(temporal_contract_path)

    gates = {
        "p0a_proxy_frozen": p0a_closeout.get("status") == "P0A_PROXY_FROZEN",
        "spatial_contract_frozen": spatial_contract.get("status") == "FROZEN_PAPER_FAITHFUL_PROXY",
        "split_pass": split.get("status") == "PASS",
        "no_group_leak": inum(split.get("group_leak_count"), 0) == 0,
        "no_recording_leak": inum(split.get("natural_recording_leak_count"), 0) == 0,
        "no_duplicate_leak": inum(split.get("duplicate_signature_leak_count"), 0) == 0,
        "p0b_r1_allows_p0c": bool(p0b_r1_accept.get("p0c_allowed")),
        "p0b_refinement_closed": p0b_r1_accept.get("further_p0b_refinement_allowed") is False,
        "temporal_proxy_advisory_or_frozen": temporal_contract.get("status") in {
            "ADVISORY_ENSEMBLE_ONLY", "SOURCE_AND_REFINED_AGREEMENT_ONLY",
            "PRIMARY_FROZEN", "PRIMARY_FROZEN_WITH_ENSEMBLE_DIAGNOSTIC",
        },
        "private_scorers_not_claimed": (
            spatial_contract.get("exact_private_scorer_recovered") is False
            and temporal_contract.get("exact_private_detector_recovered") is False
        ),
    }
    if not all(gates.values()):
        failed = [k for k, v in gates.items() if not v]
        raise ProtocolError(f"Upstream gate failure: {failed}")

    source_run = Path(str(p0a_closeout.get("source_run", ""))).expanduser().resolve()
    source_manifest_path = required_file(source_run / "source_resolution_manifest.csv", "P0A source-resolution manifest")

    annotations = read_csv(annotation_path)
    folds = read_csv(fold_path)
    holds = read_csv(hold_path)
    sources = read_csv(source_manifest_path)
    if not annotations:
        raise ProtocolError("Annotation table is empty")

    ann_by_id = {r.get("sample_id", ""): r for r in annotations}
    fold_by_id = {r.get("sample_id", ""): r for r in folds}
    hold_by_id = {r.get("sample_id", ""): r for r in holds}
    source_by_id = {r.get("sample_id", ""): r for r in sources}
    ids = [r.get("sample_id", "") for r in annotations]
    if len(set(ids)) != len(ids):
        raise ProtocolError("sample_id is not unique in annotation CSV")

    missing_join = {
        "fold": sorted(set(ids) - set(fold_by_id)),
        "hold": sorted(set(ids) - set(hold_by_id)),
        "source": sorted(set(ids) - set(source_by_id)),
    }
    if any(missing_join.values()):
        raise ProtocolError("Join coverage failure: " + ", ".join(f"{k}={len(v)}" for k, v in missing_join.items()))

    signature_cols = [c for c in annotations[0].keys() if c != "sample_id"]
    manifest: list[dict[str, Any]] = []
    missing_source_files: list[dict[str, Any]] = []
    fold_values: set[int] = set()
    for idx, ann in enumerate(annotations):
        sid = str(ann["sample_id"])
        fr = fold_by_id[sid]
        hr = hold_by_id[sid]
        sr = source_by_id[sid]
        fold = inum(fr.get("fold"))
        fold_values.add(fold)
        domain = str(fr.get("domain") or sr.get("domain") or hr.get("domain"))
        source_files = [str(x) for x in parse_serialized_list(sr.get("source_files"))]
        source_segments = parse_serialized_list(sr.get("clip_segments"))
        for p in source_files:
            if p and not Path(p).is_file():
                missing_source_files.append({"sample_id": sid, "source_file": p})
        clip_length = inum(ann.get("clip_length"))
        peak = inum(ann.get("peak_frame"))
        manifest.append({
            "row_index": idx,
            "sample_id": sid,
            "recording_id": ann.get("recording_id", ""),
            "domain": domain,
            "hand": inum(ann.get("hand")),
            "fold": fold,
            "leakage_group_id": fr.get("leakage_group_id", ""),
            "annotation_signature": stable_signature(ann, signature_cols),
            "clip_length": clip_length,
            "released_peak_frame": peak,
            "released_peak_ratio": peak / max(1, clip_length - 1),
            "hold_start": inum(hr.get("hold_start")),
            "hold_end": inum(hr.get("hold_end")),
            "hold_width": inum(hr.get("hold_width")),
            "gender": hr.get("gender", ""),
            "room_id": ann.get("room_id", ""),
            "object_id": ann.get("object_id", ""),
            "target_pelvis_x": ann.get("target_pelvis_x", ""),
            "target_pelvis_y": ann.get("target_pelvis_y", ""),
            "target_pelvis_z": ann.get("target_pelvis_z", ""),
            "target_world_x": ann.get("target_world_x", ""),
            "target_world_y": ann.get("target_world_y", ""),
            "target_world_z": ann.get("target_world_z", ""),
            "alignment_angle_deg": ann.get("alignment_angle_deg", ""),
            "alignment_score": ann.get("alignment_score", ""),
            "source_status": sr.get("status", ""),
            "source_files": json.dumps(source_files, ensure_ascii=False),
            "clip_segments": json.dumps(source_segments, ensure_ascii=False),
            "global_start": sr.get("global_start", ""),
            "global_end": sr.get("global_end", ""),
        })

    write_csv(args.output_dir / "missing_source_files.csv", missing_source_files)
    if missing_source_files:
        raise ProtocolError(f"Missing {len(missing_source_files)} source file bindings")
    if any(r["source_status"] != "OK" for r in manifest):
        raise ProtocolError("One or more source manifest rows are not OK")
    if sorted(fold_values) != list(range(len(fold_values))):
        raise ProtocolError(f"Unexpected fold IDs: {sorted(fold_values)}")

    write_csv(args.output_dir / "shared_generation_manifest.csv", manifest)

    # Donor policy: evaluation query in fold f may only retrieve donors from folds != f.
    # Natural same-hand motion templates are primary; synthetic is controller supervision only.
    donor_rows: list[dict[str, Any]] = []
    donor_starvation: list[dict[str, Any]] = []
    for fold in sorted(fold_values):
        val = [r for r in manifest if r["fold"] == fold]
        train = [r for r in manifest if r["fold"] != fold]
        for hand in (0, 1):
            q = [r for r in val if r["hand"] == hand]
            natural_donors = [r for r in train if r["hand"] == hand and r["domain"] == "natural"]
            synthetic_supervision = [r for r in train if r["hand"] == hand and r["domain"] == "synthetic"]
            donor_rows.append({
                "fold": fold,
                "hand": hand,
                "validation_queries": len(q),
                "training_all": len(train),
                "primary_natural_motion_donors": len(natural_donors),
                "synthetic_geometry_supervision": len(synthetic_supervision),
                "minimum_required_natural_motion_donors": args.min_natural_donors_per_hand,
                "donor_gate": "PASS" if len(natural_donors) >= args.min_natural_donors_per_hand else "FAIL",
            })
            if len(natural_donors) < args.min_natural_donors_per_hand:
                donor_starvation.append({"fold": fold, "hand": hand, "count": len(natural_donors)})
    write_csv(args.output_dir / "donor_pool_summary.csv", donor_rows)
    if donor_starvation:
        raise ProtocolError(f"Donor starvation: {donor_starvation}")

    domain_counts = Counter(r["domain"] for r in manifest)
    hand_counts = Counter(str(r["hand"]) for r in manifest)
    fold_counts = Counter(str(r["fold"]) for r in manifest)

    frozen_dir = args.output_dir / "frozen_inputs"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_bindings = [
        copy_bound_file(fold_path, frozen_dir / "fold_assignments.csv"),
        copy_bound_file(p0b_split_path, frozen_dir / "split_integrity.json"),
        copy_bound_file(hold_path, frozen_dir / "hold_region_manifest.csv"),
        copy_bound_file(spatial_contract_path, frozen_dir / "spatial_proxy_contract_v1_0_0.json"),
        copy_bound_file(temporal_contract_path, frozen_dir / "temporal_proxy_contract_v1_1_0.json"),
        copy_bound_file(source_model_path, frozen_dir / "apex_detector_model_v1_0_0.npz"),
        copy_bound_file(refined_model_path, frozen_dir / "apex_detector_model_v1_1_0.npz"),
    ]

    shared_split_contract = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_utc": now_utc(),
        "status": "FROZEN_GROUP_DISJOINT_5_FOLD",
        "sample_count": len(manifest),
        "fold_count": len(fold_values),
        "domain_counts": dict(domain_counts),
        "hand_counts": dict(hand_counts),
        "fold_counts": dict(fold_counts),
        "leakage_policy": {
            "natural": "all samples with the same recording_id remain in one fold",
            "duplicates": "all exact annotation-signature duplicates remain in one fold",
            "synthetic": "repeated synthetic recording_ids remain in one fold",
            "retrieval": "a validation query may use donors only from folds different from its own fold",
        },
        "split_integrity": split,
        "frozen_bindings": frozen_bindings,
    }
    json_dump(args.output_dir / "shared_split_contract_v1_0_0.json", shared_split_contract)

    output_contract = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "status": "FROZEN",
        "motion": {
            "poses": {"accepted_shapes": ["(T,165)", "(T,55,3)"], "meaning": "SMPL-X axis-angle joint rotations"},
            "trans": {"required_shape": "(T,3)", "meaning": "global translation"},
            "length": "T must exactly equal clip_length",
            "finite": "NaN and Inf are forbidden",
        },
        "evaluation_representations": {
            "oracle_peak": "released peak_frame, diagnostic only",
            "self_detected": "advisory P0B ensemble-selected frame",
            "official": "organizer server is the final source of truth",
        },
    }
    json_dump(args.output_dir / "common_motion_output_contract_v1_0_0.json", output_contract)

    b0_contract = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "status": "READY_TO_IMPLEMENT",
        "name": "B0 modular target-control baseline",
        "claim": "OmniControl-PT-style internal baseline; not an exact reproduction of unavailable official code/weights",
        "shared_inputs": ["speech timing", "active hand", "target coordinates", "body shape", "clip length"],
        "timing": "single deterministic speech-based timing heuristic; no learned phase model",
        "spatial_controller": "small MLP predicts elbow/wrist control anchors from target geometry and body shape",
        "motion_backend": "same frozen motion-template/retargeting backend used by P0C where possible, to isolate controller differences",
        "excluded": ["scene graph", "target-distractor relations", "multi-candidate learned reranking", "residual flow"],
        "candidate_budget": {"primary": 1, "diagnostic_equal_budget": 8},
    }
    json_dump(args.output_dir / "b0_internal_baseline_contract_v1_0_0.json", b0_contract)

    p0c_contract = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "status": "READY_TO_IMPLEMENT",
        "name": "P0C RePointer-v0 retrieval + shape-aware IK",
        "retrieval": {
            "primary_motion_donors": "natural, same hand, training folds only",
            "descriptor": ["target azimuth", "target elevation", "target distance", "clip length", "apex ratio", "hand"],
            "top_k": 8,
            "no_same_fold": True,
        },
        "time_warp": "piecewise linear warp preserving donor apex and query planned apex",
        "ik": "SMPL-X shape-aware local optimization of clavicle/shoulder/elbow/wrist around one clear apex/hold",
        "reranking": {
            "primary": ["frozen spatial proxy", "smoothness", "joint limits", "foot stability"],
            "advisory_temporal": ["source logreg_raw", "refined phase_peak_soft component", "reach-stability heuristic"],
            "guardrail": "temporal surrogate agreement is advisory, never represented as the private official detector",
        },
        "excluded_in_v0": ["scene graph", "residual flow", "listener state"],
    }
    json_dump(args.output_dir / "p0c_retrieval_ik_contract_v1_0_0.json", p0c_contract)

    comparison_contract = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "status": "FROZEN",
        "models": ["B0 modular target-control baseline", "P0C RePointer-v0"],
        "same_for_both": [
            "five group-disjoint folds", "source motion corpus", "SMPL-X body models",
            "output schema", "spatial proxy", "advisory temporal diagnostics",
            "evaluation code", "random seeds", "maximum training data",
        ],
        "report": {
            "primary_internal": [
                "self-detected Score1b proxy20", "self-detected STG proxy",
                "oracle-peak Score1b proxy20", "median and p90 angular error",
            ],
            "motion_quality": ["joint jerk", "foot sliding", "joint-limit violations", "pose deviation from donor"],
            "slices": ["fold", "natural/synthetic", "left/right", "target distance", "target azimuth/elevation"],
            "statistics": "paired per-sample differences with bootstrap confidence intervals; fold-level consistency",
        },
        "internal_p0c_success_gate": {
            "required": [
                "P0C mean self-detected Score1b > B0",
                "P0C mean self-detected STG > B0",
                "improvement direction positive in at least 4 of 5 folds",
                "no material degradation in motion-quality diagnostics",
            ],
            "note": "Internal scientific gate only; official server remains the final benchmark source of truth",
        },
    }
    json_dump(args.output_dir / "comparison_protocol_v1_0_0.json", comparison_contract)

    source_binding_paths = [
        annotation_path, source_manifest_path, p0a_closeout_path, spatial_contract_path,
        p0b_split_path, fold_path, hold_path, p0b_accept_path,
        p0b_r1_accept_path, temporal_contract_path, source_model_path, refined_model_path,
    ]
    source_binding = [{
        "name": p.name, "path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)
    } for p in source_binding_paths]
    json_dump(args.output_dir / "source_binding.json", source_binding)

    acceptance = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "created_utc": now_utc(),
        "status": "P0C0_SHARED_PROTOCOL_FROZEN",
        "gates": gates,
        "sample_count": len(manifest),
        "fold_count": len(fold_values),
        "donor_pool_gate": "PASS",
        "source_file_gate": "PASS",
        "join_gate": "PASS",
        "b0_allowed": True,
        "p0c_m1_allowed": True,
        "further_p0b_refinement_allowed": False,
        "next_action": "Implement and run B0 and P0C-M1 generation in parallel under the frozen comparison protocol.",
    }
    json_dump(args.output_dir / "P0C0_ACCEPTANCE.json", acceptance)

    env = {
        "created_utc": now_utc(), "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": sys.version,
        "python_executable": sys.executable, "cwd": os.getcwd(),
    }
    json_dump(args.output_dir / "environment.json", env)

    summary = [
        "REPOINTER P0C0 — SHARED B0/P0C GENERATION PROTOCOL FREEZE",
        "=" * 78,
        f"version={VERSION}", f"created_utc={now_utc()}",
        f"data_root={args.data_root}", f"output_dir={args.output_dir}",
        "dataset_writes=0", "downloads=0", "training=0", "motion_generation=0",
        "smplx_forward_kinematics=0", "git_writes=0", "source_artifact_mutation=0", "",
        "[UPSTREAM]",
        f"p0a_status={p0a_closeout.get('status')}",
        f"p0b_r1_status={p0b_r1_accept.get('status')}",
        f"p0b_r1_freeze_mode={p0b_r1_accept.get('freeze_mode')}",
        f"p0b_r1_no_more_refinement={p0b_r1_accept.get('further_p0b_refinement_allowed') is False}", "",
        "[CORPUS]",
        f"samples={len(manifest)}", f"natural={domain_counts.get('natural', 0)}",
        f"synthetic={domain_counts.get('synthetic', 0)}", f"folds={len(fold_values)}",
        f"left={hand_counts.get('0', 0)}", f"right={hand_counts.get('1', 0)}", "",
        "[GATES]",
        "upstream_gate=PASS", "join_gate=PASS", "source_file_gate=PASS",
        "split_gate=PASS", "donor_pool_gate=PASS", "",
        "[CLOSEOUT]",
        "status=P0C0_SHARED_PROTOCOL_FROZEN", "b0_allowed=TRUE", "p0c_m1_allowed=TRUE",
        "further_p0b_refinement_allowed=FALSE",
        "next_action=Run B0 and P0C-M1 generation in parallel under the frozen protocol.", "",
        "[PRIMARY_ARTIFACTS]",
        "P0C0_ACCEPTANCE.json", "shared_generation_manifest.csv", "donor_pool_summary.csv",
        "shared_split_contract_v1_0_0.json", "common_motion_output_contract_v1_0_0.json",
        "b0_internal_baseline_contract_v1_0_0.json", "p0c_retrieval_ik_contract_v1_0_0.json",
        "comparison_protocol_v1_0_0.json", "source_binding.json",
    ]
    (args.output_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    run_manifest = []
    for p in sorted(args.output_dir.rglob("*")):
        if p.is_file() and p.name != "run_file_manifest.json":
            run_manifest.append({
                "relative_path": str(p.relative_to(args.output_dir)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    json_dump(args.output_dir / "run_file_manifest.json", run_manifest)

    print("\n".join(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
