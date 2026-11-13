"""Read-only M1 artifact audits. Reports contain no sample IDs or local paths."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
import zlib

import numpy as np

WORKER = "_code/repointer_b0_p0c_m1_full_fold_v0_2_0.py"
AGGREGATOR = "_code/repointer_b0_p0c_m1_full_5fold_aggregate_v0_2_0.py"
METHOD_DIRS = {"b0": "b0", "b0eq": "b0_equal_budget", "p0c": "p0c"}
FOLD_FILES = (
    "B0_P0C_M1_ACCEPTANCE.json", "validation_gt_usage_audit.json", "FOLD_COMPLETE.json",
    "per_sample_comparison.csv", "candidate_metrics.csv", "retrieval_manifest.csv",
    "timing_manifest.csv", "schema_errors.csv", "run_file_manifest.json",
)
AGGREGATE_FILES = (
    "per_sample_comparison_all_folds.csv", "candidate_metrics_all_folds.csv",
    "retrieval_manifest_all_folds.csv", "timing_manifest_all_folds.csv",
    "fold_coverage.csv", "engineering_errors.csv", "missing_motion_files.csv",
)


def _member(root: Path, name: str) -> Path:
    """Manifest members cannot escape their run, including through symlinks."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid artifact member")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in (".", "..") for part in name.split("/")) or ":" in name:
        raise ValueError("unsafe artifact member")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact member escapes its run")
    return path


def load_m1_reference(path: str | Path) -> dict:
    reference = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(reference, dict) or reference.get("schema") != "repointer.m1-reference.v1":
        raise ValueError("unsupported M1 reference schema")
    for name in ("sample_count", "candidate_count", "selected_motion_count"):
        if type(reference.get(name)) is not int or reference[name] < 1:
            raise ValueError("invalid M1 reference counts")
    if reference.get("fold_count") != 5 or reference.get("acceptance_status") != "B0_P0C_M1_FULL_5FOLD_ACCEPTED":
        raise ValueError("invalid M1 reference protocol")
    hashes = reference.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("M1 reference must bind evidence hashes")
    for name, digest in hashes.items():
        _member(Path(path).resolve().parent, name)
        if not isinstance(digest, str) or re.fullmatch("[a-f0-9]{64}", digest) is None:
            raise ValueError("invalid M1 reference hash")
    return reference


class _Audit:
    def __init__(self, root: Path):
        self.root = root
        self.errors: Counter = Counter()
        self.warnings: Counter = Counter()

    def json(self, name: str):
        try:
            return json.loads(_member(self.root, name).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            self.errors["unreadable_json"] += 1
            return None

    def csv(self, name: str) -> list[dict]:
        try:
            with _member(self.root, name).open(encoding="utf-8-sig", newline="") as stream:
                return list(csv.DictReader(stream))
        except (OSError, ValueError, TypeError, csv.Error):
            self.errors["unreadable_csv"] += 1
            return []

    def require(self, condition: bool, category: str) -> None:
        if not condition:
            self.errors[category] += 1

    def manifest(self, name: str, prefix: str = "") -> dict[str, dict]:
        data = self.json(name)
        if not isinstance(data, list):
            self.errors["invalid_manifest"] += 1
            return {}
        records = {}
        for record in data:
            try:
                relative = prefix + record["relative_path"]
                _member(self.root, relative)
                size, digest = record["size_bytes"], record["sha256"]
                if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
                    raise ValueError("invalid manifest record")
                if relative in records:
                    raise ValueError("duplicate manifest record")
                records[relative] = {"size_bytes": size, "sha256": digest}
            except (KeyError, TypeError, ValueError):
                self.errors["invalid_manifest_record"] += 1
        return records


def _motion_schema(payload: bytes, frames: int) -> bool:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            if sorted(data.files) != ["poses", "trans"]:
                return False
            poses, trans = data["poses"], data["trans"]
            return bool(poses.shape in ((frames, 165), (frames, 55, 3)) and trans.shape == (frames, 3)
                        and poses.dtype == np.float32 and trans.dtype == np.float32
                        and np.isfinite(poses).all() and np.isfinite(trans).all())
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, zlib.error):
        return False


def audit_m1_run(run_dir: str | Path, *, reference: dict | None = None,
                 verify_hashes: bool = False, verify_motions: bool = False) -> dict:
    """Audit an external run without importing its code or loading model weights.

    By default all manifest members are checked for presence/size, non-motion
    members are hashed, and coverage/donor tables are checked. Full motion hashes
    and strict NPZ schema checks are opt-in because historical runs can be large.
    Append-only .log growth is reported separately from scientific-file failures.
    A PASS is artifact evidence, never an official evaluation or generation rerun.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError("M1 run directory does not exist")
    audit = _Audit(root)
    required = ["FULL_5FOLD_ACCEPTANCE.json", "coverage_audit.json", "method_parity_audit.json",
                "run_file_manifest.json", WORKER, AGGREGATOR]
    required += [f"aggregate/{name}" for name in AGGREGATE_FILES]
    required += [f"folds/fold_{fold}/{name}" for fold in range(5) for name in FOLD_FILES]
    for name in required:
        try:
            audit.require(_member(root, name).is_file(), "missing_required_file")
        except ValueError:
            audit.errors["unsafe_artifact_path"] += 1

    acceptance = audit.json("FULL_5FOLD_ACCEPTANCE.json") or {}
    coverage = audit.json("coverage_audit.json") or {}
    parity = audit.json("method_parity_audit.json") or {}
    if not all(isinstance(value, dict) for value in (acceptance, coverage, parity)):
        raise ValueError("M1 acceptance/audit records must be JSON objects")
    audit.require(acceptance.get("status") == "B0_P0C_M1_FULL_5FOLD_ACCEPTED", "acceptance_status")
    for gate in ("engineering_gate", "fold_consistency_gate", "quality_gate", "p1_allowed"):
        audit.require(acceptance.get(gate) is True, "acceptance_gate")
    audit.require(acceptance.get("fold_count") == 5, "fold_count")
    audit.require(coverage.get("status") == "PASS" and parity.get("status") == "PASS", "historical_audit_status")
    top_k = acceptance.get("top_k")
    if type(top_k) is not int or top_k < 1:
        audit.errors["invalid_candidate_budget"] += 1
        top_k = 8
    selected = audit.csv("aggregate/per_sample_comparison_all_folds.csv")
    ids = [row.get("sample_id", "") for row in selected]
    audit.require(bool(ids) and all(ids) and len(ids) == len(set(ids)), "sample_id_coverage")
    by_id = {row.get("sample_id"): row for row in selected}
    audit.require(len(selected) == acceptance.get("sample_count") == coverage.get("actual_unique_sample_count"), "sample_count")
    for field in ("leakage_group_id", "recording_id", "annotation_signature"):
        groups: dict[str, set] = defaultdict(set)
        for row in selected:
            if row.get(field):
                groups[row[field]].add(row.get("fold"))
        audit.require(all(len(folds) == 1 for folds in groups.values()), "cross_fold_identity_leakage")

    counts = {"samples": len(selected), "folds": 5, "candidate_rows": 0,
              "selected_motions": 0, "motion_schemas_checked": 0,
              "manifest_files": 0, "files_hashed": 0}
    expected_motions: dict[str, int] = {}
    fold_counts = []
    combined = {"candidate": [], "retrieval": [], "timing": []}
    fold_manifests = []
    for fold in range(5):
        prefix = f"folds/fold_{fold}/"
        fold_acceptance = audit.json(prefix + "B0_P0C_M1_ACCEPTANCE.json") or {}
        gt_audit = audit.json(prefix + "validation_gt_usage_audit.json") or {}
        completion = audit.json(prefix + "FOLD_COMPLETE.json") or {}
        if not all(isinstance(value, dict) for value in (fold_acceptance, gt_audit, completion)):
            audit.errors["invalid_fold_audit"] += 1
            continue
        audit.require(fold_acceptance.get("status") == "B0_P0C_M1_FULL_FOLD_COMPLETED" and fold_acceptance.get("engineering_gate") is True, "fold_acceptance")
        audit.require(gt_audit.get("status") == "PASS" and gt_audit.get("all_generation_views_exclude_prohibited_keys") is True, "gt_usage_audit")
        audit.require(completion.get("status") == "FOLD_COMPLETE", "fold_completion")
        rows = audit.csv(prefix + "per_sample_comparison.csv")
        fold_counts.append(len(rows))
        expected_ids = {sid for sid, row in by_id.items() if row.get("fold") == str(fold)}
        actual_ids = [row.get("sample_id") for row in rows]
        audit.require(set(actual_ids) == expected_ids and len(actual_ids) == len(expected_ids), "fold_coverage")
        audit.require(fold_acceptance.get("validation_query_count") == len(rows), "fold_sample_count")
        for row in rows:
            sid = row.get("sample_id", "")
            try:
                frames = int(row["clip_length"])
                if frames < 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sid):
                    raise ValueError("invalid sample metadata")
                for motion_dir in METHOD_DIRS.values():
                    expected_motions[f"{prefix}motions/{motion_dir}/{sid}.npz"] = frames
            except (ValueError, KeyError, TypeError):
                audit.errors["invalid_motion_metadata"] += 1
        candidates = audit.csv(prefix + "candidate_metrics.csv")
        retrieval = audit.csv(prefix + "retrieval_manifest.csv")
        timing = audit.csv(prefix + "timing_manifest.csv")
        expected_candidates = len(rows) * (1 + 2 * top_k)
        audit.require(len(candidates) == len(retrieval) == expected_candidates == fold_acceptance.get("candidate_count"), "candidate_count")
        counts["candidate_rows"] += len(candidates)
        for row in retrieval:
            query, donor = by_id.get(row.get("query_id")), by_id.get(row.get("donor_id"))
            if query is None or donor is None:
                audit.errors["unknown_query_or_donor"] += 1
                continue
            audit.require(query.get("fold") == str(fold) and query.get("fold") != donor.get("fold"), "donor_fold_leakage")
            audit.require(query.get("hand") == donor.get("hand") and donor.get("domain") == "natural", "donor_hand_or_domain")
            for field in ("leakage_group_id", "recording_id"):
                audit.require(not query.get(field) or query[field] != donor.get(field), "donor_identity_leakage")
        validation = [row for row in timing if row.get("role") in ("validation", "validation_generation")]
        training = [row for row in timing if row.get("role") == "training"]
        audit.require(len(validation) == len(expected_ids) and {r.get("sample_id") for r in validation} == expected_ids, "validation_timing_coverage")
        audit.require(all(row.get("fold") != str(fold) for row in training), "training_timing_leakage")
        audit.require(not audit.csv(prefix + "schema_errors.csv"), "historical_schema_errors")
        combined["candidate"].extend(candidates)
        combined["retrieval"].extend(retrieval)
        combined["timing"].extend(timing)
        fold_manifests.append(audit.manifest(prefix + "run_file_manifest.json", prefix))

    for kind, filename in (("candidate", "candidate_metrics_all_folds.csv"),
                           ("retrieval", "retrieval_manifest_all_folds.csv"),
                           ("timing", "timing_manifest_all_folds.csv")):
        aggregate = audit.csv("aggregate/" + filename)
        signature = lambda row: json.dumps(row, sort_keys=True)
        audit.require(Counter(map(signature, aggregate)) == Counter(map(signature, combined[kind])), "aggregate_table_mismatch")
    audit.require(counts["candidate_rows"] == acceptance.get("candidate_count"), "aggregate_candidate_count")
    fold_coverage = audit.csv("aggregate/fold_coverage.csv")
    audit.require(len(fold_coverage) == 5 and all(row.get("status") == "PASS" for row in fold_coverage), "fold_coverage_audit")
    for filename in ("engineering_errors.csv", "missing_motion_files.csv"):
        audit.require(not audit.csv("aggregate/" + filename), "historical_engineering_errors")
    audit.require(len(expected_motions) == len(selected) * 3, "motion_coverage")
    actual_motions = {path.relative_to(root).as_posix() for path in (root / "folds").glob("fold_*/motions/*/*.npz")}
    audit.require(actual_motions == set(expected_motions), "motion_file_set")
    counts["selected_motions"] = len(actual_motions)

    manifest = audit.manifest("run_file_manifest.json")
    counts["manifest_files"] = len(manifest)
    audit.require(set(expected_motions).issubset(manifest), "unlisted_selected_motion")
    for name, record in manifest.items():
        try:
            path = _member(root, name)
            is_motion = "/motions/" in name and name.endswith(".npz")
            needs_bytes = not is_motion or verify_hashes or verify_motions
            payload = path.read_bytes() if needs_bytes else None
            size = len(payload) if payload is not None else path.stat().st_size
            digest = hashlib.sha256(payload).hexdigest() if payload is not None else None
            if name.endswith(".log") and size > record["size_bytes"] and payload is not None:
                prefix_hash = hashlib.sha256(payload[:record["size_bytes"]]).hexdigest()
                if prefix_hash == record["sha256"]:
                    audit.warnings["append_only_logs"] += 1
                else:
                    audit.errors["log_prefix_changed"] += 1
            else:
                audit.require(size == record["size_bytes"], "manifest_size_mismatch")
                if not is_motion or verify_hashes:
                    audit.require(digest == record["sha256"], "manifest_hash_mismatch")
            if not is_motion or verify_hashes:
                counts["files_hashed"] += 1
            if verify_motions and name in expected_motions:
                counts["motion_schemas_checked"] += 1
                audit.require(_motion_schema(payload, expected_motions[name]), "motion_schema")
        except (OSError, ValueError, TypeError):
            audit.errors["unreadable_manifest_member"] += 1
    for records in fold_manifests:
        for name, record in records.items():
            canonical = manifest.get(name)
            if canonical == record:
                continue
            if name.endswith(".log"):
                try:
                    payload = _member(root, name).read_bytes()
                    if len(payload) >= record["size_bytes"] and hashlib.sha256(payload[:record["size_bytes"]]).hexdigest() == record["sha256"]:
                        audit.warnings["append_only_fold_logs"] += 1
                        continue
                except (OSError, ValueError):
                    pass
            audit.errors["fold_manifest_mismatch"] += 1

    if reference is not None:
        audit.require(counts["samples"] == reference["sample_count"]
                      and counts["candidate_rows"] == reference["candidate_count"]
                      and counts["selected_motions"] == reference["selected_motion_count"], "reference_counts")
        audit.require(fold_counts == reference.get("fold_sample_counts"), "reference_fold_counts")
        for name, expected in reference["sha256"].items():
            try:
                audit.require(hashlib.sha256(_member(root, name).read_bytes()).hexdigest() == expected, "reference_hash")
            except (OSError, ValueError):
                audit.errors["unreadable_reference_member"] += 1
    return {
        "schema": "repointer.m1-audit.v1", "status": "PASS" if not audit.errors else "FAIL",
        "reference_bound": reference is not None,
        "acceptance_status": acceptance.get("status") if acceptance.get("status") == "B0_P0C_M1_FULL_5FOLD_ACCEPTED" else "not_accepted",
        "counts": counts, "fold_sample_counts": fold_counts,
        "hash_scope": "all_manifest_files" if verify_hashes else "non_motion_manifest_files",
        "motion_schema_scope": "all_selected_motions" if verify_motions else "not_checked",
        "errors": dict(sorted(audit.errors.items())), "warnings": dict(sorted(audit.warnings.items())),
        "generation_rerun": False, "official_evaluation_run": False,
        "full_reproduction_validated": False,
    }
