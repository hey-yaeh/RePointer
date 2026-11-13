"""Invented five-fold archives test read-only audit and publication boundaries."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from repointer.artifacts import AGGREGATOR, WORKER, audit_m1_run, load_m1_reference
from repointer.assets import asset_status, load_asset_paths
from repointer.cli import main


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def manifest(root):
    return [{"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != "run_file_manifest.json"]


def refresh_manifests(root):
    for fold in range(5):
        folder = root / "folds" / f"fold_{fold}"
        write_json(folder / "run_file_manifest.json", manifest(folder))
    write_json(root / "run_file_manifest.json", manifest(root))


@pytest.fixture
def archive(tmp_path):
    """Small fabricated audit inputs only, never historical benchmark evidence."""
    root = tmp_path / "invented-run"
    samples = [dict(sample_id=f"invented_f{fold}_h{hand}_{j}", fold=str(fold), hand=str(hand),
                    domain="natural", clip_length="7", leakage_group_id=f"g{fold}_{hand}_{j}",
                    recording_id=f"r{fold}_{hand}_{j}", annotation_signature=f"a{fold}_{hand}_{j}")
               for fold in range(5) for hand in range(2) for j in range(2)]
    combined = {"candidate": [], "retrieval": [], "timing": []}
    for fold in range(5):
        folder = root / "folds" / f"fold_{fold}"
        selected = [row for row in samples if row["fold"] == str(fold)]
        write_csv(folder / "per_sample_comparison.csv", selected)
        candidates, retrieval = [], []
        for query in selected:
            donors = [row for row in samples if row["fold"] != str(fold) and row["hand"] == query["hand"]]
            for method, budget in (("b0", 1), ("b0eq", 8), ("p0c", 8)):
                for rank, donor in enumerate(donors[:budget], 1):
                    record = dict(query_id=query["sample_id"], donor_id=donor["sample_id"], method=method, rank=str(rank))
                    candidates.append(record)
                    retrieval.append(record.copy())
            for method in ("b0", "b0_equal_budget", "p0c"):
                path = folder / "motions" / method / f"{query['sample_id']}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, poses=np.zeros((7, 165), np.float32), trans=np.zeros((7, 3), np.float32))
        timing = [{**row, "role": "validation_generation" if row["fold"] == str(fold) else "training"} for row in samples]
        for kind, rows, filename in (("candidate", candidates, "candidate_metrics.csv"),
                                     ("retrieval", retrieval, "retrieval_manifest.csv"),
                                     ("timing", timing, "timing_manifest.csv")):
            write_csv(folder / filename, rows)
            combined[kind].extend(rows)
        write_csv(folder / "schema_errors.csv", [], ["error"])
        write_json(folder / "B0_P0C_M1_ACCEPTANCE.json", dict(status="B0_P0C_M1_FULL_FOLD_COMPLETED",
                   engineering_gate=True, validation_query_count=4, candidate_count=68))
        write_json(folder / "validation_gt_usage_audit.json", dict(status="PASS", all_generation_views_exclude_prohibited_keys=True))
        write_json(folder / "FOLD_COMPLETE.json", dict(status="FOLD_COMPLETE"))
        (folder / "python.log").write_text("begin\n")
    for kind, filename in (("candidate", "candidate_metrics_all_folds.csv"),
                           ("retrieval", "retrieval_manifest_all_folds.csv"),
                           ("timing", "timing_manifest_all_folds.csv")):
        write_csv(root / "aggregate" / filename, combined[kind])
    write_csv(root / "aggregate/per_sample_comparison_all_folds.csv", samples)
    write_csv(root / "aggregate/fold_coverage.csv", [dict(fold=fold, status="PASS") for fold in range(5)])
    for name in ("engineering_errors.csv", "missing_motion_files.csv"):
        write_csv(root / "aggregate" / name, [], ["error"])
    write_json(root / "FULL_5FOLD_ACCEPTANCE.json", dict(status="B0_P0C_M1_FULL_5FOLD_ACCEPTED", fold_count=5,
               sample_count=20, candidate_count=340, top_k=8, engineering_gate=True,
               fold_consistency_gate=True, quality_gate=True, p1_allowed=True))
    write_json(root / "coverage_audit.json", dict(status="PASS", actual_unique_sample_count=20))
    write_json(root / "method_parity_audit.json", dict(status="PASS"))
    for name in (WORKER, AGGREGATOR):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("raise RuntimeError('auditor must never execute archived code')\n")
    (root / "aggregate.log").write_text("")
    refresh_manifests(root)
    reference = dict(schema="repointer.m1-reference.v1", acceptance_status="B0_P0C_M1_FULL_5FOLD_ACCEPTED",
                     fold_count=5, sample_count=20, candidate_count=340, selected_motion_count=60,
                     fold_sample_counts=[4] * 5, sha256={"run_file_manifest.json": hashlib.sha256((root / "run_file_manifest.json").read_bytes()).hexdigest()})
    reference_path = tmp_path / "reference.json"
    write_json(reference_path, reference)
    return root, reference_path


def test_full_audit_preserves_archive_and_omits_private_context(archive):
    root, reference = archive
    before = manifest(root)
    report = audit_m1_run(root, reference=load_m1_reference(reference), verify_hashes=True, verify_motions=True)
    assert report["status"] == "PASS" and report["reference_bound"]
    assert report["counts"]["selected_motions"] == report["counts"]["motion_schemas_checked"] == 60
    assert report["counts"]["candidate_rows"] == 340
    assert not report["generation_rerun"] and not report["full_reproduction_validated"]
    assert str(root) not in json.dumps(report) and "invented_f" not in json.dumps(report)
    assert before == manifest(root)


def test_appended_logs_are_distinguished_from_corruption(archive):
    root, _ = archive
    for path in [root / "aggregate.log", *root.glob("folds/fold_*/python.log")]:
        with path.open("a") as stream:
            stream.write("done\n")
    report = audit_m1_run(root)
    assert report["status"] == "PASS" and report["warnings"]["append_only_logs"] == 6
    path = root / "folds/fold_0/python.log"
    path.write_text("altered prefix\n")
    assert audit_m1_run(root)["errors"]["log_prefix_changed"] == 1


def test_full_hash_flag_detects_same_size_motion_change(archive):
    root, _ = archive
    path = next(root.glob("folds/fold_*/motions/*/*.npz"))
    data = bytearray(path.read_bytes())
    data[100] ^= 1
    path.write_bytes(data)
    assert audit_m1_run(root)["status"] == "PASS"  # motion hashes explicitly unchecked
    report = audit_m1_run(root, verify_hashes=True)
    assert report["status"] == "FAIL" and report["errors"]["manifest_hash_mismatch"] == 1


def test_motion_schema_checks_shapes_beyond_manifest_hashes(archive):
    root, _ = archive
    path = next(root.glob("folds/fold_*/motions/*/*.npz"))
    np.savez(path, poses=np.zeros((7, 164), np.float32), trans=np.zeros((7, 3), np.float32))
    refresh_manifests(root)
    assert audit_m1_run(root, verify_hashes=True)["status"] == "PASS"
    assert audit_m1_run(root, verify_motions=True)["errors"]["motion_schema"] == 1


def test_array_disguised_as_npz_reports_schema_failure(archive):
    root, _ = archive
    payload = io.BytesIO()
    np.save(payload, np.zeros((7, 165), np.float32))
    path = next(root.glob("folds/fold_*/motions/*/*.npz"))
    path.write_bytes(payload.getvalue())
    refresh_manifests(root)
    assert audit_m1_run(root, verify_motions=True)["errors"]["motion_schema"] == 1


@pytest.mark.parametrize("changed", ["donor", "aggregate", "missing_motion", "acceptance"])
def test_failures_are_reported_without_sample_ids(archive, changed):
    root, _ = archive
    if changed == "donor":
        path = root / "folds/fold_0/retrieval_manifest.csv"
        rows = read_csv(path)
        rows[0]["donor_id"] = rows[0]["query_id"]
        write_csv(path, rows)
    elif changed == "aggregate":
        path = root / "aggregate/candidate_metrics_all_folds.csv"
        rows = read_csv(path)
        write_csv(path, rows[:-1])
    elif changed == "missing_motion":
        next(root.glob("folds/fold_*/motions/*/*.npz")).unlink()
    else:
        path = root / "FULL_5FOLD_ACCEPTANCE.json"
        data = json.loads(path.read_text())
        data["engineering_gate"] = False
        write_json(path, data)
    report = audit_m1_run(root)
    assert report["status"] == "FAIL" and report["errors"]
    assert str(root) not in json.dumps(report) and "invented_f" not in json.dumps(report)


def test_reference_detects_replaced_history(archive):
    root, reference = archive
    (root / "aggregate.log").write_text("changed after manifest refresh\n")
    refresh_manifests(root)
    assert audit_m1_run(root, reference=load_m1_reference(reference))["errors"]["reference_hash"] == 1


def test_manifest_cannot_read_outside_run(archive, tmp_path):
    root, _ = archive
    secret = tmp_path / "outside.txt"
    secret.write_text("not an artifact")
    path = root / "run_file_manifest.json"
    records = json.loads(path.read_text())
    records.append(dict(relative_path="../outside.txt", size_bytes=15, sha256="0" * 64))
    write_json(path, records)
    report = audit_m1_run(root, verify_hashes=True)
    assert report["errors"]["invalid_manifest_record"] == 1
    assert secret.read_text() == "not an artifact"


def test_audit_cli_writes_exclusive_report_outside_assets(archive, tmp_path, capsys):
    root, reference = archive
    report = tmp_path / "report.json"
    args = ["audit-m1", "--run-dir", str(root), "--reference", str(reference), "--report", str(report)]
    assert main(args) == 0
    assert json.loads(report.read_text())["status"] == "PASS"
    assert main(args) == 1
    assert main(["audit-m1", "--run-dir", str(root), "--report", str(root / "forbidden.json")]) == 1
    assert not (root / "forbidden.json").exists()


def test_report_refuses_dangling_symlink(archive, tmp_path):
    root, _ = archive
    target = tmp_path / "unrelated.json"
    report = tmp_path / "report-link.json"
    report.symlink_to(target)
    assert main(["audit-m1", "--run-dir", str(root), "--report", str(report)]) == 1
    assert not target.exists()


def test_manifest_symlink_cannot_escape_run(archive, tmp_path):
    root, _ = archive
    outside = tmp_path / "external-source.py"
    outside.write_text("not an archive member")
    worker = root / WORKER
    worker.unlink()
    worker.symlink_to(outside)
    report = audit_m1_run(root, verify_hashes=True)
    assert report["status"] == "FAIL" and report["errors"]["unsafe_artifact_path"] == 1
    assert str(outside) not in json.dumps(report)


def test_external_config_expansion_and_status(archive, tmp_path, monkeypatch):
    root, reference = archive
    monkeypatch.setenv("REPOINTER_TEST_M1", str(root))
    config = tmp_path / "assets.json"
    write_json(config, dict(schema="repointer.external-assets.v1", paths={"m1_full_dir": "${REPOINTER_TEST_M1}"}))
    paths = load_asset_paths(config)
    assert paths["m1_full_dir"] == root
    status = asset_status(paths)
    assert status["assets"]["m1_full_dir"]["status"] == "directory_present"
    assert status["assets"]["p1_full_dir"]["status"] == "unconfigured"
    assert str(root) not in json.dumps(status)
    assert main(["assets-status", "--config", str(config)]) == 0
    assert main(["audit-m1", "--config", str(config), "--reference", str(reference)]) == 0
    monkeypatch.delenv("REPOINTER_TEST_M1")
    with pytest.raises(ValueError, match="environment variable"):
        load_asset_paths(config)


def test_external_paths_resolve_relative_to_config_and_stay_outside_repo(tmp_path):
    config = tmp_path / "assets.json"
    write_json(config, dict(schema="repointer.external-assets.v1", paths={"m1_full_dir": "external"}))
    assert load_asset_paths(config)["m1_full_dir"] == tmp_path / "external"
    repo = Path(__file__).resolve().parents[1]
    write_json(config, dict(schema="repointer.external-assets.v1", paths={"m1_full_dir": str(repo / "private")}))
    with pytest.raises(ValueError, match="outside"):
        load_asset_paths(config)


def test_installed_package_does_not_treat_its_ancestor_as_public_source(tmp_path, monkeypatch):
    import repointer.assets as assets
    monkeypatch.setattr(assets, "__file__", str(tmp_path / "site-packages/repointer/assets.py"))
    config = tmp_path / "assets.json"
    write_json(config, dict(schema="repointer.external-assets.v1", paths={"m1_full_dir": "external"}))
    assert assets.load_asset_paths(config)["m1_full_dir"] == tmp_path / "external"


@pytest.mark.parametrize("document", [{}, {"schema": "wrong", "paths": {}},
    {"schema": "repointer.external-assets.v1", "paths": {"typo": None}},
    {"schema": "repointer.external-assets.v1", "paths": {"m1_full_dir": ""}}])
def test_bad_asset_configs_rejected(tmp_path, document):
    config = tmp_path / "bad.json"
    write_json(config, document)
    with pytest.raises(ValueError):
        load_asset_paths(config)
