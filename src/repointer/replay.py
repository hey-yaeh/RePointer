"""Run immutable historical stages into a new private directory using explicit assets.

CPU replay is supported here. Timing retraining is separate from the pretrained
inference chain: new timing weights require new P1_FULL evidence before reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence


STAGES = {
    "p1a": ("frozen/timing/repointer_p1a_wordphase_timing_oof_v0_1_0.py", "P1A_ACCEPTANCE.json", "P1A_WORDPHASENET_ACCEPTED_P1_GENERATION_ALLOWED"),
    "p2a": ("frozen/candidates/repointer_p2a_phase_router_oof_v0_1_0.py", "P2A_ACCEPTANCE.json", "P2A_PHASE_ROUTER_ACCEPTED"),
    "p2c1": ("frozen/challenge/repointer_p2c1_official_test_materialization_v0_1_0.py", "P2C1_ACCEPTANCE.json", "P2C1_OFFICIAL_TEST_MATERIALIZATION_COMPLETE"),
    "p2c2": ("frozen/selector/repointer_p2c2_spatial_first_detector_robust_candidate_c_v0_1_0.py", "P2C2_ACCEPTANCE.json", "P2C2_CANDIDATE_C_MATERIALIZATION_COMPLETE"),
}


@dataclass(frozen=True)
class ReplayStage:
    name: str
    command: tuple[str, ...]
    output: Path
    acceptance: str
    expected_status: str


def plan_replay(paths: Mapping[str, Path | None], output_root: Path, research_python: str,
                repo_root: Path, stages: Sequence[str] = ("p2a", "p2c1", "p2c2")) -> list[ReplayStage]:
    """Validate all paths and frozen hashes before creating any output."""
    stages = tuple(stages)
    if not stages or any(stage not in STAGES for stage in stages) or len(set(stages)) != len(stages):
        raise ValueError("replay stages must be unique p1a/p2a/p2c1/p2c2 names")
    if "p1a" in stages and len(stages) != 1:
        raise ValueError("timing retraining is separate; regenerate P1_FULL before using its new weights downstream")
    order = list(STAGES)
    if list(stages) != sorted(stages, key=order.index):
        raise ValueError("replay stages must follow method order")
    output_root, repo_root = Path(output_root).resolve(), Path(repo_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("replay output root must be new")
    if output_root.is_relative_to(repo_root):
        raise ValueError("real motions and learned weights must stay outside the public source repository")
    for path in paths.values():
        if path is not None and (output_root.is_relative_to(path) or path.is_relative_to(output_root)):
            raise ValueError("replay outputs must be separate from every external input")
    executable = shutil.which(research_python)
    if executable is None:
        raise ValueError("research Python interpreter is unavailable")
    inventory = json.loads((repo_root / "docs/SOURCE_INVENTORY.json").read_text(encoding="utf-8"))
    state = dict(paths)

    def required(key: str) -> str:
        path = state.get(key)
        if path is None or (not path.is_dir() and path not in planned_outputs):
            raise ValueError(f"required asset is not configured or present: {key}")
        return str(path)

    plans = []
    planned_outputs: set[Path] = set()
    for name in stages:
        source, acceptance, status = STAGES[name]
        script = repo_root / source
        if source not in inventory or hashlib.sha256(script.read_bytes()).hexdigest() != inventory[source]["sha256"]:
            raise ValueError("frozen source hash mismatch; replay refused")
        output = output_root / name
        command = [executable, "-B", str(script), "--output-dir", str(output)]
        if name == "p2a":
            command += ["--p1-full-dir", required("p1_full_dir"), "--ridge-lambda", "0.5", "--seed", "20260811"]
        else:
            command += ["--data-root", required("dataset_root"), "--p0c0-dir", required("p0c0_dir"),
                        "--m1-full-dir", required("m1_full_dir"), "--device", "cpu"]
            if name == "p1a":
                command += ["--epochs", "60", "--batch-size", "128", "--bins", "64",
                            "--hash-buckets", "2048", "--max-words", "64", "--seed", "20260810"]
            else:
                command += ["--project-root", str(output_root), "--smplx-model-dir", required("smplx_model_dir"),
                            "--fk-batch-size", "32"]
                if name == "p2c1":
                    for key in ("p1a_dir", "p1_full_dir", "p2a_dir", "p2a_mh_dir", "p2c0_dir"):
                        command += ["--" + key.replace("_", "-"), required(key)]
                    command += ["--ik-batch-size", "64"]
                else:
                    command += ["--p2c1-dir", required("p2c1_dir")]
        plans.append(ReplayStage(name, tuple(command), output, acceptance, status))
        planned_outputs.add(output)
        state[{"p1a": "p1a_dir", "p2a": "p2a_dir", "p2c1": "p2c1_dir", "p2c2": "p2c2_dir"}[name]] = output
    return plans


def execute_replay(plans: Sequence[ReplayStage], output_root: Path, threads: int = 4) -> dict:
    """Execute a reviewed plan; raw logs, checkpoints and commands remain private."""
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not plans:
        raise ValueError("no replay stages requested")
    output_root = Path(output_root).resolve()
    if len({stage.name for stage in plans}) != len(plans):
        raise ValueError("replay stages must be unique")
    for stage in plans:
        if stage.name not in STAGES or stage.output.resolve() != output_root / stage.name:
            raise ValueError("stage outputs must remain inside the fresh replay root")
        if stage.output.exists():
            raise FileExistsError("stage output already exists")
        index = stage.command.index("--output-dir")
        if Path(stage.command[index + 1]).resolve() != stage.output.resolve():
            raise ValueError("stage command output differs from its planned private directory")
    output_root.mkdir(parents=True, exist_ok=False)
    private_plan = [{"stage": item.name, "command": list(item.command)} for item in plans]
    (output_root / "commands.private.json").write_text(json.dumps(private_plan, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[key] = str(threads)
    results = []
    for stage in plans:
        stage.output.mkdir(exist_ok=False)
        with (output_root / f"{stage.name}.log").open("x", encoding="utf-8") as log:
            process = subprocess.run(stage.command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        acceptance = stage.output / stage.acceptance
        observed = None
        if acceptance.is_file():
            observed = json.loads(acceptance.read_text(encoding="utf-8")).get("status")
        passed = process.returncode == 0 and observed == stage.expected_status
        results.append({"stage": stage.name, "returncode": process.returncode, "accepted": passed})
        if not passed:
            break
    report = {"schema": "repointer.historical-replay.v1", "status": "PASS" if len(results) == len(plans) and all(r["accepted"] for r in results) else "FAIL",
              "device": "cpu", "stages": results, "synthetic": False,
              "official_server_scores_reproduced": False}
    (output_root / "replay.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
