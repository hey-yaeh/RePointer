"""Local source audit, synthetic experiments, and output-contract CLI."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from .contracts import Motion

EXPECTED = [
    'frozen/p0c0/repointer_p0c0_shared_generation_protocol_freeze_v1_0_0.py',
    'frozen/m1/repointer_b0_p0c_m1_full_fold_v0_2_0.py',
    'frozen/m1/repointer_b0_p0c_m1_full_5fold_aggregate_v0_2_0.py',
    'frozen/timing/repointer_p1a_wordphase_timing_oof_v0_1_0.py',
    'frozen/candidates/repointer_p2a_phase_router_oof_v0_1_0.py',
    'frozen/candidates/repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py',
    'frozen/challenge/repointer_p2c1_official_test_materialization_v0_1_0.py',
    'frozen/selector/repointer_p2c2_spatial_first_detector_robust_candidate_c_v0_1_0.py',
]


def audit(repo_root: Path) -> int:
    """Check exact archived bytes and syntax without executing frozen code."""
    problems = []
    try:
        inventory = json.loads((repo_root/'docs/SOURCE_INVENTORY.json').read_text(encoding='utf-8'))
    except (OSError,ValueError) as exc:
        print(f'FAIL: cannot read source inventory: {exc}')
        return 1
    if set(inventory) != set(EXPECTED):
        problems.append('inventory does not cover exactly the eight expected frozen sources')
    actual = {str(p.relative_to(repo_root)) for p in (repo_root/'frozen').rglob('*') if p.is_file()}
    if actual != set(EXPECTED): problems.append('frozen file set differs from inventory (including generated files)')
    for rel in EXPECTED:
        path = repo_root/rel
        try:
            content = path.read_bytes()
            ast.parse(content.decode('utf-8'),filename=rel)
            expected = inventory.get(rel,{})
            if hashlib.sha256(content).hexdigest()!=expected.get('sha256') or len(content)!=expected.get('bytes'):
                problems.append(f'HASH/SIZE MISMATCH: {rel}')
        except (OSError,SyntaxError,UnicodeError) as exc:
            problems.append(f'SOURCE ERROR: {rel}: {exc}')
    if problems:
        print('\n'.join(problems))
        return 1
    print(f'PASS: {len(EXPECTED)} frozen files match archived SHA-256, sizes and Python syntax')
    print('This verifies source integrity, not inference or challenge results.')
    return 0


def validate_motion(path: Path, expected_frames: int | None = None) -> int:
    try:
        motion = Motion.load(path,expected_frames)
        print(f'PASS: motion schema valid for T={motion.frames} frames')
        print('Array schema only; body-model accuracy and grounding are unverified.')
        return 0
    except (ValueError,KeyError,OSError,TypeError,EOFError) as exc:
        print(f'FAIL: {exc}')
        return 1


def _report_destination(path: Path | None, protected: list[Path]) -> Path | None:
    if path is None:
        return None
    if path.exists() or path.is_symlink():
        raise FileExistsError('audit report already exists; choose a fresh output path')
    destination = path.resolve()
    frozen = Path(__file__).resolve().parents[2] / 'frozen'
    if destination.is_relative_to(frozen) or any(destination.is_relative_to(root) for root in protected):
        raise ValueError('audit reports must stay outside frozen sources and external asset directories')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('audit report already exists; choose a fresh output path')
    return destination


def _emit_report(report: dict, destination: Path | None) -> None:
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='repointer',description='RePointer research tools; synthetic runs do not reproduce official results')
    sub = parser.add_subparsers(dest='command',required=True)
    source = sub.add_parser('audit-source',help='verify frozen source hashes and syntax')
    source.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    motion = sub.add_parser('validate-motion',help='check a motion NPZ schema')
    motion.add_argument('motion_npz',type=Path); motion.add_argument('--expected-frames',type=int)
    assets = sub.add_parser('assets-status', help='report configured external assets without exposing local paths')
    assets.add_argument('--config', type=Path, required=True)
    assets.add_argument('--report', type=Path)
    m1 = sub.add_parser('audit-m1', help='read-only integrity audit of an external historical M1 run')
    location = m1.add_mutually_exclusive_group(required=True)
    location.add_argument('--config', type=Path)
    location.add_argument('--run-dir', type=Path)
    m1.add_argument('--reference', type=Path, help='bind audit to a sanitized reference JSON')
    m1.add_argument('--verify-hashes', action='store_true', help='also hash every motion file')
    m1.add_argument('--verify-motions', action='store_true', help='also validate every selected NPZ schema')
    m1.add_argument('--report', type=Path, help='write path-free JSON to a new file outside the audited run')
    replay = sub.add_parser('replay', help='run frozen real-data stages into a new external private output root')
    replay.add_argument('--config', type=Path, required=True)
    replay.add_argument('--output-root', type=Path, required=True)
    replay.add_argument('--python', dest='research_python', required=True, help='existing research interpreter; its environment is not modified')
    replay.add_argument('--repo-root', type=Path, default=Path.cwd())
    replay.add_argument('--stages', default='p2a,p2c1,p2c2')
    replay.add_argument('--threads', type=int, default=4)
    replay.add_argument('--dry-run', action='store_true')
    for name,help_text in (
        ('make-fixture','create invented training/transcript/donor data'),
        ('train','fit synthetic-prior or optional WordPhaseNet plus router/ranker on fixtures'),
        ('infer','materialize fixture candidates A/B and unchanged selector output'),
        ('smoke','run fixture creation, training, inference and local ZIP validation')):
        command = sub.add_parser(name,help=help_text)
        command.add_argument('--config',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.command=='audit-source': return audit(args.repo_root.resolve())
    if args.command=='validate-motion': return validate_motion(args.motion_npz,args.expected_frames)
    try:
        if args.command == 'replay':
            from .assets import load_asset_paths
            from .replay import execute_replay, plan_replay
            paths = load_asset_paths(args.config)
            plans = plan_replay(paths, args.output_root, args.research_python, args.repo_root,
                                tuple(args.stages.split(',')))
            if args.dry_run:
                _emit_report({'schema':'repointer.replay-plan.v1', 'device':'cpu',
                              'stages':[stage.name for stage in plans], 'executed':False,
                              'full_reproduction_validated':False}, None)
                return 0
            report = execute_replay(plans, args.output_root, args.threads)
            _emit_report(report, None)
            return 0 if report['status'] == 'PASS' else 1
        if args.command in ('assets-status', 'audit-m1'):
            from .assets import asset_status, load_asset_paths
            paths = load_asset_paths(args.config) if args.config else {}
            protected = [path for path in paths.values() if path is not None]
            if args.command == 'assets-status':
                destination = _report_destination(args.report, protected)
                report = asset_status(paths)
                _emit_report(report, destination)
                return 0
            from .artifacts import audit_m1_run, load_m1_reference
            run = args.run_dir or paths.get('m1_full_dir')
            if run is None:
                raise ValueError('m1_full_dir is unconfigured')
            run = run.resolve()
            destination = _report_destination(args.report, protected + [run])
            reference = load_m1_reference(args.reference) if args.reference else None
            report = audit_m1_run(run, reference=reference,
                                  verify_hashes=args.verify_hashes, verify_motions=args.verify_motions)
            _emit_report(report, destination)
            return 0 if report['status'] == 'PASS' else 1
        from .config import load_config
        from .fixtures import write_fixture
        from .pipeline import infer_fixture,train_fixture,write_json
        config = load_config(args.config)
        if args.command in ('make-fixture','smoke'):
            write_fixture(config.fixture_root,seed=config.pipeline.seed)
            print('Created invented fixture motions and timed transcripts.',flush=True)
        if args.command in ('train','smoke'):
            training = train_fixture(config.fixture_root,config.checkpoint_root,config.pipeline)
            print(f"Trained {training['training_queries']} synthetic queries in {training['folds']} folds; timing={config.pipeline.timing_backend}.",flush=True)
        if args.command in ('infer','smoke'):
            result = infer_fixture(config.fixture_root,config.checkpoint_root,config.output_root,config.pipeline)
            print(f"Validated {len(result['decisions'])} selected synthetic SMPL-X-layout outputs.",flush=True)
        if args.command=='smoke':
            from .challenge import package_submission
            manifest = {row['sample_id']:row['frames'] for row in result['decisions']}
            package_submission(config.output_root/'selected',config.output_root/'synthetic_submission.zip',manifest)
            write_json(config.output_root/'smoke.json',dict(status='SYNTHETIC_SMOKE_PASSED',
                outputs=len(manifest),paper_parity=False,official_validator_run=False))
            print('PASS: synthetic smoke, exact A/B copies and local ZIP schema. Official results remain unverified.')
        return 0
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
