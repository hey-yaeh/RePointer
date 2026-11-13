"""Replay orchestration tests use invented inputs and subprocess stubs only."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from repointer.replay import STAGES, execute_replay, plan_replay


@pytest.fixture
def setup(tmp_path):
    repo = tmp_path / 'source'
    assets = tmp_path / 'assets'
    keys = ('dataset_root','p0c0_dir','m1_full_dir','p1a_dir','p1_full_dir',
            'p2a_dir','p2a_mh_dir','p2c0_dir','p2c1_dir','smplx_model_dir')
    paths = {key:assets/key for key in keys}
    for path in paths.values(): path.mkdir(parents=True)
    inventory = {}
    for script,_,_ in STAGES.values():
        path = repo / script
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("raise RuntimeError('invented source must never execute in a unit test')\n")
        inventory[script] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (repo/'docs').mkdir()
    (repo/'docs/SOURCE_INVENTORY.json').write_text(json.dumps(inventory))
    return repo,paths,tmp_path/'new-private-output'


def test_pretrained_chain_connects_new_router_and_candidates(setup):
    repo,paths,output = setup
    plans = plan_replay(paths,output,sys.executable,repo)
    assert [p.name for p in plans] == ['p2a','p2c1','p2c2']
    assert plans[1].command[plans[1].command.index('--p2a-dir')+1] == str(output/'p2a')
    assert plans[2].command[plans[2].command.index('--p2c1-dir')+1] == str(output/'p2c1')
    assert '-B' in plans[0].command and not output.exists()


def test_new_timing_weights_require_new_motion_evidence(setup):
    repo,paths,output = setup
    with pytest.raises(ValueError,match='regenerate P1_FULL'):
        plan_replay(paths,output,sys.executable,repo,('p1a','p2c1'))
    single = plan_replay(paths,output,sys.executable,repo,('p1a',))
    assert single[0].command[single[0].command.index('--epochs')+1] == '60'


@pytest.mark.parametrize('stages',[(),('unknown',),('p2a','p2a'),('p2c2','p2c1')])
def test_bad_stage_order_rejected(setup,stages):
    repo,paths,output = setup
    with pytest.raises(ValueError):plan_replay(paths,output,sys.executable,repo,stages)


def test_input_output_and_public_tree_boundaries(setup):
    repo,paths,output = setup
    for forbidden in (repo/'outputs/replay',paths['p1_full_dir']/'new-run'):
        with pytest.raises(ValueError):plan_replay(paths,forbidden,sys.executable,repo)
    output.mkdir()
    with pytest.raises(FileExistsError):plan_replay(paths,output,sys.executable,repo)


def test_source_tamper_and_missing_dependencies_fail_before_outputs(setup):
    repo,paths,output = setup
    paths['p1_full_dir'] = None
    with pytest.raises(ValueError,match='p1_full_dir'):plan_replay(paths,output,sys.executable,repo)
    paths['p1_full_dir'] = paths['p1a_dir']
    (repo/STAGES['p2a'][0]).write_text('changed source')
    with pytest.raises(ValueError,match='hash mismatch'):plan_replay(paths,output,sys.executable,repo)
    assert not output.exists()


def test_execution_uses_private_logs_and_stops_on_failed_gate(setup,monkeypatch):
    repo,paths,output = setup
    plans = plan_replay(paths,output,sys.executable,repo)
    calls = []
    def run(command,**kwargs):
        calls.append(command)
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE']=='1'
        assert kwargs['env']['OMP_NUM_THREADS']=='2'
        assert kwargs.get('shell') is None
        destination = Path(command[command.index('--output-dir')+1])
        name = destination.name
        (destination/STAGES[name][1]).write_text(json.dumps({'status':STAGES[name][2] if name=='p2a' else 'FAILED'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('repointer.replay.subprocess.run',run)
    report = execute_replay(plans,output,threads=2)
    assert report['status']=='FAIL' and len(calls)==2
    assert not report['official_server_scores_reproduced']
    assert str(output) not in json.dumps(report)
    assert (output/'commands.private.json').exists()
    assert not (output/'p2c2').exists()


def test_constructed_stage_cannot_write_outside_replay_root(setup):
    repo,paths,output = setup
    plans = plan_replay(paths,output,sys.executable,repo)
    invalid = replace(plans[0],output=paths['p1_full_dir'])
    with pytest.raises(ValueError,match='inside'):execute_replay([invalid],output)
    assert not output.exists()
