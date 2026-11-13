"""Asset-free CLI integration and Candidate B budget regression checks."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from repointer.cli import main
from repointer.config import PipelineConfig
from repointer.contracts import Candidate, Motion, Query
from repointer.ranking import build_hypotheses, ranker_feature_names
from repointer.timing import prior_distribution


def _config(path, fixture, checkpoint, output, settings):
    path.write_text(json.dumps({
        'pipeline': asdict(settings),
        'paths': {'fixture_root': str(fixture), 'checkpoint_root': str(checkpoint), 'output_root': str(output)},
    }), encoding='utf-8')
    return path


@pytest.fixture(scope='module')
def completed_smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp('repointer_smoke')
    fixture, checkpoint, output = root / 'fixture', root / 'checkpoints', root / 'output'
    settings = PipelineConfig(seed=5, top_k=2, donors_per_hypothesis=2, ik_steps=1)
    config = _config(root / 'config.json', fixture, checkpoint, output, settings)
    assert main(['smoke', '--config', str(config)]) == 0
    return SimpleNamespace(root=root, fixture=fixture, checkpoint=checkpoint, output=output,
                           settings=settings, config=config)


def test_cli_smoke_materializes_two_float32_outputs_without_official_claim(completed_smoke):
    run = completed_smoke
    training = json.loads((run.checkpoint / 'training.json').read_text())
    inference = json.loads((run.output / 'inference.json').read_text())
    smoke = json.loads((run.output / 'smoke.json').read_text())
    assert training['synthetic'] is True and training['paper_parity'] is False
    assert inference['synthetic'] is True and inference['paper_parity'] is False
    assert training['training_queries'] == 8 and training['folds'] == 3
    assert training['ranker_rows'] == 64
    assert smoke == dict(status='SYNTHETIC_SMOKE_PASSED', outputs=2,
                         paper_parity=False, official_validator_run=False)
    assert len(inference['decisions']) == 2
    assert len(list((run.output / 'selected').glob('*.npz'))) == 2
    for decision in inference['decisions']:
        assert decision['candidate_budget'] == 8
        selected = run.output / 'selected' / f"{decision['sample_id']}.npz"
        original_dir = 'candidate_b' if decision['choice'] == 'B' else 'candidate_a'
        original = run.output / original_dir / selected.name
        assert selected.read_bytes() == original.read_bytes()
        with np.load(selected, allow_pickle=False) as data:
            assert sorted(data.files) == ['poses', 'trans']
            assert data['poses'].shape == (decision['frames'], 165)
            assert data['trans'].shape == (decision['frames'], 3)
            for name in data.files:
                assert data[name].dtype == np.float32
                assert np.isfinite(data[name]).all()
    assert (run.output / 'synthetic_submission.zip').is_file()


def test_smoke_rerun_refuses_to_overwrite_existing_run(completed_smoke):
    run = completed_smoke
    before = {path.name: path.read_bytes() for path in (run.output / 'selected').glob('*.npz')}
    assert main(['smoke', '--config', str(run.config)]) == 1
    assert before == {path.name: path.read_bytes() for path in (run.output / 'selected').glob('*.npz')}


def test_cli_inference_repeats_arrays_in_new_output_directory(completed_smoke, tmp_path):
    run = completed_smoke
    second_output = tmp_path / 'second_inference'
    config = _config(tmp_path / 'infer.json', run.fixture, run.checkpoint, second_output, run.settings)
    assert main(['infer', '--config', str(config)]) == 0
    for source in (run.output / 'selected').glob('*.npz'):
        with np.load(source, allow_pickle=False) as original, np.load(second_output / 'selected' / source.name, allow_pickle=False) as repeated:
            for key in ('poses', 'trans'):
                np.testing.assert_array_equal(original[key], repeated[key])
    assert main(['infer', '--config', str(config)]) == 1


@pytest.mark.parametrize('changed', ['manifest', 'donor', 'checkpoint'])
def test_changed_input_bytes_rejected_before_output_creation(completed_smoke, tmp_path, changed):
    run = completed_smoke
    if changed == 'manifest':
        path = run.fixture / 'manifest.json'
        addition = b'\n'
    elif changed == 'donor':
        manifest = json.loads((run.fixture / 'manifest.json').read_text())
        path = run.fixture / manifest['donors'][0]['motion']
        # ZIP readers accept trailing bytes, so schema remains valid while the
        # artifact fingerprint changes. The binding must detect the change.
        addition = b'changed-donor-bytes'
    else:
        path = run.checkpoint / 'router.npz'
        addition = b'changed-checkpoint-bytes'
    before = path.read_bytes()
    output = tmp_path / 'rejected'
    config = _config(tmp_path / 'infer.json', run.fixture, run.checkpoint, output, run.settings)
    try:
        path.write_bytes(before + addition)
        assert main(['infer', '--config', str(config)]) == 1
        assert not output.exists()
    finally:
        path.write_bytes(before)


def test_inference_rejects_configuration_mismatch_before_output_creation(completed_smoke, tmp_path):
    run = completed_smoke
    settings = PipelineConfig(**{**asdict(run.settings), 'top_k': 3})
    output = tmp_path / 'mismatch'
    config = _config(tmp_path / 'infer.json', run.fixture, run.checkpoint, output, settings)
    assert main(['infer', '--config', str(config)]) == 1
    assert not output.exists()


def test_inference_rejects_changed_implementation_binding(completed_smoke,tmp_path):
    run = completed_smoke
    path = run.checkpoint/'training.json'
    before = path.read_bytes()
    report = json.loads(before)
    report['implementation_sha256']['pipeline.py'] = '0'*64
    output = tmp_path/'changed_implementation'
    config = _config(tmp_path/'infer.json',run.fixture,run.checkpoint,output,run.settings)
    try:
        path.write_text(json.dumps(report),encoding='utf-8')
        assert main(['infer','--config',str(config)]) == 1
        assert not output.exists()
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize('fallback_id', ['d0', 'd2'])
@pytest.mark.parametrize('source',['median','wordphase'])
def test_candidate_budget_reserves_slot_at_fallback_timing_before_donor_truncation(monkeypatch, fallback_id,source):
    import repointer.pipeline as pipeline
    query = Query('invented', 64, 0, np.asarray([0., 0, 3.]))
    motion = Motion(np.zeros((64, 165), dtype=np.float32), np.zeros((64, 3), dtype=np.float32))
    settings = PipelineConfig(top_k=3, donors_per_hypothesis=2, ik_steps=1)
    donors = [SimpleNamespace(context=SimpleNamespace(sample_id=f'd{i}')) for i in range(3)]
    probability, median = prior_distribution(.5), 10
    hypotheses = build_hypotheses(probability, query.frames, median)
    fallback = Candidate(motion, median, fallback_id, .2, source)

    def metrics(planned):
        return dict(planned_frame=planned, self_frame=planned,
                    det_source=planned, det_refined=planned, det_heuristic=planned,
                    detector_agreement=1., planned_score1b_proxy20=.95,
                    self_score1b_proxy20=.95, self_angle_deg=1,
                    joint_jerk=0., foot_sliding=0., pose_deviation=0., max_reach=1.,
                    joint_limit_fraction=0.)

    a_pair = (fallback, {**metrics(median), 'donor_rank': 3, 'donor_distance': .2})

    def retrieve(_query, _donors, planned, count):
        # d2 is outside the first two donors. A still consumes one slot in its
        # own timing hypothesis, so the last hypothesis retains both donors.
        return [(donor, float(index)) for index, donor in enumerate(donors)][:count]

    def realize(_query, donor, distance, planned, source, engine, **kwargs):
        return Candidate(motion, planned, donor.context.sample_id, distance, source), metrics(planned)

    monkeypatch.setattr(pipeline, 'retrieve', retrieve)
    monkeypatch.setattr(pipeline, 'realize', realize)
    pairs, features = pipeline.candidate_pool(query, donors, a_pair, probability, median, settings)
    assert len(pairs) == 8 and features.shape == (8, 48)
    assert pairs[0] is a_pair
    assert features[0,ranker_feature_names().index('hyp_expected')] == float(source=='wordphase')
    assert Counter(candidate.planned_frame for candidate, _ in pairs) == {
        hypothesis['planned_frame']: 2 for hypothesis in hypotheses
    }
    fresh_at_fallback = [(candidate, metric) for candidate, metric in pairs[1:] if candidate.planned_frame == median]
    assert len(fresh_at_fallback) == 1
    assert fresh_at_fallback[0][0].donor_id != fallback_id
    assert fresh_at_fallback[0][1]['donor_rank'] == 1
    rank_column = ranker_feature_names().index('donor_rank_norm')
    for index, (_, metric) in enumerate(pairs):
        assert features[index, rank_column] == pytest.approx(metric['donor_rank'] / 8)
