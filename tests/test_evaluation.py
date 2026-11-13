"""Local geometry/temporal diagnostics on invented joints; no licensed assets."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from repointer.contracts import Candidate, Motion, Query
from repointer.evaluation import (
    evaluate_candidate, evaluate_hold, motion_quality, spatial_scores,
    synthetic_detectors,
)
from repointer.kinematics import SyntheticKinematics


def _joints(frames=12):
    joints = np.zeros((frames, 55, 3), dtype=np.float32)
    joints[:, [20, 21], 2] = 1.
    joints[:, [16, 17], 2] = -1.
    return joints


def _candidate(frames=12):
    motion = Motion(np.zeros((frames, 165), dtype=np.float32), np.zeros((frames, 3), dtype=np.float32))
    return Candidate(motion, 5, 'invented_donor', .1, 'median')


def _detector(joints, hand):
    assert joints.shape[1:] == (55, 3) and hand in (0, 1)
    return dict(source=5, refined=5, heuristic=6, ensemble=5, agreement=1.)


class FixedKinematics:
    """Only a geometry fixture: deliberately not the synthetic detector backend."""
    name = 'invented-joints'

    def __init__(self, joints):
        self.values = joints

    def joints(self, motion, betas, gender='neutral'):
        assert motion.frames == len(self.values)
        return self.values.copy()


def test_forearm_geometry_point_sphere_and_degenerate_rays():
    joints = _joints()
    for hand in (0, 1):
        score, angle = spatial_scores(joints, hand, [0, 0, 3])
        np.testing.assert_array_equal(angle, 0)
        np.testing.assert_array_equal(score, 1)
        score, angle = spatial_scores(joints, hand, [3, 0, 3])
        np.testing.assert_allclose(angle, 45)
        np.testing.assert_allclose(score, np.exp(-45 ** 2 / (2 * 20 ** 2)))
        # Bounding sphere angular margin subtracts 45 degrees at this radius.
        score, angle = spatial_scores(joints, hand, [3, 0, 3], half_diagonal=3)
        np.testing.assert_allclose(angle, 0, atol=1e-6)
        np.testing.assert_allclose(score, 1)
    score, angle = spatial_scores(joints, 0, [0, 0, 0])
    np.testing.assert_array_equal(angle, 180)
    assert np.all(score < 1e-10)
    joints[:, 20] = joints[:, 18]
    _, angle = spatial_scores(joints, 0, [0, 0, 3])
    np.testing.assert_array_equal(angle, 180)


@pytest.mark.parametrize('joints,hand,target,extent', [
    (np.zeros((12, 21, 3)), 0, [0, 0, 3], 0),
    (_joints(), 2, [0, 0, 3], 0),
    (_joints(), 0, [0, 0], 0),
    (_joints(), 0, [0, np.nan, 3], 0),
    (_joints(), 0, [0, 0, 3], -1),
])
def test_spatial_contract_validation(joints, hand, target, extent):
    with pytest.raises(ValueError):
        spatial_scores(joints, hand, target, extent)


def test_temporal_hold_includes_both_endpoints_and_uses_frame_distance():
    for frame in (10, 11, 15, 20):
        assert evaluate_hold(frame, 10, 20) == 1
    assert evaluate_hold(5, 10, 20) == pytest.approx(np.exp(-.5))
    assert evaluate_hold(25, 10, 20) == pytest.approx(np.exp(-.5))
    assert evaluate_hold(25, 10, 20, sigma=10) == pytest.approx(np.exp(-.125))
    for values in [(-1, 10, 20), (5, 20, 10), (5., 10, 20)]:
        with pytest.raises(ValueError):
            evaluate_hold(*values)
    with pytest.raises(ValueError):
        evaluate_hold(5, 10, 20, sigma=0)


def test_synthetic_detectors_depend_only_on_motion_and_active_hand():
    candidate = _candidate()
    engine = SyntheticKinematics()
    query = Query('invented', 12, 0, np.asarray([-1., 0, 2.]))
    first = evaluate_candidate(candidate, query, engine)
    # Transcript and target changes cannot affect a motion-only detector.
    second = evaluate_candidate(replace(candidate, planned_frame=7),
        replace(query, target=np.asarray([2., 1., -3.]), words=('unrelated transcript',)), engine)
    for key in ('det_source', 'det_refined', 'det_heuristic', 'det_ensemble', 'self_frame', 'detector_agreement'):
        assert first[key] == second[key]
    assert first['self_score1b_proxy20'] != second['self_score1b_proxy20']
    assert not {'hold_start', 'hold_end', 'released_peak_frame', 'gt_peak'} & set(Query.__dataclass_fields__)
    assert not {'temporal_proxy_sigma5', 'self_stg_proxy_sigma5', 'oracle_frame'} & set(first)
    det = synthetic_detectors(engine.joints(candidate.motion, query.betas), query.hand)
    assert all(2 <= det[key] <= query.frames - 3 for key in ('source', 'refined', 'heuristic', 'ensemble'))


def test_external_kinematics_requires_explicit_motion_only_detector():
    candidate = _candidate()
    query = Query('invented', 12, 0, np.asarray([0., 0, 3.]))
    backend = FixedKinematics(_joints())
    with pytest.raises(ValueError, match='explicitly supplied'):
        evaluate_candidate(candidate, query, backend)
    metrics = evaluate_candidate(candidate, query, backend, detector=_detector)
    assert metrics['robust_spatial_utility'] == pytest.approx(1)
    with pytest.raises(ValueError, match='frame count mismatch'):
        evaluate_candidate(candidate, replace(query, frames=13), backend, detector=_detector)


def test_bbox_inside_policy_ignores_world_point_only_inside_bounds():
    candidate = _candidate()
    backend = FixedKinematics(_joints())
    query = Query('invented', 12, 0, np.asarray([1., 0, 3.]),
        target_world=np.asarray([1., 0, 3.]), bbox_center=np.asarray([0., 0, 3.]),
        bbox_size=np.asarray([2.2, .2, .2]))
    inside = evaluate_candidate(candidate, query, backend, detector=_detector)
    assert inside['world_point_window_q10_score1b'] < 1
    assert inside['robust_spatial_utility'] == pytest.approx(1)
    outside = evaluate_candidate(candidate, replace(query, target=np.asarray([2., 0, 3.]),
        target_world=np.asarray([2., 0, 3.])), backend, detector=_detector)
    assert outside['world_bbox_window_q10_score1b'] == 1
    assert outside['robust_spatial_utility'] < .9


def test_relative_and_fixed_world_geometry_detect_translation_drift():
    joints = _joints()
    # Ray stays parallel to +z while pelvis translates away from fixed world target.
    joints[:, :, 0] += np.arange(12)[:, None] * .2
    query = Query('invented', 12, 0, np.asarray([0., 0, 3.]), target_world=np.asarray([0., 0, 3.]))
    metrics = evaluate_candidate(_candidate(), query, FixedKinematics(joints), detector=_detector)
    assert metrics['planned_score1b_proxy20'] == 1
    assert metrics['relative_window_q10_score1b'] == 1
    assert metrics['world_point_window_q10_score1b'] < .7
    assert metrics['robust_spatial_utility'] < .9


def test_quality_uses_historical_divide_by_fps_scale_and_matches_selector():
    candidate = _candidate()
    joints = _joints().astype(float)
    joints[:, [10, 11], 0] = np.arange(12)[:, None] * .3
    quality = motion_quality(joints, candidate.motion)
    assert quality['foot_sliding'] == pytest.approx(.01)
    assert quality['generic_foot_sliding'] == quality['foot_sliding']
    assert quality['joint_jerk'] == pytest.approx(0, abs=1e-16)
    # Differential check against the frozen selector's actual diagnostic scale.
    path = Path(__file__).resolve().parents[1] / 'frozen/selector/repointer_p2c2_spatial_first_detector_robust_candidate_c_v0_1_0.py'
    spec = importlib.util.spec_from_file_location('frozen_quality_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    module = SimpleNamespace(JOINT_INDEX={'left_foot': 10, 'right_foot': 11})
    expected = reference.generic_motion_quality(joints, candidate.motion.poses, module)
    for key in expected:
        assert quality[key] == pytest.approx(expected[key])
    poses = candidate.motion.poses.copy()
    poses[:, 3] = 4
    changed = Motion(poses, candidate.motion.trans)
    assert motion_quality(joints, changed)['joint_limit_fraction'] == pytest.approx(1 / 21)
    assert motion_quality(joints, changed, candidate.motion)['pose_deviation'] == pytest.approx(4 / 55)
