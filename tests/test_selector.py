import copy
from pathlib import Path
import importlib.util
import numpy as np
import pytest

from repointer.selector import choose_candidate, robust_metrics, select_candidate


def _base():
    return dict(detector_floor_score1b=.9, window_q10_score1b=.91, planned_floor_score1b=.95,
                world_bbox_window_q10_score1b=.92, world_point_window_q10_score1b=.9,
                forearm_direction_dispersion_deg=2., detector_spread_frames=2.,
                generic_joint_jerk=.0003, generic_foot_sliding=.001,
                joint_limit_fraction=0., robust_spatial_utility=.92)


def test_selector_matches_frozen_for_safe_unsafe_and_tied_candidates():
    path = Path(__file__).resolve().parents[1] / 'frozen/selector/repointer_p2c2_spatial_first_detector_robust_candidate_c_v0_1_0.py'
    spec = importlib.util.spec_from_file_location('frozen_selector', path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    a = _base()
    b = {**a, 'robust_spatial_utility': .95, 'detector_floor_score1b': .94, 'window_q10_score1b': .95}
    assert choose_candidate(a, b) == ref.choose_candidate(a, b, .001)
    for key, value in [('planned_floor_score1b', .7), ('generic_joint_jerk', .1),
                       ('foot_sliding', .1), ('joint_limit_fraction', .1)]:
        if key == 'foot_sliding':
            key = 'generic_foot_sliding'
        unsafe = {**b, key: value}
        assert choose_candidate(a, unsafe) == ref.choose_candidate(a, unsafe, .001)
        assert choose_candidate(a, unsafe)['choice'] == 'A'
    assert choose_candidate(a, a)['choice'] == 'A'


def test_selection_preserves_complete_candidate_identity():
    a = {'poses': np.zeros((5, 165)), 'trans': np.zeros((5, 3))}
    b = {'poses': np.ones((5, 165)), 'trans': np.ones((5, 3))}
    before_a, before_b = copy.deepcopy(a), copy.deepcopy(b)
    metrics = _base()
    selected, decision = select_candidate(a, b, metrics, {**metrics, 'robust_spatial_utility': .95})
    assert selected is b and decision['choice'] == 'B'
    selected, decision = select_candidate(a, b, metrics, metrics)
    assert selected is a and decision['choice'] == 'A'
    for data, before in [(a, before_a), (b, before_b)]:
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])


def test_multi_frame_utility_uses_planned_and_detector_windows():
    score = np.ones(10)
    score[3] = .2
    quality = dict(generic_joint_jerk=0, generic_foot_sliding=0, joint_limit_fraction=0)
    metrics = robust_metrics(score, score, score, 4, [4, 5, 6, 5], 1., 0., quality, 1, 0)
    assert metrics['planned_floor_score1b'] == .2
    assert metrics['detector_floor_score1b'] == 1
    assert metrics['window_q10_score1b'] == pytest.approx(.44)
    assert metrics['robust_frame_count'] == 4
    expected = .34 + .26 * .44 + .18 * .2 + .10 + .06 * .8 + .04 + .02
    assert metrics['robust_spatial_utility'] == pytest.approx(expected)
    # Exact world target can be ignored only for POINT_INSIDE_AABB resolution.
    point = np.zeros(10)
    assert robust_metrics(np.ones(10), np.ones(10), point, 4, [4] * 4, 1, 0, quality)['robust_spatial_utility'] == pytest.approx(.06)
    assert robust_metrics(np.ones(10), np.ones(10), point, 4, [4] * 4, 1, 0, quality, point_inside_bbox=True)['robust_spatial_utility'] == pytest.approx(1.)


@pytest.mark.parametrize('key', ['robust_spatial_utility', 'planned_floor_score1b', 'generic_joint_jerk'])
def test_selector_rejects_nonfinite_metrics(key):
    a = _base()
    with pytest.raises(ValueError, match='finite'):
        choose_candidate(a, {**a, key: float('nan')})
    with pytest.raises(ValueError):
        choose_candidate(a, a, -.01)


def test_robust_shape_and_index_checks():
    score = np.ones(10)
    quality = dict(generic_joint_jerk=0, generic_foot_sliding=0, joint_limit_fraction=0)
    with pytest.raises(ValueError, match='matching shape'):
        robust_metrics(score, np.ones(9), score, 4, [4] * 4, 1, 0, quality)
    with pytest.raises(ValueError, match='detector frames'):
        robust_metrics(score, score, score, 4, [4, 4, 4, 10], 1, 0, quality)
    with pytest.raises(ValueError, match='radii'):
        robust_metrics(score, score, score, 4, [4] * 4, 1, 0, quality, -1)
