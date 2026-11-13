from pathlib import Path
import importlib.util
import numpy as np
import pytest

from repointer.ranking import (
    RidgeModel, build_hypotheses, build_ranker_feature, build_router_features,
    cross_fit_ridge, fit_domain_hand_medians, median_frame, phase_route,
    ranker_feature_names, select_with_threshold,
)

ROOT = Path(__file__).resolve().parents[1]


def _reference(filename):
    path = ROOT / 'frozen' / 'candidates' / filename
    spec = importlib.util.spec_from_file_location('frozen_ranking_reference', path)
    mod = importlib.util.module_from_spec(spec)
    # The dataclass decorator resolves module names through sys.modules.
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_router_and_ranker_features_match_frozen_allowlist():
    router = _reference('repointer_p2a_phase_router_oof_v0_1_0.py')
    mh = _reference('repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py')
    row = {
        'domain': 'natural', 'hand': 1, 'clip_length': 120,
        'target_distance': 2.3, 'target_azimuth_deg': 42, 'target_elevation_deg': 13,
        'median_peak': 51, 'wordphase_peak': 62,
        'median_planned_frame': 51, 'wordphase_planned_frame': 62,
        'wordphase_det_source': 60, 'wordphase_det_refined': 64,
        'wordphase_self_score1b_proxy20': .9, 'median_self_score1b_proxy20': .95,
        'gt_peak': 100000, 'hold_start': -999, 'oracle_score': 123,
    }
    X, names, columns = build_router_features([row])
    ref_X, ref_names, ref_columns = router.build_features([row])
    np.testing.assert_allclose(X, ref_X)
    assert names == ref_names and columns == ref_columns and X.shape == (1, 53)
    scrubbed = {k: v for k, v in row.items() if k not in {'gt_peak', 'hold_start', 'oracle_score'}}
    np.testing.assert_array_equal(X, build_router_features([scrubbed])[0])
    timing = {'hypothesis': 'wp_q25', 'wordphase_probability': .25, 'wordphase_entropy_norm': .8}
    candidate = {'planned_frame': 60, 'self_frame': 64, 'donor_rank': 2, 'joint_jerk': .001}
    np.testing.assert_allclose(build_ranker_feature(row, timing, candidate), mh.build_ranker_feature(row, timing, candidate))
    assert len(ranker_feature_names()) == 48


def test_group_weighted_ridge_matches_frozen_and_serializes(tmp_path):
    reference = _reference('repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py')
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(12, 3)), rng.normal(size=12)
    group = np.asarray(['a'] * 6 + ['b'] * 4 + ['c'] * 2)
    query = np.asarray(['q1'] * 2 + ['q2'] * 4 + ['q3'] * 4 + ['q4'] * 2)
    model = RidgeModel.fit(X, y, group, .5, query)
    beta, mu, sd = reference.fit_weighted_ridge(X, y, group, query, .5)
    np.testing.assert_allclose(model.beta, beta)
    np.testing.assert_allclose(model.mean, mu)
    np.testing.assert_allclose(model.std, sd)
    np.testing.assert_allclose(model.predict(X), reference.ridge_predict(X, beta, mu, sd))
    path = tmp_path / 'model.npz'
    model.save(path)
    np.testing.assert_array_equal(RidgeModel.load(path).predict(X), model.predict(X))
    np.testing.assert_array_equal(RidgeModel.from_dict(model.to_dict()).predict(X), model.predict(X))
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError, match='dimension'):
        model.predict(np.ones((3, 4)))


def test_cross_fit_excludes_validation_targets_and_statistics():
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = np.arange(8, dtype=float)
    groups = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']
    folds = np.repeat(np.arange(4), 2)
    result = cross_fit_ridge(X, y, groups, folds)
    changed_y = y.copy()
    changed_y[folds == 0] += 1000
    second = cross_fit_ridge(X, changed_y, groups, folds)
    np.testing.assert_array_equal(result.predictions[folds == 0], second.predictions[folds == 0])
    np.testing.assert_array_equal(result.models[0].mean, X[folds != 0].mean(axis=0))
    assert len(result.models) == 4 and np.isfinite(result.predictions).all()


def test_cross_fit_rejects_group_and_query_leakage():
    X, y = np.ones((4, 2)), np.arange(4.)
    with pytest.raises(ValueError, match='spans'):
        cross_fit_ridge(X, y, ['a', 'b', 'a', 'c'], [0, 0, 1, 1])
    with pytest.raises(ValueError, match='spans'):
        cross_fit_ridge(X, y, ['a', 'b', 'c', 'd'], [0, 0, 1, 1], ['q', 'r', 'q', 's'])
    with pytest.raises(ValueError, match='at least two'):
        cross_fit_ridge(X, y, ['a', 'b', 'c', 'd'], [0, 0, 0, 0])


def test_medians_and_router_boundary():
    medians = fit_domain_hand_medians(['natural'] * 3 + ['synthetic'], [0, 0, 1, 1], [.2, .4, .8, .7])
    assert medians['natural', 0] == pytest.approx(.3)
    assert median_frame(medians, 'natural', 0, 101) == 30
    assert phase_route(0) == 'wordphase' and phase_route(-1e-10) == 'median'
    with pytest.raises(KeyError):
        median_frame(medians, 'synthetic', 0, 101)
    with pytest.raises(ValueError):
        phase_route(float('nan'))
    with pytest.raises(ValueError):
        fit_domain_hand_medians(['natural'], [0], [1.1])


def test_four_hypotheses_match_frozen_with_collapse_fill():
    mh = _reference('repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py')
    p = np.exp(-((np.arange(64) - 31.5) / 3) ** 2)
    p /= p.sum()
    centers = np.linspace(0, 1, 64)
    mean = float(p @ centers)
    dist = dict(clip_length=120, prob=p, expected_ratio=mean,
                expected_frame=round(mean * 119), std_ratio=float(np.sqrt(p @ (centers - mean) ** 2)),
                entropy_norm=float(-(p * np.log(np.maximum(p, 1e-12))).sum() / np.log(64)),
                max_prob=float(p.max()), top_bin_margin=float(np.sort(p)[-1] - np.sort(p)[-2]))
    actual = build_hypotheses(p, 120, 60)
    reference = mh.build_hypotheses(dist, 60, 4, 6)
    assert len(actual) == len({h['planned_frame'] for h in actual}) == 4
    assert [h['hypothesis'] for h in actual] == [h['hypothesis'] for h in reference]
    for a, b in zip(actual, reference):
        for key in a:
            if isinstance(a[key], str):
                assert a[key] == b[key]
            else:
                assert a[key] == pytest.approx(b[key])
    assert actual[0]['hypothesis'] == 'median'
    for invalid in (np.zeros(64), np.ones(63), np.full(64, np.nan), -np.ones(64)):
        with pytest.raises(ValueError):
            build_hypotheses(invalid, 120, 60)
    with pytest.raises(ValueError):
        build_hypotheses(p, 5, 2)


def test_ranker_retains_base_on_unsafe_high_score():
    base = dict(self_score1b_proxy20=.95, planned_score1b_proxy20=.98,
                joint_limit_fraction=0, foot_sliding=.001, pose_deviation=.001, donor_rank=2)
    safe = {**base, 'donor_rank': 1}
    unsafe = {**safe, 'planned_score1b_proxy20': .5}
    assert select_with_threshold([base, safe, unsafe], np.asarray([.8, .9, 1]), 0, .01) == 1
    assert select_with_threshold([base, safe, unsafe], np.asarray([.8, .805, 1]), 0, .01) == 0


def test_nested_calibration_excludes_outer_targets():
    from repointer.ranking import nested_ranker_calibration
    X = np.arange(24.).reshape(12, 2)
    y = np.linspace(0, 1, 12)
    folds = np.repeat(np.arange(3), 4)
    groups = np.repeat(['a', 'b', 'c'], 4)
    queries = np.repeat(['q1', 'q2', 'q3', 'q4', 'q5', 'q6'], 2)
    row = dict(self_score1b_proxy20=.95, planned_score1b_proxy20=.98,
               joint_limit_fraction=0, foot_sliding=0, pose_deviation=0, donor_rank=1)
    rows = [dict(row) for _ in range(12)]
    bases = {q: i * 2 for i, q in enumerate(np.unique(queries))}
    first = nested_ranker_calibration(X, y, groups, folds, queries, rows, bases, (.1, .5), (0., .02))
    changed_y = y.copy()
    changed_y[folds == 0] += 999
    second = nested_ranker_calibration(X, changed_y, groups, folds, queries, rows, bases, (.1, .5), (0., .02))
    assert first[0].calibration == second[0].calibration
    assert first[0].ridge_lambda == second[0].ridge_lambda
    assert first[0].replacement_threshold == second[0].replacement_threshold
    np.testing.assert_array_equal(first[0].model.beta, second[0].model.beta)
