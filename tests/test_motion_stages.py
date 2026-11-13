"""Asset-free contracts and M1 tests, including historical warp comparison."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from repointer.contracts import ACTIVE_JOINTS, Candidate, Donor, Motion, Query
from repointer.ik import optimize_apex
from repointer.kinematics import SMPLXKinematics, SyntheticKinematics
from repointer.retrieval import descriptor, retrieve
from repointer.rotations import axis_angle_to_quat, nlerp, quat_to_axis_angle, rotation_matrix
from repointer.warping import apply_hold, hold_weights, piecewise_positions, warp_clip


def query(**kwargs):
    fields = dict(sample_id="query", frames=15, hand=0, target=np.array([.3, .8, 2.]),
                  group="query-group", recording="query-recording", fold=2)
    fields.update(kwargs)
    return Query(**fields)


def motion(frames=15):
    rng = np.random.default_rng(11)
    return Motion(rng.normal(0, .05, (frames, 165)), np.linspace(0, 1, frames)[:, None] * [[1., 2., 3.]])


def donor(name="donor", **kwargs):
    fields = dict(sample_id=name, group=f"{name}-group", recording=f"{name}-recording", fold=1)
    fields.update(kwargs)
    context = query(**fields)
    return Donor(context, motion(context.frames), 7)


def test_motion_normalizes_55_joint_layout_and_copies_inputs(tmp_path):
    poses = np.zeros((5, 55, 3))
    translations = np.ones((5, 3))
    value = Motion(poses, translations)
    poses[:] = 1
    translations[:] = 2
    assert value.poses.shape == (5, 165)
    np.testing.assert_array_equal(value.poses, 0)
    np.testing.assert_array_equal(value.trans, 1)
    assert value.poses.dtype == value.trans.dtype == np.float32
    with pytest.raises(ValueError):
        value.poses[0, 0] = 2
    path = tmp_path / "smplx-motion.npz"
    value.save(path)
    restored = Motion.load(path, expected_frames=5)
    np.testing.assert_array_equal(restored.poses, value.poses)
    np.testing.assert_array_equal(restored.trans, value.trans)
    with pytest.raises(ValueError, match="frame count"):
        Motion.load(path, expected_frames=6)
    with pytest.raises(FileExistsError):
        value.save(path)


@pytest.mark.parametrize("poses,translations", [
    (np.zeros((0, 165)), np.zeros((0, 3))),
    (np.zeros((5, 164)), np.zeros((5, 3))),
    (np.zeros((5, 55, 4)), np.zeros((5, 3))),
    (np.zeros((5, 165), dtype=complex), np.zeros((5, 3))),
    (np.full((5, 165), np.nan), np.zeros((5, 3))),
    (np.zeros((5, 165)), np.zeros((4, 3))),
    (np.zeros((5, 165)), np.full((5, 3), np.inf)),
])
def test_motion_rejects_invalid_schema(poses, translations):
    with pytest.raises(ValueError):
        Motion(poses, translations)


@pytest.mark.parametrize("change", [dict(frames=0), dict(frames=4), dict(frames=5.0), dict(hand=-1), dict(hand=2),
                                    dict(hand=True), dict(target=np.zeros(3)), dict(target=[1, np.nan, 2]),
                                    dict(betas=np.zeros(9)), dict(gender="unknown"), dict(domain="unknown"),
                                    dict(sample_id="../escape"), dict(group=""), dict(recording=""),
                                    dict(clip_start=-1), dict(bbox_size=[1, 1, 1]),
                                    dict(bbox_size=[1, -1, 1], bbox_center=[0, 0, 0])])
def test_query_rejects_invalid_context(change):
    with pytest.raises(ValueError):
        query(**change)


def test_generation_contract_excludes_evaluation_labels():
    assert not {"gt_peak", "released_peak_frame", "hold_start", "hold_end", "audio"} & set(Query.__dataclass_fields__)
    assert ACTIVE_JOINTS[0] == (13, 16, 18, 20)  # left arm per P0C0
    assert ACTIVE_JOINTS[1] == (14, 17, 19, 21)


def test_donor_and_candidate_contracts():
    with pytest.raises(ValueError, match="length mismatch"):
        Donor(query(), motion(5), 2)
    for apex in (0, 14, 7.0):
        with pytest.raises(ValueError, match="interior frame"):
            Donor(query(), motion(), apex)
    for planned in (0, 14):
        with pytest.raises(ValueError, match="interior"):
            Candidate(motion(), planned, "donor", 0., "median")
    for distance in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="distance"):
            Candidate(motion(), 7, "donor", distance, "median")


@pytest.mark.parametrize("excluded", [dict(sample_id="query"), dict(group="query-group"),
                                     dict(recording="query-recording"), dict(fold=2),
                                     dict(hand=1), dict(domain="synthetic")])
def test_retrieval_excludes_each_leakage_and_eligibility_violation(excluded):
    with pytest.raises(ValueError, match="no eligible"):
        retrieve(query(), [donor(**excluded)], 7)


def test_retrieval_is_same_hand_natural_deterministic_and_budgeted():
    near = donor("near")
    tied = donor("tied")
    far = donor("far", target=[-3., -2., .1])
    ranked = retrieve(query(), [far, tied, near], planned_frame=7, top_k=2)
    assert [item[0].context.sample_id for item in ranked] == ["tied", "near"]
    assert [item[1] for item in ranked] == [0., 0.]
    # An official test query has no frozen fold; group/recording rules still apply.
    same_fold = donor("test-allowed", fold=2)
    assert retrieve(replace(query(), fold=-1), [same_fold], 7)[0][0] is same_fold
    with pytest.raises(ValueError, match="budget"):
        retrieve(query(), [near], 7, top_k=0)
    with pytest.raises(ValueError, match="interior"):
        retrieve(query(), [near], 0)
    with pytest.raises(ValueError):
        descriptor(query(), np.nan)


def test_rotation_round_trip_and_proper_matrices():
    angles = np.array([[0., 0., 0.], [1e-10, 0, 0], [.2, -.3, .5], [0, np.pi, 0], [0, 0, 4.]])
    quaternion = axis_angle_to_quat(angles)
    np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-12)
    rotation = rotation_matrix(angles)
    np.testing.assert_allclose(rotation_matrix(quat_to_axis_angle(quaternion)), rotation, atol=1e-12)
    np.testing.assert_allclose(rotation @ rotation.swapaxes(-1, -2), np.broadcast_to(np.eye(3), (5, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-12)


def test_quaternion_interpolation_takes_short_path_across_pi():
    first = axis_angle_to_quat(np.array([0, 0, np.pi - .1]))
    second = axis_angle_to_quat(np.array([0, 0, -np.pi + .1]))
    midpoint = quat_to_axis_angle(nlerp(first, second, .5))
    # The midpoint rotates x to -x, not back to x through zero radians.
    np.testing.assert_allclose(rotation_matrix(midpoint) @ [1, 0, 0], [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(nlerp(first, -first, .5), first, atol=1e-12)


@pytest.mark.parametrize("arguments", [(1, 0, 15, 7), (15, 0, 15, 7), (15, 14, 15, 7),
                                        (15, 7, 15, 15), (15, 7, 15, 7.5)])
def test_warp_rejects_invalid_geometry(arguments):
    with pytest.raises(ValueError):
        piecewise_positions(*arguments)


def test_warp_preserves_apex_endpoints_and_centers_translation():
    source = motion()
    positions = piecewise_positions(15, 7, 23, 9)
    assert positions[0] == 0 and positions[9] == 7 and positions[-1] == 14
    assert np.all(np.diff(positions) > 0)
    result = warp_clip(source, 7, 23, 9)
    assert result.frames == 23
    np.testing.assert_allclose(result.poses[9], source.poses[7], atol=1e-7)
    np.testing.assert_allclose(result.poses[[0, -1]], source.poses[[0, -1]], atol=1e-7)
    np.testing.assert_array_equal(result.trans[9], 0)
    np.testing.assert_allclose(result.trans[-1] - result.trans[0], source.trans[-1] - source.trans[0])


def test_warp_and_hold_match_actual_frozen_functions():
    path = Path(__file__).resolve().parents[1] / "frozen/m1/repointer_b0_p0c_m1_full_fold_v0_2_0.py"
    spec = importlib.util.spec_from_file_location("_repointer_frozen_m1_test", path)
    historical = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = historical
    try:
        spec.loader.exec_module(historical)
        source = motion()
        cleaned = warp_clip(source, 7, 23, 9)
        archived = historical.warp_clip({"pose": source.poses, "trans": source.trans}, 7, 23, 9)
        np.testing.assert_allclose(cleaned.poses, archived["pose"], atol=1e-7)
        np.testing.assert_allclose(cleaned.trans, archived["trans"], atol=3e-7)
        apex = np.full(165, .25, dtype=np.float32)
        held = apply_hold(cleaned, 9, apex, 0, plateau=5, taper=4)
        archived_hold = historical.apply_hold(archived["pose"], 9, apex, 0, plateau=5, taper=4)
        np.testing.assert_allclose(held.poses, archived_hold, atol=1e-7)
    finally:
        del sys.modules[spec.name]


@pytest.mark.parametrize("hand", [0, 1])
def test_taper_preserves_inactive_joints_and_translation(hand):
    source = motion(31)
    apex = np.full(165, .4, dtype=np.float32)
    result = apply_hold(source, peak=15, optimized_apex=apex, hand=hand, plateau=5, taper=8)
    active = [3 * joint + axis for joint in ACTIVE_JOINTS[hand] for axis in range(3)]
    inactive = sorted(set(range(165)) - set(active))
    np.testing.assert_array_equal(result.poses[:, inactive], source.poses[:, inactive])
    np.testing.assert_array_equal(result.trans, source.trans)
    np.testing.assert_allclose(result.poses[13:18, active], np.broadcast_to(apex[active], (5, 12)), atol=1e-7)
    np.testing.assert_allclose(result.poses[[0, -1]], source.poses[[0, -1]], atol=1e-7)
    weights = hold_weights(31, 15, plateau=5, taper=8)
    assert weights[15] == 1 and weights[25] == 0
    np.testing.assert_array_equal(weights, weights[::-1])
    assert np.all(np.diff(weights[17:]) <= 0)


def test_invalid_hold_context():
    with pytest.raises(ValueError):
        apply_hold(motion(), 7, np.zeros(165), hand=2)
    with pytest.raises(ValueError):
        apply_hold(motion(), 7, np.zeros(164), hand=0)
    with pytest.raises(ValueError):
        hold_weights(15, 15)


def test_synthetic_kinematics_is_shape_dependent_and_translation_equivariant():
    engine = SyntheticKinematics()
    source = Motion(np.zeros((5, 165)), np.zeros((5, 3)))
    neutral = engine.joints(source, np.zeros(10))
    shaped = engine.joints(source, np.r_[2., np.zeros(9)])
    assert neutral.shape == (5, 55, 3)
    np.testing.assert_allclose(shaped[:, 20], neutral[:, 20] * (1 + .03 * np.tanh(2)), atol=1e-7)
    assert np.linalg.norm(shaped[0, 20] - neutral[0, 20]) > .01
    translation = np.tile([1., 2., 3.], (5, 1))
    shifted = engine.joints(Motion(source.poses, translation), np.zeros(10))
    np.testing.assert_allclose(shifted - neutral, np.broadcast_to(translation[:, None], neutral.shape), atol=2e-7)


@pytest.mark.parametrize("hand", [0, 1])
def test_synthetic_ik_improves_target_ray_and_preserves_inactive_rotations(hand):
    engine = SyntheticKinematics()
    context = query(hand=hand, betas=np.r_[1., np.zeros(9)])
    pose = np.zeros(165, dtype=np.float32)
    optimized = optimize_apex(engine, pose, context, steps=25, lr=.1)

    def cosine(value):
        joints = engine.joints(Motion(value[None], np.zeros((1, 3))), context.betas)[0]
        forearm = joints[20 + hand] - joints[18 + hand]
        target = joints[0] + context.target - joints[18 + hand]
        return np.dot(forearm, target) / (np.linalg.norm(forearm) * np.linalg.norm(target))

    assert cosine(optimized) > .98
    assert cosine(optimized) > cosine(pose) + .5
    active = {3 * joint + axis for joint in ACTIVE_JOINTS[hand] for axis in range(3)}
    inactive = sorted(set(range(165)) - active)
    np.testing.assert_array_equal(optimized[inactive], pose[inactive])
    np.testing.assert_array_equal(pose, 0)
    assert np.isfinite(optimized).all()


@pytest.mark.parametrize("change", [dict(steps=0), dict(lr=0), dict(lr=np.nan), dict(max_delta=-1)])
def test_ik_rejects_invalid_optimizer_parameters(change):
    with pytest.raises(ValueError):
        optimize_apex(SyntheticKinematics(), np.zeros(165), query(), **change)


def test_smplx_adapter_passes_correct_pose_slices_without_assets():
    """Exercise the licensed adapter boundary with only an ndarray tensor stub."""
    class Tensor:
        def __init__(self, value):
            self.value = value
            self.dtype = value.dtype
            self.device = "cpu"

        def __len__(self):
            return len(self.value)

        def __getitem__(self, key):
            return self.value[key]

    calls = {}
    expected_joints = np.zeros((2, 55, 3))

    def model(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(joints=expected_joints)

    adapter = SMPLXKinematics.__new__(SMPLXKinematics)
    adapter.torch = SimpleNamespace(zeros=lambda shape, dtype, device: np.zeros(shape, dtype=dtype))
    adapter.model = lambda gender: model
    poses = np.arange(330, dtype=np.float32).reshape(2, 165)
    translations = np.ones((2, 3))
    betas = np.zeros((2, 10))
    result = adapter.forward_torch(Tensor(poses), translations, betas, "neutral")
    assert result is expected_joints
    expected = {"global_orient": (0, 3), "body_pose": (3, 66), "jaw_pose": (66, 69),
                "leye_pose": (69, 72), "reye_pose": (72, 75),
                "left_hand_pose": (75, 120), "right_hand_pose": (120, 165)}
    for name, (start, end) in expected.items():
        np.testing.assert_array_equal(calls[name], poses[:, start:end])
    assert calls["transl"] is translations and calls["betas"] is betas
    np.testing.assert_array_equal(calls["expression"], np.zeros((2, 10)))
    assert calls["return_verts"] is False


@pytest.mark.parametrize("change", [dict(fold=-2), dict(fold=1.0), dict(fold=True),
                                     dict(group=123), dict(group="  "), dict(recording=["id"]),
                                     dict(recording="\t"), dict(clip_start=True), dict(clip_start=[0.])])
def test_query_rejects_noninteger_fold_and_nonstring_leakage_keys(change):
    with pytest.raises(ValueError):
        query(**change)


@pytest.mark.parametrize("planned", [7.5, True, np.bool_(True)])
def test_candidate_requires_an_integer_frame(planned):
    with pytest.raises(ValueError, match="interior integer"):
        Candidate(motion(), planned, "donor", 0., "median")


@pytest.mark.parametrize("distance", [True, [0.], "0", complex(0, 1)])
def test_candidate_requires_real_scalar_distance(distance):
    with pytest.raises(ValueError):
        Candidate(motion(), 7, "donor", distance, "median")


@pytest.mark.parametrize("expected", [True, 5.0, 0, -1])
def test_motion_frame_expectation_requires_positive_integer(tmp_path, expected):
    path = tmp_path / "motion.npz"
    motion().save(path)
    with pytest.raises(ValueError, match="positive integer"):
        Motion.load(path, expected_frames=expected)


@pytest.mark.parametrize("arguments", [(True, 0, 0, 0), (15, 7.5, 5, 8), (15, True, 5, 8),
                                        (15, 7, 5.5, 8), (15, 7, 5, True), (15, 7, False, 8)])
def test_hold_counts_require_integers(arguments):
    with pytest.raises(ValueError):
        hold_weights(*arguments)


def test_hold_rejects_boolean_hand_and_nonreal_apex():
    with pytest.raises(ValueError):
        apply_hold(motion(), 7, np.zeros(165), True)
    with pytest.raises(ValueError):
        apply_hold(motion(), 7, np.zeros(165, dtype=complex), 0)


@pytest.mark.parametrize("change", [dict(top_k=True), dict(top_k=1.5), dict(planned_frame=True), dict(planned_frame=7.5)])
def test_retrieval_counts_require_integers(change):
    fields = dict(planned_frame=7, top_k=1)
    fields.update(change)
    with pytest.raises(ValueError):
        retrieve(query(), [donor()], **fields)


@pytest.mark.parametrize("change", [dict(steps=True), dict(steps=1.5), dict(lr=True), dict(lr=[.1]),
                                    dict(max_delta=True), dict(max_delta="1")])
def test_ik_rejects_noninteger_steps_and_nonreal_scalars(change):
    with pytest.raises(ValueError):
        optimize_apex(SyntheticKinematics(), np.zeros(165), query(), **change)


@pytest.mark.parametrize("betas", [np.zeros(9), np.zeros((1, 10)), np.full(10, np.nan),
                                   np.full(10, np.inf), np.zeros(10, dtype=complex)])
def test_direct_kinematics_calls_validate_shape_betas(betas):
    for engine in (SyntheticKinematics(), SMPLXKinematics.__new__(SMPLXKinematics)):
        # Invalid input fails before any model/device access, so no assets needed.
        with pytest.raises(ValueError, match="betas"):
            engine.joints(motion(), betas)


def test_direct_kinematics_calls_validate_gender():
    for engine in (SyntheticKinematics(), SMPLXKinematics.__new__(SMPLXKinematics)):
        with pytest.raises(ValueError, match="gender"):
            engine.joints(motion(), np.zeros(10), gender="unknown")


@pytest.mark.parametrize("batch_size", [True, 1.5, 0])
def test_smplx_adapter_batch_size_requires_positive_integer(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        SMPLXKinematics(tmp_path, batch_size=batch_size)
