"""P1A contracts, feature parity and optional learned-model round trip."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import repointer.timing as timing

from repointer.timing import (
    TimedWord, TimingInput, WordPhaseNet, apex_from_distribution,
    domain_hand_medians, encode_transcript, fit_synthetic_prior, fit_timing,
    load_checkpoint, load_synthetic_prior, prior_distribution, ratio_to_frame,
    save_checkpoint, synthetic_training_fixture, token_id, token_normalize,
)


def query(words=(), **kwargs):
    options = dict(words=words, frames=60, hand=0, domain="natural", target=(1, 0, 2), fixed_peak=48)
    options.update(kwargs)
    return TimingInput(**options)


def test_transcript_features_match_frozen_preprocessing(tmp_path):
    """Numerical comparison against the actual archived feature function."""
    path = Path(__file__).resolve().parents[1] / "frozen/timing/repointer_p1a_wordphase_timing_oof_v0_1_0.py"
    spec = importlib.util.spec_from_file_location("_repointer_frozen_timing_test", path)
    historical = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = historical
    try:
        spec.loader.exec_module(historical)
        transcript = tmp_path / "invented.tsv"
        transcript.write_text("2.0 2.1 out\n3.1 3.3 Please,\n3.6 3.9 point\n4.1 4.5 THERE!\n5.5 5.8 out\n", encoding="utf-8")
        record = dict(sample_id="invented", recording_id="invented", leakage_group_id="g", fold=0,
                      domain="natural", hand=0, clip_length=60, released_peak_frame=30,
                      hold_start=28, hold_end=32, global_start=90,
                      target_pelvis_x=1, target_pelvis_y=0, target_pelvis_z=2)
        old = historical.make_example(record, {"transcript_path": str(transcript), "planned_peak": 48}, {}, {}, 2048, 64)
        words = (TimedWord(2, 2.1, "out"), TimedWord(3.1, 3.3, "Please,"), TimedWord(3.6, 3.9, "point"),
                 TimedWord(4.1, 4.5, "THERE!"), TimedWord(5.5, 5.8, "out"))
        new = encode_transcript(query(words, clip_start=3))
        np.testing.assert_array_equal(new.token_ids, old.token_ids)
        np.testing.assert_allclose(new.word_features, old.word_features, atol=1e-7)
        np.testing.assert_allclose(new.base_clip_features, old.base_clip_features, atol=1e-7)
        assert new.word_count == old.word_count == 3
    finally:
        del sys.modules[spec.name]


def test_missing_words_and_deterministic_hashing():
    encoded = encode_transcript(query())
    np.testing.assert_array_equal(encoded.token_ids, [1])
    np.testing.assert_array_equal(encoded.word_features, np.zeros((1, 10)))
    assert encoded.base_clip_features.shape == (15,)
    assert encoded.word_count == 0
    assert token_normalize(" THERE!! ") == "there"
    assert token_id("there") == token_id(token_normalize("There!"))
    assert 2 <= token_id("there") < 2050


def test_timestamp_seconds_and_apex_frame_intervals_differ():
    encoded = encode_transcript(query((TimedWord(0, 1, "this"),)))
    # A 60-frame, 30-fps transcript has duration 2 s. Frame ratios use 59.
    assert encoded.word_features[0, 1] == 0.5
    assert encoded.base_clip_features[-1] == pytest.approx(48 / 59)
    assert ratio_to_frame(0, 60) == 2
    assert ratio_to_frame(1, 60) == 57
    assert ratio_to_frame(0.5, 61) == 30
    assert ratio_to_frame(0.5, 5) == 2


def test_max_words_retains_broad_coverage():
    words = [TimedWord(i / 30, (i + 0.5) / 30, str(i)) for i in range(60)]
    encoded = encode_transcript(query(words), max_words=4)
    assert encoded.word_count == 4
    assert encoded.word_features[0, 0] == 0
    assert encoded.word_features[-1, 0] > 0.95


@pytest.mark.parametrize("change", [dict(frames=4), dict(hand=2), dict(domain="other"), dict(target=[0, 0, 0]),
                                     dict(target=[1, 2]), dict(fixed_peak=60), dict(fps=0), dict(clip_start=-1)])
def test_invalid_query_context(change):
    with pytest.raises(ValueError):
        query(**change)


@pytest.mark.parametrize("probability", [np.ones(63), np.zeros(64), np.ones(64), np.full(64, np.nan), np.r_[-1, np.ones(63) / 31]])
def test_distribution_shape_and_probability_validation(probability):
    with pytest.raises(ValueError):
        apex_from_distribution(probability, 60)


def test_distribution_conversion_uses_expectation():
    probability = np.zeros(64)
    probability[10], probability[50] = 0.25, 0.75
    assert apex_from_distribution(probability, 64) == 40
    assert prior_distribution(0.5).sum() == pytest.approx(1)


def test_prior_training_and_json_round_trip(tmp_path):
    inputs, labels = synthetic_training_fixture(n=8, seed=4)
    model = fit_synthetic_prior(inputs, labels)
    probabilities = model.predict(inputs)
    assert probabilities.shape == (8, 64)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    model.save(tmp_path / "synthetic.json")
    restored = load_synthetic_prior(tmp_path / "synthetic.json")
    assert restored.backend == "synthetic-prior"
    np.testing.assert_array_equal(restored.predict(inputs), probabilities)
    assert restored.predict([]).shape == (0, 64)
    assert domain_hand_medians([query()], [15])["natural_hand0"] == pytest.approx(15 / 59)
    assert domain_hand_medians([query()], [15])["synthetic_hand1"] == 0.5


def test_labels_are_separate_and_validated():
    inputs, labels = synthetic_training_fixture()
    assert "gt_peak" not in TimingInput.__dataclass_fields__
    assert "hold_start" not in TimingInput.__dataclass_fields__
    with pytest.raises(ValueError):
        fit_synthetic_prior(inputs, labels[:-1])
    with pytest.raises(ValueError):
        fit_synthetic_prior(inputs, [-1] * len(inputs))
    with pytest.raises(ValueError):
        fit_synthetic_prior([], [])


def test_invalid_word_interval():
    with pytest.raises(ValueError):
        TimedWord(1, 0, "point")
    with pytest.raises(ValueError):
        TimedWord(0, float("nan"), "point")


def test_optional_wordphase_dependency_error():
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("PyTorch installed; learned round trip covers this environment")
    with pytest.raises(RuntimeError, match="optional timing dependency"):
        WordPhaseNet()


def test_wordphase_training_and_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch", minversion="2.6")
    # Small fixtures test gradients and persistence, not benchmark accuracy.
    torch.set_num_threads(1)
    inputs, labels = synthetic_training_fixture(n=4, seed=7)
    model = fit_timing(inputs, labels, epochs=1, buckets=32, seed=5)
    assert len(model.history) == 1 and np.isfinite(model.history[0])
    probability = model.predict(inputs)
    assert probability.shape == (4, 64)
    np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=1e-5)
    assert model.network.net.gru.bidirectional
    save_checkpoint(model, tmp_path / "timing.pt")
    restored = load_checkpoint(tmp_path / "timing.pt")
    np.testing.assert_array_equal(restored.predict(inputs), probability)
    original = (tmp_path / "timing.pt").read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(model, tmp_path / "timing.pt")
    assert (tmp_path / "timing.pt").read_bytes() == original


@pytest.mark.parametrize("change", [dict(frames=True), dict(frames=60.0), dict(hand=True), dict(hand=0.0),
                                    dict(fixed_peak=True), dict(fixed_peak=48.), dict(fps=True), dict(fps=[30]),
                                    dict(clip_start=False), dict(clip_start=[0]), dict(words=None), dict(words="point"),
                                    dict(target=np.ones(3, dtype=complex))])
def test_timing_query_requires_integer_counts_and_real_context(change):
    with pytest.raises(ValueError):
        query(**change)


@pytest.mark.parametrize("start,end,text", [(True, 1., "point"), ([0.], 1., "point"), (0., False, "point"), (0., 1., None)])
def test_word_times_require_numeric_scalars(start, end, text):
    with pytest.raises(ValueError):
        TimedWord(start, end, text)


@pytest.mark.parametrize("ratio", [True, [0.5], "0.5", complex(.5, 0)])
def test_apex_ratio_requires_real_scalar(ratio):
    with pytest.raises(ValueError):
        ratio_to_frame(ratio, 60)
    with pytest.raises(ValueError):
        prior_distribution(ratio)


def test_prior_distribution_handles_extremely_narrow_uncertainty():
    probability = prior_distribution(.5, sigma_bins=np.nextafter(0., 1.))
    assert np.isfinite(probability).all()
    assert probability.sum() == pytest.approx(1)
    assert probability[31] == probability[32] == .5
    with pytest.raises(ValueError):
        prior_distribution(.5, sigma_bins=True)


@pytest.mark.parametrize("change", [dict(max_words=True), dict(max_words=3.5), dict(max_words=0),
                                    dict(buckets=True), dict(buckets=2048.)])
def test_transcript_encoder_counts_require_integers(change):
    with pytest.raises(ValueError):
        encode_transcript(query(), **change)


@pytest.mark.parametrize("change", [dict(buckets=True), dict(word_dim=10.), dict(clip_dim=16.), dict(bins=64.)])
def test_network_dimensions_validate_before_optional_torch_import(change):
    with pytest.raises(ValueError):
        WordPhaseNet(**change)


@pytest.mark.parametrize("change", [dict(epochs=True), dict(epochs=1.5), dict(batch_size=True), dict(batch_size=4.),
                                    dict(seed=True), dict(seed=-1), dict(learning_rate=True), dict(learning_rate=[.1]),
                                    dict(buckets=True), dict(max_words=3.5)])
def test_training_configuration_rejects_noninteger_counts_without_torch(change):
    inputs, labels = synthetic_training_fixture()
    with pytest.raises(ValueError):
        fit_timing(inputs, labels, **change)


@pytest.mark.parametrize("change", [dict(n=True), dict(n=3.), dict(seed=True), dict(seed=-1)])
def test_fixture_counts_require_integers(change):
    with pytest.raises(ValueError):
        synthetic_training_fixture(**change)


def test_boolean_labels_and_nonnumeric_distributions_are_rejected():
    inputs, labels = synthetic_training_fixture()
    with pytest.raises(ValueError):
        fit_synthetic_prior(inputs, [True] * len(labels))
    with pytest.raises(ValueError):
        fit_synthetic_prior([None], [1])
    with pytest.raises(ValueError):
        apex_from_distribution(np.ones(64, dtype=complex) / 64, 60)
    with pytest.raises(ValueError):
        token_id("point", buckets=True)
    with pytest.raises(ValueError):
        token_id(None)


def test_synthetic_prior_never_overwrites_training_artifact(tmp_path):
    inputs, labels = synthetic_training_fixture()
    model = fit_synthetic_prior(inputs, labels)
    path = tmp_path / "existing.json"
    model.save(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        model.save(path)
    assert path.read_bytes() == original
    document = json.loads(original)
    document["medians"]["natural_hand0"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_synthetic_prior(invalid)


def test_checkpoint_creation_is_exclusive_without_requiring_torch(tmp_path, monkeypatch):
    """Check file-open policy with a serialization stub, not model execution."""
    calls = []

    def save(payload, stream):
        calls.append(payload)
        stream.write(b"invented checkpoint bytes")

    monkeypatch.setattr(timing, "_torch", lambda: SimpleNamespace(from_numpy=lambda value: value, save=save))
    model = SimpleNamespace(network=SimpleNamespace(net=SimpleNamespace(state_dict=lambda: {})),
                            word_mean=np.zeros(10), word_std=np.ones(10), clip_mean=np.zeros(16), clip_std=np.ones(16),
                            medians={}, buckets=32, max_words=64, history=())
    path = tmp_path / "protected.pt"
    save_checkpoint(model, path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(model, path)
    assert len(calls) == 1
    assert path.read_bytes() == original


def test_checkpoint_loader_always_requests_restricted_loading(tmp_path, monkeypatch):
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        return {"schema": "unsupported"}

    monkeypatch.setattr(timing, "_torch", lambda: SimpleNamespace(load=load))
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(tmp_path / "unused.pt")
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_optional_backend_rejects_torch_before_26(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.5.1"))
    with pytest.raises(RuntimeError, match="PyTorch >= 2.6"):
        WordPhaseNet()
