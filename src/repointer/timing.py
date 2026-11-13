"""Timed-transcript encoding and WordPhaseNet (P1A).

The feature definitions and network follow frozen P1A v0.1.0.  No waveform,
motion, released apex or hold labels enter :class:`TimingInput`. Training a
small fixture validates code paths; it does not recreate the historical folds.
PyTorch is an optional dependency, imported only for learned model operations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from .contracts import finite_array, finite_scalar

BINS = 64
HASH_BUCKETS = 2048
DEICTIC = {"this", "that", "these", "those", "here", "there", "it", "one"}
SPATIAL = {"left", "right", "front", "behind", "back", "above", "below", "under", "over", "next", "near", "beside", "between", "on", "in", "at", "corner", "side"}
FUNCTION = {"the", "a", "an", "to", "of", "is", "are", "was", "were", "and", "or", "with", "you", "can", "could", "please"}


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_priors(medians: Any) -> dict[str, float]:
    keys = {f"{domain}_hand{hand}" for domain in ("natural", "synthetic") for hand in (0, 1)}
    if not isinstance(medians, dict) or set(medians) != keys:
        raise ValueError("timing prior must contain all domain/hand priors")
    result = {key: finite_scalar(value, "median prior") for key, value in medians.items()}
    if any(not 0 <= value <= 1 for value in result.values()):
        raise ValueError("timing median priors must lie in [0,1]")
    return result


@dataclass(frozen=True)
class TimedWord:
    """Word interval in seconds on the transcript timeline."""

    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        start = finite_scalar(self.start, "word start")
        end = finite_scalar(self.end, "word end")
        if end < start:
            raise ValueError("word intervals must be finite and end >= start")
        if not isinstance(self.text, str):
            raise ValueError("word text must be a string")


@dataclass(frozen=True)
class TimingInput:
    """Disclosed query context; hand 0 is left and hand 1 is right.

    ``target`` is the pelvis-relative target vector. ``fixed_peak`` is the
    baseline planned apex, not a released label. Natural recording transcripts
    can use ``clip_start``; fixture and clip-relative transcripts use zero.
    """

    words: Sequence[TimedWord]
    frames: int
    hand: int
    domain: str
    target: Sequence[float]
    fixed_peak: int
    clip_start: float = 0.0
    fps: float = 30.0

    def __post_init__(self) -> None:
        _integer(self.frames, "timing clip length", 5)
        if type(self.hand) is not int or self.hand not in (0, 1) or not isinstance(self.domain, str) or self.domain not in ("natural", "synthetic"):
            raise ValueError("hand must be 0 or 1; domain must be natural or synthetic")
        if type(self.fixed_peak) is not int or not 0 <= self.fixed_peak < self.frames:
            raise ValueError("fixed_peak must be an in-range frame index")
        target = finite_array(self.target, (3,), "timing target")
        if np.linalg.norm(target.astype(float)) < 1e-8:
            raise ValueError("target must be a finite nonzero pelvis-relative 3-vector")
        if finite_scalar(self.fps, "fps") <= 0 or finite_scalar(self.clip_start, "clip_start") < 0:
            raise ValueError("fps must be positive and clip_start nonnegative")
        if not isinstance(self.words, Sequence) or isinstance(self.words, (str, bytes)) or not all(isinstance(word, TimedWord) for word in self.words):
            raise ValueError("words must contain TimedWord values")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "words", tuple(self.words))


@dataclass(frozen=True)
class EncodedTranscript:
    token_ids: np.ndarray
    word_features: np.ndarray
    base_clip_features: np.ndarray
    word_count: int


def token_normalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("token text must be a string")
    return re.sub(r"^[^\w']+|[^\w']+$", "", text.strip().lower()) or "<empty>"


def token_id(token: str, buckets: int = HASH_BUCKETS) -> int:
    """Stable hashing: PAD=0, NO_WORD=1, words=2..buckets+1."""
    _integer(buckets, "hash buckets")
    if not isinstance(token, str):
        raise ValueError("token must be a string")
    if token == "<no_word>":
        return 1
    digest = hashlib.blake2b(token.encode("utf-8", errors="replace"), digest_size=8).digest()
    return 2 + int.from_bytes(digest, "little") % buckets


def encode_transcript(query: TimingInput, *, buckets: int = HASH_BUCKETS, max_words: int = 64) -> EncodedTranscript:
    """Encode the historical 10 word features and 15 query features.

    Timestamp normalization uses T/fps seconds, while apex normalization uses
    T-1 frame intervals. Keep this distinction when reading model outputs.
    """
    _integer(max_words, "max_words")
    _integer(buckets, "hash buckets")
    if not isinstance(query, TimingInput):
        raise ValueError("transcript encoding requires TimingInput")
    seconds = query.frames / query.fps
    start = query.clip_start
    selected = []
    for word in query.words:
        a, b = word.start, word.end
        if b >= start - 0.05 and a <= start + seconds + 0.05:
            ar, br = max(0.0, a - start), min(seconds, b - start)
            if br >= -0.05 and ar <= seconds + 0.05:
                selected.append((max(0.0, ar), max(0.0, min(seconds, br)), token_normalize(word.text)))
    selected.sort(key=lambda word: (word[0], word[1]))
    if len(selected) > max_words:
        indices = np.linspace(0, len(selected) - 1, max_words).round().astype(int)
        selected = [selected[int(index)] for index in indices]
    tokens, features, pauses, durations = [], [], [], []
    count = len(selected)
    for index, (a, b, word) in enumerate(selected):
        before = max(0.0, a - (selected[index - 1][1] if index else 0.0))
        after = max(0.0, (selected[index + 1][0] if index + 1 < count else seconds) - b)
        duration = max(0.0, b - a)
        durations.append(duration)
        pauses.extend([before, after])
        tokens.append(token_id(word, buckets))
        features.append([
            a / seconds, b / seconds, duration / seconds, before / seconds, after / seconds,
            index / max(1, count - 1), min(len(word), 24) / 24.0,
            float(word in DEICTIC), float(word in SPATIAL), float(word in FUNCTION),
        ])
    if not tokens:
        tokens, features = [1], [[0.0] * 10]
    target = np.asarray(query.target, dtype=np.float64)
    distance = float(np.linalg.norm(target))
    unit = target / distance
    first = selected[0][0] / seconds if selected else 0.0
    last = selected[-1][1] / seconds if selected else 0.0
    base = np.asarray([
        float(query.domain == "natural"), float(query.hand), query.frames / 180.0,
        np.log(max(distance, 1e-4)), *unit,
        count / max_words, first, last, max(0.0, last - first),
        np.mean(durations) / seconds if durations else 0.0,
        np.mean(pauses) / seconds if pauses else 0.0,
        np.max(pauses) / seconds if pauses else 0.0,
        query.fixed_peak / (query.frames - 1),
    ], dtype=np.float32)
    return EncodedTranscript(np.asarray(tokens, dtype=np.int64), np.asarray(features, dtype=np.float32), base, count)


def validate_distribution(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities)
    if values.shape != (BINS,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("apex distribution must contain 64 finite nonnegative values")
    values = values.astype(np.float64)
    total = float(values.sum())
    if total <= 0 or not np.isclose(total, 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("apex distribution must sum to one")
    return values / total


def ratio_to_frame(ratio: float, frames: int) -> int:
    """Historical rounded apex conversion with two-frame endpoint margins."""
    _integer(frames, "frames", 5)
    if not 0 <= finite_scalar(ratio, "apex ratio") <= 1:
        raise ValueError("apex ratio must lie in [0,1]")
    return int(np.clip(round(float(ratio) * (frames - 1)), 2, frames - 3))


def apex_from_distribution(probabilities: np.ndarray, frames: int) -> int:
    return ratio_to_frame(float(validate_distribution(probabilities) @ np.linspace(0, 1, BINS)), frames)


def prior_distribution(ratio: float, *, sigma_bins: float = 1.5) -> np.ndarray:
    """Explicit median/fixture prior, not a learned WordPhaseNet prediction."""
    ratio = finite_scalar(ratio, "apex ratio")
    sigma_bins = finite_scalar(sigma_bins, "sigma_bins")
    if not 0 <= ratio <= 1 or sigma_bins <= 0:
        raise ValueError("ratio must lie in [0,1] and sigma_bins must be positive")
    distance = (np.arange(BINS) - ratio * (BINS - 1)) ** 2
    # Subtract the minimum squared distance before scaling. Even vanishingly
    # small positive sigma then leaves at least one finite maximum-probability bin.
    with np.errstate(over="ignore", under="ignore"):
        result = np.exp(-0.5 * ((distance - distance.min()) / sigma_bins) / sigma_bins)
    return result / result.sum()


def domain_hand_medians(inputs: Sequence[TimingInput], apex_frames: Sequence[int]) -> dict[str, float]:
    """Fit priors from training labels only; never pass held-out labels here."""
    _validate_labels(inputs, apex_frames)
    result = {}
    for domain in ("natural", "synthetic"):
        for hand in (0, 1):
            values = [apex / (query.frames - 1) for query, apex in zip(inputs, apex_frames) if (query.domain, query.hand) == (domain, hand)]
            result[f"{domain}_hand{hand}"] = float(np.median(values)) if values else 0.5
    return result


@dataclass(frozen=True)
class SyntheticPriorModel:
    """Small NumPy backend for integration fixtures, not WordPhaseNet.

    Fitting median apex ratios exercises label-to-training-artifact and
    inference contracts without PyTorch or protected assets. The generated
    Gaussian distributions must not be presented as learned P1A output.
    """

    medians: dict[str, float]
    backend: str = "synthetic-prior"

    def __post_init__(self) -> None:
        if self.backend != "synthetic-prior":
            raise ValueError("SyntheticPriorModel backend must be synthetic-prior")
        object.__setattr__(self, "medians", _validate_priors(self.medians))

    def predict(self, inputs: Sequence[TimingInput]) -> np.ndarray:
        _validate_inputs(inputs)
        if not inputs:
            return np.empty((0, BINS), dtype=np.float32)
        return np.stack([prior_distribution(self.medians[f"{query.domain}_hand{query.hand}"]) for query in inputs])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps({
            "schema": "repointer.synthetic-prior.v1", "backend": self.backend,
            "bins": BINS, "medians": self.medians, "paper_parity": False,
        }, indent=2, allow_nan=False) + "\n"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)


def fit_synthetic_prior(inputs: Sequence[TimingInput], apex_frames: Sequence[int]) -> SyntheticPriorModel:
    return SyntheticPriorModel(domain_hand_medians(inputs, apex_frames))


def load_synthetic_prior(path: str | Path) -> SyntheticPriorModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("schema") != "repointer.synthetic-prior.v1"
            or data.get("backend") != "synthetic-prior" or type(data.get("bins")) is not int or data.get("bins") != BINS):
        raise ValueError("unsupported synthetic prior schema")
    return SyntheticPriorModel(_validate_priors(data.get("medians")))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("WordPhaseNet requires the optional timing dependency: pip install -e '.[timing]'") from exc
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < (2, 6):
        raise RuntimeError("WordPhaseNet requires PyTorch >= 2.6 for restricted checkpoint loading")
    return torch


class WordPhaseNet:
    """Historical module names and shapes, with a lazily constructed ``net``."""

    def __init__(self, buckets: int = HASH_BUCKETS, word_dim: int = 10, clip_dim: int = 16, bins: int = BINS):
        _integer(buckets, "hash buckets")
        if any(type(value) is not int for value in (word_dim, clip_dim, bins)) or word_dim != 10 or clip_dim != 16 or bins != BINS:
            raise ValueError("WordPhaseNet requires positive buckets, 10 word features, 16 clip features, and 64 bins")
        torch = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(buckets + 2, 24, padding_idx=0)
                self.word_proj = nn.Sequential(nn.Linear(24 + word_dim, 64), nn.SiLU(), nn.LayerNorm(64))
                self.gru = nn.GRU(64, 64, num_layers=1, batch_first=True, bidirectional=True)
                self.att = nn.Linear(128, 1)
                self.clip_proj = nn.Sequential(nn.Linear(clip_dim, 64), nn.SiLU(), nn.LayerNorm(64))
                self.head = nn.Sequential(nn.Linear(192, 128), nn.SiLU(), nn.Dropout(0.10), nn.Linear(128, bins))

            def forward(self, tok: Any, wf: Any, mask: Any, cf: Any) -> Any:
                x = self.word_proj(torch.cat([self.emb(tok), wf], dim=-1))
                x, _ = self.gru(x)
                attention = self.att(x).squeeze(-1).masked_fill(~mask, -1e4)
                pooled = torch.sum(x * torch.softmax(attention, dim=1)[:, :, None], dim=1)
                return self.head(torch.cat([pooled, self.clip_proj(cf)], dim=-1))

        self.net = Net()


@dataclass
class TimingModel:
    network: WordPhaseNet
    word_mean: np.ndarray
    word_std: np.ndarray
    clip_mean: np.ndarray
    clip_std: np.ndarray
    medians: dict[str, float]
    buckets: int = HASH_BUCKETS
    max_words: int = 64
    history: tuple[float, ...] = ()

    def _batch(self, inputs: Sequence[TimingInput]) -> tuple[Any, ...]:
        torch = _torch()
        encoded = [encode_transcript(query, buckets=self.buckets, max_words=self.max_words) for query in inputs]
        length = max(len(item.token_ids) for item in encoded)
        tokens = np.zeros((len(inputs), length), dtype=np.int64)
        words = np.zeros((len(inputs), length, 10), dtype=np.float32)
        mask = np.zeros((len(inputs), length), dtype=bool)
        clips = np.zeros((len(inputs), 16), dtype=np.float32)
        for i, (query, item) in enumerate(zip(inputs, encoded)):
            n = len(item.token_ids)
            tokens[i, :n] = item.token_ids
            words[i, :n] = (item.word_features - self.word_mean) / self.word_std
            mask[i, :n] = True
            raw = np.r_[item.base_clip_features, self.medians[f"{query.domain}_hand{query.hand}"]]
            clips[i] = (raw - self.clip_mean) / self.clip_std
        return tuple(torch.from_numpy(array) for array in (tokens, words, mask, clips))

    def predict(self, inputs: Sequence[TimingInput]) -> np.ndarray:
        _validate_inputs(inputs)
        if not inputs:
            return np.empty((0, BINS), dtype=np.float32)
        torch = _torch()
        self.network.net.eval()
        with torch.no_grad():
            return torch.softmax(self.network.net(*self._batch(inputs)), dim=1).cpu().numpy()


def _validate_inputs(inputs: Sequence[TimingInput]) -> None:
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)) or not all(isinstance(query, TimingInput) for query in inputs):
        raise ValueError("timing inputs must be a sequence of TimingInput values")


def _validate_labels(inputs: Sequence[TimingInput], labels: Sequence[int]) -> None:
    _validate_inputs(inputs)
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or not inputs or len(inputs) != len(labels):
        raise ValueError("training inputs and labels must be nonempty with equal length")
    for query, apex in zip(inputs, labels):
        if type(apex) is not int or not 0 <= apex < query.frames:
            raise ValueError("training apex labels must be in-range integer frames")


def fit_timing(inputs: Sequence[TimingInput], apex_frames: Sequence[int], *, epochs: int = 2, seed: int = 0,
               buckets: int = HASH_BUCKETS, max_words: int = 64, batch_size: int = 128,
               learning_rate: float = 0.002) -> TimingModel:
    """Train one CPU model; the caller must supply a training-only partition.

    Matches the frozen AdamW/soft-target plus 0.5 SmoothL1 objective. Historical
    reproduction additionally requires frozen five-fold groups and acceptance
    artifacts, none of which a fixture training run can provide.
    """
    _validate_labels(inputs, apex_frames)
    _integer(epochs, "epochs")
    _integer(batch_size, "batch_size")
    _integer(seed, "seed", 0)
    if finite_scalar(learning_rate, "learning_rate") <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    encoded = [encode_transcript(query, buckets=buckets, max_words=max_words) for query in inputs]
    medians = domain_hand_medians(inputs, apex_frames)
    word = np.concatenate([item.word_features for item in encoded]).astype(np.float64)
    clip = np.stack([np.r_[item.base_clip_features, medians[f"{query.domain}_hand{query.hand}"]] for query, item in zip(inputs, encoded)])
    wm, ws, cm, cs = word.mean(0), word.std(0), clip.mean(0), clip.std(0)
    ws[ws < 1e-6], cs[cs < 1e-6] = 1.0, 1.0
    torch = _torch()
    torch.manual_seed(seed)
    model = TimingModel(WordPhaseNet(buckets), *(array.astype(np.float32) for array in (wm, ws, cm, cs)), medians, buckets, max_words)
    optimizer = torch.optim.AdamW(model.network.net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.05)
    rng = np.random.default_rng(seed)
    history = []
    centers = torch.linspace(0, 1, BINS)
    for _epoch in range(epochs):
        model.network.net.train()
        total = 0.0
        indices = rng.permutation(len(inputs))
        for start in range(0, len(inputs), batch_size):
            ids = indices[start:start + batch_size]
            queries = [inputs[int(i)] for i in ids]
            ratio = torch.tensor([apex_frames[int(i)] / (inputs[int(i)].frames - 1) for i in ids], dtype=torch.float32)
            logits = model.network.net(*model._batch(queries))
            soft = torch.exp(-0.5 * ((centers[None, :] - ratio[:, None]) / (1.5 / (BINS - 1))) ** 2)
            soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-12)
            cross_entropy = -(soft * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            expectation = (torch.softmax(logits, dim=1) * centers).sum(dim=1)
            loss = cross_entropy + 0.5 * torch.nn.functional.smooth_l1_loss(expectation, ratio, beta=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.net.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        scheduler.step()
        history.append(total / len(inputs))
    model.history = tuple(history)
    model.network.net.eval()
    return model


def save_checkpoint(model: TimingModel, path: str | Path) -> None:
    """Write weights and preprocessing statistics to an explicitly chosen path."""
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "repointer.wordphase.v1", "model_state": model.network.net.state_dict(),
        **{name: torch.from_numpy(getattr(model, name)) for name in ("word_mean", "word_std", "clip_mean", "clip_std")},
        "medians": model.medians, "hash_buckets": model.buckets, "max_words": model.max_words,
        "bins": BINS, "history": list(model.history),
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)


def load_checkpoint(path: str | Path) -> TimingModel:
    """Load cleaned tensor-only checkpoints with PyTorch's restricted loader.

    Historical numpy/pickle checkpoints need a separately audited conversion;
    this entry point deliberately never enables unrestricted pickle loading.
    """
    torch = _torch()
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (not isinstance(state, dict) or state.get("schema") != "repointer.wordphase.v1"
            or type(state.get("bins")) is not int or state.get("bins") != BINS):
        raise ValueError("unsupported timing checkpoint schema or bin count")
    arrays = []
    for name, shape in (("word_mean", (10,)), ("word_std", (10,)), ("clip_mean", (16,)), ("clip_std", (16,))):
        array = state[name].cpu().numpy()
        if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all() or (name.endswith("std") and np.any(array <= 0)):
            raise ValueError(f"invalid checkpoint statistic: {name}")
        arrays.append(finite_array(array, shape, name).copy())
    medians = _validate_priors(state.get("medians"))
    _integer(state.get("max_words"), "checkpoint max_words")
    _integer(state.get("hash_buckets"), "checkpoint hash buckets")
    network = WordPhaseNet(state["hash_buckets"])
    network.net.load_state_dict(state["model_state"], strict=True)
    network.net.eval()
    return TimingModel(network, *arrays, medians, state["hash_buckets"], state["max_words"], tuple(state.get("history", ())))


def synthetic_training_fixture(n: int = 8, seed: int = 0) -> tuple[list[TimingInput], list[int]]:
    """Generate invented transcripts and labels, independent of challenge data."""
    _integer(n, "fixture size")
    _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    inputs, labels = [], []
    for i in range(n):
        frames = int(rng.integers(30, 61))
        seconds = frames / 30.0
        peak = int(round((0.45 + 0.1 * (i % 2)) * (frames - 1)))
        words = [TimedWord(0.05 * seconds, 0.2 * seconds, "point"), TimedWord(0.35 * seconds, 0.6 * seconds, "there")]
        inputs.append(TimingInput(words, frames, i % 2, "natural" if (i // 2) % 2 else "synthetic", [1.0, 0.2, 2.0], max(2, frames - 12)))
        labels.append(peak)
    return inputs, labels
