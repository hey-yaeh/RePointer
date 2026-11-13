"""Strict public contracts. Evaluation labels never enter a generation Query."""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
import re
import numpy as np

FPS = 30.0
ACTIVE_JOINTS = {0: (13, 16, 18, 20), 1: (14, 17, 19, 21)}


def finite_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite numeric {shape}, got {array.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(array, dtype=np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} exceeds float32 range")
    array.setflags(write=False)
    return array


def finite_scalar(value, name: str) -> float:
    """Accept real numeric scalars, excluding booleans and array coercions."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} exceeds float64 range")
    return converted


def sample_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("sample ID must be a nonempty filename-safe identifier")
    return value


@dataclass(frozen=True, eq=False)
class Motion:
    """SMPL-X axis-angle layout, radians; translations in dataset length units."""
    poses: np.ndarray
    trans: np.ndarray

    def __post_init__(self):
        poses = np.asarray(self.poses)
        if poses.ndim == 3 and poses.shape[1:] == (55, 3):
            poses = poses.reshape(len(poses), 165)
        if poses.ndim != 2 or poses.shape[1:] != (165,) or len(poses) < 1:
            raise ValueError("poses must be nonempty (T,165) or (T,55,3)")
        object.__setattr__(self, "poses", finite_array(poses, (len(poses), 165), "poses"))
        object.__setattr__(self, "trans", finite_array(self.trans, (len(poses), 3), "trans"))

    @property
    def frames(self) -> int:
        return len(self.poses)

    @classmethod
    def load(cls, path: Path, expected_frames: int | None = None) -> Motion:
        if expected_frames is not None and (type(expected_frames) is not int or expected_frames < 1):
            raise ValueError("expected_frames must be a positive integer")
        with np.load(path, allow_pickle=False) as data:
            motion = cls(data["poses"], data["trans"])
        if expected_frames is not None and motion.frames != expected_frames:
            raise ValueError(f"frame count {motion.frames} != expected {expected_frames}")
        return motion

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite an existing run or any archived source.
        with path.open("xb") as stream:
            np.savez_compressed(stream, poses=self.poses, trans=self.trans)


@dataclass(frozen=True, eq=False)
class Query:
    sample_id: str
    frames: int
    hand: int
    target: np.ndarray
    betas: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float32))
    gender: str = "neutral"
    domain: str = "natural"
    group: str = "query"
    recording: str = "query"
    fold: int = -1
    words: tuple = ()
    clip_start: float = 0.0
    target_world: np.ndarray | None = None
    bbox_size: np.ndarray | None = None
    bbox_center: np.ndarray | None = None

    def __post_init__(self):
        sample_name(self.sample_id)
        if type(self.frames) is not int or self.frames < 5:
            raise ValueError("generation requires integer frames >= 5")
        if type(self.hand) is not int or self.hand not in ACTIVE_JOINTS:
            raise ValueError("hand must be 0 (left) or 1 (right)")
        if type(self.fold) is not int or self.fold < -1:
            raise ValueError("fold must be an integer >= -1")
        if not isinstance(self.gender, str) or self.gender not in ("neutral", "male", "female"):
            raise ValueError("gender must be neutral, male or female")
        if not isinstance(self.domain, str) or self.domain not in ("natural", "synthetic"):
            raise ValueError("domain must be natural or synthetic")
        if (not isinstance(self.group, str) or not self.group.strip()
                or not isinstance(self.recording, str) or not self.recording.strip()
                or finite_scalar(self.clip_start, "clip_start") < 0):
            raise ValueError("group, recording and nonnegative clip_start are required")
        object.__setattr__(self, "target", finite_array(self.target, (3,), "target"))
        if np.linalg.norm(self.target) < 1e-6:
            raise ValueError("target pelvis-relative vector must be nonzero")
        object.__setattr__(self, "betas", finite_array(self.betas, (10,), "betas"))
        for name in ("target_world", "bbox_size", "bbox_center"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, finite_array(value, (3,), name))
        if self.bbox_size is not None and np.any(self.bbox_size < 0):
            raise ValueError("bbox_size must be nonnegative")
        if (self.bbox_size is None) != (self.bbox_center is None):
            raise ValueError("bbox_center and bbox_size must be supplied together")
        object.__setattr__(self, "words", tuple(self.words))


@dataclass(frozen=True, eq=False)
class Donor:
    context: Query
    motion: Motion
    apex: int

    def __post_init__(self):
        if not isinstance(self.context, Query) or not isinstance(self.motion, Motion):
            raise ValueError("donor requires Query context and Motion arrays")
        if self.context.frames != self.motion.frames:
            raise ValueError("donor metadata/motion length mismatch")
        if type(self.apex) is not int or not 1 <= self.apex < self.motion.frames - 1:
            raise ValueError("donor apex must be an interior frame")


@dataclass(frozen=True, eq=False)
class Candidate:
    motion: Motion
    planned_frame: int
    donor_id: str
    donor_distance: float
    timing_source: str

    def __post_init__(self):
        if not isinstance(self.motion, Motion):
            raise ValueError("candidate requires Motion arrays")
        if type(self.planned_frame) is not int or not 1 <= self.planned_frame < self.motion.frames - 1:
            raise ValueError("candidate planned frame must be an interior integer")
        if finite_scalar(self.donor_distance, "donor distance") < 0:
            raise ValueError("donor distance must be finite and nonnegative")
        sample_name(self.donor_id)
        if not isinstance(self.timing_source, str) or not self.timing_source.strip():
            raise ValueError("candidate timing source must be a nonempty string")
