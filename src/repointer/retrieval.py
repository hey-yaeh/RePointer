"""Natural same-hand donors, with executable fold/group/recording exclusions."""
from __future__ import annotations
import numpy as np
from .contracts import Donor, Query, finite_scalar


def descriptor(query: Query, apex_ratio: float) -> np.ndarray:
    x, y, z = query.target.astype(float)
    distance = float(np.linalg.norm(query.target.astype(float)))
    if not 0 <= finite_scalar(apex_ratio, "apex ratio") <= 1:
        raise ValueError("apex ratio must lie in [0,1]")
    return np.array([np.arctan2(x, z)/np.pi, np.arctan2(y, max(np.hypot(x,z),1e-8))/(np.pi/2),
                     np.log(max(distance,1e-4)), query.frames/180., apex_ratio])


def retrieve(query: Query, donors: list[Donor], planned_frame: int, top_k: int = 8) -> list[tuple[Donor, float]]:
    if (type(top_k) is not int or top_k < 1 or type(planned_frame) is not int
            or not 1 <= planned_frame < query.frames - 1):
        raise ValueError("positive retrieval budget and interior apex required")
    pool = [d for d in donors if d.context.domain == "natural" and d.context.hand == query.hand
            and d.context.sample_id != query.sample_id and d.context.group != query.group
            and d.context.recording != query.recording
            and (query.fold < 0 or d.context.fold != query.fold)]
    if not pool:
        raise ValueError("no eligible natural same-hand training donors after leakage exclusions")
    matrix = np.stack([descriptor(d.context, d.apex/(d.motion.frames-1)) for d in pool])
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.
    distances = np.linalg.norm((matrix - descriptor(query, planned_frame/(query.frames-1)))/std, axis=1)
    indices = np.argsort(distances, kind="stable")[:top_k]
    return [(pool[i], float(distances[i])) for i in indices]
