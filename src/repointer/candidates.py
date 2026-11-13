"""Shared M1 realization. Timing routes always produce complete motion clips."""
from __future__ import annotations
import numpy as np
from .contracts import Candidate, Donor, Motion, Query
from .evaluation import evaluate_candidate
from .ik import optimize_apex
from .kinematics import Kinematics
from .retrieval import retrieve
from .warping import apply_hold, warp_clip


def realize(query: Query, donor: Donor, distance: float, planned: int, source: str,
            engine: Kinematics, *, steps: int = 50, plateau: int = 5, taper: int = 8) -> tuple[Candidate,dict]:
    warped = warp_clip(donor.motion,donor.apex,query.frames,planned)
    apex = optimize_apex(engine,warped.poses[planned],query,steps=steps)
    motion = apply_hold(warped,planned,apex,query.hand,plateau,taper)
    if query.target_world is not None:
        joints = engine.joints(motion,query.betas,query.gender)
        shift = (query.target_world-query.target)-joints[planned,0]
        motion = Motion(motion.poses,motion.trans+shift)
    candidate = Candidate(motion,planned,donor.context.sample_id,distance,source)
    return candidate,evaluate_candidate(candidate,query,engine,warped)


def construct_route(query: Query, donors: list[Donor], planned: int, source: str,
                    engine: Kinematics, *, top_k: int = 8, steps: int = 50,
                    plateau: int = 5, taper: int = 8) -> tuple[Candidate,dict]:
    candidates = []
    for rank,(donor,distance) in enumerate(retrieve(query,donors,planned,top_k),1):
        candidate,metrics = realize(query,donor,distance,planned,source,engine,
                                    steps=steps,plateau=plateau,taper=taper)
        metrics = {**metrics,'donor_rank':rank,'donor_distance':distance,'planned_frame':planned}
        candidates.append((candidate,metrics))
    # Same lexicographic M1 ordering; no evaluation labels influence the choice.
    return min(candidates,key=lambda item:(-item[1]['planned_score1b_proxy20'],
        -item[1]['self_score1b_proxy20'],-item[1]['detector_agreement'],
        item[1]['joint_jerk'],item[0].donor_distance))


def query_features(query: Query) -> dict:
    x,y,z = query.target.astype(float)
    return dict(domain=query.domain,hand=query.hand,clip_length=query.frames,
        target_distance=float(np.linalg.norm(query.target)),target_azimuth_deg=float(np.degrees(np.arctan2(x,z))),
        target_elevation_deg=float(np.degrees(np.arctan2(y,max(np.hypot(x,z),1e-8)))))
