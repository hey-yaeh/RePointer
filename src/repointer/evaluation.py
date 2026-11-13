"""Transparent local geometry proxies; no organizer scores are computed here."""
from __future__ import annotations
from typing import Callable
import numpy as np
from .contracts import Candidate, Motion, Query
from .kinematics import Kinematics, SyntheticKinematics
from .selector import angular_dispersion_deg, clipped_window, robust_metrics


def spatial_scores(joints: np.ndarray, hand: int, target: np.ndarray,
                   half_diagonal: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """Forearm ray to point/bounding sphere; 20-degree Gaussian local proxy."""
    j = np.asarray(joints, dtype=float)
    if j.ndim != 3 or j.shape[1] < 22 or j.shape[2] != 3 or not np.isfinite(j).all() or hand not in (0,1):
        raise ValueError("finite (T,J>=22,3) joints and active hand required")
    targets = np.asarray(target, dtype=float)
    if targets.shape not in ((3,), (len(j),3)) or not np.isfinite(targets).all() or half_diagonal < 0 or not np.isfinite(half_diagonal):
        raise ValueError("invalid spatial target or extent")
    elbow, wrist = j[:,18+hand], j[:,20+hand]
    fore, to = wrist-elbow, targets-elbow
    denom = np.linalg.norm(fore,axis=1)*np.linalg.norm(to,axis=1)
    cos = np.sum(fore*to,axis=1)/np.maximum(denom,1e-12)
    angle = np.degrees(np.arccos(np.clip(cos,-1,1)))
    margin = np.degrees(np.arcsin(np.clip(half_diagonal/np.maximum(np.linalg.norm(to,axis=1),1e-12),0,1)))
    angle = np.maximum(0,angle-margin)
    # A zero-length forearm/target ray has no direction, hence no grounding.
    angle = np.where(denom < 1e-12,180.,angle)
    return np.exp(-angle**2/(2*20**2)), angle


def synthetic_detectors(joints: np.ndarray, hand: int) -> dict:
    """Three motion-only test heuristics, NOT the frozen P0B detector ensemble."""
    s,e,w = joints[:,16+hand],joints[:,18+hand],joints[:,20+hand]
    norm = lambda x: np.linalg.norm(x,axis=-1)
    reach = norm(w-s)/np.maximum(norm(e-s)+norm(w-e),1e-8)
    speed = norm(np.gradient(w.astype(float),axis=0))
    direction = (w-e)/np.maximum(norm(w-e)[:,None],1e-8)
    turn = norm(np.gradient(direction,axis=0))
    def standardized(x):
        return (x-x.mean())/max(float(x.std()),1e-6)
    valid = np.ones(len(joints),dtype=bool); valid[:2] = False; valid[-2:] = False
    tracks = [reach-.1*speed, standardized(reach)-.5*standardized(speed),
              standardized(reach)-.5*standardized(turn)-.25*standardized(speed)]
    frames = [int(np.argmax(np.where(valid,t,-np.inf))) for t in tracks]
    ensemble = int(np.median(frames))
    return dict(source=frames[0],refined=frames[1],heuristic=frames[2],ensemble=ensemble,
                agreement=float(np.exp(-np.std(frames)**2/(2*5**2))))


def motion_quality(joints: np.ndarray, motion: Motion, donor: Motion | None = None) -> dict[str,float]:
    """Historical diagnostic scale: each finite difference is divided by 30.

    These are local diagnostic quantities, not SI velocity/jerk or a naturalness
    assessment. Retaining this scale is necessary for frozen selector thresholds.
    """
    difference = lambda x: np.zeros_like(x,dtype=float) if len(x)<=1 else np.gradient(np.asarray(x,dtype=float),axis=0)/30.
    jerk = difference(difference(difference(joints)))
    feet = joints[:,[10,11]]
    speed = np.linalg.norm(difference(feet)[:,:,[0,2]],axis=2)
    contact = feet[:,:,1] <= feet[:,:,1].min(axis=0)[None]+.05
    aa = motion.poses[:,3:66].reshape(motion.frames,21,3)
    quality = dict(joint_jerk=float(np.linalg.norm(jerk[:,:22],axis=2).mean()),
                   foot_sliding=float((speed*contact).sum()/max(1,int(contact.sum()))),
                   joint_limit_fraction=float((np.linalg.norm(aa,axis=2)>np.pi).mean()))
    quality['generic_joint_jerk'] = quality['joint_jerk']
    quality['generic_foot_sliding'] = quality['foot_sliding']
    quality['pose_deviation'] = 0. if donor is None else float(np.linalg.norm((motion.poses-donor.poses).reshape(-1,55,3),axis=2).mean())
    return quality


def evaluate_candidate(candidate: Candidate, query: Query, engine: Kinematics,
                       donor: Motion | None = None, detector: Callable | None = None) -> dict:
    """Generation-safe metrics: no released peak, hold, or alignment labels."""
    if candidate.motion.frames != query.frames:
        raise ValueError("query/candidate frame count mismatch")
    if detector is None:
        if not isinstance(engine,SyntheticKinematics):
            raise ValueError("licensed backend requires an explicitly supplied audited motion-only detector")
        detector = synthetic_detectors
    joints = engine.joints(candidate.motion, query.betas, query.gender)
    det = detector(joints,query.hand)
    point_world = query.target_world
    if point_world is None:
        point_world = joints[candidate.planned_frame,0]+query.target
    center_world = point_world if query.bbox_center is None else query.bbox_center
    ref_pelvis = point_world-query.target
    half_diag = 0. if query.bbox_size is None else .5*float(np.linalg.norm(query.bbox_size))
    relative, angles = spatial_scores(joints, query.hand, joints[:,0]+center_world-ref_pelvis, half_diag)
    world_bbox, _ = spatial_scores(joints, query.hand, center_world, half_diag)
    world_point, _ = spatial_scores(joints, query.hand, point_world)
    frames = [int(det[key]) for key in ('source','refined','heuristic','ensemble')]
    indices = set(clipped_window(candidate.planned_frame,2,query.frames))
    for frame in frames: indices.update(clipped_window(frame,2,query.frames))
    forearms = joints[:,20+query.hand]-joints[:,18+query.hand]
    dispersion = angular_dispersion_deg(forearms[sorted(indices)])
    quality = motion_quality(joints,candidate.motion,donor)
    quality['max_reach'] = float(np.linalg.norm(joints[:,20+query.hand]-joints[:,16+query.hand],axis=1).max())
    inside = query.bbox_size is not None and bool(np.all(np.abs(point_world-center_world)<=query.bbox_size/2+1e-6))
    robust = robust_metrics(relative,world_bbox,world_point,candidate.planned_frame,frames,
                            det['agreement'],dispersion,quality,point_inside_bbox=inside)
    return {**quality,**robust,'self_frame':frames[-1],
            'planned_score1b_proxy20':float(relative[candidate.planned_frame]),
            'self_score1b_proxy20':float(relative[frames[-1]]),'self_angle_deg':float(angles[frames[-1]])}


def evaluate_hold(detected_frame: int, hold_start: int, hold_end: int, sigma: float = 5.) -> float:
    """Explicit evaluation-only temporal proxy, separate from generation."""
    if not all(type(v) is int for v in (detected_frame,hold_start,hold_end)) or not 0 <= hold_start <= hold_end or detected_frame < 0:
        raise ValueError("invalid detected frame or ground-truth hold interval")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("temporal sigma must be finite and positive")
    distance = max(hold_start-detected_frame,detected_frame-hold_end,0)
    return float(np.exp(-distance**2/(2*sigma**2)))
