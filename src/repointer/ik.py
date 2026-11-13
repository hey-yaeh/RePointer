"""Shape-aware target-ray apex optimization; inactive joints stay unchanged."""
from __future__ import annotations
import numpy as np
from .contracts import ACTIVE_JOINTS, Motion, Query, finite_array, finite_scalar
from .kinematics import Kinematics, SMPLXKinematics


def _geometry(joints, hand):
    shoulder, elbow, wrist = (joints[k] for k in (16+hand, 18+hand, 20+hand))
    reach = np.linalg.norm(wrist-shoulder) / max(np.linalg.norm(elbow-shoulder)+np.linalg.norm(wrist-elbow), 1e-6)
    bend = np.dot(shoulder-elbow, wrist-elbow) / max(np.linalg.norm(shoulder-elbow)*np.linalg.norm(wrist-elbow), 1e-6)
    return elbow, wrist, reach, bend


def optimize_apex(engine: Kinematics, pose: np.ndarray, query: Query,
                  steps: int = 50, lr: float = .05, max_delta: float = 1.5) -> np.ndarray:
    pose = finite_array(pose, (165,), "IK pose")
    if (type(steps) is not int or steps < 1
            or finite_scalar(lr, "IK learning rate") <= 0
            or finite_scalar(max_delta, "IK maximum delta") <= 0):
        raise ValueError("positive finite IK parameters required")
    if isinstance(engine, SMPLXKinematics):
        return _optimize_torch(engine, pose, query, steps, lr, max_delta)
    base = np.asarray(pose, dtype=float).copy()
    indices = np.array([3*j+k for j in ACTIVE_JOINTS[query.hand] for k in range(3)])
    def joints(value):
        return engine.joints(Motion(value[None], np.zeros((1,3))), query.betas, query.gender)[0]
    initial = joints(base)
    _, _, reach0, bend0 = _geometry(initial, query.hand)
    desired = max(reach0, .88)
    def loss(delta):
        changed = base.copy()
        changed[indices] += delta
        j = joints(changed)
        elbow, wrist, reach, bend = _geometry(j, query.hand)
        fore, target = wrist-elbow, j[0]+query.target-elbow
        cosine = np.dot(fore, target)/max(np.linalg.norm(fore)*np.linalg.norm(target), 1e-6)
        large = np.maximum(np.linalg.norm(delta.reshape(4,3),axis=1)-1,0)
        return 8*(1-np.clip(cosine,-1,1))+.8*(reach-desired)**2+.15*(bend-bend0)**2+.015*np.mean(delta**2)+.02*np.mean(large**2)
    # Finite differences are only for the tiny asset-free test skeleton. Licensed
    # SMPL-X uses autograd below with the same M1 objective.
    delta, first, second = np.zeros(12), np.zeros(12), np.zeros(12)
    best, best_loss = delta.copy(), loss(delta)
    eps = 1e-3
    for step in range(1, steps+1):
        gradient = np.empty(12)
        for k in range(12):
            offset = np.zeros(12); offset[k] = eps
            gradient[k] = (loss(delta+offset)-loss(delta-offset))/(2*eps)
        first = .9*first+.1*gradient
        second = .999*second+.001*gradient**2
        delta = np.clip(delta-lr*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8), -max_delta, max_delta)
        value = loss(delta)
        if value < best_loss:
            best, best_loss = delta.copy(), value
    base[indices] += best
    return base.astype(np.float32)


def _optimize_torch(engine, pose, query, steps, lr, max_delta):
    torch = engine.torch
    base = torch.tensor(pose[None], dtype=torch.float32, device=engine.device)
    betas = torch.tensor(query.betas[None].copy(), dtype=torch.float32, device=engine.device)
    trans = torch.zeros((1,3), device=engine.device)
    indices = [3*j+k for j in ACTIVE_JOINTS[query.hand] for k in range(3)]
    delta = torch.zeros((1,12), device=engine.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    def geometry(j):
        s, e, w = j[:,16+query.hand],j[:,18+query.hand],j[:,20+query.hand]
        norm = lambda x: torch.linalg.norm(x, dim=1)
        reach = norm(w-s)/torch.clamp(norm(e-s)+norm(w-e), min=1e-6)
        bend = torch.sum((s-e)*(w-e),dim=1)/torch.clamp(norm(s-e)*norm(w-e),min=1e-6)
        return e,w,reach,bend
    with torch.no_grad():
        _,_,r0,b0 = geometry(engine.forward_torch(base,trans,betas,query.gender))
        desired = torch.clamp(r0,min=.88)
    target = torch.tensor(query.target[None].copy(), device=engine.device)
    for _ in range(steps):
        changed = base.clone(); changed[:,indices] += delta
        j = engine.forward_torch(changed,trans,betas,query.gender)
        e,w,r,b = geometry(j)
        fore,to = w-e,j[:,0]+target-e
        cosine = torch.sum(fore*to,dim=1)/torch.clamp(torch.linalg.norm(fore,dim=1)*torch.linalg.norm(to,dim=1),min=1e-6)
        loss = (8*(1-cosine.clamp(-1,1))+.8*(r-desired)**2+.15*(b-b0)**2).mean()
        loss += .015*(delta**2).mean()+.02*(torch.relu(torch.linalg.norm(delta.reshape(1,4,3),dim=2)-1)**2).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad(): delta.clamp_(-max_delta,max_delta)
    out = np.array(pose,copy=True)
    out[indices] += delta.detach().cpu().numpy()[0]
    return out.astype(np.float32)
