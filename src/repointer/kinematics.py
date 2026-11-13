"""Asset-free test skeleton and an optional licensed SMPL-X adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Protocol
import numpy as np
from .contracts import Motion, finite_array
from .rotations import rotation_matrix


class Kinematics(Protocol):
    name: str
    def joints(self, motion: Motion, betas: np.ndarray, gender: str = "neutral") -> np.ndarray: ...


class SyntheticKinematics:
    """Tiny forward-kinematics test double. Not an SMPL-X body model."""
    name = "synthetic-skeleton"

    def joints(self, motion: Motion, betas: np.ndarray, gender: str = "neutral") -> np.ndarray:
        betas = finite_array(betas, (10,), "betas")
        if not isinstance(gender, str) or gender not in ("neutral", "male", "female"):
            raise ValueError("invalid kinematics gender")
        offsets = np.zeros((55, 3), dtype=np.float64)
        parents = np.zeros(55, dtype=int)
        parents[0] = -1
        parents[[3, 6, 9, 12, 15]] = [0, 3, 6, 9, 12]
        offsets[[3, 6, 9, 12, 15], 1] = [.18, .18, .18, .12, .18]
        for side, sign in ((0, -1.), (1, 1.)):
            hip, knee, ankle, foot = 1+side, 4+side, 7+side, 10+side
            parents[[hip, knee, ankle, foot]] = [0, hip, knee, ankle]
            offsets[hip] = [sign*.10, -.08, 0]
            offsets[knee] = [0, -.42, 0]
            offsets[ankle] = [0, -.42, 0]
            offsets[foot] = [0, -.03, .14]
            collar, shoulder, elbow, wrist = 13+side, 16+side, 18+side, 20+side
            parents[[collar, shoulder, elbow, wrist]] = [9, collar, shoulder, elbow]
            offsets[collar] = [sign*.08, .10, 0]
            offsets[shoulder] = [sign*.12, 0, 0]
            offsets[elbow] = [sign*.28, 0, 0]
            offsets[wrist] = [sign*.25, 0, 0]
            index = 27 if side == 0 else 42
            parents[index] = wrist
            offsets[index] = [sign*.08, 0, 0]
        offsets *= 1 + .03 * np.tanh(float(np.asarray(betas)[0]))
        rotations = rotation_matrix(motion.poses.reshape(-1, 55, 3))
        world = np.empty_like(rotations)
        joints = np.empty((motion.frames, 55, 3), dtype=np.float64)
        for joint, parent in enumerate(parents):
            if parent < 0:
                joints[:, joint] = motion.trans
                world[:, joint] = rotations[:, joint]
            else:
                joints[:, joint] = joints[:, parent] + np.einsum('tij,j->ti', world[:, parent], offsets[joint])
                world[:, joint] = world[:, parent] @ rotations[:, joint]
        return joints.astype(np.float32)


class SMPLXKinematics:
    """Direct model adapter; model files must be supplied separately under license.

    No automatic downloads, workstation discovery, or gender fallback. This
    adapter is unvalidated against historical RePointer outputs.
    """
    name = "smplx"

    def __init__(self, model_dir: Path, device: str = "cpu", batch_size: int = 128):
        if not Path(model_dir).is_dir():
            raise ValueError("configured SMPL-X model directory does not exist")
        if type(batch_size) is not int or batch_size < 1 or device not in ("cpu", "cuda"):
            raise ValueError("positive batch_size and cpu/cuda device required")
        try:
            import torch
            import smplx
        except ImportError as exc:
            raise RuntimeError("SMPL-X inference requires the optional [models] dependencies and licensed assets") from exc
        self.torch, self.smplx = torch, smplx
        self.model_dir, self.device, self.batch_size = Path(model_dir), device, batch_size
        self.models = {}

    def model(self, gender: str):
        if not isinstance(gender, str) or gender not in ("neutral", "male", "female"):
            raise ValueError("invalid SMPL-X gender")
        if gender not in self.models:
            model = self.smplx.create(str(self.model_dir), model_type="smplx", gender=gender,
                use_pca=False, flat_hand_mean=True, num_betas=10, num_expression_coeffs=10).to(self.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models[gender] = model
        return self.models[gender]

    def forward_torch(self, poses, trans, betas, gender: str):
        return self.model(gender)(betas=betas, global_orient=poses[:, :3], body_pose=poses[:, 3:66],
            jaw_pose=poses[:, 66:69], leye_pose=poses[:, 69:72], reye_pose=poses[:, 72:75],
            left_hand_pose=poses[:, 75:120], right_hand_pose=poses[:, 120:165],
            expression=self.torch.zeros((len(poses), 10), dtype=poses.dtype, device=poses.device),
            transl=trans, return_verts=False).joints

    def joints(self, motion: Motion, betas: np.ndarray, gender: str = "neutral") -> np.ndarray:
        betas = finite_array(betas, (10,), "betas")
        if not isinstance(gender, str) or gender not in ("neutral", "male", "female"):
            raise ValueError("invalid SMPL-X gender")
        torch = self.torch
        chunks = []
        with torch.no_grad():
            for start in range(0, motion.frames, self.batch_size):
                end = min(start+self.batch_size, motion.frames)
                def tensor(x):
                    return torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)
                joints = self.forward_torch(tensor(motion.poses[start:end]), tensor(motion.trans[start:end]),
                    tensor(np.broadcast_to(betas, (end-start, 10)).copy()), gender)
                chunks.append(joints.detach().cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)
