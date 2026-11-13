"""Local challenge schema checks and byte-preserving materialization.

These checks do not run the organizer evaluator or establish paper reproduction.
The official test manifest and validator must be obtained separately. No data or
submission is uploaded by these functions.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np

from .contracts import sample_name

_FROZEN_ROOT = Path(__file__).resolve().parents[2] / 'frozen'
_METADATA = {
    'system': 'RePointer',
    'validation': 'local_schema_only',
    'official_scores_reproduced': False,
}


def _destination(path: str | Path) -> Path:
    path = Path(path)
    if path.resolve().is_relative_to(_FROZEN_ROOT.resolve()):
        raise ValueError('cannot write into frozen historical implementation')
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'destination already exists: {path.name}')
    return path


def _snapshot(path: str | Path, expected_frames: int) -> tuple[bytes, dict[str, Any]]:
    """Validate the exact bytes subsequently copied, without loading pickles."""
    if type(expected_frames) is not int or expected_frames < 1:
        raise ValueError('expected frame count must be a positive integer')
    path = Path(path)
    payload = path.read_bytes()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if sorted(archive.files) != ['poses', 'trans']:
                raise ValueError('motion NPZ must contain exactly poses and trans')
            poses, trans = archive['poses'], archive['trans']
            if poses.shape not in ((expected_frames, 165), (expected_frames, 55, 3)):
                raise ValueError('poses must match expected frames and have (T,165) or (T,55,3) shape')
            if trans.shape != (expected_frames, 3):
                raise ValueError('trans must match expected frames with (T,3) shape')
            if poses.dtype != np.float32 or trans.dtype != np.float32:
                raise ValueError('challenge motion arrays must have float32 dtype')
            if not np.isfinite(poses).all() or not np.isfinite(trans).all():
                raise ValueError('motion contains nonfinite values')
            layout = 'T,165' if poses.ndim == 2 else 'T,55,3'
    except (KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise ValueError('invalid motion NPZ archive') from exc
    return payload, {
        'frames': expected_frames, 'pose_layout': layout,
        'sha256': hashlib.sha256(payload).hexdigest(),
    }


def validate_motion_file(path: str | Path, expected_frames: int) -> dict[str, Any]:
    """Validate strict output schema and return a digest; no kinematics implied."""
    return _snapshot(path, expected_frames)[1]


def materialize_candidate(
    a_path: str | Path,
    b_path: str | Path,
    decision_choice: str,
    output_path: str | Path,
    expected_frames: int,
) -> dict[str, Any]:
    """Copy the complete selected A or B archive, preserving every source byte."""
    if decision_choice not in ('A', 'B'):
        raise ValueError('candidate choice must be A or B')
    output_path = _destination(output_path)
    payload, metadata = _snapshot(b_path if decision_choice == 'B' else a_path, expected_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('xb') as stream:
        stream.write(payload)
    return {'choice': decision_choice, **metadata}


def _manifest(expected_frames: Mapping[str, int]) -> dict[str, int]:
    if not expected_frames:
        raise ValueError('submission manifest cannot be empty')
    out = {}
    for sample_id, frames in expected_frames.items():
        sample_name(sample_id)
        if type(frames) is not int or frames < 1:
            raise ValueError('manifest frame counts must be positive integers')
        out[sample_id] = frames
    if len({sample_id.casefold() for sample_id in out}) != len(out):
        raise ValueError('sample IDs collide on case-insensitive filesystems')
    return out


def _snapshots(
    motions: Mapping[str, Path], expected_frames: Mapping[str, int],
) -> dict[str, tuple[bytes, dict[str, Any]]]:
    expected_frames = _manifest(expected_frames)
    if set(motions) != set(expected_frames):
        raise ValueError('motion IDs must match the manifest exactly (missing or extra IDs)')
    return {sample_id: _snapshot(motions[sample_id], expected_frames[sample_id]) for sample_id in sorted(expected_frames)}


def create_submission(
    motions: Mapping[str, Path],
    expected_frames: Mapping[str, int],
    output_dir: str | Path,
) -> dict[str, str]:
    """Create a local submission folder after checking the entire manifest.

    Inputs are captured before writing so schema errors leave no partial output,
    and concurrent source changes cannot alter a materialized clip. Output paths
    must be new. Returned hashes are keyed by sample ID and contain no local paths.
    """
    output_dir = _destination(output_dir)
    snapshots = _snapshots(motions, expected_frames)
    output_dir.mkdir(parents=True, exist_ok=False)
    motion_dir = output_dir / 'motions'
    motion_dir.mkdir()
    for sample_id, (payload, _) in snapshots.items():
        with (motion_dir / f'{sample_id}.npz').open('xb') as stream:
            stream.write(payload)
    with (output_dir / 'metadata.json').open('x', encoding='utf-8') as stream:
        json.dump(_METADATA, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return {sample_id: info['sha256'] for sample_id, (_, info) in snapshots.items()}


def validate_motion_directory(
    motion_dir: str | Path, expected_frames: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Require exactly one NPZ per manifest ID, with no unexpected files."""
    expected = _manifest(expected_frames)
    motion_dir = Path(motion_dir)
    wanted = {f'{sample_id}.npz' for sample_id in expected}
    if {path.name for path in motion_dir.iterdir()} != wanted:
        raise ValueError('motion directory does not match manifest exactly (missing or extra files)')
    return {
        sample_id: validate_motion_file(motion_dir / f'{sample_id}.npz', expected[sample_id])
        for sample_id in sorted(expected)
    }


def package_submission(
    motion_dir: str | Path,
    zip_path: str | Path,
    expected_frames: Mapping[str, int],
) -> dict[str, Any]:
    """Write a deterministic local ZIP with safe, stable challenge filenames.

    Only validated motion files and generated minimal metadata enter the archive.
    Archive timestamps are fixed, and existing ZIP files are never overwritten.
    The official validator remains a separate, required reproduction step.
    """
    zip_path = _destination(zip_path)
    motion_dir = Path(motion_dir)
    validate_motion_directory(motion_dir, expected_frames)
    snapshots = _snapshots({sid: motion_dir / f'{sid}.npz' for sid in expected_frames}, expected_frames)
    contents = {'metadata.json': (json.dumps(_METADATA, sort_keys=True, indent=2) + '\n').encode('utf-8')}
    contents.update({f'motions/{sid}.npz': payload for sid, (payload, _) in snapshots.items()})
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zip_path.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return {'sha256': hashlib.sha256(zip_path.read_bytes()).hexdigest(),
            'motions': {sid: info['sha256'] for sid, (_, info) in snapshots.items()}}
