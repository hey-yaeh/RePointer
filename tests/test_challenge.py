import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pytest

from repointer.challenge import (
    create_submission, materialize_candidate, package_submission,
    validate_motion_directory, validate_motion_file,
)


def _motion(path: Path, layout=(6, 165), dtype=np.float32, **extra):
    np.savez_compressed(path, poses=np.zeros(layout, dtype=dtype), trans=np.zeros((6, 3), dtype=dtype), **extra)
    return path


@pytest.mark.parametrize('layout', [(6, 165), (6, 55, 3)])
def test_materialization_preserves_source_bytes_and_no_overwrite(tmp_path, layout):
    a = _motion(tmp_path / 'a.npz', layout)
    b = _motion(tmp_path / 'b.npz')
    out = tmp_path / 'materialized' / 'out.npz'
    result = materialize_candidate(a, b, 'A', out, 6)
    assert out.read_bytes() == a.read_bytes()
    assert result['sha256'] == hashlib.sha256(a.read_bytes()).hexdigest()
    assert result['frames'] == 6 and result['choice'] == 'A'
    with pytest.raises(FileExistsError):
        materialize_candidate(a, b, 'B', out, 6)
    with pytest.raises(ValueError, match='A or B'):
        materialize_candidate(a, b, 'C', tmp_path / 'invalid.npz', 6)


@pytest.mark.parametrize('layout,dtype,extra', [
    ((6, 164), np.float32, {}), ((6, 165), np.float64, {}),
    ((6, 165), np.float32, {'betas': np.zeros(10)}),
])
def test_strict_challenge_schema_rejects_invalid_archives(tmp_path, layout, dtype, extra):
    bad = _motion(tmp_path / 'bad.npz', layout, dtype, **extra)
    with pytest.raises(ValueError):
        validate_motion_file(bad, 6)
    with pytest.raises(ValueError):
        create_submission({'bad': bad}, {'bad': 6}, tmp_path / 'submission')
    assert not (tmp_path / 'submission').exists()


def test_schema_rejects_nonfinite_frame_count_and_pickle(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, poses=np.full((6, 165), np.nan, dtype=np.float32), trans=np.zeros((6, 3), dtype=np.float32))
    with pytest.raises(ValueError, match='nonfinite'):
        validate_motion_file(path, 6)
    _motion(path)
    with pytest.raises(ValueError, match='expected frames'):
        validate_motion_file(path, 7)
    np.savez(path, poses=np.zeros((6, 165), dtype=object), trans=np.zeros((6, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        validate_motion_file(path, 6)


@pytest.mark.parametrize('sample_id', ['../outside', 'a/b', r'a\b', '.', '..', '/absolute', ''])
def test_rejects_unsafe_manifest_identifiers(tmp_path, sample_id):
    path = _motion(tmp_path / 'valid.npz')
    with pytest.raises(ValueError, match='identifier'):
        create_submission({sample_id: path}, {sample_id: 6}, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_exact_manifest_missing_extra_and_case_collision(tmp_path):
    path = _motion(tmp_path / 'valid.npz')
    with pytest.raises(ValueError, match='exactly'):
        create_submission({'a': path}, {'a': 6, 'b': 6}, tmp_path / 'out')
    with pytest.raises(ValueError, match='exactly'):
        create_submission({'a': path, 'b': path}, {'a': 6}, tmp_path / 'out')
    with pytest.raises(ValueError, match='collide'):
        create_submission({'A': path, 'a': path}, {'A': 6, 'a': 6}, tmp_path / 'out')


def test_directory_and_deterministic_zip_preserve_bytes(tmp_path):
    a, b = _motion(tmp_path / 'a.npz'), _motion(tmp_path / 'b.npz', (6, 55, 3))
    output = tmp_path / 'submission'
    manifest = {'z': 6, 'a': 6}
    hashes = create_submission({'z': a, 'a': b}, manifest, output)
    assert set(hashes) == {'z', 'a'}
    assert (output / 'motions' / 'a.npz').read_bytes() == b.read_bytes()
    validated = validate_motion_directory(output / 'motions', manifest)
    assert validated['a']['sha256'] == hashes['a']
    first, second = tmp_path / 'first.zip', tmp_path / 'second.zip'
    package_submission(output / 'motions', first, manifest)
    package_submission(output / 'motions', second, manifest)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ['metadata.json', 'motions/a.npz', 'motions/z.npz']
        assert archive.read('motions/a.npz') == b.read_bytes()
        assert archive.read('motions/z.npz') == a.read_bytes()
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    with pytest.raises(FileExistsError):
        package_submission(output / 'motions', first, manifest)
    with pytest.raises(FileExistsError):
        create_submission({'z': a, 'a': b}, manifest, output)
    extra = output / 'motions' / 'extra.txt'
    extra.write_text('unexpected')
    with pytest.raises(ValueError, match='exactly'):
        validate_motion_directory(output / 'motions', manifest)


def test_frozen_destination_rejected_even_through_symlink(tmp_path):
    import repointer.challenge as challenge
    # Substitute a fixture immutable archive, so the real frozen tree stays untouched.
    frozen = tmp_path / 'historical'
    frozen.mkdir()
    original = challenge._FROZEN_ROOT
    challenge._FROZEN_ROOT = frozen
    try:
        motion = _motion(tmp_path / 'motion.npz')
        alias = tmp_path / 'alias'
        alias.symlink_to(frozen, target_is_directory=True)
        with pytest.raises(ValueError, match='frozen'):
            materialize_candidate(motion, motion, 'A', alias / 'bad.npz', 6)
        assert not list(frozen.iterdir())
    finally:
        challenge._FROZEN_ROOT = original
