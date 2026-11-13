from pathlib import Path
import json
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
from repointer.cli import EXPECTED,audit,validate_motion

def test_sources_parse():
    assert audit(ROOT)==0


def test_source_audit_detects_modified_and_extra_files(tmp_path):
    (tmp_path/'docs').mkdir()
    shutil.copyfile(ROOT/'docs/SOURCE_INVENTORY.json',tmp_path/'docs/SOURCE_INVENTORY.json')
    for name in EXPECTED:
        destination=tmp_path/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    assert audit(tmp_path)==0
    with (tmp_path/EXPECTED[0]).open('a') as stream:
        stream.write('\n# changed snapshot\n')
    assert audit(tmp_path)==1
    shutil.copyfile(ROOT/EXPECTED[0],tmp_path/EXPECTED[0])
    (tmp_path/'frozen'/'unexpected.txt').write_text('generated file')
    assert audit(tmp_path)==1

def test_motion_schema_valid(tmp_path):
    p=tmp_path/'valid.npz'
    np.savez(p,poses=np.zeros((5,165),np.float32),trans=np.zeros((5,3),np.float32))
    assert validate_motion(p,expected_frames=5)==0

def test_motion_schema_invalid(tmp_path):
    p=tmp_path/'invalid.npz'
    np.savez(p,poses=np.zeros((5,164),np.float32),trans=np.zeros((5,3),np.float32))
    assert validate_motion(p)==1

def test_motion_schema_nan(tmp_path):
    p=tmp_path/'nan.npz'
    poses=np.zeros((5,55,3),np.float32)
    poses[2,8,0]=np.nan
    np.savez(p,poses=poses,trans=np.zeros((5,3),np.float32))
    assert validate_motion(p)==1

def test_no_private_paths_in_public_texts():
    private_roots = ('/' + 'media' + '/' + 'astra' + '/', '/' + 'home' + '/' + 'astra' + '/')
    for p in ROOT.rglob('*'):
        if any(part in {'.venv','.git','__pycache__','build','dist','outputs','.pytest_cache'} or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts):
            continue
        if not p.is_file() or p.suffix.lower() not in {'.py','.md','.json','.toml','.txt'}:
            continue
        s=p.read_text(encoding='utf-8')
        assert not any(root in s for root in private_roots), str(p)
