"""Portable path resolution, external learned weights and run isolation."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from repointer.config import PipelineConfig, load_config


def config_file(tmp_path, **sections):
    value = {'paths':{'fixture_root':'fixture','checkpoint_root':'weights','output_root':'run'},'pipeline':{}}
    value.update(sections)
    path = tmp_path/'settings.json'
    path.write_text(json.dumps(value),encoding='utf-8')
    return path


def test_config_relative_to_file_independent_of_working_directory(tmp_path,monkeypatch):
    path = config_file(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(path)
    assert config.fixture_root == tmp_path/'fixture'
    assert config.checkpoint_root == tmp_path/'weights'
    assert config.pipeline.hypotheses == 4


def test_environment_expansion_is_explicit_and_unset_fails(tmp_path,monkeypatch):
    path = config_file(tmp_path,paths=dict(fixture_root='${REPOINTER_TEST_ROOT}/fixture',checkpoint_root='weights',output_root='run'))
    monkeypatch.delenv('REPOINTER_TEST_ROOT',raising=False)
    with pytest.raises(ValueError,match='environment variable'): load_config(path)
    monkeypatch.setenv('REPOINTER_TEST_ROOT',str(tmp_path))
    assert load_config(path).fixture_root == tmp_path/'fixture'


@pytest.mark.parametrize('paths',[
    dict(fixture_root='same',checkpoint_root='same',output_root='run'),
    dict(fixture_root='parent',checkpoint_root='parent/child',output_root='run'),
    dict(fixture_root='',checkpoint_root='weights',output_root='run'),
    dict(fixture_root='fixture',output_root='run'),
])
def test_invalid_paths_rejected(tmp_path,paths):
    with pytest.raises(ValueError): load_config(config_file(tmp_path,paths=paths))


@pytest.mark.parametrize('settings',[dict(top_k=0),dict(ik_steps=1.5),dict(seed=True),dict(hypotheses=9),
    dict(donors_per_hypothesis=-1),dict(timing_backend='diffusion'),dict(selector_threshold=float('nan'))])
def test_invalid_parameters(settings):
    with pytest.raises(ValueError): PipelineConfig(**settings)


def test_unknown_settings_rejected(tmp_path):
    with pytest.raises(ValueError): load_config(config_file(tmp_path,unknown={}))
    with pytest.raises(TypeError): load_config(config_file(tmp_path,pipeline={'typo':3}))


def test_learned_weights_must_stay_outside_public_tree(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = config_file(tmp_path,pipeline={'timing_backend':'wordphase'},paths={
        'fixture_root':str(tmp_path/'fixture'),'output_root':str(tmp_path/'run'),
        'checkpoint_root':str(root/'outputs'/'forbidden-weights')})
    with pytest.raises(ValueError,match='outside'): load_config(path)


def test_toml_configuration(tmp_path):
    pytest.importorskip('tomllib')
    path = tmp_path/'settings.toml'
    path.write_text('[paths]\nfixture_root="fixture"\ncheckpoint_root="weights"\noutput_root="run"\n[pipeline]\nik_steps=1\n',encoding='utf-8')
    assert load_config(path).pipeline.ik_steps==1
