"""JSON/TOML configuration; paths are relative to the config, never cwd."""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
from string import Template


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 0
    top_k: int = 8
    donors_per_hypothesis: int = 4
    hypotheses: int = 4
    ik_steps: int = 50
    hold_plateau: int = 5
    hold_taper: int = 8
    timing_backend: str = "synthetic-prior"
    epochs: int = 2
    selector_threshold: float = .001

    def __post_init__(self):
        import math
        for key in ('seed','top_k','donors_per_hypothesis','hypotheses','ik_steps','hold_plateau','hold_taper','epochs'):
            value = getattr(self,key)
            if type(value) is not int or value < (0 if key in ('seed','hold_plateau','hold_taper') else 1):
                raise ValueError(f"{key} must be an integer in its valid range")
        if self.hypotheses != 4:
            raise ValueError("the final method requires four timing hypotheses")
        if self.timing_backend not in ('synthetic-prior','wordphase'):
            raise ValueError("timing_backend must be synthetic-prior or wordphase")
        if not math.isfinite(self.selector_threshold) or self.selector_threshold < 0:
            raise ValueError("selector_threshold must be finite and nonnegative")


@dataclass(frozen=True)
class RunConfig:
    pipeline: PipelineConfig
    fixture_root: Path
    checkpoint_root: Path
    output_root: Path


def load_config(path: Path) -> RunConfig:
    path = Path(path).resolve()
    if path.suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
    elif path.suffix == '.toml':
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError as exc:
                raise RuntimeError("TOML on Python 3.10 needs [config]; JSON requires no extra dependency") from exc
        data = tomllib.loads(path.read_text(encoding='utf-8'))
    else:
        raise ValueError("configuration must be .json or .toml")
    if set(data) - {'pipeline','paths'}:
        raise ValueError("unknown configuration section")
    pipeline = PipelineConfig(**data.get('pipeline',{}))
    paths = data.get('paths',{})
    expected = {'fixture_root','checkpoint_root','output_root'}
    if set(paths) != expected:
        raise ValueError(f"paths must contain exactly {sorted(expected)}")
    resolved = {}
    for key,value in paths.items():
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string")
        try:
            expanded = Template(value).substitute(os.environ)
        except (KeyError,ValueError) as exc:
            raise ValueError(f"{key} contains an unset or invalid environment variable") from exc
        candidate = Path(expanded).expanduser()
        resolved[key] = (path.parent/candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    values = list(resolved.values())
    if any(a == b or a in b.parents or b in a.parents for i,a in enumerate(values) for b in values[i+1:]):
        raise ValueError("fixture, checkpoint and output roots must be separate non-nested directories")
    repo = Path(__file__).resolve().parents[2]
    for value in values:
        if value == repo or any(value.is_relative_to(repo/name) for name in ('frozen','src','tests','docs','protocol','.git')):
            raise ValueError("run paths must not point into source or historical directories")
    if pipeline.timing_backend == 'wordphase' and resolved['checkpoint_root'].is_relative_to(repo):
        raise ValueError("learned weights must be kept outside the public source repository")
    return RunConfig(pipeline,**resolved)
