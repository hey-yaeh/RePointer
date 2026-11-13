"""Explicit external research-asset paths; no discovery, downloads or copying."""
from __future__ import annotations

import json
import os
from pathlib import Path
from string import Template
from typing import Mapping


ASSET_ROLES = {
    "m1_full_dir": "Accepted M1 five-fold run directory, including aggregate/ and folds/",
    "p0c0_dir": "Private shared manifest, folds, holds and apex detector weights",
    "p1a_dir": "Five WordPhaseNet timing checkpoints and OOF evidence",
    "p1_full_dir": "Accepted full timing-to-motion comparisons and audits",
    "p2a_dir": "Phase router models, routes and acceptance",
    "p2a_mh_dir": "Multi-hypothesis distributions, rankers and acceptance",
    "p2c0_dir": "Official test-input audit and acceptance",
    "p2c1_dir": "Validated historical candidate A/B materialization, for standalone selector replay",
    "dataset_root": "Authorized HSI dataset, timed words, scenes and shape inputs",
    "smplx_model_dir": "Separately licensed SMPL-X model directory",
}


def load_asset_paths(config: str | Path) -> dict[str, Path | None]:
    """Read JSON paths relative to the config, expanding only explicit variables.

    Null entries mean unconfigured. Existing directories are checked separately;
    a path declaration alone does not establish scientific readiness.
    """
    config = Path(config).resolve()
    document = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"schema", "paths"}:
        raise ValueError("external asset config requires schema and paths only")
    if document["schema"] != "repointer.external-assets.v1":
        raise ValueError("unsupported external asset config schema")
    paths = document["paths"]
    if not isinstance(paths, dict) or set(paths) - set(ASSET_ROLES):
        raise ValueError("external asset config contains unknown path names")
    module_dir = Path(__file__).resolve().parent
    repo = module_dir.parents[1]
    source_checkout = (repo / "src" / "repointer").resolve() == module_dir
    resolved: dict[str, Path | None] = {}
    for name in ASSET_ROLES:
        value = paths.get(name)
        if value is None:
            resolved[name] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string or null")
        try:
            expanded = Template(value).substitute(os.environ)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{name} contains an unset or invalid environment variable") from exc
        path = Path(expanded).expanduser()
        path = (config.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if source_checkout and path.is_relative_to(repo):
            raise ValueError(f"{name} must stay outside the public source repository")
        resolved[name] = path
    return resolved


def asset_status(paths: Mapping[str, Path | None]) -> dict:
    """Return path-free availability, never a full reproduction-ready claim."""
    return {
        "schema": "repointer.external-asset-status.v1",
        "full_reproduction_validated": False,
        "assets": {
            name: {
                "status": "unconfigured" if paths.get(name) is None else (
                    "directory_present" if paths[name].is_dir() else "missing_or_not_directory"
                ),
                "role": description,
            }
            for name, description in ASSET_ROLES.items()
        },
    }
