# Installation, training and inference

The supported runnable path initially uses invented fixtures. It exercises final-method
stage boundaries without redistributing challenge data, weights or SMPL-X models.
It does not reproduce the paper. See [Source Audit](SOURCE_AUDIT.md) for historical
contracts and [Reproducibility Status](REPRODUCIBILITY_STATUS.md) for remaining work.

## Install

Python 3.10+ and NumPy 1.24+ are required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m repointer --help
python -m repointer audit-source
python -m pytest -q
```

`repointer` and the compatibility alias `repointer-audit` also invoke the CLI. For a
source-only check with dependencies already available, use
`PYTHONPATH=src python3 -m repointer audit-source`. Pytest config adds `src` automatically.

Core commands do not import PyTorch or `smplx`. Optional extras are `.[timing]` for
WordPhaseNet, `.[models]` for the licensed backend adapter, and `.[config]` for TOML
on Python 3.10. Python 3.11+ reads TOML through the standard library. Dependency
ranges are compatibility bounds, not a recovered historical lockfile.

## Synthetic workflow

Run all steps together:

```bash
python -m repointer smoke --config configs/synthetic.json
```

Or use the individual stages with a fresh configuration:

```bash
python -m repointer make-fixture --config configs/synthetic.json
python -m repointer train --config configs/synthetic.json
python -m repointer infer --config configs/synthetic.json
python -m repointer validate-motion outputs/synthetic-inference/selected/query_h0.npz --expected-frames 34
```

Choose either the combined or individual sequence for each configuration. Both create
new directories and refuse to overwrite existing runs. After a smoke run, reuse its
training artifact by copying the configuration and changing only `output_root` to a
fresh directory before `infer`. Keep pipeline settings identical.

`configs/synthetic.json` resolves paths relative to its own location, even when invoked
from another working directory. Copying the config changes the base directory of its
relative paths. Set `fixture_root`, `checkpoint_root` and `output_root` to three separate,
non-nested directories. They must not point into source or historical directories.
`${VARIABLE}` expansion is explicit; unset variables cause an error. Seeds and all
budgets live under `pipeline`; unknown options fail validation.

The fixture uses top-two retrieval and two donor slots for each of four hypotheses,
with six IK steps. `PipelineConfig` defaults retain top-eight retrieval, four donors,
four hypotheses and 50 IK steps. Both are transparent local configurations; neither
restores missing historical run settings or assets automatically.

Generated artifacts:

| Artifact | Meaning |
|---|---|
| `synthetic-fixture/manifest.json`, `donors/*.npz` | Invented timed words, labels, shapes and donor arrays; no HSI content |
| `synthetic-training/timing.json` | Fitted Gaussian timing prior based on synthetic training medians; **not WordPhaseNet** |
| `synthetic-training/router.npz`, `ranker.npz` | Numeric ridge parameters trained on synthetic candidate diagnostics |
| `synthetic-training/training.json` | Configuration, synthetic status, staged OOF diagnostics and input/model hashes |
| `synthetic-inference/candidate_a/*.npz`, `candidate_b/*.npz` | Complete candidate clips |
| `synthetic-inference/selected/*.npz` | Exact bytes copied from the selected A or B |
| `synthetic-inference/inference.json` | Router, ranker and selector decisions, identities and output hashes |
| `synthetic-inference/synthetic_submission.zip`, `smoke.json` | Locally schema-checked package and explicit non-official smoke status |

Training uses three group-disjoint fixture folds, fitting timing statistics without each
query's fold and cross-fitting the ridge models. These staged OOF diagnostics are code
checks, not fully nested end-to-end generalization estimates. Synthetic ridge calibration
uses a fixed lambda and replacement threshold. `ranking.nested_ranker_calibration`
provides a separately tested inner-OOF calibration utility for later research integration.

Inference verifies the exact fixture manifest, donor bytes, checkpoint hashes, package
source hashes and stored configuration. Training records Python/NumPy and optional
PyTorch versions without machine paths. Inference currently runs the fixture query
split bound to that training artifact.
A general HSI manifest adapter is pending historical artifact recovery and parity work.

## Optional learned timing on fixtures

To train the actual cleaned WordPhaseNet rather than the NumPy fixture prior:

```bash
python -m pip install -e '.[test,timing]'
export REPOINTER_PRIVATE_ROOT="$(mktemp -d)"
python -m repointer smoke --config configs/wordphase.example.json
python -m pytest -q tests/test_timing.py
```

Keep this directory outside the public source repository. The command fits fold-specific
and full-data CPU timing models and records the chosen backend. It still uses synthetic
motion and detectors. PyTorch is absent in the supplied execution environment, so the
learned training/checkpoint round-trip test is explicitly skipped there.

Cleaned checkpoints use a tensor-only schema and restricted `weights_only=True` loading.
Historical checkpoints contain NumPy/pickle structures and are not directly accepted;
an audited migration is still required. Do not enable unrestricted pickle loading for
untrusted files. Real P2C1 inference also needs the five-checkpoint ensemble and historical
batch/padding behavior; this fixture runner is not its replacement.

## Input contracts and licensed backend

Frame indices are zero-based at 30 fps. Frame ratios use `T - 1`; word times are seconds
and clip duration is `T / 30`. `Query` requires at least five frames, hand `0=left` or
`1=right`, a nonzero pelvis-relative target vector, ten shape betas, gender and explicit
sample/group/recording identities. Four distinct timing hypotheses additionally require
enough interior frames. Inference queries exclude peak/hold/alignment supervision and
raw audio. Supervised labels live separately in fixture training records.

`Motion` accepts finite numeric `(T,165)` or `(T,55,3)` poses plus `(T,3)` translations
and normalizes to float32. Challenge materialization additionally requires exact keys
`poses` and `trans`, float32 dtype and exact manifest coverage. Array layout does not
certify kinematics, grounding or naturalness.

`SMPLXKinematics` is an optional low-level adapter requiring an explicit licensed model
directory. Its FK/IK path is not asset-tested here. Production candidate integration
also needs an audited motion-only detector and historical feature/checkpoint conversion;
the native `train`/`infer` CLI supports the explicit synthetic backend only. Missing dependencies
never trigger an automatic fallback. `configs/local_paths.example.toml` remains a
historical path checklist; frozen programs take explicit CLI flags and do not read it.

Full benchmark commands and missing acceptance artifacts are listed in
[Reproducibility Status](REPRODUCIBILITY_STATUS.md). Keep original research runs read-only
and use fresh output roots for any historical launcher.

The owner's historical M1 five-fold archive is now located and integrity-checked
externally. Use `assets-status` / `audit-m1` with
`configs/external_assets.example.json` to inspect it; see
[External Assets](EXTERNAL_ASSETS.md). This separate configuration never copies data
or checkpoints into source and does not change the fixture-only training/inference
scope above. The [Documentation Index](README.md) maps the remaining guides.

For actual historical training/inference, use the separate `replay` command and
`configs/historical_replay.example.json`. It runs preserved programs with an
explicit external research interpreter, using real assets and a new private output
directory. Real CPU training and 100-test inference have been executed; setup,
measured differences and limitations are in [Real Reproduction](REAL_REPRODUCTION.md).
