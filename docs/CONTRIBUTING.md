# Working on RePointer

Keep `frozen/` immutable. Run `python -m repointer audit-source` before and after
changes; do not update the archived hashes to make changed source pass. Cleaned
implementations and interfaces belong in `src/repointer/`, with tests in `tests/`.

Use invented fixtures for tests. Keep actual challenge data, derived labels, weights,
SMPL-X files, credentials and run outputs outside public version control. The public
fixture runner uses a reduced budget and synthetic kinematics/detectors. A passing
fixture test is not a paper or challenge reproduction.

Before proposing a change, run `python -m pytest -q` and the synthetic smoke command
in [Getting Started](GETTING_STARTED.md). Add tests for meaningful stage invariants
and boundaries. Keep inference queries separate from supervision and evaluation
labels. Compare scientific transforms to frozen code where assets are unnecessary.

Record dependencies, seeds, configurations, input hashes and numerical differences
when evaluating a real backend. Do not silently substitute a synthetic model when
licensed models are missing. Do not add the excluded residual/diffusion pilot to the
main pipeline. Review generated artifacts for local paths before sharing them.

Use `assets-status` and `audit-m1` to inspect configured private artifacts. Keep
archive audits read-only and write reports outside their inputs. In audits and tests,
do not import archived `_code/` files, enable pickle loading, copy real motions, or
replace provenance hashes to silence a failed audit. New audit tests must use
invented archives. See [External Assets](EXTERNAL_ASSETS.md).

Real `replay` runs require an explicitly selected research interpreter and a fresh
output root outside source and all inputs. Keep commands/logs and checkpoint outputs
private; publish only path-free aggregate evidence. Do not combine newly trained
timing weights with historical P1_FULL features without regenerating that evidence.

Contributions to RePointer-owned source use the [MIT License](../LICENSE). Retain
third-party notices and their original license terms. Checkpoint/data redistribution
requires separate approval. CI configuration is provided; remote workflow results
must be checked after publication rather than inferred from local tests.
