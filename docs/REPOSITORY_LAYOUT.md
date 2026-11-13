# Public source and external research assets

Keep the established stage names and filenames so historical references remain
auditable. Cleaned code belongs in `src/repointer/`; `frozen/` remains immutable.

```text
RePointer source repository/
  README.md                    Entry point and evidence limits
  pyproject.toml               Core dependencies and optional model extras
  src/repointer/               Cleaned implementation and read-only audits
  frozen/                     Eight immutable historical programs
  protocol/
    p0c0/                     Public shared protocol contracts
    m1/                       Sanitized external-run reference
  configs/
    synthetic.json            Generated-data workflow
    wordphase.example.json    Optional learned timing on fixtures
    external_assets.example.json  Explicit private asset paths
  tests/                      Invented fixtures and stage/integrity tests
  docs/                       Instructions, provenance and audit summaries
  .github/workflows/          CI definitions; no remote run implied
  outputs/                    Ignored local synthetic runs and audit reports

External private workspace/   Not copied, symlinked or committed into source
  authorized dataset and transcripts
  licensed SMPL-X assets
  historical M1/P1/P2 run directories
  private model checkpoints and per-sample evidence
```

## What to keep where

| Material | Public source repository | External private directory |
|---|---|---|
| Algorithm source, tests and config templates | Yes | Optional historical copies |
| Public protocol definitions | Yes | Original private bindings |
| Aggregate audit status, counts and provenance hashes | Sanitized reference only | Full original records |
| Dataset, transcripts, sample manifests and labels | No | Yes |
| Generated challenge-derived motions and per-sample metrics | No | Yes |
| Learned weights, B0 checkpoints and SMPL-X files | No | Yes |
| Original logs/environment reports with local paths | No | Yes |
| Invented smoke outputs and temporary test parameters | Ignored local outputs only | Optional |

`.gitignore` excludes run directories, research tables, media, model formats,
credentials and local path configs. `MANIFEST.in` includes source/documentation
explicitly and excludes those formats from source distributions. These rules do
not grant redistribution rights or remove files already tracked in a real Git
repository. Review the actual staged files before any owner-authorized publication.

No usable Git history is available in the supplied checkout: `.git` is an empty,
read-only directory. Repository organization and package checks can proceed, but
creating commits in that directory requires writable Git metadata. An independent
local source snapshot and verified Git bundle have been prepared. The owner approved
MIT licensing and public source publication. The public source is hosted at
[hey-yaeh/RePointer](https://github.com/hey-yaeh/RePointer).

## Reproduction entry points

Use `make-fixture` / `train` / `infer` / `smoke` for executable synthetic checks.
Use `assets-status` and `audit-m1` for external dependency availability and integrity.
The latter do not import historical programs, load `.pt` models or execute training.
An archived `_code/` worker is hashed as evidence and remains in its original run.

Use `replay` for explicit real historical-stage execution, including trusted legacy
checkpoints and licensed body models in a separate research environment. All raw
outputs stay private. It is distinct from read-only auditing and native synthetic
execution; see [Real Reproduction](REAL_REPRODUCTION.md).

The reference run is not a substitute for a reconstruction path. Researchers need
either authorized access to compatible outputs or the scripts, inputs and protocol
evidence to regenerate them. The remaining reconstruction gaps are explicit in
[Reproducibility Status](REPRODUCIBILITY_STATUS.md).
