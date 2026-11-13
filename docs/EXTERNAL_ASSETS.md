# External assets and historical M1

The accepted M1 run `20260810T200141Z` was found in the owner's private research
workspace on 13 November 2026. Its data, generated motions, sample tables, logs and
weights remain outside the public repository. Only aggregate audit status and a
small provenance reference are included under `protocol/m1/`.
The [full M1 audit snapshot](validation/m1_external_audit_20261113.json) and
[configured-asset snapshot](validation/external_asset_status_20261113.json) record
the checks performed without disclosing any private directory or sample identity.

## Configure paths

Set `REPOINTER_M1_FULL_DIR` to your existing accepted run directory using a shell
environment variable. It must contain `FULL_5FOLD_ACCEPTANCE.json`, `aggregate/`,
`folds/fold_0/` through `folds/fold_4/`, and the historical `_code/` copies. Point to
the timestamped run, not its parent archive or `runs/` container.

```bash
python -m repointer assets-status --config configs/external_assets.example.json
python -m repointer audit-m1 --config configs/external_assets.example.json \
  --reference protocol/m1/external_run_reference.json
```

For a one-off directory check, use an explicit shell variable:

```bash
python -m repointer audit-m1 --run-dir "$REPOINTER_M1_FULL_DIR" \
  --reference protocol/m1/external_run_reference.json
```

You may copy the template to `configs/external_assets.local.json` and configure
additional paths. `*.local.json` and `*.local.toml` are ignored and excluded from
source distributions. Do not commit real machine paths. This external-assets
configuration uses JSON and a separate schema from the synthetic JSON/TOML runner.

Paths resolve relative to the config file, and `${VARIABLE}` expansion fails if an
explicitly referenced variable is unset. `null` means unconfigured. All actual asset
paths must resolve outside this source tree; ignored private subdirectories inside
the repository are not supported asset destinations. No private-path file or symlink
is needed in the public source tree.

## Check scope and results

`assets-status` prints path-free availability for each named dependency. A successful
command means the status was read; `directory_present` establishes only existence.
It does not imply valid checkpoints, compatible schemas or full reproduction readiness.

By default, `audit-m1` checks:

- Required five-fold layout, acceptance/GT audits and per-fold completion markers.
- Sample coverage and cross-fold group/recording/signature isolation.
- Natural same-hand training-fold donors and timing-label partition boundaries.
- Candidate budgets and consistency of combined/fold tables.
- Selected-motion file coverage and manifest sizes.
- Non-motion file hashes, fold/root manifest consistency and reference evidence hashes.

To hash and inspect all selected motion arrays, run:

```bash
python -m repointer audit-m1 --config configs/external_assets.example.json \
  --reference protocol/m1/external_run_reference.json \
  --verify-hashes --verify-motions --report outputs/m1-audit.json
```

This reads all motion bytes and can take several minutes. Arrays must have exact
`poses` / `trans` keys, float32 dtype, expected frame counts, finite values, and the
public SMPL-X layout. The report states precisely whether motion hashes and arrays
were checked. Output reports omit local paths, sample IDs, transcripts and labels.
An existing report is never overwritten; choose another output filename on rerun.
Reports cannot be written into the audited asset directories or `frozen/`.

Append-only `.log` growth is reported separately when its recorded prefix still
matches the historical hash. A changed prefix or scientific file fails the audit.
No checkpoint is deserialized, archived code executed, data copied or output motion
regenerated. `PASS` is an integrity result, not official evaluation or renewed author
acceptance of a changed experiment.

The public reference binds one known historical archive. A newly generated run should
produce its own audited evidence; a reference hash mismatch does not justify changing
the original reference or fabricating historical acceptance files. Without `--reference`,
the report explicitly says it is not bound to the known historical run.

## What has been located

| Evidence | Verified historical archive |
|---|---|
| Acceptance | `B0_P0C_M1_FULL_5FOLD_ACCEPTED` |
| Samples | 2,525 unique samples over five folds |
| Fold sizes | 503 / 509 / 506 / 506 / 501 |
| Candidate and retrieval rows | 42,925 each |
| Selected motions | 7,575: B0, B0 equal-budget and P0C for each query |
| Integrity | Scientific tables, source copies, weights and selected-motion hashes match |
| Motion schema | All 7,575 passed frame/shape/float32/finite-value checks |
| Logs | Six files have append-only growth with matching recorded prefixes |
| Source lineage | Copied worker matches the documented original M1 digest; current public frozen worker retains its pre-existing path sanitization |

Historical results are local proxy evidence. Neither locating this archive nor
validating its files establishes newly reproduced paper or organizer scores.

## Requirements for another researcher

For training and experimental reproduction, downstream historical programs consume
M1 tables/coverage/source copies. Researchers need compatible outputs or a complete
way to generate them from authorized inputs. The original pilot evidence and
P1_FULL/P2C0 launchers/artifacts are now located externally, but are not bundled.
Full from-scratch motion-training reconstruction has not been rerun.

Inference should ultimately depend on a donor bank, model parameters, licensed
kinematics and validated inputs rather than requiring every training-fold motion.
The `train`/`infer` commands still use synthetic fixtures. The separate `replay`
command supports real historical-stage execution; a 100-test CPU replay has now
passed local validation. See [Real Reproduction](REAL_REPRODUCTION.md).
