# Reproducibility status

The synthetic workflow and real historical-stage replay run locally. **The real
100-test inference chain has passed the official local validator; full from-scratch
paper reproduction and organizer-server scores remain unverified.** See
[Real Reproduction](REAL_REPRODUCTION.md) for the measured results and exact scope.

| Component | Current evidence |
|---|---|
| Eight frozen sources | Present; exact SHA-256/size checks against original public inventory |
| Cleaned timing, motion, candidate, ranker and selector modules | Implemented; synthetic unit/integration tests and selected frozen-function comparisons |
| Asset-free train/infer/smoke CLI | Executed with synthetic-prior timing, invented donors, test skeleton and motion-only heuristics |
| Learned WordPhaseNet | Historical real-data CPU training completed: 2,525 examples, five folds, 60 epochs each; native host optional test remains skipped |
| Licensed SMPL-X FK/IK | Historical backend executed on real inputs for all 100 regenerated A/B pairs; cleaned adapter not independently certified |
| P0C0 shared manifest/folds/holds/apex detectors | Located externally; bound non-motion files checked; not bundled |
| Historical P1A/router/ranker checkpoints | Located externally; all 19 P1A archive entries verified; final full-ranker retraining matches history to floating-point precision; not bundled |
| M1 full five-fold run, selected clips and acceptance | Located externally as run `20260810T200141Z`; all 7,575 selected motions integrity/schema checked; not bundled |
| Original M1 pilot closeout/source | Located externally; full M1 training reconstruction was not rerun |
| P1_FULL comparison/candidate/parity/coverage/GT audit run | Located and checked externally; its launcher was recovered from the historical run; not added to frozen archive |
| P2C0 official test input audit/acceptance | Located and checked externally, including its launcher; not bundled |
| HSI data, official manifest/validator, SMPL-X models | Owner's authorized assets located externally and used in CPU replay; other researchers must obtain their own authorized access |
| P2A router training replay | All 2,525 choices match history; max parameter difference about `2.44e-15` |
| Final 100-test generation/selection replay | Regenerated A/B and selector completed; all choices match, 14 B selections, max final pose difference about `1.73e-6` radians |
| Final paper, supplement, submission and organizer score report | Not supplied here; reported numbers in README come from incoming documentation |
| Exact byte parity | Historical-A/B selector replay: 100/100 selected files match; fresh CPU-generated motions differ slightly from historical CUDA output |

Current results and commands are in [Progress](PROGRESS.md) and
[Local Validation](LOCAL_VALIDATION.json). That record distinguishes this run from
older archive checks. [Source Audit](SOURCE_AUDIT.md) lists all inputs/checkpoints and
[Data and Licensing](DATA_AND_LICENSE.md) states redistribution limits.

## External M1 evidence

On 13 November 2026 the owner identified the original M1 archive. Read-only checks
confirmed accepted five-fold status, 2,525 unique samples, 42,925 candidates, no
observed identity/donor leakage, and 7,575 selected motion files with matching hashes
and valid arrays. The copied worker matches the documented original digest. Six
stdout logs grew only by appending bytes after their manifest snapshots.

This resolves M1 artifact discovery. It does not add those files to the public
repository, supply the remaining stages, or establish a new generation/evaluation
run. Configure it through [External Assets](EXTERNAL_ASSETS.md); the path-free
reference is [external_run_reference.json](../protocol/m1/external_run_reference.json).
Another researcher must obtain compatible authorized outputs or rebuild them.

## Real replay follow-up

The previously unlocated P0C0/P1A/P1_FULL/P2A/MH/P2C0/P2C1/P2C2 stages and original
M1 pilot evidence were subsequently found in the owner's external workspace. The
P1_FULL and P2C0 launchers are preserved there; the public eight-file frozen archive
was not expanded or modified. Required data and licensed assets are also present
privately. P1A retraining, P2A retraining and real 100-test inference replay have now
run in independent private directories.

This changes asset discovery and local execution status. It does not establish a
full from-scratch training reproduction, final-manuscript audit, checkpoint-release
permission or freshly measured organizer-server scores. Follow the separate
`replay` instructions in [Real Reproduction](REAL_REPRODUCTION.md).

## What the tests establish

Tests check shapes, finite arrays, seconds/frame conversion, preprocessing parity,
leakage exclusions, quaternion warp and hold behavior, synthetic IK, ridge fitting and
cross-fitting, candidate budgets, selector guards, exact A/B byte copying, checkpoint
bindings and the CLI. A stubbed SMPL-X call verifies all 165 pose slices without assets.
Local spatial/temporal proxies are distinct from the private organizer evaluator.

Synthetic training uses a fitted timing prior unless `timing_backend=wordphase` is
explicitly configured. Its skeleton and detectors are test doubles. Reduced budgets,
fixed calibration and staged OOF diagnostic losses have no benchmark performance
meaning. The staged feature tables are not a fully nested end-to-end validation
estimate. No private checkpoints or protected motions are included.

## Path to full paper reproduction

1. Obtain the final manuscript, supplement, exact final configuration, original
   submission identifiers and organizer evaluation evidence from the authors.
2. Acquire authorized HSI data, word transcripts, scene/shape inputs and the exact
   test manifest/validator. Acquire SMPL-X directly under its license. Keep these and
   all real learned weights outside this repository; use explicit environment paths.
3. Restore P0C0's accepted manifest, folds, holds, source bindings and two detector
   checkpoints. Verify hashes, 2,525-sample coverage, five folds, duplicate signatures
   and recording/group isolation. The public acceptance JSON alone is insufficient.
4. Restore the original environment if available, or record a new one: Python, NumPy,
   PyTorch, `smplx`, transitive packages, CUDA/cuDNN, model files, hardware and seeds.
   Compatibility ranges in this package are not historical pins. Record private
   asset hashes without publishing paths, labels or restricted parameters.
5. Configure and audit the externally located M1 run, or regenerate five M1 folds
   using authorized inputs and recovered pilot/worker evidence. Keep original runs
   read-only. A new reconstruction still needs valid coverage, leakage, quality and
   GT-usage audits; the public reference is not a replacement for those inputs.
6. Restore/train the five P1A timing checkpoints, then restore/reconstruct P1_FULL
   generation and audits. Compare timed-word hashes/features, normalization, apex
   distributions and expected frames before downstream motion comparisons.
7. Restore P2A OOF routes/full router, then P2A-MH distributions, calibration evidence
   and fold rankers. Validate donor eligibility and selection for every fold. Its
   historical launcher also checks a P2B closeout acceptance, but does not execute
   the residual method. Recover that evidence; do not fabricate acceptance records.
8. Restore/reconstruct P2C0's audited official input manifest and acceptance. Verify
   no hidden labels enter test generation. Restore P2C1 source copies and reproduce
   the five-checkpoint ensemble in original manifest order/batches. Historical BiGRU
   padding changes predictions; keep the 100 tests in one batch per checkpoint.
9. Materialize Candidate A and Candidate B, audit fixed-world/pelvis alignment and
   exact shapes, then run P2C2. Verify each selected output is byte-identical to
   complete A or B. Preserve candidate counts, source/checkpoint/configuration hashes,
   decisions and validator logs in a private run directory.
10. Run the official local validator; compare the resulting ZIP and intermediates to
    historical evidence. Obtain organizer scores through the authorized evaluation
    route before claiming official reproduction. Evaluate naturalness separately.

Recreated missing artifacts are a **new run**, not recovered historical evidence.
Cleaned historical-checkpoint migration, full frozen detector integration, a general
HSI manifest adapter and intermediate numerical parity are still required before
substituting `src/repointer/` for the historical runner.

## Historical command templates

These templates describe the archived CLIs after all dependencies are configured.
The P1A/P2A/P2C1/P2C2 stages have now run on real HSI inputs as documented in
[Real Reproduction](REAL_REPRODUCTION.md); the full M1/P1_FULL/MH training chain
has not been regenerated. Set the uppercase variables to private paths and fresh
output directories. Do not point outputs at original research runs.
The frozen launchers can overwrite their own outputs; the cleaned no-overwrite policy
does not change immutable historical behavior.

```bash
# After restoring accepted P0C0 and the original pilot evidence:
for fold in 0 1 2 3 4; do
  python frozen/m1/repointer_b0_p0c_m1_full_fold_v0_2_0.py \
    --data-root "$DATA_ROOT" --project-root "$PRIVATE_ROOT" \
    --p0c0-dir "$P0C0_DIR" --smplx-model-dir "$SMPLX_MODEL_DIR" \
    --output-dir "$NEW_M1_RUN/folds/fold_$fold" --fold "$fold" \
    --queries-per-slice 0 --top-k 8 --device cpu
 done

python frozen/m1/repointer_b0_p0c_m1_full_5fold_aggregate_v0_2_0.py \
  --run-dir "$NEW_M1_RUN" --p0c0-dir "$P0C0_DIR" \
  --pilot-dir "$PILOT_DIR" --pilot-source "$PILOT_SOURCE" \
  --worker-source frozen/m1/repointer_b0_p0c_m1_full_fold_v0_2_0.py

python frozen/timing/repointer_p1a_wordphase_timing_oof_v0_1_0.py \
  --data-root "$DATA_ROOT" --p0c0-dir "$P0C0_DIR" \
  --m1-full-dir "$NEW_M1_RUN" --output-dir "$NEW_P1A_RUN" \
  --epochs 60 --batch-size 128 --bins 64 --hash-buckets 2048 --max-words 64

# BLOCKER: P1_FULL source/artifacts must be recovered before proceeding.
python frozen/candidates/repointer_p2a_phase_router_oof_v0_1_0.py \
  --p1-full-dir "$P1_FULL_DIR" --output-dir "$NEW_P2A_RUN" --ridge-lambda 0.5

python frozen/candidates/repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py \
  --data-root "$DATA_ROOT" --project-root "$PRIVATE_ROOT" \
  --p0c0-dir "$P0C0_DIR" --m1-full-dir "$NEW_M1_RUN" \
  --p1a-dir "$NEW_P1A_RUN" --p1-full-dir "$P1_FULL_DIR" \
  --p2a-dir "$NEW_P2A_RUN" --p2b-dir "$P2B_CLOSEOUT_DIR" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" --output-dir "$NEW_MH_RUN" \
  --queries-per-slice 0 --hypothesis-count 4 --donors-per-hypothesis 4 \
  --wordphase-batch-size 128

# BLOCKER: P2C0 audited inputs/source/acceptance must be recovered.
python frozen/challenge/repointer_p2c1_official_test_materialization_v0_1_0.py \
  --data-root "$DATA_ROOT" --project-root "$PRIVATE_ROOT" \
  --p0c0-dir "$P0C0_DIR" --m1-full-dir "$NEW_M1_RUN" \
  --p1a-dir "$NEW_P1A_RUN" --p1-full-dir "$P1_FULL_DIR" \
  --p2a-dir "$NEW_P2A_RUN" --p2a-mh-dir "$NEW_MH_RUN" --p2c0-dir "$P2C0_DIR" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" --output-dir "$NEW_P2C1_RUN"

python frozen/selector/repointer_p2c2_spatial_first_detector_robust_candidate_c_v0_1_0.py \
  --data-root "$DATA_ROOT" --project-root "$PRIVATE_ROOT" \
  --p0c0-dir "$P0C0_DIR" --m1-full-dir "$NEW_M1_RUN" --p2c1-dir "$NEW_P2C1_RUN" \
  --smplx-model-dir "$SMPLX_MODEL_DIR" --output-dir "$NEW_P2C2_RUN" \
  --replacement-threshold 0.001
```

Inspect each archived `--help` and compare arguments to final author configuration.
Successful help, syntax, local schema checks or advisory proxies cannot replace an
actual licensed model run and organizer validation.
