# Source and dependency audit

Initial audit: 12 November 2026; external M1 follow-up: 13 November 2026.
This audit concerns the supplied source bundle and explicitly identified external
evidence. The final
manuscript, organizer score reports, original experiment environment, challenge data,
and private companion assets were not supplied in this workspace. The method description
and frozen programs establish implementation lineage; they do not establish numerical
reproduction of the submission.

The owner subsequently located accepted M1 run `20260810T200141Z` outside this
repository. Its five-fold tables, source copies, private B0 weights and 7,575 selected
motions were checked read-only; all scientific hashes and motion schemas passed.
Six stdout logs have verified append-only growth. No data or weights were copied into
source. See [External Assets](EXTERNAL_ASSETS.md) and the sanitized `protocol/m1/`
reference. Earlier statements about missing M1 outputs refer to the public bundle,
not absence from the owner's original research workspace.

A further real-data pass recovered the original pilot, P0C0/P1A/P1_FULL/P2A/MH and
P2C0/P2C1/P2C2 assets, including external P1_FULL and P2C0 launchers. Real CPU P1A
training, P2A training and 100-test A/B/selector inference now have execution evidence.
The main audit below still describes what the public source archive itself contains;
external assets and historical-script replay are tracked in
[Real Reproduction](REAL_REPRODUCTION.md).

All eight files under `frozen/` parse, and their SHA-256 digests match
[`SOURCE_INVENTORY.json`](SOURCE_INVENTORY.json). These are the public snapshot digests.
[`SOURCE_HASHES.json`](SOURCE_HASHES.json) separately records the original M1 digest:
the public snapshot already removed three workstation model-search paths before this
cleanup. Keep that distinction when comparing an original run to this release.

## Existing implementation

The original implementation comprises eight standalone scripts, about 9,100 lines.
Later scripts dynamically import a copied M1 worker from historical run directories.
Their dependencies include acceptance documents, CSV tables, distribution caches and
source copies as well as model weights. Installing Python packages does not restore
those inputs. The original `src/repointer/` contained only syntax and motion-schema
audit commands; the cleaned modules are a separate implementation.

| Frozen source | Runtime dependencies beyond Python | Principal inputs | Outputs / role |
|---|---|---|---|
| `p0c0/repointer_p0c0_shared_generation_protocol_freeze_v1_0_0.py` | None | Training annotation CSV; P0A source-resolution manifest and closeout; P0B splits, holds and acceptance; P0B-R1 acceptance/contracts; two apex models | Shared manifest, source hashes, frozen inputs and contracts, donor-pool checks, `P0C0_ACCEPTANCE.json`; no training or generation |
| `m1/repointer_b0_p0c_m1_full_fold_v0_2_0.py` | NumPy, PyTorch, `smplx`, licensed model files | P0C0 private manifest and detector models, resolved source clips, transcripts, shape and scene metadata; selected fold | Same-hand retrieval, apex warp, IK and hold; B0/B0 equal-budget/P0C clips and candidate metrics; fold acceptance and GT audit |
| `m1/repointer_b0_p0c_m1_full_5fold_aggregate_v0_2_0.py` | NumPy | All five fold outputs; P0C0 contracts; pilot acceptance and source; worker source | Coverage/leakage checks, bootstrap comparisons, quality gates, `FULL_5FOLD_ACCEPTANCE.json`; no generation |
| `timing/repointer_p1a_wordphase_timing_oof_v0_1_0.py` | NumPy, PyTorch | P0C0 manifest, M1 aggregate timing/comparison/coverage tables, timed transcripts | Five WordPhaseNet checkpoints, OOF timing predictions and acceptance; raw waveforms are not model inputs |
| `candidates/repointer_p2a_phase_router_oof_v0_1_0.py` | NumPy | Accepted P1_FULL run, per-sample median/WordPhase comparisons and parity/coverage audits | Five cross-fitted ridge routers, full-data router, OOF routes and acceptance |
| `candidates/repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py` | NumPy, PyTorch; imported M1 adds `smplx` | P0C0, M1 source/timing/detectors, P1A five checkpoints and predictions, P1_FULL candidate tables, P2A routes; P2B closeout | OOF probability cache, four timing hypotheses, multiple donor candidates, cross-fitted ranker, whole selected clips and acceptance |
| `challenge/repointer_p2c1_official_test_materialization_v0_1_0.py` | NumPy, PyTorch; imported M1 adds `smplx`; organizer validator subprocess | Accepted upstream runs, official manifest and P2C0 audited input paths, body shapes, licensed assets, copied M1/MH sources | Five-checkpoint timing ensemble, full ranker, world-aligned A/B clips, validator logs and local ZIPs; not organizer scores |
| `selector/repointer_p2c2_spatial_first_detector_robust_candidate_c_v0_1_0.py` | NumPy; imported M1 adds PyTorch/`smplx`; organizer validator subprocess | Complete P2C1 A/B outputs, planned-frame/world-alignment audit, detector weights, shapes/scenes, licensed assets | Detector-robust metrics, guarded A/B decision, exact selected file copies and final candidate ZIP |

The stage paths in the table are relative to `frozen/`. See
[`SOURCE_STAGE_MAP.md`](SOURCE_STAGE_MAP.md) for the scientific lineage.

## Input and output contracts

- **Protocol and leakage:** P0C0 records 2,525 training examples and five group-disjoint
  folds. Natural recording IDs and duplicate annotation signatures must remain together.
  A held-out query retrieves only natural, same-hand donors from other folds. Historical
  P0C0 requires at least 100 natural donors per hand/fold; M1 defaults to top eight.
  Held-out peak/hold/alignment labels are for evaluation, not generation.
- **Motion:** SMPL-X axis-angle poses have shape `(T, 165)` or `(T, 55, 3)`;
  translations have shape `(T, 3)`. All entries must be finite and `T` must equal the
  query clip length. Historical official materialization writes only `poses` and
  `trans`, as float32. Array validity alone says nothing about physical accuracy.
- **Geometry:** Hand `0` means left, `1` means right. Targets distinguish pelvis-relative
  coordinates, fixed world points and optional scene AABBs. Shape uses ten betas and
  an explicit gender/model policy. A forearm ray starts at the active elbow and points
  towards the active wrist. Target frame conventions must match motion translations.
- **Timing:** Frozen feature extraction uses 30 fps. Normalized apex ratio uses
  `frame / (T - 1)`; transcript clip duration uses `T / fps`. Natural transcripts may
  use recording-relative timestamps and require `global_start / fps` subtraction.
  Learned expected ratios become rounded frames, clipped to `[2, T - 3]` for `T > 4`.
  Transcripts are word intervals, not audio embeddings or raw samples.
- **Warp and hold:** Piecewise temporal resampling maps the donor apex exactly to
  the planned query apex. A shape-aware local clavicle/shoulder/elbow/wrist correction
  is spread around the planned apex with a plateau and taper. This is retrieval plus
  adaptation, not a residual/diffusion generator.
- **Candidate A:** Generate and rerank separate top-eight median and WordPhase-expected
  pools; a 53-feature ridge router predicts WordPhase-minus-median local STG utility.
  Select the learned route when predicted delta is at least zero. Historical official
  tests use all-training natural medians by hand; OOF training uses fold-specific
  domain/hand statistics.
- **Candidate B:** Four timing hypotheses and four donor slots per hypothesis give a
  budget of 16, including complete Candidate A as the explicit fallback. Reserve one
  slot for A, generate 15 alternatives, predict candidate utility and apply quality/
  spatial guards before replacing A. Ranker standardization is fitted on training
  folds. Ridge fitting weights both leakage groups and queries.
- **Official timing ensemble:** P2C1 averages five fold-checkpoint distributions.
  The historical un-packed bidirectional GRU is padding-sensitive: test examples
  remain in manifest order, with all 100 examples in one batch of at most 128 per
  checkpoint. Changing padding or batching may change predictions.
- **World alignment:** P2C1 adds a single translation offset,
  `reference_pelvis_world - pelvis_at_planned`, to every frame. Pose angles remain
  unchanged. This requires body-model FK and an audited world/pelvis convention.
- **Selector:** P2C2 scores planned/detected-frame neighborhoods with relative and
  fixed-world spatial proxies, emphasizing detector floors and lower quantiles.
  Candidate B replaces A only on guarded robust utility gain (default `0.001`).
  The selector copies one complete candidate unchanged; it does not mix frames,
  blend poses or rerun IK.
- **Evaluation:** Local spatial/temporal proxies and advisory apex detectors are not
  the private organizer scorer. Oracle-peak and self-detected results are distinct.
  Submission schema validation does not validate the reported paper scores.

## Checkpoint inventory and compatibility

No model checkpoints are present in the public bundle. Private companion presence
reported in older documentation is historical evidence, not a check of current files.

| Asset | Frozen names / payload | Required for |
|---|---|---|
| Apex surrogates | `apex_detector_model_v1_0_0.npz`, `apex_detector_model_v1_1_0.npz`; coefficients, intercept, feature mean/std, variant metadata | M1 reranking, candidate metrics and robust selector |
| Timing | `wordphasenet_fold0.pt` through `wordphasenet_fold4.pt`; `model_state`, word/clip normalization, domain/hand medians, bins, hash buckets, fold/version | OOF timing and official five-model ensemble |
| Phase router | `phase_router_fold*.npz`, `phase_router_full_model.npz`; beta, feature mean/std/names, lambda and decision | Candidate A |
| Candidate ranker | `multi_hypothesis_ranker_fold*.npz`; P2C1 creates `multi_hypothesis_ranker_full_model_v0_1_0.npz` with feature names and replacement threshold | Candidate B |
| B0 controller | M1 trains/saves a small control-anchor MLP for the internal baseline | Historical controlled comparison, not final RePointer |
| SMPL-X | Separately licensed model files and required supporting model data | Shape-aware FK and production IK |

WordPhaseNet has a 24-dimensional hashed-token embedding, ten word features,
a 64-dimensional projection, one 64-hidden-unit bidirectional GRU, attention pooling,
64-dimensional clip projection and a 64-bin head. Fifteen base clip features gain
a training-domain/hand median feature before inference. Feature ordering, token hashing,
normalization, padding and state-dictionary names all belong to the checkpoint contract.
Cleaned checkpoint formats must be validated or explicitly migrated before loading
historical weights; matching layer dimensions is insufficient.

Historical `.pt` loading permits pickle (`weights_only=False` in the MH helper).
Only load known, trusted historical files. Cleaned synthetic examples do not require
historical checkpoint deserialization.

## Scoped local review probes

Independent asset-free probes on 12 November 2026 compared cleaned deterministic
helpers with the imported frozen M1 reference over 20 random valid clips. Piecewise
time maps, warped pose arrays and tapered-hold pose arrays matched exactly. The largest
absolute translation difference was `2.98e-08`; descriptor difference was `8.30e-08`
after the cleaned query's float32 target conversion. These probes cover only those
deterministic NumPy helpers, not donor corpus identity, learned models, shape-aware
SMPL-X IK, full M1 outputs or official metrics.

Eight additional synthetic-skeleton IK probes covered both hands and four targets.
All reduced the pointing angle; inactive pose coordinates remained unchanged and
active angle deltas stayed within the configured bound. Final errors ranged from
`0.59` to `4.80` degrees after 30 steps. This verifies behavior of the synthetic test
double only.

A separate temporary-directory fixture run used seed `19`, two retrieval donors,
two donors per hypothesis and three IK steps. It trained eight queries across three
folds and generated two inference queries. Candidate B filled its configured budget
of eight, and each selected file matched its chosen complete A/B file byte for byte.
Repeating fixture creation, training or inference against the same destinations
raised `FileExistsError`. This bounded run does not use challenge data or licensed
models. See the main progress report for the final pytest/CLI checks and their exact
commands; these review probes are narrower than an end-to-end scientific parity test.

## Dependency and portability gaps

The frozen scripts import only NumPy, PyTorch and `smplx` beyond the Python standard
library. No pandas, SciPy or scikit-learn import is required by these eight sources.
`smplx` and PyTorch can introduce additional transitive dependencies; their complete
historical lockfile, CUDA/cuDNN versions, hardware and exact body-model package are
not supplied. Unpinned minimum requirements do not establish numerical equivalence.

At the start of this audit, the available `python3` environment had NumPy and pytest;
PyTorch and `smplx` were absent. A plain `python` executable was unavailable. This is
an environment observation, not a portable installation instruction. Use the documented
virtual environment and record the actual versions used for each run.

Further portability issues in the immutable history:

- Run directories are executable inputs: later stages expect `_code/` copies, exact
  artifact filenames and specific acceptance statuses. Some scripts search a user's
  Downloads directory for stage source when the run copy is missing.
- Manifest `source_files`, transcript/body-shape paths and source-binding/environment
  reports can contain absolute paths and host metadata. Keep real run artifacts
  private; sanitize any report before publishing it.
- Frozen materializers remove/recreate their own submission directories and ZIP files.
  Run them only in a fresh, isolated output directory, away from original research runs.
- Torch and body-model imports are deferred in several scripts. Successful `--help`,
  AST parsing or static symbol checks do not establish model execution.
- P2A-MH requires `P2B_ACCEPTANCE.json` with status
  `P2B_RESIDUAL_PROJECTION_NO_MATERIAL_HEADROOM`. This is a historical closeout gate;
  the residual pilot is not executed by the final method and must not be added to it.

## Evidence needed for independent full reproduction

The following remain required, independent of the cleaned synthetic workflow:

1. Authorized HSI dataset, official test manifest/validator, transcripts, shape and
   scene files, and separately acquired licensed SMPL-X assets.
2. Private P0C0 shared manifest, exact folds, holds, source bindings and apex models.
3. Access to the externally located M1 five-fold motions, candidate/timing/retrieval/
   comparison tables, GT audits, source copies, coverage and acceptance, or an
   independently validated reconstruction. These are deliberately not bundled.
   Rebuilding the aggregate additionally needs original pilot closeout/source.
4. P1A checkpoints, normalization and exact OOF transcript batching; P1_FULL motion
   comparisons/candidate tables, parity/coverage audits and acceptance. A P1_FULL
   generation launcher is not among the eight supplied frozen sources.
5. P2A OOF routes and full router; P2A-MH OOF distribution cache, ranker/calibration
   evidence and outputs; the historical P2B closeout if replaying its exact launcher.
6. P2C0 audited test-input manifest and acceptance. A P2C0 launcher is not among the
   eight supplied sources. Do not fabricate an acceptance file to bypass that gap.
7. P2C1 A/B output evidence, P2C2 decisions and selected-file hashes, final submission
   identity and organizer evaluation report; compare actual rerun results to them.
8. Final manuscript/supplement and final configuration to audit prose, hyperparameters,
   ablations and submission identity against the implementation.

Reconstructing missing artifacts from authorized data creates a new documented run.
It does not recover a missing historical run. See
[`REPRODUCIBILITY_STATUS.md`](REPRODUCIBILITY_STATUS.md) for current execution evidence
and [`DATA_AND_LICENSE.md`](DATA_AND_LICENSE.md) for distribution constraints.
