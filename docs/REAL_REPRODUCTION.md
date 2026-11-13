# Real-data reproduction — 13 November 2026

Real CPU experiments have now been executed with the owner's authorized dataset,
licensed SMPL-X assets and verified historical checkpoints. The final 100-test
inference chain was rerun from timed words through A/B generation and the selector.
**Organizer-server scores and an entirely retrained paper experiment remain unverified.**

The path-free machine-readable evidence is
[real_reproduction_20261113.json](validation/real_reproduction_20261113.json).
Private checkpoints, sample tables, motions, commands and logs remain outside source.
The located external stage acceptance/manifest hashes are recorded in
[external_pipeline_reference.json](../protocol/final/external_pipeline_reference.json).

## Executed experiments

| Experiment | Actual result |
|---|---|
| P1A training from initialization | 2,525 real examples, five folds, 60 epochs per fold; historical acceptance gates passed |
| P2A phase-router retraining | All 2,525 OOF route choices match history; max parameter difference `2.44e-15` |
| P2C2 replay using historical A/B | All 100 choices match; selected files are byte-identical to the historical selected files; official local validator passed |
| P2C1 regeneration | Recomputed five-checkpoint timing ensemble/full ranker, retrieved/warped donors, optimized real SMPL-X IK and regenerated 100 A plus 100 B clips; all directory/ZIP validators passed |
| P2C2 using newly generated A/B | All 100 choices match history, including 14 B selections; each selected output copies its newly generated candidate unchanged; official local validator passed |

The newly generated final outputs are **numerically close, not byte-identical**, to
the original CUDA submission. Maximum pose-component difference is
`1.7285346984863281e-6` radians; maximum translation-component difference is
`1.1920928955078125e-7` in dataset units. Mean clip pose RMSE is about `9.69e-9`
radians. These values are measured differences, not a pre-registered official gate.

One Candidate A donor ID differs; its source files, clip segments, global frame
range, clip length, source apex and hand are identical to the historical donor.
This is a source-slice alias rather than a different motion input. It is retained
explicitly in the report rather than claiming exact donor-ID parity.

Newly trained CPU timing models are a **separate experiment**. They are not substituted
into the submission replay, which uses the five historical P1A checkpoints and
historical M1/P1_FULL/MH training evidence. A new timing model requires regenerated
P1_FULL evidence and downstream models before claiming a newly trained full system.

## Environments

The package/control environment uses Python 3.12.3 and NumPy. Its optional PyTorch
test remains skipped because PyTorch is not installed there. An existing external
research environment supplies Python 3.9.23, PyTorch 2.5.1+cu118 and `smplx`.
That environment was used without installation or modification. CUDA is unavailable
in the current execution context; all reported real experiments use CPU.

The cleaned native timing API requires its separately declared modern PyTorch extra.
The replay tool instead executes the preserved historical programs with an explicitly
chosen research interpreter. It does not certify the cleaned SMPL-X adapter or migrate
legacy checkpoint formats. Only trusted, hash-verified historical checkpoints were
used; their legacy deserialization is not an interface for untrusted downloaded files.

## Repeat the pretrained inference chain

Configure environment variables named by `configs/historical_replay.example.json`.
Each must refer to authorized external inputs. Set `REPOINTER_RESEARCH_PYTHON` to
your research interpreter and `REPOINTER_REPLAY_OUTPUT` to a fresh directory outside
the public source tree and every input archive.

```bash
PYTHONPATH=src python3 -m repointer replay \
  --config configs/historical_replay.example.json \
  --python "$REPOINTER_RESEARCH_PYTHON" --output-root "$REPOINTER_REPLAY_OUTPUT" \
  --dry-run

PYTHONPATH=src python3 -m repointer replay \
  --config configs/historical_replay.example.json \
  --python "$REPOINTER_RESEARCH_PYTHON" --output-root "$REPOINTER_REPLAY_OUTPUT"
```

The default stages are `p2a,p2c1,p2c2`. The new router feeds the new A/B materializer,
and its outputs feed the new selector. Historical stage sources are verified against
`SOURCE_INVENTORY.json`. Execution uses CPU, 32-frame FK batches, the original
64-candidate IK batches, original model hyperparameters and four CPU threads.
The tool does not write into inputs, permit existing output roots or change frozen
code. Child interpreters run with bytecode writes disabled.

Raw logs and `commands.private.json` may contain local paths and labels; keep the
entire output root private. The printed summary omits private paths and sample IDs.
A `PASS` means the requested historical stage acceptance checks completed; it does
not imply numerical parity or organizer-server scoring. Compare new intermediates
and output arrays explicitly, as done for this report.

To retrain timing separately, choose another fresh output directory and use
`--stages p1a`. Combining P1A retraining with the pretrained inference chain is
rejected because its P1_FULL training evidence would otherwise be inconsistent.
For standalone selector replay, configure `p2c1_dir` and use `--stages p2c2`.

## Remaining work

- Audit the final manuscript and original organizer report before asserting all paper
  numbers and ablations are reproduced. No submission was uploaded in this work.
- Rebuild M1/P1_FULL/MH motion-training artifacts if full from-scratch training
  reproduction is the objective; they were reused here as verified historical inputs.
- Test the cleaned native backends with migrated historical checkpoints, independently
  of historical-script replay, and validate the intended GPU environment separately.
- Establish redistribution permission for any checkpoint or derived artifact release.
  MIT source licensing and public source publication are owner-approved; third-party
  asset redistribution remains separate.
