# Stage lineage and evidence limits

- P0C0 (`shared_generation_protocol_freeze`): frozen shared folds, data boundaries,
  proxy definitions and output schema; it does not generate motion.

- M1 (`b0_p0c_m1_full_fold_v0_2_0`): core retrieval, geometric retargeting, time warp and internal local-proxy evaluation.
- M1 aggregate: aggregates completed M1 five-fold runs; **not** a generation worker.
- P1A (`wordphase_timing_oof`): independent WordPhaseNet 5-fold supervised timing stage.
- P2A (`phase_router_oof`): offline training/selection of the two conservative timing routes.
- P2A-MH (`multi_hypothesis_timing_pilot`): multiple apex/donor candidates and a cross-fitted ranker. Archived source from accepted run 20260812T065037Z.
- P2C1 (`official_test_materialization`): uses frozen stage artifacts for official test candidate A/B materialization.
- P2C2 (`spatial_first_detector_robust_candidate_c`): final robust A/B selector, copies the selected complete motion.

The separate P2B residual-projection pilot is **not** part of the final submitted method. V3 residual-generator and 9-hypothesis scorer scripts are intentionally excluded.

The historical P2A-MH launcher checks a P2B closeout acceptance as provenance only.
Do not treat that artifact dependency as part of generation. Cleaned interfaces are
mapped in [Architecture](ARCHITECTURE.md); exact runtime inputs and missing artifacts
are listed in [Source Audit](SOURCE_AUDIT.md).
