from pathlib import Path
import ast
import pytest

ROOT = Path(__file__).resolve().parent.parent
NEEDED = {
  "frozen/m1/repointer_b0_p0c_m1_full_fold_v0_2_0.py": {"SMPLXEngine", "warp_clip", "piecewise_positions", "descriptor", "optimize_apex_batch", "apply_hold", "evaluate_candidate"},
  "frozen/timing/repointer_p1a_wordphase_timing_oof_v0_1_0.py": {"WordPhaseNet", "make_soft_targets", "train_one_fold"},
  "frozen/candidates/repointer_p2a_phase_router_oof_v0_1_0.py": {"build_features", "fit_weighted_ridge"},
  "frozen/candidates/repointer_p2a_mh_multi_hypothesis_timing_pilot_v0_1_2.py": {"build_hypotheses", "fit_weighted_ridge"},
  "frozen/challenge/repointer_p2c1_official_test_materialization_v0_1_0.py": {"formal_materialization", "create_submission_zip"},
  "frozen/selector/repointer_p2c2_spatial_first_detector_robust_candidate_c_v0_1_0.py": {"evaluate_motion", "choose_candidate", "formal_run"},
}
@pytest.mark.parametrize('rel,required',NEEDED.items())
def test_frozen_stage_symbols(rel,required):
    path = ROOT / rel
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    missing = required - names
    assert not missing, f"Missing {missing} from {rel}"
