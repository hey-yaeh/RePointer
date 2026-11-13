"""Executable synthetic training/inference wiring of the final method stages.

This exercises architecture and contracts. It is not a paper reproduction runner:
the body model and motion-only detectors here are explicitly synthetic doubles.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from .candidates import construct_route, query_features, realize
from .config import PipelineConfig
from .contracts import Candidate, Donor, Query
from .evaluation import evaluate_hold
from .fixtures import load_fixture
from .kinematics import SyntheticKinematics
from .ranking import (RidgeModel, build_hypotheses, build_ranker_feature, build_router_features,
                      cross_fit_ridge, phase_route, select_with_threshold)
from .retrieval import retrieve
from .selector import select_candidate
from .timing import (TimingInput, apex_from_distribution, fit_synthetic_prior, fit_timing,
                     load_checkpoint, load_synthetic_prior, ratio_to_frame, save_checkpoint)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False,sort_keys=True)
        stream.write('\n')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_bindings(root: Path) -> dict[str,str]:
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    paths = ['manifest.json']+[item['motion'] for item in manifest['donors']]
    bindings = {}
    for name in paths:
        source = (root/name).resolve()
        if not source.is_relative_to(root):
            raise ValueError("fixture motion path escapes fixture root")
        bindings[str(source.relative_to(root))] = digest(source)
    return bindings


def implementation_bindings() -> dict[str,str]:
    """Bind executed package source without recording any workstation paths."""
    return {path.name:digest(path) for path in sorted(Path(__file__).parent.glob('*.py'))}


def timing_input(query: Query, fallback_ratio: float = .5) -> TimingInput:
    words = [w for w in query.words if w.end>=query.clip_start-.05 and w.start<=query.clip_start+query.frames/30+.05]
    fixed = ratio_to_frame(fallback_ratio,query.frames)
    if words:
        fixed = int(np.clip(round((max(w.end for w in words)-query.clip_start)*30)-11,2,query.frames-3))
    return TimingInput(query.words,query.frames,query.hand,query.domain,query.target,fixed,query.clip_start)


def fit_clock(records: list[dict], settings: PipelineConfig):
    from .timing import domain_hand_medians
    base_inputs = [timing_input(r['query']) for r in records]
    labels = [r['apex_frame'] for r in records]
    medians = domain_hand_medians(base_inputs,labels)
    inputs = [timing_input(r['query'],medians[f"{r['query'].domain}_hand{r['query'].hand}"]) for r in records]
    if settings.timing_backend == 'wordphase':
        return fit_timing(inputs,labels,epochs=settings.epochs,seed=settings.seed)
    return fit_synthetic_prior(inputs,labels)


def route_options(query, donors, model, settings):
    engine = SyntheticKinematics()
    fallback_ratio = model.medians[f'{query.domain}_hand{query.hand}']
    prob = model.predict([timing_input(query,fallback_ratio)])[0]
    median = ratio_to_frame(fallback_ratio,query.frames)
    expected = apex_from_distribution(prob,query.frames)
    options = {}
    # Reuse identical donor/apex realizations across the two timing routes.
    for name,frame in (('median',median),('wordphase',expected)):
        if name == 'wordphase' and expected == median:
            previous,metrics = options['median']
            options[name] = (replace(previous,timing_source=name),metrics)
        else:
            options[name] = construct_route(query,donors,frame,name,engine,top_k=settings.top_k,
                steps=settings.ik_steps,plateau=settings.hold_plateau,taper=settings.hold_taper)
    row = {**query_features(query),'median_peak':median,'wordphase_peak':expected}
    for name,(_,metrics) in options.items():
        row.update({f'{name}_{key}':value for key,value in metrics.items()})
    features,_,_ = build_router_features([row])
    return options,features[0],prob,median


def candidate_pool(query, donors, a_pair, prob, median, settings):
    engine = SyntheticKinematics()
    hypotheses = build_hypotheses(prob,query.frames,median,count=settings.hypotheses)
    a,metrics = a_pair
    fallback_timing = min(hypotheses,key=lambda h:(abs(h['planned_frame']-a.planned_frame),h['hypothesis_rank']))
    fallback_features = {**fallback_timing,'hypothesis':'wp_expected' if a.timing_source=='wordphase' else 'median'}
    pairs, timings = [a_pair],[fallback_features]
    budget = settings.hypotheses*settings.donors_per_hypothesis
    for hypothesis in hypotheses:
        planned = hypothesis['planned_frame']
        # Reserve the fallback slot in A's timing hypothesis, excluding A before
        # truncating donor order. Other hypotheses keep their complete budget.
        retrieved = retrieve(query,donors,planned,len(donors))
        count = settings.donors_per_hypothesis
        if hypothesis is fallback_timing:
            retrieved = [(d,dist) for d,dist in retrieved if d.context.sample_id!=a.donor_id]
            count -= 1
        retrieved = retrieved[:count]
        if len(retrieved) != count:
            raise ValueError("not enough eligible donors for the configured Candidate B budget")
        for rank,(donor,distance) in enumerate(retrieved,1):
            pair = realize(query,donor,distance,planned,hypothesis['hypothesis'],engine,
                steps=settings.ik_steps,plateau=settings.hold_plateau,taper=settings.hold_taper)
            pair = pair[0],{**pair[1],'donor_rank':rank,'donor_distance':distance}
            pairs.append(pair); timings.append(hypothesis)
    if len(pairs) != budget:
        raise ValueError("candidate search did not fill the configured budget")
    features = np.stack([build_ranker_feature(query_features(query),h,m) for (_,m),h in zip(pairs,timings)])
    return pairs,features


def _training_utility(record: dict, metrics: dict) -> float:
    """Training-only supervision. Neither hold endpoint is an inference input."""
    return metrics['self_score1b_proxy20']*evaluate_hold(metrics['self_frame'],record['hold_start'],record['hold_end'])


def train_fixture(fixture_root: Path, checkpoint_root: Path, settings: PipelineConfig) -> dict:
    donors,records,queries = load_fixture(fixture_root)
    if len(records)<3 or len({r['query'].fold for r in records})<3:
        raise ValueError("fixture training requires three group-disjoint folds")
    for r in records:
        q = r['query']
        if not (type(r['apex_frame']) is int and 0<=r['apex_frame']<q.frames
                and 0<=r['hold_start']<=r['hold_end']<q.frames):
            raise ValueError("invalid fixture training labels")
    train_context = {(r['query'].sample_id,r['query'].group,r['query'].recording) for r in records}
    for q in queries:
        if any(q.sample_id==sid or q.group==group or q.recording==recording for sid,group,recording in train_context):
            raise ValueError("training/inference sample, group or recording overlap")
    groups = [r['query'].group for r in records]
    folds = np.array([r['query'].fold for r in records])
    # Validate split boundaries before fitting timing networks or touching output.
    for attr in ('group','recording','sample_id'):
        for value in {getattr(r['query'],attr) for r in records}:
            if len({r['query'].fold for r in records if getattr(r['query'],attr)==value})!=1:
                raise ValueError("training identity spans folds")
    checkpoint_root = Path(checkpoint_root)
    checkpoint_root.mkdir(parents=True,exist_ok=False)
    clocks = {int(fold):fit_clock([r for r in records if r['query'].fold!=fold],settings) for fold in np.unique(folds)}
    route_data = [route_options(r['query'],donors,clocks[r['query'].fold],settings) for r in records]
    X = np.stack([d[1] for d in route_data])
    y = np.array([_training_utility(r,d[0]['wordphase'][1])-_training_utility(r,d[0]['median'][1]) for r,d in zip(records,route_data)])
    router_oof = cross_fit_ridge(X,y,groups,folds)
    router = RidgeModel.fit(X,y,groups)
    pool_features,pool_targets,pool_groups,pool_folds,pool_queries = [],[],[],[],[]
    for i,(record,data) in enumerate(zip(records,route_data)):
        options,_,prob,median = data
        choice = phase_route(float(router_oof.predictions[i]))
        pairs,features = candidate_pool(record['query'],donors,options[choice],prob,median,settings)
        pool_features.extend(features)
        pool_targets.extend(_training_utility(record,m) for _,m in pairs)
        pool_groups.extend([record['query'].group]*len(pairs))
        pool_folds.extend([record['query'].fold]*len(pairs))
        pool_queries.extend([record['query'].sample_id]*len(pairs))
    rank_X,rank_y = np.array(pool_features),np.array(pool_targets)
    rank_oof = cross_fit_ridge(rank_X,rank_y,pool_groups,pool_folds,pool_queries)
    ranker = RidgeModel.fit(rank_X,rank_y,pool_groups,query_ids=pool_queries)
    clock = fit_clock(records,settings)
    clock_name = 'timing.pt' if settings.timing_backend=='wordphase' else 'timing.json'
    if settings.timing_backend=='wordphase': save_checkpoint(clock,checkpoint_root/clock_name)
    else: clock.save(checkpoint_root/clock_name)
    router.save(checkpoint_root/'router.npz'); ranker.save(checkpoint_root/'ranker.npz')
    report = dict(schema='repointer.synthetic-training.v1',synthetic=True,paper_parity=False,
        kinematics='synthetic-skeleton',detectors='synthetic-motion-heuristics',pipeline=asdict(settings),
        timing_file=clock_name,training_queries=len(records),folds=len(clocks),ranker_rows=len(rank_X),
        router_oof_mse=float(np.mean((router_oof.predictions-y)**2)),
        ranker_oof_mse=float(np.mean((rank_oof.predictions-rank_y)**2)),
        ranker_replacement_threshold=.01,
        fixture_manifest_sha256=digest(Path(fixture_root)/'manifest.json'),
        fixture_files_sha256=fixture_bindings(fixture_root),
        implementation_sha256=implementation_bindings(),
        environment=dict(python=platform.python_version(),numpy=np.__version__,
            **({'torch':__import__('torch').__version__} if settings.timing_backend=='wordphase' else {})),
        files={name:digest(checkpoint_root/name) for name in (clock_name,'router.npz','ranker.npz')},
        limitations=['Synthetic fixture experiment only; not an official benchmark result.',
                     'Fixed ridge lambda and replacement threshold; historical calibration is not restored.',
                     'Staged OOF diagnostics are not a fully nested end-to-end generalization estimate.'])
    write_json(checkpoint_root/'training.json',report)
    return report


def infer_fixture(fixture_root: Path, checkpoint_root: Path, output_root: Path, settings: PipelineConfig) -> dict:
    from .challenge import materialize_candidate
    donors,_,queries = load_fixture(fixture_root)
    checkpoint_root,output_root = Path(checkpoint_root),Path(output_root)
    report = json.loads((checkpoint_root/'training.json').read_text(encoding='utf-8'))
    if report.get('schema')!='repointer.synthetic-training.v1' or report.get('synthetic') is not True:
        raise ValueError("unsupported training artifact schema")
    if report['pipeline'] != asdict(settings):
        raise ValueError("inference configuration differs from the trained artifact configuration")
    if report['implementation_sha256'] != implementation_bindings():
        raise ValueError("implementation changed since training; create a new documented run")
    if report['fixture_manifest_sha256'] != digest(Path(fixture_root)/'manifest.json'):
        raise ValueError("fixture manifest changed since training")
    # Bind the actual input arrays, not just their filenames in the manifest.
    if fixture_bindings(fixture_root) != report['fixture_files_sha256']:
        raise ValueError("donor motion hash mismatch")
    for name,expected in report['files'].items():
        if Path(name).name!=name or digest(checkpoint_root/name)!=expected:
            raise ValueError("training artifact hash mismatch")
    clock = load_checkpoint(checkpoint_root/'timing.pt') if settings.timing_backend=='wordphase' else load_synthetic_prior(checkpoint_root/'timing.json')
    router,ranker = RidgeModel.load(checkpoint_root/'router.npz'),RidgeModel.load(checkpoint_root/'ranker.npz')
    output_root.mkdir(parents=True,exist_ok=False)
    decisions = []
    for query in queries:
        options,X,prob,median = route_options(query,donors,clock,settings)
        route = phase_route(float(router.predict(X[None])[0]))
        a_pair = options[route]
        pairs,features = candidate_pool(query,donors,a_pair,prob,median,settings)
        scores = ranker.predict(features)
        index = select_with_threshold([m for _,m in pairs],scores,0,report['ranker_replacement_threshold'])
        b_pair = pairs[index]
        selected,decision = select_candidate(a_pair[0],b_pair[0],a_pair[1],b_pair[1],settings.selector_threshold)
        a_path,b_path = output_root/'candidate_a'/f'{query.sample_id}.npz',output_root/'candidate_b'/f'{query.sample_id}.npz'
        a_pair[0].motion.save(a_path); b_pair[0].motion.save(b_path)
        materialized = materialize_candidate(a_path,b_path,decision['choice'],output_root/'selected'/f'{query.sample_id}.npz',query.frames)
        decisions.append(dict(sample_id=query.sample_id,frames=query.frames,route=route,
            candidate_b_index=index,candidate_budget=len(pairs),selected_donor=selected.donor_id,
            planned_frame=selected.planned_frame,**decision,sha256=materialized['sha256']))
    result = dict(schema='repointer.synthetic-inference.v1',synthetic=True,paper_parity=False,
        kinematics='synthetic-skeleton',detectors='synthetic-motion-heuristics',
        timing_backend=settings.timing_backend,decisions=decisions,
        training_sha256=digest(checkpoint_root/'training.json'))
    write_json(output_root/'inference.json',result)
    return result
