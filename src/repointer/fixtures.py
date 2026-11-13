"""Invented transcripts and motions only; nothing is derived from HSI."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .contracts import Donor, Motion, Query
from .timing import TimedWord


def query_to_dict(query: Query) -> dict:
    record = asdict(query)
    for key in ('target','betas','target_world','bbox_size','bbox_center'):
        if record[key] is not None: record[key] = record[key].tolist()
    return record


def query_from_dict(record: dict) -> Query:
    # Dataclass construction rejects unknown fields, including evaluation labels.
    record = dict(record)
    record['words'] = tuple(TimedWord(**word) for word in record.get('words',()))
    return Query(**record)


def write_fixture(root: Path, *, seed: int = 0) -> Path:
    root = Path(root)
    root.mkdir(parents=True,exist_ok=False)
    rng = np.random.default_rng(seed)
    donors = []
    for hand in (0,1):
        for i in range(4):
            frames,peak = 32+i*2,14+i
            poses = np.zeros((frames,55,3),dtype=np.float32)
            weight = np.exp(-.5*((np.arange(frames)-peak)/7.)**2)
            sign = -1. if hand == 0 else 1.
            poses[:,16+hand,1] = -sign*(.5+.2*i)*weight
            poses[:,16+hand,2] = sign*(.1+.04*i)*weight
            poses[:,18+hand,1] = -sign*(.15+.02*i)*weight
            poses[:,0,1] = .03*np.sin(np.linspace(0,2*np.pi,frames))
            trans = np.zeros((frames,3),dtype=np.float32)
            trans[:,0] = .01*np.sin(np.linspace(0,np.pi,frames))
            sid = f'donor_h{hand}_{i}'
            query = Query(sid,frames,hand,np.array([sign*(1+.2*i),.3,2.]),
                group=sid,recording=sid,fold=-1)
            relative = f'donors/{sid}.npz'
            Motion(poses,trans).save(root/relative)
            donors.append(dict(query=query_to_dict(query),motion=relative,apex=peak))
    train = []
    for i in range(8):
        hand = i%2
        frames = 32+2*(i%3)
        apex = int(round((.43+.1*hand+.015*(i//4))*(frames-1)))
        sid = f'train_{i}'
        words = (TimedWord(.05,.3,'point'),TimedWord(.35,apex/30+.25,'there'))
        query = Query(sid,frames,hand,np.array([(-1 if hand==0 else 1)*(.8+.1*i),.2,2.]),
            betas=rng.normal(0,.1,10),domain='natural' if i%4<2 else 'synthetic',
            group=sid,recording=sid,fold=i%3,words=words)
        train.append(dict(query=query_to_dict(query),apex_frame=apex,hold_start=apex-2,hold_end=apex+2))
    queries = [query_to_dict(Query(f'query_h{hand}',34,hand,np.array([(-1 if hand==0 else 1)*1.1,.25,2.]),
        group=f'test_{hand}',recording=f'test_{hand}',words=(TimedWord(.05,.3,'point'),TimedWord(.4,.9,'there')))) for hand in (0,1)]
    manifest = dict(schema='repointer.synthetic-fixture.v1',synthetic=True,seed=seed,donors=donors,train=train,queries=queries)
    path = root/'manifest.json'
    path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path


def load_fixture(root: Path) -> tuple[list[Donor],list[dict],list[Query]]:
    root = Path(root).resolve()
    data = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if data.get('schema') != 'repointer.synthetic-fixture.v1' or data.get('synthetic') is not True:
        raise ValueError("this CLI currently accepts only explicitly synthetic fixtures")
    donors = []
    for item in data['donors']:
        motion_path = (root/item['motion']).resolve()
        if not motion_path.is_relative_to(root):
            raise ValueError("fixture motion path escapes fixture root")
        donors.append(Donor(query_from_dict(item['query']),Motion.load(motion_path),item['apex']))
    train = [{**item,'query':query_from_dict(item['query'])} for item in data['train']]
    queries = [query_from_dict(item) for item in data['queries']]
    for rows in (donors,train,queries):
        ids = [r.context.sample_id if isinstance(r,Donor) else (r['query'].sample_id if isinstance(r,dict) else r.sample_id) for r in rows]
        if len(ids) != len(set(ids)): raise ValueError("duplicate sample IDs in fixture split")
    return donors,train,queries
