"""Shared I/O for the second release-granularity revision."""
import hashlib
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def freeze(path,obj):
    if path.exists():assert json.loads(path.read_text())['design']==obj,(path,'frozen design changed')
    else:save(path,dict(design=obj,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256={p.name:sha(p) for p in (ROOT/'src').glob('*.py')}))


def key(job):
    identity={k:job[k] for k in ['dataset','n','epsilon','seed','method','k','decoder','estimator','proxy_r2','proxy_r3']}
    return '_'.join([job['dataset'],str(job['n']),str(job['epsilon']),str(job['seed']),job['method'],
        str(job['k']),job['decoder'],job['estimator'],str(job['proxy_r2']),str(job['proxy_r3'])])
