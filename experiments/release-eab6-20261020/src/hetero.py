"""Class heterogeneity of the source-disjoint partition versus its unconstrained target (no query).

For all ten disjoint cohorts (frozen cohort construction of release-disjoint-20261020): mean
top-class share per client and median number of classes per client, overall and for the last 100
clients (drawn when per-class pools are partly exhausted). References: the reused-source CIFAR-1k
verification cohorts (release-value-20261019, 600 records/client) and unconstrained Dirichlet(.1)
labels with 40 records per client (seed 20261027).
"""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;DJ=EXP/'release-disjoint-20261020';VALUE=EXP/'release-value-20261019'
sys.path.insert(0,str(EXP/'release-verify-teacher-20261020/src'));sys.path.insert(0,str(DJ/'src'));sys.path.insert(0,str(VALUE/'src'))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
C=10
def stats(lab):
    p=np.stack([(lab==c).mean(1) for c in range(C)],1);return dict(top_class_share=float(p.max(1).mean()),median_classes=float(np.median((p>0).sum(1))))
def main():
    dj=load('djrun',DJ/'src/run.py');pv=load('pv',VALUE/'src/private_validation.py')
    y=np.load(EXP/'release-frontier-20261019/inputs/cifar10.npz')['y_private'];res=[];last=[]
    for s in json.loads((DJ/'DISJOINT-DESIGN.json').read_text())['design']['cohort_seeds']:
        tr,va,_=dj.cohort(y,s);m=np.concatenate([tr,va],1);res.append(stats(y[m]));last.append(stats(y[m[-100:]]))
    reused=[]
    for s in pv.SEEDS:
        tr,va,_=pv.cohort(y,s);reused.append(stats(y[np.concatenate([tr,va],1)]))
    rng=np.random.default_rng(20261027);P=rng.dirichlet(np.full(C,.1),size=1000);free=stats(np.array([rng.choice(C,40,p=q) for q in P]))
    mean=lambda L,k:float(np.mean([x[k] for x in L]))
    out=dict(disjoint=dict(top_class_share=mean(res,'top_class_share'),median_classes=mean(res,'median_classes')),
        disjoint_last100=dict(top_class_share=mean(last,'top_class_share')),reused_source_600=dict(top_class_share=mean(reused,'top_class_share'),median_classes=mean(reused,'median_classes')),
        unconstrained_dirichlet_40=free,cohorts=len(res))
    (ROOT/'analysis/disjoint-heterogeneity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=1))
if __name__=='__main__':main()
