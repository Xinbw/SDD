"""Exact loss decomposition of regional naming and the audit-sample circularity check.

Offline research diagnostics; nothing here is released or routes a decision.
(1) For every K=C regional release behind tab-ceiling, decode the stored CLEAN vote counts
    (counts/n, never published) with the same centered decoder as the protected release. Then
    ceiling - clean = teacher-content loss, clean - released = release-noise loss.
(2) Audit circularity: a labelled public audit sample of size a estimates the one-to-one ceiling
    by the best matching on its own confusion matrix. That estimate is exactly the in-sample
    accuracy of the matching it induces; we report it next to the held-out accuracy of the same
    matching (test complement), i.e. the public naming the audit already completes.
Assignments use the shared public map and must match the recorded geometry hashes.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;FRONTIER=EXP/'release-frontier-20261019'
sys.path.insert(0,str(FRONTIER/'src'))
import torch
import engine_base as base
from common import CLASSES,SETTINGS,SEEDS,release_key
from run_private import decode,metrics
CONFIRM=[('cifar10',1000,.1),('cifar10',10000,.1),('mnist',10000,.1)];CONFIRM_SEEDS=list(range(8801,8811))
AUDIT_MULTIPLES=[1,2,5,10,20,50];AUDIT_REPS=200

def assignment(task,k,recorded):
    with np.load(FRONTIER/'inputs'/f'{task}.npz') as b:data={x:b[x] for x in ['x_proxy','x_development','x_test','y_test']}
    _,_,centres=base.partition(data,k)
    points=torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda')
    a=torch.cdist(points,torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()[len(data['x_development']):]
    assert hashlib.sha256(a.tobytes()).hexdigest()==recorded[f'{task}/K{k}'],(task,k)
    return a,data['y_test']

def weights(task,y,C):
    return (1/np.maximum(np.bincount(y,minlength=C),1)[y])/C if task=='web' else np.full(len(y),1/len(y))

def matching(conf):
    r,c=linear_sum_assignment(-conf);m=np.full(conf.shape[0],-1);m[r]=c;return m

def main():
    torch.backends.cuda.matmul.allow_tf32=False
    recorded=json.loads((FRONTIER/'analysis/canonical-regional-readouts.json').read_text())['geometries']
    canonical=json.loads((FRONTIER/'analysis/canonical-regional-readouts.json').read_text())['records']
    rows=[];audit=[];cache={}
    for d,n,e in SETTINGS:
        C=CLASSES[d];metric='balanced_accuracy' if d=='web' else 'accuracy'
        if d not in cache:cache[d]=assignment(d,C,recorded)
        a,y=cache[d];conf=(d,n,e) in CONFIRM;stage=FRONTIER/'confirmation' if conf else FRONTIER
        per=[]
        for seed in (CONFIRM_SEEDS if conf else SEEDS):
            key=release_key(d,n,e,seed,180,'regional',C,'M');path=stage/'results'/(key+'.json');r=json.loads(path.read_text())
            z=np.load(stage/r['diagnostic_file']);w=np.bincount(z['ids'],minlength=C)
            noisy=decode(z['scores'],w,'center');assert (noisy==np.array(r['endpoints']['center']['labels'])).all(),key
            clean=decode(z['counts']/n,w,'center')
            released=metrics(noisy[a],y,C)[metric];ref=canonical[str(path.relative_to(EXP.parent))]['releases']['1']['center'][metric]
            assert abs(released-ref)<1e-12,key
            per.append(dict(seed=seed,released=released,clean=metrics(clean[a],y,C)[metric],flipped_regions=int((clean!=noisy).sum())))
        rows.append(dict(dataset=d,n=n,epsilon=e,cohorts='confirmation' if conf else 'development',metric=metric,
            clean_named=float(np.mean([p['clean'] for p in per])),released_named=float(np.mean([p['released'] for p in per])),
            mean_flipped_regions=float(np.mean([p['flipped_regions'] for p in per])),per_seed=per))
        print(d,n,e,round(100*rows[-1]['clean_named'],1),round(100*rows[-1]['released_named'],1),flush=True)
    rng=np.random.default_rng(20261020)
    for d,(a,y) in cache.items():
        C=CLASSES[d];w=weights(d,y,C);W=np.zeros((C,C));np.add.at(W,(a,y),w);true=float(W[np.arange(C),matching(W)].sum())
        for mult in AUDIT_MULTIPLES:
            size=mult*C
            if size>=len(y)//2:continue
            est=[];held=[];cover=[]
            for _ in range(AUDIT_REPS):
                idx=rng.choice(len(y),size,replace=False);mask=np.ones(len(y),bool);mask[idx]=False
                wa=weights(d,y[idx],C);Wa=np.zeros((C,C));np.add.at(Wa,(a[idx],y[idx]),wa)
                m=matching(Wa);est.append(float(Wa[np.arange(C),m].sum()))
                pred=m[a[mask]];held.append(float(np.sum(weights(d,y[mask],C)*(pred==y[mask]))));cover.append(len(np.unique(a[idx]))==C)
            audit.append(dict(dataset=d,audit_labels=size,labels_per_class=mult,true_one_to_one_ceiling=true,
                in_sample_estimate=float(np.mean(est)),estimate_abs_error_p90=float(np.quantile(np.abs(np.array(est)-true),.9)),
                heldout_naming=float(np.mean(held)),heldout_naming_p10=float(np.quantile(held,.1)),all_regions_covered=float(np.mean(cover)),repetitions=AUDIT_REPS))
            x=audit[-1];print(d,size,round(100*x['in_sample_estimate'],1),round(100*x['heldout_naming'],1),round(100*true,1),flush=True)
    (ROOT/'analysis/naming-decomposition.json').write_text(json.dumps(dict(records=rows,
        scope='offline diagnostic: clean counts are never published; test labels never route decisions'),indent=2)+'\n')
    (ROOT/'analysis/audit-circularity.json').write_text(json.dumps(dict(records=audit,
        scope='random labelled audit drawn from the test split; held-out accuracy on its complement; offline diagnostic'),indent=2)+'\n')

if __name__=='__main__':main()
