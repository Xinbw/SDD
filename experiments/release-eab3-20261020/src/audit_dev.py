"""Audit circularity with the audit sample drawn from DEVELOPMENT, never from test.

Offline research diagnostic; nothing here is released or routes a decision.
A labelled public audit sample of size a, drawn from the public development split, estimates the
one-to-one ceiling by the best matching on its own confusion matrix (its in-sample accuracy). The
same matching is scored once on the test split, and the true one-to-one ceiling is the best matching
on the full test confusion matrix. Test labels therefore only score; they never form an audit.
Supersedes the test-split audit of release-eab2-20261020 in the manuscript; that record is kept.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;FRONTIER=EXP/'release-frontier-20261019'
sys.path.insert(0,str(FRONTIER/'src'))
import torch
import engine_base as base
from common import CLASSES
AUDIT_MULTIPLES=[1,2,5,10,20,50];AUDIT_REPS=200;SEED=20261021

def assignment(task,k,recorded):
    with np.load(FRONTIER/'inputs'/f'{task}.npz') as b:data={x:b[x] for x in ['x_proxy','x_development','y_development','x_test','y_test']}
    _,_,centres=base.partition(data,k)
    points=torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda')
    a=torch.cdist(points,torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy();dev,test=a[:len(data['x_development'])],a[len(data['x_development']):]
    assert hashlib.sha256(test.tobytes()).hexdigest()==recorded[f'{task}/K{k}'],(task,k)
    return dev,data['y_development'],test,data['y_test']

def weights(task,y,C):
    return (1/np.maximum(np.bincount(y,minlength=C),1)[y])/C if task=='web' else np.full(len(y),1/len(y))

def confusion(task,a,y,C):
    W=np.zeros((C,C));np.add.at(W,(a,y),weights(task,y,C));return W

def matching(conf):
    r,c=linear_sum_assignment(-conf);m=np.full(conf.shape[0],-1);m[r]=c;return m

def main():
    torch.backends.cuda.matmul.allow_tf32=False
    recorded=json.loads((FRONTIER/'analysis/canonical-regional-readouts.json').read_text())['geometries']
    rng=np.random.default_rng(SEED);audit=[]
    for task in ['cifar10','mnist','web','femnist62']:
        C=CLASSES[task];da,dy,ta,ty=assignment(task,C,recorded)
        W=confusion(task,ta,ty,C);true=float(W[np.arange(C),matching(W)].sum())
        for mult in AUDIT_MULTIPLES:
            size=mult*C
            if size>len(dy)//2:continue
            est=[];held=[];cover=[]
            for _ in range(AUDIT_REPS):
                idx=rng.choice(len(dy),size,replace=False);Wa=confusion(task,da[idx],dy[idx],C);m=matching(Wa)
                est.append(float(Wa[np.arange(C),m].sum()));held.append(float(np.sum(weights(task,ty,C)*(m[ta]==ty))));cover.append(len(np.unique(da[idx]))==C)
            audit.append(dict(dataset=task,audit_split='development',scored_split='test',audit_labels=size,labels_per_class=mult,true_one_to_one_ceiling=true,
                in_sample_estimate=float(np.mean(est)),estimate_abs_error_p90=float(np.quantile(np.abs(np.array(est)-true),.9)),
                heldout_naming=float(np.mean(held)),heldout_naming_p10=float(np.quantile(held,.1)),all_regions_covered=float(np.mean(cover)),repetitions=AUDIT_REPS))
            x=audit[-1];print(task,size,round(100*x['in_sample_estimate'],1),round(100*x['heldout_naming'],1),round(100*true,1),flush=True)
    (ROOT/'analysis/audit-circularity-dev.json').write_text(json.dumps(dict(records=audit,seed=SEED,
        scope='random labelled audit drawn from the public development split; its induced matching scored on test; offline diagnostic'),indent=2)+'\n')

if __name__=='__main__':main()
