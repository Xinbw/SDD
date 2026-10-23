"""Numerically audit the preceding conditional lemma in its stated space."""
import json
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import ROOT,PARENT,save,sha
from run_private import project

def assignment_gap(a,weights):
    utility=a*(weights/weights.sum())[:,None]
    r,c=linear_sum_assignment(-utility);best=float(utility[r,c].sum());second=-math.inf
    for row,col in zip(r,c):
        altered=utility.copy();altered[row,col]=-1e100
        rr,cc=linear_sum_assignment(-altered);second=max(second,float(utility[rr,cc].sum()))
    return best-second

def main():
    plan=json.loads((PARENT/'STUDY-SCHEDULE.json').read_text())['design'];out=[]
    for role in plan['roles']:
        if role['role']!='public-selected':continue
        path=PARENT/'results'/(role['job']+'.json');row=json.loads(path.read_text())
        profile=np.load(PARENT/'public-selection'/f"{row['dataset']}_K{row['k']}_{row['estimator']}.npz")
        diagnostic=np.load(PARENT/row['diagnostic_file']);q=diagnostic['r1_counts']/row['n'];qp=profile['q']
        delta=float(np.max(np.abs(q-qp)));v=row['trace'][0]['privacy']['residual_std']
        weights=profile['proxy_region_weights'];a=project(qp);k,c=q.shape
        raw_gap=np.sort(q,axis=1)[:,-1,None]-q
        best=q.argmax(1);terms=np.exp(-raw_gap**2/(4*v*v));terms[np.arange(k),best]=0
        per_region=np.minimum(1,terms.sum(1));raw=float(np.average(per_region,weights=weights))
        item=dict(dataset=row['dataset'],n=row['n'],epsilon=row['epsilon'],seed=row['seed'],
            decoder=row['decoder'],k=k,estimator=row['estimator'],sup_shift_unprojected=delta,
            noise_std=v,raw_private_winner_weighted_flip_bound=raw,source_sha256=sha(path),
            clean_private_counts_are_offline_only=True)
        if row['decoder']=='center':
            score=a-(a*weights[:,None]).sum(0)/weights.sum();top=np.sort(score,axis=1)[:,-2:]
            gap=float(np.min(top[:,1]-top[:,0]));radius=gap/8-delta
            bound=1. if radius<=0 else min(1.,2*k*c*math.exp(-radius*radius/(2*v*v)))
            item.update(gap=gap,radius=radius,coupled_bound=bound,criterion='centered g/8 - delta')
        elif row['decoder']=='matching':
            gap=assignment_gap(a,weights);radius=gap/4-delta
            bound=1. if radius<=0 else min(1.,2*k*c*math.exp(-radius*radius/(2*v*v)))
            item.update(gap=gap,radius=radius,coupled_bound=bound,criterion='normalized assignment gap/4 - delta')
        else:item.update(coupled_bound=None,criterion='raw: coupled lemma not applicable; separate vote-gap bound')
        out.append(item)
    save(ROOT/'analysis/bound-diagnostics.json',out)
    print('Actual coupled bounds:',sorted({r['coupled_bound'] for r in out if r['coupled_bound'] is not None}))

if __name__=='__main__':main()
