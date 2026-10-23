"""Predeclared feature-selected annotation, separate from passive label access."""
import json
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
import engine_base as base
import public_frontier as frontier
from common import ROOT,CLASSES,SETTINGS,SEEDS,freeze,save,sha,release_key
from run_private import metrics

def main(only=None):
    freeze('GEOMETRY-LABEL-DESIGN.json',dict(tasks=list(CLASSES),multipliers=[1,3,5,10,20,50],
        policy='For each public K=C region, request q nearest development features to its centroid; if short, fill with remaining globally closest-to-assigned-centroid points.',
        uses_unqueried_labels=False,distinct_sample_labels_counted=True,random_draws=1,
        acquisition_right='feature-selected annotation queries; separate from existing passive labels',
        reason='Avoid attributing private value to passive class-coverage gaps when public representative annotation is available.'))
    for task,c in CLASSES.items():
        if only and task!=only:continue
        data=frontier.public_features(task);data['dataset']=task
        proxy_ids,reps,centres=base.partition(data,c);weights=np.bincount(proxy_ids,minlength=c)
        allx=np.concatenate([data['x_development'],data['x_test']])
        assignment=torch.cdist(torch.as_tensor(allx,device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
        data['proxy_assignment']=proxy_ids;xd=data['x_development'];dev_ids=assignment[:len(xd)]
        distance=((xd-centres[dev_ids])**2).sum(1);pools=[]
        for r in range(c):
            idx=np.where(dev_ids==r)[0];pools.append(idx[np.argsort(distance[idx],kind='stable')])
        graphs={a:frontier.graph_operator(task,data,a) for a in [.8,.95]}
        cases=frontier.private_cases(task)
        for q in [1,3,5,10,20,50]:
            output=ROOT/'public'/f'{task}_geometry_0_m{q*c}.json'
            if output.exists():continue
            idx=np.concatenate([p[:q] for p in pools]);chosen=set(idx.tolist())
            if len(idx)<q*c:
                other=[int(i) for i in np.argsort(distance,kind='stable') if int(i) not in chosen]
                idx=np.concatenate([idx,np.array(other[:q*c-len(idx)])])
            assert len(idx)==q*c and len(set(idx.tolist()))==q*c
            # Only now acquire exactly the selected labels.
            with np.load(ROOT/'inputs'/f'{task}.npz') as bundle:y=bundle['y_development'][idx]
            sample=dict(scheme='geometry',draw=0,q=q,m=len(idx),indices=idx.tolist(),labels=y.tolist(),
                missing_classes=int((np.bincount(y,minlength=c)==0).sum()),
                sample_sha256=hashlib.sha256(idx.tobytes()+y.tobytes()).hexdigest())
            candidates,choice,extra=frontier.public_candidates(data,sample,graphs,assignment,centres,weights);public=candidates[choice]
            out=dict(dataset=task,**sample,selected_public=public['name'],label_budget=len(idx),external_validation_labels=0,
                baselines={},hybrid=[],label_acquisition='feature-selected development annotations')
            with np.load(ROOT/'inputs'/f'{task}.npz') as b:truth=b['y_test']
            for a in candidates:out['baselines'][a['name']]=dict(**metrics(a['test'].argmax(1),truth,c),loo_score=a['loo_score'])
            out['public_portfolio']=out['baselines'][public['name']]
            out['hybrid']=frontier.evaluate_cases(cases,public,sample,data,assignment,centres,weights,extra,truth,out['public_portfolio'])
            save(output,out);print(task,'geometry labels',len(idx),'public',round(out['public_portfolio']['accuracy']*100,3),public['name'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CLASSES));main(p.parse_args().dataset)
