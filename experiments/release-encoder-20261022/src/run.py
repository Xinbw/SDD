"""Weak-encoder ablation runner (ENCODER-DESIGN.json, frozen before this file).

Reuses the unchanged release-frontier modules with their roots pointed at this
stage: confirm_joint.private (cohorts, 180 steps, K=C release and fixed head),
geometry-label selection, anchored_fusion.main(confirm=True) (fourteen public
learners, joint naming, exact LOO).
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;FRONTIER=EXP/'release-frontier-20261019'
sys.path.insert(0,str(FRONTIER/'src'));sys.path.insert(0,str(ROOT/'src'))
import design;design.main()
import engine_base as base,public_frontier as public,run_private,confirm_joint,anchored_fusion
for m in [base,public,run_private,confirm_joint,anchored_fusion]:m.ROOT=ROOT
confirm_joint.CONFIRM=ROOT/'confirmation';confirm_joint.CONFIRM_SETTINGS=[('cifar10',1000,.1)]
from common import save

def link_random_samples():
    folder=ROOT/'confirmation/samples';folder.mkdir(parents=True,exist_ok=True)
    for p in sorted((FRONTIER/'confirmation/samples').glob('cifar10_random_*_m*.json')):
        t=folder/p.name
        if not t.exists():t.symlink_to(os.path.relpath(p,folder))

def geometry_samples():
    """geometry_labels.py policy in the ResNet-18 geometry; labels read only for the selected indices."""
    task,c='cifar10',10;data=public.public_features(task);data['dataset']=task
    proxy_ids,reps,centres=base.partition(data,c)
    allx=np.concatenate([data['x_development'],data['x_test']])
    assignment=torch.cdist(torch.as_tensor(allx,device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
    xd=data['x_development'];dev=assignment[:len(xd)];distance=((xd-centres[dev])**2).sum(1)
    pools=[np.where(dev==r)[0][np.argsort(distance[dev==r],kind='stable')] for r in range(c)]
    for q in [1,3,5]:
        out=ROOT/'confirmation/samples'/f'{task}_geometry_0_m{q*c}.json'
        if out.exists():continue
        idx=np.concatenate([p[:q] for p in pools]);chosen=set(idx.tolist())
        if len(idx)<q*c:
            other=[int(i) for i in np.argsort(distance,kind='stable') if int(i) not in chosen];idx=np.concatenate([idx,np.array(other[:q*c-len(idx)])])
        assert len(idx)==q*c and len(set(idx.tolist()))==q*c
        with np.load(ROOT/'inputs'/f'{task}.npz') as b:y=b['y_development'][idx]
        save(out,dict(dataset=task,scheme='geometry',draw=0,q=q,m=len(idx),indices=idx.tolist(),labels=y.tolist(),
            missing_classes=int((np.bincount(y,minlength=c)==0).sum()),sample_sha256=hashlib.sha256(idx.tobytes()+y.tobytes()).hexdigest()))

def main(step):
    for d in ['checkpoints','public','analysis','confirmation/results','confirmation/diagnostics']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    if step in ['private','all']:confirm_joint.private()
    if step in ['samples','all']:link_random_samples();geometry_samples()
    if step in ['joint','all']:anchored_fusion.main('cifar10',confirm=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['private','samples','joint','all']);main(p.parse_args().step)
