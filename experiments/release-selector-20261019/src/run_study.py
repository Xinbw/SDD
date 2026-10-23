"""Freeze all fresh end-to-end, FEMNIST resolution, and mixing conditions."""
import json
import time
import numpy as np
import torch

import engine_base as base
from common import ROOT,key,freeze,save,sha
from selection import TASKS
from learning import run
from public_models import select as select_public_model,fit,evaluate


def head_clip(dataset,n,epsilon):
    if dataset=='femnist62':
        path=ROOT.parent/'release-granularity-20261019/public-selection/femnist62_quality.json'
        row=next(r for r in json.loads(path.read_text())['records'] if r['n']==n and r['epsilon']==epsilon)
        return row['selected_clip'],str(path.relative_to(ROOT.parent))
    population=1000 if dataset=='web' else n
    alpha='None' if dataset=='web' else '0.1'
    path=ROOT.parent/'review-evidence-20261012/public-selection'/f'{dataset}_a{alpha}_n{population}.json'
    return json.loads(path.read_text())['budgets'][str(epsilon)]['clip'],str(path.relative_to(ROOT.parent))


def job(dataset,n,epsilon,seed,method='SDD',k=None,decoder='center',estimator='M',ratio=(64,64)):
    k=k or (62 if dataset=='femnist62' else 6 if dataset=='web' else 10)
    out=dict(dataset=dataset,n=n,epsilon=epsilon,seed=seed,method=method,k=k,decoder=decoder,estimator=estimator,
        proxy_r2=ratio[0],proxy_r3=ratio[1])
    if method=='DP-Head':out['clip'],out['public_clipping_source']=head_clip(dataset,n,epsilon)
    return out


def make_plan():
    assert (ROOT/'PUBLIC-CHOICES-FROZEN.json').exists()
    public=select_public_model()
    registry={};roles=[]
    def add(j,role,condition):
        identifier=key(j)
        if identifier in registry:assert registry[identifier]==j
        else:registry[identifier]=j
        roles.append(dict(role=role,condition=condition,job=identifier))
    for dataset,spec in TASKS.items():
        design=json.loads((ROOT/'public-selection'/f'{dataset}.json').read_text())
        for cfg in design['records']:
            n,epsilon=cfg['n'],cfg['epsilon'];condition=dict(dataset=dataset,n=n,epsilon=epsilon)
            for seed in range(8501,8511):
                add(job(dataset,n,epsilon,seed,**cfg['selected']),'public-selected',condition)
                add(job(dataset,n,epsilon,seed),'fixed-center',condition)
                add(job(dataset,n,epsilon,seed,method='DP-FedCT'),'per-example',condition)
                add(job(dataset,n,epsilon,seed,method='DP-Head'),'head',condition)
                add(job(dataset,n,0.,seed,method='Local-only',ratio=(0,0)),'local',condition)
    for n in [1000,2000]:
        for seed in range(8501,8511):
            add(job('femnist62',n,0.,seed,method='Local+Public'),'public-source-local',dict(dataset='femnist62',n=n))
    for n in [1000,2000]:
        for epsilon in [.1,1.]:
            for k in [31,62,124,248,500]:
                for seed in range(8501,8506):
                    add(job('femnist62',n,epsilon,seed,k=k),'natural-resolution-grid',dict(dataset='femnist62',n=n,epsilon=epsilon,k=k))
    for seed in range(8601,8611):
        for ratio in [(0,0),(32,32),(64,64),(96,96),(128,128),(64,128)]:
            add(job('cifar10',10000,.1,seed,decoder='matching',ratio=ratio),'training-mix',dict(dataset='cifar10',n=10000,epsilon=.1,proxy_r2=ratio[0],proxy_r3=ratio[1]))
    plan=dict(jobs=list(registry.values()),roles=roles,fresh_end_to_end_seeds=list(range(8501,8511)),
        training_mix_seeds=list(range(8601,8611)),public_only_final_seeds=list(range(8501,8511)),
        public_only_configurations={k:{p:public[k][p] for p in ['lr','steps']} for k in ['selected','matched_180']},
        public_choices_sha256=sha(ROOT/'PUBLIC-CHOICES-FROZEN.json'),
        all_private_jobs_fixed_before_new_private_outcomes=True,all_endpoints_reported=True,
        K_grid_does_not_route_by_private_test=True,deduplicated_identical_training_configs=True)
    freeze(ROOT/'STUDY-SCHEDULE.json',plan)
    return plan


def main():
    plan=make_plan();public_cfg=plan['public_only_configurations'];public_data=None;teachers={}
    print('FROZEN UNIQUE TRAINING JOBS',len(plan['jobs']),'comparison-role entries',len(plan['roles']),flush=True)
    for seed in range(8501,8511):
        with np.load(ROOT/'inputs/femnist62.npz',allow_pickle=False) as b:
            public_data={k:b[k] for k in ['x_development','y_development','x_test','y_test']}
        for label in ['matched_180','selected']:
            output=ROOT/'results'/f'femnist62_Public-only_{label}_{seed}.json'
            if output.exists():continue
            choice=public_cfg[label];w=fit(public_data,seed,choice['lr'],choice['steps'])
            result=dict(dataset='femnist62',seed=seed,method='Public-only-'+label,**evaluate(w,public_data,'test'),
                private_data_used=False,public_development_labels=5000,optimizer_steps=choice['steps'],lr=choice['lr'],
                private_privacy_cost=0,logical_report_coordinates=0,configuration_used_test=False)
            save(output,result)
            print(output.stem,result['accuracy'],flush=True)
    for i,j in enumerate(plan['jobs']):
        output=ROOT/'results'/(key(j)+'.json')
        if not output.exists():
            teacher=None
            if j['method']=='Local+Public':
                if j['seed'] not in teachers:
                    cfg=public_cfg['selected'];teachers[j['seed']]=fit(public_data,j['seed'],cfg['lr'],cfg['steps'])[0]
                teacher=teachers[j['seed']]
            run(j,teacher)
        save(ROOT/'STUDY-STATUS.json',dict(completed=i+1,total=len(plan['jobs']),last=key(j)))
    save(ROOT/'STUDY-COMPLETE.json',dict(unique_private_training_conditions=len(plan['jobs']),public_only_conditions=20,
        comparison_role_entries=len(plan['roles']),utc=time.time()))


if __name__=='__main__':main()
