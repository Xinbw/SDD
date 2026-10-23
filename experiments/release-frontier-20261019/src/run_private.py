"""Actual-cohort T=1 controls at 60 and 180 pre-release private SGD steps."""
import json
import math
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import engine_base as base
import privacy
from common import ROOT,PARENT,CLASSES,SETTINGS,SEEDS,freeze,frozen_design,release_key,save,sha

def project(a):
    u=np.sort(a,axis=-1)[:,::-1];z=u.cumsum(1)-1;j=np.arange(1,a.shape[1]+1)
    r=(u-z/j>0).sum(1).clip(1);theta=z[np.arange(len(r)),r-1]/r
    return np.maximum(a-theta[:,None],0)

def coverage(scores,weights):
    """Exact surjective assignment: free winners plus minimum-loss class representatives."""
    k,c=scores.shape
    assert k>=c
    utility=scores*weights[:,None];labels=utility.argmax(1)
    loss=utility.max(1)[:,None]-utility
    classes,regions=linear_sum_assignment(loss.T)
    labels[regions]=classes
    assert len(np.unique(labels))==c
    return labels

def decode(a,weights,mode):
    a=project(a)
    if mode=='coverage':return coverage(a,weights)
    if mode=='center':a=a-(a*weights[:,None]).sum(0)/weights.sum()
    return a.argmax(1)

def metrics(pred,truth,c):
    recall=np.array([(pred[truth==j]==j).mean() for j in range(c)])
    return dict(accuracy=float(np.mean(pred==truth)),balanced_accuracy=float(recall.mean()),
        per_class_recall=recall.tolist(),minimum_recall=float(recall.min()))

def regions(dataset,n,epsilon):
    c=CLASSES[dataset];out={(c,'M'),(c,'R')}
    prior=json.loads((PARENT/'public-selection'/f'{dataset}.json').read_text())
    p=next(x['selected'] for x in prior['records'] if x['n']==n and x['epsilon']==epsilon)
    out.add((p['k'],p['estimator']))
    if dataset=='web':out.add((12,'M'))
    if dataset=='femnist62':out.add((124,'M'))
    return sorted(out)

def reference_clip(dataset,n,epsilon):
    path=PARENT/'STUDY-SCHEDULE.json'
    job=next(x for x in json.loads(path.read_text())['design']['jobs']
        if x['dataset']==dataset and x['n']==n and x['epsilon']==epsilon and x['method']=='DP-Head')
    return job['clip']

def initial(data,n,seed):
    # Identical frozen parent checkpoints are a cache, never a different cohort.
    for path in (PARENT/'checkpoints').glob(f'private_*_n{n}_a*_s{seed}.pt'):
        fingerprint=__import__('hashlib').sha256(data['x_private'].tobytes()).hexdigest()[:10]
        if not path.name.startswith('private_'+fingerprint):continue
        local=ROOT/'checkpoints'/path.name
        if not local.exists():local.symlink_to(path.resolve())
    return base.initial_state(data,n,None if 'owners_private' in data else .1,seed)

def release(weights,data,dataset,n,epsilon,seed,steps,method,k,estimator,pop,counts=None):
    name=release_key(dataset,n,epsilon,seed,steps,method,k,estimator);output=ROOT/'results'/(name+'.json')
    if output.exists():return
    c=CLASSES[dataset]
    if method=='head':
        clip=reference_clip(dataset,n,epsilon);scale=1<<16
        radius=math.ceil(clip*scale+math.sqrt(128*c)/2)
        cert=privacy.certificate(epsilon,1e-5,1,n,integer_sensitivity=2*radius,dimension=128*c)
        cert['integer_report_radius']=radius
        norms=weights.flatten(1).norm(dim=1).clamp_min(1e-12)
        clipped=weights*(clip/norms).clamp(max=1)[:,None,None]
        quantized=torch.round(clipped.double()*scale).long()
        assert radius**2<2**53
        within=quantized.double().square().sum((1,2))<=radius**2
        quantized[~within]=0
        encoded=quantized.sum(0).cpu().numpy()
        noise=privacy.noise_sum(cert,128*c,seed=seed*1000003+steps*101+20261019)
        scores=(encoded+noise.reshape(128,c))/(n*scale)
        pred=(data['x_test']@scores).argmax(1)
        endpoints={'broadcast':metrics(pred,data['y_test'],c)};diagnostic=dict(scores=scores,integer_ball_guard_fraction=float((~within).float().mean()))
        logical_coordinates=128*c
    else:
        xp=torch.as_tensor(data['x_proxy'],device=base.DEVICE)
        if method=='per-example':
            ids=np.arange(len(xp));centres=data['x_proxy']
            if counts is None:counts=base.vote_counts(weights,xp)
        else:
            ids,reps,centres=base.partition(data,k)
            if counts is None:counts=base.vote_counts(weights,xp,ids,reps if estimator=='R' else None)
        k=len(counts);cert=privacy.certificate(epsilon,1e-5,1,n,clusters=k,dimension=k*c)
        scores=base.noisy_votes(counts,cert,seed*1000003+steps*101+20261019).cpu().numpy()
        row_weights=np.bincount(ids,minlength=k)
        test_ids=torch.cdist(torch.as_tensor(data['x_test'],device=base.DEVICE),torch.as_tensor(centres,device=base.DEVICE)).argmin(1).cpu().numpy()
        modes=['raw','center']+(['coverage'] if k>=c and method=='regional' else [])
        endpoints={mode:dict(**metrics(decode(scores,row_weights,mode)[test_ids],data['y_test'],c),
            labels=decode(scores,row_weights,mode).tolist()) for mode in modes}
        diagnostic=dict(counts=counts.cpu().numpy(),scores=scores,ids=ids,centres=centres)
        logical_coordinates=k*c
    path=ROOT/'diagnostics'/(name+'.npz');np.savez_compressed(path,**diagnostic)
    save(output,dict(dataset=dataset,n=n,epsilon=epsilon,seed=seed,steps=steps,method=method,k=k,estimator=estimator,
        releases=1,privacy=cert,population=pop,endpoints=endpoints,logical_report_coordinates=logical_coordinates,
        public_task_labels=0 if method!='head' else 'parent public clipping configuration',
        all_training_before_release_private_only=True,private_model_published=False,
        diagnostic_file=str(path.relative_to(ROOT)),diagnostic_sha256=sha(path)))
    print(name,{m:round(v['accuracy']*100,2) for m,v in endpoints.items()},flush=True)

def main():
    freeze('DESIGN-FROZEN.json',frozen_design())
    plan=[]
    for dataset,n,epsilon in SETTINGS:
        for seed in SEEDS:
            for steps in [60,180]:
                for k,e in regions(dataset,n,epsilon):plan.append(release_key(dataset,n,epsilon,seed,steps,'regional',k,e))
                plan.extend(release_key(dataset,n,epsilon,seed,steps,m,0,'-') for m in ['head','per-example'])
    freeze('PRIVATE-SCHEDULE.json',dict(expected_results=plan,settings=SETTINGS,seeds=SEEDS,
        privacy_per_counterfactual='one calibrated release, never joint access to all candidate releases'))
    for dataset in CLASSES:
        data=base.load_data(dataset)
        for n in sorted({n for d,n,e in SETTINGS if d==dataset}):
            epsilons=[e for d,nn,e in SETTINGS if d==dataset and nn==n]
            for seed in SEEDS:
                expected=[key for key in plan if key.startswith(f'{dataset}_{n}_') and f'_{seed}_T1_' in key]
                if all((ROOT/'results'/(key+'.json')).exists() for key in expected):continue
                w,v,idx,sizes,pop=initial(data,n,seed);w,v=w.clone(),v.clone()
                x,y,index,sz=[torch.as_tensor(a,device=base.DEVICE) for a in [data['x_private'],data['y_private'],idx,sizes]]
                for steps in [60,180]:
                    if steps==180:
                        for r in [2,3]:w,v=base.train_round(w,v,x,y,index,sz,seed,r)
                    pairs=sorted(set(pair for e in epsilons for pair in regions(dataset,n,e)))
                    xp=torch.as_tensor(data['x_proxy'],device=base.DEVICE)
                    for k,estimator in pairs:
                        pending=[e for e in epsilons if (k,estimator) in regions(dataset,n,e)
                            and not (ROOT/'results'/(release_key(dataset,n,e,seed,steps,'regional',k,estimator)+'.json')).exists()]
                        if not pending:continue
                        ids,reps,_=base.partition(data,k)
                        counts=base.vote_counts(w,xp,ids,reps if estimator=='R' else None)
                        for e in pending:release(w,data,dataset,n,e,seed,steps,'regional',k,estimator,pop,counts)
                    for method in ['head','per-example']:
                        counts=None
                        for e in epsilons:
                            key=release_key(dataset,n,e,seed,steps,method,0,'-')
                            if (ROOT/'results'/(key+'.json')).exists():continue
                            if method=='per-example' and counts is None:counts=base.vote_counts(w,xp)
                            release(w,data,dataset,n,e,seed,steps,method,0,'-',pop,counts)
                save(ROOT/'PRIVATE-STATUS.json',dict(dataset=dataset,n=n,seed=seed,
                    completed=len(list((ROOT/'results').glob('*.json'))),planned=len(plan)))
    assert all((ROOT/'results'/(key+'.json')).exists() for key in plan)
    save(ROOT/'PRIVATE-COMPLETE.json',dict(conditions=len(plan),utc=time.time()))

if __name__=='__main__':main()
