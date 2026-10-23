"""Freeze a deployable public-only resolution/estimator/decoder rule.

Public teacher bootstraps and deployment Gaussian noise are averaged. No private
or test arrays are read, and no private margins select a deployment option.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import engine_base as base
import privacy
from common import ROOT,freeze,save,sha

TASKS={'cifar10':dict(cohorts=[1000,10000],budgets=[.1],ks=[5,10,20,50,100],metric='accuracy'),
    'mnist':dict(cohorts=[10000],budgets=[.1],ks=[5,10,20,50,100],metric='accuracy'),
    'web':dict(cohorts=[3000],budgets=[1.],ks=[3,5,6,10,20,50],metric='balanced_accuracy'),
    'femnist62':dict(cohorts=[1000,2000],budgets=[.1,1.],ks=[31,62,124,248,500],metric='accuracy')}


def public_data(dataset):
    with np.load(ROOT/'inputs'/f'{dataset}.npz',allow_pickle=False) as b:
        keys=['x_proxy','x_development','y_development','x_validation','y_validation']
        if 'owners_development' in b.files:keys.append('owners_development')
        data={k:b[k] for k in keys}
    data['x_private']=data['x_development'];data['y_private']=data['y_development']
    if 'owners_development' in data:data['owners_private']=data['owners_development']
    return data,keys


def opinions(weights,xp,ids,reps,estimator):
    k=int(ids.max())+1
    assignment=F.one_hot(torch.as_tensor(ids,device=base.DEVICE),k).float()
    assignment/=assignment.sum(0,keepdim=True)
    output=[]
    for first in range(0,len(weights),256):
        logits=torch.matmul(xp[reps] if estimator=='R' else xp,weights[first:first+256])
        if estimator=='M':logits=torch.matmul(logits.softmax(-1).transpose(1,2),assignment).transpose(1,2)
        output.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(output)


def project(values):
    order=np.sort(values,axis=-1)[...,::-1];cssv=order.cumsum(-1)-1;ranks=np.arange(1,values.shape[-1]+1)
    rho=np.maximum((order-cssv/ranks>0).sum(-1),1)
    theta=np.take_along_axis(cssv,(rho-1)[...,None],axis=-1)[...,0]/rho
    return np.maximum(values-theta[...,None],0)


def decode(values,row_weights,decoder):
    a=project(values)
    if decoder=='center':return (a-(a*row_weights[:,None]).sum(-2,keepdims=True)/row_weights.sum()).argmax(-1)
    if decoder=='raw':return a.argmax(-1)
    labels=np.empty(a.shape[:-1],dtype=np.int64)
    for i in range(len(a)):
        rows,cols=linear_sum_assignment(-a[i]*row_weights[:,None]);labels[i,rows]=cols
    return labels


def truth_table(data,centres,classes,metric):
    assign=torch.cdist(torch.as_tensor(data['x_validation'],device=base.DEVICE),
        torch.as_tensor(centres,device=base.DEVICE)).argmin(-1).cpu().numpy()
    counts=np.stack([np.bincount(data['y_validation'][assign==i],minlength=classes) for i in range(len(centres))])
    if metric=='balanced_accuracy':
        counts=counts/np.maximum(counts.sum(0,keepdims=True),1)
    return counts/counts.sum()


def select_dataset(dataset):
    path=ROOT/'public-selection'/f'{dataset}.json'
    if path.exists():
        out=json.loads(path.read_text());assert out['configuration_frozen'];return out
    spec=TASKS[dataset];data,loaded=public_data(dataset)
    public_n=min(1000,len(np.unique(data['owners_private']))) if 'owners_private' in data else 1000
    w,_,_,_,population=base.initial_state(data,public_n,None if dataset=='web' else .1,1729,public=True)
    classes=w.shape[-1];xp=torch.as_tensor(data['x_proxy'],device=base.DEVICE)
    public_profiles={};tables={};region_weights={}
    for k in spec['ks']:
        ids,reps,centres=base.partition(data,k)
        tables[k]=truth_table(data,centres,classes,spec['metric'])
        region_weights[k]=np.bincount(ids,minlength=k).astype(np.float64)
        for estimator in ['M','R']:
            op=opinions(w,xp,ids,reps,estimator)
            q=np.stack([(op==c).mean(0) for c in range(classes)],axis=-1)
            boot=[]
            rng=np.random.default_rng(92001)
            for b in range(8):
                draw=op[rng.integers(public_n,size=public_n)]
                boot.append(np.stack([(draw==c).mean(0) for c in range(classes)],axis=-1))
            public_profiles[(k,estimator)]=(q,np.stack(boot))
            np.savez_compressed(ROOT/'public-selection'/f'{dataset}_K{k}_{estimator}.npz',q=q,bootstrap_q=np.stack(boot),
                teacher_opinions=op,validation_class_weights=tables[k],proxy_region_weights=region_weights[k])
        print('PUBLIC PROFILE',dataset,k,flush=True)
    records=[]
    for n in spec['cohorts']:
        for epsilon in spec['budgets']:
            options=[]
            for k in spec['ks']:
                cert=privacy.certificate(epsilon,1e-5,3,n,clusters=k,dimension=k*classes)
                rng=np.random.default_rng(93001)
                noise=rng.standard_normal((8,256,k,classes))*cert['residual_std']
                for estimator in ['M','R']:
                    q,boot=public_profiles[(k,estimator)]
                    sample=(boot[:,None]+noise).reshape(2048,k,classes)
                    for decoder in ['center','raw']+(['matching'] if k==classes else []):
                        labels=decode(sample,region_weights[k],decoder)
                        scores=tables[k][np.arange(k)[None],labels].sum(1)
                        by_teacher=scores.reshape(8,256).mean(1)
                        options.append(dict(k=k,estimator=estimator,decoder=decoder,score=float(scores.mean()),
                            public_teacher_bootstrap_sd=float(by_teacher.std(ddof=1)),noise_mc_sem=float(scores.std(ddof=1)/np.sqrt(2048)),
                            public_teacher_scores=by_teacher.tolist(),sigma=cert['residual_std']))
            chosen=max(options,key=lambda c:(c['score'],-c['k'],c['estimator']=='M',c['decoder']=='center'))
            records.append(dict(n=n,epsilon=epsilon,selected={k:chosen[k] for k in ['k','estimator','decoder']},
                selected_predicted_score=chosen['score'],candidates=options))
            print('FROZEN CHOICE',dataset,n,epsilon,chosen,flush=True)
    out=dict(dataset=dataset,configuration_frozen=True,private_data_access=False,test_access=False,loaded_source_arrays=loaded,
        public_teacher_population=population,public_teachers=public_n,metric=spec['metric'],records=records,
        teacher_bootstraps=8,noise_repeats_per_bootstrap=256,selection_seed=1729,noise_seed=93001,
        candidate_estimators=['M','R'],candidate_decoders=['center','raw','matching only at K=C'],
        criterion='expected public validation nearest-centroid accuracy; balanced accuracy for Web',
        private_test_outcomes_select_configuration=False,
        source_sha256={p.name:sha(p) for p in (ROOT/'src').glob('*.py')})
    save(path,out)
    return out


def main():
    freeze(ROOT/'PUBLIC-DESIGN-FROZEN.json',dict(tasks=TASKS,teacher_bootstraps=8,noise_repeats_per_bootstrap=256,
        estimators=['M','R'],decoders=['center','raw','matching only at K=C'],private_data_access=False,test_access=False,
        private_seeds=list(range(8501,8511)),mix_seeds=list(range(8601,8611))))
    for dataset in TASKS:select_dataset(dataset)
    save(ROOT/'PUBLIC-CHOICES-FROZEN.json',dict(files={p.name:sha(p) for p in sorted((ROOT/'public-selection').glob('*.json'))},
        all_four_tasks_frozen=True,private_evaluation_started=False))


if __name__=='__main__':main()
