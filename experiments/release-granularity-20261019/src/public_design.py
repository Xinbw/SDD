"""Public-only Gaussian quality selector and independently tuned head clips.

The loader explicitly omits all private and final-test arrays. Validation labels
are used only for public configuration. Private clean counts never enter it.
"""
import hashlib
import json
import math

import numpy as np
import torch

import engine_base as base
import privacy
from experiment_engine import ROOT, save
from quality_model import predict


def public_data(dataset):
    bundle=np.load(ROOT/'inputs'/f'{dataset}.npz',allow_pickle=False)
    keys=['x_proxy','y_proxy','x_development','y_development','x_validation','y_validation']
    if 'owners_development' in bundle.files:keys.append('owners_development')
    data={key:bundle[key] for key in keys}
    data['x_private']=data['x_development'];data['y_private']=data['y_development']
    if 'owners_development' in data:data['owners_private']=data['owners_development']
    assert all('test' not in key for key in data)
    return data


def select(dataset, cohorts, budgets, ks):
    output=ROOT/'public-selection'/f'{dataset}_quality.json'
    if output.exists():
        result=json.loads(output.read_text());assert result['configuration_frozen'];return result
    data=public_data(dataset)
    public_n=min(1000,len(np.unique(data['owners_private']))) if 'owners_private' in data else 1000
    weights,_,_,_,population=base.initial_state(data,public_n,.1,1729,public=True)
    xp=torch.as_tensor(data['x_proxy'],device=base.DEVICE)
    xv=torch.as_tensor(data['x_validation'],device=base.DEVICE)
    classes=weights.shape[-1]
    candidates={}
    for k in ks:
        ids,_,centres=base.partition(data,k)
        assign=torch.cdist(xv,torch.as_tensor(centres,device=base.DEVICE)).argmin(-1).cpu().numpy()
        truth=np.stack([np.bincount(data['y_validation'][assign==i],minlength=classes) for i in range(k)])
        candidates[k]=dict(q=(base.vote_counts(weights,xp,ids).double()/public_n).cpu().numpy(),truth=truth,
            row_weights=np.bincount(ids,minlength=k))
    w=weights.cpu().numpy();norms=np.linalg.norm(w.reshape(public_n,-1),axis=1)
    clips=np.quantile(norms,[.5,.75,.9,.95,.99,1.]).tolist()
    records=[]
    for n in cohorts:
        for epsilon in budgets:
            options=[]
            for k,v in candidates.items():
                cert=privacy.certificate(epsilon,1e-5,3,n,clusters=k,dimension=k*classes)
                pred=predict(v['q'],v['truth'],v['row_weights'],cert['residual_std'],repeats=4096)
                options.append(dict(k=k,mode='center',**pred))
            chosen=max(options,key=lambda x:(x['predicted_accuracy'],-x['k']))
            head=[]
            # Six public-teacher quantiles, same criterion and all noise scales
            # evaluated at the actual deployment cohort size.
            for clip in clips:
                mean=(w*np.minimum(clip/np.maximum(norms,1e-12),1)[:,None,None]).mean(0)
                radius=math.ceil(clip*(1<<16)+math.sqrt(mean.size)/2)
                cert=privacy.certificate(epsilon,1e-5,3,n,integer_sensitivity=2*radius,dimension=mean.size)
                generator=torch.Generator(device=base.DEVICE).manual_seed(91001)
                noise=torch.randn((128,)+mean.shape,device=base.DEVICE,generator=generator)*cert['residual_std']
                scores=torch.matmul(xv,torch.as_tensor(mean,device=base.DEVICE)+noise)
                value=float((scores.argmax(-1)==torch.as_tensor(data['y_validation'],device=base.DEVICE)[None]).float().mean())
                head.append(dict(clip=clip,validation_accuracy=value))
            best=max(head,key=lambda x:x['validation_accuracy'])
            records.append(dict(n=n,epsilon=epsilon,selected_k=chosen['k'],quality_candidates=options,
                selected_clip=best['clip'],head_candidates=head))
            print('PUBLIC ONLY',dataset,n,epsilon,'K',chosen['k'],'clip',best['clip'],flush=True)
    result=dict(dataset=dataset,configuration_frozen=True,test_access=False,private_data_access=False,
        loaded_source_arrays=[k for k in data if k not in ['x_private','y_private','owners_private']],
        public_teachers=public_n,public_population=population,mode='center',candidate_k=ks,
        head_clip_quantiles=[.5,.75,.9,.95,.99,1.],criterion='expected public validation regional-label accuracy; head validation accuracy',
        gaussian_approximation=True,records=records,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/public_design.py',ROOT/'src/quality_model.py']})
    save(output,result)
    return result


if __name__=='__main__':
    select('cifar10',[100,1000,10000],[.025,.1,1.],[5,10,20,50,100])
    select('femnist62',[1000,2000],[.1,1.],[31,62,124])
