"""Freeze release configurations using public development data only.

The objective is label-transfer accuracy on the public validation split, averaged
over five predeclared noise repeats. Sealed final test arrays are never loaded.
"""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch

import engine
import privacy

ROOT=engine.ROOT
EPS=[.01,.025,.05,.1,.25,.5,1.]
KS=[5,10,20,50,100]
PUBLIC_SEED=1729
REPEATS=[91001,91002,91003,91004,91005]


def design_noise(mean,cert,seed):
    noise=privacy.noise_sum(cert,mean.numel(),seed=seed)
    return mean+torch.as_tensor(noise.reshape(mean.shape)/(cert['actual_clients']*cert['quantization_scale']),
                               device='cuda',dtype=torch.float32)


def metric(prediction,data):
    acc,balanced=engine.score_labels(prediction.cpu().numpy(),data['y_validation'],int(data['y_private'].max())+1)
    return balanced if 'owners_private' in data else acc


def select(dataset,alpha,target_n):
    output=ROOT/'public-selection'/f'{dataset}_a{alpha}_n{target_n}.json'
    if output.exists() and json.loads(output.read_text()).get('configuration_frozen'):return
    started=time.time();data=engine.load_data(dataset,public=True)
    public_n=min(1000,len(np.unique(data['owners_private']))) if 'owners_private' in data else 1000
    weights,_,_,_,population=engine.initial_state(data,public_n,alpha,PUBLIC_SEED,public=True)
    xp=torch.as_tensor(data['x_proxy'],device='cuda');xv=torch.as_tensor(data['x_validation'],device='cuda')
    c=weights.shape[-1];p=len(xp)
    result={'dataset':dataset,'alpha':alpha,'target_private_clients':target_n,
            'public_bootstrap_teachers':public_n,'public_population':population,
            'candidate_clusters':KS,'noise_repeats':REPEATS,'selection_seed':PUBLIC_SEED,
            'objective':'balanced validation accuracy' if dataset=='web' else 'validation accuracy',
            'selection_uses_only_public_data':True,'test_access':False,
            'public_pilot_noise_is_a_design_surrogate_not_a_private_DP_evaluation':True,'budgets':{}}
    counts_cache={};assign_cache={};centres_cache={}
    for k in KS:
        ids,reps,centres=engine.partition(data,k,'semantic')
        counts_cache[k]=engine.vote_counts(weights,xp,ids).float()/public_n
        centres_cache[k]=centres
        assign_cache[k]=torch.cdist(xv,torch.as_tensor(centres,device='cuda')).argmin(-1)
    full_counts=engine.vote_counts(weights,xp).float()/public_n
    w_cpu=weights.cpu().numpy();norms=np.linalg.norm(w_cpu.reshape(public_n,-1),axis=1)
    clips=np.quantile(norms,[.5,.75,.9,.95,.99,1.0]).tolist()
    for epsilon in EPS:
        candidates=[]
        for k in KS:
            mean=counts_cache[k]
            ids,_,_=engine.partition(data,k,'semantic')
            row_weights=torch.bincount(torch.as_tensor(ids,device='cuda'),minlength=k).float()
            cert=privacy.certificate(epsilon,1e-5,3,target_n,clusters=k,dimension=k*c)
            values={'raw':[],'center':[]}
            for seed in REPEATS:
                score=design_noise(mean,cert,seed)
                for mode in values:
                    prediction=engine.labels_from_scores(score,mode,row_weights)[assign_cache[k]]
                    values[mode].append(metric(prediction,data))
            for mode,vals in values.items():candidates.append({'k':k,'mode':mode,'score':float(np.mean(vals)),'repeats':vals})
        # Prefer smaller payload, then raw consensus, on exactly tied scores.
        chosen=max(candidates,key=lambda x:(x['score'],-x['k'],x['mode']=='raw'))
        head_candidates=[]
        for clip in clips:
            factors=np.minimum(clip/np.maximum(norms,1e-12),1.)
            mean=torch.as_tensor((w_cpu*factors[:,None,None]).mean(0),device='cuda')
            cert=privacy.certificate(epsilon,1e-5,3,target_n,clip=clip,dimension=mean.numel())
            vals=[]
            for seed in REPEATS:
                head=design_noise(mean,cert,seed)
                vals.append(metric(torch.matmul(xv,head).argmax(-1),data))
            head_candidates.append({'clip':clip,'score':float(np.mean(vals)),'repeats':vals})
        best_head=max(head_candidates,key=lambda x:x['score'])
        pate_candidates=[]
        for threshold in [.1,.2,.3]:
            for cap_fraction in [.25,.5]:
                for screen_fraction in [.25,.5]:
                    cap=int(p*cap_fraction)
                    screen=privacy.certificate(epsilon,1e-5,3,target_n,l2_sensitivity=math.sqrt(p),dimension=p,
                        rho_fraction=screen_fraction,accounting='rdp')
                    answer=privacy.certificate(epsilon,1e-5,3,target_n,clusters=cap,dimension=cap*c,
                        rho_fraction=1-screen_fraction,accounting='rdp')
                    vals=[];coverage=[]
                    for seed in REPEATS[:2]:
                        scores=design_noise(full_counts.max(-1).values[:,None],screen,seed).flatten()
                        selected=torch.where(scores>=threshold)[0]
                        selected=selected[torch.argsort(scores[selected],descending=True)[:cap]]
                        if len(selected)==0:vals.append(0.);coverage.append(0.);continue
                        answer_scores=design_noise(full_counts[selected],answer,seed+37)
                        labels=engine.labels_from_scores(answer_scores,chosen['mode'])
                        closest=torch.cdist(xv,xp[selected]).argmin(-1)
                        vals.append(metric(labels[closest],data));coverage.append(len(selected)/p)
                    pate_candidates.append(dict(threshold=threshold,cap_fraction=cap_fraction,
                        screen_rho_fraction=screen_fraction,score=float(np.mean(vals)),coverage=float(np.mean(coverage))))
        best_pate=max(pate_candidates,key=lambda x:x['score'])
        result['budgets'][str(epsilon)]={'k':chosen['k'],'mode':chosen['mode'],'clip':best_head['clip'],
            'pate':{k:best_pate[k] for k in ['threshold','cap_fraction','screen_rho_fraction']},
            'semantic_candidates':candidates,'head_candidates':head_candidates,'pate_candidates':pate_candidates}
        print(dataset,alpha,target_n,epsilon,'K',chosen['k'],chosen['mode'],'public score',round(chosen['score'],4),flush=True)
        output.write_text(json.dumps(result,indent=2)+'\n')
    result['seconds']=time.time()-started;result['configuration_frozen']=True
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    for dataset in ['cifar10','mnist']:
        for alpha in [.1,1.]:
            for n in [100,1000,10000]:select(dataset,alpha,n)
    for n in [100,1000,3000]:select('web',None,n)
    outputs=sorted((ROOT/'public-selection').glob('*_a*_n*.json'))
    manifest={'all_configs_frozen':True,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
              'sealed_test_evaluation_started':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'src').glob('*.py')}}
    (ROOT/'public-selection/FROZEN.json').write_text(json.dumps(manifest,indent=2)+'\n')
