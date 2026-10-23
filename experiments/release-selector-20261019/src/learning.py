"""Actual-client end-to-end releases, with public and private endpoints separate."""
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import engine_base as base
import privacy
from common import ROOT,key,save,sha


def decode(scores,weights,mode):
    a=base.project_rows(scores)
    if mode=='center':return (a-(a*weights[:,None]).sum(0)/weights.sum()).argmax(-1)
    if mode=='raw':return a.argmax(-1)
    rows,columns=linear_sum_assignment(-(a*weights[:,None]).cpu().numpy())
    labels=np.empty(len(a),dtype=np.int64);labels[rows]=columns
    return torch.as_tensor(labels,device=base.DEVICE)


def train(weights,velocity,x,y,index,sizes,seed,r,xp,targets,proxy_count):
    if targets is None or proxy_count==0:
        return base.train_round(weights,velocity,x,y,index,sizes,seed,r)
    if proxy_count==64:return base.train_round(weights,velocity,x,y,index,sizes,seed,r,xp,targets)
    n=len(weights);private_count=128-proxy_count
    gen=torch.Generator(device=base.DEVICE).manual_seed(seed*100003+r*101)
    genp=torch.Generator(device=base.DEVICE).manual_seed(seed*100019+r*103)
    for _ in range(60):
        local=(torch.rand((n,private_count),device=base.DEVICE,generator=gen)*sizes[:,None]).long()
        gids=index.gather(1,local)
        for first in range(0,n,1024):
            last=min(n,first+1024)
            picks=torch.randint(len(xp),(last-first,proxy_count),device=base.DEVICE,generator=genp)
            features=torch.cat([x[gids[first:last]],xp[picks]],1)
            labels=torch.cat([y[gids[first:last]],targets[picks]],1)
            w=weights[first:last];residual=torch.bmm(features,w).softmax(-1)
            residual.scatter_add_(-1,labels[...,None],-torch.ones_like(labels[...,None],dtype=residual.dtype))
            grad=torch.bmm(features.transpose(1,2),residual)/128+base.WEIGHT_DECAY*w
            velocity[first:last].mul_(base.MOMENTUM).add_(grad);w.add_(velocity[first:last],alpha=-base.LR)
    return weights,velocity


def release(weights,data,job,r,diagnostic,assignment):
    n,_,c=weights.shape;xp=torch.as_tensor(data['x_proxy'],device=base.DEVICE)
    if job['method']=='DP-FedCT':
        ids=np.arange(len(xp));counts=base.vote_counts(weights,xp);eval_ids=assignment
    else:
        ids,reps,centres=base.partition(data,job['k'])
        counts=base.vote_counts(weights,xp,ids,reps if job['estimator']=='R' else None);eval_ids=assignment
    k=len(counts);cert=privacy.certificate(job['epsilon'],1e-5,3,n,clusters=k,dimension=k*c)
    scores=base.noisy_votes(counts,cert,job['seed']*1000003+r*101)
    row_weights=torch.bincount(torch.as_tensor(ids,device=base.DEVICE),minlength=k).float()
    label=decode(scores,row_weights,job['decoder'])
    clean=decode(counts.float()/n,row_weights,job['decoder'])
    truth=np.stack([np.bincount(data['y_proxy'][ids==i],minlength=c) for i in range(k)])
    diagnostic.update({f'r{r}_counts':counts.cpu().numpy(),f'r{r}_scores':scores.cpu().numpy(),
        f'r{r}_labels':label.cpu().numpy(),'truth_counts':truth,'ids':ids})
    pred=label.cpu().numpy()[eval_ids]
    acc,balanced=base.score_labels(pred,data['y_test'],c)
    info=dict(logical_report_coordinates=k*c,public_cluster_purity=float(truth.max(1).sum()/truth.sum()),
        proxy_label_accuracy=float((label.cpu().numpy()[ids]==data['y_proxy']).mean()),
        clean_proxy_label_accuracy=float((clean.cpu().numpy()[ids]==data['y_proxy']).mean()),
        distinct_target_classes=int(len(label.unique())),flip_fraction=float((label!=clean).float().mean()),
        global_classifier_accuracy=acc,global_classifier_balanced_accuracy=balanced,
        global_classifier='nearest proxy example' if job['method']=='DP-FedCT' else 'nearest public semantic centroid',
        public_postprocessing_only=True,additional_privacy_cost=0)
    return label[torch.as_tensor(ids,device=base.DEVICE)],cert,info


def run(job,public_teacher=None):
    start=time.time();data=base.load_data(job['dataset'])
    w,v,idx,sizes,pop=base.initial_state(data,job['n'],None if job['dataset']=='web' else .1,job['seed'])
    w,v=w.clone(),v.clone()
    x,y,xp,xt,yt,index,sizes_t=[torch.as_tensor(z,device=base.DEVICE) for z in
        [data['x_private'],data['y_private'],data['x_proxy'],data['x_test'],data['y_test'],idx,sizes]]
    if job['method']=='DP-FedCT':assignment=torch.cdist(xt,xp).argmin(-1).cpu().numpy()
    elif job['method']=='SDD':
        _,_,centres=base.partition(data,job['k'])
        assignment=torch.cdist(xt,torch.as_tensor(centres,device=base.DEVICE)).argmin(-1).cpu().numpy()
    else:assignment=None
    targets=None if public_teacher is None else (xp@public_teacher).argmax(-1)
    reference=torch.zeros_like(w[0]);trace=[];diagnostic={};head_local=None
    for r in range(1,4):
        if r>1:
            proxy=job['proxy_r2'] if r==2 else job['proxy_r3']
            w,v=train(w,v,x,y,index,sizes_t,job['seed'],r,xp,targets,proxy)
        if job['method'] in ['Local-only','Local+Public']:continue
        if job['method']=='DP-Head':
            if r==3:
                a,b=base.evaluate(w,xt,yt);head_local=dict(accuracy=a,balanced_accuracy=b,steps=180)
            reference,cert,info=base.parameter_release(w,reference,job['clip'],job['epsilon'],job['seed'],r)
            w=reference[None].expand(job['n'],-1,-1).clone();v.zero_();targets=(xp@reference).argmax(-1)
        else:targets,cert,info=release(w,data,job,r,diagnostic,assignment)
        trace.append(dict(round=r,privacy=cert,diagnostics=info))
    acc,balanced=base.evaluate(w,xt,yt)
    result=dict(**job,accuracy=acc,balanced_accuracy=balanced,per_class_recall=base.evaluate.last_recall,
        actual_clients=job['n'],calibrated_clients=job['n'],population=pop,trace=trace,head_local_endpoint=head_local,
        optimizer_steps=180,batch_size=128,release_rounds=len(trace),delta=1e-5,
        private_model_released=False,global_endpoint_is_DP_postprocessing=True if job['method'] in ['SDD','DP-FedCT','DP-Head'] else None,
        final_label_release_supervises_local_model=False,selection_used_private_data=False,selection_used_final_test=False,
        seconds=time.time()-start,metrics_and_clean_counts_are_offline_diagnostics=True)
    if diagnostic:
        file=ROOT/'diagnostics'/(key(job)+'.npz');np.savez_compressed(file,**diagnostic)
        result['diagnostic_file']=str(file.relative_to(ROOT));result['diagnostic_sha256']=sha(file)
        result['global_endpoint']=dict(accuracy=trace[-1]['diagnostics']['global_classifier_accuracy'],
            balanced_accuracy=trace[-1]['diagnostics']['global_classifier_balanced_accuracy'],release=3,
            trained_on_private_data=False,extra_private_queries=0)
    elif job['method']=='DP-Head':result['global_endpoint']=dict(accuracy=acc,balanced_accuracy=balanced,release=3)
    save(ROOT/'results'/(key(job)+'.json'),result)
    print(key(job),'local',round(acc*100,4),'global',round(result.get('global_endpoint',{}).get('accuracy',0)*100,4),
        round(result['seconds'],1),flush=True)
    return result
