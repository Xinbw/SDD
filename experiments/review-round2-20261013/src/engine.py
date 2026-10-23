"""Matched-population, vectorized independent-client learning experiments.

Each tensor row is a separately trained classifier with its own client data.
Vectorization changes execution order, not the number of private units or reports.
"""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

import privacy

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
DEVICE='cuda'
STEPS=60
BATCH=128
LR=.1
MOMENTUM=.9
WEIGHT_DECAY=5e-4


def load_data(dataset,public=False):
    bundle=np.load(ROOT/'inputs'/f'{dataset}.npz',allow_pickle=False)
    keys=[k for k in bundle.files if public is False or 'test' not in k]
    data={k:bundle[k] for k in keys}
    if public:
        data['x_private']=data['x_development'];data['y_private']=data['y_development']
        if 'owners_development' in data:data['owners_private']=data['owners_development']
    return data


def make_population(data,n,alpha,seed):
    rng=np.random.default_rng(seed)
    y=data['y_private'];classes=int(y.max())+1
    if 'owners_private' in data:
        owners=np.unique(data['owners_private'])
        if n>len(owners):raise ValueError(f'{n} clients requested, {len(owners)} distinct authors available')
        chosen=rng.choice(owners,n,replace=False)
        indices=[np.where(data['owners_private']==owner)[0] for owner in chosen]
        length=max(map(len,indices));matrix=np.zeros((n,length),dtype=np.int64)
        sizes=np.array([len(i) for i in indices],dtype=np.int64)
        for j,idx in enumerate(indices):matrix[j,:len(idx)]=idx
        owner_hash=hashlib.sha256(chosen.tobytes()).hexdigest()
    else:
        sizes=np.full(n,600,dtype=np.int64);matrix=np.empty((n,600),dtype=np.int64)
        pools=[np.where(y==c)[0] for c in range(classes)]
        proportions=rng.dirichlet(np.full(classes,alpha),size=n)
        for j in range(n):
            labels=rng.choice(classes,size=600,p=proportions[j])
            for c in range(classes):
                mask=labels==c
                matrix[j,mask]=rng.choice(pools[c],size=int(mask.sum()),replace=True)
        owner_hash=None
    return matrix,sizes,dict(actual_clients=n,unique_client_identifiers=n,
        per_client_records_min=int(sizes.min()),per_client_records_median=float(np.median(sizes)),
        per_client_records_max=int(sizes.max()),distinct_author_hash=owner_hash,
        client_sample_indices_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        client_generation_seed=seed,alpha=alpha if owner_hash is None else None)


def partition(data,k,kind='semantic'):
    folder=ROOT/'inputs/partitions';folder.mkdir(exist_ok=True)
    digest=hashlib.sha256(data['x_proxy'].tobytes()).hexdigest()[:12]
    path=folder/f'{digest}_K{k}_{kind}.npz'
    if path.exists():
        p=np.load(path);return p['ids'].astype(np.int64),p['representatives'].astype(np.int64),p['centres']
    if kind=='random':
        semantic,_,_=partition(data,k,'semantic')
        ids=np.empty_like(semantic);permutation=np.random.default_rng(1729+k).permutation(len(semantic))
        ids[permutation]=np.sort(semantic)
        x=data['x_proxy'];centres=np.stack([x[ids==c].mean(0) for c in range(k)])
    else:
        x=data['x_proxy_weak'] if kind=='weak' else data['x_proxy']
        fitted=KMeans(n_clusters=k,n_init=5,max_iter=100,random_state=1729).fit(x)
        ids=fitted.labels_.astype(np.int64);centres=fitted.cluster_centers_.copy()
        # Public deterministic repair handles duplicate feature rows without changing K.
        for c in range(k):
            if not (ids==c).any():
                counts=np.bincount(ids,minlength=k)
                candidates=np.where(counts[ids]>1)[0]
                residual=((x[candidates]-centres[ids[candidates]])**2).sum(1)
                point=candidates[np.argmax(residual)];ids[point]=c;centres[c]=x[point]
    assert len(np.unique(ids))==k
    representatives=np.array([np.where(ids==c)[0][np.argmin(((x[ids==c]-centres[c])**2).sum(1))] for c in range(k)])
    np.savez_compressed(path,ids=ids,representatives=representatives,centres=centres)
    return ids,representatives,centres


def train_round(weights,velocity,x,y,index,sizes,seed,round_number,proxy_x=None,targets=None):
    n,d,c=weights.shape
    generator=torch.Generator(device=DEVICE).manual_seed(seed*100003+round_number*101)
    generator_proxy=torch.Generator(device=DEVICE).manual_seed(seed*100019+round_number*103)
    valid=None if targets is None else torch.where(targets>=0)[0]
    use_proxy=valid is not None and len(valid)>0
    private_batch=BATCH//2 if use_proxy else BATCH
    for step in range(STEPS):
        local=(torch.rand((n,private_batch),device=DEVICE,generator=generator)*sizes[:,None]).long()
        global_ids=index.gather(1,local)
        for start in range(0,n,1024):
            end=min(n,start+1024)
            features=x[global_ids[start:end]];labels=y[global_ids[start:end]]
            if use_proxy:
                picks=torch.randint(len(valid),(end-start,BATCH-private_batch),device=DEVICE,generator=generator_proxy)
                pids=valid[picks]
                features=torch.cat([features,proxy_x[pids]],dim=1)
                labels=torch.cat([labels,targets[pids]],dim=1)
            w=weights[start:end]
            logits=torch.bmm(features,w)
            residual=logits.softmax(-1)
            residual.scatter_add_(-1,labels[...,None],-torch.ones_like(labels[...,None],dtype=residual.dtype))
            grad=torch.bmm(features.transpose(1,2),residual)/BATCH+WEIGHT_DECAY*w
            velocity[start:end].mul_(MOMENTUM).add_(grad)
            w.add_(velocity[start:end],alpha=-LR)
    return weights,velocity


def initial_state(data,n,alpha,seed,public=False):
    stage='public' if public else 'private'
    path=ROOT/'checkpoints'/f'{stage}_{hashlib.sha256(data["x_private"].tobytes()).hexdigest()[:10]}_n{n}_a{alpha}_s{seed}.pt'
    indices,sizes,meta=make_population(data,n,alpha,seed)
    parent_cache=ROOT.parent/'review-evidence-20261012'/'checkpoints'/path.name
    cache=path if path.exists() else parent_cache
    if cache.exists():
        state=torch.load(cache,map_location=DEVICE,weights_only=False)
        assert state['population']==meta
        return state['weights'],state['velocity'],indices,sizes,meta
    x=torch.as_tensor(data['x_private'],device=DEVICE);y=torch.as_tensor(data['y_private'],device=DEVICE)
    generator=torch.Generator(device=DEVICE).manual_seed(seed)
    weights=torch.randn((n,x.shape[1],int(y.max())+1),device=DEVICE,generator=generator)*.01
    velocity=torch.zeros_like(weights)
    weights,velocity=train_round(weights,velocity,x,y,torch.as_tensor(indices,device=DEVICE),
        torch.as_tensor(sizes,device=DEVICE),seed,1)
    torch.save({'weights':weights.cpu(),'velocity':velocity.cpu(),'population':meta},path)
    return weights,velocity,indices,sizes,meta


def vote_counts(weights,proxy_x,ids=None,representatives=None):
    n,d,c=weights.shape
    if ids is None:
        rows=len(proxy_x);assignment=None
    else:
        rows=int(ids.max())+1
        assignment=F.one_hot(torch.as_tensor(ids,device=DEVICE),rows).float()
        assignment=assignment/assignment.sum(0,keepdim=True)
    counts=torch.zeros((rows,c),device=DEVICE,dtype=torch.int64)
    for start in range(0,n,512):
        w=weights[start:start+512]
        if representatives is not None:
            logits=torch.matmul(proxy_x[representatives],w)
            opinions=logits.argmax(-1)
        else:
            logits=torch.matmul(proxy_x,w)
            if assignment is None:opinions=logits.argmax(-1)
            else:
                probabilities=logits.softmax(-1)
                means=torch.matmul(probabilities.transpose(1,2),assignment).transpose(1,2)
                opinions=means.argmax(-1)
        counts+=F.one_hot(opinions,c).sum(0)
    return counts


def project_rows(a):
    sorted_a,_=torch.sort(a,dim=-1,descending=True)
    sums=sorted_a.cumsum(-1)-1
    ranks=torch.arange(1,a.shape[-1]+1,device=a.device,dtype=a.dtype)
    active=(sorted_a-sums/ranks)>0
    rho=active.sum(-1).clamp_min(1)
    theta=sums.gather(-1,(rho-1)[...,None]).squeeze(-1)/rho
    return torch.clamp(a-theta[...,None],min=0)


def labels_from_scores(scores,mode='raw',row_weights=None):
    a=project_rows(scores)
    if mode=='center':
        if row_weights is None:prior=a.mean(0)
        else:prior=(a*row_weights[:,None]).sum(0)/row_weights.sum()
        a=a-prior
    return a.argmax(-1)


def noisy_votes(counts,cert,seed):
    shape=counts.shape
    noise=privacy.noise_sum(cert,counts.numel(),seed=seed)
    quantized=counts.cpu().numpy().astype(np.int64)*cert['quantization_scale']
    released=quantized+noise.reshape(shape)
    return torch.as_tensor(released.astype(np.float64)/(cert['actual_clients']*cert['quantization_scale']),device=DEVICE,dtype=torch.float32)


def score_labels(prediction,labels,classes):
    prediction=np.asarray(prediction);labels=np.asarray(labels)
    acc=float((prediction==labels).mean())
    balanced=float(np.mean([(prediction[labels==c]==c).mean() for c in range(classes) if (labels==c).any()]))
    return acc,balanced


def evaluate(weights,x,y):
    all_acc=[];all_balanced=[];all_recall=[];classes=weights.shape[-1]
    for start in range(0,len(weights),128):
        pred=torch.matmul(x,weights[start:start+128]).argmax(-1)
        correct=pred==y[None]
        all_acc.extend(correct.float().mean(-1).cpu().tolist())
        recall=torch.stack([correct[:,y==c].float().mean(-1) for c in range(classes) if (y==c).any()],dim=1)
        all_recall.extend(recall.cpu().tolist())
        balanced=recall.mean(1)
        all_balanced.extend(balanced.cpu().tolist())
    evaluate.last_recall=np.mean(all_recall,axis=0).tolist()
    return float(np.mean(all_acc)),float(np.mean(all_balanced))


def release_labels(weights,data,method,k,epsilon,seed,round_number,mode='raw',pate=None,colluders=0):
    n,d,c=weights.shape;proxy_x=torch.as_tensor(data['x_proxy'],device=DEVICE)
    rows=len(proxy_x)
    noise_seed=seed*1000003+round_number*101
    if method in ['SDD','random','representative','weak','random-representative']:
        kind={'SDD':'semantic','random':'random','representative':'semantic','weak':'weak','random-representative':'random'}[method]
        ids,reps,_=partition(data,k,kind)
        counts=vote_counts(weights,proxy_x,ids,reps if method in ['representative','random-representative'] else None)
        cert=privacy.certificate(epsilon,1e-5,3,n,clusters=k,dimension=k*c,colluders=colluders)
        scores=noisy_votes(counts,cert,noise_seed)
        label=labels_from_scores(scores,mode,torch.bincount(torch.as_tensor(ids,device=DEVICE),minlength=k).float())
        targets=label[torch.as_tensor(ids,device=DEVICE)]
        clean=labels_from_scores(counts.float()/n,mode,torch.bincount(torch.as_tensor(ids,device=DEVICE),minlength=k).float())
        purity=sum(np.bincount(data['y_proxy'][ids==i],minlength=c).max() for i in range(k))/rows
        margins=torch.topk(counts.float()/n,2,dim=-1).values
        diagnostics={'public_cluster_purity':float(purity),'clean_pseudolabel_accuracy':float((clean.cpu().numpy()[ids]==data['y_proxy']).mean()),
                     'cluster_flip_fraction':float((label!=clean).float().mean()),'median_raw_vote_margin':float((margins[:,0]-margins[:,1]).median()),
                     'logical_report_coordinates':k*c}
        certificate=cert
    elif method=='DP-FedCT':
        counts=vote_counts(weights,proxy_x)
        cert=privacy.certificate(epsilon,1e-5,3,n,clusters=rows,dimension=rows*c,colluders=colluders)
        scores=noisy_votes(counts,cert,noise_seed)
        targets=labels_from_scores(scores,mode)
        diagnostics={'logical_report_coordinates':rows*c}
        certificate=cert
    elif method=='PATE-DI':
        counts=vote_counts(weights,proxy_x)
        threshold=pate['threshold'];cap=min(rows,int(rows*pate['cap_fraction']));fraction=pate['screen_rho_fraction']
        screen=privacy.certificate(epsilon,1e-5,3,n,integer_sensitivity_squared=rows*(1<<16)**2,dimension=rows,
                                  rho_fraction=fraction,accounting='rdp',colluders=colluders)
        max_counts=counts.max(-1).values
        noisy_max=noisy_votes(max_counts[:,None],screen,noise_seed).flatten()
        selected=torch.where(noisy_max>=threshold)[0]
        selected=selected[torch.argsort(noisy_max[selected],descending=True)[:cap]]
        answer=privacy.certificate(epsilon,1e-5,3,n,clusters=cap,dimension=cap*c,
                                  rho_fraction=1-fraction,accounting='rdp',colluders=colluders)
        targets=torch.full((rows,),-1,device=DEVICE,dtype=torch.long)
        if len(selected):
            scores=noisy_votes(counts[selected],answer,noise_seed+37)
            targets[selected]=labels_from_scores(scores,mode)
        rho_total=screen['rho_total_upper']+answer['rho_total_upper']
        bound=privacy.epsilon_from_rho(rho_total,1e-5,'rdp');assert bound<=epsilon
        certificate={'screen':screen,'answers':answer,'rho_total_upper':rho_total,'epsilon_upper':bound,
                     'view':'noisy confidence scores and selected noisy counts; ideal protected screening'}
        diagnostics={'answered_queries':len(selected),'max_answered_queries':cap,'threshold':threshold,
                     'logical_answer_coordinates_max':cap*c,'logical_screen_coordinates':rows}
    else:raise ValueError(method)
    valid=targets>=0
    diagnostics['pseudolabel_coverage']=float(valid.float().mean())
    if valid.any():
        a,b=score_labels(targets[valid].cpu().numpy(),data['y_proxy'][valid.cpu().numpy()],c)
        diagnostics.update(pseudolabel_accuracy=a,pseudolabel_balanced_accuracy=b)
    return targets,certificate,diagnostics


def parameter_release(weights,reference,clip,epsilon,seed,round_number):
    n,d,c=weights.shape
    updates=weights-reference
    norms=updates.flatten(1).norm(dim=1)
    factor=(clip/norms.clamp_min(1e-12)).clamp_max(1)
    updates=updates*factor[:,None,None]
    scale=1<<16
    radius=math.ceil(clip*scale+math.sqrt(d*c)/2)
    quantized_clients=torch.round(updates.double()*scale).long()
    # In these configurations squared integer norms are <2**53, so this
    # double reduction is exact. The guard also handles non-finite local fits.
    assert radius**2<2**53
    within_ball=quantized_clients.double().square().sum((1,2))<=radius**2
    quantized_clients[~within_ball]=0
    cert=privacy.certificate(epsilon,1e-5,3,n,integer_sensitivity=2*radius,dimension=d*c)
    cert['integer_report_radius']=radius
    quantized=quantized_clients.sum(0).cpu().numpy()
    noise=privacy.noise_sum(cert,d*c,seed*1000003+round_number*101).reshape(d,c)
    released=reference+torch.as_tensor((quantized+noise).astype(np.float64)/(n*cert['quantization_scale']),device=DEVICE,dtype=torch.float32)
    return released,cert,{'clip':clip,'integer_report_radius':radius,
        'integer_ball_guard_fraction':float((~within_ball).float().mean()),
        'clipped_fraction':float((norms>clip).float().mean()),'logical_report_coordinates':d*c}


def run(dataset,n,alpha,seed,method,k,epsilon,mode='raw',clip=1.,pate=None,eval_split='test',public=False,colluders=0):
    start=time.time();data=load_data(dataset,public)
    weights,velocity,idx,sizes,population=initial_state(data,n,alpha,seed,public)
    weights=weights.clone();velocity=velocity.clone()
    x=torch.as_tensor(data['x_private'],device=DEVICE);y=torch.as_tensor(data['y_private'],device=DEVICE)
    proxy_x=torch.as_tensor(data['x_proxy'],device=DEVICE)
    x_eval=torch.as_tensor(data[f'x_{eval_split}'],device=DEVICE)
    y_eval=torch.as_tensor(data[f'y_{eval_split}'],device=DEVICE)
    index=torch.as_tensor(idx,device=DEVICE);sizes_t=torch.as_tensor(sizes,device=DEVICE)
    trace=[];targets=None;reference=torch.zeros_like(weights[0])
    initial_accuracy,initial_balanced=evaluate(weights,x_eval,y_eval)
    initial_recall=evaluate.last_recall.copy();local_endpoint=None
    for r in range(1,4):
        if r>1:
            weights,velocity=train_round(weights,velocity,x,y,index,sizes_t,seed,r,proxy_x,targets)
        if method=='DP-Head':
            if r==3:
                local_acc,local_bal=evaluate(weights,x_eval,y_eval)
                local_endpoint={'accuracy':local_acc,'balanced_accuracy':local_bal,'recall':evaluate.last_recall.copy(),
                    'optimizer_steps':180,'extra_adaptation_steps':0,'endpoint':'before third parameter aggregation; previous two broadcasts retained'}
            reference,cert,diagnostics=parameter_release(weights,reference,clip,epsilon,seed,r)
            weights=reference[None].expand(n,-1,-1).clone();velocity.zero_()
            targets=torch.matmul(proxy_x,reference).argmax(-1)
        else:
            targets,cert,diagnostics=release_labels(weights,data,method,k,epsilon,seed,r,mode,pate,colluders)
        trace.append({'round':r,'privacy':cert,'diagnostics':diagnostics})
    accuracy,balanced=evaluate(weights,x_eval,y_eval)
    result=dict(dataset=dataset,actual_clients=n,calibrated_clients=n,alpha=alpha,seed=seed,method=method,k=k,
                epsilon=epsilon,delta=1e-5,mode=mode,clip=clip if method=='DP-Head' else None,
                public_development=public,evaluation_split=eval_split,accuracy=accuracy,balanced_accuracy=balanced,
                initial_accuracy=initial_accuracy,initial_balanced_accuracy=initial_balanced,
                population=population,rounds=3,optimizer_steps_per_round=STEPS,batch_size=BATCH,
                lr=LR,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY,
                private_model='128-dimensional frozen public features; independently trained no-intercept softmax heads',
                trace=trace,seconds=time.time()-start,diagnostics_are_offline_benchmark_measurements=True,
                per_class_recall=evaluate.last_recall.copy(),initial_per_class_recall=initial_recall,local_endpoint=local_endpoint)
    print(json.dumps({k:v for k,v in result.items() if k not in ['trace','population']}),flush=True)
    return result
