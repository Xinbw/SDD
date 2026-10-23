"""Inductive public baselines and budget-matched DP/public postprocessing.

Only queried development labels enter fitting, leave-one-out selection or gates.
Final-test labels are first loaded by the evaluator, after each rule is fixed.
"""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
import engine_base as base
from run_private import project,coverage,decode,metrics
from common import ROOT,CLASSES,SETTINGS,SEEDS,freeze,frozen_design,release_key,save,sha

DEVICE='cuda'
torch.backends.cuda.matmul.allow_tf32=False
WEIGHTS=[0.,.1,.3,.5,.7,.9,1.]
LAMBDAS=[.001,.01,.1,1.,10.]

def normalize(a):
    if torch.is_tensor(a):
        a=a.clamp_min(0);s=a.sum(-1,keepdim=True)
        return torch.where(s>1e-20,a/s.clamp_min(1e-20),torch.full_like(a,1/a.shape[-1]))
    a=np.maximum(a,0);s=a.sum(-1,keepdims=True)
    return np.divide(a,s,out=np.full_like(a,1/a.shape[-1]),where=s>1e-20)

def observed_score(pred,y,c,balanced):
    if balanced:return float(np.mean([(pred[y==j]==j).mean() for j in range(c) if (y==j).any()]))
    return float(np.mean(pred==y))

def samples(dataset):
    file=ROOT/'public'/f'{dataset}-label-samples.json'
    if file.exists():return json.loads(file.read_text())
    with np.load(ROOT/'inputs'/f'{dataset}.npz') as b:y=b['y_development']
    c=CLASSES[dataset];rows=[];task_id=list(CLASSES).index(dataset)
    for scheme,draws in [('random',100)]+([('stratified',50)] if dataset in ['cifar10','mnist'] else []):
        for draw in range(draws):
            rng=np.random.default_rng(970001+task_id*10000+draw+(100000 if scheme=='stratified' else 0))
            if scheme=='random':permutation=rng.permutation(len(y))
            else:pools=[rng.permutation(np.where(y==j)[0]) for j in range(c)]
            for q in [1,3,5,10,20,50]:
                ids=permutation[:c*q] if scheme=='random' else np.concatenate([p[:q] for p in pools])
                assert len(ids)==c*q and len(np.unique(ids))==len(ids)
                labels=y[ids];counts=np.bincount(labels,minlength=c)
                rows.append(dict(scheme=scheme,draw=draw,q=q,m=len(ids),indices=ids.tolist(),labels=labels.tolist(),
                    missing_classes=int(np.sum(counts==0)),sample_sha256=hashlib.sha256(ids.tobytes()+labels.tobytes()).hexdigest()))
    save(file,rows);return rows

def public_features(dataset):
    with np.load(ROOT/'inputs'/f'{dataset}.npz') as b:
        return {k:b[k] for k in ['x_proxy','x_development','x_test']}

def graph_operator(dataset,data,alpha):
    """Fixed unlabeled graph; return propagation coefficients from development seeds."""
    cache=ROOT/'checkpoints'/f'graph_{dataset}_{alpha}.pt'
    if cache.exists():return torch.load(cache,map_location=DEVICE,weights_only=False)
    xd,xp,xt=data['x_development'],data['x_proxy'],data['x_test']
    nodes=np.concatenate([xd,xp]);nd=len(xd);n=len(nodes)
    nn=NearestNeighbors(n_neighbors=21,n_jobs=8).fit(nodes)
    distance,index=nn.kneighbors(nodes);distance,index=distance[:,1:],index[:,1:]
    bandwidth=max(float(np.median(distance[:,-1]**2)),1e-8)
    values=np.exp(-distance**2/bandwidth).astype(np.float32)
    w=torch.zeros((n,n),device=DEVICE)
    w[torch.arange(n,device=DEVICE)[:,None],torch.as_tensor(index,device=DEVICE)]=torch.as_tensor(values,device=DEVICE)
    w=(w+w.T)*.5;degree=w.sum(1).clamp_min(1e-12).rsqrt();s=w*degree[:,None]*degree[None]
    matrix=torch.eye(n,device=DEVICE)-alpha*s
    right=torch.eye(n,device=DEVICE)[:,:nd]
    factor=torch.linalg.cholesky(matrix)
    h=(1-alpha)*torch.cholesky_solve(right,factor)
    residual=float((matrix@h-(1-alpha)*right).abs().max())
    assert residual<2e-5,residual
    distance,index=nn.kneighbors(xt,n_neighbors=20)
    query=np.exp(-distance**2/bandwidth);query/=np.maximum(query.sum(1,keepdims=True),1e-30)
    test=torch.zeros((len(xt),nd),device=DEVICE)
    for start in range(0,len(xt),64):
        idx=torch.as_tensor(index[start:start+64],device=DEVICE)
        q=torch.as_tensor(query[start:start+64],device=DEVICE,dtype=torch.float32)
        test[start:start+64]=(h[idx]*q[...,None]).sum(1)
    out=dict(development=h[:nd],proxy=h[nd:],test=test,bandwidth=bandwidth,
        numerical_residual=residual,alpha=alpha,training_nodes=n,test_nodes_in_training_graph=0)
    torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in out.items()},cache)
    save(ROOT/'public'/f'graph-{dataset}-{alpha}.json',dict(alpha=alpha,neighbors=20,training_nodes=n,
        public_development_feature_nodes=nd,proxy_nodes=len(xp),test_nodes_in_training_graph=0,
        bandwidth=bandwidth,linear_system_max_residual=residual,labels_used_in_graph_construction=0))
    return out

def name_regions(counts,centres,weights,constrained=False):
    nonempty=np.where(counts.sum(1)>0)[0]
    if not len(nonempty):return np.zeros(len(counts),dtype=np.int64)
    nearest=((centres[:,None]-centres[nonempty][None])**2).sum(2).argmin(1)
    p=normalize(counts[nonempty][nearest].astype(np.float64))
    return coverage(p,weights) if constrained else p.argmax(1)

def region_public(data,idx,y,c,assignment,centres,weights):
    region=assignment[idx];counts=np.zeros((c,c));np.add.at(counts,(region,y),1)
    pairs,inverse=np.unique(np.stack([region,y],axis=1),axis=0,return_inverse=True)
    candidates=[];loo_counts=[]
    for r,label in pairs:
        a=counts.copy();a[r,label]-=1;loo_counts.append(a)
    for constrained in [False,True]:
        labels=name_regions(counts,centres,weights,constrained)
        by_pair=np.stack([name_regions(a,centres,weights,constrained) for a in loo_counts])
        cv=by_pair[inverse,region]
        candidates.append(dict(name='cluster-coverage' if constrained else 'cluster-majority',
            loo=np.eye(c)[cv],test=np.eye(c)[labels[assignment[len(data['x_development']):]]],
            proxy=None,regional_labels=labels))
    return candidates,counts,np.array(loo_counts),pairs,inverse

def prototype(x,y,queries,c,bandwidth):
    counts=torch.bincount(y,minlength=c);sums=torch.zeros((c,x.shape[1]),device=DEVICE)
    sums.index_add_(0,y,x);means=sums/counts[:,None].clamp_min(1)
    scores=-torch.cdist(queries,means).square()/bandwidth
    scores[:,counts==0]=-1e6
    # Each row removes its own feature and label before forming prototypes.
    loo=means[None].expand(len(x),-1,-1).clone()
    own=(sums[y]-x)/(counts[y]-1).clamp_min(1)[:,None]
    loo[torch.arange(len(x),device=DEVICE),y]=own
    ls=-(x[:,None]-loo).square().sum(2)/bandwidth
    ls[:,counts==0]=-1e6;ls[torch.arange(len(x),device=DEVICE)[counts[y]==1],y[counts[y]==1]]=-1e6
    return ls.softmax(1),scores.softmax(1)

def public_candidates(data,sample,graphs,assignment,centres,weights):
    idx=np.array(sample['indices']);y=np.array(sample['labels']);c=CLASSES[data['dataset']]
    x=torch.as_tensor(data['x_development'][idx],device=DEVICE);ty=torch.as_tensor(y,device=DEVICE)
    queries=torch.as_tensor(np.concatenate([data['x_test'],data['x_proxy']]),device=DEVICE)
    candidates,counts,loo_counts,pairs,inverse=region_public(data,idx,y,c,assignment,centres,weights)
    for candidate in candidates:
        labels=candidate['regional_labels'];proxy_ids=data['proxy_assignment']
        candidate['proxy']=np.eye(c)[labels[proxy_ids]]
    bandwidth=graphs[.95]['bandwidth']
    loo,scores=prototype(x,ty,queries,c,bandwidth)
    candidates.append(dict(name='prototype',loo=loo.cpu().numpy(),test=scores[:len(data['x_test'])].cpu().numpy(),proxy=scores[len(data['x_test']):].cpu().numpy()))
    target=torch.nn.functional.one_hot(ty,c).float();gram=x.T@x
    eig,u=torch.linalg.eigh(gram);eig=eig.clamp_min(0);rot=x@u;rhs=rot.T@target
    for lam in LAMBDAS:
        coef=(u/(eig+lam)[None])@rhs
        fitted=x@coef;leverage=(rot.square()/(eig+lam)[None]).sum(1).clamp(max=1-1e-6)
        # Exact ridge leave-one-out score, up to a positive row-wise factor.
        ls=fitted-leverage[:,None]*target
        ps=queries@coef
        ls=ls/(1-leverage[:,None]).clamp_min(1e-6)
        candidates.append(dict(name=f'ridge-{lam:g}',loo=ls.softmax(1).cpu().numpy(),
            test=ps[:len(data['x_test'])].softmax(1).cpu().numpy(),proxy=ps[len(data['x_test']):].softmax(1).cpu().numpy()))
    counts_class=np.bincount(y,minlength=c)
    for alpha,g in graphs.items():
        hid=torch.as_tensor(idx,device=DEVICE)
        # H depends only on public features. Subtracting H_ii removes the held-out label exactly.
        full={split:g[split][:,hid]@target for split in ['development','test','proxy']}
        ls=full['development'][hid].clone();ls[torch.arange(len(idx),device=DEVICE),ty]-=g['development'][hid,hid]
        for mass in [False,True]:
            if mass:
                divisor=np.maximum(counts_class,1).astype(np.float32)
                loo_div=np.broadcast_to(divisor,(len(idx),c)).copy();loo_div[np.arange(len(idx)),y]=np.maximum(counts_class[y]-1,1)
                pl=ls/torch.as_tensor(loo_div,device=DEVICE);pt=full['test']/torch.as_tensor(divisor,device=DEVICE)
                pp=full['proxy']/torch.as_tensor(divisor,device=DEVICE)
            else:pl,pt,pp=ls,full['test'],full['proxy']
            candidates.append(dict(name=f'graph-{alpha:g}-'+('classmass' if mass else 'ordinary'),
                loo=normalize(pl).cpu().numpy(),test=normalize(pt).cpu().numpy(),proxy=normalize(pp).cpu().numpy()))
    balanced=data['dataset']=='web'
    for a in candidates:a['loo_score']=observed_score(a['loo'].argmax(1),y,c,balanced)
    chosen=max(range(len(candidates)),key=lambda j:(candidates[j]['loo_score'],-j))
    return candidates,chosen,(counts,loo_counts,pairs,inverse)

def evidence(counts,centres):
    nonempty=np.where(counts.sum(1)>0)[0]
    if not len(nonempty):return np.full(counts.shape,1/counts.shape[1])
    nearest=((centres[:,None]-centres[nonempty][None])**2).sum(2).argmin(1)
    return normalize(counts[nonempty][nearest].astype(float))

def hybrid(public,sample,private,data,assignment,centres,weights,extra):
    c=CLASSES[data['dataset']];y=np.array(sample['labels']);idx=np.array(sample['indices'])
    rid=assignment[idx];test_ids=assignment[len(data['x_development']):]
    counts,loo_counts,pairs,inverse=extra
    e=evidence(counts,centres);elo=np.stack([evidence(a,centres) for a in loo_counts])
    a=project(private);options=[dict(name='public',loo=public['loo'],kind='public',weight=0.,mode='public')]
    for mode in ['center','coverage','raw']:
        labels=decode(private,weights,mode)
        for fraction in WEIGHTS[1:]:
            loo=(1-fraction)*public['loo']+fraction*np.eye(c)[labels[rid]]
            options.append(dict(name=f'graphmix-{mode}-{fraction:g}',loo=loo,kind='mixture',weight=fraction,mode=mode,labels=labels))
        for fraction in WEIGHTS:
            labels=decode((1-fraction)*e+fraction*a,weights,mode)
            pl=np.stack([decode((1-fraction)*ee+fraction*a,weights,mode) for ee in elo])
            loo=np.eye(c)[pl[inverse,rid]]
            options.append(dict(name=f'regional-{mode}-{fraction:g}',loo=loo,kind='regional',weight=fraction,mode=mode,labels=labels))
    for opt in options:opt['score']=observed_score(opt['loo'].argmax(1),y,c,data['dataset']=='web')
    best=max(range(len(options)),key=lambda j:(options[j]['score'],-j));opt=options[best]
    if opt['kind']=='public':test=public['test'];proxy=public['proxy']
    elif opt['kind']=='mixture':
        f=opt['weight'];labels=opt['labels']
        test=(1-f)*public['test']+f*np.eye(c)[labels[test_ids]]
        proxy=(1-f)*public['proxy']+f*np.eye(c)[labels[data['proxy_assignment']]]
    else:
        test=np.eye(c)[opt['labels'][test_ids]];proxy=np.eye(c)[opt['labels'][data['proxy_assignment']]]
    pcorrect=(public['loo'].argmax(1)==y);hcorrect=(opt['loo'].argmax(1)==y)
    difference=hcorrect.astype(float)-pcorrect
    if data['dataset']=='web':
        cc=np.bincount(y,minlength=c);observed=max(1,np.sum(cc>0));difference*=len(y)/(observed*cc[y])
    gain=float(difference.mean());lower=gain-1.96*float(difference.std(ddof=1))/np.sqrt(len(y))
    recall_lower=[]
    for j in range(c):
        n=int(np.sum(y==j));k=int(np.sum(hcorrect[y==j]))
        if n==0:recall_lower.append(0.);continue
        p=k/n;z=1.96
        recall_lower.append(float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)))
    covered=len(np.unique(proxy.argmax(1)))==c
    accepted=bool(lower>0 and covered and min(recall_lower)>1e-12 and opt['kind']!='public')
    return test,dict(selected=opt['name'],loo_score=opt['score'],public_loo_score=public['loo_score'],
        paired_loo_gain=gain,paired_loo_gain_lower=lower,minimum_loo_wilson_lower=min(recall_lower),
        all_classes_in_proxy_predictions=covered,private_augmentation_published=accepted,
        gate_scope='predeclared LOO heuristic; selection and correlated LOO errors do not provide a risk certificate',
        candidate_loo_scores={o['name']:o['score'] for o in options})

def main(dataset=None):
    freeze('DESIGN-FROZEN.json',frozen_design())
    freeze('PUBLIC-EXECUTION-FROZEN.json',dict(public_tasks=list(CLASSES),samples=100,stratified=50,
        public_label_budget='m unique development labels; no validation labels',test_labels_only_in_evaluator=True))
    for task in ([dataset] if dataset else CLASSES):
        data=public_features(task);data['dataset']=task;c=CLASSES[task]
        proxy_ids,reps,centres=base.partition(data,c);weights=np.bincount(proxy_ids,minlength=c)
        all_points=np.concatenate([data['x_development'],data['x_test']])
        assignment=torch.cdist(torch.as_tensor(all_points,device=DEVICE),torch.as_tensor(centres,device=DEVICE)).argmin(1).cpu().numpy()
        data['proxy_assignment']=proxy_ids
        graphs={a:graph_operator(task,data,a) for a in [.8,.95]}
        rows=samples(task)
        for number,sample in enumerate(rows):
            name=f"{task}_{sample['scheme']}_{sample['draw']}_m{sample['m']}"
            output=ROOT/'public'/(name+'.json')
            if output.exists():continue
            candidates,choice,extra=public_candidates(data,sample,graphs,assignment,centres,weights)
            chosen=candidates[choice];out=dict(dataset=task,**sample,selected_public=chosen['name'],
                label_budget=sample['m'],external_validation_labels=0,baselines={},hybrid=[])
            # Test labels enter only evaluation; no subsequent choice uses any test quantity.
            with np.load(ROOT/'inputs'/f'{task}.npz') as b:truth=b['y_test']
            for cand in candidates:out['baselines'][cand['name']]=dict(**metrics(cand['test'].argmax(1),truth,c),loo_score=cand['loo_score'])
            out['public_portfolio']=out['baselines'][chosen['name']]
            for d,n,epsilon in SETTINGS:
                if d!=task:continue
                for seed in SEEDS:
                    key=release_key(task,n,epsilon,seed,180,'regional',c,'M')
                    source=ROOT/'results'/(key+'.json')
                    if not source.exists():raise RuntimeError('Complete private stage first: '+key)
                    result=json.loads(source.read_text());diag=ROOT/result['diagnostic_file']
                    private=np.load(diag)['scores']
                    prediction,gate=hybrid(chosen,sample,private,data,assignment,centres,weights,extra)
                    ungated=metrics(prediction.argmax(1),truth,c)
                    gated=ungated if gate['private_augmentation_published'] else out['public_portfolio']
                    out['hybrid'].append(dict(n=n,epsilon=epsilon,seed=seed,protected_source=key,source_sha256=sha(source),
                        private_only={mode:result['endpoints'][mode] for mode in ['center','coverage','raw']},
                        ungated=ungated,gated=gated,gate=gate))
            save(output,out)
            if number%10==0:print(name,'public',round(out['public_portfolio']['accuracy']*100,2),chosen['name'],flush=True)
            save(ROOT/'PUBLIC-STATUS.json',dict(dataset=task,completed=number+1,total=len(rows)))
        save(ROOT/f'PUBLIC-{task}-COMPLETE.json',dict(label_sets=len(rows),utc=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CLASSES));main(p.parse_args().dataset)
