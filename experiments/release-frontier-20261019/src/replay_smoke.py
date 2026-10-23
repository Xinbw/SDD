"""Fresh actual-cohort and public-pipeline reconstruction, isolated from evidence."""
import json
from pathlib import Path
import numpy as np
import torch
import engine_base as base
import privacy
from run_private import metrics,decode
from common import ROOT,CLASSES,release_key,save,sha

def main():
    project=ROOT.parents[1];output=project/'reproduction-output/smoke-frontier';output.mkdir(parents=True,exist_ok=True)
    old_root=base.ROOT;base.ROOT=output
    for folder in ['checkpoints','inputs/partitions']:(output/folder).mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'inputs').glob('*.npz'):
        link=output/'inputs'/source.name
        if not link.exists():link.symlink_to(source.resolve())
    for source in (ROOT/'inputs/partitions').glob('*.npz'):
        link=output/'inputs/partitions'/source.name
        if not link.exists():link.symlink_to(source.resolve())
    checks=[]
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        for d,n,e,seed,steps,k,stage in [('cifar10',1000,.1,8501,60,10,ROOT),('femnist62',2000,1.,8501,180,124,ROOT),('cifar10',1000,.1,8801,180,10,ROOT/'confirmation')]:
            data=base.load_data(d);w,v,idx,sizes,pop=base.initial_state(data,n,None if d=='femnist62' else .1,seed)
            if steps==180:
                x,y,index,sz=[torch.as_tensor(z,device='cuda') for z in [data['x_private'],data['y_private'],idx,sizes]]
                for r in [2,3]:w,v=base.train_round(w,v,x,y,index,sz,seed,r)
            ids,reps,centres=base.partition(data,k);counts=base.vote_counts(w,torch.as_tensor(data['x_proxy'],device='cuda'),ids)
            key=release_key(d,n,e,seed,steps,'regional',k,'M');row=json.loads((stage/'results'/(key+'.json')).read_text())
            expected=np.load(stage/row['diagnostic_file']);assert np.array_equal(counts.cpu().numpy(),expected['counts'])
            cert=privacy.certificate(e,1e-5,1,n,clusters=k,dimension=k*CLASSES[d])
            scores=base.noisy_votes(counts,cert,seed*1000003+steps*101+20261019).cpu().numpy()
            assert np.array_equal(scores,expected['scores'])
            checks.append(dict(source=key,cohort=pop,clean_votes_equal=True,exact_discrete_release_equal=True))
    finally:base.ROOT=old_root
    import public_frontier as public
    torch.backends.cuda.matmul.allow_tf32=False
    for task,n,e,m in [('cifar10',1000,.1,10),('femnist62',2000,1.,62)]:
        source=ROOT/'public'/f'{task}_random_0_m{m}.json';record=json.loads(source.read_text())
        data=public.public_features(task);data['dataset']=task;c=CLASSES[task]
        ids,rep,centres=base.partition(data,c);data['proxy_assignment']=ids;weights=np.bincount(ids,minlength=c)
        allx=np.concatenate([data['x_development'],data['x_test']]);assignment=torch.cdist(torch.as_tensor(allx,device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
        # Graph caches go to isolated output, never to the protected record tree.
        original_public_root=public.ROOT;public.ROOT=output
        try:graphs={a:public.graph_operator(task,data,a) for a in [.8,.95]}
        finally:public.ROOT=original_public_root
        candidates,choice,extra=public.public_candidates(data,record,graphs,assignment,centres,weights)
        assert candidates[choice]['name']==record['selected_public']
        with np.load(ROOT/'inputs'/f'{task}.npz') as b:truth=b['y_test']
        for a in candidates:
            actual=metrics(a['test'].argmax(1),truth,c)
            for metric in ['accuracy','balanced_accuracy','per_class_recall']:assert actual[metric]==record['baselines'][a['name']][metric],(task,a['name'],metric)
        case=next(x for x in record['hybrid'] if x['n']==n and x['epsilon']==e and x['seed']==8501)
        row=json.loads((ROOT/'results'/(case['protected_source']+'.json')).read_text());private=np.load(ROOT/row['diagnostic_file'])['scores']
        scores,gate=public.hybrid(candidates[choice],record,private,data,assignment,centres,weights,extra)
        actual=metrics(scores.argmax(1),truth,c)
        assert actual==case['ungated'] and gate==case['gate'],task
        checks.append(dict(public_label_source=source.name,all_twelve_baselines_equal=True,selected_fusion_and_gate_equal=True))
    # Independent primary case: fourteen public learners, joint LOO, gates,
    # and identically decoded head/point views, reconstructed from features.
    from anchored_fusion import count_loo,uniform_loo,LAMBDAS,score_correct,gate
    from run_private import project
    from scipy.special import softmax
    task='cifar10';c=10;m=10;n=1000;e=.1;seed=8801
    record=json.loads((ROOT/'analysis/joint-confirmation/cifar10_random_0_m10.json').read_text())
    data=public.public_features(task);data['dataset']=task
    ids,rep,centres=base.partition(data,c);weights=np.bincount(ids,minlength=c);data['proxy_assignment']=ids
    allx=np.concatenate([data['x_development'],data['x_test']]);assignment=torch.cdist(torch.as_tensor(allx,device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
    dev,test=assignment[:len(data['x_development'])],assignment[len(data['x_development']):]
    oldroot=public.ROOT;public.ROOT=output
    try:graphs={a:public.graph_operator(task,data,a) for a in [.8,.95]}
    finally:public.ROOT=oldroot
    candidates,_,_=public.public_candidates(data,record,graphs,assignment,centres,weights)
    with np.load(ROOT/'inputs/cifar10.npz') as b:truth=b['y_test']
    y=np.array(record['labels']);region=dev[record['indices']];counts=np.zeros((c,c));np.add.at(counts,(region,y),1)
    models=[]
    for candidate in candidates:
        correct=candidate['loo'].argmax(1)==y;value=dict(**metrics(candidate['test'].argmax(1),truth,c),loo_score=score_correct(correct,y,c,False))
        assert value==record['baselines'][candidate['name']]
        models.append(dict(name=candidate['name'],correct=correct,score=value['loo_score'],metric=value,coverage=len(np.unique(candidate['proxy'].argmax(1)))==c,private=False))
    for name,fn in [('count-matching',lambda:count_loo(counts,region,y)),('uniform-empty-matching',lambda:uniform_loo(counts,weights,region,y))]:
        label,correct=fn();value=dict(**metrics(label[test],truth,c),loo_score=score_correct(correct,y,c,False));assert value==record['baselines'][name]
        models.append(dict(name=name,correct=correct,score=value['loo_score'],metric=value,coverage=True,private=False))
    publicbest=max(range(len(models)),key=lambda j:(models[j]['score'],-j));best=models[publicbest]
    assert best['name']==record['selected_public']
    source=next(h for h in record['hybrid'] if h['n']==n and h['seed']==seed)
    protected=json.loads((ROOT/'confirmation/results'/(source['protected_source']+'.json')).read_text());released=np.load(ROOT/'confirmation'/protected['diagnostic_file'])['scores']
    def choices(released,native=None):
        options=list(models);a=project(released)*weights[:,None]/weights.mean()
        for lam in LAMBDAS:
            label,correct=count_loo(counts+lam*a,region,y);options.append(dict(name=f'joint-{lam:g}',correct=correct,score=score_correct(correct,y,c,False),labels=label,private=True,coverage=True))
        if native:
            correct=native[0][record['indices']]==y;options.append(dict(name='native-head',score=score_correct(correct,y,c,False),metric=native[1],private=False))
        for mode in ['center','coverage']:
            label=decode(released,weights,mode);correct=label[region]==y
            options.append(dict(name=('private-' if native is None else 'private-region-')+mode,labels=label,correct=correct,score=score_correct(correct,y,c,False),private=True,coverage=len(np.unique(label))==c))
        index=max(range(len(options)),key=lambda j:(options[j]['score'],-j));chosen=options[index]
        value=metrics(chosen['labels'][test],truth,c) if chosen['private'] else chosen['metric']
        return chosen,value,options
    chosen,value,options=choices(released);assert value==source['ungated'] and chosen['name']==source['gate']['selected']
    g=gate(chosen['correct'],best['correct'],y,c,chosen['coverage'],chosen['private'])
    for key,value in g.items():assert value==source['gate'][key]
    expected=json.loads((ROOT/'analysis/joint-controls-confirmation/cifar10_random_0_m10.json').read_text())
    for method in ['head-fixed','per-example']:
        key=release_key(task,n,e,seed,180,method,0,'-');r=json.loads((ROOT/'confirmation/results'/(key+'.json')).read_text());a=np.load(ROOT/'confirmation'/r['diagnostic_file'])['scores']
        if method=='head-fixed':
            posterior=softmax(data['x_proxy']@a,axis=1);regional=np.stack([posterior[ids==k].mean(0) for k in range(c)]);native=((data['x_development']@a).argmax(1),r['endpoints']['broadcast'])
        else:regional=np.stack([a[ids==k].mean(0) for k in range(c)]);native=None
        chosen,value,_=choices(regional,native);h=next(h for h in expected['controls'] if h['n']==n and h['seed']==seed and h['method']==method)
        assert value==h['metric']
    checks.append(dict(independent_joint_confirmation=True,all_fourteen_public_learners_equal=True,exact_joint_selection_and_gate_equal=True,head_and_pooled_point_controls_equal=True))
    # Exact correctness comparison against unrestricted full refits, including ties.
    from anchored_fusion import match
    rng=np.random.default_rng(20261019)
    for c in [3,6,10,62]:
        for m in [10,100]:
            region=rng.integers(c,size=m);y=rng.integers(c,size=m)
            for tied in [False,True]:
                score=rng.integers(0,4,size=(c,c)).astype(float) if tied else rng.normal(size=(c,c))
                _,fast=count_loo(score,region,y);slow=[]
                for k,label in zip(region,y):
                    a=score.copy();a[k,label]-=1;slow.append(match(a)[k]==label)
                assert np.array_equal(fast,slow)
    checks.append(dict(matched_edge_loo_equal_to_full_refit=True,classes=[3,6,10,62],tied_matrices_included=True))
    save(output/'checks.json',checks);print('PASS: three reconstructed cohorts and exact releases; legacy and independent fourteen-model joint pipelines; matched head/point controls; exact LOO full-refit equivalence.')

if __name__=='__main__':main()
