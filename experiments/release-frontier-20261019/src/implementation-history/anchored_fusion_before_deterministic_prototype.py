"""Joint naming from observed public counts and one protected vote matrix.

Exact LOO accuracy needs re-solving only matched observed edges. A removed
label on an unmatched edge cannot become correct when that edge decreases.
All original policies and records remain immutable comparisons.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import engine_base as base
import public_frontier as public
from run_private import project,decode,metrics
from common import ROOT,PARENT,CLASSES,SETTINGS,SEEDS,freeze,save,sha,release_key

LAMBDAS=[.01,.03,.1,.3,1.,3.,10.,30.]

def match(score):
    rows,classes=linear_sum_assignment(-score);labels=np.zeros(len(score),dtype=np.int64);labels[rows]=classes
    return labels

def count_loo(score,region,y):
    labels=match(score);correct=np.zeros(len(y),dtype=bool)
    # Exact: if the removed edge is unused, its decrease cannot make it win.
    for r in range(len(labels)):
        c=labels[r];mask=(region==r)&(y==c)
        if not mask.any():continue
        altered=score.copy();altered[r,c]-=1
        correct[mask]=(match(altered)[r]==c)
    return labels,correct

def uniform_loo(counts,weights,region,y):
    def matrix(a):
        denominator=a.sum(1,keepdims=True)
        return np.divide(a,denominator,out=np.full_like(a,1/a.shape[1]),where=denominator>0)*weights[:,None]
    labels=match(matrix(counts));correct=np.zeros(len(y),dtype=bool)
    for r in range(len(labels)):
        c=labels[r];mask=(region==r)&(y==c)
        if not mask.any():continue
        a=counts.copy();a[r,c]-=1
        correct[mask]=(match(matrix(a))[r]==c)
    return labels,correct

def score_correct(correct,y,c,balanced):
    return float(np.mean([correct[y==j].mean() for j in range(c) if (y==j).any()])) if balanced else float(correct.mean())

def gate(correct,public_correct,y,c,coverage,private_branch):
    delta=correct.astype(float)-public_correct.astype(float)
    if c==6:
        count=np.bincount(y,minlength=c);observed=max(1,np.sum(count>0));delta*=len(y)/(observed*count[y])
    gain=float(delta.mean());lower=gain-1.96*float(delta.std(ddof=1))/np.sqrt(len(y));bounds=[]
    for j in range(c):
        n=int((y==j).sum());k=int(correct[y==j].sum())
        if not n:bounds.append(0.);continue
        p=k/n;z=1.96
        bounds.append(float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)))
    minimum=min(bounds);accepted=bool(private_branch and coverage and lower>0 and minimum>1e-12)
    return dict(paired_loo_gain=gain,paired_loo_gain_lower=lower,minimum_loo_wilson_lower=minimum,
        all_classes_in_proxy_predictions=bool(coverage),private_augmentation_published=accepted,
        gate_scope='predeclared LOO heuristic, not a statistical/local-utility certificate')

def design():
    freeze('JOINT-NAMING-DESIGN.json',dict(
        motivation='Use observed public counts directly; empty regions contribute no fabricated public label certainty.',
        tasks=list(CLASSES),settings=SETTINGS,seeds=SEEDS,budgets=[1,3,5,10,20,50],all_original_acquisitions=True,
        primary_private_source='T1,180 steps,K=C,M; same existing protected matrix',
        score='L_public + lambda * (public region weight / mean region weight) * projected DP matrix',
        lambdas=LAMBDAS,public_baselines='original twelve plus observed-count matching and uniform-empty matching',
        candidates='all fourteen public models, joint lambdas in ascending order, fixed private centered and coverage rules',
        selection='exact LOO accuracy/balanced accuracy; candidate order breaks ties; no external validation labels',
        loo_acceleration='re-solve matched observed edges only; exact held-out correctness, including ties',
        publication_gate='same paired LOO heuristic; compare with within-budget selected fourteen-model public portfolio',
        class_checked_publication='Publish selected private augmentation or public fallback only with positive per-class LOO score and proxy coverage; otherwise clients retain local models.',
        private_local_utility='offline benchmark only, never a selection input',
        prior_outputs_unchanged=True,new_private_queries=0,new_labels=0,test_routes_choice=False))

def main(only=None,confirm=False,freeze_only=False):
    design()
    if freeze_only:return
    from confirm_joint import CONFIRM,CONFIRM_SETTINGS,CONFIRM_SEEDS
    outdir=ROOT/('analysis/joint-confirmation' if confirm else 'analysis/joint-naming');outdir.mkdir(exist_ok=True)
    tasks=['cifar10','mnist'] if confirm else list(CLASSES)
    for task in ([only] if only else tasks):
        data=public.public_features(task);data['dataset']=task;c=CLASSES[task]
        ids,reps,centres=base.partition(data,c);weights=np.bincount(ids,minlength=c);data['proxy_assignment']=ids
        allx=np.concatenate([data['x_development'],data['x_test']])
        assignment=torch.cdist(torch.as_tensor(allx,device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
        dev=assignment[:len(data['x_development'])];test=assignment[len(data['x_development']):]
        with np.load(ROOT/'inputs'/f'{task}.npz') as b:truth=b['y_test']
        graphs={a:public.graph_operator(task,data,a) for a in [.8,.95]}
        if confirm:
            cases=[]
            for d,n,e in CONFIRM_SETTINGS:
                if d!=task:continue
                for seed in CONFIRM_SEEDS:
                    key=release_key(task,n,e,seed,180,'regional',c,'M');source=CONFIRM/'results'/(key+'.json')
                    result=json.loads(source.read_text());private=np.load(CONFIRM/result['diagnostic_file'])['scores']
                    cases.append((n,e,seed,key,sha(source),result,private))
            input_folder=CONFIRM/'samples'
        else:cases=public.private_cases(task);input_folder=ROOT/'public'
        locals={}
        if not confirm:
            plan=json.loads((PARENT/'STUDY-SCHEDULE.json').read_text())['design']['jobs']
            from common import SEEDS as original_seeds
            for j in plan:
                if j['dataset']==task and j['method']=='Local-only':
                    filename=f"{task}_{j['n']}_0.0_{j['seed']}_Local-only_{c}_center_M_0_0.json"
                    local=json.loads((PARENT/'results'/filename).read_text());locals[(j['n'],j['seed'])]=local
        inputs=sorted(input_folder.glob(f'{task}_*_m*.json'))
        for number,path in enumerate(inputs):
            target=outdir/path.name
            if target.exists():continue
            row=json.loads(path.read_text());y=np.array(row['labels']);region=dev[row['indices']]
            fitted,choice,extra=public.public_candidates(data,row,graphs,assignment,centres,weights)
            counts=np.zeros((c,c));np.add.at(counts,(region,y),1)
            count_labels,count_correct=count_loo(counts,region,y)
            uni_labels,uni_correct=uniform_loo(counts,weights,region,y)
            additions=[dict(name='count-matching',labels=count_labels,correct=count_correct),dict(name='uniform-empty-matching',labels=uni_labels,correct=uni_correct)]
            models=[];baselines={}
            for candidate in fitted:
                name=candidate['name'];correct=candidate['loo'].argmax(1)==y
                value=dict(**metrics(candidate['test'].argmax(1),truth,c),loo_score=score_correct(correct,y,c,task=='web'))
                if not confirm:
                    assert value==row['baselines'][name],(task,name)
                models.append(dict(name=name,correct=correct,score=value['loo_score'],metric=value,
                    coverage=len(np.unique(candidate['proxy'].argmax(1)))==c,private=False))
                baselines[name]=value
            for item in additions:
                value=dict(**metrics(item['labels'][test],truth,c),loo_score=score_correct(item['correct'],y,c,task=='web'))
                models.append(dict(name=item['name'],correct=item['correct'],score=value['loo_score'],metric=value,coverage=True,private=False))
                baselines[item['name']]=value
            pindex=max(range(len(models)),key=lambda j:(models[j]['score'],-j));best_public=models[pindex]
            def one(case):
                n,e,seed,key,digest,source,private=case;options=list(models);a=project(private)*weights[:,None]/weights.mean()
                for lam in LAMBDAS:
                    label,correct=count_loo(counts+lam*a,region,y)
                    options.append(dict(name=f'joint-{lam:g}',labels=label,correct=correct,score=score_correct(correct,y,c,task=='web'),private=True,coverage=True))
                for mode in ['center','coverage']:
                    label=decode(private,weights,mode);correct=label[region]==y
                    options.append(dict(name='private-'+mode,labels=label,correct=correct,score=score_correct(correct,y,c,task=='web'),private=True,coverage=len(np.unique(label))==c))
                index=max(range(len(options)),key=lambda j:(options[j]['score'],-j));chosen=options[index]
                value=metrics(chosen['labels'][test],truth,c) if chosen['private'] else chosen['metric']
                g=gate(chosen['correct'],best_public['correct'],y,c,chosen['coverage'],chosen['private'])
                g.update(selected=chosen['name'],loo_score=chosen['score'],public_loo_score=best_public['score'],candidate_loo_scores={x['name']:x['score'] for x in options})
                pub=gate(best_public['correct'],best_public['correct'],y,c,best_public['coverage'],False)
                checked=g if g['private_augmentation_published'] else pub
                published=bool(checked['all_classes_in_proxy_predictions'] and checked['minimum_loo_wilson_lower']>1e-12)
                fallback=value if g['private_augmentation_published'] else best_public['metric']
                local=source['private_local_only_endpoint'] if confirm else locals[(n,seed)]
                utility=fallback if published else local
                g.update(public_classifier_published=published,private_local_utility_used_for_selection=False)
                return dict(n=n,epsilon=e,seed=seed,protected_source=key,source_sha256=digest,
                    protected_group='confirmation' if confirm else 'development',
                    ungated=value,gated=fallback,gate=g,
                    abstention_client_utility={key:utility[key] for key in ['accuracy','balanced_accuracy']},
                    selected_uses_private_votes=chosen['private'])
            if task=='femnist62':
                with ThreadPoolExecutor(max_workers=8) as pool:outputs=list(pool.map(one,cases))
            else:outputs=list(map(one,cases))
            result={k:row[k] for k in ['dataset','scheme','draw','q','m','indices','labels','missing_classes','sample_sha256']}
            result.update(label_budget=row['m'],external_validation_labels=0,source=path.name,source_sha256=sha(path),
                selected_public=best_public['name'],baselines=baselines,public_portfolio=best_public['metric'],hybrid=outputs,
                policy='joint observed-count naming; no imputed evidence; original policy preserved separately')
            save(target,result)
            if number%20==0:print(task,number,'/',len(inputs),row['scheme'],row['m'],'public',round(best_public['metric']['accuracy']*100,2),flush=True)
        save(ROOT/(f'JOINT-{task}'+('-CONFIRMATION' if confirm else '')+'-COMPLETE.json'),dict(label_sets=len(inputs),utc=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CLASSES));p.add_argument('--confirm',action='store_true');p.add_argument('--freeze-only',action='store_true');a=p.parse_args();main(a.dataset,a.confirm,a.freeze_only)
