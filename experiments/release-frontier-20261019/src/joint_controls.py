"""Identical joint naming on protected head and pooled point-query views."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy.special import softmax
import engine_base as base
from anchored_fusion import LAMBDAS,count_loo,score_correct
from run_private import decode,metrics,project
from common import ROOT,CLASSES,SETTINGS,SEEDS,freeze,save,sha,release_key
from confirm_joint import CONFIRM,CONFIRM_SETTINGS,CONFIRM_SEEDS

def main(confirm=False):
    freeze('JOINT-RELEASE-CONTROLS.json',dict(
        objects=['fixed-B=1 private head','per-example votes pooled over public regions after release'],
        joint_rule='same L + lambda weighted private regional score, same lambda grid and fourteen public baselines',
        head_region='mean softmax posterior of protected head over proxy region',
        point_region='average unprojected released per-example scores within each public region, then project',
        head_native_classifier_is_additional_fixed_candidate=True,lambdas=LAMBDAS,
        private_labels_or_clean_counts_used=False,extra_private_query_from_postprocessing=False,
        selection='exact LOO; fixed public-first ties; no test routing'))
    folder=ROOT/('analysis/joint-confirmation' if confirm else 'analysis/joint-naming')
    output=ROOT/('analysis/joint-controls-confirmation' if confirm else 'analysis/joint-controls');output.mkdir(exist_ok=True)
    settings=CONFIRM_SETTINGS if confirm else SETTINGS;seeds=CONFIRM_SEEDS if confirm else SEEDS;stage=CONFIRM if confirm else ROOT
    for task in (['cifar10','mnist'] if confirm else CLASSES):
        c=CLASSES[task]
        with np.load(ROOT/'inputs'/f'{task}.npz') as b:data={k:b[k] for k in ['x_proxy','x_development','x_test','y_test']}
        ids,reps,centres=base.partition(data,c);weights=np.bincount(ids,minlength=c)
        torch.backends.cuda.matmul.allow_tf32=False
        assignment=torch.cdist(torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
        dev,test=assignment[:len(data['x_development'])],assignment[len(data['x_development']):]
        cases=[]
        for d,n,e in settings:
            if d!=task:continue
            for seed in seeds:
                for method in ['head-fixed','per-example']:
                    key=release_key(task,n,e,seed,180,method,0,'-');path=stage/'results'/(key+'.json')
                    row=json.loads(path.read_text());released=np.load(stage/row['diagnostic_file'])['scores']
                    if method=='head-fixed':
                        posterior=softmax(data['x_proxy']@released,axis=1)
                        regional=np.stack([posterior[ids==k].mean(0) for k in range(c)])
                        native_dev=(data['x_development']@released).argmax(1);native_metric=row['endpoints']['broadcast']
                    else:
                        regional=np.stack([released[ids==k].mean(0) for k in range(c)])
                        native_dev=native_metric=None
                    cases.append(dict(n=n,epsilon=e,seed=seed,method=method,source=key,source_sha256=sha(path),
                        regional=regional,native_dev=native_dev,native_metric=native_metric))
        for number,path in enumerate(sorted(folder.glob(f'{task}_*_m*.json'))):
            target=output/path.name
            if target.exists():continue
            r=json.loads(path.read_text());y=np.array(r['labels']);region=dev[r['indices']]
            counts=np.zeros((c,c));np.add.at(counts,(region,y),1)
            public=[dict(name=name,score=value['loo_score'],metric=value,private=False) for name,value in r['baselines'].items()]
            def one(case):
                a=project(case['regional'])
                a=a*weights[:,None]/weights.mean();options=list(public)
                for lam in LAMBDAS:
                    label,correct=count_loo(counts+lam*a,region,y)
                    options.append(dict(name=f'joint-{lam:g}',score=score_correct(correct,y,c,task=='web'),labels=label,private=True))
                if case['native_dev'] is not None:
                    correct=case['native_dev'][r['indices']]==y
                    options.append(dict(name='native-head',score=score_correct(correct,y,c,task=='web'),metric=case['native_metric'],private=False))
                for mode in ['center','coverage']:
                    label=decode(case['regional'],weights,mode);correct=label[region]==y
                    options.append(dict(name='private-region-'+mode,score=score_correct(correct,y,c,task=='web'),labels=label,private=True))
                choice=max(range(len(options)),key=lambda j:(options[j]['score'],-j));chosen=options[choice]
                value=metrics(chosen['labels'][test],data['y_test'],c) if chosen['private'] else chosen['metric']
                return dict(n=case['n'],epsilon=case['epsilon'],seed=case['seed'],method=case['method'],source=case['source'],source_sha256=case['source_sha256'],
                    selected=chosen['name'],loo_score=chosen['score'],metric=value,extra_private_queries=0)
            if task=='femnist62':
                with ThreadPoolExecutor(max_workers=8) as pool:result=list(pool.map(one,cases))
            else:result=list(map(one,cases))
            save(target,dict(dataset=task,scheme=r['scheme'],draw=r['draw'],q=r['q'],m=r['m'],source=path.name,source_sha256=sha(path),controls=result))
            if number%50==0:print(task,'joint controls',number,'/',len(list(folder.glob(f'{task}_*_m*.json'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--confirm',action='store_true');main(p.parse_args().confirm)
