"""Public-resource and matched-computation attribution, with sealed selection.

All metrics are offline benchmark diagnostics, outside deployment transcripts.
The parent experiment's learning, noise and privacy routines remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch

import engine

ROOT=engine.ROOT
PARENT=ROOT.parent/'review-evidence-20261012'
PUBLIC_SEEDS=[91001,91002,91003,91004,91005]
LABELS={'web':['comp','rec','sci','talk','alt/soc','misc']}


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def public_data(dataset,split):
    # Selection never loads private or final-test arrays from the feature bundle.
    with np.load(ROOT/'inputs'/f'{dataset}.npz',allow_pickle=False) as bundle:
        return {k:bundle[k] for k in ['x_development','y_development',f'x_{split}',f'y_{split}']}


def fit_public(data,seed,lr,steps):
    x=torch.as_tensor(data['x_development'],device=engine.DEVICE)
    y=torch.as_tensor(data['y_development'],device=engine.DEVICE)
    c=int(y.max())+1
    g=torch.Generator(device=engine.DEVICE).manual_seed(seed)
    w=torch.randn((1,x.shape[1],c),device=engine.DEVICE,generator=g)*.01
    velocity=torch.zeros_like(w)
    index=torch.arange(len(x),device=engine.DEVICE)[None]
    sizes=torch.tensor([len(x)],device=engine.DEVICE)
    old_lr=engine.LR;engine.LR=lr
    try:
        for r in range(1,steps//engine.STEPS+1):
            w,velocity=engine.train_round(w,velocity,x,y,index,sizes,seed,r)
    finally:
        engine.LR=old_lr
    return w


def evaluate(w,data,split):
    acc,balanced=engine.evaluate(w,torch.as_tensor(data[f'x_{split}'],device=engine.DEVICE),
        torch.as_tensor(data[f'y_{split}'],device=engine.DEVICE))
    return {'accuracy':acc,'balanced_accuracy':balanced,'per_class_recall':engine.evaluate.last_recall.copy()}


def select_public():
    path=ROOT/'PUBLIC-RESOURCE-FROZEN.json'
    if path.exists():return json.loads(path.read_text())
    configs={}
    for dataset in ['cifar10','mnist','web']:
        data=public_data(dataset,'validation');candidates=[]
        field='balanced_accuracy' if dataset=='web' else 'accuracy'
        for steps in [60,180,600]:
            for lr in [.03,.1,.3]:
                runs=[evaluate(fit_public(data,s,lr,steps),data,'validation') for s in PUBLIC_SEEDS]
                candidates.append({'steps':steps,'lr':lr,'validation_mean':float(np.mean([r[field] for r in runs])),
                    'validation_runs':runs})
        matched=max([q for q in candidates if q['steps']==180],key=lambda q:(q['validation_mean'],-q['lr']))
        strongest=max(candidates,key=lambda q:(q['validation_mean'],-q['steps'],-q['lr']))
        configs[dataset]={'public_training_examples':len(data['y_development']),'matched_180':matched,
            'validation_selected':strongest,'candidates':candidates}
        print('PUBLIC SELECT',dataset,'180',matched['lr'],matched['validation_mean'],
            'validated',strongest['steps'],strongest['lr'],strongest['validation_mean'],flush=True)
    result={'selection_uses_private_data':False,'selection_uses_final_test':False,
        'selection_seeds':PUBLIC_SEEDS,'learning_rates':[.03,.1,.3],'training_steps':[60,180,600],
        'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'datasets':configs,
        'parent_frozen_configs_sha256':hashlib.sha256((ROOT/'public-selection/FROZEN.json').read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}}
    save(path,result);return result


def original(dataset,n,alpha,epsilon,seed,method):
    if dataset=='web':name=f'web_fixed_n{n}_e{epsilon}_s{seed}_{method}.json'
    elif n==1000:name=f'cifar_confirmation_n1000_a0.1_e{epsilon}_s{seed}_{method}.json'
    else:name=f'{dataset}_n{n}_a{alpha}_e{epsilon}_s{seed}_{method}.json'
    path=PARENT/'results'/name
    return (json.loads(path.read_text()),path) if path.exists() else (None,path)


def config(dataset,n,alpha,epsilon):
    cn=1000 if dataset=='web' else n
    path=ROOT/'public-selection'/f'{dataset}_a{alpha}_n{cn}.json'
    return json.loads(path.read_text())['budgets'][str(epsilon)],path


def local_control(dataset,n,alpha,seed,method,public_config):
    data=engine.load_data(dataset)
    w,v,idx,sizes,pop=engine.initial_state(data,n,alpha,seed)
    w=w.clone();v=v.clone()
    x=torch.as_tensor(data['x_private'],device=engine.DEVICE);y=torch.as_tensor(data['y_private'],device=engine.DEVICE)
    index=torch.as_tensor(idx,device=engine.DEVICE);sizes_t=torch.as_tensor(sizes,device=engine.DEVICE)
    xp=torch.as_tensor(data['x_proxy'],device=engine.DEVICE);targets=None
    if method=='Local+Public':
        pd=public_data(dataset,'validation')
        teacher=fit_public(pd,seed,public_config['lr'],public_config['steps'])
        targets=torch.matmul(xp,teacher[0]).argmax(-1)
    for r in [2,3]:w,v=engine.train_round(w,v,x,y,index,sizes_t,seed,r,xp,targets)
    return dict(dataset=dataset,actual_clients=n,alpha=alpha,seed=seed,method=method,
        **evaluate(w,data,'test'),population=pop,optimizer_steps=180,batch_size=128,
        private_examples_per_later_batch=128 if targets is None else 64,
        public_examples_per_later_batch=0 if targets is None else 64,
        shared_private_information=False,private_model_released=False,
        public_teacher_configuration=public_config if targets is not None else None,
        experiment_role='Independent private learning and public-source transfer counterfactual',
        diagnostics_are_offline_benchmark_measurements=True)


def main():
    frozen=select_public()
    conditions=[]
    for dataset,n,seeds in [('cifar10',10000,list(range(801,806))),('mnist',10000,list(range(801,806))),
        ('cifar10',1000,list(range(8101,8111)))]:
        for s in seeds:conditions.append(dict(dataset=dataset,n=n,alpha=.1,epsilon=.1,seed=s))
    for n in [1000,3000]:
        for epsilon in [.25,1.]:
            for s in range(8101,8111):conditions.append(dict(dataset='web',n=n,alpha=None,epsilon=epsilon,seed=s))
    public_jobs=[dict(dataset=d,seed=s,method=m) for d,seeds in [('cifar10',list(range(801,806))+list(range(8101,8111))),
        ('mnist',list(range(801,806))),('web',list(range(8101,8111)))] for s in seeds
        for m in ['Public-only-180','Public-only-selected']]
    unique={tuple(j[k] for k in ['dataset','n','alpha','seed']):j for j in conditions}
    controls=[dict(dataset=j['dataset'],n=j['n'],alpha=j['alpha'],seed=j['seed'],method=m)
        for j in unique.values() for m in ['Local-only','Local+Public']]
    replays=[]
    for j in conditions:
        for m in (['SDD','representative','DP-FedCT','DP-Head','PATE-DI','random-representative'] if j['dataset']=='web' else ['DP-Head']):
            replays.append(dict(**j,method=m))
    jobs=public_jobs+controls+replays
    plan={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'conditions':conditions,'jobs':jobs,
        'public_configuration_frozen_sha256':hashlib.sha256((ROOT/'PUBLIC-RESOURCE-FROZEN.json').read_bytes()).hexdigest(),
        'private_learner_optimizer_steps':180,'dp_head_extra_adaptation_steps':0,
        'web_replay_includes_all_six_previously_reported_methods':True,
        'public_control_uses_all_original_public_development_labels':True}
    schedule=ROOT/'ROUND2-SCHEDULE.json'
    if schedule.exists():assert json.loads(schedule.read_text())['jobs']==jobs
    else:save(schedule,plan)
    if not (ROOT/'ROUND2-START.json').exists():save(ROOT/'ROUND2-START.json',{'utc':time.time(),'schedule_sha256':hashlib.sha256(schedule.read_bytes()).hexdigest()})
    print('ROUND2 JOBS',len(jobs),flush=True)
    for i,j in enumerate(jobs):
        name='_'.join(str(j[k]) for k in ['dataset','n','alpha','epsilon','seed','method'] if k in j)
        path=ROOT/'results'/f'{name}.json'
        if path.exists():continue
        start=time.time();dataset=j['dataset'];method=j['method'];seed=j['seed']
        if method.startswith('Public-only'):
            key='matched_180' if method=='Public-only-180' else 'validation_selected'
            choice=frozen['datasets'][dataset][key]
            data=public_data(dataset,'test');w=fit_public(data,seed,choice['lr'],choice['steps'])
            result=dict(**j,**evaluate(w,data,'test'),public_labels=len(data['y_development']),
                optimizer_steps=choice['steps'],lr=choice['lr'],private_data_used=False,
                private_privacy_cost=0,logical_private_report_coordinates=0,
                selection_used_final_test=False,diagnostics_are_offline_benchmark_measurements=True)
        elif method in ['Local-only','Local+Public']:
            choice=frozen['datasets'][dataset]['validation_selected']
            result=local_control(dataset,j['n'],j['alpha'],seed,method,
                {k:choice[k] for k in ['lr','steps']})
        else:
            cfg,cp=config(dataset,j['n'],j['alpha'],j['epsilon'])
            result=engine.run(dataset,j['n'],j['alpha'],seed,method,cfg['k'],j['epsilon'],
                mode=cfg['mode'],clip=cfg['clip'],pate=cfg['pate'])
            parent,pp=original(dataset,j['n'],j['alpha'],j['epsilon'],seed,method)
            if parent is not None:
                error=max(abs(result[k]-parent[k]) for k in ['accuracy','balanced_accuracy','initial_accuracy','initial_balanced_accuracy'])
                assert error<2e-6,(pp,error)
                result['parent_result_file']=pp.name;result['parent_result_sha256']=hashlib.sha256(pp.read_bytes()).hexdigest()
                result['parent_endpoint_max_abs_difference']=error
            else:result['parent_result_file']=None
            result['public_configuration_file']=cp.name
            result['public_configuration_sha256']=hashlib.sha256(cp.read_bytes()).hexdigest()
            result['experiment_role']='Matched local endpoint and per-class offline evaluation; same privacy mechanism'
        result['seconds']=time.time()-start
        save(path,result)
        status={'completed':i+1,'total':len(jobs),'last_job':name,'utc':time.time()}
        save(ROOT/'ROUND2-STATUS.json',status)
        field='balanced_accuracy' if dataset=='web' else 'accuracy'
        print('COMPLETE',i+1,'/',len(jobs),name,round(result[field],5),flush=True)
    save(ROOT/'ROUND2-COMPLETE.json',{'completed':len(jobs),'utc':time.time()})


if __name__=='__main__':main()
