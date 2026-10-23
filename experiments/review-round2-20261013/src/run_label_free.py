"""Fixed taxonomy-resolution checks requiring no public training labels.

K equals the publicly specified number of task classes; consensus is centered.
Both estimators are reported. No validation or test labels select a setting.
Public feature maps are the existing label-independent maps, held fixed.
"""
import hashlib
import json
import time

import engine
from attribution import save

ROOT=engine.ROOT
jobs=[]
for dataset,n,seeds in [('cifar10',10000,range(801,806)),('mnist',10000,range(801,806)),
    ('cifar10',1000,range(8101,8111))]:
    for seed in seeds:
        for method in ['SDD','DP-FedCT']:
            jobs.append(dict(dataset=dataset,n=n,alpha=.1,epsilon=.1,seed=seed,method=method,k=10))
for n,epsilon in [(1000,.25),(3000,1.)]:
    for seed in range(8101,8111):
        for method in ['SDD','representative','DP-FedCT']:
            jobs.append(dict(dataset='web',n=n,alpha=None,epsilon=epsilon,seed=seed,method=method,k=6))
schedule=ROOT/'LABEL-FREE-SCHEDULE.json'
plan={'jobs':jobs,'rule':'K equals the public class taxonomy size; centered consensus for every dataset',
    'public_labels_used_for_training':0,'public_labels_used_for_configuration':0,
    'validation_labels_used_for_configuration':0,'both_estimators_reported_without_routing':True,
    'public_features_label_independent_and_unchanged':True,
    'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}}
if schedule.exists():assert json.loads(schedule.read_text())['jobs']==jobs
else:save(schedule,plan)
if not (ROOT/'LABEL-FREE-START.json').exists():save(ROOT/'LABEL-FREE-START.json',{'utc':time.time(),'schedule_sha256':hashlib.sha256(schedule.read_bytes()).hexdigest()})
for i,j in enumerate(jobs):
    name='label_free_'+'_'.join(str(j[k]) for k in ['dataset','n','alpha','epsilon','seed','method'])
    path=ROOT/'results'/f'{name}.json'
    if path.exists():continue
    result=engine.run(j['dataset'],j['n'],j['alpha'],j['seed'],j['method'],j['k'],j['epsilon'],mode='center')
    result.update(configuration_rule='fixed taxonomy K=C, centered consensus',public_training_labels=0,
        public_configuration_labels=0,public_configuration_reselected=False,
        experiment_role='Label-free regional knowledge reuse; geometry and taxonomy fixed without public labels')
    save(path,result)
    save(ROOT/'LABEL-FREE-STATUS.json',{'completed':i+1,'total':len(jobs),'last_job':name})
    print('LABEL FREE',i+1,'/',len(jobs),name,flush=True)
save(ROOT/'LABEL-FREE-COMPLETE.json',{'completed':len(jobs),'utc':time.time()})
