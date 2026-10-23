"""Execute the predeclared matrix after every public configuration is frozen."""
from pathlib import Path
import hashlib
import json
import time
import traceback
import engine

ROOT=engine.ROOT
EPS=[.01,.025,.05,.1,.25,.5,1.]
METHODS=['SDD','DP-FedCT','DP-Head','PATE-DI']
jobs={}

def add(dataset,n,alpha,epsilon,seed,method,role,override=None):
    key=f'{dataset}_n{n}_a{alpha}_e{epsilon}_s{seed}_{method}'
    if override:key+='_'+override
    if key in jobs:jobs[key]['roles'].append(role);return
    jobs[key]=dict(id=key,dataset=dataset,n=n,alpha=alpha,epsilon=epsilon,seed=seed,method=method,roles=[role],override=override)

for dataset in ['cifar10','mnist']:
    for alpha in [.1,1.]:
        for epsilon in EPS:
            for seed in [801,802,803]:
                for method in METHODS:add(dataset,10000,alpha,epsilon,seed,method,'complete budget curve')
        for seed in [804,805]:
            for method in METHODS:add(dataset,10000,alpha,.1,seed,method,'five-seed decisive budget')
        for epsilon in [.1,1.]:
            for seed in [801,802,803,804,805]:
                for method in ['random','representative','weak']:
                    add(dataset,10000,alpha,epsilon,seed,method,'same-K semantic attribution')
                add(dataset,10000,alpha,epsilon,seed,'SDD','paired semantic attribution')
        for seed in [801,802,803,804,805]:
            add(dataset,10000,alpha,.1,seed,'SDD','postprocessing attribution','raw')
            add(dataset,10000,alpha,1.,seed,'SDD','ten-percent collusion calibration','collusion')
    for n in [100,1000]:
        for epsilon in [.1,1.]:
            for seed in [801,802,803]:
                for method in ['SDD','DP-FedCT','DP-Head']:
                    add(dataset,n,.1,epsilon,seed,method,'matched actual population scaling')
for n in [100,1000,3000]:
    for epsilon in [.1,.25,1.]:
        for seed in [801,802,803]:
            for method in METHODS:add('web',n,None,epsilon,seed,method,'natural-author Web content task')
    for seed in [804,805]:
        for method in METHODS:add('web',n,None,.1,seed,method,'five-seed Web point')
for epsilon in [.1,1.]:
    for seed in [801,802,803,804,805]:
        for method in ['SDD','random','representative']:
            add('web',3000,None,epsilon,seed,method,'Web semantic attribution')

schedule=ROOT/'schedule.json'
matrix={'declared_before_final_test_access':True,'primary_image_metric':'mean client accuracy',
        'primary_web_metric':'mean client balanced accuracy','full_grid_seeds':[801,802,803],
        'decisive_seeds':[801,802,803,804,805],'jobs':list(jobs.values()),
        'legacy_ResNet_diagnostic':{'dataset':'cifar10','alpha':.1,'actual_clients':100,'calibrated_clients':100,
          'K':10,'budgets':[.1,1.],'seeds':[42,43,44],'methods':['SDD','DP-FedCT'],
          'epochs_per_round':20,'T':3,'old_R1_checkpoints_reused':True}}
if schedule.exists():assert json.loads(schedule.read_text())==matrix
else:schedule.write_text(json.dumps(matrix,indent=2)+'\n')
print('Predeclared jobs:',len(jobs),flush=True)
while not (ROOT/'public-selection/FROZEN.json').exists():time.sleep(15)
frozen=json.loads((ROOT/'public-selection/FROZEN.json').read_text())
assert frozen['all_configs_frozen']
for name,digest in frozen['files'].items():
    assert hashlib.sha256((ROOT/'public-selection'/name).read_bytes()).hexdigest()==digest
    assert json.loads((ROOT/'public-selection'/name).read_text())['configuration_frozen']
assert len(frozen['files'])==15
start=ROOT/'FINAL-TEST-START.json'
if not start.exists():
    start.write_text(json.dumps({'unix_time':time.time(),'schedule_sha256':hashlib.sha256(schedule.read_bytes()).hexdigest(),
        'public_freeze_sha256':hashlib.sha256((ROOT/'public-selection/FROZEN.json').read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}},indent=2)+'\n')
started=time.time();completed=0
for job in jobs.values():
    output=ROOT/'results'/f'{job["id"]}.json'
    if output.exists():completed+=1;continue
    config=json.loads((ROOT/'public-selection'/f'{job["dataset"]}_a{job["alpha"]}_n{job["n"]}.json').read_text())['budgets'][str(job['epsilon'])]
    mode='raw' if job['override']=='raw' else config['mode']
    colluders=job['n']//10 if job['override']=='collusion' else 0
    print('START',completed+1,'/',len(jobs),job['id'],flush=True)
    try:
        result=engine.run(job['dataset'],job['n'],job['alpha'],job['seed'],job['method'],config['k'],
            job['epsilon'],mode=mode,clip=config['clip'],pate=config['pate'],colluders=colluders)
        result['experiment_roles']=job['roles'];result['configuration_override']=job['override']
        result['public_configuration_file']=f'{job["dataset"]}_a{job["alpha"]}_n{job["n"]}.json'
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(output)
        completed+=1
        (ROOT/'STATUS.json').write_text(json.dumps({'completed':completed,'total':len(jobs),
            'last_job':job['id'],'seconds':time.time()-started},indent=2)+'\n')
    except Exception:
        (ROOT/'logs'/f'{job["id"]}.error.txt').write_text(traceback.format_exc())
        raise
(ROOT/'SUITE-COMPLETE.json').write_text(json.dumps({'jobs_completed':completed,'seconds':time.time()-started},indent=2)+'\n')
