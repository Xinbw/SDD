"""Fresh-seed fixed-public-resolution corroboration on natural author units."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import engine

ROOT=engine.ROOT
SEEDS=list(range(8101,8111))
METHODS=['SDD','DP-FedCT','DP-Head','PATE-DI','random','representative']
jobs=[dict(dataset='web',alpha=None,n=n,epsilon=e,seed=s,method=m,
    id=f'web_fixed_n{n}_e{e}_s{s}_{m}')
    for n in [1000,3000] for e in [.25,1.] for s in SEEDS for m in METHODS]
jobs.extend(dict(dataset='cifar10',alpha=.1,n=1000,epsilon=.1,seed=s,method=m,
    id=f'cifar_confirmation_n1000_a0.1_e0.1_s{s}_{m}') for s in SEEDS for m in METHODS[:4])
plan={'reason':'Separate actual participation from public resolution on sparse natural-author clients; fresh independent seeds.',
    'cohorts':[1000,3000],'budgets':[.25,1.],'seeds':SEEDS,'public_configuration_source':'frozen web n=1000 configs',
    'same_heldout_author_test_as_initial_study':True,'independent_confirmation_is_over_population_training_and_noise_seeds':True,
    'no_private_configuration_reselection':True,'all_public_partition_counts_are_5':True,
    'jobs':jobs,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}}
path=ROOT/'WEB-CONFIRMATION-SCHEDULE.json'
if not path.exists():path.write_text(json.dumps(plan,indent=2)+'\n')
else:assert json.loads(path.read_text())['jobs']==jobs
resume_pid=int(os.environ.get('SDD_RESUME_LEGACY_PID','0'))
print('Fresh-seed fixed-K jobs:',len(jobs),flush=True)
try:
    # Serialise GPU phases: the legacy parent is suspended between child jobs.
    while True:
        status=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(status.splitlines()[0])<4000:break
        time.sleep(15)
    (ROOT/'WEB-CONFIRMATION-START.json').write_text(json.dumps({'time':time.time(),
        'schedule_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
    completed=0
    for job in jobs:
        output=ROOT/'results'/f'{job["id"]}.json'
        if output.exists():completed+=1;continue
        configuration_file=f'{job["dataset"]}_a{job["alpha"]}_n1000.json'
        config=json.loads((ROOT/'public-selection'/configuration_file).read_text())['budgets'][str(job['epsilon'])]
        if job['dataset']=='web':assert config['k']==5
        print('START',completed+1,'/',len(jobs),job['id'],flush=True)
        result=engine.run(job['dataset'],job['n'],job['alpha'],job['seed'],job['method'],config['k'],job['epsilon'],
            mode=config['mode'],clip=config['clip'],pate=config['pate'])
        result['configuration_override']='fixed-public-1k'
        result['experiment_roles']=['fresh-seed fixed-resolution natural-author confirmation']
        result['public_configuration_file']=configuration_file
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(output)
        completed+=1
        (ROOT/'WEB-CONFIRMATION-STATUS.json').write_text(json.dumps({'completed':completed,'total':len(jobs),
            'last_job':job['id']},indent=2)+'\n')
    (ROOT/'WEB-CONFIRMATION-COMPLETE.json').write_text(json.dumps({'completed':completed},indent=2)+'\n')
finally:
    if resume_pid:
        os.kill(resume_pid,18) # SIGCONT on Linux; resume the original diagnostic queue.
