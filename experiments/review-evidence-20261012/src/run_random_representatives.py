"""Same-estimator, same-size random-region control on natural author clients."""
import hashlib
import json
from pathlib import Path
import engine

ROOT=engine.ROOT
jobs=[dict(n=n,epsilon=e,seed=s,id=f'web_fixed_n{n}_e{e}_s{s}_random-representative')
    for n in [1000,3000] for e in [.25,1.] for s in range(8101,8111)]
manifest={'purpose':'Separate semantic partition quality from representative versus mean opinion estimation.',
    'configuration_source':'frozen public web n=1000; K=5','jobs':jobs,'no_private_tuning':True,
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}}
path=ROOT/'RANDOM-REPRESENTATIVE-SCHEDULE.json'
if not path.exists():path.write_text(json.dumps(manifest,indent=2)+'\n')
for job in jobs:
    output=ROOT/'results'/f'{job["id"]}.json'
    if output.exists():continue
    config=json.loads((ROOT/'public-selection/web_aNone_n1000.json').read_text())['budgets'][str(job['epsilon'])]
    assert config['k']==5
    result=engine.run('web',job['n'],None,job['seed'],'random-representative',5,job['epsilon'],mode=config['mode'])
    result['configuration_override']='fixed-public-1k'
    result['experiment_roles']=['size-matched random partition with representative estimator']
    result['public_configuration_file']='web_aNone_n1000.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
(ROOT/'RANDOM-REPRESENTATIVE-COMPLETE.json').write_text(json.dumps({'runs':len(jobs)},indent=2)+'\n')
