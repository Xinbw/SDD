"""Real writer clients, disjoint client records and evaluation writers."""
import hashlib
import json
import time

from experiment_engine import ROOT, key, run, save
from public_design import select


def main():
    design=select('femnist62',[1000,2000],[.1,1.],[31,62,124])
    jobs=[]
    for n in [1000,2000]:
        for seed in range(8401,8406):
            jobs.append(dict(dataset='femnist62',n=n,epsilon=0.,seed=seed,method='Local-only',k=62,
                decoder='center',alpha=None,group='natural-writers'))
            for epsilon in [.1,1.]:
                cfg=next(r for r in design['records'] if r['n']==n and r['epsilon']==epsilon)
                for method,decoder in [('SDD','center'),('SDD','matching'),('DP-FedCT','center'),('DP-Head','center')]:
                    job=dict(dataset='femnist62',n=n,epsilon=epsilon,seed=seed,method=method,k=62,
                        decoder=decoder,alpha=None,group='natural-writers')
                    if method=='DP-Head':job['clip']=cfg['selected_clip']
                    jobs.append(job)
    plan=dict(jobs=jobs,declared_before_natural_private_results=True,private_writers_disjoint=True,
        public_private_validation_test_writers_disjoint=True,all_training_records_per_selected_writer=True,
        steps_per_client=180,fixed_regional_rule='K=C=62; center or prespecified weighted matching',
        public_configuration_labels='DP-Head clipping only, 5000 development labels; SDD uses none',
        public_design_sha256=hashlib.sha256((ROOT/'public-selection/femnist62_quality.json').read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    schedule=ROOT/'NATURAL-SCHEDULE.json'
    if schedule.exists():assert json.loads(schedule.read_text())['jobs']==jobs
    else:save(schedule,plan)
    for i,job in enumerate(jobs):
        path=ROOT/'results'/(key(job)+'.json')
        if not path.exists():run(job)
        save(ROOT/'NATURAL-STATUS.json',dict(completed=i+1,total=len(jobs),last=key(job)))
    save(ROOT/'NATURAL-COMPLETE.json',dict(completed=len(jobs),utc=time.time()))


if __name__=='__main__':main()
