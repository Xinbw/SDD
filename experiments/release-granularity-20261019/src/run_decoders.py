"""Prespecified original/fresh seed groups; every decoder is retained."""
import hashlib
import json
import time

from experiment_engine import ROOT, key, run, save


def jobs():
    out = []
    for dataset, n, epsilon, seeds, decoders, group in [
        ('cifar10', 10000, .1, range(801,806), ['center','matching','soft','center-soft'], 'original-five'),
        ('cifar10', 10000, .1, range(8201,8206), ['center','matching','soft','center-soft'], 'fresh-five'),
        ('mnist', 10000, .1, range(8201,8206), ['center','matching','soft','center-soft'], 'fresh-five'),
        ('cifar10', 1000, .1, range(8201,8206), ['center','matching'], 'fresh-five-small-cohort'),
        ('web', 3000, 1., range(8101,8106), ['center','matching','soft','center-soft'], 'existing-fixed-six')]:
        for seed in seeds:
            for decoder in decoders:
                out.append(dict(dataset=dataset,n=n,epsilon=epsilon,seed=seed,method='SDD',
                    k=6 if dataset=='web' else 10,decoder=decoder,alpha=None if dataset=='web' else .1,group=group))
    for seed in range(8201,8206):
        out.append(dict(dataset='cifar10',n=10000,epsilon=.1,seed=seed,method='uniform-query',k=10,
            decoder='center',alpha=.1,group='uniform-query-control'))
    return out


def main():
    plan = dict(jobs=jobs(), declared_before_new_private_results=True,
        decoder_selection='All prespecified variants reported; no test-driven routing.',
        matching='public-size-weighted assignment, K=C only', soft='projected vote proportions',
        center_soft='softmax(projected vote proportions minus public-size-weighted prior), temperature 0.1',
        local_steps=180, label_release_rounds=3, final_round_label_release_unused=True,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    schedule=ROOT/'DECODER-SCHEDULE.json'
    if schedule.exists():
        assert json.loads(schedule.read_text())['jobs']==plan['jobs']
    else:
        save(schedule,plan)
    for i,job in enumerate(plan['jobs']):
        path=ROOT/'results'/(key(job)+'.json')
        result=json.loads(path.read_text()) if path.exists() else run(job)
        if job['group']=='original-five' and job['decoder']=='center':
            parent=ROOT.parent/'review-round2-20261013/results'/f"label_free_cifar10_10000_0.1_0.1_{job['seed']}_SDD.json"
            expected=json.loads(parent.read_text())
            assert abs(result['accuracy']-expected['accuracy'])<2e-6, (result['accuracy'],expected['accuracy'])
            assert result['population']==expected['population']
        save(ROOT/'DECODER-STATUS.json',dict(completed=i+1,total=len(plan['jobs']),last=key(job)))
    save(ROOT/'DECODER-COMPLETE.json',dict(completed=len(plan['jobs']),utc=time.time()))


if __name__=='__main__':
    main()
