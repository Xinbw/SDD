"""Complete the prior uniform-query control with the same point estimator.

Only query placement/propagation geometry changes. Both methods query ten
points, use centered decoding, and retain matched privacy and client steps.
"""
from common import ROOT,key,freeze,save,sha
from learning import run

jobs=[dict(dataset='cifar10',n=10000,epsilon=.1,seed=s,method='SDD',k=10,decoder='center',
    estimator='R',proxy_r2=64,proxy_r3=64) for s in range(8201,8206)]
parent=ROOT.parent/'release-granularity-20261019/results'
freeze(ROOT/'QUERY-CONTROL-SCHEDULE.json',dict(jobs=jobs,role='matched point-query placement control',
    comparison='prior fixed uniform-query/nearest-query propagation, K=10, centered',
    seeds_are_paired_replays_not_a_new_confirmation_group=True,
    uniform_sources={str(s):sha(parent/f'cifar10_10000_0.1_{s}_uniform-query_10_center.json') for s in range(8201,8206)},
    model_queries_per_client_round=10,client_steps=180))
for j in jobs:
    if not (ROOT/'results'/(key(j)+'.json')).exists():run(j)
save(ROOT/'QUERY-CONTROL-COMPLETE.json',dict(completed=len(jobs)))
