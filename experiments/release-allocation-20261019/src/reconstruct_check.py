"""Gate: stored clean statistics must reproduce every recorded validation histogram exactly."""
import json,sys
import numpy as np
from alloc_common import ROOT,VALUE,pv,load_public,private_regions,clean_counts,save
maps,data,samples=load_public();regions=private_regions(data['x_private'],maps['centres'])
sizes=np.bincount(maps['proxy'],minlength=10);checked=0;mismatch=0
for seed in pv.SEEDS:
    clean_counts(seed);tr,va,pop=pv.cohort(data['y_private'],seed)
    for e in ['0.1','1','3']:
        named=json.loads((VALUE/'results/naming'/f's{seed}_e{e}.json').read_text());assert named['population']==pop
        scores=np.load(VALUE/named['diagnostic_file'])['scores']
        for draw in range(100):
            r=json.loads((VALUE/'results/validation'/f's{seed}_e{e}_m10_d{draw}.json').read_text())
            chosen,pub,_=pv.select(scores,sizes,samples[draw],maps['development'])
            assert chosen['name']==r['candidate']
            h=pv.hist_counts(chosen['labels'],pub['labels'],regions[va],data['y_private'][va])
            stored=np.load(VALUE/r['diagnostic_file'])['client_joint_counts']
            checked+=1;mismatch+=int(not np.array_equal(h,stored))
    print(seed,checked,mismatch,flush=True)
save(ROOT/'results/reconstruction-check.json',dict(cases=checked,histogram_mismatches=mismatch,candidate_reselection_exact=True))
assert mismatch==0
