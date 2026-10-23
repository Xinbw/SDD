"""Test-set minimum class recall of classifiers PUBLISHED by each verification rule (post-processing).

Same 3000 protected validation outputs and the same fixed 50/25/25 split as threshold-policies;
the test metrics are offline diagnostics and never route a decision.
"""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;VALUE=EXP/'release-value-20261019'
QUANTILES=[.05,.1,.25,.5]
def main():
    rows=[]
    for p in sorted((VALUE/'results/validation').glob('*.json')):
        r=json.loads(p.read_text());lin=json.loads((VALUE/'analysis/linear-validation'/p.name).read_text());d=r['research_diagnostics']
        rows.append(dict(e=r['epsilon_total'],accept_gain=bool(r['candidate_uses_private'] and r['protected_validation']['gain_lower']>0),
            accept_class=bool(r['candidate_uses_private'] and r['protected_validation']['gain_lower']>0 and min(lin['bounds']['recall_lower'])>0),
            proposed=bool(r['candidate_uses_private']),cand=min(d['candidate_test']['per_class_recall']),pub=min(d['public_test']['per_class_recall'])))
    out={}
    for e in [.1,1.,3.]:
        R=[x for x in rows if x['e']==e];res={}
        for name,key in [('proposed','proposed'),('mean-gain rule','accept_gain'),('all-recall>0 rule','accept_class')]:
            m=np.array([x['cand'] for x in R if x[key]])
            res[name]=dict(published=int(len(m)),cases=len(R),
                minimum_recall_quantiles={f'{q:g}':float(np.quantile(m,q)) for q in QUANTILES} if len(m) else None,
                share_below={f'{t:g}':float(np.mean(m<t)) if len(m) else None for t in [.01,.1,.25,.5]},
                values=sorted(float(v) for v in m))
        res['public fallback']=dict(minimum_recall_median=float(np.median([x['pub'] for x in R])))
        out[str(e)]=res
        print(e,{k:(v['published'],v['share_below']) if 'published' in v else v for k,v in res.items()})
    (ROOT/'analysis/published-minimum-recall.json').write_text(json.dumps(dict(by_epsilon=out,new_private_queries=0,new_privacy_budget=0,
        scope='test minimum class recall of the private candidate in cases each rule publishes; offline diagnostic'),indent=2)+'\n')
if __name__=='__main__':main()
