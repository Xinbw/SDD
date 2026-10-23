"""Meaningful publication thresholds on the SAME released validation outputs (no new query/budget).

Policies (each a separately declared rule, alpha_h = .025 for its own class family):
  gain >= gamma            : released gain lower bound > gamma
  recall >= r0             : gain lower > 0 and every direct-linear recall lower bound >= r0
  non-degradation eta      : gain lower > 0 and, for every class c,
                             lower[ h_c,10 - h_c,01 + eta * sum_ab h_c,ab ] >= 0,
                             i.e. recall_f,c - recall_b,c >= -eta on the finite held-out cohort.
Non-degradation band: coefficient vector (eta, eta-1, 1+eta, eta) on (00,01,10,11) has L2 norm
sqrt(2+4 eta^2); one-sided union over C targets, z = sqrt(2 log(C/alpha)); rounding (2+2 eta)/s.
Test labels and clean counts are loaded only for the offline utility/coverage diagnostics.
"""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];VALUE=ROOT.parent/'release-value-20261019'
C=10;S=1<<16;ALPHA=.025
GAMMA=[0.,.05,.1,.2];R0=[0.,.25,.5,.6,.7];ETA=[0.,.05,.1,.25,.5]
def nondeg_lower(h,v,eta):
    h=np.asarray(h).reshape(C,4);t=h[:,2]-h[:,1]+eta*h.sum(1)
    return t-v*math.sqrt(2+4*eta*eta)*math.sqrt(2*math.log(C/ALPHA))-(2+2*eta)/S
rows=[]
for p in sorted((VALUE/'results/validation').glob('*.json')):
    r=json.loads(p.read_text());lin=json.loads((VALUE/'analysis/linear-validation'/p.name).read_text())
    z=np.load(VALUE/r['diagnostic_file']);hist=z['released_class_histogram'];clean=z['client_joint_counts'].sum(0).astype(float)
    mass=clean.sum(1);true_diff=np.divide(clean[:,2]-clean[:,1],mass,out=np.zeros(C),where=mass>0)
    d=r['research_diagnostics'];v=r['privacy_classes']['residual_std']
    rows.append(dict(e=r['epsilon_total'],private=r['candidate_uses_private'],gl=r['protected_validation']['gain_lower'],
        rl=lin['bounds']['recall_lower'],nd={eta:nondeg_lower(hist,v,eta).min() for eta in ETA},
        cand=d['candidate_test']['accuracy'],pub=d['public_test']['accuracy'],cand_min_recall=d['candidate_test']['minimum_recall'] if 'minimum_recall' in d['candidate_test'] else None,
        true_gain=d['true_gain'],true_recall=d['true_recall'],true_diff=true_diff.tolist()))
out={}
for e in [.1,1.,3.]:
    R=[x for x in rows if x['e']==e];res={}
    def summarize(name,accept,valid):
        a=np.array(accept);u=np.array([x['cand'] if ok else x['pub'] for x,ok in zip(R,accept)])
        res[name]=dict(accept=100*a.mean(),policy_test=100*u.mean(),false_accepts=int(sum(ok and not val for ok,val in zip(accept,valid))))
    for g in GAMMA:summarize(f'gain>={g}',[x['private'] and x['gl']>g for x in R],[x['true_gain']>g for x in R])
    for r0 in R0:summarize(f'recall>={r0}',[x['private'] and x['gl']>0 and min(x['rl'])>=r0 and (r0>0 or min(x['rl'])>0) for x in R],
        [x['true_gain']>0 and min(x['true_recall'])>=r0 and min(x['true_recall'])>0 for x in R])
    for eta in ETA:summarize(f'nondeg eta={eta}',[x['private'] and x['gl']>0 and x['nd'][eta]>=0 for x in R],
        [x['true_gain']>0 and min(x['true_diff'])>=-eta for x in R])
    out[str(e)]=res
(ROOT/'analysis/threshold-policies.json').write_text(json.dumps(dict(policies=out,new_private_queries=0,new_privacy_budget=0,
    scope='post-processing of the 3000 existing protected validation outputs; finite held-out cohort; false accepts use clean held-out diagnostics'),indent=2)+'\n')
for e,res in out.items():
    print('eps',e);[print(f'  {k:18s} accept={v["accept"]:5.1f} policy={v["policy_test"]:5.2f} FA={v["false_accepts"]}') for k,v in res.items()]
