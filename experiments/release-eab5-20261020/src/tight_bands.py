"""Tighter one-sided class bands on the SAME 3000 protected validation outputs (no query, no budget).

The recorded class rule uses simultaneous two-sided bands (4C events) for all numerators and
denominators. A publication decision only needs a LOWER recall bound:
  recalls > 0   : every correct-mass numerator lower > 0; C one-sided events, z=sqrt(2 log(C/a))
  recalls >= r0 : numerator lower / denominator upper >= r0; 2C one-sided events, z=sqrt(2 log(2C/a))
with the same proxies (sqrt2 v, 2 v) and rounding terms (2/s, 4/s), alpha=.025 for the class family.
Gain rule unchanged (gain lower > 0). Test labels and clean held-out recalls only score.
"""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];VALUE=ROOT.parent/'release-value-20261019';EAB=ROOT.parent/'release-eab-20261020'
C=10;S=1<<16;ALPHA=.025;R0=[0.,.25,.5]

def tight_lower(h,v,r0):
    h=np.asarray(h).reshape(C,4);mass=h.sum(1);correct=h[:,2]+h[:,3]
    if r0==0:
        z=math.sqrt(2*math.log(C/ALPHA));return np.where(correct-(math.sqrt(2)*v*z+2/S)>0,1e-9,0.)
    z=math.sqrt(2*math.log(2*C/ALPHA));nl=np.clip(correct-(math.sqrt(2)*v*z+2/S),0,1);du=np.clip(mass+2*v*z+4/S,0,1)
    return np.clip(np.divide(nl,du,out=np.zeros(C),where=du>0),0,1)

def main():
    rows=[]
    for p in sorted((VALUE/'results/validation').glob('*.json')):
        r=json.loads(p.read_text());z=np.load(VALUE/r['diagnostic_file']);d=r['research_diagnostics'];v=r['privacy_classes']['residual_std']
        lin=json.loads((VALUE/'analysis/linear-validation'/p.name).read_text())
        gain=bool(r['candidate_uses_private'] and r['protected_validation']['gain_lower']>0)
        rows.append(dict(e=r['epsilon_total'],gain=gain,cand=d['candidate_test']['accuracy'],pub=d['public_test']['accuracy'],true_gain=d['true_gain'],true_recall=np.array(d['true_recall']),
            tight={r0:bool(gain and (tight_lower(z['released_class_histogram'],v,r0).min()>0 if r0==0 else tight_lower(z['released_class_histogram'],v,r0).min()>=r0)) for r0 in R0},
            recorded={r0:bool(gain and min(lin['bounds']['recall_lower'])>0) if r0==0 else bool(gain and min(lin['bounds']['recall_lower'])>=r0) for r0 in R0}))
    ladder=json.loads((EAB/'analysis/threshold-policies.json').read_text())['policies'];out={}
    for e in [.1,1.,3.]:
        R=[x for x in rows if x['e']==e];res={}
        for r0 in R0:
            for kind in ['recorded','tight']:
                acc=np.array([x[kind][r0] for x in R]);util=np.array([x['cand'] if a else x['pub'] for x,a in zip(R,acc)])
                fa=int(sum(a and not (x['true_gain']>0 and x['true_recall'].min()>=r0 and x['true_recall'].min()>0) for x,a in zip(R,acc)))
                res[f'recall>={r0:g} {kind}']=dict(accept=100*float(acc.mean()),policy_test=100*float(util.mean()),false_accepts=fa)
            k=f'recall>={r0}';assert abs(res[f'recall>={r0:g} recorded']['accept']-ladder[str(e)][k]['accept'])<1e-9,(e,r0)
        out[str(e)]=res
    (ROOT/'analysis/tight-bands.json').write_text(json.dumps(dict(by_epsilon=out,alpha=ALPHA,new_private_queries=0,new_privacy_budget=0,
        scope='one-sided lower class bands on the recorded protected histograms; recorded rows reproduce tab-verify-ladder'),indent=2)+'\n')
    for e,res in out.items():print(e,{k:(round(v['accept'],1),round(v['policy_test'],1),v['false_accepts']) for k,v in res.items()})

if __name__=='__main__':main()
