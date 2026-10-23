"""Balanced-accuracy verification control on the 6000 verify-teacher cases (no query, no budget).

Uses only the already released C x 4 class histogram. Balanced held-out gain
  B = (1/C) sum_c (h_c,10 - h_c,01) / m_c ,  m_c = sum_ab h_c,ab.
Simultaneous one-sided events: C lower bounds on t_c = h_c,10 - h_c,01 (proxy sqrt2 v, rounding 2/s)
and 2C two-sided bounds on m_c (proxy 2 v, rounding 4/s); z = sqrt(2 log(3C/alpha)), alpha=.025.
Lower B_lo = (1/C) sum_c t_lo / (m_up if t_lo >= 0 else m_lo), with m_lo floored at 1e-12 (t_lo<0 and
m_lo=0 gives -inf, i.e. no certificate). Policy: candidate uses private votes and B_lo > 0.
Test labels only score; this is an additional declared rule, not the pre-registered one.
"""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];VT=ROOT.parent/'release-verify-teacher-20261020'
S=1<<16;ALPHA=.025

def balanced_lower(h,v,C):
    h=np.asarray(h).reshape(C,4);t=h[:,2]-h[:,1];m=h.sum(1);z=math.sqrt(2*math.log(3*C/ALPHA))
    tl=t-(math.sqrt(2)*v*z+2/S);mu=m+2*v*z+4/S;ml=m-2*v*z-4/S
    terms=np.where(tl>=0,tl/np.maximum(mu,1e-12),np.where(ml>0,tl/np.maximum(ml,1e-12),-np.inf))
    return float(terms.mean())

def main():
    out={}
    for task in ['mnist','cifar10','web']:
        for e in [.1,1.]:
            R=[json.loads(p.read_text()) for p in sorted((VT/'results').glob(f'{task}_*_e{e:g}_d*.json'))];R=[r for r in R if r['draw']>=50]
            C={'web':6}.get(task,10);priv=[r for r in R if r['candidate_uses_private']]
            acc=[r for r in priv if balanced_lower(r['protected_validation']['released_class_histogram'],r['privacy']['classes']['residual_std'],C)>0]
            ids={id(r) for r in acc}
            def util(metric,mode):
                return float(np.mean([(r['test']['gated'] if (id(r) in ids if mode=='balanced' else r['decisions']['gated_gain_verified']) else r['test']['public'])[metric] for r in R]))
            bal_diff=[r['test']['gated']['balanced_accuracy']-r['test']['public']['balanced_accuracy'] for r in acc]
            out[f'{task}/{e:g}']=dict(cases=len(R),private_candidate_cases=len(priv),balanced_acceptance=len(acc)/len(priv) if priv else None,
                accuracy_acceptance=float(np.mean([r['decisions']['gated_gain_verified'] for r in priv])) if priv else None,
                accepted_test_balanced_difference=float(np.mean(bal_diff)) if acc else None,accepted_share_balanced_positive=float(np.mean(np.array(bal_diff)>0)) if acc else None,
                utility_balanced={'public':float(np.mean([r['test']['public']['balanced_accuracy'] for r in R])),'loo':float(np.mean([r['test']['loo']['balanced_accuracy'] for r in R])),
                    'gated':float(np.mean([r['test']['gated']['balanced_accuracy'] for r in R])),'accuracy_verified':util('balanced_accuracy','accuracy'),'balanced_verified':util('balanced_accuracy','balanced')},
                utility_accuracy={'public':float(np.mean([r['test']['public']['accuracy'] for r in R])),'loo':float(np.mean([r['test']['loo']['accuracy'] for r in R])),
                    'gated':float(np.mean([r['test']['gated']['accuracy'] for r in R])),'accuracy_verified':util('accuracy','accuracy'),'balanced_verified':util('accuracy','balanced')})
            x=out[f'{task}/{e:g}'];print(task,e,'acc-accept %.3f bal-accept %s'%(x['accuracy_acceptance'] or 0,x['balanced_acceptance']),'| bal util',{k:round(100*v,1) for k,v in x['utility_balanced'].items()},'| acc util',{k:round(100*v,1) for k,v in x['utility_accuracy'].items()})
    (ROOT/'analysis/balanced-verification.json').write_text(json.dumps(dict(by_setting=out,draws='50-99',alpha=ALPHA,new_private_queries=0,new_privacy_budget=0,
        status='additional declared rule on released histograms; not part of the pre-registered design'),indent=2)+'\n')

if __name__=='__main__':main()
