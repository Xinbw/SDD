"""POST-HOC (not pre-registered) decomposition, written after the primary outcome was known.

Splits private-candidate cases at epsilon=1 (draws 50-99) by the protected gain decision and reports
their test differences against the comparator and their clean held-out gains (offline diagnostics).
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    out={}
    for task in ['mnist','cifar10','web']:
        R=[json.loads(p.read_text()) for p in (ROOT/'results').glob(f'{task}_*_e1_d*.json')]
        R=[r for r in R if r['draw']>=50 and r['candidate_uses_private']];res={}
        for name,flag in [('accepted',True),('rejected',False)]:
            S=[r for r in R if r['decisions']['gated_gain_verified']==flag]
            if not S:res[name]=dict(cases=0);continue
            bal=np.array([r['test']['gated']['balanced_accuracy']-r['test']['public']['balanced_accuracy'] for r in S])
            acc=np.array([r['test']['gated']['accuracy']-r['test']['public']['accuracy'] for r in S])
            res[name]=dict(cases=len(S),test_balanced_difference=float(bal.mean()),share_balanced_positive=float((bal>0).mean()),
                test_accuracy_difference=float(acc.mean()),share_accuracy_positive=float((acc>0).mean()),
                clean_client_mean_gain=float(np.mean([r['research_diagnostics']['clean_client_mean_gain'] for r in S])))
        out[task]=res
    (ROOT/'analysis/post-hoc-decision-split.json').write_text(json.dumps(dict(status='post hoc; not pre-registered',epsilon=1,draws='50-99',by_task=out),indent=2)+'\n')
    print(json.dumps(out,indent=1))
if __name__=='__main__':main()
