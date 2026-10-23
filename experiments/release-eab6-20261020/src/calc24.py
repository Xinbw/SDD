"""Closed-form acceptance check on ALL 24 cells of the verification ladder (8 targets x 3 budgets).

Uses the unchanged accept_probability of release-eab3-20261020/src/planning.py (which checked 18
cells: it omitted gain>0 and gain>=.20). Inputs per case: clean client-mean gain, clean class masses,
clean correct masses and clean worst-class differences of the recorded held-out cohort.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;VALUE=EXP/'release-value-20261019';EAB=EXP/'release-eab-20261020'
sys.path.insert(0,str(EXP/'release-eab3-20261020/src'));import planning
TARGETS=[('gain>=0.0','gain',0.),('gain>=0.1','gain',.1),('gain>=0.2','gain',.2),('recall>=0.0','recall',0.),('recall>=0.25','recall',.25),
         ('recall>=0.5','recall',.5),('nondeg eta=0.25','nondeg',.25),('nondeg eta=0.5','nondeg',.5)]

def main():
    C=10;cases=[]
    for p in sorted((VALUE/'results/validation').glob('*.json')):
        r=json.loads(p.read_text());z=np.load(VALUE/r['diagnostic_file']);h=z['client_joint_counts'].sum(0).astype(float);h=h/h.sum()
        mass=h.sum(1);correct=h[:,2]+h[:,3];diff=np.divide(h[:,2]-h[:,1],mass,out=np.zeros(C),where=mass>0)
        cases.append(dict(e=r['epsilon_total'],n=r['n'],private=r['candidate_uses_private'],gain=r['research_diagnostics']['true_gain'],mass=mass,correct=correct,diff=diff))
    ladder=json.loads((EAB/'analysis/threshold-policies.json').read_text())['policies'];out={}
    for key,kind,level in TARGETS:
        row={}
        for e in [.1,1.,3.]:
            R=[x for x in cases if x['e']==e]
            pred=[planning.accept_probability(kind,level,e,x['n'],C,x['gain'],x['mass'],x['correct'],x['diff']) if x['private'] else 0. for x in R]
            row[f'{e:g}']=dict(predicted=100*float(np.mean(pred)),recorded=ladder[str(e)][key]['accept'])
        out[key]=row
    gap=max(abs(c['predicted']-c['recorded']) for r in out.values() for c in r.values())
    (ROOT/'analysis/calculator-24.json').write_text(json.dumps(dict(cells=24,max_gap_points=gap,by_target=out,
        planning_source_sha256=__import__('hashlib').sha256((EXP/'release-eab3-20261020/src/planning.py').read_bytes()).hexdigest()),indent=2)+'\n')
    for k,r in out.items():print(k,{e:(round(c['predicted'],1),round(c['recorded'],1)) for e,c in r.items()})
    print('max gap',round(gap,2))

if __name__=='__main__':main()
