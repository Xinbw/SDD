"""Out-of-sample calibration of the frozen planning calculator on the 6000 verify-teacher cases.

release-eab3-20261020/src/planning.py (accept_probability) was written and run before the
verify-teacher design was frozen; it is imported unchanged. The new cases differ in validation
clients (10000 constructed / 1249 natural Usenet authors), classes (10 / 6) and cohorts.
Inputs per case: its clean client-mean gain, the clean class masses of its validation cohort
(rebuilt deterministically from the frozen cohort construction) and clean candidate recalls.
This checks the noise/band calibration across settings; effect sizes remain inputs.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;VT=EXP/'release-verify-teacher-20261020'
sys.path.insert(0,str(EXP/'release-eab3-20261020/src'));sys.path.insert(0,str(VT/'src'))
import planning
import run as vt

def masses(task,seed,data,C):
    _,(vm,vs),_=vt.cohorts(task,data,seed);mask=np.arange(vm.shape[1])[None,:]<vs[:,None]
    y=data['y_private'][vm];per=np.stack([((y==c)&mask).sum(1)/vs for c in range(C)],1);return per.mean(0)

def main():
    out={}
    for task in ['mnist','cifar10','web']:
        C={'web':6}.get(task,10);data=vt.base.load_data(task);M={s:masses(task,s,data,C) for s in vt.SEEDS[task]}
        for e in [.1,1.]:
            R=[json.loads(p.read_text()) for p in sorted((VT/'results').glob(f'{task}_*_e{e:g}_d*.json'))]
            nv=vt.TASKS[task]['validation_clients'];pg=[];pc=[];rg=[];rc=[]
            for r in R:
                if not r['candidate_uses_private']:continue
                m=M[r['seed']];g=r['research_diagnostics']['clean_client_mean_gain'];corr=np.array(r['research_diagnostics']['true_recall'])*m
                pg.append(planning.accept_probability('gain',0.,e,nv,C,g,m,corr,None));pc.append(planning.accept_probability('recall',0.,e,nv,C,g,m,corr,None))
                rg.append(r['decisions']['gated_gain_verified']);rc.append(r['decisions']['gated_class_verified'])
            out[f'{task}/{e:g}']=dict(private_candidate_cases=len(pg),gain=dict(predicted=100*float(np.mean(pg)),recorded=100*float(np.mean(rg))),
                recall=dict(predicted=100*float(np.mean(pc)),recorded=100*float(np.mean(rc))),class_masses_min=float(min(M[s].min() for s in M)),class_masses_max=float(max(M[s].max() for s in M)))
            print(task,e,out[f'{task}/{e:g}'],flush=True)
    gap=max(abs(v[k]['predicted']-v[k]['recorded']) for v in out.values() for k in ['gain','recall'])
    (ROOT/'analysis/planning-out-of-sample.json').write_text(json.dumps(dict(by_setting=out,max_gap_points=gap,cells=2*len(out),
        planning_source_sha256=__import__('hashlib').sha256((EXP/'release-eab3-20261020/src/planning.py').read_bytes()).hexdigest(),
        scope='calibration of the frozen closed-form acceptance model on new settings, given clean effect sizes'),indent=2)+'\n')
    print('max gap',gap)

if __name__=='__main__':main()
