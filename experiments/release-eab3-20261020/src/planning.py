"""Closed-form verification planning map (no private data, no query, no budget).

The validation bands of release-value-20261019 / release-eab-20261020 are closed form in the
calibrated noise std v(eps, n, C) of the two validation reports (rho fractions .25/.25 of one total
budget; the naming fraction .5 is not used by verification). Each released statistic is the clean
statistic plus independent Gaussian noise of known std, so the acceptance probability is closed form:
  gain >= gamma : P = Phi((g - 2 v_g sqrt(2 log(1/a)) - 1/s - gamma) / (2 v_g))
  recalls >= r0 : times prod_c Phi((r_c m_c - r_n - r0 (m_c + r_d)) / (v_h sqrt(2(1-r0)^2 + 2 r0^2)));
                  r_n = sqrt2 v_h z + 2/s, r_d = 2 v_h z + 4/s, z = sqrt(2 log(4C/a))
  non-degr. eta : times prod_c Phi(((d_c + eta) m_c - v_h sqrt(2+4 eta^2) sqrt(2 log(C/a)) - (2+2 eta)/s) / (v_h sqrt(2+4 eta^2)))
(the recall>0 rule uses r0 = 0; the per-class noises are independent coordinates, the two reports
independent releases). A target is called feasible at probability 1/2.
Check: applied case by case to the CLEAN held-out statistics of the 3000 recorded validation cases
(offline diagnostics), it is compared with the 3000 recorded protected decisions.
Map: minimum total epsilon over n and C at fixed effect sizes (median clean values of the private
candidates at eps=3, class mass 1/C). Effect sizes at other tasks are an input, not a claim.
"""
import json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;VALUE=EXP/'release-value-20261019';EAB=EXP/'release-eab-20261020'
sys.path.insert(0,str(EXP/'release-frontier-20261019/src'))
import privacy
S=1<<16;A=.025;DELTA=1e-5;FRAC=.25
TARGETS=[('gain>=0.1','gain',.1),('recall>0','recall',0.),('recall>=0.25','recall',.25),('recall>=0.5','recall',.5),('nondeg eta=0.5','nondeg',.5),('nondeg eta=0.25','nondeg',.25)]
LADDER={'gain>=0.1':'gain>=0.1','recall>0':'recall>=0.0','recall>=0.25':'recall>=0.25','recall>=0.5':'recall>=0.5','nondeg eta=0.5':'nondeg eta=0.5','nondeg eta=0.25':'nondeg eta=0.25'}
N_GRID=np.unique(np.round(np.logspace(2,5,61)).astype(int)).tolist();C_GRID=[2,4,6,8,10,15,20,30,40,62,80,100]

def noise(e,n,C):
    g=privacy.certificate(e,DELTA,1,n,integer_sensitivity=S,dimension=1,rho_fraction=FRAC)['residual_std']
    h=privacy.certificate(e,DELTA,1,n,integer_sensitivity_squared=2*S*S,dimension=4*C,rho_fraction=FRAC)['residual_std']
    return g,h

def Phi(x):return 0.5*(1+np.vectorize(math.erf)(np.asarray(x,float)/math.sqrt(2)))

def accept_probability(kind,level,e,n,C,gain,mass,correct,diff):
    vg,vh=noise(e,n,C);p=float(Phi((gain-2*vg*math.sqrt(2*math.log(1/A))-1/S-(level if kind=='gain' else 0))/(2*vg)))
    if kind=='gain':return p
    mass=np.asarray(mass);correct=np.asarray(correct)
    if kind=='recall':
        z=math.sqrt(2*math.log(4*C/A));rn=math.sqrt(2)*vh*z+2/S;rd=2*vh*z+4/S
        return p*float(np.prod(Phi((correct-rn-level*(mass+rd))/(vh*math.sqrt(2*(1-level)**2+2*level**2)))))
    k=math.sqrt(2+4*level*level)
    return p*float(np.prod(Phi(((np.asarray(diff)+level)*mass-vh*k*math.sqrt(2*math.log(C/A))-(2+2*level)/S)/(vh*k))))

def feasible(*args):return accept_probability(*args)>=.5

def min_epsilon(kind,level,n,C,gain,recall,diff):
    m=np.full(C,1/C);args=(gain,m,recall*m,np.full(C,diff))
    lo,hi=1e-3,1e3
    if not feasible(kind,level,hi,n,C,*args):return None
    for _ in range(60):
        mid=math.sqrt(lo*hi)
        if feasible(kind,level,mid,n,C,*args):hi=mid
        else:lo=mid
    return hi

def main():
    cases=[];C=10
    for p in sorted((VALUE/'results/validation').glob('*.json')):
        r=json.loads(p.read_text());z=np.load(VALUE/r['diagnostic_file']);h=z['client_joint_counts'].sum(0).astype(float);h=h/h.sum()
        mass=h.sum(1);correct=h[:,2]+h[:,3];diff=np.divide(h[:,2]-h[:,1],mass,out=np.zeros(C),where=mass>0)
        cases.append(dict(e=r['epsilon_total'],n=r['n'],private=r['candidate_uses_private'],gain=r['research_diagnostics']['true_gain'],mass=mass,correct=correct,diff=diff,
            recall=np.divide(correct,mass,out=np.zeros(C),where=mass>0)))
    ladder=json.loads((EAB/'analysis/threshold-policies.json').read_text())['policies']
    check={}
    for name,kind,level in TARGETS:
        row={}
        for e in [.1,1.,3.]:
            R=[x for x in cases if x['e']==e];pred=[accept_probability(kind,level,e,x['n'],C,x['gain'],x['mass'],x['correct'],x['diff']) if x['private'] else 0. for x in R]
            row[f'{e:g}']=dict(predicted_accept=100*float(np.mean(pred)),recorded_accept=ladder[f'{e}'][LADDER[name]]['accept'])
        check[name]=row
    P=[x for x in cases if x['e']==3. and x['private']]
    effect=dict(gain=float(np.median([x['gain'] for x in P])),minimum_recall=float(np.median([x['recall'].min() for x in P])),
        minimum_recall_difference=float(np.median([x['diff'].min() for x in P])),source='median clean held-out values of private candidates, CIFAR-10 n=1000, eps=3 (offline)',cases=len(P))
    def curve(name,kind,level,ns,Cs):
        return [min_epsilon(kind,level,n,c,effect['gain'],effect['minimum_recall'],effect['minimum_recall_difference']) for n,c in zip(ns,Cs)]
    by_n={name:dict(C=10,n=N_GRID,epsilon=curve(name,k,l,N_GRID,[10]*len(N_GRID))) for name,k,l in TARGETS}
    by_C={name:dict(n=1000,C=C_GRID,epsilon=curve(name,k,l,[1000]*len(C_GRID),C_GRID)) for name,k,l in TARGETS}
    anchor={name:min_epsilon(k,l,1000,10,effect['gain'],effect['minimum_recall'],effect['minimum_recall_difference']) for name,k,l in TARGETS}
    out=dict(effect_sizes=effect,alpha=A,rho_fractions=dict(naming=.5,gain=FRAC,classes=FRAC),delta=DELTA,anchor_n1000_C10=anchor,
        decision_check=check,by_n=by_n,by_C=by_C,new_private_queries=0,new_privacy_budget=0,
        scope='closed-form acceptance probability (feasible at 1/2) of each publication target; effect sizes are inputs; check uses offline clean held-out statistics')
    (ROOT/'analysis/verification-planning.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(effect=effect,anchor=anchor,check=check),indent=1))

if __name__=='__main__':main()
