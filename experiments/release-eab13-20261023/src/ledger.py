"""EAB round 13 ledger (writes analysis/eab13-ledger.json; post-processing of frozen records, no new query).

1. Pilot resources of the CIFAR budget-split selection: labels, protected releases and, if one pilot
   population answered every grid configuration, the zCDP budget it would accumulate.
2. FEMNIST public reference: strongest public learners on exactly the confirmation label draws (50-99)
   and test split, from the inherited release-frontier public records.
3. Applicability of the discrete-Gaussian sum result (sigma >= 1/2): smallest integer noise scale b over
   every recorded certificate.
"""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
BC=EXP/'release-budgetconf-20261021';FR=EXP/'release-frontier-20261019';FC=EXP/'release-femnistconf-20261022'
KEYS={'sigma_share_integer','honest_clients','dimension','sensitivity_integer_squared'}

def rho(e,delta=1e-5):L=math.log(1/delta);return (math.sqrt(L+e)-math.sqrt(L))**2
def eps(r,delta=1e-5):return r+2*math.sqrt(r*math.log(1/delta))

def pilot():
    D=json.loads((BC/'BUDGETCONF-DESIGN.json').read_text())['design'];A=json.loads((BC/'analysis/budgetconf-analysis.json').read_text())
    draws=10;gain=D['configurations']['gain_grid']['phi_naming'];cls=D['configurations']['class_grid']['phi_naming']
    # one naming release per configuration (shared by its draws), one gain and (class grid) one histogram per draw
    units=sum(p+draws*(1-p) for p in gain)+sum(p+draws*(1-p) for p in cls)
    releases=len(gain)*(1+draws)+len(cls)*(1+2*draws)
    with np.load(FR/'inputs/cifar10.npz') as z:validation=int(len(z['y_validation']))
    return dict(cohorts=1,clients=D['clients'],label_draws=draws,labels_per_draw=10,development_labels=10*draws,validation_labels=validation,
        configurations=len(gain)+len(cls),protected_releases=releases,rho_multiple_of_deployment=units,
        epsilon_if_same_clients={f'{e:g}':eps(units*rho(e)) for e in D['epsilon_total']},
        pilot_subsamples=A['pilot']['all recalls>0|1']['subsamples'])

def femnist_public():
    D=json.loads((FC/'FEMNISTCONF-DESIGN.json').read_text())['design'];acc=[];base={}
    for d in D['label_draws']['confirmation']:
        r=json.loads((FR/'public'/f'femnist62_random_{d}_m62.json').read_text());acc.append(r['public_portfolio']['accuracy'])
        for k,v in r['baselines'].items():base.setdefault(k,[]).append(v['accuracy'])
    best=max(base,key=lambda k:np.mean(base[k]))
    return dict(portfolio=100*float(np.mean(acc)),strongest=dict(name=best,accuracy=100*float(np.mean(base[best]))),draws=len(acc))

def min_b():
    out=[]
    def walk(x):
        if isinstance(x,dict):
            if KEYS<=x.keys():out.append(int(x['sigma_share_integer']))
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    for p in sorted(EXP.rglob('*.json')):
        if 'release-eab13-20261023' in p.parts or p.stat().st_size>200_000_000:continue
        try:walk(json.loads(p.read_text()))
        except ValueError:continue
    return dict(certificates=len(out),min_integer_noise_scale=min(out))

def main():
    res=dict(pilot=pilot(),femnist_public=femnist_public(),noise_scale=min_b())
    (ROOT/'analysis').mkdir(exist_ok=True);(ROOT/'analysis/eab13-ledger.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=1))
if __name__=='__main__':main()
