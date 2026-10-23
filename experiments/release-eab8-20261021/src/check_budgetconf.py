"""Re-check the budget-split confirmation study from its records (CPU, no private query).

1. Every certificate in every development/confirmation record is recomputed from its declared
   parameters and must match; every deployment's composed rho must not exceed the eps target.
2. The frozen SELECTION.json is re-derived from the development records alone (protected decisions and
   validation accuracies) and must match; its development-results hashes must match the records.
3. Every protected decision is re-derived from the released bounds stored in the record.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
EXP=Path(__file__).resolve().parents[2];BC=EXP/'release-budgetconf-20261021'
sys.path.insert(0,str(EXP/'release-frontier-20261019/src'));import privacy
D=json.loads((BC/'BUDGETCONF-DESIGN.json').read_text())['design'];SEL=json.loads((BC/'SELECTION.json').read_text())
N=D['clients'];C=10;S=1<<16;DELTA=D['delta_total']

def configs():
    out=[('fixed',.5,.25,.25)]+[(f'gain{p:g}',p,1-p,0.) for p in D['configurations']['gain_grid']['phi_naming']]
    return out+[(f'class{p:g}',p,(1-p)/4,3*(1-p)/4) for p in D['configurations']['class_grid']['phi_naming']]

def main():
    files=sorted((BC/'results').glob('*.json'));assert len(files)==20;cache={};n_cert=0;n_rows=0
    for f in files:
        rec=json.loads(f.read_text())
        for e in D['epsilon_total']:
            for name,fn,fg,fc in configs():
                c=rec['certificates'][f'{e:g}/{name}']
                k=(e,name)
                if k not in cache:
                    cn=privacy.certificate(e,DELTA,1,N,clusters=C,dimension=C*C,rho_fraction=fn)
                    cg=privacy.certificate(e,DELTA,1,N,integer_sensitivity=S,dimension=1,rho_fraction=fg)
                    cc=privacy.certificate(e,DELTA,1,N,integer_sensitivity_squared=2*S*S,dimension=4*C,rho_fraction=fc) if fc>0 else None
                    cache[k]=json.loads(json.dumps(dict(naming=cn,gain=cg,classes=cc)))
                assert {x:c[x] for x in ['naming','gain','classes']}==cache[k],(f.name,k)
                assert c['total_rho_upper']<=privacy.rho_from_epsilon(e,DELTA,'zcdp')*(1+1e-12);n_cert+=1
        for r in rec['rows']:
            priv=r['private'];gl=r['gain_lower'];a=r['accept']
            if 'gain>0' in a:assert a['gain>0']==(priv and gl>0)
            if 'min_recall_lower' in r:
                rl=r['min_recall_lower'];assert a['all recalls>0']==(priv and gl>0 and rl>0) and a['all recalls>=.50']==(priv and gl>0 and rl>=.5)
            n_rows+=1
    dev=[f for f in files if f.name.startswith('dev_')]
    assert SEL['development_results_sha256']=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dev}
    rows=[dict(e=r['e'],config=r['config'],accept=r['accept'],cand=r['candidate_eval']['accuracy'],pub=r['public_eval']['accuracy']) for f in dev for r in json.loads(f.read_text())['rows']]
    pick=lambda cu:sorted([p for p,v in cu.items() if v>=max(cu.values())-1e-12],key=lambda p:(abs(p-.5),p))[0]
    for key,s in SEL['selection'].items():
        t,e=key.split('|');fam=s['family'];grid=D['configurations']['gain_grid' if fam=='gain' else 'class_grid']['phi_naming'];util={};acc={}
        for p in grid:
            R=[r for r in rows if r['e']==float(e) and r['config']==f'{fam}{p:g}'];x=np.array([r['accept'][t] for r in R])
            util[p]=100*float(np.mean([r['cand'] if y else r['pub'] for r,y in zip(R,x)]));acc[p]=100*float(x.mean())
        assert pick(util)==s['primary_phi'] and pick(acc)==s['secondary_phi'],key
        assert all(abs(util[p]-s['validation_utility_curve'][f'{p:g}'])<1e-9 for p in grid),key
    print(f'PASS: budget-split confirmation: {n_cert} deployment certificates recomputed and composed within budget, {n_rows} protected decisions re-derived, frozen selection re-derived from development records only')
if __name__=='__main__':main()
