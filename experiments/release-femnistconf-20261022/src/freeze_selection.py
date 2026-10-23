"""Freeze the FEMNIST budget split per target and eps from the DEVELOPMENT phase only (writes SELECTION.json; same rule as release-budgetconf).

Reads, per development case, only the protected decisions and the validation accuracy of the candidate
and comparator. Clean statistics (true_gain, true_min_recall) are not read. Refuses to run if any
confirmation result exists.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=json.loads((ROOT/'FEMNISTCONF-DESIGN.json').read_text())['design']
GRID=dict(gain=D['configurations']['gain_grid']['phi_naming'],cls=D['configurations']['class_grid']['phi_naming'])
TARGETS=[('gain>0','gain'),('all recalls>0','class'),('all recalls>=.50','class')]

def pick(curve):
    best=max(curve.values());tied=[p for p,v in curve.items() if v>=best-1e-12]
    return sorted(tied,key=lambda p:(abs(p-.5),p))[0]

def main():
    assert not list((ROOT/'results').glob('confirm_*.json')),'confirmation results exist'
    files=sorted((ROOT/'results').glob('dev_s*.json'));assert len(files)==len(D['development_cohort_seeds'])
    rows=[]
    for f in files:
        for r in json.loads(f.read_text())['rows']:
            rows.append(dict(e=r['e'],config=r['config'],accept=r['accept'],cand=r['candidate_eval']['accuracy'],pub=r['public_eval']['accuracy']))
    out={}
    for t,fam in TARGETS:
        for e in D['epsilon_total']:
            util={};acc={}
            for p in GRID['gain' if fam=='gain' else 'cls']:
                R=[r for r in rows if r['e']==e and r['config']==f'{fam}{p:g}']
                assert len(R)==len(files)*50
                a=np.array([r['accept'][t] for r in R]);u=np.array([r['cand'] if x else r['pub'] for r,x in zip(R,a)])
                util[p]=100*float(u.mean());acc[p]=100*float(a.mean())
            F=[r for r in rows if r['e']==e and r['config']=='fixed'];af=np.array([r['accept'][t] for r in F])
            uf=100*float(np.mean([r['cand'] if x else r['pub'] for r,x in zip(F,af)]))
            out[f'{t}|{e:g}']=dict(target=t,epsilon=e,family=fam,primary_phi=pick(util),secondary_phi=pick(acc),
                validation_utility_curve={f'{p:g}':v for p,v in util.items()},development_accept_curve={f'{p:g}':v for p,v in acc.items()},
                fixed_development=dict(accept=100*float(af.mean()),validation_utility=uf))
    rec=dict(selection=out,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        development_results_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        select_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_read='protected decisions; validation accuracy of candidate and comparator',inputs_not_read='true_gain, true_min_recall, any test label, any confirmation record')
    path=ROOT/'SELECTION.json';assert not path.exists(),'SELECTION.json already frozen'
    path.write_text(json.dumps(rec,indent=2)+'\n')
    for k,v in out.items():print(k,'primary',v['primary_phi'],'secondary',v['secondary_phi'],'fixed',{a:round(b,1) for a,b in v['fixed_development'].items()},'best util %.1f acc %.1f'%(v['validation_utility_curve'][f"{v['primary_phi']:g}"],v['development_accept_curve'][f"{v['secondary_phi']:g}"]))
if __name__=='__main__':main()
