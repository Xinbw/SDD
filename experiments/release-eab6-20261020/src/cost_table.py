"""Cost comparison on ONE evaluation subset (label draws 50-99), with acceptance and selected shares.

Post-processing only. Sources:
  fixed / tight : the 3000 recorded protected validation outputs (release-value-20261019), recorded
                  linear bands and the one-sided bands of release-eab5-20261020/tight_bands.py;
  optimized     : the pre-registered allocation study (release-allocation-20261019): naming share phi
                  chosen on draws 0-49 by unconditional test utility (gain target: phi in .1..9, no
                  histogram; class targets: phi in .1..8 with the gain release taking psi=.25 of the
                  verification budget), against its re-released fixed 50/25/25 reference; both on 50-99;
  disjoint      : the source-disjoint cohorts (release-disjoint-20261020), same rules.
The optimized utilities are asserted equal to allocation-analysis.json.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;VALUE=EXP/'release-value-20261019';ALLOC=EXP/'release-allocation-20261019';DJ=EXP/'release-disjoint-20261020'
sys.path.insert(0,str(EXP/'release-eab5-20261020/src'));import tight_bands as tb
B=set(range(50,100));EPS=['0.1','1.0','3.0']
TARGETS=[('gain>0',0.,'gain','gain_target_accept',('reference_gain_accept',.5)),
         ('all recalls>0',0.,'class r0=0.0 psi=0.25','class_accept_psi0.25_r0.0',('class_accept_psi0.5_r0.0',.5)),
         ('all recalls>=.50',.5,'class r0=0.5 psi=0.25','class_accept_psi0.25_r0.5',('class_accept_psi0.5_r0.5',.5))]

def recorded():
    rows=[]
    for p in sorted((VALUE/'results/validation').glob('*.json')):
        r=json.loads(p.read_text())
        if r['draw'] not in B:continue
        z=np.load(VALUE/r['diagnostic_file']);lin=json.loads((VALUE/'analysis/linear-validation'/p.name).read_text());d=r['research_diagnostics'];v=r['privacy_classes']['residual_std']
        gain=bool(r['candidate_uses_private'] and r['protected_validation']['gain_lower']>0);rl=min(lin['bounds']['recall_lower'])
        row=dict(e=f"{r['epsilon_total']}",cand=d['candidate_test']['accuracy'],pub=d['public_test']['accuracy'],gain=gain)
        for r0 in [0.,.5]:
            row[f'fixed{r0}']=gain and (rl>0 if r0==0 else rl>=r0)
            t=tb.tight_lower(z['released_class_histogram'],v,r0).min();row[f'tight{r0}']=gain and (t>0 if r0==0 else t>=r0)
        rows.append(row)
    return rows

def summarize(cases,flag):
    a=np.array([c[flag] for c in cases]);u=np.array([c['cand'] if x else c['pub'] for c,x in zip(cases,a)]);return dict(accept=100*float(a.mean()),utility=100*float(u.mean()))

def allocation():
    rows=[r for s in sorted((ALLOC/'results').glob('alloc_s*.json')) for r in json.loads(s.read_text())['rows']]
    def frame(e,phi,field,draws):
        sel=[r for r in rows if r['epsilon']==e and abs(r['phi_naming']-phi)<1e-9 and r['draw'] in draws and field in r];by={}
        for r in sel:by.setdefault(r['seed'],[]).append((r['candidate_test'] if r.get(field) else r['public_test'],float(bool(r[field]))))
        s=sorted(by);return np.array([np.mean([x[0] for x in by[k]]) for k in s]),np.array([np.mean([x[1] for x in by[k]]) for k in s])
    return frame

def main():
    R=recorded();frame=allocation();an=json.loads((ALLOC/'results/allocation-analysis.json').read_text())['targets']
    dj=[json.loads(p.read_text()) for p in sorted((DJ/'results').glob('*.json'))];dj=[r for r in dj if r['draw'] in B]
    out={}
    for name,r0,akey,field,(rfield,rphi) in TARGETS:
        res={}
        for e in EPS:
            cases=[c for c in R if c['e']==e];flag='gain' if name=='gain>0' else None
            fixed=summarize(cases,'gain' if name=='gain>0' else f'fixed{r0}');tight=fixed if name=='gain>0' else summarize(cases,f'tight{r0}')
            x=an[akey][e];phi=x['selected_phi_naming'];us,as_=frame(float(e),phi,field,B);ur,ar=frame(float(e),rphi,rfield,B)
            assert abs(100*us.mean()-x['heldout_selected'])<1e-9 and abs(100*ur.mean()-x['heldout_reference'])<1e-9,(name,e)
            curve=list(x['selection_curve_draws0_49'].values());flat=max(curve)-min(curve)<1e-9
            psi=None if name=='gain>0' else .25
            split=[phi,1-phi,0.] if psi is None else [phi,(1-phi)*psi,(1-phi)*(1-psi)]
            D=[r for r in dj if f"{r['epsilon_total']}"==e]
            def dec(r):
                pv=r['protected_validation'];g=r['candidate_uses_private'] and pv['gain_lower']>0
                if name=='gain>0':return g
                rl=min(pv['recall_lower']);return g and (rl>0 if r0==0 else rl>=r0)
            da=np.array([dec(r) for r in D]);du=np.array([r['research_diagnostics']['candidate_test' if a else 'public_test']['accuracy'] for r,a in zip(D,da)])
            res[e]=dict(fixed=fixed,tight=tight,optimized_reference=dict(accept=100*float(ar.mean()),utility=100*float(ur.mean())),
                optimized=dict(accept=100*float(as_.mean()),utility=100*float(us.mean()),phi_naming=phi,split_naming_gain_classes=split,flat_curve=flat),
                disjoint=dict(accept=100*float(da.mean()),utility=100*float(du.mean())))
        out[name]=res
    (ROOT/'analysis/cost-table.json').write_text(json.dumps(dict(by_target=out,draws='50-99',
        grids=dict(gain='phi_naming in {.1,...,.9}; no class histogram',classes='phi_naming in {.1,...,.8}; gain share of verification psi=.25 (pre-registered target; psi=.5 analysed separately in the artifact)'),
        criterion='unconditional test utility on draws 0-49 (pre-registered)',new_private_queries=0,new_privacy_budget=0),indent=2)+'\n')
    for n,res in out.items():
        for e,x in res.items():print(n,e,{k:{a:round(b,1) if isinstance(b,float) else b for a,b in v.items()} for k,v in x.items()})

if __name__=='__main__':main()
