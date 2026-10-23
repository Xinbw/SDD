"""Pre-registered analysis of the allocation feasibility study (see ALLOCATION-DESIGN.json)."""
import json
import numpy as np
from scipy import stats
from alloc_common import ROOT,save
rows=[r for s in sorted((ROOT/'results').glob('alloc_s*.json')) for r in json.loads(s.read_text())['rows']]
EPS=[.1,.3,1.,3.];PHI=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
def util(r,field):return r['candidate_test'] if r.get(field) else r['public_test']
def frame(e,phi,field,draws):
    sel=[r for r in rows if r['epsilon']==e and abs(r['phi_naming']-phi)<1e-9 and r['draw'] in draws and field in r]
    by={}
    for r in sel:by.setdefault(r['seed'],[]).append((util(r,field),float(bool(r[field]))))
    seeds=sorted(by);u=np.array([np.mean([x[0] for x in by[s]]) for s in seeds]);a=np.array([np.mean([x[1] for x in by[s]]) for s in seeds])
    return u,a
A=set(range(50));B=set(range(50,100))
targets={'gain':('gain_target_accept',PHI,('reference_gain_accept',.5))}
for r0 in ['0.0','0.5']:
    for psi in ['0.25','0.5']:targets[f'class r0={r0} psi={psi}']=(f'class_accept_psi{psi}_r{r0}',PHI[:-1],(f'class_accept_psi0.5_r{r0}',.5))
report={'sanity_reference_all_draws':{},'targets':{}}
for e in EPS:
    ug,ag=frame(e,.5,'reference_gain_accept',A|B);uc,ac=frame(e,.5,'class_accept_psi0.5_r0.0',A|B)
    report['sanity_reference_all_draws'][str(e)]=dict(gain_accept=100*ag.mean(),gain_utility=100*ug.mean(),class_accept=100*ac.mean(),class_utility=100*uc.mean())
for name,(field,grid,(rfield,rphi)) in targets.items():
    out={}
    for e in EPS:
        curve={str(p):100*frame(e,p,field,A)[0].mean() for p in grid}
        best=max(grid,key=lambda p:frame(e,p,field,A)[0].mean())
        ub,ab=frame(e,best,field,B);ur,ar=frame(e,rphi,rfield,B);d=100*(ub-ur)
        lo,hi=stats.t.interval(.95,len(d)-1,loc=d.mean(),scale=stats.sem(d)) if d.std()>0 else (d.mean(),d.mean())
        full={str(p):dict(utility=100*frame(e,p,field,A|B)[0].mean(),accept=100*frame(e,p,field,A|B)[1].mean()) for p in grid}
        out[str(e)]=dict(selected_phi_naming=best,interior=best not in (grid[0],grid[-1]),heldout_selected=100*ub.mean(),heldout_reference=100*ur.mean(),
            improvement=d.mean(),ci95=[lo,hi],selection_curve_draws0_49=curve,all_draw_curve=full)
    report['targets'][name]=out
def go(name):
    o=report['targets'][name];wins=[e for e,x in o.items() if x['improvement']>=1 and x['ci95'][0]>0]
    return dict(passing_epsilons=wins,interior_any=any(x['interior'] for x in o.values()),go=len(wins)>=2 and any(x['interior'] for x in o.values()))
report['go_criterion']={k:go(k) for k in ['gain','class r0=0.0 psi=0.25','class r0=0.0 psi=0.5']}
report['go']=any(v['go'] for v in report['go_criterion'].values())
save(ROOT/'results/allocation-analysis.json',report)
print(json.dumps(report['sanity_reference_all_draws'],indent=1))
for name,o in report['targets'].items():
    print('==',name)
    for e,x in o.items():print(f"  eps={e:>4} best_phi={x['selected_phi_naming']} interior={x['interior']} sel={x['heldout_selected']:.2f} ref={x['heldout_reference']:.2f} d={x['improvement']:+.2f} CI=[{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]")
print(json.dumps(report['go_criterion'],indent=1),'GO=',report['go'])
