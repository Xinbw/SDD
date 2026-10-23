"""Pre-specified analysis of VERIFY-TEACHER-DESIGN.json (primary: epsilon=1, label draws 50-99).

Post-processing of the stored case records; no new query. Bootstrap: crossed over label draws x
cohort seeds, 2000 replicates, seed 20261022 (same estimator as export_frontier.crossed).
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'VERIFY-TEACHER-DESIGN.json').read_text())['design']
TASKS=[t['dataset'] for t in D['tasks']];SEEDS=D['cohorts']['seeds'];PRIMARY=list(range(50,100));ALL=list(range(100))
POLICIES=['public','loo','gated','gated_gain_verified','gated_class_verified']

def crossed(a):
    a=np.asarray(a,dtype=float);l,s=a.shape;rng=np.random.default_rng(20261022)
    left=rng.multinomial(l,np.full(l,1/l),size=2000)/l;right=rng.multinomial(s,np.full(s,1/s),size=2000)/s
    b=np.einsum('bi,ij,bj->b',left,a,right)
    return dict(mean=float(a.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)))

def load():
    R={}
    for p in (ROOT/'results').glob('*.json'):
        r=json.loads(p.read_text());R[(r['task'],r['epsilon_total'],r['draw'],r['seed'])]=r
    assert len(R)==sum(len(SEEDS[t]) for t in TASKS)*len(D['epsilon_total'])*100,len(R)
    return R

def summary(R,task,e,draws):
    seeds=SEEDS[task];metric='balanced_accuracy' if task=='web' else 'accuracy';cases=[R[(task,e,d,s)] for d in draws for s in seeds]
    priv=[r for r in cases if r['candidate_uses_private']];acc=[r['decisions']['gated_gain_verified'] for r in priv]
    cls=[r['decisions']['gated_class_verified'] for r in priv]
    rej=[r for r in priv if not r['decisions']['gated_gain_verified']]
    M={p:np.array([[R[(task,e,d,s)]['test'][p][metric] for s in seeds] for d in draws]) for p in POLICIES}
    A={p:np.array([[R[(task,e,d,s)]['test'][p]['accuracy'] for s in seeds] for d in draws]) for p in POLICIES}
    out=dict(metric=metric,cases=len(cases),private_candidate_cases=len(priv),gate_fires=float(np.mean([r['gate_fires'] for r in cases])),
        gain_acceptance=float(np.mean(acc)) if priv else None,class_acceptance=float(np.mean(cls)) if priv else None,
        false_gain_accepts=int(sum(r['research_diagnostics']['false_gain_accept'] for r in cases)),
        gain_bound_violations=int(sum(not r['research_diagnostics']['gain_lower_valid'] for r in cases)),
        mean_clean_client_gain_private=float(np.mean([r['research_diagnostics']['clean_client_mean_gain'] for r in priv])) if priv else None,
        rejections=len(rej),rejections_clean_gain_nonpositive=int(sum(r['research_diagnostics']['clean_client_mean_gain']<=0 for r in rej)),
        rejections_clean_gain_positive=int(sum(r['research_diagnostics']['clean_client_mean_gain']>0 for r in rej)),
        mean_gain_band_radius=float(np.mean([r['protected_validation']['released_gain']-r['protected_validation']['gain_lower'] for r in cases])),
        utility={p:float(M[p].mean()) for p in POLICIES},utility_accuracy={p:float(A[p].mean()) for p in POLICIES},
        versus_public={p:crossed(M[p]-M['public']) for p in POLICIES if p!='public'},
        versus_loo={p:crossed(M[p]-M['loo']) for p in ['gated','gated_gain_verified','gated_class_verified']},
        clean_center_test=float(np.mean([r['research_diagnostics']['clean_center_test'][metric] for r in cases])))
    return out

def main():
    R=load();res={f'{t}/{e:g}/{w}':summary(R,t,e,dr) for t in TASKS for e in D['epsilon_total'] for w,dr in [('primary',PRIMARY),('all',ALL)]}
    m=res['mnist/1/primary'];u=res['web/1/primary'];c=res['cifar10/1/primary']
    C1=m['gain_acceptance']>=.5;C2=u['gain_acceptance']<=.1
    predictions={
        'C1 holds (MNIST acceptance >= 50%)':C1,'C2 holds (Usenet acceptance <= 10%)':C2,
        'CIFAR-10k acceptance >= 50%':c['gain_acceptance']>=.5,
        'gated_gain_verified - LOO > 0 on MNIST':m['versus_loo']['gated_gain_verified']['mean']>0,
        'gated_gain_verified - LOO >= 0 on Usenet':u['versus_loo']['gated_gain_verified']['mean']>=0,
        'gated above LOO on MNIST':m['versus_loo']['gated']['mean']>0,'gated below LOO on Usenet':u['versus_loo']['gated']['mean']<0,
        'gate closed at epsilon=.1 on all tasks':all(res[f'{t}/0.1/all']['gate_fires']==0 for t in TASKS)}
    out=dict(design_sha256=__import__('hashlib').sha256((ROOT/'VERIFY-TEACHER-DESIGN.json').read_bytes()).hexdigest(),
        primary=dict(C1=C1,C2=C2,success=bool(C1 and C2),mnist_acceptance=m['gain_acceptance'],usenet_acceptance=u['gain_acceptance']),
        predictions=predictions,results=res,new_private_queries_beyond_design=0)
    (ROOT/'analysis/verify-teacher-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(primary=out['primary'],predictions=predictions),indent=1))
    for k,v in res.items():
        print(k.ljust(22),'priv %4d acc %s cls %s rej %d (clean<=0 %d, >0 %d) FA %d viol %d radius %.3f'%(v['private_candidate_cases'],
            'NA' if v['gain_acceptance'] is None else '%.3f'%v['gain_acceptance'],'NA' if v['class_acceptance'] is None else '%.3f'%v['class_acceptance'],
            v['rejections'],v['rejections_clean_gain_nonpositive'],v['rejections_clean_gain_positive'],v['false_gain_accepts'],v['gain_bound_violations'],v['mean_gain_band_radius']),
            '| util',{p:round(100*x,1) for p,x in v['utility'].items()},'| vsLOO',{p:'%+.1f[%+.1f,%+.1f]'%(100*x['mean'],100*x['ci_low'],100*x['ci_high']) for p,x in v['versus_loo'].items()})

if __name__=='__main__':main()
