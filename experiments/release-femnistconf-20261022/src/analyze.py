"""Analysis of the FEMNIST replication (FEMNISTCONF-DESIGN.json, SELECTION.json, PREDICTIONS.json).

Primary F1 (planner prediction within 5 points of confirmation acceptance in every fixed and frozen cell)
and F2 (false-accept rate <= 5% of accepted cases in every fixed and frozen cell), registered predictions
and descriptive quantities. Crossed bootstrap over 50 draws x 10 cohorts, 2000 replicates, seed 20261027.
"""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=json.loads((ROOT/'FEMNISTCONF-DESIGN.json').read_text())['design']
SEL=json.loads((ROOT/'SELECTION.json').read_text());PRED=json.loads((ROOT/'PREDICTIONS.json').read_text())
TARGETS=[('gain>0','gain',0.),('all recalls>0','class',0.),('all recalls>=.50','class',.5)];EPS=D['epsilon_total']

def load(phase):
    seeds=D[f'{"development" if phase=="dev" else "confirmation"}_cohort_seeds'];R={}
    for j,s in enumerate(seeds):
        for r in json.loads((ROOT/'results'/f'{phase}_s{s}.json').read_text())['rows']:R[(r['e'],r['config'],r['draw'],j)]=r
    return R,len(seeds)

def crossed(a):
    a=np.asarray(a,float);l,s=a.shape;rng=np.random.default_rng(20261027)
    L=rng.multinomial(l,np.full(l,1/l),size=2000)/l;Rr=rng.multinomial(s,np.full(s,1/s),size=2000)/s;b=np.einsum('bi,ij,bj->b',L,a,Rr)
    return dict(mean=float(a.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)))

def valid(r,t,r0):return r['true_gain']>0 and (t=='gain>0' or (r['true_min_recall']>=r0 and r['true_min_recall']>0))

def grid(R,ns,e,config,t,r0,draws):
    acc=np.zeros((len(draws),ns));util=np.zeros_like(acc);fa=0;tail=[];pub=np.zeros_like(acc);truth=[]
    for i,d in enumerate(draws):
        for j in range(ns):
            r=R[(e,config,d,j)];ok=bool(r['accept'][t]);acc[i,j]=ok;truth.append(valid(r,t,r0) and r['private'])
            util[i,j]=100*(r['candidate_eval'] if ok else r['public_eval'])['accuracy'];pub[i,j]=100*r['public_eval']['accuracy']
            fa+=int(ok and not valid(r,t,r0))
            if ok:tail.append(r['candidate_eval']['minimum_recall']<.1)
    return acc,util,dict(accept=100*float(acc.mean()),utility=float(util.mean()),false_accepts=fa,false_accept_rate=100*fa/max(1,acc.sum()),
        published=int(acc.sum()),published_min_recall_below_10pct=100*float(np.mean(tail)) if tail else None,public_fallback=float(pub.mean()),
        target_true_rate=100*float(np.mean(truth)))

def main():
    R,ns=load('confirm');draws=D['label_draws']['confirmation'];out={}
    for t,fam,r0 in TARGETS:
        for e in EPS:
            k=f'{t}|{e:g}';s=SEL['selection'][k];p=PRED['predictions'][k];frozen=f"{fam}{s['primary_phi']:g}"
            af,uf,xf=grid(R,ns,e,'fixed',t,r0,draws);ap,up,xp=grid(R,ns,e,frozen,t,r0,draws);curve={}
            for c in p['predicted_by_config']:
                if c!='fixed':curve[c]=grid(R,ns,e,c,t,r0,draws)[2]
            best=max(curve,key=lambda c:curve[c]['utility'])
            out[k]=dict(target=t,epsilon=e,fixed=dict(**xf,predicted=p['predicted_fixed']),
                frozen=dict(config=frozen,phi=s['primary_phi'],**xp,predicted=p['predicted_frozen'],accept_diff=crossed(ap-af),utility_diff=crossed(up-uf)),
                confirmation_best=dict(config=best,utility=curve[best]['utility'],regret_points=curve[best]['utility']-xp['utility']),
                curve={c:dict(accept=v['accept'],utility=v['utility'],predicted=p['predicted_by_config'][c]) for c,v in curve.items()})
    gaps=[abs(v[x]['accept']-v[x]['predicted']) for v in out.values() for x in ['fixed','frozen']]
    F1=bool(max(gaps)<=5);F2=bool(all(v[x]['false_accept_rate']<=5 for v in out.values() for x in ['fixed','frozen']))
    pred={'F1 holds':F1,'F2 holds':F2,
        'no frozen cell more than 1 point below fixed':bool(all(v['frozen']['utility_diff']['mean']>=-1 for v in out.values())),
        'gain>0 accepted more often than all recalls>0 at every eps (fixed)':bool(all(out[f'gain>0|{e:g}']['fixed']['accept']>out[f'all recalls>0|{e:g}']['fixed']['accept'] for e in EPS)),
        'no class target accepted in more than 5% of cases at eps<=3':bool(all(out[f'{t}|{e:g}'][x]['accept']<=5 for t in ['all recalls>0','all recalls>=.50'] for e in EPS if e<=3 for x in ['fixed','frozen']))}
    res=dict(design_sha256=hashlib.sha256((ROOT/'FEMNISTCONF-DESIGN.json').read_bytes()).hexdigest(),selection_sha256=hashlib.sha256((ROOT/'SELECTION.json').read_bytes()).hexdigest(),
        predictions_sha256=hashlib.sha256((ROOT/'PREDICTIONS.json').read_bytes()).hexdigest(),primary=dict(F1=F1,F2=F2,success=F1 and F2,max_planner_gap_points=float(max(gaps))),
        registered_predictions=pred,cells=out,cases=len(R))
    (ROOT/'analysis').mkdir(exist_ok=True);(ROOT/'analysis/femnistconf-analysis.json').write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(dict(primary=res['primary'],predictions=pred,cases=len(R)),indent=1))
    for k,v in out.items():
        f,p=v['fixed'],v['frozen']
        print('%-22s fixed %5.1f/%5.1f pred %5.1f FA %d true %.1f tail %s | %s %5.1f/%5.1f pred %5.1f FA %d dU %+5.1f [%+.1f,%+.1f] | best %s regret %.2f'%(k,f['accept'],f['utility'],f['predicted'],f['false_accepts'],f['target_true_rate'],
            None if f['published_min_recall_below_10pct'] is None else round(f['published_min_recall_below_10pct'],1),p['config'],p['accept'],p['utility'],p['predicted'],p['false_accepts'],p['utility_diff']['mean'],p['utility_diff']['ci_low'],p['utility_diff']['ci_high'],v['confirmation_best']['config'],v['confirmation_best']['regret_points']))
if __name__=='__main__':main()
