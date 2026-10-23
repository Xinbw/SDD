"""Analysis of the budget-split confirmation study (BUDGETCONF-DESIGN.json, SELECTION.json).

Primary H1 and registered predictions, plus the registered descriptive quantities. Crossed bootstrap
over the 50 confirmation draws x 10 confirmation cohorts (2000 replicates, seed 20261026).
"""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=json.loads((ROOT/'BUDGETCONF-DESIGN.json').read_text())['design'];SEL=json.loads((ROOT/'SELECTION.json').read_text())
TARGETS=[('gain>0','gain',0.),('all recalls>0','class',0.),('all recalls>=.50','class',.5)];EPS=D['epsilon_total']

def load(phase):
    seeds=D[f'{"development" if phase=="dev" else "confirmation"}_cohort_seeds'];R={}
    for j,s in enumerate(seeds):
        for r in json.loads((ROOT/'results'/f'{phase}_s{s}.json').read_text())['rows']:R[(r['e'],r['config'],r['draw'],j)]=r
    return R,len(seeds)

def crossed(a):
    a=np.asarray(a,float);l,s=a.shape;rng=np.random.default_rng(20261026)
    L=rng.multinomial(l,np.full(l,1/l),size=2000)/l;Rr=rng.multinomial(s,np.full(s,1/s),size=2000)/s;b=np.einsum('bi,ij,bj->b',L,a,Rr)
    return dict(mean=float(a.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)))

def grid(R,ns,e,config,t,r0,draws):
    acc=np.zeros((len(draws),ns));util=np.zeros_like(acc);fa=0;tail=[];pub=np.zeros_like(acc)
    for i,d in enumerate(draws):
        for j in range(ns):
            r=R[(e,config,d,j)];ok=bool(r['accept'][t]);acc[i,j]=ok
            util[i,j]=100*(r['candidate_eval'] if ok else r['public_eval'])['accuracy'];pub[i,j]=100*r['public_eval']['accuracy']
            valid=r['true_gain']>0 and (t=='gain>0' or (r['true_min_recall']>=r0 and r['true_min_recall']>0))
            fa+=int(ok and not valid)
            if ok:tail.append(r['candidate_eval']['minimum_recall']<.1)
    return acc,util,dict(accept=100*float(acc.mean()),utility=float(util.mean()),false_accepts=fa,false_accept_rate=100*fa/max(1,acc.sum()),
        published=int(acc.sum()),published_min_recall_below_10pct=100*float(np.mean(tail)) if tail else None,public_fallback=float(pub.mean()))

def cfg(fam,p):return f'{fam}{p:g}'

def main():
    R,ns=load('confirm');draws=list(range(50,100));out={}
    for t,fam,r0 in TARGETS:
        for e in EPS:
            s=SEL['selection'][f'{t}|{e:g}'];phis=[float(p) for p in s['validation_utility_curve']]
            af,uf,xf=grid(R,ns,e,'fixed',t,r0,draws);curve={}
            for p in phis:_,_,x=grid(R,ns,e,cfg(fam,p),t,r0,draws);curve[f'{p:g}']=x
            ap,up,xp=grid(R,ns,e,cfg(fam,s['primary_phi']),t,r0,draws);a2,u2,x2=grid(R,ns,e,cfg(fam,s['secondary_phi']),t,r0,draws)
            best=max(curve,key=lambda k:curve[k]['utility'])
            out[f'{t}|{e:g}']=dict(target=t,epsilon=e,fixed=xf,primary=dict(phi=s['primary_phi'],**xp,accept_diff=crossed(ap-af),utility_diff=crossed(up-uf)),
                secondary=dict(phi=s['secondary_phi'],**x2,accept_diff=crossed(a2-af),utility_diff=crossed(u2-uf)),
                confirmation_best=dict(phi=float(best),utility=curve[best]['utility'],regret_points=curve[best]['utility']-xp['utility']),confirmation_curve=curve)
    h=out['all recalls>0|1'];H1=h['primary']['accept_diff']['ci_low']>0 and h['primary']['utility_diff']['ci_low']>0
    pred={'H1 holds':H1,
        'eps=3 recall>=.50 acceptance gain lower bound > 0':out['all recalls>=.50|3']['primary']['accept_diff']['ci_low']>0,
        'no cell more than 1 point below fixed':all(v['primary']['utility_diff']['mean']>=-1 for v in out.values()),
        'false-accept rate <= 5% in every frozen cell':all(v['primary']['false_accept_rate']<=5 for v in out.values()),
        'class phi below .5 at eps<=1':all(out[f'{t}|{e:g}']['primary']['phi']<.5 for t in ['all recalls>0','all recalls>=.50'] for e in [.1,1.])}
    # pilot size (descriptive): selection from one development cohort x 10 development draws
    Rd,nd=load('dev');pilot={}
    for t,fam,r0 in TARGETS:
        for e in [1.,3.]:
            s=SEL['selection'][f'{t}|{e:g}'];phis=[float(p) for p in s['validation_utility_curve']];picks=[]
            for j in range(nd):
                for b in range(5):
                    dd=range(10*b,10*b+10);u={}
                    for p in phis:
                        u[p]=np.mean([(Rd[(e,cfg(fam,p),d,j)]['candidate_eval'] if Rd[(e,cfg(fam,p),d,j)]['accept'][t] else Rd[(e,cfg(fam,p),d,j)]['public_eval'])['accuracy'] for d in dd])
                    m=max(u.values());picks.append(sorted([p for p in phis if u[p]>=m-1e-12],key=lambda p:(abs(p-.5),p))[0])
            cu=out[f'{t}|{e:g}']['confirmation_curve']
            pilot[f'{t}|{e:g}']=dict(subsamples=len(picks),same_as_frozen=100*float(np.mean([p==s['primary_phi'] for p in picks])),
                mean_confirmation_utility=float(np.mean([cu[f'{p:g}']['utility'] for p in picks])),fixed_utility=out[f'{t}|{e:g}']['fixed']['utility'],
                pick_counts={f'{p:g}':int(sum(q==p for q in picks)) for p in phis if any(q==p for q in picks)})
    res=dict(design_sha256=hashlib.sha256((ROOT/'BUDGETCONF-DESIGN.json').read_bytes()).hexdigest(),selection_sha256=hashlib.sha256((ROOT/'SELECTION.json').read_bytes()).hexdigest(),
        primary=dict(H1=H1,cell=dict(accept_diff=h['primary']['accept_diff'],utility_diff=h['primary']['utility_diff'])),predictions=pred,cells=out,pilot=pilot)
    (ROOT/'analysis').mkdir(exist_ok=True);(ROOT/'analysis/budgetconf-analysis.json').write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(dict(primary=res['primary'],predictions=pred),indent=1))
    for k,v in out.items():
        f,p=v['fixed'],v['primary']
        print('%-22s fixed %5.1f/%5.1f | phi %.1f %5.1f/%5.1f dAcc %+5.1f [%+.1f,%+.1f] dU %+5.1f [%+.1f,%+.1f] FA %d tail %s | sec phi %.1f %5.1f/%5.1f | best %.1f regret %.2f'%(k,f['accept'],f['utility'],p['phi'],p['accept'],p['utility'],
            100*p['accept_diff']['mean'],100*p['accept_diff']['ci_low'],100*p['accept_diff']['ci_high'],p['utility_diff']['mean'],p['utility_diff']['ci_low'],p['utility_diff']['ci_high'],p['false_accepts'],
            None if p['published_min_recall_below_10pct'] is None else round(p['published_min_recall_below_10pct'],1),v['secondary']['phi'],v['secondary']['accept'],v['secondary']['utility'],v['confirmation_best']['phi'],v['confirmation_best']['regret_points']))
    for k,v in pilot.items():print('pilot',k,v)
if __name__=='__main__':main()
