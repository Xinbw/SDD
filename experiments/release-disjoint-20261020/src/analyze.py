"""Pre-specified analysis of DISJOINT-DESIGN.json: the ladder on source-disjoint clients.

Same rules as release-eab-20261020/thresholds.py (each a separately declared rule, alpha=.025),
compared with the reused-source ladder on the same label draws. Crossed bootstrap over label draws x
cohorts, 2000 replicates, seed 20261025.
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EAB=ROOT.parent/'release-eab-20261020'
D=json.loads((ROOT/'DISJOINT-DESIGN.json').read_text())['design'];SEEDS=D['cohort_seeds']
RULES=[('gain>=0.0','gain>0'),('gain>=0.1','gain>=.10'),('gain>=0.2','gain>=.20'),('recall>=0.0','all recalls>0'),('recall>=0.25','all recalls>=.25'),
       ('recall>=0.5','all recalls>=.50'),('nondeg eta=0.25','non-degradation eta=.25'),('nondeg eta=0.5','non-degradation eta=.50')]

def crossed(a):
    a=np.asarray(a,float);l,s=a.shape;rng=np.random.default_rng(20261025)
    L=rng.multinomial(l,np.full(l,1/l),size=2000)/l;R=rng.multinomial(s,np.full(s,1/s),size=2000)/s;b=np.einsum('bi,ij,bj->b',L,a,R)
    return dict(mean=float(a.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)))

def decide(r,key):
    pv=r['protected_validation'];d=r['research_diagnostics'];priv=r['candidate_uses_private'];gl=pv['gain_lower'];rl=min(pv['recall_lower'])
    tg=d['true_gain'];tr=min(d['true_recall']);td=min(d['true_diff'])
    if key.startswith('gain'):g=float(key.split('>=')[1]);return priv and gl>g,tg>g
    if key.startswith('recall'):
        r0=float(key.split('>=')[1]);ok=priv and gl>0 and rl>=r0 and (r0>0 or rl>0);return ok,tg>0 and tr>=r0 and tr>0
    eta=key.split('=')[1];return priv and gl>0 and pv['nondeg_lower'][eta]>=0,tg>0 and td>=-float(eta)

def main():
    R={(r['seed'],r['epsilon_total'],r['draw']):r for r in (json.loads(p.read_text()) for p in (ROOT/'results').glob('*.json'))}
    assert len(R)==len(SEEDS)*3*100,len(R)
    ref=json.loads((EAB/'analysis/threshold-policies.json').read_text())['policies'];out={}
    for e in D['epsilon_total']:
        res={}
        for key,name in RULES:
            acc=np.zeros((100,len(SEEDS)),bool);util=np.zeros((100,len(SEEDS)));fa=0
            for i in range(100):
                for j,s in enumerate(SEEDS):
                    r=R[(s,e,i)];ok,valid=decide(r,key);acc[i,j]=ok;fa+=int(ok and not valid)
                    util[i,j]=r['research_diagnostics']['candidate_test' if ok else 'public_test']['accuracy']
            pub=np.array([[R[(s,e,i)]['research_diagnostics']['public_test']['accuracy'] for s in SEEDS] for i in range(100)])
            refkey={'gain>=0.0':'gain>=0.0'}.get(key,key)
            res[name]=dict(accept=100*float(acc.mean()),policy_test=100*float(util.mean()),false_accepts=fa,improvement_over_public=crossed(util-pub),
                reused_source=dict(accept=ref[str(e)][refkey]['accept'],policy_test=ref[str(e)][refkey]['policy_test']))
        cand=np.array([[R[(s,e,i)]['research_diagnostics']['candidate_test']['accuracy'] for s in SEEDS] for i in range(100)])
        out[f'{e:g}']=dict(rules=res,candidate_test=100*float(cand.mean()),public_test=100*float(np.mean([[R[(s,e,i)]['research_diagnostics']['public_test']['accuracy'] for s in SEEDS] for i in range(100)])),
            private_candidate_share=float(np.mean([R[k]['candidate_uses_private'] for k in R if k[1]==e])))
    g1=out['1']['rules']['gain>0']['improvement_over_public'];S1=g1['ci_low']>0
    mono=all(out[e]['rules'][a]['accept']>=out[e]['rules'][b]['accept']-1e-9 for e in out for a,b in [('gain>0','gain>=.10'),('gain>=.10','gain>=.20'),('all recalls>0','all recalls>=.25'),('all recalls>=.25','all recalls>=.50'),('non-degradation eta=.50','non-degradation eta=.25')])
    noFA=all(out[e]['rules'][n]['false_accepts']==0 for e in ['1','3'] for _,n in RULES)
    reused=out['1']['rules']['gain>0']['reused_source']['policy_test']-62.3
    smaller=100*g1['mean']<reused
    pred={'S1 holds':S1,'monotone ladder':mono,'no false accept at eps>=1':noFA,'eps=1 improvement smaller than reused-source':smaller}
    result=dict(design_sha256=__import__('hashlib').sha256((ROOT/'DISJOINT-DESIGN.json').read_bytes()).hexdigest(),primary=dict(S1=S1,eps1_gain_policy_improvement=g1),predictions=pred,
        reused_source_eps1_gain_improvement_points=reused,by_epsilon=out)
    (ROOT/'analysis/disjoint-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(primary=result['primary'],predictions=pred),indent=1))
    for e,v in out.items():
        print('eps',e,'cand %.1f pub %.1f priv %.2f'%(v['candidate_test'],v['public_test'],v['private_candidate_share']))
        for n,x in v['rules'].items():print('  %-26s acc %5.1f util %5.1f FA %d | reused %5.1f / %5.1f'%(n,x['accept'],x['policy_test'],x['false_accepts'],x['reused_source']['accept'],x['reused_source']['policy_test']))

if __name__=='__main__':main()
