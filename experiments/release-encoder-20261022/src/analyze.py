"""Analysis of the weak-encoder ablation (ENCODER-DESIGN.json). DINOv2 reference values are recomputed
with the same code from the unchanged release-frontier confirmation records."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;FRONTIER=EXP/'release-frontier-20261019'
SEEDS=list(range(8801,8811))

def crossed(a):
    a=np.asarray(a,float);l,s=a.shape;rng=np.random.default_rng(20261022)
    L=rng.multinomial(l,np.full(l,1/l),size=2000)/l;R=rng.multinomial(s,np.full(s,1/s),size=2000)/s;b=np.einsum('bi,ij,bj->b',L,a,R)
    return dict(mean=float(a.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)))

def frontier(stage,scheme,q,draws):
    rows=[json.loads((stage/'analysis/joint-confirmation'/f'cifar10_{scheme}_{d}_m{10*q}.json').read_text()) for d in draws]
    a=np.array([[h['ungated']['accuracy'] for h in sorted([h for h in r['hybrid'] if h['n']==1000],key=lambda h:h['seed'])] for r in rows])
    assert a.shape==(len(draws),10)
    base={k:np.array([r['baselines'][k]['accuracy'] for r in rows]) for k in rows[0]['baselines']};assert len(base)==14
    strongest=max(base,key=lambda k:base[k].mean());portfolio=np.array([r['public_portfolio']['accuracy'] for r in rows])
    priv=np.mean([[h['selected_uses_private_votes'] for h in r['hybrid'] if h['n']==1000] for r in rows])
    return dict(joint=100*a.mean(),strongest_public=dict(name=strongest,accuracy=100*base[strongest].mean()),portfolio=100*portfolio.mean(),
        gain_vs_strongest=crossed(100*(a-base[strongest][:,None])),private_selected_rate=100*float(priv)),a,portfolio

def ceilings(stage):
    sys.path.insert(0,str(FRONTIER/'src'));import engine_base as base
    old=base.ROOT;base.ROOT=stage
    try:
        with np.load(stage/'inputs/cifar10.npz') as z:data={k:z[k] for k in ['x_proxy','x_test','y_test']}
        ids,reps,centres=base.partition(data,10)
    finally:base.ROOT=old
    t=torch.cdist(torch.as_tensor(data['x_test']),torch.as_tensor(centres)).argmin(1).numpy();y=data['y_test']
    M=np.zeros((10,10));np.add.at(M,(t,y),1);r,c=linear_sum_assignment(-M)
    return dict(one_to_one=100*M[r,c].sum()/len(y),many_to_one=100*M.max(1).sum()/len(y))

def head(stage):
    v=[json.loads((stage/'confirmation/results'/f'cifar10_1000_0.1_{s}_T1_s180_head-fixed_K0_-.json').read_text())['endpoints']['broadcast']['accuracy'] for s in SEEDS]
    return 100*float(np.mean(v))

def main():
    out={}
    for name,stage in [('resnet18',ROOT),('dinov2',FRONTIER)]:
        r={'ceilings':ceilings(stage),'native_head':head(stage),'random':{},'directed':{}}
        for q in [1,3,5,10,20,50]:r['random'][f'm{10*q}'],_,_=frontier(stage,'random',q,range(100))
        out[name]=r
    rand,a,_=frontier(ROOT,'random',1,range(100));out['resnet18']['random']['m10']=rand
    for q in [1,3,5]:
        g,ga,gp=frontier(ROOT,'geometry',q,[0]);g['joint_minus_directed_public']=100*float(ga.mean()-gp.mean());out['resnet18']['directed'][f'm{10*q}']=g
    pub=json.loads((FRONTIER/'public/cifar10_geometry_0_m10.json').read_text())['public_portfolio']['accuracy']
    out['dinov2']['directed']['m10']=dict(portfolio=100*pub,note='inherited public-only record (protocol cohorts not needed: public only)')
    E1=bool(rand['gain_vs_strongest']['ci_low']>0);E2=bool(out['resnet18']['directed']['m10']['portfolio']>=rand['joint'])
    pred={'E1 holds':E1,'E2 holds':E2,
        'partition ceilings lower than with DINOv2':bool(all(out['resnet18']['ceilings'][k]<out['dinov2']['ceilings'][k] for k in ['one_to_one','many_to_one'])),
        'random-acquisition joint gain at m=10 smaller than with DINOv2':bool(rand['gain_vs_strongest']['mean']<out['dinov2']['random']['m10']['gain_vs_strongest']['mean']),
        'directed m=10: joint exceeds directed public by at most 1 point':bool(out['resnet18']['directed']['m10']['joint_minus_directed_public']<=1)}
    res=dict(design_sha256=hashlib.sha256((ROOT/'ENCODER-DESIGN.json').read_bytes()).hexdigest(),primary=dict(E1=E1,E2=E2,success=E1 and E2),predictions=pred,results=out)
    (ROOT/'analysis/encoder-analysis.json').write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(dict(primary=res['primary'],predictions=pred),indent=1))
    for n in ['dinov2','resnet18']:
        o=out[n];print(n,'ceil',{k:round(v,1) for k,v in o['ceilings'].items()},'head %.1f'%o['native_head'])
        for k,v in o['random'].items():print('  random',k,'public %.1f (%s) joint %.1f gain %+.1f [%+.1f,%+.1f] priv %.0f%%'%(v['strongest_public']['accuracy'],v['strongest_public']['name'],v['joint'],v['gain_vs_strongest']['mean'],v['gain_vs_strongest']['ci_low'],v['gain_vs_strongest']['ci_high'],v['private_selected_rate']))
        for k,v in o['directed'].items():print('  directed',k,{a:(round(b,1) if isinstance(b,float) else b) for a,b in v.items() if a in ['portfolio','joint','joint_minus_directed_public']})

if __name__=='__main__':main()
