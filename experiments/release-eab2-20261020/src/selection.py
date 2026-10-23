"""Selection-rule study frozen in SELECTION-RULE-DESIGN.json (post-processing; no new query or budget).

Rules act on one released regional naming and its m=C labels; test labels only score them.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;FRONTIER=EXP/'release-frontier-20261019'
sys.path.insert(0,str(FRONTIER/'src'))
import privacy
from common import CLASSES,SETTINGS,SEEDS,release_key
from export_frontier import crossed
CONFIRM=[('cifar10',1000,.1),('cifar10',10000,.1),('mnist',10000,.1)];CONFIRM_SEEDS=list(range(8801,8811))
V0=[.01,.03,.1];RULES=['loo','cover']+[f'cover-noise-{v:g}' for v in V0]

def load(d,n,e):
    C=CLASSES[d];metric='balanced_accuracy' if d=='web' else 'accuracy';conf=(d,n,e) in CONFIRM
    seeds=CONFIRM_SEEDS if conf else SEEDS;stage=FRONTIER/'confirmation' if conf else FRONTIER
    canonical=json.loads((FRONTIER/'analysis/canonical-regional-readouts.json').read_text())['records']
    private=np.array([canonical[str((stage/'results'/(release_key(d,n,e,s,180,'regional',C,'M')+'.json')).relative_to(EXP.parent))]['releases']['1']['center'][metric] for s in seeds])
    folder=FRONTIER/'analysis'/('joint-confirmation' if conf else 'joint-naming')
    samples=sorted([json.loads(p.read_text()) for p in folder.glob(f'{d}_random_*_m{C}.json')],key=lambda x:x['draw']);assert [x['draw'] for x in samples]==list(range(100))
    loo=[];checked=0
    for x in samples:
        hs=sorted([h for h in x['hybrid'] if h['n']==n and h['epsilon']==e],key=lambda h:h['seed']);assert [h['seed'] for h in hs]==seeds
        loo.append([h['ungated'][metric] for h in hs])
        for j,h in enumerate(hs):
            if h['gate']['selected']=='private-center':assert abs(h['ungated'][metric]-private[j])<1e-12;checked+=1
    names=list(samples[0]['baselines']);assert len(names)==14
    public=np.array([[x['baselines'][k][metric] for k in names] for x in samples])
    v=privacy.certificate(e,1e-5,1,n,clusters=C,dimension=C*C)['residual_std']
    return dict(loo=np.array(loo),private=private,missing=np.array([x['missing_classes']>0 for x in samples]),public=public,names=names,v=v,checked=checked,cohorts='confirmation' if conf else 'development')

def apply(rule,s):
    if rule=='loo':use=np.zeros(len(s['missing']),bool)
    elif rule=='cover':use=s['missing']
    else:use=s['missing']&(s['v']<=float(rule.split('-')[-1]))
    return np.where(use[:,None],s['private'][None,:],s['loo'])

def gain(rule,s,draws):
    a=apply(rule,s)[draws];p=s['public'][draws];best=int(p.mean(0).argmax())
    return crossed(a-p[:,best][:,None])[0]|dict(strongest_public=s['names'][best],public=float(p[:,best].mean()),utility=float(a.mean()))

def main():
    S={x:load(*x) for x in SETTINGS};first=np.arange(50);second=np.arange(50,100);every=np.arange(100)
    fixed={r:{f'{d}/{n}/{e:g}':gain(r,S[(d,n,e)],every) for d,n,e in SETTINGS} for r in RULES}
    score={r:float(np.mean([gain(r,s,first)['mean'] for s in S.values()])) for r in RULES};chosen=max(RULES,key=lambda r:score[r])
    held={f'{d}/{n}/{e:g}':dict(selected=gain(chosen,S[(d,n,e)],second),reference=gain('loo',S[(d,n,e)],second)) for d,n,e in SETTINGS}
    out=dict(design_sha256=__import__('hashlib').sha256((ROOT/'SELECTION-RULE-DESIGN.json').read_bytes()).hexdigest(),
        settings={f'{d}/{n}/{e:g}':dict(v=s['v'],cohorts=s['cohorts'],missing_probability=float(s['missing'].mean()),private_center_matches_checked=s['checked']) for (d,n,e),s in S.items()},
        fixed_rules_all_draws=fixed,selection_scores_draws0_49=score,selected_rule=chosen,heldout_draws50_99=held,new_private_queries=0,new_privacy_budget=0)
    (ROOT/'analysis/selection-rules.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in RULES:print(r.ljust(16),' '.join(f"{k}:{100*v['mean']:+.1f}[{100*v['ci_low']:+.1f},{100*v['ci_high']:+.1f}]" for k,v in fixed[r].items()))
    print('scores',{k:round(100*v,2) for k,v in score.items()},'chosen',chosen)
    for k,v in held.items():print(k,f"sel {100*v['selected']['mean']:+.1f} ref {100*v['reference']['mean']:+.1f}")

if __name__=='__main__':main()
