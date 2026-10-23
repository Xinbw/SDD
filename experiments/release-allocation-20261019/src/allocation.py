"""Feasibility study: joint allocation of one whole-client budget between private
supervision (naming) and private verification.

New releases of STORED CLEAN statistics with FRESH exact discrete-Gaussian noise.
Existing records are read only. Research diagnostics (clean counts, test labels)
never route a decision. The design is frozen before any result is computed.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from alloc_common import ROOT,VALUE,pv,privacy,bands,metrics,load_public,private_regions,clean_counts,save,sha

EPSILONS=[.1,.3,1.,3.]
PHI_N=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
PSI=[.25,.5]                     # gain share of the verification budget, class-target design
R0=[0.,.5]                       # class recall thresholds
N=1000;C=10;S=1<<16;D=4*C;ALPHA=.025
DESIGN=dict(
    purpose='go/no-go feasibility of supervision-versus-verification budget allocation',
    data='stored clean regional vote counts and reconstructed client held-out joint counts (3000/3000 exact reconstruction gate)',
    seeds=pv.SEEDS,draws=list(range(100)),epsilon_total=EPSILONS,delta=1e-5,
    gain_target=dict(releases='naming + paired gain (no class histogram)',phi_naming=PHI_N,phi_gain='1-phi_naming',rule='private candidate AND gain lower > 0'),
    class_target=dict(releases='naming + gain + C x 4 histogram',phi_naming=[p for p in PHI_N if p<.9],gain_share_of_verification=PSI,
        recall_thresholds=R0,rule='private candidate AND gain lower > 0 AND min simultaneous recall lower (direct linear bands) > r0'),
    reference='phi=(.5,.25,.25), the published fixed split, re-run with fresh noise in this same design',
    naming_noise='one fresh exact release per (seed, epsilon, phi_naming), shared by all draws and targets with that phi',
    validation_noise='fresh exact releases per case',
    outcome='unconditional independent-test accuracy of the published classifier (public fallback on rejection); acceptance rate',
    selection_honesty='best allocation chosen on draws 0-49, evaluated on draws 50-99',
    go_criterion='for gain or class(r0=0) target at >=2 of 4 epsilons: held-out-draw improvement of the selected allocation over reference >= 1.0 point with cohort-paired 95% t-interval excluding 0; and an interior optimum (phi_naming not at grid edge) for >=1 epsilon',
    registered_prediction='optimal naming share decreases as total epsilon grows, because candidate quality saturates while verification remains noise-limited',
    seeds_noise='sha256-derived replay keys with prefix ALLOC-20261019; distinct from all earlier keys')

def freeze():
    path=ROOT/'ALLOCATION-DESIGN.json';design=json.loads(json.dumps(DESIGN))
    if path.exists():assert json.loads(path.read_text())['design']==design
    else:save(path,dict(design=design,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256={p.name:sha(p) for p in (ROOT/'src').glob('*.py')}))

def key(*parts):return int.from_bytes(__import__('hashlib').sha256(('ALLOC-20261019:'+':'.join(map(str,parts))).encode()).digest()[:7],'big')

def cert(e,fraction,**kw):return privacy.certificate(e,1e-5,1,N,rho_fraction=fraction,**kw)

def release_naming(counts,e,phi,seed):
    c=cert(e,phi,clusters=C,dimension=C*C);noise=privacy.noise_sum(c,C*C,seed=key('naming',seed,e,phi))
    return ((counts.astype(np.int64)*S+noise.reshape(C,C)).astype(np.float64)/(N*S)).astype(np.float32)

def gain_lower(g,e,fraction,seed,tag):
    c=cert(e,fraction,integer_sensitivity=S,dimension=1);noise=privacy.noise_sum(c,1,seed=key('gain',tag,seed))
    unit=(int(g.sum())+int(noise[0]))/(N*S)
    return 2*unit-1-2*c['residual_std']*math.sqrt(2*math.log(1/ALPHA))-1/S

def class_lower(q,e,fraction,seed,tag):
    c=cert(e,fraction,integer_sensitivity_squared=2*S*S,dimension=D);noise=privacy.noise_sum(c,D,seed=key('hist',tag,seed))
    return bands((q.sum(0)+noise)/(N*S),c)['minimum_recall_lower']

def run_seed(seed,maps,data,samples,regions):
    out=ROOT/'results'/f'alloc_s{seed}.json'
    if out.exists():return
    counts=clean_counts(seed);tr,va,pop=pv.cohort(data['y_private'],seed);vr=regions[va];vt=data['y_private'][va]
    sizes=np.bincount(maps['proxy'],minlength=C);test=maps['test'];yt=data['y_test'];rows=[]
    for e in EPSILONS:
        for phi in PHI_N:
            scores=release_naming(counts,e,phi,seed)
            for draw in range(100):
                chosen,pub,_=pv.select(scores,sizes,samples[draw],maps['development'])
                cand=float(metrics(chosen['labels'][test],yt,C)['accuracy']);public=float(metrics(pub['labels'][test],yt,C)['accuracy'])
                h=pv.hist_counts(chosen['labels'],pub['labels'],vr,vt);g,q=pv.quantize(h)
                tag=f'{seed}:{e}:{phi}:{draw}'
                row=dict(seed=seed,epsilon=e,phi_naming=phi,draw=draw,candidate=chosen['name'],private=chosen['private'],candidate_test=cand,public_test=public)
                row['gain_target_accept']=bool(chosen['private'] and gain_lower(g,e,1-phi,seed,tag+':G')>0)
                for psi in PSI:
                    if phi>=.9:continue
                    v=1-phi;gl=gain_lower(g,e,v*psi,seed,tag+f':C{psi}');cl=class_lower(q,e,v*(1-psi),seed,tag+f':C{psi}')
                    for r0 in R0:row[f'class_accept_psi{psi}_r{r0}']=bool(chosen['private'] and gl>0 and cl>r0)
                    if psi==.5 and abs(phi-.5)<1e-9:row['reference_gain_accept']=bool(chosen['private'] and gl>0)
                rows.append(row)
        print(seed,e,'done',flush=True)
    save(out,dict(seed=seed,population=pop,rows=rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);a=p.parse_args();freeze()
    maps,data,samples=load_public();regions=private_regions(data['x_private'],maps['centres'])
    for s in ([a.seed] if a.seed else pv.SEEDS):run_seed(s,maps,data,samples,regions)
