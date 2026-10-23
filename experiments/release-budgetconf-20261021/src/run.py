"""Budget-split confirmation study (BUDGETCONF-DESIGN.json, frozen before this file).

Phase 'dev': development cohorts from source pool A, label draws 0-49, utility on the public validation
split; test labels are never loaded. Phase 'confirm': requires the frozen SELECTION.json; confirmation
cohorts from pool B, label draws 50-99, utility on the test split. Per deployment the server sees one
protected naming matrix, one protected paired gain and (if its split has one) one protected class
histogram. Clean statistics and evaluation labels are loaded only after the protected decisions.
"""
import argparse,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;FRONTIER=EXP/'release-frontier-20261019';VALUE=EXP/'release-value-20261019';VT=EXP/'release-verify-teacher-20261020'
sys.path.insert(0,str(FRONTIER/'src'));sys.path.insert(0,str(VALUE/'src'));sys.path.insert(0,str(VT/'src'))
import engine_base as base
import privacy
from run_private import metrics
from run import quantize,class_bands,histogram
sys.path.insert(0,str(ROOT/'src'));import design as frozen
spec=importlib.util.spec_from_file_location('pv',VALUE/'src/private_validation.py');pv=importlib.util.module_from_spec(spec);spec.loader.exec_module(pv)

D=json.loads((ROOT/'BUDGETCONF-DESIGN.json').read_text())['design']
C=10;N=D['clients'];REC=D['records_per_client'];TRAIN=D['training_records_per_client'];HELD=D['heldout_records_per_client']
S=1<<16;ALPHA=.025;DELTA=D['delta_total'];DEV='cuda'
PHASES=dict(dev=dict(seeds=D['development_cohort_seeds'],draws=range(0,50),pool=0,split='validation'),
            confirm=dict(seeds=D['confirmation_cohort_seeds'],draws=range(50,100),pool=1,split='test'))
def configurations():
    out=[('fixed',.5,.25,.25)]
    out+=[(f'gain{p:g}',p,1-p,0.) for p in D['configurations']['gain_grid']['phi_naming']]
    out+=[(f'class{p:g}',p,(1-p)/4,3*(1-p)/4) for p in D['configurations']['class_grid']['phi_naming']]
    return out
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(*parts):return int.from_bytes(hashlib.sha256(('BUDGETCONF-20261021:'+':'.join(map(str,parts))).encode()).digest()[:7],'big')
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def pools(y):
    rng=np.random.default_rng(20261021);A=[];B=[]
    for c in range(C):
        idx=rng.permutation(np.where(y==c)[0]);h=len(idx)//2;A.append(idx[:h]);B.append(idx[h:])
    A=np.sort(np.concatenate(A));B=np.sort(np.concatenate(B));assert len(np.intersect1d(A,B))==0
    return A,B

def cohort(y,pool,seed):
    rng=np.random.default_rng(seed);yp=y[pool];per=[list(pool[rng.permutation(np.where(yp==c)[0])]) for c in range(C)];m=np.zeros((N,REC),dtype=np.int64)
    for i in range(N):
        p=rng.dirichlet(np.full(C,.1));take=[]
        for _ in range(REC):
            avail=np.array([len(q)>0 for q in per],float);w=p*avail
            w=w/w.sum() if w.sum()>0 else avail/avail.sum();c=rng.choice(C,p=w);take.append(per[c].pop())
        m[i]=take
    assert len(np.unique(m))==N*REC and np.all(np.isin(m,pool))
    tr,va=m[:,:TRAIN],m[:,TRAIN:]
    return tr,va,dict(seed=seed,clients=N,training_records=TRAIN,heldout_records=HELD,distinct_sources=int(len(np.unique(m))),source_reuse=0,
        pool_sha256=hashlib.sha256(pool.tobytes()).hexdigest(),training_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),heldout_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest())

def main(phase,seeds=None,draws=None,out=ROOT):
    frozen.main();P=PHASES[phase];selection=None
    if phase=='confirm':
        sel=ROOT/'SELECTION.json';assert sel.exists(),'confirmation requires the frozen SELECTION.json';selection=dict(sha256=sha(sel),frozen_utc=json.loads(sel.read_text())['frozen_utc'])
    keys=['x_private','y_private','x_proxy','x_development','x_validation','x_test']
    with np.load(FRONTIER/'inputs/cifar10.npz') as z:data={k:z[k] for k in keys}
    data['num_classes']=C;ids,reps,centres=base.partition(data,C);sizes=np.bincount(ids,minlength=C)
    torch.backends.cuda.matmul.allow_tf32=False
    assign=lambda x:torch.cdist(torch.as_tensor(x,device=DEV),torch.as_tensor(centres,device=DEV)).argmin(1).cpu().numpy()
    a=assign(np.concatenate([data['x_development'],data['x_test']]));dev,test=a[:len(data['x_development'])],a[len(data['x_development']):]
    rec=json.loads((FRONTIER/'analysis/canonical-regional-readouts.json').read_text())['geometries'];assert hashlib.sha256(test.tobytes()).hexdigest()==rec['cifar10/K10']
    evalregion=test if P['split']=='test' else assign(data['x_validation'])
    regions=assign(data['x_private']);pool=pools(data['y_private'])[P['pool']];cfgs=configurations()
    for seed in (seeds or P['seeds']):
        path=out/'results'/f'{phase}_s{seed}.json'
        if path.exists():continue
        tr,va,meta=cohort(data['y_private'],pool,seed);ck=out/'checkpoints'/f'{phase}_s{seed}.pt';ck.parent.mkdir(parents=True,exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32=True
        if ck.exists():st=torch.load(ck,map_location=DEV,weights_only=False);assert st['cohort']==meta;w=st['weights']
        else:
            gen=torch.Generator(device=DEV).manual_seed(seed);w=torch.randn((N,128,C),device=DEV,generator=gen)*.01;v=torch.zeros_like(w)
            x=torch.as_tensor(data['x_private'],device=DEV);y=torch.as_tensor(data['y_private'],device=DEV);idx=torch.as_tensor(tr,device=DEV);sz=torch.full((N,),TRAIN,device=DEV)
            for r in [1,2,3]:w,v=base.train_round(w,v,x,y,idx,sz,seed,r)
            torch.save(dict(weights=w.cpu(),cohort=meta),ck)
        counts=base.vote_counts(w,torch.as_tensor(data['x_proxy'],device=DEV),ids);torch.backends.cuda.matmul.allow_tf32=False
        vr=regions[va];vt=data['y_private'][va];vs=np.full(N,HELD,dtype=np.int64);rows=[];certs={}
        for e in D['epsilon_total']:
            for name,fn,fg,fc in cfgs:
                certn=privacy.certificate(e,DELTA,1,N,clusters=C,dimension=C*C,rho_fraction=fn)
                certg=privacy.certificate(e,DELTA,1,N,integer_sensitivity=S,dimension=1,rho_fraction=fg)
                certc=privacy.certificate(e,DELTA,1,N,integer_sensitivity_squared=2*S*S,dimension=4*C,rho_fraction=fc) if fc>0 else None
                total=certn['rho_total_upper']+certg['rho_total_upper']+(certc['rho_total_upper'] if certc else 0.)
                target=privacy.rho_from_epsilon(e,DELTA,'zcdp');assert total<=target*(1+1e-12),(e,name,total,target)
                certs[f'{e:g}/{name}']=dict(naming=certn,gain=certg,classes=certc,total_rho_upper=total,total_rho_target=target)
                scores=base.noisy_votes(counts,certn,key(phase,'naming',seed,e,name)).cpu().numpy()
                for draw in (draws if draws is not None else P['draws']):
                    lr=json.loads((VALUE/'public'/f'labels_{draw}.json').read_text())
                    chosen,pub,_=pv.select(scores,sizes,lr,dev)
                    h=histogram(chosen['labels'][vr],pub['labels'][vr],vt,va,vs,C);g,q=quantize(h,vs)
                    ng=privacy.noise_sum(certg,1,seed=key(phase,'gain',seed,e,name,draw));gl=2*(int(g.sum())+int(ng[0]))/(N*S)-1-2*certg['residual_std']*math.sqrt(2*math.log(1/ALPHA))-1/S
                    priv=bool(chosen['private']);row=dict(e=e,config=name,draw=draw,candidate=chosen['name'],private=priv,gain_lower=gl)
                    if certc:
                        nc=privacy.noise_sum(certc,4*C,seed=key(phase,'hist',seed,e,name,draw));b=class_bands((q.sum(0)+nc)/(N*S),certc,C);rl=b['minimum_recall_lower'];row['min_recall_lower']=rl
                    dec={}
                    if name=='fixed' or name.startswith('gain'):dec['gain>0']=priv and gl>0
                    if certc:dec['all recalls>0']=priv and gl>0 and rl>0;dec['all recalls>=.50']=priv and gl>0 and rl>=.5
                    row['accept']=dec
                    # ---- protected decisions fixed; clean diagnostics follow (evaluation labels below, after the loop).
                    tot=h.sum(0).astype(float)/(N*HELD);mass=tot.sum(1);recall=np.divide(tot[:,2]+tot[:,3],mass,out=np.zeros(C),where=mass>0)
                    row['true_gain']=float((tot[:,2]+tot[:,3]).sum()-(tot[:,1]+tot[:,3]).sum());row['true_min_recall']=float(recall.min())
                    row['_cand']=chosen['labels'].tolist();row['_pub']=pub['labels'].tolist();rows.append(row)
            print(phase,'seed',seed,'eps',e,'done',flush=True)
        # all protected decisions of this cohort are fixed: load the phase's evaluation labels now
        with np.load(FRONTIER/'inputs/cifar10.npz') as z:ye=z['y_test' if P['split']=='test' else 'y_validation']
        for r in rows:
            mc=metrics(np.array(r.pop('_cand'))[evalregion],ye,C);mp=metrics(np.array(r.pop('_pub'))[evalregion],ye,C)
            r['candidate_eval']=dict(accuracy=mc['accuracy'],minimum_recall=mc['minimum_recall']);r['public_eval']=dict(accuracy=mp['accuracy'],minimum_recall=mp['minimum_recall'])
        save(path,dict(phase=phase,seed=seed,cohort=meta,evaluation_split=P['split'],selection=selection,certificates=certs,rows=rows,
            server_view='per deployment: protected naming matrix, protected paired gain, protected class histogram when its split has one'))
        del w;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','confirm'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:
        t0=time.time();main(a.phase,[PHASES[a.phase]['seeds'][0]],list(PHASES[a.phase]['draws'])[:2],ROOT/'smoke')
        save(ROOT/'smoke'/f'SMOKE-{a.phase}.json',dict(seconds=time.time()-t0,seed=PHASES[a.phase]['seeds'][0]))
    else:main(a.phase)
