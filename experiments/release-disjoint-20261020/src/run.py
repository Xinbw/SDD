"""Source-disjoint CIFAR-10 verification ladder (DISJOINT-DESIGN.json, frozen before this file).

Server view per deployment: one protected naming matrix, one protected paired gain, one protected
class histogram. Clean statistics and test labels are research diagnostics loaded after decisions.
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
import design as frozen
spec=importlib.util.spec_from_file_location('pv',VALUE/'src/private_validation.py');pv=importlib.util.module_from_spec(spec);spec.loader.exec_module(pv)

D=json.loads((ROOT/'DISJOINT-DESIGN.json').read_text())['design']
C=10;N=D['clients'];REC=D['records_per_client'];TRAIN=D['training_records_per_client'];HELD=D['heldout_records_per_client']
S=1<<16;ALPHA=.025;DELTA=D['delta_total'];FR=D['zcdp_budget_fractions'];DEV='cuda'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def cohort(y,seed):
    rng=np.random.default_rng(seed);pools=[list(rng.permutation(np.where(y==c)[0])) for c in range(C)];m=np.zeros((N,REC),dtype=np.int64)
    for i in range(N):
        p=rng.dirichlet(np.full(C,.1));take=[]
        for _ in range(REC):
            avail=np.array([len(q)>0 for q in pools],float);w=p*avail
            w=w/w.sum() if w.sum()>0 else avail/avail.sum();c=rng.choice(C,p=w);take.append(pools[c].pop())
        m[i]=take
    assert len(np.unique(m))==N*REC
    tr,va=m[:,:TRAIN],m[:,TRAIN:]
    return tr,va,dict(seed=seed,clients=N,training_records=TRAIN,heldout_records=HELD,distinct_sources=int(len(np.unique(m))),source_reuse=0,
        training_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),heldout_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest())

def nondeg_lower(h,v,eta):
    h=np.asarray(h).reshape(C,4);t=h[:,2]-h[:,1]+eta*h.sum(1)
    return float((t-v*math.sqrt(2+4*eta*eta)*math.sqrt(2*math.log(C/ALPHA))-(2+2*eta)/S).min())

def main(seeds=None,draws=None,out=ROOT):
    frozen.main()
    with np.load(FRONTIER/'inputs/cifar10.npz') as z:data={k:z[k] for k in ['x_private','y_private','x_proxy','x_development','y_development','x_test']}
    data['num_classes']=C;ids,reps,centres=base.partition(data,C);sizes=np.bincount(ids,minlength=C)
    torch.backends.cuda.matmul.allow_tf32=False
    a=torch.cdist(torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device=DEV),torch.as_tensor(centres,device=DEV)).argmin(1).cpu().numpy()
    dev,test=a[:len(data['x_development'])],a[len(data['x_development']):]
    rec=json.loads((FRONTIER/'analysis/canonical-regional-readouts.json').read_text())['geometries'];assert hashlib.sha256(test.tobytes()).hexdigest()==rec['cifar10/K10']
    regions=torch.cdist(torch.as_tensor(data['x_private'],device=DEV),torch.as_tensor(centres,device=DEV)).argmin(1).cpu().numpy()
    for seed in (seeds or D['cohort_seeds']):
        tr,va,meta=cohort(data['y_private'],seed);ck=out/'checkpoints'/f'cifar_disjoint_s{seed}.pt';ck.parent.mkdir(parents=True,exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32=True
        if ck.exists():st=torch.load(ck,map_location=DEV,weights_only=False);assert st['cohort']==meta;w=st['weights']
        else:
            gen=torch.Generator(device=DEV).manual_seed(seed);w=torch.randn((N,128,C),device=DEV,generator=gen)*.01;v=torch.zeros_like(w)
            x=torch.as_tensor(data['x_private'],device=DEV);y=torch.as_tensor(data['y_private'],device=DEV);idx=torch.as_tensor(tr,device=DEV);sz=torch.full((N,),TRAIN,device=DEV)
            for r in [1,2,3]:w,v=base.train_round(w,v,x,y,idx,sz,seed,r)
            torch.save(dict(weights=w.cpu(),cohort=meta),ck)
        counts=base.vote_counts(w,torch.as_tensor(data['x_proxy'],device=DEV),ids);torch.backends.cuda.matmul.allow_tf32=False
        vr=regions[va];vt=data['y_private'][va];vs=np.full(N,HELD,dtype=np.int64)
        for e in D['epsilon_total']:
            certn=privacy.certificate(e,DELTA,1,N,clusters=C,dimension=C*C,rho_fraction=FR['naming'])
            scores=base.noisy_votes(counts,certn,seed*1000003+int(round(e*100))*901+20261025).cpu().numpy()
            diag=out/'diagnostics'/f's{seed}_e{e:g}_naming.npz';diag.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(diag,counts=counts.cpu().numpy(),scores=scores)
            certg=privacy.certificate(e,DELTA,1,N,integer_sensitivity=S,dimension=1,rho_fraction=FR['gain'])
            certc=privacy.certificate(e,DELTA,1,N,integer_sensitivity_squared=2*S*S,dimension=4*C,rho_fraction=FR['classes'])
            for draw in (draws if draws is not None else range(100)):
                path=out/'results'/f's{seed}_e{e:g}_d{draw}.json'
                if path.exists():continue
                lr=json.loads((VALUE/'public'/f'labels_{draw}.json').read_text())
                chosen,pub,choices=pv.select(scores,sizes,lr,dev)
                h=histogram(chosen['labels'][vr],pub['labels'][vr],vt,va,vs,C);g,q=quantize(h,vs)
                key=seed*1000000007+int(round(e*100))*100003+draw*103+20261026
                ng=privacy.noise_sum(certg,1,seed=key);nc=privacy.noise_sum(certc,4*C,seed=key+73)
                rg=(int(g.sum())+int(ng[0]))/(N*S);rh=(q.sum(0)+nc)/(N*S)
                gl=2*rg-1-2*certg['residual_std']*math.sqrt(2*math.log(1/ALPHA))-1/S;bands=class_bands(rh,certc,C)
                nd={f'{eta:g}':nondeg_lower(rh,certc['residual_std'],eta) for eta in [.25,.5]}
                # ---- protected quantities fixed; research diagnostics and test labels follow.
                with np.load(FRONTIER/'inputs/cifar10.npz') as z:yt=z['y_test']
                total=h.sum(0).astype(float)/(N*HELD);mass=total.sum(1)
                recall=np.divide(total[:,2]+total[:,3],mass,out=np.zeros(C),where=mass>0);diff=np.divide(total[:,2]-total[:,1],mass,out=np.zeros(C),where=mass>0)
                save(path,dict(seed=seed,epsilon_total=e,draw=draw,cohort=meta,naming=dict(certificate=certn,diagnostic_file=str(diag.relative_to(out)),diagnostic_sha256=sha(diag)),
                    candidate=chosen['name'],candidate_uses_private=bool(chosen['private']),candidate_loo_scores=choices,comparator=pub['name'],
                    privacy=dict(gain=certg,classes=certc,total_rho_upper=certn['rho_total_upper']+certg['rho_total_upper']+certc['rho_total_upper'],total_rho_target=privacy.rho_from_epsilon(e,DELTA,'zcdp')),
                    protected_validation=dict(released_gain=2*rg-1,gain_lower=gl,released_class_histogram=rh.tolist(),nondeg_lower=nd,**bands),
                    research_diagnostics=dict(true_gain=float((total[:,2]+total[:,3]).sum()-(total[:,1]+total[:,3]).sum()),true_recall=recall.tolist(),true_diff=diff.tolist(),
                        candidate_test=metrics(chosen['labels'][test],yt,C),public_test=metrics(pub['labels'][test],yt,C)),
                    server_view='protected naming matrix, protected paired gain, protected class histogram'))
            print('seed',seed,'eps',e,'done',flush=True)
        del w;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:
        t0=time.time();main([D['cohort_seeds'][0]],[0,1],ROOT/'smoke');save(ROOT/'smoke/SMOKE.json',dict(seconds=time.time()-t0,seed=D['cohort_seeds'][0],draws=[0,1]))
    else:main()
