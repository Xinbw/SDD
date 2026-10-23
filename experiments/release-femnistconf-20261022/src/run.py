"""FEMNIST replication runner (FEMNISTCONF-DESIGN.json, frozen before this file).

Phase 'dev': pool-A writers, label draws 0-49, utility on the public validation split; test labels never
loaded. Phase 'confirm': requires SELECTION.json and PREDICTIONS.json; pool-B writers, draws 50-99,
utility on the test split. Per deployment the server sees one protected naming matrix, one protected
paired gain and (if its split has one) one protected class histogram. Clean held-out statistics and
evaluation labels are computed only after the protected decisions of the cohort are fixed.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;FRONTIER=EXP/'release-frontier-20261019';VT=EXP/'release-verify-teacher-20261020'
sys.path.insert(0,str(FRONTIER/'src'));sys.path.insert(0,str(VT/'src'))
import engine_base as base
import privacy
from run_private import metrics,decode,project
from anchored_fusion import count_loo,LAMBDAS
from run import quantize,class_bands,histogram
sys.path.insert(0,str(ROOT/'src'));import design as frozen

D=json.loads((ROOT/'FEMNISTCONF-DESIGN.json').read_text())['design']
C=62;N=D['clients'];S=1<<16;ALPHA=.025;DELTA=D['delta_total'];DEV='cuda';PREFIX='FEMNISTCONF-20261022'
PHASES=dict(dev=dict(seeds=D['development_cohort_seeds'],draws=D['label_draws']['development'],pool=0,split='validation'),
            confirm=dict(seeds=D['confirmation_cohort_seeds'],draws=D['label_draws']['confirmation'],pool=1,split='test'))
def configurations():
    out=[('fixed',.5,.25,.25)]
    out+=[(f'gain{p:g}',p,1-p,0.) for p in D['configurations']['gain_grid']['phi_naming']]
    out+=[(f'class{p:g}',p,(1-p)/4,3*(1-p)/4) for p in D['configurations']['class_grid']['phi_naming']]
    return out
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(*parts):return int.from_bytes(hashlib.sha256((PREFIX+':'+':'.join(map(str,parts))).encode()).digest()[:7],'big')
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def writer_pools(owners):
    writers=np.unique(owners);perm=np.random.default_rng(20261022).permutation(len(writers));h=len(writers)//2
    A=np.sort(writers[perm[:h]]);B=np.sort(writers[perm[h:]]);assert not len(np.intersect1d(A,B));return A,B

def cohort(owners,groups,pool,seed):
    rng=np.random.default_rng(seed);chosen=np.sort(rng.choice(pool,N,replace=False));tr=[];va=[]
    for w in chosen:
        idx=groups[w][rng.permutation(len(groups[w]))];k=min(int(math.floor(.8*len(idx))),len(idx)-1);tr.append(np.sort(idx[:k]));va.append(np.sort(idx[k:]))
    def pad(g):
        m=np.zeros((N,max(map(len,g))),dtype=np.int64);s=np.array([len(x) for x in g],dtype=np.int64)
        for i,x in enumerate(g):m[i,:len(x)]=x
        return m,s
    (tm,ts),(vm,vs)=pad(tr),pad(va);assert ts.min()>=1 and vs.min()>=1
    meta=dict(seed=seed,clients=N,writers_sha256=hashlib.sha256(chosen.astype('U8').tobytes()).hexdigest(),training_records=int(ts.sum()),heldout_records=int(vs.sum()),
        heldout_min=int(vs.min()),heldout_median=float(np.median(vs)),training_indices_sha256=hashlib.sha256(tm.tobytes()).hexdigest(),heldout_indices_sha256=hashlib.sha256(vm.tobytes()).hexdigest())
    return tm,ts,vm,vs,meta

def select(scores,weights,record,dev):
    region=dev[record['indices']];y=np.array(record['labels']);counts=np.zeros((C,C));np.add.at(counts,(region,y),1)
    label,correct=count_loo(counts,region,y);options=[dict(name='public-count',labels=label,score=float(correct.mean()),private=False)]
    weighted=project(scores)*weights[:,None]/weights.mean()
    for lam in LAMBDAS:
        label,correct=count_loo(counts+lam*weighted,region,y);options.append(dict(name=f'joint-{lam:g}',labels=label,score=float(correct.mean()),private=True))
    for mode in ['center','coverage']:
        label=decode(scores,weights,mode);options.append(dict(name='private-'+mode,labels=label,score=float((label[region]==y).mean()),private=True))
    i=max(range(len(options)),key=lambda j:(options[j]['score'],-j));return options[i],options[0]

def main(phase,seeds=None,draws=None,out=ROOT):
    frozen.main();P=PHASES[phase];selection=None
    if phase=='confirm':
        for f in ['SELECTION.json','PREDICTIONS.json']:assert (ROOT/f).exists(),'confirmation requires '+f
        selection={f:sha(ROOT/f) for f in ['SELECTION.json','PREDICTIONS.json']}
    with np.load(FRONTIER/'inputs/femnist62.npz') as z:
        data={k:z[k] for k in ['x_private','y_private','owners_private','x_proxy','x_development','x_validation','x_test']}
    data['num_classes']=C;ids,reps,centres=base.partition(data,C);sizes=np.bincount(ids,minlength=C)
    torch.backends.cuda.matmul.allow_tf32=False
    ct=torch.as_tensor(centres,device=DEV)
    def assign(x):return np.concatenate([torch.cdist(torch.as_tensor(x[s:s+65536],device=DEV),ct).argmin(1).cpu().numpy() for s in range(0,len(x),65536)])
    a=assign(np.concatenate([data['x_development'],data['x_test']]));dev,test=a[:len(data['x_development'])],a[len(data['x_development']):]
    rec=json.loads((FRONTIER/'analysis/canonical-regional-readouts.json').read_text())['geometries'];assert hashlib.sha256(test.tobytes()).hexdigest()==rec['femnist62/K62']
    evalregion=test if P['split']=='test' else assign(data['x_validation'])
    regions=assign(data['x_private'])
    owners=data['owners_private'];order=np.argsort(owners,kind='stable');uw,start=np.unique(owners[order],return_index=True)
    groups={w:np.sort(order[s:e]) for w,s,e in zip(uw,start,list(start[1:])+[len(order)])}
    pool=writer_pools(owners)[P['pool']];cfgs=configurations()
    labelsets={(r['draw']):r for r in json.loads((FRONTIER/'public/femnist62-label-samples.json').read_text()) if r['scheme']=='random' and r['q']==1}
    x=torch.as_tensor(data['x_private'],device=DEV);y=torch.as_tensor(data['y_private'],device=DEV);xp=torch.as_tensor(data['x_proxy'],device=DEV)
    for seed in (seeds or P['seeds']):
        path=out/'results'/f'{phase}_s{seed}.json'
        if path.exists():continue
        tm,ts,vm,vs,meta=cohort(owners,groups,pool,seed)
        torch.backends.cuda.matmul.allow_tf32=True
        gen=torch.Generator(device=DEV).manual_seed(seed);w=torch.randn((N,128,C),device=DEV,generator=gen)*.01;v=torch.zeros_like(w)
        idx=torch.as_tensor(tm,device=DEV);sz=torch.as_tensor(ts,device=DEV)
        for r in [1,2,3]:w,v=base.train_round(w,v,x,y,idx,sz,seed,r)
        counts=base.vote_counts(w,xp,ids);torch.backends.cuda.matmul.allow_tf32=False
        vr=regions[vm];vt=data['y_private'][vm];rows=[];certs={};clean=[]
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
                    chosen,pub=select(scores,sizes,labelsets[draw],dev)
                    h=histogram(chosen['labels'][vr],pub['labels'][vr],vt,vm,vs,C);g,q=quantize(h,vs)
                    ng=privacy.noise_sum(certg,1,seed=key(phase,'gain',seed,e,name,draw));gl=2*(int(g.sum())+int(ng[0]))/(N*S)-1-2*certg['residual_std']*math.sqrt(2*math.log(1/ALPHA))-1/S
                    priv=bool(chosen['private']);row=dict(e=e,config=name,draw=draw,candidate=chosen['name'],private=priv,gain_lower=gl)
                    if certc:
                        nc=privacy.noise_sum(certc,4*C,seed=key(phase,'hist',seed,e,name,draw));b=class_bands((q.sum(0)+nc)/(N*S),certc,C);rl=b['minimum_recall_lower'];row['min_recall_lower']=rl
                    dec={}
                    if name=='fixed' or name.startswith('gain'):dec['gain>0']=priv and gl>0
                    if certc:dec['all recalls>0']=priv and gl>0 and rl>0;dec['all recalls>=.50']=priv and gl>0 and rl>=.5
                    row['accept']=dec
                    # ---- protected decisions fixed; clean client-mean held-out statistics follow (diagnostic only).
                    tot=(h/vs[:,None,None]).mean(0);mass=tot.sum(1);correct=tot[:,2]+tot[:,3]
                    recall=np.divide(correct,mass,out=np.zeros(C),where=mass>0)
                    row['true_gain']=float(correct.sum()-(tot[:,1]+tot[:,3]).sum());row['true_min_recall']=float(recall.min())
                    row['clean_index']=len(clean);clean.append(np.stack([mass,correct]))
                    row['_cand']=chosen['labels'].tolist();row['_pub']=pub['labels'].tolist();rows.append(row)
            print(phase,'seed',seed,'eps',e,'done',flush=True)
        with np.load(FRONTIER/'inputs/femnist62.npz') as z:ye=z['y_test' if P['split']=='test' else 'y_validation']
        for r in rows:
            mc=metrics(np.array(r.pop('_cand'))[evalregion],ye,C);mp=metrics(np.array(r.pop('_pub'))[evalregion],ye,C)
            r['candidate_eval']=dict(accuracy=mc['accuracy'],minimum_recall=mc['minimum_recall']);r['public_eval']=dict(accuracy=mp['accuracy'],minimum_recall=mp['minimum_recall'])
        cpath=out/'results'/f'{phase}_s{seed}_clean.npz';cpath.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cpath,mass_correct=np.stack(clean).astype(np.float64))
        save(path,dict(phase=phase,seed=seed,cohort=meta,evaluation_split=P['split'],selection=selection,certificates=certs,rows=rows,clean_file=cpath.name,clean_sha256=sha(cpath),
            server_view='per deployment: protected naming matrix, protected paired gain, protected class histogram when its split has one'))
        del w,v;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','confirm'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:
        t0=time.time();main(a.phase,[PHASES[a.phase]['seeds'][0]],list(PHASES[a.phase]['draws'])[:2],ROOT/'smoke')
        save(ROOT/'smoke'/f'SMOKE-{a.phase}.json',dict(seconds=time.time()-t0,seed=PHASES[a.phase]['seeds'][0]))
    else:main(a.phase)
