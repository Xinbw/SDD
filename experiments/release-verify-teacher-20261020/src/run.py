"""Gated selection rule + total-budget private verification on new independent cohorts.

Implements VERIFY-TEACHER-DESIGN.json (frozen before this file existed; design.py asserts it).
Server view per deployment: one protected naming matrix, one protected paired gain, one protected
class histogram. Clean counts, clean held-out statistics and test labels are research diagnostics;
test labels are loaded only after every protected decision of a case.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;FRONTIER=EXP/'release-frontier-20261019'
sys.path.insert(0,str(FRONTIER/'src'))
import engine_base as base
import privacy
import public_frontier as public
from anchored_fusion import count_loo,uniform_loo,score_correct,LAMBDAS
from run_private import project,decode,metrics
import design as frozen

D=json.loads((ROOT/'VERIFY-TEACHER-DESIGN.json').read_text())['design']
TASKS={t['dataset']:t for t in D['tasks']};SEEDS=D['cohorts']['seeds'];EPSILONS=D['epsilon_total'];DELTA=D['delta_total']
FR=D['zcdp_budget_fractions'];V0=.03;S=1<<16;ALPHA=.025;TRAIN=600;HELD=120;POOL_SEED=2026101901
DEV='cuda'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

# ---------------------------------------------------------------- cohorts
def source_pools(y,C):
    rng=np.random.default_rng(POOL_SEED);train=[];held=[]
    for c in range(C):
        a=rng.permutation(np.where(y==c)[0]);cut=int(.8*len(a));train.append(a[:cut]);held.append(a[cut:])
    assert not np.intersect1d(np.concatenate(train),np.concatenate(held)).size
    return train,held

def constructed(pools,n,size,C,rng):
    prob=rng.dirichlet(np.full(C,.1),size=n);m=np.zeros((n,size),dtype=np.int64)
    for i in range(n):
        lab=rng.choice(C,size,p=prob[i])
        for c in range(C):
            mask=lab==c;m[i,mask]=rng.choice(pools[c],int(mask.sum()),replace=True)
    return m,np.full(n,size,dtype=np.int64)

def padded(groups):
    sizes=np.array([len(g) for g in groups],dtype=np.int64);m=np.zeros((len(groups),sizes.max()),dtype=np.int64)
    for i,g in enumerate(groups):m[i,:len(g)]=g
    return m,sizes

def cohorts(task,data,seed):
    t=TASKS[task];C=t['classes'];n=t['naming_clients'];nv=t['validation_clients']
    if task=='web':
        owners=np.unique(data['owners_private']);rng=np.random.default_rng(seed)
        chosen=rng.choice(owners,n,replace=False);rest=np.setdiff1d(owners,chosen);assert len(rest)==nv and not np.intersect1d(chosen,rest).size
        nm,ns=padded([np.where(data['owners_private']==o)[0] for o in chosen]);vm,vs=padded([np.where(data['owners_private']==o)[0] for o in rest])
        extra=dict(naming_authors_sha256=hashlib.sha256(chosen.tobytes()).hexdigest(),validation_authors_sha256=hashlib.sha256(rest.tobytes()).hexdigest(),author_overlap=0)
    else:
        trainp,heldp=source_pools(data['y_private'],C)
        nm,ns=constructed(trainp,n,TRAIN,C,np.random.default_rng(seed));vm,vs=constructed(heldp,nv,HELD,C,np.random.default_rng(seed+500000))
        extra=dict(training_pool_sha256=hashlib.sha256(np.concatenate(trainp).tobytes()).hexdigest(),heldout_pool_sha256=hashlib.sha256(np.concatenate(heldp).tobytes()).hexdigest())
        assert not np.intersect1d(nm,vm).size
    meta=dict(task=task,seed=seed,naming_clients=n,validation_clients=nv,naming_sizes=[int(ns.min()),float(np.median(ns)),int(ns.max())],
        validation_sizes=[int(vs.min()),float(np.median(vs)),int(vs.max())],naming_indices_sha256=hashlib.sha256(nm.tobytes()).hexdigest(),
        validation_indices_sha256=hashlib.sha256(vm.tobytes()).hexdigest(),source_overlap_naming_validation=0,**extra)
    return (nm,ns),(vm,vs),meta

def train(task,data,seed,naming,meta):
    path=ROOT/'checkpoints'/f'{task}_s{seed}.pt'
    if path.exists():
        st=torch.load(path,map_location=DEV,weights_only=False);assert st['cohort']==meta;return st['weights']
    torch.backends.cuda.matmul.allow_tf32=True
    nm,ns=naming;n=len(ns);C=TASKS[task]['classes']
    gen=torch.Generator(device=DEV).manual_seed(seed)
    w=torch.randn((n,data['x_private'].shape[1],C),device=DEV,generator=gen)*.01;v=torch.zeros_like(w)
    x=torch.as_tensor(data['x_private'],device=DEV);y=torch.as_tensor(data['y_private'],device=DEV)
    idx=torch.as_tensor(nm,device=DEV);sz=torch.as_tensor(ns,device=DEV)
    for r in [1,2,3]:w,v=base.train_round(w,v,x,y,idx,sz,seed,r)
    torch.save(dict(weights=w.cpu(),cohort=meta),path);return w

# ---------------------------------------------------------------- validation reports
def quantize(h,sizes):
    """h: (n,C,4) integer counts per client; sizes: records per client. Generalizes release-value quantize."""
    n=len(sizes);Dm=h.shape[1]*4;flat=h.reshape(n,Dm).astype(np.int64);sz=sizes[:,None].astype(np.int64)
    scaled=flat*S;q=scaled//sz;residual=S-q.sum(1);order=np.argsort(-(scaled%sz),axis=1,kind='stable')
    q[np.arange(n)[:,None],order]+=(np.arange(Dm)[None,:]<residual[:,None])
    assert np.all(q>=0) and np.all(q.sum(1)==S)
    delta=flat.reshape(n,-1,4)[:,:,2].sum(1)-flat.reshape(n,-1,4)[:,:,1].sum(1)
    g=(S*(delta+sizes)+sizes)//(2*sizes);assert np.all((g>=0)&(g<=S))
    return g,q

def class_bands(released,cert,C):
    z=math.sqrt(2*math.log(4*C/ALPHA));v=cert['residual_std'];rn=math.sqrt(2)*v*z+2/S;rd=2*v*z+4/S
    h=np.asarray(released).reshape(C,4);mass=h.sum(1);correct=h[:,2]+h[:,3]
    dl=np.clip(mass-rd,0,1);du=np.clip(mass+rd,0,1);nl=np.clip(correct-rn,0,1);nu=np.clip(correct+rn,0,1)
    lower=np.clip(np.divide(nl,du,out=np.zeros(C),where=du>0),0,1);upper=np.clip(np.divide(nu,dl,out=np.ones(C),where=dl>0),0,1)
    bad=lower>upper;lower[bad]=0;upper[bad]=1
    return dict(numerator_radius=rn,denominator_radius=rd,recall_lower=lower.tolist(),recall_upper=upper.tolist(),minimum_recall_lower=float(lower.min()))

def histogram(cand,base_pred,truth,vm,vs,C):
    n=len(vs);mask=np.arange(vm.shape[1])[None,:]<vs[:,None]
    out=2*(cand==truth)+(base_pred==truth);index=(np.arange(n)[:,None]*(C*4)+truth*4+out)[mask]
    h=np.bincount(index,minlength=n*C*4).reshape(n,C,4);assert np.all(h.sum((1,2))==vs);return h

# ---------------------------------------------------------------- public side
class PublicTask:
    """Public learners on one task with inductive queries over test AND the private pool."""
    def __init__(self,task):
        self.task=task;self.C=TASKS[task]['classes'];torch.backends.cuda.matmul.allow_tf32=False
        data=public.public_features(task);data['dataset']=task
        with np.load(FRONTIER/'inputs'/f'{task}.npz') as b:self.x_private=b['x_private'];self.y_test=b['y_test']
        ids,reps,centres=base.partition(data,self.C);self.weights=np.bincount(ids,minlength=self.C);data['proxy_assignment']=ids
        allx=np.concatenate([data['x_development'],data['x_test']])
        a=torch.cdist(torch.as_tensor(allx,device=DEV),torch.as_tensor(centres,device=DEV)).argmin(1).cpu().numpy()
        self.dev=a[:len(data['x_development'])];self.test=a[len(data['x_development']):]
        recorded=json.loads((FRONTIER/'analysis/canonical-regional-readouts.json').read_text())['geometries']
        assert hashlib.sha256(self.test.tobytes()).hexdigest()==recorded[f'{task}/K{self.C}']
        self.private=torch.cdist(torch.as_tensor(self.x_private,device=DEV),torch.as_tensor(centres,device=DEV)).argmin(1).cpu().numpy()
        self.nt=len(data['x_test']);self.centres=centres
        graphs={al:public.graph_operator(task,data,al) for al in [.8,.95]}
        nodes=np.concatenate([data['x_development'],data['x_proxy']]);nn=NearestNeighbors(n_neighbors=21,n_jobs=8).fit(nodes)
        for al,g in graphs.items():
            h=torch.cat([torch.as_tensor(g['development'],device=DEV),torch.as_tensor(g['proxy'],device=DEV)])
            ext=self.queries(nn,h,self.x_private,g['bandwidth'])
            check=self.queries(nn,h,data['x_test'][:256],g['bandwidth'])
            assert torch.allclose(check,torch.as_tensor(g['test'][:256],device=DEV),atol=1e-6,rtol=0),'graph query reconstruction'
            graphs[al]={**{k:(torch.as_tensor(v,device=DEV) if torch.is_tensor(v) else v) for k,v in g.items()},'test':torch.cat([torch.as_tensor(g['test'],device=DEV),ext])}
        ext=dict(data);ext['x_test']=np.concatenate([data['x_test'],self.x_private])
        self.data=ext;self.graphs=graphs;self.assignment=np.concatenate([self.dev,self.test,self.private])
        folder=FRONTIER/D['tasks'][[t['dataset'] for t in D['tasks']].index(task)]['public_records'].split('release-frontier-20261019/')[1]
        self.records=folder

    @staticmethod
    def queries(nn,h,x,bandwidth):
        distance,index=nn.kneighbors(x,n_neighbors=20);q=np.exp(-distance**2/bandwidth);q/=np.maximum(q.sum(1,keepdims=True),1e-30)
        out=torch.zeros((len(x),h.shape[1]),device=DEV)
        for s in range(0,len(x),64):
            idx=torch.as_tensor(index[s:s+64],device=DEV);w=torch.as_tensor(q[s:s+64],device=DEV,dtype=torch.float32)
            out[s:s+64]=(h[idx]*w[...,None]).sum(1)
        return out

    def draw(self,number):
        torch.backends.cuda.matmul.allow_tf32=False
        C=self.C;task=self.task;rec=json.loads((self.records/f'{task}_random_{number}_m{C}.json').read_text())
        y=np.array(rec['labels']);region=self.dev[rec['indices']]
        fitted,_,_=public.public_candidates(self.data,rec,self.graphs,self.assignment,self.centres,self.weights)
        counts=np.zeros((C,C));np.add.at(counts,(region,y),1)
        cl,cc=count_loo(counts,region,y);ul,uc=uniform_loo(counts,self.weights,region,y)
        models=[]
        for cand in fitted:
            pred=cand['test'].argmax(1);correct=cand['loo'].argmax(1)==y
            value=dict(**metrics(pred[:self.nt],self.y_test,C),loo_score=score_correct(correct,y,C,task=='web'))
            assert value==rec['baselines'][cand['name']],(task,number,cand['name'])
            models.append(dict(name=cand['name'],correct=correct,score=value['loo_score'],metric=value,pool=pred[self.nt:],private=False,
                coverage=len(np.unique(cand['proxy'].argmax(1)))==C))
        for name,lab,cor in [('count-matching',cl,cc),('uniform-empty-matching',ul,uc)]:
            value=dict(**metrics(lab[self.test],self.y_test,C),loo_score=score_correct(cor,y,C,task=='web'))
            assert value==rec['baselines'][name],(task,number,name)
            models.append(dict(name=name,correct=cor,score=value['loo_score'],metric=value,pool=lab[self.private],private=False,coverage=True))
        best=max(range(len(models)),key=lambda j:(models[j]['score'],-j))
        assert models[best]['name']==rec['selected_public'] and models[best]['metric']==rec['public_portfolio'],(task,number)
        return dict(record=rec,y=y,region=region,counts=counts,models=models,best=best,missing=rec['missing_classes']>0)

def select(pub,scores,weights,C,balanced):
    options=list(pub['models']);a=project(scores)*weights[:,None]/weights.mean()
    for lam in LAMBDAS:
        lab,cor=count_loo(pub['counts']+lam*a,pub['region'],pub['y'])
        options.append(dict(name=f'joint-{lam:g}',labels=lab,correct=cor,score=score_correct(cor,pub['y'],C,balanced),private=True))
    for mode in ['center','coverage']:
        lab=decode(scores,weights,mode);cor=lab[pub['region']]==pub['y']
        options.append(dict(name='private-'+mode,labels=lab,correct=cor,score=score_correct(cor,pub['y'],C,balanced),private=True))
    i=max(range(len(options)),key=lambda j:(options[j]['score'],-j));return options[i],options

# ---------------------------------------------------------------- one task
def run_task(task,seeds,epsilons,draws,out):
    t=TASKS[task];C=t['classes'];balanced=task=='web';data=base.load_data(task)
    pt=PublicTask(task);prepared={}
    for seed in seeds:
        naming,val,meta=cohorts(task,data,seed);w=train(task,data,seed,naming,meta)
        torch.backends.cuda.matmul.allow_tf32=True  # private training/voting: engine_base convention
        counts=base.vote_counts(w,torch.as_tensor(data['x_proxy'],device=DEV),pt.data['proxy_assignment'])
        torch.backends.cuda.matmul.allow_tf32=False  # all public computation: public_frontier convention
        vm,vs=val;truth=data['y_private'][vm]
        for e in epsilons:
            certn=privacy.certificate(e,DELTA,1,t['naming_clients'],clusters=C,dimension=C*C,rho_fraction=FR['naming'])
            scores=base.noisy_votes(counts,certn,seed*1000003+int(round(e*100))*901+20261023).cpu().numpy()
            diag=out/'diagnostics'/f'{task}_s{seed}_e{e:g}_naming.npz';diag.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(diag,counts=counts.cpu().numpy(),scores=scores)
            prepared[(seed,e)]=dict(scores=scores,certn=certn,vm=vm,vs=vs,truth=truth,meta=meta,diag=diag,
                clean_center=decode(counts.cpu().numpy()/t['naming_clients'],pt.weights,'center'))
        del w;torch.cuda.empty_cache()
    for number in draws:
        pending=[(s,e) for s in seeds for e in epsilons if not (out/'results'/f'{task}_s{s}_e{e:g}_d{number}.json').exists()]
        if not pending:continue
        pub=pt.draw(number);b=pub['models'][pub['best']]
        for seed,e in pending:
            p=prepared[(seed,e)];v=p['certn']['residual_std']
            chosen,options=select(pub,p['scores'],pt.weights,C,balanced)
            gate_fires=bool(pub['missing'] and v<=V0)
            cand=next(o for o in options if o['name']=='private-center') if gate_fires else chosen
            pool=lambda o:o['labels'][pt.private] if o['private'] else o['pool']
            vm,vs=p['vm'],p['vs'];h=histogram(pool(cand)[vm],b['pool'][vm],p['truth'],vm,vs,C)
            g,q=quantize(h,vs);nv=len(vs)
            certg=privacy.certificate(e,DELTA,1,nv,integer_sensitivity=S,dimension=1,rho_fraction=FR['gain'])
            certc=privacy.certificate(e,DELTA,1,nv,integer_sensitivity_squared=2*S*S,dimension=4*C,rho_fraction=FR['classes'])
            key=seed*1000000007+int(round(e*100))*100003+number*103+20261024
            ng=privacy.noise_sum(certg,1,seed=key);nc=privacy.noise_sum(certc,4*C,seed=key+73)
            rg=(int(g.sum())+int(ng[0]))/(nv*S);rh=(q.sum(0)+nc)/(nv*S)
            gain_lower=2*rg-1-2*certg['residual_std']*math.sqrt(2*math.log(1/ALPHA))-1/S;bands=class_bands(rh,certc,C)
            gain_ok=bool(cand['private'] and gain_lower>0);class_ok=bool(gain_ok and bands['minimum_recall_lower']>0)
            # ---- all protected decisions are fixed; research diagnostics and test labels follow.
            with np.load(FRONTIER/'inputs'/f'{task}.npz') as z:ytest=z['y_test']
            test=lambda o:metrics(o['labels'][pt.test],ytest,C) if o['private'] else {k:o['metric'][k] for k in ['accuracy','balanced_accuracy','per_class_recall','minimum_recall']}
            mb=test(b);mloo=test(chosen);mg=test(cand)
            total=h.sum(0).astype(float);per=h.astype(float)/vs[:,None,None]
            dclient=(per[:,:,2].sum(1)+per[:,:,3].sum(1))-(per[:,:,1].sum(1)+per[:,:,3].sum(1))
            mass=total.sum(1);true_recall=np.divide(total[:,2]+total[:,3],mass,out=np.zeros(C),where=mass>0)
            save(out/'results'/f'{task}_s{seed}_e{e:g}_d{number}.json',dict(task=task,seed=seed,epsilon_total=e,draw=number,m=C,missing_classes=int(pub['record']['missing_classes']),
                label_source=pub['record']['source'] if 'source' in pub['record'] else f'{task}_random_{number}_m{C}.json',label_sample_sha256=pub['record']['sample_sha256'],
                cohort=p['meta'],naming=dict(certificate=p['certn'],noise_std=v,diagnostic_file=str(p['diag'].relative_to(out)),diagnostic_sha256=sha(p['diag'])),
                comparator=b['name'],loo_choice=chosen['name'],loo_uses_private=bool(chosen['private']),gate_fires=gate_fires,
                candidate=cand['name'],candidate_uses_private=bool(cand['private']),candidate_loo_scores={o['name']:o['score'] for o in options},
                privacy=dict(gain=certg,classes=certc,total_rho_upper=p['certn']['rho_total_upper']+certg['rho_total_upper']+certc['rho_total_upper'],
                    total_rho_target=privacy.rho_from_epsilon(e,DELTA,'zcdp'),composition='sequential'),
                protected_validation=dict(released_gain=2*rg-1,gain_lower=gain_lower,released_class_histogram=rh.tolist(),**bands),
                decisions=dict(gated_gain_verified=gain_ok,gated_class_verified=class_ok,uses_test_labels=False,uses_clean_statistics=False),
                test=dict(public=mb,loo=mloo,gated=mg,gated_gain_verified=mg if gain_ok else mb,gated_class_verified=mg if class_ok else mb),
                research_diagnostics=dict(clean_client_mean_gain=float(dclient.mean()),true_recall=true_recall.tolist(),
                    gain_lower_valid=bool(gain_lower<=dclient.mean()+1e-12),false_gain_accept=bool(gain_ok and dclient.mean()<=0),
                    clean_center_test=metrics(p['clean_center'][pt.test],ytest,C)),
                server_view='protected naming matrix, protected paired gain, protected class histogram'))
        print(task,'draw',number,'done',len(pending),'cases',flush=True)

# ---------------------------------------------------------------- smoke reproduction checks
def reference_checks():
    import importlib.util
    sys.path.insert(0,str(EXP/'release-value-20261019/src'))
    spec=importlib.util.spec_from_file_location('pv',EXP/'release-value-20261019/src/private_validation.py');pv=importlib.util.module_from_spec(spec);spec.loader.exec_module(pv)
    spec=importlib.util.spec_from_file_location('lb',EXP/'release-value-20261019/src/linear_bands.py');lb=importlib.util.module_from_spec(spec);spec.loader.exec_module(lb)
    rng=np.random.default_rng(1);n=pv.N;h=np.zeros((n,10,4),dtype=np.int64)
    for i in range(n):np.add.at(h[i],(rng.integers(0,10,120),rng.integers(0,4,120)),1)
    g0,q0=pv.quantize(h);g1,q1=quantize(h,np.full(n,120));assert (g0==g1).all() and (q0==q1).all()
    cert=privacy.certificate(1.,DELTA,1,1000,integer_sensitivity_squared=2*S*S,dimension=40,rho_fraction=.25);r=rng.random(40)/40
    assert class_bands(r,cert,10)==lb.bands(r,cert);checks=['quantize equals release-value quantize (1000 clients x 120)','class bands equal linear_bands.bands']
    # Selection logic reproduces a recorded old-cohort hybrid choice exactly.
    pt=PublicTask('mnist');CONF=FRONTIER/'confirmation'
    for number in [0,1]:
        pub=pt.draw(number);rec=pub['record']
        for hy in [x for x in rec['hybrid'] if x['n']==10000 and x['epsilon']==.1][:3]:
            res=json.loads((CONF/'results'/(hy['protected_source']+'.json')).read_text());scores=np.load(CONF/res['diagnostic_file'])['scores']
            chosen,_=select(pub,scores,pt.weights,10,False)
            m=metrics(chosen['labels'][pt.test],pt.y_test,10) if chosen['private'] else chosen['metric']
            assert chosen['name']==hy['gate']['selected'] and abs(m['accuracy']-hy['ungated']['accuracy'])<1e-12,(number,hy['seed'])
    checks.append('LOO selection reproduces 6 recorded old-cohort MNIST hybrid choices and metrics exactly; public learners on extended queries equal recorded test metrics')
    return checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--task',choices=list(TASKS));a=p.parse_args()
    frozen.main()
    if a.smoke:
        out=ROOT/'smoke';t0=time.time();checks=reference_checks();print('\n'.join(checks),flush=True)
        for task in ([a.task] if a.task else list(TASKS)):run_task(task,[SEEDS[task][0]],[1.],[0,1],out)
        save(out/'SMOKE.json',dict(checks=checks,tasks=[a.task] if a.task else list(TASKS),seeds='first seed per task',epsilon=[1.],draws=[0,1],seconds=time.time()-t0,
            note='draws 0-1 are secondary draws; primary draws 50-99 are not touched by the smoke test'))
    else:
        for task in ([a.task] if a.task else list(TASKS)):run_task(task,SEEDS[task],EPSILONS,list(range(100)),ROOT)

if __name__=='__main__':main()
