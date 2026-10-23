"""Whole-client DP validation of an already protected public classifier.

New independent actual cohorts. Source-disjoint train/held-out private pools.
The three conditional releases share ONE declared total rho budget.
Research diagnostics (clean counts/test labels) never route a public decision.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from study import ROOT,PARENT,sha,save,freeze
sys.path.insert(0,str(PARENT/'src'))
import engine_base as base
import privacy
from anchored_fusion import count_loo,LAMBDAS
from run_private import decode,metrics,project

SEEDS=list(range(8901,8911));EPSILONS=[.1,1.,3.]
N=1000;C=10;TRAIN=480;HELD=120;D=4*C;S=1<<16
FRACTIONS=dict(naming=.5,gain=.25,classes=.25)
ALPHA_GAIN=.025;ALPHA_CLASSES=.025

def design():
    freeze('PRIVATE-VALIDATION-DESIGN.json',dict(
        dataset='cifar10',clients=N,private_seeds=SEEDS,epsilon_total=EPSILONS,delta_total=1e-5,
        zcdp_budget_fractions=FRACTIONS,public_labels=10,public_draws=100,public_label_seed=12000001,
        source_pool_partition_seed=2026101901,source_pool_partition='stratified 80/20 private source split, globally disjoint training and held-out sources',
        per_client_training_records=TRAIN,per_client_heldout_records=HELD,dirichlet_alpha=.1,
        private_training_steps=180,all_private_steps_before_naming=True,batch=128,learning_rate=.1,momentum=.9,weight_decay=5e-4,
        naming='K=C/M; one protected aggregate; rho fraction .5',
        public_baseline='fixed observed-count region matching; frozen comparator from completed development study',
        candidate_family='restricted regional demonstration: public count match; eight existing joint weights; fixed private centered and coverage',
        candidate_weights=LAMBDAS,candidate_choice='exact LOO accuracy; public-first fixed ties; no external public validation labels',
        verification_target='finite actual held-out cohort mean candidate-minus-public-baseline utility; not unseen-population or per-client benefit',
        gain_report='round half-up s*(1+paired gain)/2; integer sensitivity s',
        class_report='C x 4 joint correctness histogram per client normalized by 120, exact largest-remainder integer simplex; squared sensitivity 2*s*s',
        validation_noise='independent exact distributed discrete Gaussian shares for each of the two validation reports',
        gain_alpha=ALPHA_GAIN,class_alpha=ALPHA_CLASSES,
        gain_lower='released paired gain - 2*v_gain*sqrt(2 log(1/alpha_gain)) - 1/s',
        class_coordinate_radius='v_class*sqrt(2 log(2*D/alpha_class)) + 1/s; raw unprojected coordinates',
        gain_policy='private candidate selected AND gain lower > 0, otherwise public baseline',
        class_checked_policy='gain policy AND positive simultaneous lower class recall in every class, otherwise public baseline',
        class_recall_threshold=0.,family_failure_probability_upper=ALPHA_GAIN+ALPHA_CLASSES,
        private_local_heldout_utility='offline context only; never a selection/validation input',
        test_labels='loaded only after candidate selection, protected validation and publication decisions',
        privacy_counterfactual_scope='one label set, one cohort, one total epsilon; all research cases are separate deployments',
        no_public_annotations_for_private_verification=True,all_cases_retained=True))

def source_pools(y):
    rng=np.random.default_rng(2026101901);train=[];held=[]
    for c in range(C):
        a=rng.permutation(np.where(y==c)[0]);cut=int(.8*len(a));train.append(a[:cut]);held.append(a[cut:])
    assert not np.intersect1d(np.concatenate(train),np.concatenate(held)).size
    return train,held

def cohort(y,seed):
    trainp,heldp=source_pools(y);rng=np.random.default_rng(seed);prob=rng.dirichlet(np.full(C,.1),size=N)
    tr=np.zeros((N,TRAIN),dtype=np.int64);va=np.zeros((N,HELD),dtype=np.int64)
    for i in range(N):
        ty=rng.choice(C,TRAIN,p=prob[i]);vy=rng.choice(C,HELD,p=prob[i])
        for c in range(C):
            mask=ty==c;tr[i,mask]=rng.choice(trainp[c],int(mask.sum()),replace=True)
            mask=vy==c;va[i,mask]=rng.choice(heldp[c],int(mask.sum()),replace=True)
    assert not np.intersect1d(tr,va).size
    meta=dict(actual_clients=N,unique_client_identifiers=N,client_generation_seed=seed,alpha=.1,
        training_records_per_client=TRAIN,heldout_records_per_client=HELD,
        training_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),heldout_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest(),
        training_source_pool_sha256=hashlib.sha256(np.concatenate(trainp).tobytes()).hexdigest(),heldout_source_pool_sha256=hashlib.sha256(np.concatenate(heldp).tobytes()).hexdigest(),
        source_overlap_train_heldout=0,source_copies_across_clients='allowed; constructed-client adjacency')
    return tr,va,meta

def train(data,seed):
    tr,va,pop=cohort(data['y_private'],seed);path=ROOT/'checkpoints'/f'cifar1k_holdout_s{seed}.pt'
    if path.exists():
        state=torch.load(path,map_location='cuda',weights_only=False);assert state['population']==pop
        return state['weights'],va,pop
    torch.backends.cuda.matmul.allow_tf32=True
    gen=torch.Generator(device='cuda').manual_seed(seed)
    w=torch.randn((N,128,C),device='cuda',generator=gen)*.01;v=torch.zeros_like(w)
    x=torch.as_tensor(data['x_private'],device='cuda');y=torch.as_tensor(data['y_private'],device='cuda');idx=torch.as_tensor(tr,device='cuda');sizes=torch.full((N,),TRAIN,device='cuda')
    for r in [1,2,3]:w,v=base.train_round(w,v,x,y,idx,sizes,seed,r)
    torch.save(dict(weights=w.cpu(),population=pop),path)
    return w,va,pop

def sample_labels():
    with np.load(ROOT/'inputs/cifar10.npz') as z:y=z['y_development']
    for draw in range(100):
        path=ROOT/'public'/f'labels_{draw}.json'
        if path.exists():continue
        idx=np.random.default_rng(12000001+draw).permutation(len(y))[:10];labels=y[idx]
        save(path,dict(draw=draw,m=10,indices=idx.tolist(),labels=labels.tolist(),missing_classes=int((np.bincount(labels,minlength=C)==0).sum()),sample_sha256=hashlib.sha256(idx.tobytes()+labels.tobytes()).hexdigest()))

def select(scores,weights,label_record,development_assignment):
    region=development_assignment[label_record['indices']];y=np.array(label_record['labels']);counts=np.zeros((C,C));np.add.at(counts,(region,y),1)
    label,correct=count_loo(counts,region,y);options=[dict(name='public-count',labels=label,score=float(correct.mean()),private=False)]
    weighted=project(scores)*weights[:,None]/weights.mean()
    for lam in LAMBDAS:
        label,correct=count_loo(counts+lam*weighted,region,y)
        options.append(dict(name=f'joint-{lam:g}',labels=label,score=float(correct.mean()),private=True))
    for mode in ['center','coverage']:
        label=decode(scores,weights,mode);options.append(dict(name='private-'+mode,labels=label,score=float((label[region]==y).mean()),private=True))
    index=max(range(len(options)),key=lambda j:(options[j]['score'],-j));return options[index],options[0],{x['name']:x['score'] for x in options}

def hist_counts(candidate,public,regions,truth):
    pn=candidate[regions];pb=public[regions];outcome=2*(pn==truth)+(pb==truth)
    index=np.arange(N)[:,None]*(C*4)+truth*4+outcome
    h=np.bincount(index.ravel(),minlength=N*C*4).reshape(N,C,4)
    assert np.all(h.sum((1,2))==HELD)
    return h.astype(np.int16)

def quantize(h):
    flat=h.reshape(N,D).astype(np.int64);scaled=flat*S;q=scaled//HELD
    residual=S-q.sum(1);order=np.argsort(-(scaled%HELD),axis=1,kind='stable')
    q[np.arange(N)[:,None],order]+=(np.arange(D)[None,:]<residual[:,None])
    assert np.all(q>=0) and np.all(q.sum(1)==S)
    delta=flat.reshape(N,C,4)[:,:,2].sum(1)-flat.reshape(N,C,4)[:,:,1].sum(1)
    g=(S*(delta+HELD)+HELD)//(2*HELD)
    assert np.all((g>=0)&(g<=S))
    return g,q

def class_bands(released,cert):
    a=cert['residual_std']*math.sqrt(2*math.log(2*D/ALPHA_CLASSES))+1/S
    h=released.reshape(C,4);mass=h.sum(1);correct=h[:,2]+h[:,3]
    dlow=np.clip(mass-4*a,0,1);dup=np.clip(mass+4*a,0,1)
    nlow=np.clip(correct-2*a,0,1);nup=np.clip(correct+2*a,0,1)
    lower=np.divide(nlow,dup,out=np.zeros(C),where=dup>0);upper=np.divide(nup,dlow,out=np.ones(C),where=dlow>0)
    lower=np.clip(lower,0,1);upper=np.clip(upper,0,1)
    invalid=lower>upper;lower[invalid]=0;upper[invalid]=1
    return dict(coordinate_radius=a,recall_lower=lower.tolist(),recall_upper=upper.tolist(),minimum_recall_lower=float(lower.min()))

def evaluate_case(h,cert_gain,cert_class,seed):
    g,q=quantize(h)
    noiseg=privacy.noise_sum(cert_gain,1,seed=seed)
    noisec=privacy.noise_sum(cert_class,D,seed=seed+73)
    released_g=(int(g.sum())+int(noiseg[0]))/(N*S)
    released_h=(q.sum(0)+noisec)/(N*S)
    lower=2*released_g-1-2*cert_gain['residual_std']*math.sqrt(2*math.log(1/ALPHA_GAIN))-1/S
    bands=class_bands(released_h,cert_class)
    return dict(released_gain=float(2*released_g-1),gain_lower=float(lower),**bands),dict(client_joint_counts=h,gain_noise_sum=noiseg,class_noise_sum=noisec,released_gain_unit=np.array(released_g),released_class_histogram=released_h)

def main(only_seed=None,only_epsilon=None):
    design();sample_labels();base.ROOT=ROOT
    with np.load(ROOT/'inputs/cifar10.npz') as z:data={k:z[k] for k in ['x_private','y_private','x_proxy','x_development','y_development','x_test']}
    data['num_classes']=C
    ids,reps,centres=base.partition(data,C);sizes=np.bincount(ids,minlength=C)
    torch.backends.cuda.matmul.allow_tf32=False
    assign=torch.cdist(torch.as_tensor(np.concatenate([data['x_development'],data['x_test']]),device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
    dev,test=assign[:len(data['x_development'])],assign[len(data['x_development']):]
    private_regions=torch.cdist(torch.as_tensor(data['x_private'],device='cuda'),torch.as_tensor(centres,device='cuda')).argmin(1).cpu().numpy()
    for seed in ([only_seed] if only_seed is not None else SEEDS):
        eps=[only_epsilon] if only_epsilon is not None else EPSILONS
        if all((ROOT/'results/validation'/f's{seed}_e{e:g}_m10_d{draw}.json').exists() for e in eps for draw in range(100)):continue
        torch.backends.cuda.matmul.allow_tf32=True
        w,val,pop=train(data,seed)
        counts=base.vote_counts(w,torch.as_tensor(data['x_proxy'],device='cuda'),ids)
        val_regions=private_regions[val];val_truth=data['y_private'][val]
        with torch.no_grad():local=(torch.bmm(torch.as_tensor(data['x_private'][val],device='cuda'),w).argmax(-1).cpu().numpy()==val_truth).mean()
        for e in eps:
            key=f's{seed}_e{e:g}';named=ROOT/'results/naming'/(key+'.json')
            certn=privacy.certificate(e,1e-5,1,N,clusters=C,dimension=C*C,rho_fraction=FRACTIONS['naming'])
            certg=privacy.certificate(e,1e-5,1,N,integer_sensitivity=S,dimension=1,rho_fraction=FRACTIONS['gain'])
            certc=privacy.certificate(e,1e-5,1,N,integer_sensitivity_squared=2*S*S,dimension=D,rho_fraction=FRACTIONS['classes'])
            if named.exists():
                naming=json.loads(named.read_text());diag=np.load(ROOT/naming['diagnostic_file']);scores=diag['scores']
            else:
                scores=base.noisy_votes(counts,certn,seed*1000003+int(e*100)*901+20261021).cpu().numpy()
                diag=ROOT/'diagnostics/naming'/(key+'.npz');np.savez_compressed(diag,counts=counts.cpu().numpy(),scores=scores)
                save(named,dict(seed=seed,n=N,epsilon_total=e,population=pop,privacy=certn,diagnostic_file=str(diag.relative_to(ROOT)),diagnostic_sha256=sha(diag),private_local_heldout_accuracy=float(local),public_labels_for_validation=0))
            for draw in range(100):
                path=ROOT/'results/validation'/f'{key}_m10_d{draw}.json'
                if path.exists():continue
                lp=ROOT/'public'/f'labels_{draw}.json';label_record=json.loads(lp.read_text())
                chosen,pub,choices=select(scores,sizes,label_record,dev)
                h=hist_counts(chosen['labels'],pub['labels'],val_regions,val_truth)
                released,diagnostic=evaluate_case(h,certg,certc,seed*1000000007+int(e*100)*100003+draw*103+20261022)
                global_accept=bool(chosen['private'] and released['gain_lower']>0)
                class_accept=bool(global_accept and released['minimum_recall_lower']>0)
                # Only after protected decisions: load independent test labels.
                with np.load(ROOT/'inputs/cifar10.npz') as z:truth=z['y_test']
                newmetric=metrics(chosen['labels'][test],truth,C);pubmetric=metrics(pub['labels'][test],truth,C)
                total=h.sum(0).astype(float)/(N*HELD);mass=total.sum(1);oracle_recall=np.divide(total[:,2]+total[:,3],mass,out=np.zeros(C),where=mass>0)
                newheld=float((total[:,2]+total[:,3]).sum());pubheld=float((total[:,1]+total[:,3]).sum());gain=newheld-pubheld
                diag=ROOT/'diagnostics/validation'/(path.stem+'.npz');np.savez_compressed(diag,**diagnostic)
                save(path,dict(seed=seed,n=N,epsilon_total=e,draw=draw,m=10,naming_source=named.name,naming_source_sha256=sha(named),label_source=lp.name,label_source_sha256=sha(lp),
                    candidate=chosen['name'],candidate_uses_private=chosen['private'],candidate_labels=chosen['labels'].tolist(),public_labels=pub['labels'].tolist(),candidate_loo_scores=choices,
                    privacy_gain=certg,privacy_classes=certc,total_rho_upper=certn['rho_total_upper']+certg['rho_total_upper']+certc['rho_total_upper'],
                    protected_validation=released,policy=dict(global_gain_accepted=global_accept,class_checked_accepted=class_accept,uses_clean_statistics=False,uses_test_labels=False,private_local_used=False),
                    research_diagnostics=dict(candidate_heldout_accuracy=newheld,public_heldout_accuracy=pubheld,true_gain=gain,true_recall=oracle_recall.tolist(),
                        gain_lower_valid=bool(released['gain_lower']<=gain+1e-12),class_bands_valid=bool(np.all(np.array(released['recall_lower'])<=oracle_recall+1e-12)&np.all(np.array(released['recall_upper'])>=oracle_recall-1e-12)),
                        candidate_test=newmetric,public_test=pubmetric,global_policy_test=newmetric if global_accept else pubmetric,class_policy_test=newmetric if class_accept else pubmetric,
                        global_policy_heldout_accuracy=newheld if global_accept else pubheld,class_policy_heldout_accuracy=newheld if class_accept else pubheld),
                    diagnostic_file=str(diag.relative_to(ROOT)),diagnostic_sha256=sha(diag),server_view='protected naming matrix, protected gain and class histogram; research diagnostics excluded'))
            print('Completed',key,'100 held-out validation cases',flush=True)
        del w
    save(ROOT/'PRIVATE-VALIDATION-COMPLETE.json',dict(naming_conditions=len(list((ROOT/'results/naming').glob('*.json'))),validation_cases=len(list((ROOT/'results/validation').glob('*.json'))),utc=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--epsilon',type=float);a=p.parse_args();design()
    if not a.freeze_only:main(a.seed,a.epsilon)
