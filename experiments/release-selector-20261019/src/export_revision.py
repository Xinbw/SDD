"""Generate the second revision from frozen identities and actual endpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from common import ROOT,key,save


def stats(values):
    a=np.array(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.,values=a.tolist())


def paired(a,b):
    s=stats(np.array(a)-np.array(b));h=float(t.ppf(.975,s['n']-1)*s['sd']/np.sqrt(s['n']))
    return dict(**s,ci_low=s['mean']-h,ci_high=s['mean']+h)


def cell(s):return f"{s['mean']*100:.2f}$\\pm${s['sd']*100:.2f}"


def table(path,caption,label,cols,head,rows,wide=True):
    env='table*' if wide else 'table'
    text=f'\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\\small\n\\centering\n'
    text+=f'\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\\toprule\n'+' & '.join(head)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(row)+r' \\' for row in rows)
    text+=f'\n\\bottomrule\\end{{tabular}}\n\\end{{{env}}}\n';path.write_text(text)


def export(paper,partial=False):
    paper.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'STUDY-SCHEDULE.json').read_text())['design']
    results={p.stem:json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))}
    readers={p.stem:json.loads(p.read_text()) for p in sorted((ROOT/'analysis/public-readouts').glob('*.json'))}
    out=dict(conditions=[],mix=[],natural_resolution=[],public_references=[],centroid_replay_count=390,
        original_training_conditions=1771,actual_new_private_conditions=sum('population' in r for r in results.values()),
        new_public_reference_conditions=sum('Public-only' in r['method'] for r in results.values()),
        public_student_readouts=len(readers),all_private_conditions_planned=len(plan['jobs']))
    roles=plan['roles'];condition_ids=[]
    table(paper/'tab-study-map.tex',
        r'Comparison conditions and seed identities. Seeds are replications; the writer grid and mixture study reuse specified paired subsets. '
        r'Public labels used for selection are listed separately from the protected client supervision used for targets.',
        'tab:study-map','llll',['Condition','Configuration resource','Seed group','Reported endpoints'],[
            ['Fixed CIFAR matching','K=10, M, matching; no task labels','801--805; 8201--8205; 8601--8610','Public NN; private client'],
            ['Public rule, four tasks','Development/validation; joint K/e/d','8501--8510, eight settings','NN; public student; client'],
            ['CIFAR mixture intervention','Fixed K=10, M, matching; six mixtures','8601--8610 (same ten as above)','NN; client, fixed 180 steps'],
            ['Natural-writer K grid','Five fixed K; M, centered','8501--8505 (paired subset)','NN/client; accuracy/macro'],
            ['Point-query placement','Ten point queries, centered; paired replay','8201--8205','Private client'],
            ['FEMNIST source controls','5000 development labels; validation','8501--8510','Public-only; Local+Public']])
    for role in roles:
        if role['role']=='public-selected' and role['condition'] not in condition_ids:condition_ids.append(role['condition'])
    def records(role,condition):
        ids=[r['job'] for r in roles if r['role']==role and r['condition']==condition]
        if not all(i in results for i in ids):raise KeyError((role,condition))
        return [results[i] for i in ids]
    rows=[];extra=[]
    task_name={'cifar10':'CIFAR','mnist':'MNIST','web':'Usenet','femnist62':'FEMNIST'}
    for condition in condition_ids:
        dataset=condition['dataset'];metric='balanced_accuracy' if dataset=='web' else 'accuracy'
        try:
            selected=records('public-selected',condition);fixed=records('fixed-center',condition)
            head=records('head',condition);ct=records('per-example',condition);local=records('local',condition)
        except KeyError:
            if partial:continue
            raise
        psel=json.loads((ROOT/'public-selection'/f'{dataset}.json').read_text())
        cfg=next(r for r in psel['records'] if r['n']==condition['n'] and r['epsilon']==condition['epsilon'])
        choice=cfg['selected'];profile=np.load(ROOT/'public-selection'/f"{dataset}_K{choice['k']}_{choice['estimator']}.npz")
        metric_weights=profile['validation_class_weights'];pw=profile['proxy_region_weights'];pw=pw/pw.sum()
        actual_validation=[];shift_tv=[];shift_inf=[]
        for r in selected:
            d=np.load(ROOT/r['diagnostic_file']);q=d['r1_counts']/r['n'];label=d['r1_labels']
            actual_validation.append(float(metric_weights[np.arange(len(label)),label].sum()))
            shift_tv.append(float((np.abs(q-profile['q']).sum(1)*pw).sum()/2))
            shift_inf.append(float(np.abs(q-profile['q']).max()))
        glob=stats([r['global_endpoint'][metric] for r in selected]);fl=stats([r['global_endpoint'][metric] for r in fixed])
        private=stats([r[metric] for r in selected]);hg=stats([r['global_endpoint'][metric] for r in head])
        loc=stats([r[metric] for r in local]);ctl=stats([r[metric] for r in ct]);hl=stats([r['head_local_endpoint'][metric] for r in head])
        sr=[readers.get(key(r)) for r in selected];cr=[readers.get(key(r)) for r in ct]
        student=stats([r[metric] for r in sr]) if all(sr) else None
        ctstudent=stats([r[metric] for r in cr]) if all(cr) else None
        row=dict(**condition,metric=metric,seeds=list(range(8501,8511)),choice=choice,selected_global=glob,
            fixed_global=fl,selected_local=private,head_global=hg,head_local=hl,ct_local=ctl,local_only=loc,
            student=student,ct_student=ctstudent,public_predicted_validation=cfg['selected_predicted_score'],
            actual_first_release_validation=stats(actual_validation),private_public_weighted_TV=stats(shift_tv),
            private_public_sup_shift=stats(shift_inf),global_gain_vs_fixed=paired(glob['values'],fl['values']),
            global_gain_vs_head=paired(glob['values'],hg['values']),local_gain_vs_ct=paired(private['values'],ctl['values']))
        row['selected_client_macro']=stats([r['balanced_accuracy'] for r in selected])
        row['selected_global_macro']=stats([r['global_endpoint']['balanced_accuracy'] for r in selected])
        row['local_only_macro']=stats([r['balanced_accuracy'] for r in local])
        out['conditions'].append(row)
        desc=f"{task_name[dataset]}, {condition['n']//1000}k, {condition['epsilon']:g}"
        short={'center':'ctr','matching':'match','raw':'raw'}[choice['decoder']]
        rows.append([desc,f"{choice['k']}/{choice['estimator']}/{short}",cell(glob),cell(fl),cell(private),cell(hg)])
        extra.append([desc,cell(ctl),cell(ctstudent) if ctstudent else '--',cell(student) if student else '--',cell(hl),cell(loc)])
    table(paper/'tab-selector-e2e.tex',
        r'End-to-end public selection, ten fresh paired seeds (8501--8510). K/estimator/decoder freezes before private evaluation. '
        r'Global: nearest public centroid using the third protected label release. Client: 180-step private local endpoint. '
        r'Fixed uses taxonomy K=C, M, centered consensus. Head: final DP-Head broadcast. '
        r'Entries are accuracy (Usenet: balanced accuracy), \%, mean$\pm$sample SD; both head endpoints and other readouts appear in Table~\ref{tab:selector-readouts}.',
        'tab:selector-e2e','lcrrrr',['Task, clients, $\\varepsilon$','K/e/d','Chosen global','Fixed global','Chosen client','Head global'],rows)
    table(paper/'tab-selector-readouts.tex',
        r'Complementary endpoints for exactly the paired cohorts of Table~\ref{tab:selector-e2e}, mean$\pm$sample SD (\%). '
        r'Public students train for 180 additional server steps on final protected targets, initialized independently of private models. '
        r'CT denotes per-example DP-FedCT. Head local precedes its third broadcast with no additional adaptation.',
        'tab:selector-readouts','lrrrrr',['Task, clients, $\\varepsilon$','CT client','CT student','SDD student','Head local','Local-only'],extra)
    web=next((r for r in out['conditions'] if r['dataset']=='web'),None)
    if web:
        selected=records('public-selected',dict(dataset='web',n=3000,epsilon=1.))
        with np.load(ROOT/'inputs/web.npz',allow_pickle=False) as b:
            xt=b['x_test'];yt=b['y_test'];xp=b['x_proxy']
        import hashlib
        digest=hashlib.sha256(xp.tobytes()).hexdigest()[:12]
        centres=np.load(ROOT/'inputs/partitions'/f'{digest}_K{web["choice"]["k"]}_semantic.npz')['centres']
        assignment=((xt[:,None]-centres[None])**2).sum(2).argmin(1)
        arrays={}
        for role,label in [('public-selected','SDD client'),('per-example','CT client'),('head','Head global'),('local','Local-only')]:
            rr=records(role,dict(dataset='web',n=3000,epsilon=1.))
            arrays[label]=np.array([r['per_class_recall'] for r in rr])
        nn=[]
        for r in selected:
            labels=np.load(ROOT/r['diagnostic_file'])['r3_labels'];pred=labels[assignment]
            nn.append([(pred[yt==c]==c).mean() for c in range(6)])
        arrays['SDD NN']=np.array(nn)
        if all(key(r) in readers for r in selected):arrays['SDD student']=np.array([readers[key(r)]['per_class_recall'] for r in selected])
        out['web_recall']={name:dict(mean=values.mean(0).tolist(),per_seed_minimum=stats(values.min(1))) for name,values in arrays.items()}
        rrows=[[name]+[f'{v*100:.1f}' for v in row['mean']]+[cell(row['per_seed_minimum'])] for name,row in out['web_recall'].items()]
        table(paper/'tab-web-new-recall.tex',
            r'Fresh ten-seed Usenet class allocation. Per-family recall means (\%); the last column takes the six-class minimum within each seed before the mean$\pm$sample SD. '
            r'Client recall vectors first average client models within a seed. Public NN uses the final regional labels.',
            'tab:web-new-recall','lrrrrrrr',['Endpoint','comp','rec','sci','talk','alt/soc','misc','Min./seed'],rrows)
    for a,b in [(0,0),(32,32),(64,64),(96,96),(128,128),(64,128)]:
        condition=dict(dataset='cifar10',n=10000,epsilon=.1,proxy_r2=a,proxy_r3=b)
        try:r=records('training-mix',condition)
        except KeyError:
            if partial:continue
            raise
        out['mix'].append(dict(proxy_r2=a,proxy_r3=b,seeds=list(range(8601,8611)),accuracy=stats([x['accuracy'] for x in r]),
            global_accuracy=stats([x['global_endpoint']['accuracy'] for x in r]),
            first_release_accuracy=stats([x['trace'][0]['diagnostics']['global_classifier_accuracy'] for x in r])))
    fixed=next((r for r in out['mix'] if r['proxy_r2']==64 and r['proxy_r3']==64),None)
    if fixed:
        replay=json.loads((ROOT/'analysis/centroid-replays.json').read_text())
        old_global=[r['accuracy'] for r in replay if r['dataset']=='cifar10' and r['n']==10000 and r['decoder']=='matching' and r['release']==3]
        parent=ROOT.parent/'release-granularity-20261019/results'
        old_seeds=list(range(801,806))+list(range(8201,8206))
        old_local=[json.loads((parent/f'cifar10_10000_0.1_{s}_SDD_10_matching.json').read_text())['accuracy'] for s in old_seeds]
        assert len(old_global)==10
        out['fixed_taxonomy_no_labels']=dict(current_ten=fixed,
            pooled_global=stats(old_global+fixed['global_accuracy']['values']),
            pooled_local=stats(old_local+fixed['accuracy']['values']),
            seeds=old_seeds+list(range(8601,8611)),group_sizes=[5,5,10],
            prior_global=stats(old_global),prior_local=stats(old_local))
        last=next(r for r in out['mix'] if r['proxy_r2']==128 and r['proxy_r3']==128)
        out['pure_proxy_gain']=paired(last['accuracy']['values'],fixed['accuracy']['values'])
    natural_rows=[]
    data=np.load(ROOT/'inputs/femnist62.npz',allow_pickle=False)
    for k in [31,62,124,248,500]:
        # Partition cache is public and deterministic; avoid importing GPU training here.
        import hashlib
        digest=hashlib.sha256(data['x_proxy'].tobytes()).hexdigest()[:12]
        par=np.load(ROOT/'inputs/partitions'/f'{digest}_K{k}_semantic.npz');ids=par['ids']
        proxy_purity=float(sum(np.bincount(data['y_proxy'][ids==i],minlength=62).max() for i in range(k))/len(ids))
        profile=np.load(ROOT/'public-selection'/f'femnist62_K{k}_M.npz')
        validation_purity=float(profile['validation_class_weights'].max(1).sum())
        for n in [1000,2000]:
            for epsilon in [.1,1.]:
                condition=dict(dataset='femnist62',n=n,epsilon=epsilon,k=k)
                try:rr=records('natural-resolution-grid',condition)
                except KeyError:
                    if partial:continue
                    raise
                row=dict(n=n,epsilon=epsilon,k=k,proxy_purity=proxy_purity,validation_purity=validation_purity,
                    seeds=list(range(8501,8506)),global_accuracy=stats([r['global_endpoint']['accuracy'] for r in rr]),
                    local_accuracy=stats([r['accuracy'] for r in rr]),global_macro=stats([r['global_endpoint']['balanced_accuracy'] for r in rr]),
                    local_macro=stats([r['balanced_accuracy'] for r in rr]))
                out['natural_resolution'].append(row)
                if n==2000:natural_rows.append([str(k),str(epsilon),f'{proxy_purity:.3f}',f'{validation_purity:.3f}',cell(row['global_accuracy']),cell(row['local_accuracy'])])
    table(paper/'tab-femnist-resolution.tex',
        r'Natural-writer resolution grid: 2000 actual writers, mean regional votes and centered decoding at every K, five paired seeds. '
        r'Proxy and validation purities measure their respective constant regional classifiers. '
        r'The public rule may choose a different estimator or decoder; it is evaluated separately.',
        'tab:femnist-resolution','rrrrrr',['K',r'$\varepsilon$','Proxy pur.','Val. pur.','Global acc.','Client acc.'],natural_rows)
    references=[]
    for variant in ['matched_180','selected']:
        rr=[results.get(f'femnist62_Public-only_{variant}_{s}') for s in range(8501,8511)]
        if not all(rr):continue
        row=dict(method='Public-only '+('180' if variant=='matched_180' else 'selected'),
            n=None,steps=rr[0]['optimizer_steps'],lr=rr[0]['lr'],accuracy=stats([r['accuracy'] for r in rr]),
            balanced_accuracy=stats([r['balanced_accuracy'] for r in rr]),development_labels=5000,coordinates=0)
        out['public_references'].append(row)
    for n in [1000,2000]:
        try:rr=records('public-source-local',dict(dataset='femnist62',n=n))
        except KeyError:
            if partial:continue
            raise
        out['public_references'].append(dict(method=f'Local+Public {n//1000}k',n=n,steps=180,accuracy=stats([r['accuracy'] for r in rr]),
            balanced_accuracy=stats([r['balanced_accuracy'] for r in rr]),development_labels=5000,coordinates=0))
    for r in out['public_references']:references.append([r['method'],str(r['steps']),cell(r['accuracy']),cell(r['balanced_accuracy'])])
    table(paper/'tab-femnist-public.tex',
        r'FEMNIST supervision-source references, ten seeds, 5000 public development labels. '
        r'Public validation chooses public learning rate and training length; final-test arrays do not participate. '
        r'Local+Public retains 180 private-client steps and 64/64 mixing, taking proxy targets from the selected public teacher. '
        r'All entries require zero private report coordinates.',
        'tab:femnist-public','lrrr',['Reference','Steps','Accuracy','Macro recall'],references,False)
    query_ids=[f'cifar10_10000_0.1_{s}_SDD_10_center_R_64_64' for s in range(8201,8206)]
    if all(identifier in results for identifier in query_ids):
        sem=[results[i] for i in query_ids]
        old=ROOT.parent/'release-granularity-20261019/results'
        uniform=[json.loads((old/f'cifar10_10000_0.1_{s}_uniform-query_10_center.json').read_text()) for s in range(8201,8206)]
        assert all(a['population']==b['population'] for a,b in zip(sem,uniform))
        out['matched_query_control']=dict(semantic=stats([r['accuracy'] for r in sem]),uniform=stats([r['accuracy'] for r in uniform]),
            gain=paired([r['accuracy'] for r in sem],[r['accuracy'] for r in uniform]),seeds=list(range(8201,8206)),
            model_queries_per_client_round=10,decoder='center',report_coordinates=100,optimizer_steps=180)
    save(ROOT/'summary.json',out);(paper/'selector-data.json').write_text(json.dumps(out,indent=2)+'\n')
    values={}
    def percent(name,value):values[name]=f'{value*100:.1f}\\%'
    if fixed:
        percent('FixedGlobal',out['fixed_taxonomy_no_labels']['pooled_global']['mean'])
        percent('FixedGlobalSd',out['fixed_taxonomy_no_labels']['pooled_global']['sd'])
        percent('FixedClient',out['fixed_taxonomy_no_labels']['pooled_local']['mean'])
        percent('MixHalf',fixed['accuracy']['mean']);percent('MixFull',last['accuracy']['mean'])
        percent('MixNN',fixed['global_accuracy']['mean'])
        values['MixGain']=f"{out['pure_proxy_gain']['mean']*100:.1f}"
        values['MixGainCI']=f"[{out['pure_proxy_gain']['ci_low']*100:.1f},{out['pure_proxy_gain']['ci_high']*100:.1f}]"
        final=next(r for r in out['mix'] if r['proxy_r2']==64 and r['proxy_r3']==128)
        percent('MixFinal',final['accuracy']['mean'])
    for r in out['conditions']:
        if r['dataset']=='cifar10' and r['n']==10000:
            for macro,field in [('CifChosenGlobal','selected_global'),('CifChosenClient','selected_local'),('CifHead','head_global'),('CifCTClient','ct_local')]:percent(macro,r[field]['mean'])
            if r['student']:percent('CifStudent',r['student']['mean'])
            if r['ct_student']:percent('CifCTStudent',r['ct_student']['mean'])
        if r['dataset']=='web':
            percent('WebChosenGlobal',r['selected_global']['mean']);percent('WebChosenClient',r['selected_local']['mean'])
            percent('WebCTClient',r['ct_local']['mean']);values['WebGain']=f"{r['local_gain_vs_ct']['mean']*100:.1f}"
            values['WebGainCI']=f"[{r['local_gain_vs_ct']['ci_low']*100:.1f},{r['local_gain_vs_ct']['ci_high']*100:.1f}]"
        if r['dataset']=='femnist62' and r['n']==2000:
            prefix='FemLow' if r['epsilon']==.1 else 'FemHigh'
            percent(prefix+'Global',r['selected_global']['mean']);percent(prefix+'Client',r['selected_local']['mean'])
            percent('FemLocal',r['local_only']['mean'])
            percent(prefix+'Macro',r['selected_client_macro']['mean']);percent('FemLocalMacro',r['local_only_macro']['mean'])
    for r in out['public_references']:
        if r['method']=='Public-only 180':percent('FemPublicMatched',r['accuracy']['mean'])
        if r['method']=='Public-only selected':percent('FemPublicSelected',r['accuracy']['mean'])
        if r['method']=='Local+Public 2k':percent('FemLocalPublic',r['accuracy']['mean'])
    if 'matched_query_control' in out:
        r=out['matched_query_control'];percent('QuerySemantic',r['semantic']['mean']);percent('QueryUniform',r['uniform']['mean'])
        values['QueryGain']=f"{r['gain']['mean']*100:.1f}"
        values['QueryGainCI']=f"[{r['gain']['ci_low']*100:.1f},{r['gain']['ci_high']*100:.1f}]"
    expected=['FixedGlobal','FixedGlobalSd','FixedClient','MixHalf','MixFull','MixNN','MixGain','MixGainCI','MixFinal',
        'CifChosenGlobal','CifChosenClient','CifHead','CifCTClient','CifStudent','CifCTStudent','WebChosenGlobal',
        'WebChosenClient','WebCTClient','WebGain','WebGainCI','FemLowGlobal','FemLowClient','FemHighGlobal',
        'FemHighClient','FemLocal','FemPublicMatched','FemPublicSelected','FemLocalPublic']
    expected+=['FemLowMacro','FemHighMacro','FemLocalMacro']
    expected+=['QuerySemantic','QueryUniform','QueryGain','QueryGainCI']
    if partial:
        for name in expected:values.setdefault(name,r'\textit{pending}')
    else:assert all(name in values for name in expected),set(expected)-set(values)
    (paper/'selector-values.tex').write_text('% Generated numerical prose values.\n'+
        '\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in values.items())+'\n')
    plot(paper,out)
    print('Generated',len(out['conditions']),'end-to-end settings;',len(out['mix']),'mix settings;',len(out['natural_resolution']),'natural K cells.')


def plot(paper,data):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'pdf.fonttype':42})
    folder=paper/'figures';folder.mkdir(exist_ok=True)
    def finish(fig,name):
        fig.tight_layout(pad=.6,w_pad=1.8);fig.savefig(folder/name,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.5))
    row=next((r for r in data['conditions'] if r['dataset']=='cifar10' and r['n']==10000),None)
    if row and row['student'] and row['ct_student']:
        vals=[row['selected_global'],row['student'],row['selected_local'],row['ct_student'],row['head_global']]
        ax[0].bar(np.arange(5),[r['mean']*100 for r in vals],yerr=[r['sd']*100 for r in vals],
            color=['#237a57','#63a37d','#9ac3a3','#4477aa','#cc8833'],capsize=2,width=.65)
        ax[0].set_xticks(range(5),['Region\nNN','Region\nstudent','Region\nclient','CT\nstudent','Head\nglobal'])
        ax[0].set_ylim(78,96)
    ax[0].set(title='(a) Readouts of protected supervision',ylabel='CIFAR-10 accuracy (%)')
    mixes=data['mix']
    if mixes:
        x=np.arange(len(mixes));ax[1].errorbar(x,[r['accuracy']['mean']*100 for r in mixes],
            yerr=[r['accuracy']['sd']*100 for r in mixes],fmt='o-',color='#237a57',capsize=2,ms=4,label='Client endpoint')
        ax[1].axhline(mixes[0]['first_release_accuracy']['mean']*100,color='#cc8833',ls='--',lw=1,label='First-release NN')
        ax[1].plot(x,[r['global_accuracy']['mean']*100 for r in mixes],'s:',color='#4477aa',ms=3,lw=1,label='Final-release NN')
        ax[1].set_xticks(x,[f"{r['proxy_r2']/128:g}/{r['proxy_r3']/128:g}" for r in mixes])
        ax[1].legend(fontsize=7,frameon=False,loc='lower right')
    ax[1].set(title='(b) Fixed 180-step client budget',ylabel='Accuracy (%)',xlabel='Proxy fraction in rounds 2 / 3',ylim=(38,96))
    for a in ax:a.grid(axis='y',alpha=.15)
    finish(fig,'fig-readouts.pdf')
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.45))
    colors={'cifar10':'#237a57','mnist':'#4477aa','web':'#aa3377','femnist62':'#cc8833'}
    for dataset,color in colors.items():
        rr=[r for r in data['conditions'] if r['dataset']==dataset]
        if not rr:continue
        ax[0].errorbar([r['public_predicted_validation']*100 for r in rr],
            [r['actual_first_release_validation']['mean']*100 for r in rr],
            yerr=[r['actual_first_release_validation']['sd']*100 for r in rr],fmt='o',color=color,ms=4,capsize=2,label=dataset)
    ax[0].plot([0,100],[0,100],':',color='.5');ax[0].set(xlim=(0,100),ylim=(0,100),
        xlabel='Public-pilot predicted quality (%)',ylabel='Actual private-release quality (%)',title='(a) Public-to-private transfer')
    ax[0].legend(fontsize=6,frameon=False,loc='upper left')
    rr=data['conditions'];x=np.arange(len(rr))
    ax[1].bar(x,[r['private_public_weighted_TV']['mean'] for r in rr],
        yerr=[r['private_public_weighted_TV']['sd'] for r in rr],color=[colors[r['dataset']] for r in rr],capsize=2)
    labels=[{'cifar10':'C','mnist':'M','web':'W','femnist62':'F'}[r['dataset']]+f"{r['n']//1000}k/{r['epsilon']:g}" for r in rr]
    ax[1].set_xticks(x,labels,rotation=45,ha='right',fontsize=7)
    ax[1].set(title='(b) Clean vote-distribution shift',ylabel='Public-size-weighted total variation')
    for a in ax:a.grid(axis='y',alpha=.15)
    finish(fig,'fig-selection.pdf')
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.45))
    rr=data['natural_resolution']
    rows=sorted([r for r in rr if r['n']==2000 and r['epsilon']==.1],key=lambda r:r['k'])
    if rows:
        for field,label,color in [('proxy_purity','Proxy purity','#237a57'),('validation_purity','Validation purity','#4477aa')]:
            ax[0].plot([r['k'] for r in rows],[r[field] for r in rows],'o-',color=color,ms=3,label=label)
    for epsilon,color in [(.1,'#4477aa'),(1.,'#cc8833')]:
        rows=sorted([r for r in rr if r['n']==2000 and r['epsilon']==epsilon],key=lambda r:r['k'])
        if rows:
            ax[1].errorbar([r['k'] for r in rows],[r['global_accuracy']['mean']*100 for r in rows],
                yerr=[r['global_accuracy']['sd']*100 for r in rows],fmt='o-',ms=3,color=color,capsize=2,label=rf'M, center: $\epsilon={epsilon:g}$')
        choice=next((r for r in data['conditions'] if r['dataset']=='femnist62' and r['n']==2000 and r['epsilon']==epsilon),None)
        if choice:ax[1].scatter(choice['choice']['k'],choice['selected_global']['mean']*100,marker='*',s=65,color=color,zorder=5)
    for a in ax:
        a.set_xscale('log');a.set_xticks([31,62,124,248,500],[31,62,124,248,500]);a.set_xlabel('Public regions K');a.grid(alpha=.15);a.legend(fontsize=7,frameon=False)
    ax[0].set(title='(a) Natural-writer geometry',ylabel='Constant-region purity',ylim=(.2,.8))
    ax[1].set(title='(b) Actual 2000-writer releases',ylabel='Public classifier accuracy (%)',ylim=(0,50))
    finish(fig,'fig-femnist-resolution.pdf')
    (folder/'selector-figure-data.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();export(a.paper,a.partial)
