"""All manuscript numbers derive from the frozen study's actual records."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,PARENT,CLASSES,SETTINGS,SEEDS,release_key,save

def stats(a):
    a=np.asarray(a,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.,
        p05=float(np.quantile(a,.05)),p50=float(np.median(a)),p95=float(np.quantile(a,.95)),values=a.tolist())

def paired(a,b):
    x=np.asarray(a)-np.asarray(b);s=stats(x);half=t.ppf(.975,len(x)-1)*s['sd']/np.sqrt(len(x))
    return dict(**s,ci_low=s['mean']-half,ci_high=s['mean']+half)

def crossed(a):
    a=np.asarray(a,dtype=float);l,s=a.shape;rng=np.random.default_rng(982019)
    left=rng.multinomial(l,np.full(l,1/l),size=2000)/l
    right=rng.multinomial(s,np.full(s,1/s),size=2000)/s
    boots=np.einsum('bi,ij,bj->b',left,a,right)
    return dict(mean=float(a.mean()),label_draws=l,cohort_seeds=s,
        label_draw_sd=float(a.mean(1).std(ddof=1)) if l>1 else 0.,
        cohort_mean_sd=float(a.mean(0).std(ddof=1)),
        ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),
        label_mean_p05=float(np.quantile(a.mean(1),.05)),label_mean_p95=float(np.quantile(a.mean(1),.95))),boots

def cell(s):return f"{100*s['mean']:.2f}$\\pm${100*s['sd']:.2f}"
def number(s):return f"{100*s['mean']:.2f}"
def table(path,caption,label,columns,head,rows,wide=True):
    env='table*' if wide else 'table'
    value=f'\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\\small\n\\centering\n'
    value+=f'\\begin{{tabular}}{{@{{}}{columns}@{{}}}}\\toprule\n'+' & '.join(head)+r' \\'+'\n\\midrule\n'
    value+='\n'.join(' & '.join(row)+r' \\' for row in rows)
    value+=f'\n\\bottomrule\\end{{tabular}}\n\\end{{{env}}}\n';path.write_text(value)

def export(paper,partial=False):
    paper.mkdir(parents=True,exist_ok=True);(paper/'figures').mkdir(exist_ok=True)
    rows={p.stem:json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))}
    old=json.loads((PARENT/'STUDY-SCHEDULE.json').read_text())['design']
    oldr={p.stem:json.loads(p.read_text()) for p in (PARENT/'results').glob('*.json')}
    students={p.stem:json.loads(p.read_text()) for p in (ROOT/'analysis/students').glob('*.json')}
    pub=[json.loads(p.read_text()) for p in sorted((ROOT/'public').glob('*_m*.json'))]
    canonical=json.loads((ROOT/'analysis/canonical-regional-readouts.json').read_text())
    def regional_metric(key,mode,release='1',parent=False):
        source='experiments/'+(PARENT.name if parent else ROOT.name)+'/results/'+key+'.json'
        return canonical['records'][source]['releases'][str(release)][mode]
    out=dict(single_release=[],frontiers=[],fewshot=[],geometry=[],bounds=json.loads((ROOT/'analysis/bound-diagnostics.json').read_text()),
        primary_single_release_records=len(rows),public_label_sets=len(pub),public_students=len(students),
        original_primary_records=2336,random_draws=100,stratified_draws=50,cohort_seeds=SEEDS)
    names={'cifar10':'CIFAR','mnist':'MNIST','web':'Usenet','femnist62':'FEMNIST'}
    def source(d,n,e,s,steps,method,k=0,est='-'):
        return rows[release_key(d,n,e,s,steps,method,k,est)]
    def student(d,n,e,s,steps,method,k,est,decoder):
        return students[release_key(d,n,e,s,steps,method,k,est)+'_'+decoder]
    trows=[];srows=[];seedrows=[];variants=[]
    for d,n,e in SETTINGS:
        c=CLASSES[d];metric='balanced_accuracy' if d=='web' else 'accuracy'
        old_ids=[r['job'] for r in old['roles'] if r['role']=='fixed-center' and r['condition']==dict(dataset=d,n=n,epsilon=e)]
        try:
            oldvalues=[regional_metric(i,'actual',3,True)[metric] for i in old_ids]
            first=[regional_metric(i,'actual',1,True)[metric] for i in old_ids]
            new={f'{steps}_{mode}':stats([regional_metric(release_key(d,n,e,s,steps,'regional',c,'M'),mode)[metric] for s in SEEDS])
                for steps in [60,180] for mode in ['center','coverage','raw']}
            heads=stats([source(d,n,e,s,180,'head-fixed')['endpoints']['broadcast'][metric] for s in SEEDS])
            tuned=stats([source(d,n,e,s,180,'head')['endpoints']['broadcast'][metric] for s in SEEDS])
            std={mode:stats([student(d,n,e,s,180,'regional',c,'M',mode)[metric] for s in SEEDS]) for mode in ['center','coverage']}
            ct=stats([student(d,n,e,s,180,'per-example',0,'-','center')[metric] for s in SEEDS])
        except KeyError:
            if partial:continue
            raise
        record=dict(dataset=d,n=n,epsilon=e,metric=metric,T3_fixed=stats(oldvalues),T3_first=stats(first),T1=new,head_fixed=heads,head_parent_tuned=tuned,
            students=std,ct_student=ct,T1_60_vs_T3=paired(new['60_center']['values'],oldvalues),
            T1_60_vs_T3_first=paired(new['60_center']['values'],first),
            T1_180_vs_T3=paired(new['180_center']['values'],oldvalues))
        out['single_release'].append(record)
        label=f'{names[d]}, {n//1000}k, {e:g}'
        trows.append([label,cell(record['T3_first']),cell(new['60_center']),cell(record['T3_fixed']),cell(new['180_center']),cell(new['180_coverage']),cell(heads)])
        srows.append([label,cell(std['center']),cell(std['coverage']),cell(ct),cell(tuned)])
        vr={f'{est}_{mode}':stats([regional_metric(release_key(d,n,e,s,180,'regional',c,est),mode)[metric] for s in SEEDS])
            for est in ['M','R'] for mode in ['center','coverage']}
        variants.append([label]+[cell(vr[key]) for key in ['M_center','R_center','M_coverage','R_coverage']])
        for s in SEEDS:
            rr=source(d,n,e,s,180,'regional',c,'M')
            seedrows.append(dict(dataset=d,n=n,epsilon=e,seed=s,
                T3=oldvalues[SEEDS.index(s)],T1_60=new['60_center']['values'][SEEDS.index(s)],
                T1_180=new['180_center']['values'][SEEDS.index(s)],T1_180_coverage=new['180_coverage']['values'][SEEDS.index(s)],
                T1_180_class_recall=regional_metric(release_key(d,n,e,s,180,'regional',c,'M'),'center')['per_class_recall']))
    out['single_release_per_seed']=seedrows
    table(paper/'tab-rounds.tex',r'One versus three releases on the same ten actual cohorts. '
        r'Fixed $K=C$, M, centered; Cov is a separate coverage-constrained decoder of the same matrix. '
        r'T=1 spends the full $(\varepsilon,10^{-5})$ budget after 60/180 private-only SGD steps; T=3 uses 180 steps with feedback. '
        r'All entries are public classifiers (NN; Head broadcast), mean$\pm$sample SD, \%; Web uses balanced accuracy. '
        r'Head fixes clipping at $B=1$ with no public labels.',
        'tab:rounds','lrrrrrr',['Task, clients, $\\varepsilon$','T=3 first/60','T=1/60','T=3 final/180','T=1/180','T=1/180 Cov','Head/180'],trows)
    table(paper/'tab-round-students.tex',r'Additional public readouts for Table~\ref{tab:rounds}, 180 pre-release private steps. '
        r'Every student adds 180 public SGD steps from independent initialization, with no private model state. '
        r'The final column is a head broadcast with the preceding public-label-tuned clipping configuration, separately from fixed $B=1$. '
        r'Mean$\pm$sample SD, \%; Web uses balanced accuracy.',
        'tab:round-students','lrrrr',['Task, clients, $\\varepsilon$','SDD student','SDD Cov student','CT student','Public-tuned Head'],srows)
    table(paper/'tab-release-variants.tex',r'Regional estimators and decoder allocation at T=1/180, K=C, ten paired cohorts. '
        r'M pools regional probabilities; R queries the public centroid representative. M/R are separate protected protocols; centered/coverage decoding shares each source matrix. '
        r'Every entry is a public NN classifier, mean$\pm$sample SD (\%); Web uses balanced accuracy. No test metric routes a deployment choice.',
        'tab:release-variants','lrrrr',['Task, clients, $\\varepsilon$','M/center','R/center','M/coverage','R/coverage'],variants)
    for d,n,e in SETTINGS:
        metric='balanced_accuracy' if d=='web' else 'accuracy'
        for scheme in ['random','stratified','geometry']:
            group=[];bootstrap=[];all_bootstrap=[];private_bootstrap=[]
            for q in [1,3,5,10,20,50]:
                r=sorted([x for x in pub if x['dataset']==d and x['scheme']==scheme and x['q']==q],key=lambda x:x['draw'])
                expected=100 if scheme=='random' else 50 if scheme=='stratified' else 1
                if not r:continue
                if len(r)!=expected:
                    if partial:continue
                    raise ValueError((d,scheme,q,len(r),expected))
                public=np.array([x['public_portfolio'][metric] for x in r]);h=[];g=[];accept=[];minimum=[]
                for sample in r:
                    hrow=sorted([z for z in sample['hybrid'] if z['n']==n and z['epsilon']==e],key=lambda z:z['seed'])
                    assert [a['seed'] for a in hrow]==SEEDS
                    h.append([z['ungated'][metric] for z in hrow]);g.append([z['gated'][metric] for z in hrow])
                    accept.append([z['gate']['private_augmentation_published'] for z in hrow])
                    minimum.append([z['ungated']['minimum_recall'] for z in hrow])
                hybrid,hboots=crossed(h);gated,gboots=crossed(g);gain,boots=crossed(np.array(h)-public[:,None]);ggain,_=crossed(np.array(g)-public[:,None])
                private=np.array([regional_metric(release_key(d,n,e,s,180,'regional',CLASSES[d],'M'),'center')[metric] for s in SEEDS])
                pgain,pboots=crossed(np.array(h)-private[None,:])
                baseline_stats={};baseline_gains={};baseline_boots=[]
                for name in r[0]['baselines']:
                    baseline=np.array([sample['baselines'][name][metric] for sample in r])
                    baseline_stats[name]=stats(baseline)
                    value,b=crossed(np.array(h)-baseline[:,None]);baseline_gains[name]=value;baseline_boots.append(b)
                graph_values=[]
                for sample in r:
                    choices=[x for name,x in sample['baselines'].items() if name.startswith('graph-')]
                    best=max(range(len(choices)),key=lambda i:(choices[i]['loo_score'],-i));graph_values.append(choices[best][metric])
                record=dict(dataset=d,n=n,epsilon=e,scheme=scheme,q=q,m=q*CLASSES[d],metric=metric,
                    public=stats(public),hybrid=hybrid,gated=gated,gain=gain,gated_gain=ggain,gain_vs_private_center=pgain,
                    acceptance=float(np.mean(accept)),missing_class_probability=float(np.mean([x['missing_classes']>0 for x in r])),
                    graph=stats(graph_values),fixed_public_baselines=baseline_stats,gain_vs_public_methods=baseline_gains,
                    hybrid_minimum_recall=float(np.mean(minimum)),public_minimum_recall=float(np.mean([x['public_portfolio']['minimum_recall'] for x in r])),
                    selected_public_counts={name:sum(x['selected_public']==name for x in r) for name in sorted({x['selected_public'] for x in r})})
                group.append(record);bootstrap.append(boots);all_bootstrap.append(np.stack(baseline_boots));private_bootstrap.append(pboots)
            if len(group)==6:
                b=np.array(bootstrap);means=np.array([x['gain']['mean'] for x in group]);sds=b.std(1,ddof=1)
                critical=float(np.quantile(np.max(np.abs((b-means[:,None])/np.maximum(sds[:,None],1e-12)),axis=0),.95))
                for i,row in enumerate(group):row['gain'].update(simultaneous_ci_low=float(means[i]-critical*sds[i]),simultaneous_ci_high=float(means[i]+critical*sds[i]),six_budget_critical=critical)
                b=np.array(all_bootstrap);mu=np.array([[x['mean'] for x in row['gain_vs_public_methods'].values()] for row in group]);sd=b.std(2,ddof=1)
                critical=float(np.quantile(np.max(np.abs((b-mu[:,:,None])/np.maximum(sd[:,:,None],1e-12)),axis=(0,1)),.95))
                for i,row in enumerate(group):
                    lower=mu[i]-critical*sd[i];upper=mu[i]+critical*sd[i]
                    for j,(name,value) in enumerate(row['gain_vs_public_methods'].items()):value.update(simultaneous_ci_low=float(lower[j]),simultaneous_ci_high=float(upper[j]))
                    best=max(row['fixed_public_baselines'],key=lambda name:row['fixed_public_baselines'][name]['mean'])
                    row['gain_vs_strongest_mean_public']=dict(mean=float(mu[i].min()),simultaneous_ci_low=float(lower.min()),
                        simultaneous_ci_high=float(upper.min()),strongest_mean_public_method=best,
                        scope='simultaneous across all twelve frozen public baselines and all six label budgets within this task/cohort/acquisition setting',critical=critical)
                b=np.array(private_bootstrap);mu=np.array([x['gain_vs_private_center']['mean'] for x in group]);sd=b.std(1,ddof=1)
                critical=float(np.quantile(np.max(np.abs((b-mu[:,None])/np.maximum(sd[:,None],1e-12)),axis=0),.95))
                for i,row in enumerate(group):row['gain_vs_private_center'].update(simultaneous_ci_low=float(mu[i]-critical*sd[i]),simultaneous_ci_high=float(mu[i]+critical*sd[i]))
            out['frontiers']+=group
    for d in ['cifar10','mnist']:
        for q in [1,3,5]:
            r=sorted([x for x in pub if x['dataset']==d and x['scheme']=='stratified' and x['q']==q],key=lambda x:x['draw'])
            if len(r)!=50:
                if partial:continue
                raise ValueError((d,q,len(r)))
            item=dict(dataset=d,q=q,m=q*10)
            for family,prefix in [('naming','cluster-majority'),('coverage','cluster-coverage'),('prototype','prototype'),('ridge','ridge-'),('graph','graph-')]:
                values=[]
                for sample in r:
                    options=[(name,x) for name,x in sample['baselines'].items() if name.startswith(prefix)]
                    choice=max(range(len(options)),key=lambda i:(options[i][1]['loo_score'],-i))
                    values.append(options[choice][1]['accuracy'])
                item[family]=stats(values)
            out['fewshot'].append(item)
    frows=[[names[r['dataset']],str(r['q'])]+[cell(r[k]) for k in ['naming','coverage','prototype','ridge','graph']] for r in out['fewshot']]
    table(paper/'tab-fewshot.tex',r'Public-only naming and learning with exactly 1/3/5 labels per class, 50 independent stratified draws. '
        r'Ridge and graph parameters use leave-one-out selection entirely inside the stated label budget. '
        r'All use the same public representations and unlabeled development/proxy features; test queries are inductive. Mean$\pm$sample SD accuracy, \%.',
        'tab:fewshot','lrrrrrr',['Task','Labels/class','Naming','Coverage','Class centroid','Ridge','Graph SSL'],frows)
    for row in out['frontiers']:
        if row['scheme']=='geometry' and (row['dataset'],row['n'],row['epsilon']) in [('cifar10',10000,.1),('mnist',10000,.1),('web',3000,1.),('femnist62',2000,1.)]:out['geometry'].append(row)
    grows=[[names[r['dataset']],str(r['m']),number(r['public']),number(r['hybrid']),f"{100*r['acceptance']:.1f}"] for r in out['geometry'] if r['q'] in [1,3,5]]
    table(paper/'tab-geometry-labels.tex',r'Feature-selected annotation is a separate acquisition condition. '
        r'Only the stated m development points receive labels; the public graph uses no labels to choose them. '
        r'Public is the within-budget selected portfolio; Hybrid uses ten paired private seeds. Pub. is the acceptance rate of private augmentation under the declared heuristic gate. Accuracy, \%; balanced accuracy for Web.',
        'tab:geometry-labels','lrrrr',['Task','m','Public','Hybrid','Pub. (\\%)'],grows)
    anchors=[]
    for d,n,e in [('cifar10',1000,.1),('cifar10',10000,.1),('mnist',10000,.1),('web',3000,1.),('femnist62',2000,1.)]:
        rr=next((r for r in out['frontiers'] if (r['dataset'],r['n'],r['epsilon'],r['scheme'],r['q'])==(d,n,e,'random',1)),None)
        if rr is None:continue
        b=max(rr['fixed_public_baselines'].values(),key=lambda x:x['mean'])
        anchors.append([f'{names[d]} {n//1000}k',str(CLASSES[d]),number(b),number(rr['hybrid'])])
    table(paper/'tab-budget-anchors.tex',r'Random public-label operating points at m=C, 100 label draws crossed with ten cohorts. '
        r'Public is the strongest mean of the twelve frozen candidates (a diagnostic envelope, not test-driven deployment selection). '
        r'Fusion is the declared LOO rule. Accuracy, \%; balanced accuracy for Usenet.',
        'tab:budget-anchors','lrrr',['Task/cohort','m','Public','Fusion'],anchors,wide=False)
    table(paper/'tab-resource-map.tex',r'Label resources and replication units. $q\in\{1,3,5,10,20,50\}$. '
        r'Cohorts use seeds 8501--8510 paired across release protocols; label draws use independent frozen tapes.',
        'tab:resource-map','lll',['Condition','Labels','Replications'],[
            ['Fixed T=1 / T=3','0','10 cohorts/setting'],
            ['Random public / fusion','Cq','100 draws $\\times$ 10 cohorts'],
            ['Supplied balanced (C/M)','Cq','50 draws $\\times$ 10 cohorts'],
            ['Feature-selected labels','Cq','1 pool $\\times$ 10 cohorts'],
            ['Retained teacher pilot','dev. + val.','10 cohorts/setting']],wide=False)
    bounds=[]
    for d,n,e in SETTINGS:
        rr=[r for r in out['bounds'] if r['dataset']==d and r['n']==n and r['epsilon']==e]
        decoder=rr[0]['decoder'];applicable=[r['coupled_bound'] for r in rr if r['coupled_bound'] is not None]
        bounds.append([f'{names[d]}, {n//1000}k, {e:g}',decoder,f"{np.mean([r['sup_shift_unprojected'] for r in rr]):.3f}",
            '1.000' if applicable and min(applicable)==1 else 'n/a',f"{np.mean([r['raw_private_winner_weighted_flip_bound'] for r in rr]):.3f}"])
    table(paper/'tab-bound-values.tex',r'Numerical scope of the preceding conditional stability lemma, first release, ten cohorts per setting. '
        r'$\delta_q$ is the unprojected matrix sup distance. Coupled bounds are 1 in every applicable cohort; raw decoding uses the separate regional vote-gap bound. '
        r'The last column is the public-region-weighted raw-winner flip bound, not a classification error bound. Clean private votes enter these diagnostics only.',
        'tab:bound-values','llrrr',['Task, clients, $\\varepsilon$','Decoder','$\\delta_q$','Coupled','Raw gap'],bounds)
    safety=[]
    for d,n,e in [('cifar10',1000,.1),('web',3000,1.)]:
        for r in out['frontiers']:
            if (r['dataset'],r['n'],r['epsilon'],r['scheme'])!=(d,n,e,'random'):continue
            safety.append([names[d],str(r['m']),number(r['public']),number(r['hybrid']),number(r['gated']),f"{100*r['acceptance']:.1f}",f"{100*r['hybrid_minimum_recall']:.2f}"])
    table(paper/'tab-publication-gate.tex',r'Private-augmentation publication versus public-only fallback under random label access. '
        r'Each row crosses 100 label sets with ten cohorts. Gate requires positive lower LOO gain, proxy class coverage and positive per-class Wilson lower recall. '
        r'This is a tested heuristic, not a private-local-benefit certificate. All/gated are unconditional test utilities, with public-only fallback on rejection. Entries \%; Web uses balanced accuracy.',
        'tab:publication-gate','lrrrrrr',['Task','m','Public','All hybrid','Gated','Accept','Min recall'],safety)
    natural=json.loads((PARENT/'summary.json').read_text())['natural_resolution']
    for row in natural:
        condition={key:row[key] for key in ['n','epsilon','k']};condition['dataset']='femnist62'
        ids=[r['job'] for r in old['roles'] if r['role']=='natural-resolution-grid' and r['condition']==condition]
        row['native_global_accuracy']=row['global_accuracy']
        row['global_accuracy']=stats([regional_metric(i,'actual',3,True)['accuracy'] for i in ids])
    out['natural_resolution']=natural
    out['public_regional_assignment_convention']=canonical['convention']
    figures(paper,out)
    numerical_macros(paper,out)
    save(paper/'frontier-data.json',out);save(ROOT/'summary.json',out)
    print('Exported',len(out['single_release']),'round settings,',len(out['frontiers']),'budget settings and',len(pub),'label sets.')

def figures(paper,out):
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':7.5,'pdf.fonttype':42,'ps.fonttype':42})
    colors=dict(public='#3d73a1',hybrid='#b34d36',gated='#46816a',private='#666666')
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.0),sharex=True)
    for ax,(d,n,scheme) in zip(axes.flat,[('cifar10',1000,'random'),('cifar10',10000,'random'),('cifar10',10000,'stratified'),('mnist',10000,'random')]):
        rows=sorted([r for r in out['frontiers'] if r['dataset']==d and r['n']==n and r['scheme']==scheme],key=lambda r:r['q'])
        if not rows:continue
        x=[r['q'] for r in rows]
        ax.plot(x,[100*r['public']['mean'] for r in rows],'o-',color=colors['public'],label='Public portfolio')
        envelope=[100*max(r['fixed_public_baselines'].values(),key=lambda x:x['mean'])['mean'] for r in rows]
        ax.plot(x,envelope,'d--',color='#6a6091',label='Best frozen public mean',markersize=3)
        for key,style,label in [('hybrid','s-','Private + public'),('gated','^:','Gated / public fallback')]:
            y=np.array([100*r[key]['mean'] for r in rows]);ax.plot(x,y,style,color=colors[key],label=label,markersize=3)
            ax.fill_between(x,[100*r[key]['ci_low'] for r in rows],[100*r[key]['ci_high'] for r in rows],color=colors[key],alpha=.10)
        fixed=next((r for r in out['single_release'] if r['dataset']==d and r['n']==n),None)
        if fixed:ax.axhline(100*fixed['T1']['180_center']['mean'],ls='--',lw=1,color=colors['private'],label='Private-only, no task labels')
        ax.set_xscale('log');ax.set_xticks([1,3,5,10,20,50],labels=['1','3','5','10','20','50']);ax.grid(alpha=.2)
        ax.set_title(('CIFAR-10' if d=='cifar10' else 'MNIST')+f' {n//1000}k / '+scheme);ax.set_ylabel('Test accuracy (%)')
        if d=='cifar10':ax.set_ylim(40,100)
        else:ax.set_ylim(15,100)
    for ax in axes[1]:ax.set_xlabel('Public labels / class (average for random)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.035))
    fig.tight_layout(rect=[0,.075,1,1]);fig.savefig(paper/'figures/fig-label-frontier.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),gridspec_kw={'width_ratios':[1.2,1]})
    rows=out['single_release'];ax=axes[0];y=np.arange(len(rows))
    for key,offset,color,label in [('T3_first',-.21,'#999999','T=3, first / 60 steps'),('60_center',-.07,'#3d73a1','T=1 / 60 steps'),('T3_fixed',.07,'#46816a','T=3, final / 180 steps'),('180_center',.21,'#b34d36','T=1 / 180 steps')]:
        values=[r[key] if key.startswith('T3') else r['T1'][key] for r in rows]
        ax.errorbar([100*r['mean'] for r in values],y+offset,xerr=[100*r['sd'] for r in values],fmt='o',markersize=3,color=color,label=label,capsize=2)
    labelmap={'cifar10':'C','mnist':'M','web':'W','femnist62':'F'}
    ax.set_yticks(y,labels=[f"{labelmap[r['dataset']]} {r['n']//1000}k / {r['epsilon']:g}" for r in rows]);ax.invert_yaxis();ax.set_xlabel('Public NN accuracy (%)');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=6.5)
    ax.set_title('(a) Budget allocation, same cohorts')
    ax=axes[1];r=sorted([r for r in out['frontiers'] if r['dataset']=='web' and r['scheme']=='random'],key=lambda r:r['q'])
    if r:
        x=[z['m'] for z in r]
        for key,label in [('public','Public'),('hybrid','Hybrid'),('gated','Gated')]:ax.plot(x,[100*z[key]['mean'] for z in r],'o-',label=label,color=colors[key],markersize=3)
        ax.plot(x,[100*z['hybrid_minimum_recall'] for z in r],'k:',label='Hybrid minimum recall')
        ax.set_xscale('log');ax.set_xticks(x,labels=[str(z) for z in x]);ax.set_xlabel('Public task labels');ax.set_ylabel('Balanced accuracy / recall (%)');ax.legend(fontsize=6.5);ax.grid(alpha=.2)
    ax.set_title('(b) Usenet: utility and class recall')
    fig.tight_layout();fig.savefig(paper/'figures/fig-release-count.pdf',bbox_inches='tight');plt.close(fig)
    natural=out['natural_resolution'];fig,axes=plt.subplots(1,2,figsize=(7.1,2.45))
    unique={r['k']:r for r in natural};ks=sorted(unique)
    axes[0].plot(ks,[unique[k]['proxy_purity'] for k in ks],'o-',label='Proxy purity')
    axes[0].plot(ks,[unique[k]['validation_purity'] for k in ks],'s--',label='Validation purity')
    axes[0].set_xscale('log');axes[0].set_xticks(ks,labels=list(map(str,ks)));axes[0].set_xlabel('Public regions K');axes[0].set_ylabel('Constant-region purity');axes[0].legend(fontsize=7);axes[0].grid(alpha=.2);axes[0].set_title('(a) Natural writer geometry')
    for epsilon,color in [(.1,'#3d73a1'),(1.,'#b34d36')]:
        r=sorted([r for r in natural if r['n']==2000 and r['epsilon']==epsilon],key=lambda r:r['k'])
        axes[1].plot([z['k'] for z in r],[100*z['global_accuracy']['mean'] for z in r],'o-',color=color,label=f'T=3, eps={epsilon:g}')
        x=[];value=[];sd=[]
        for k in [31,62,124]:
            a=[json.loads(p.read_text()) for p in (ROOT/'results').glob(f'femnist62_2000_{epsilon:g}_*_T1_s180_regional_K{k}_M.json')]
            if len(a)!=10:continue
            can=json.loads((ROOT/'analysis/canonical-regional-readouts.json').read_text())['records']
            values=[]
            for z in a:
                key=release_key(z['dataset'],z['n'],z['epsilon'],z['seed'],z['steps'],'regional',k,'M')
                values.append(can['experiments/'+ROOT.name+'/results/'+key+'.json']['releases']['1']['center']['accuracy'])
            x.append(k);v=np.array(values);value.append(v.mean()*100);sd.append(v.std(ddof=1)*100)
        axes[1].errorbar(x,value,yerr=sd,fmt='s--',capsize=2,color=color,label=f'T=1 / 180, eps={epsilon:g}')
    axes[1].set_xscale('log');axes[1].set_xticks(ks,labels=list(map(str,ks)));axes[1].set_xlabel('Public regions K');axes[1].set_ylabel('Released NN accuracy (%)');axes[1].legend(fontsize=6.5);axes[1].grid(alpha=.2);axes[1].set_title('(b) Resolution and release count, 2k writers')
    fig.tight_layout();fig.savefig(paper/'figures/fig-natural-frontier.pdf',bbox_inches='tight');plt.close(fig)
    save(paper/'figures/frontier-figure-data.json',dict(frontiers=out['frontiers'],single_release=out['single_release'],natural_resolution=natural))

def numerical_macros(paper,out):
    cif=next((x for x in out['frontiers'] if (x['dataset'],x['n'],x['scheme'],x['q'])==('cifar10',1000,'random',1)),None)
    if not cif:return
    gain=cif['gain_vs_strongest_mean_public'];best=cif['fixed_public_baselines'][gain['strongest_mean_public_method']]
    values={'BudgetGain':f"{100*gain['mean']:.1f}",
        'BudgetGainCI':f"[{100*gain['simultaneous_ci_low']:.1f},{100*gain['simultaneous_ci_high']:.1f}]",
        'RandomPublicTen':f"{100*best['mean']:.1f}\\%",'RandomHybridTen':f"{100*cif['hybrid']['mean']:.1f}\\%",
        'RandomPortfolioTen':f"{100*cif['public']['mean']:.1f}\\%"}
    for condition,prefix in [(('cifar10',10000,.1),'Cif'),(('cifar10',1000,.1),'CifSmall'),(('femnist62',2000,.1),'FemLow'),(('femnist62',2000,1.),'FemHigh')]:
        row=next((r for r in out['single_release'] if (r['dataset'],r['n'],r['epsilon'])==condition),None)
        if row:
            values[prefix+'Single']=f"{100*row['T1']['180_center']['mean']:.1f}\\%"
            values[prefix+'HeadSingle']=f"{100*row['head_fixed']['mean']:.1f}\\%"
            values[prefix+'RoundGain']=f"{100*row['T1_180_vs_T3']['mean']:.1f}"
    (paper/'frontier-values.tex').write_text('% Generated from all retained label draws and paired cohorts.\n'+''.join('\\newcommand{\\'+key+'}{'+value+'}\n' for key,value in values.items()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();export(a.paper,a.partial)
