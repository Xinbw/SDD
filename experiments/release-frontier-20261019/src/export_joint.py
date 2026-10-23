"""Final joint-naming evidence, independent confirmation and matched controls."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,CLASSES,SETTINGS,SEEDS,save,release_key
CONFIRM_SETTINGS=[('cifar10',1000,.1),('cifar10',10000,.1),('mnist',10000,.1)]
CONFIRM_SEEDS=list(range(8801,8811))
from export_frontier import export as context_export,stats,crossed,table,cell,number

def simultaneous(group,field,all_methods=False):
    boots=np.array([r.pop('_'+field) for r in group]);mu=np.array([[v['mean'] for v in r[field].values()] if all_methods else r[field]['mean'] for r in group])
    sd=boots.std(-1,ddof=1);axes=tuple(range(boots.ndim-1))
    crit=float(np.quantile(np.max(np.abs((boots-mu[...,None])/np.maximum(sd[...,None],1e-12)),axis=axes),.95))
    for i,r in enumerate(group):
        values=list(r[field].values()) if all_methods else [r[field]]
        for j,v in enumerate(values):
            u=mu[i,j] if all_methods else mu[i];s=sd[i,j] if all_methods else sd[i]
            v.update(simultaneous_ci_low=float(u-crit*s),simultaneous_ci_high=float(u+crit*s),critical=crit)

def export(paper):
    context_export(paper)
    names={'cifar10':'CIFAR','mnist':'MNIST','web':'Usenet','femnist62':'FEMNIST'}
    out=dict(frontiers=[],label_sets=4224,crossed_cases=75480,matched_control_cases=150960,public_candidates=14,
        confirmation=dict(settings=CONFIRM_SETTINGS,seeds=CONFIRM_SEEDS,label_draws=100),
        uncertainty='2000 crossed bootstrap replicates; simultaneous public comparisons cover six budgets and fourteen frozen public learners within each phase/task/cohort/acquisition condition')
    canonical=json.loads((ROOT/'analysis/canonical-regional-readouts.json').read_text())['records']
    for phase,settings,seeds,folder,controls,stage in [('development',SETTINGS,SEEDS,'joint-naming','joint-controls',ROOT),
            ('confirmation',CONFIRM_SETTINGS,CONFIRM_SEEDS,'joint-confirmation','joint-controls-confirmation',ROOT/'confirmation')]:
        for task in CLASSES:
            inputs=[json.loads(p.read_text()) for p in sorted((ROOT/'analysis'/folder).glob(task+'_*_m*.json'))]
            if not inputs:continue
            control={p.name:json.loads(p.read_text()) for p in (ROOT/'analysis'/controls).glob(task+'_*_m*.json')}
            for d,n,e in settings:
                if d!=task:continue
                metric='balanced_accuracy' if d=='web' else 'accuracy'
                private=[];head=[];local=[]
                for seed in seeds:
                    key=release_key(d,n,e,seed,180,'regional',CLASSES[d],'M');path=stage/'results'/(key+'.json');r=json.loads(path.read_text())
                    rel=str(path.relative_to(ROOT.parents[1]));private.append(canonical[rel]['releases']['1']['center'][metric])
                    if phase=='confirmation':local.append(r['private_local_only_endpoint'][metric])
                    h=json.loads((stage/'results'/(release_key(d,n,e,seed,180,'head-fixed',0,'-')+'.json')).read_text());head.append(h['endpoints']['broadcast'][metric])
                for scheme in sorted({x['scheme'] for x in inputs}):
                    group=[]
                    for q in [1,3,5,10,20,50]:
                        samples=sorted([x for x in inputs if x['scheme']==scheme and x['q']==q],key=lambda x:x['draw']);assert len(samples)==(100 if scheme=='random' else 50 if scheme=='stratified' else 1)
                        rows=[sorted([h for h in x['hybrid'] if h['n']==n and h['epsilon']==e],key=lambda h:h['seed']) for x in samples]
                        assert all([h['seed'] for h in hs]==seeds for hs in rows)
                        public=np.array([x['public_portfolio'][metric] for x in samples]);a=np.array([[h['ungated'][metric] for h in hs] for hs in rows]);g=np.array([[h['gated'][metric] for h in hs] for hs in rows])
                        selected,boots=crossed(a-public[:,None]);priv,pb=crossed(a-np.array(private)[None,:])
                        baselines={name:stats([x['baselines'][name][metric] for x in samples]) for name in samples[0]['baselines']};assert len(baselines)==14
                        gains={};bb=[]
                        for name in baselines:
                            gains[name],b=crossed(a-np.array([x['baselines'][name][metric] for x in samples])[:,None]);bb.append(b)
                        ct={}
                        for method in ['head-fixed','per-example']:
                            values=[]
                            for x in samples:
                                key=f"{d}_{scheme}_{x['draw']}_m{x['m']}.json";hs=sorted([h for h in control[key]['controls'] if h['n']==n and h['epsilon']==e and h['method']==method],key=lambda h:h['seed']);assert [h['seed'] for h in hs]==seeds
                                values.append([h['metric'][metric] for h in hs])
                            ct[method]=dict(utility=crossed(values)[0],regional_minus_control=crossed(a-np.array(values))[0])
                        r=dict(phase=phase,dataset=d,n=n,epsilon=e,scheme=scheme,q=q,m=q*CLASSES[d],metric=metric,
                            public=stats(public),fixed_public_baselines=baselines,joint=crossed(a)[0],gated=crossed(g)[0],
                            private_center=stats(private),native_head=stats(head),private_local=stats(local) if local else None,
                            gains=gains,gain_selected_public=selected,gain_private=priv,controls=ct,
                            abstention_utility=crossed([[h['abstention_client_utility'][metric] for h in hs] for hs in rows])[0],
                            augmentation_rate=float(np.mean([[h['gate']['private_augmentation_published'] for h in hs] for hs in rows])),
                            publication_rate=float(np.mean([[h['gate']['public_classifier_published'] for h in hs] for hs in rows])),
                            missing_probability=float(np.mean([x['missing_classes']>0 for x in samples])),
                            joint_minimum_recall=float(np.mean([[h['ungated']['minimum_recall'] for h in hs] for hs in rows])),
                            gated_minimum_recall=float(np.mean([[h['gated']['minimum_recall'] for h in hs] for hs in rows])),
                            _gains=np.stack(bb),_gain_selected_public=boots,_gain_private=pb)
                        group.append(r)
                    simultaneous(group,'gains',True);simultaneous(group,'gain_selected_public');simultaneous(group,'gain_private')
                    for r in group:
                        strongest=max(r['fixed_public_baselines'],key=lambda k:r['fixed_public_baselines'][k]['mean'])
                        r['gain_strongest_public']=dict(mean=r['gains'][strongest]['mean'],simultaneous_ci_low=min(v['simultaneous_ci_low'] for v in r['gains'].values()),simultaneous_ci_high=min(v['simultaneous_ci_high'] for v in r['gains'].values()),method=strongest)
                    out['frontiers']+=group
    assert len(out['frontiers'])==132
    def find(d,n,scheme='random',q=1,phase='confirmation',e=.1):return next(r for r in out['frontiers'] if (r['dataset'],r['n'],r['epsilon'],r['scheme'],r['q'],r['phase'])==(d,n,e,scheme,q,phase))
    primary=find('cifar10',1000);gain=primary['gain_strongest_public'];out['primary']=primary
    vals={'JointGain':f"{100*gain['mean']:.1f}",'JointGainCI':f"[{100*gain['simultaneous_ci_low']:.1f},{100*gain['simultaneous_ci_high']:.1f}]",
        'JointPublic':f"{100*primary['fixed_public_baselines'][gain['method']]['mean']:.1f}\\%",'JointAccuracy':f"{100*primary['joint']['mean']:.1f}\\%",'JointPortfolio':f"{100*primary['public']['mean']:.1f}\\%"}
    (paper/'joint-values.tex').write_text('% Generated: independent joint-naming confirmation.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    anchors=[]
    for r in out['frontiers']:
        if r['scheme']!='random' or r['q']!=1:continue
        if r['phase']=='development' and r['dataset'] in ['cifar10','mnist']:continue
        best=max(r['fixed_public_baselines'].values(),key=lambda v:v['mean'])
        anchors.append([f"{names[r['dataset']]} {r['n']//1000}k / {r['epsilon']:g}",'I' if r['phase']=='confirmation' else 'P']+[f'{100*v:.1f}' for v in [best['mean'],r['joint']['mean'],r['controls']['head-fixed']['utility']['mean'],r['controls']['per-example']['utility']['mean'],r['native_head']['mean']]])
    anchors.sort(key=lambda row:(row[1]!='I',{'CIFAR':0,'MNIST':1,'Usenet':2,'FEMNIST':3}[row[0].split()[0]],int(row[0].split()[1][:-1]),float(row[0].split()[-1])))
    table(paper/'tab-joint-controls.tex',r'Equal-label comparison at random $m=C$: 100 public label draws crossed with ten cohorts per setting. I: independent confirmation; P: paired protocol cohort. Public is the strongest mean of 14 frozen public learners, a descriptive envelope. Region, Head and Point use the same count-based joint assignment and within-budget LOO selection; Point pools unprojected scores after release. Native Head uses no public labels. All entries are public classifiers, \%; Web uses balanced accuracy.',
        'tab:joint-controls','llrrrrr',[r'Task, clients / $\varepsilon$','Group','Public','Region joint','Head joint','Point joint','Native Head'],anchors)
    few=[]
    for d in ['cifar10','mnist']:
        for q in [1,3,5]:
            r=find(d,10000,'stratified',q,'development');samples=[json.loads(p.read_text()) for p in (ROOT/'analysis/joint-naming').glob(f'{d}_stratified_*_m{q*10}.json')]
            def family(prefix):
                values=[]
                for s in samples:
                    options=[v for k,v in s['baselines'].items() if k.startswith(prefix)];chosen=max(range(len(options)),key=lambda j:(options[j]['loo_score'],-j));values.append(options[chosen]['accuracy'])
                return stats(values)
            few.append([names[d],str(q)]+[cell(family(k)) for k in ['cluster-majority','count-matching','prototype','ridge-','graph-']])
    table(paper/'tab-fewshot.tex',r'Public-only naming and learning with 1/3/5 supplied labels per class, 50 independent draws. Count match uses raw observed counts, with zero evidence in empty regions. Ridge and graph parameters use exact LOO within the stated budget. All arms share public representations and unlabelled features; test queries are inductive. Mean$\pm$sample SD accuracy, \%.',
        'tab:fewshot','lrrrrrr',['Task','Labels/class','Majority naming','Count match','Class centroid','Ridge','Graph SSL'],few)
    resources=[['Paired release study','0','10 cohorts/setting'],['Joint development','Cq','3024 label sets'],['Independent confirmation','Cq','1200 sets; 30 new cohorts'],['Random label access','Cq',r'100 draws $\times$ 10 cohorts'],['Balanced (C/M)','Cq',r'50 draws $\times$ 10 cohorts'],['Feature-selected','Cq',r'1 pool $\times$ 10 cohorts']]
    table(paper/'tab-resource-map.tex',r'Resources and replication. $q\in\{1,3,5,10,20,50\}$. Paired cohorts: 8501--8510. Independent cohorts: 8801--8810 with fresh label draws. All joint selection labels are inside m.',
        'tab:resource-map','lll',['Condition','Labels','Replications'],resources,False)
    gate=[]
    for d,n,e,phase in [('cifar10',1000,.1,'confirmation'),('web',3000,1.,'development')]:
        for q in [1,3,5,10,20,50]:
            r=find(d,n,q=q,phase=phase,e=e)
            gate.append([names[d],str(r['m'])]+[f'{100*v:.1f}' for v in [r['public']['mean'],r['joint']['mean'],r['gated']['mean'],r['augmentation_rate'],r['publication_rate'],r['abstention_utility']['mean'],r['joint_minimum_recall']]])
    table(paper/'tab-publication-gate.tex',r'Two publication choices, random label access. Aug.: private augmentation rate, otherwise public-only fallback (Gate). Pub.: strict class-checked public classifier rate, otherwise clients retain their private Local-only models (Retain utility). All utilities are unconditional; Local-only utility is offline evaluation, never selection. Scores are LOO heuristics rather than risk certificates. Entries \%; Web uses balanced accuracy.',
        'tab:publication-gate','lrrrrrrrr',['Task','m','Public','Joint','Gate','Aug.','Pub.','Retain utility','Min recall'],gate)
    geometry=[]
    for d,n,e in [('cifar10',10000,.1),('mnist',10000,.1),('web',3000,1.),('femnist62',2000,1.)]:
        for q in [1,3,5]:
            r=find(d,n,'geometry',q,'development',e);geometry.append([names[d],str(r['m']),number(r['public']),number(r['joint']),f"{100*r['publication_rate']:.1f}"])
    table(paper/'tab-geometry-labels.tex',r'Feature-selected annotation on the fixed public pool; only m selected labels are revealed. Public selection uses the same 14 learners. Pub. is the class-checked publication rate. Accuracy, \%; Web uses balanced accuracy.',
        'tab:geometry-labels','lrrrr',['Task','m','Public','Joint','Pub.'],geometry)
    figures(paper,out)
    save(paper/'joint-data.json',out);save(ROOT/'joint-summary.json',out)
    print('Joint export:',len(out['frontiers']),'cells; primary gain',gain)

def figures(paper,out):
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':7,'pdf.fonttype':42})
    colors={'public':'#3d73a1','joint':'#b34d36','head':'#a77b28','point':'#49866c'}
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.05),sharex=True)
    conditions=[('cifar10',1000,'random','confirmation'),('cifar10',10000,'random','confirmation'),('cifar10',10000,'stratified','development'),('mnist',10000,'random','confirmation')]
    for ax,(d,n,s,phase) in zip(axes.flat,conditions):
        rows=sorted([r for r in out['frontiers'] if (r['dataset'],r['n'],r['scheme'],r['phase'])==(d,n,s,phase)],key=lambda r:r['q']);x=[r['q'] for r in rows]
        ax.plot(x,[100*r['public']['mean'] for r in rows],'o-',color=colors['public'],markersize=3,label='Public portfolio')
        ax.plot(x,[100*max(v['mean'] for v in r['fixed_public_baselines'].values()) for r in rows],'d--',color='#77669a',markersize=3,label='Best frozen public mean')
        y=[100*r['joint']['mean'] for r in rows];ax.plot(x,y,'s-',color=colors['joint'],markersize=3,label='Region joint');ax.fill_between(x,[100*r['joint']['ci_low'] for r in rows],[100*r['joint']['ci_high'] for r in rows],color=colors['joint'],alpha=.12)
        for method,label,color,mark in [('head-fixed','Head joint',colors['head'],'^:'),('per-example','Point joint (pooled)',colors['point'],'v:')]:ax.plot(x,[100*r['controls'][method]['utility']['mean'] for r in rows],mark,color=color,markersize=3,label=label)
        ax.axhline(100*rows[0]['private_center']['mean'],ls='--',lw=1,color='#777777',label='Region only, zero labels')
        ax.set_xscale('log');ax.set_xticks([1,3,5,10,20,50],labels=['1','3','5','10','20','50']);ax.set_ylim(35 if d=='cifar10' else 15,100);ax.grid(alpha=.2)
        ax.set_title(('CIFAR-10' if d=='cifar10' else 'MNIST')+f' {n//1000}k / '+('balanced' if s=='stratified' else 'random'));ax.set_ylabel('Test accuracy (%)')
    for ax in axes[1]:ax.set_xlabel('Public labels / class (average for random)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.015));fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(paper/'figures/fig-joint-frontier.pdf',bbox_inches='tight');plt.close(fig)
    context=json.loads((paper/'frontier-data.json').read_text());rows=context['single_release']
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),gridspec_kw={'width_ratios':[1.2,1]});ax=axes[0];y=np.arange(len(rows))
    for key,offset,color,label in [('T3_first',-.21,'#999999','T=3, first / 60 steps'),('60_center',-.07,'#3d73a1','T=1 / 60 steps'),('T3_fixed',.07,'#46816a','T=3, final / 180 steps'),('180_center',.21,'#b34d36','T=1 / 180 steps')]:
        v=[r[key] if key.startswith('T3') else r['T1'][key] for r in rows];ax.errorbar([100*r['mean'] for r in v],y+offset,xerr=[100*r['sd'] for r in v],fmt='o',markersize=3,color=color,label=label,capsize=2)
    nm={'cifar10':'C','mnist':'M','web':'W','femnist62':'F'};ax.set_yticks(y,labels=[f"{nm[r['dataset']]} {r['n']//1000}k / {r['epsilon']:g}" for r in rows]);ax.invert_yaxis();ax.set_xlabel('Public NN accuracy (Web: balanced, %)');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=6.2);ax.set_title('(a) Budget allocation, same cohorts')
    ax=axes[1];r=sorted([r for r in out['frontiers'] if r['dataset']=='web' and r['scheme']=='random'],key=lambda r:r['q']);x=[z['m'] for z in r]
    for key,label,color in [('public','Public portfolio',colors['public']),('joint','Region joint',colors['joint']),('gated','Augmentation gate',colors['point'])]:ax.plot(x,[100*z[key]['mean'] for z in r],'o-',label=label,color=color,markersize=3)
    ax.plot(x,[100*z['joint_minimum_recall'] for z in r],'k:',label='Joint minimum recall');ax.set_xscale('log');ax.set_xticks(x,labels=list(map(str,x)));ax.set_xlabel('Public task labels');ax.set_ylabel('Balanced accuracy / recall (%)');ax.legend(fontsize=6.4);ax.grid(alpha=.2);ax.set_title('(b) Usenet: utility and class recall');fig.tight_layout();fig.savefig(paper/'figures/fig-release-count.pdf',bbox_inches='tight');plt.close(fig)
    save(paper/'figures/joint-figure-data.json',dict(frontiers=out['frontiers'],single_release=context['single_release'],natural_resolution=context['natural_resolution']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);export(p.parse_args().paper)
