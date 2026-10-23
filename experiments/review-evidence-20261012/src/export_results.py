"""Generate every published number and vector figure from recorded runs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
EPS=[.01,.025,.05,.1,.25,.5,1.]
METHODS=['SDD','DP-FedCT','DP-Head','PATE-DI']
COLORS={'SDD':'#17644d','DP-FedCT':'#4178b8','DP-Head':'#a45b18','PATE-DI':'#87549a',
        'random':'#8492a5','representative':'#5e86aa','weak':'#b4a48b'}
MARKERS={'SDD':'o','DP-FedCT':'s','DP-Head':'^','PATE-DI':'D'}

def subset(dataset,n,alpha,epsilon,method,seeds=(801,802,803),override=None):
    return sorted([r for r in ROWS if r.get('model')!='full ResNet-18' and r['dataset']==dataset
        and r['actual_clients']==n and r['alpha']==alpha and r['epsilon']==epsilon
        and r['method']==method and r['seed'] in seeds and r.get('configuration_override')==override],
        key=lambda r:r['seed'])

def values(rows,field=None):
    field=field or ('balanced_accuracy' if rows and rows[0]['dataset']=='web' else 'accuracy')
    return np.array([r[field] for r in rows])

def describe(rows,field=None):
    a=values(rows,field)
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.,replicates=len(a))

def paired(a,b,field=None):
    aa={r['seed']:r for r in a};bb={r['seed']:r for r in b};keys=sorted(aa.keys()&bb.keys())
    field=field or ('balanced_accuracy' if a[0]['dataset']=='web' else 'accuracy')
    differences=np.array([aa[s][field]-bb[s][field] for s in keys]);n=len(keys)
    mean=float(differences.mean());half=float(t.ppf(.975,n-1)*differences.std(ddof=1)/np.sqrt(n))
    return dict(mean_gain=mean,ci_low=mean-half,ci_high=mean+half,replicates=n,seed_differences=differences.tolist())

def cell(rows):
    z=describe(rows);return f"{z['mean']:.3f}\\,$\\pm$\\,{z['sd']:.3f}"

def table(path,caption,label,header,body,spec,wide=False):
    env='table*' if wide else 'table'
    text='\\begin{'+env+'}[t]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
    text+='\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}'+spec+'@{}}\\toprule\n'
    text+=header+' \\\\\\midrule\n'+'\n'.join(body)+'\n\\bottomrule\\end{tabular}\n\\end{'+env+'}\n'
    path.write_text(text)

def figures(paper):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.3),sharex=True,sharey=True)
    for ax,(dataset,alpha) in zip(axes.flat,[('cifar10',.1),('mnist',.1),('cifar10',1.),('mnist',1.)]):
        for method in METHODS:
            z=[describe(subset(dataset,10000,alpha,e,method)) for e in EPS]
            y=np.array([q['mean'] for q in z]);sd=np.array([q['sd'] for q in z])
            ax.plot(EPS,y,color=COLORS[method],marker=MARKERS[method],ms=3.8,lw=1.5,label=method)
            ax.fill_between(EPS,y-sd,y+sd,color=COLORS[method],alpha=.12,lw=0)
        ax.set_xscale('log');ax.set_ylim(0,1.01);ax.grid(alpha=.16)
        ax.set_title(('CIFAR-10' if dataset=='cifar10' else 'MNIST')+fr', $\alpha={alpha:g}$')
        ax.set_xticks([.01,.025,.05,.1,.25,.5,1.],['.01','.025','.05','.1','.25','.5','1'])
    for ax in axes[:,0]:ax.set_ylabel('Mean client accuracy')
    for ax in axes[-1]:ax.set_xlabel(r'Whole-run privacy budget $\varepsilon$')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',frameon=False)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(paper/'figures/fig-main-curves.pdf');fig.savefig(paper/'figures/fig-main-curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),sharey=True)
    for ax,dataset in [(axes[1],'cifar10')]:
        for method in METHODS[:3]:
            for epsilon,style in [(.1,'-'),(1.,'--')]:
                z=[describe(subset(dataset,n,.1,epsilon,method)) for n in [100,1000,10000]]
                y=np.array([q['mean'] for q in z]);sd=np.array([q['sd'] for q in z])
                ax.errorbar([100,1000,10000],y,yerr=sd,lw=1.3,ls=style,marker=MARKERS[method],ms=4,capsize=2,
                    color=COLORS[method],label=method+fr', $\varepsilon={epsilon:g}$')
        ax.set_xscale('log');ax.set_xticks([100,1000,10000],['100','1k','10k']);ax.set_ylim(0,1.01);ax.grid(alpha=.16)
        ax.set_xlabel('Actual trained and reporting clients');ax.set_title('CIFAR-10' if dataset=='cifar10' else 'MNIST')
    ax=axes[0]
    for epsilon,style in [(.025,'--'),(.1,'-')]:
        points=[]
        for method in METHODS[:3]:
            rows=subset('cifar10',10000,.1,epsilon,method)
            coordinates=rows[0]['trace'][0]['diagnostics']['logical_report_coordinates']
            z=describe(rows);points.append((coordinates,z['mean']))
            ax.errorbar(coordinates,z['mean'],yerr=z['sd'],color=COLORS[method],marker=MARKERS[method],
                ms=5,capsize=2,label=method+fr', $\varepsilon={epsilon:g}$')
        points.sort();ax.plot([q[0] for q in points],[q[1] for q in points],color='#aaa',ls=style,lw=.8,zorder=0)
    ax.set_xscale('log');ax.set_xticks([100,1280,20000],['100','1280','20k']);ax.set_ylim(0,1.01)
    ax.set_xlabel('Integer report coordinates per client');ax.set_ylabel('Mean client accuracy')
    ax.set_title('(a) Accuracy and communication, 10k clients');ax.grid(alpha=.16)
    axes[1].set_title('(b) Actual participation, CIFAR-10')
    handles,labels=axes[1].get_legend_handles_labels();axes[1].legend(handles,labels,ncol=2,loc='lower right',fontsize=6.7,frameon=False)
    axes[0].text(.03,.96,r'$\varepsilon=0.1$: solid; $0.025$: dashed',transform=axes[0].transAxes,fontsize=7,va='top')
    fig.tight_layout();fig.savefig(paper/'figures/fig-mechanism.pdf');fig.savefig(paper/'figures/fig-mechanism.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.1,4.6))
    control_methods=['SDD','random','representative','weak'];short=['SDD','Random','Representative','Pixel PCA']
    for ax,dataset in zip(axes[0],['cifar10','mnist']):
        z=[describe(subset(dataset,10000,.1,.1,m,seeds=range(801,806))) for m in control_methods]
        ax.bar(range(4),[q['mean'] for q in z],yerr=[q['sd'] for q in z],color=[COLORS[m] for m in control_methods],
            width=.68,capsize=3,error_kw={'lw':.8})
        ax.set_xticks(range(4),short,fontsize=7.4);ax.set_ylim(0,1);ax.set_ylabel('Mean client accuracy')
        ax.set_title(('(a) CIFAR-10' if dataset=='cifar10' else '(b) MNIST')+r': same $K$, $n$, $\varepsilon$')
        for j,m in enumerate(control_methods):
            r=subset(dataset,10000,.1,.1,m,seeds=range(801,806))[0]
            purity=r['trace'][0]['diagnostics']['public_cluster_purity']
            ax.text(j,.03,f'purity\n{purity:.2f}',ha='center',va='bottom',fontsize=7.5,color='white' if j!=3 else '#222')
        ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax=axes[1,0]
    web_methods=['SDD','representative','random-representative','DP-FedCT','DP-Head','PATE-DI']
    web_names=['SDD-M','SDD-R','Random-R','DP-FedCT','DP-Head','PATE-DI']
    web_colours=[COLORS['SDD'],'#2d8b69','#8492a5',COLORS['DP-FedCT'],COLORS['DP-Head'],COLORS['PATE-DI']]
    z=[describe(subset('web',3000,None,1.,m,range(8101,8111),'fixed-public-1k')) for m in web_methods]
    ax.bar(range(6),[q['mean'] for q in z],yerr=[q['sd'] for q in z],color=web_colours,
        width=.65,capsize=2,error_kw={'lw':.7})
    ax.set_xticks(range(6),web_names,fontsize=6.4,rotation=12)
    ax.set_ylim(0,.36);ax.set_ylabel('Mean balanced accuracy')
    ax.set_title(r'(c) Usenet: 3k authors, $\varepsilon=1$');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax=axes[1,1]
    for m,label,color,marker in [('SDD','SDD-M',COLORS['SDD'],'o'),('representative','SDD-R','#2d8b69','D'),
        ('DP-FedCT','DP-FedCT',COLORS['DP-FedCT'],'s'),('DP-Head','DP-Head',COLORS['DP-Head'],'^')]:
        z=[describe(subset('web',n,None,.25,m,range(8101,8111),'fixed-public-1k')) for n in [1000,3000]]
        ax.errorbar([1000,3000],[q['mean'] for q in z],yerr=[q['sd'] for q in z],color=color,marker=marker,
            ms=4,capsize=2,lw=1.4,label=label)
    ax.set_xscale('log');ax.set_xticks([1000,3000],['1k','3k']);ax.minorticks_off();ax.set_xlabel('Distinct reporting authors')
    ax.set_ylabel('Mean balanced accuracy');ax.set_title(r'(d) Fixed $K=5$, $\varepsilon=0.25$');ax.grid(alpha=.16)
    ax.legend(fontsize=6.7,ncol=2,frameon=False)
    fig.tight_layout();fig.savefig(paper/'figures/fig-ablations.pdf');fig.savefig(paper/'figures/fig-ablations.png',dpi=180);plt.close(fig)

def export(paper):
    summary={'runs':len(ROWS),'image_budget_curves':[],'five_seed_comparisons':[],'population_scaling':[],
             'semantic_controls':[],'postprocessing':[],'collusion':[],'web':[],'legacy':[]}
    image_tables=[]
    for dataset in ['cifar10','mnist']:
        body=[]
        for alpha in [.1,1.]:
            body.append(r'\multicolumn{6}{l}{'+('CIFAR-10' if dataset=='cifar10' else 'MNIST')+fr', $\alpha={alpha:g}$'+'} \\\\')
            for e in EPS:
                rows=[subset(dataset,10000,alpha,e,m) for m in METHODS]
                assert all(len(r)==3 for r in rows),(dataset,alpha,e)
                body.append(f'{e:g} & {rows[0][0]["k"]} & '+' & '.join(cell(r) for r in rows)+r' \\')
                summary['image_budget_curves'].append(dict(dataset=dataset,alpha=alpha,epsilon=e,k=rows[0][0]['k'],
                    methods={m:describe(r) for m,r in zip(METHODS,rows)}))
        image_tables.append((dataset,body))
    image_text=''
    for dataset,body in image_tables:
        temp=paper/f'_tmp-{dataset}.tex'
        table(temp,('CIFAR-10' if dataset=='cifar10' else 'MNIST')+r': actual $n=10^4$, three independent seeds. Mean accuracy $\pm$ sample SD; all public-selected budgets.',
            'tab:new-'+dataset,r'$\varepsilon$ & $K$ & SDD & DP-FedCT & DP-Head & PATE-DI',body,'rrrrrr',wide=True)
        image_text+=temp.read_text()+'\n';temp.unlink()
    (paper/'tab-new-image-results.tex').write_text(image_text)
    effect_body=[]
    for dataset,alpha,n in [('cifar10',.1,10000),('cifar10',1.,10000),('mnist',.1,10000),('mnist',1.,10000),('web',None,3000)]:
        a=subset(dataset,n,alpha,.1,'SDD',range(801,806));b=subset(dataset,n,alpha,.1,'DP-FedCT',range(801,806))
        z=paired(a,b);summary['five_seed_comparisons'].append(dict(dataset=dataset,alpha=alpha,n=n,epsilon=.1,
            SDD=describe(a),DP_FedCT=describe(b),effect=z,k=a[0]['k']))
        name=('CIFAR' if dataset=='cifar10' else ('MNIST' if dataset=='mnist' else 'Usenet'))
        label=name+(f' ({alpha:g})' if alpha is not None else '')
        if dataset!='web':effect_body.append(label+f" & {100*describe(a)['mean']:.1f} & {100*z['mean_gain']:.1f} & [{100*z['ci_low']:.1f},{100*z['ci_high']:.1f}]"+r' \\')
    table(paper/'tab-new-effects.tex',r'Predeclared image $\varepsilon=0.1$ comparisons over five seeds. Accuracy is a percentage; gain and paired $95\%$ interval are percentage points relative to DP-FedCT.',
        'tab:new-effects','task ($\\alpha$) & SDD & gain & interval',effect_body,'lrrr')
    web_body=[]
    for n in [100,1000,3000]:
        for e in [.1,.25,1.]:
            a=[subset('web',n,None,e,m) for m in METHODS]
            assert all(len(r)==3 for r in a)
            web_body.append(f'{n} & {e:g} & {a[0][0]["k"]} & '+' & '.join(cell(r) for r in a)+r' \\')
            summary['web'].append(dict(n=n,epsilon=e,k=a[0][0]['k'],methods={m:describe(r) for m,r in zip(METHODS,a)}))
    table(paper/'tab-new-web-results.tex',r'Natural-author Usenet results. Mean client balanced accuracy $\pm$ sample SD over three seeds; each cohort contains the stated number of distinct authors.',
        'tab:new-web',r'$n$ & $\varepsilon$ & $K$ & SDD & DP-FedCT & DP-Head & PATE-DI',web_body,'rrrrrrr',wide=True)
    control_body=[]
    for dataset in ['cifar10','mnist']:
        for alpha in [.1,1.]:
            for e in [.1,1.]:
                for m in ['SDD','random','representative','weak']:
                    a=subset(dataset,10000,alpha,e,m,range(801,806));assert len(a)==5
                    diagnostics=[r['trace'][0]['diagnostics'] for r in a]
                    z=dict(dataset=dataset,alpha=alpha,epsilon=e,method=m,k=a[0]['k'],accuracy=describe(a),
                        purity=float(np.mean([q['public_cluster_purity'] for q in diagnostics])),
                        clean_pseudo=float(np.mean([q['clean_pseudolabel_accuracy'] for q in diagnostics])),
                        noisy_pseudo=float(np.mean([q['pseudolabel_accuracy'] for q in diagnostics])),
                        flips=float(np.mean([q['cluster_flip_fraction'] for q in diagnostics])))
                    summary['semantic_controls'].append(z)
                    if alpha==.1 and e==.1:
                        control_body.append(('CIFAR' if dataset=='cifar10' else 'MNIST')+' & '+m+
                            f" & {z['k']} & {z['purity']:.3f} & {z['noisy_pseudo']:.3f} & {z['flips']:.3f} & "+cell(a)+r' \\')
    table(paper/'tab-new-controls.tex',r'Semantic attribution at actual $n=10^4$, $\alpha=0.1$, $\varepsilon=0.1$, five seeds. Purity is public; pseudo-label accuracy and flips use the first release. Full controls at both budgets and heterogeneity levels are in the artifact.',
        'tab:new-controls','task & grouping & $K$ & purity & pseudo & flips & accuracy',control_body,'llrrrrr',wide=True)
    for dataset in ['cifar10','mnist']:
        for alpha in [.1,1.]:
            a=subset(dataset,10000,alpha,.1,'SDD',range(801,806));b=subset(dataset,10000,alpha,.1,'SDD',range(801,806),'raw')
            summary['postprocessing'].append(dict(dataset=dataset,alpha=alpha,selected_mode=a[0]['mode'],effect=paired(a,b)))
            a=subset(dataset,10000,alpha,1.,'SDD',range(801,806),'collusion');b=subset(dataset,10000,alpha,1.,'SDD',range(801,806))
            summary['collusion'].append(dict(dataset=dataset,alpha=alpha,effect=paired(a,b)))
        for n in [100,1000,10000]:
            for e in [.1,1.]:
                summary['population_scaling'].append(dict(dataset=dataset,n=n,epsilon=e,methods={m:describe(subset(dataset,n,.1,e,m)) for m in METHODS[:3]}))
    legacy_body=[]
    for e in [.1,1.]:
        for m in ['SDD','DP-FedCT']:
            a=[r for r in ROWS if r.get('model')=='full ResNet-18' and r['epsilon']==e and r['method']==m]
            if len(a)>=1:
                z=describe(a);summary['legacy'].append(dict(epsilon=e,method=m,**z))
                legacy_body.append(f'{e:g} & '+m+' & '+f'{z["mean"]:.3f}'+r' \\')
    if legacy_body:
        table(paper/'tab-new-legacy.tex',r'Original ResNet-18 diagnostic, actual/calibrated $n=100$, fixed seed42, matched trainer and exact integer noise.',
            'tab:new-legacy',r'$\varepsilon$ & interface & accuracy',legacy_body,'rlr')
    else:(paper/'tab-new-legacy.tex').write_text('% Legacy experiments pending; publication export must be regenerated.\n')
    confirmation=[];confirmation_body=[]
    for n in [1000,3000]:
        for e in [.25,1.]:
            groups={m:subset('web',n,None,e,m,range(8101,8111),'fixed-public-1k') for m in
                ['SDD','representative','random-representative','DP-FedCT','DP-Head','PATE-DI']}
            assert all(len(r)==10 for r in groups.values())
            confirmation.append(dict(n=n,epsilon=e,k=5,methods={m:describe(r) for m,r in groups.items()},
                representative_effects={m:paired(groups['representative'],r) for m,r in groups.items() if m!='representative'}))
            confirmation_body.append(f'{n} & {e:g} & '+' & '.join(cell(groups[m]) for m in
                ['SDD','representative','random-representative','DP-FedCT','DP-Head','PATE-DI'])+r' \\')
    summary['web_fixed_resolution_confirmation']=confirmation
    table(paper/'tab-new-confirmation.tex',r'Fixed-resolution Usenet confirmation, $K=5$, ten fresh seeds. SDD-M pools regional posteriors; SDD-R queries the public centroid representative; Random-R uses matched random groups and representatives. Balanced accuracy $\pm$ sample SD.',
        'tab:new-confirmation',r'$n$ & $\varepsilon$ & SDD-M & SDD-R & Random-R & DP-FedCT & DP-Head & PATE-DI',
        confirmation_body,'rrrrrrrr',wide=True)
    a=subset('cifar10',1000,.1,.1,'SDD',range(8101,8111),'fixed-public-1k')
    summary['compact_parameter_confirmation']=dict(n=1000,epsilon=.1,k=a[0]['k'],SDD=describe(a),
        baselines={m:describe(subset('cifar10',1000,.1,.1,m,range(8101,8111),'fixed-public-1k')) for m in METHODS[1:]},
        effects={m:paired(a,subset('cifar10',1000,.1,.1,m,range(8101,8111),'fixed-public-1k')) for m in METHODS[1:]})
    (ROOT/'analysis-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    keys=['dataset','actual_clients','calibrated_clients','alpha','epsilon','seed','method','k','mode','configuration_override','accuracy','balanced_accuracy','seconds']
    with (ROOT/'all-results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in ROWS:w.writerow({k:r.get(k) for k in keys})
    figures(paper)
    (ROOT/'export-manifest.json').write_text(json.dumps({'results':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'results').glob('*.json')},'generated':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paper.glob('tab-new-*.tex')}},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paper',type=Path,required=True);args=parser.parse_args()
    ROWS=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    assert (ROOT/'SUITE-COMPLETE.json').exists(),'All predeclared head experiments must finish before export.'
    export(args.paper)
