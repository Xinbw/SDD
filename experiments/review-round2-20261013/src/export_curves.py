"""Generate three revision figures using preserved runs and new endpoint summaries."""
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

REVISION=Path(__file__).resolve().parents[1]
ROOT=REVISION.parent/'review-evidence-20261012'
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
                z=[describe(subset(dataset,n,.1,epsilon,method,range(8101,8111) if n==1000 and epsilon==.1 else (801,802,803),
                    'fixed-public-1k' if n==1000 and epsilon==.1 else None)) for n in [100,1000,10000]]
                y=np.array([q['mean'] for q in z]);sd=np.array([q['sd'] for q in z])
                ax.errorbar([100,1000,10000],y,yerr=sd,lw=1.3,ls=style,marker=MARKERS[method],ms=4,capsize=2,
                    color=COLORS[method],label=method+fr', $\varepsilon={epsilon:g}$')
        summary=json.loads((REVISION/'round2-summary.json').read_text())
        endpoint=next(q for q in summary['attribution'] if q['dataset']==dataset and q['n']==1000)
        a=endpoint['DP_Head_local']
        ax.errorbar(1000,a['mean'],yerr=a['sd'],marker='^',ms=5,mfc='white',mec=COLORS['DP-Head'],
            color=COLORS['DP-Head'],capsize=2,ls='none',label=r'Head-local, $\varepsilon=0.1$')
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
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',fontsize=6.7,frameon=False)
    endpoint=next(q for q in summary['attribution'] if q['dataset']=='cifar10' and q['n']==10000)
    pub=endpoint['Public_only_selected']['mean']
    axes[0].axhline(pub,color='#777',lw=.9,ls=':')
    axes[0].text(.03,.995,f'Public-only: {100*pub:.1f}% (0 private coordinates)',transform=axes[0].transAxes,
        fontsize=6.8,va='top',bbox={'facecolor':'white','edgecolor':'none','pad':1,'alpha':.85})
    axes[0].text(.03,.12,r'$\varepsilon=0.1$: solid; $0.025$: dashed',transform=axes[0].transAxes,fontsize=7,va='top')
    fig.tight_layout(rect=(0,.18,1,1));fig.savefig(paper/'figures/fig-mechanism.pdf');fig.savefig(paper/'figures/fig-mechanism.png',dpi=180);plt.close(fig)
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

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paper',type=Path,required=True);args=parser.parse_args()
    ROWS=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    assert (ROOT/'SUITE-COMPLETE.json').exists(),'All predeclared head experiments must finish before export.'
    figures(args.paper)
