"""Generate every new numerical table and Figure 2 from retained records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]


def stats(values):
    v=np.asarray(values,dtype=float)
    return dict(n=len(v),mean=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else 0.,values=v.tolist())


def paired(a,b):
    v=np.asarray(a)-np.asarray(b)
    s=stats(v);half=float(t.ppf(.975,len(v)-1)*s['sd']/np.sqrt(len(v)))
    return dict(**s,ci_low=s['mean']-half,ci_high=s['mean']+half)


def cell(s,digits=2):return f"{s['mean']*100:.{digits}f}$\\pm${s['sd']*100:.{digits}f}"


def table(path,caption,label,columns,header,rows,wide=False):
    env='table*' if wide else 'table'
    text=f'\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\\small\n'
    text+=f'\\centering\n\\begin{{tabular}}{{@{{}}{columns}@{{}}}}\\toprule\n'
    text+=' & '.join(header)+r' \\'+ '\n\\midrule\n'
    text+='\n'.join(' & '.join(row)+r' \\' for row in rows)
    text+=f'\n\\bottomrule\\end{{tabular}}\n\\end{{{env}}}\n'
    path.write_text(text)


def export(paper,partial=False):
    paper.mkdir(parents=True,exist_ok=True)
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    def group(dataset,n,epsilon,method,decoder,seeds,metric='accuracy',endpoint=False):
        rows=[next(r for r in records if r['dataset']==dataset and r['n']==n and r['epsilon']==epsilon and
            r['method']==method and r['decoder']==decoder and r['seed']==seed) for seed in seeds]
        return stats([(r['local_endpoint'] if endpoint else r)[metric] for r in rows]),rows
    output=dict(decoders=[],natural=[],quality=[],selector=[])
    names={'center':'Centered hard','matching':'Weighted matching','soft':'Projected soft','center-soft':'Centered soft'}
    orig=list(range(801,806));fresh=list(range(8201,8206))
    cifar=[]
    for decoder in names:
        a,ra=group('cifar10',10000,.1,'SDD',decoder,orig)
        b,rb=group('cifar10',10000,.1,'SDD',decoder,fresh)
        complete=lambda rr:sum(r['trace'][0]['diagnostics']['distinct_released_classes']==10 for r in rr)
        row=dict(dataset='cifar10',n=10000,decoder=decoder,original=a,fresh=b,
            original_first_release_complete=complete(ra),fresh_first_release_complete=complete(rb),
            original_seeds=orig,fresh_seeds=fresh)
        output['decoders'].append(row)
        cifar.append([names[decoder],cell(a,2),cell(b,2),f'{complete(ra)}/5',f'{complete(rb)}/5'])
    table(paper/'tab-decoder-cifar.tex',
        r'CIFAR-10, actual 10k clients, $\varepsilon=0.1$, fixed $K=C=10$, no public task labels. '
        r'Accuracy (\%, mean$\pm$sample SD); complete means all ten classes occur among first-release regional targets. '
        r'Every prespecified decoder and both disjoint seed groups are retained.',
        'tab:decoder-cifar','lrrrr',['Decoder','Original five','Fresh five','Complete orig.','Complete fresh'],cifar,True)
    cross=[]
    for dataset,n,epsilon,seeds,metric in [('mnist',10000,.1,fresh,'accuracy'),
        ('cifar10',1000,.1,fresh,'accuracy'),('web',3000,1.,list(range(8101,8106)),'balanced_accuracy')]:
        vals=[]
        for decoder in names:
            try:s,rr=group(dataset,n,epsilon,'SDD',decoder,seeds,metric)
            except StopIteration:vals.append('--');continue
            output['decoders'].append(dict(dataset=dataset,n=n,epsilon=epsilon,decoder=decoder,seeds=seeds,metric=metric,summary=s))
            vals.append(cell(s))
        cross.append([f'{dataset}, {n//1000}k']+vals)
    table(paper/'tab-decoder-transfer.tex',
        r'Fixed taxonomy resolution without public task labels, five paired seeds. MNIST/CIFAR-10: accuracy at $\varepsilon=0.1$; '
        r'Web: balanced accuracy at $\varepsilon=1$, $K=6$. -- denotes an unrun prespecified cell.',
        'tab:decoder-transfer','lrrrr',['Task/cohort']+list(names.values()),cross,True)
    center=next(x for x in output['decoders'] if x['dataset']=='cifar10' and x['n']==10000 and x['decoder']=='center')
    match=next(x for x in output['decoders'] if x['dataset']=='cifar10' and x['n']==10000 and x['decoder']=='matching')
    output['cifar_matching_gain_original']=paired(match['original']['values'],center['original']['values'])
    output['cifar_matching_gain_fresh']=paired(match['fresh']['values'],center['fresh']['values'])
    output['cifar_matching_gain_pooled']=paired(match['original']['values']+match['fresh']['values'],center['original']['values']+center['fresh']['values'])
    old=ROOT.parent/'review-round2-20261013/results'
    ct=[json.loads((old/f'label_free_cifar10_10000_0.1_0.1_{s}_DP-FedCT.json').read_text())['accuracy'] for s in orig]
    local=[json.loads((old/f'cifar10_10000_0.1_{s}_Local-only.json').read_text())['accuracy'] for s in orig]
    output['cifar_matching_vs_ct']=paired(match['original']['values'],ct)
    output['cifar_matching_vs_local']=paired(match['original']['values'],local)
    uniform,ur=group('cifar10',10000,.1,'uniform-query','center',fresh)
    output['uniform_query']=dict(summary=uniform,paired_vs_center=paired(center['fresh']['values'],uniform['values']),
        paired_vs_matching=paired(match['fresh']['values'],uniform['values']),seeds=fresh)
    natural_rows=[];natural_main=[]
    for n in [1000,2000]:
        for epsilon in [.1,1.]:
            for method,decoder,endpoint in [('Local-only','center',False),('SDD','center',False),('SDD','matching',False),
                ('DP-FedCT','center',False),('DP-Head','center',False),('DP-Head','center',True)]:
                try:
                    a,rr=group('femnist62',n,0. if method=='Local-only' else epsilon,method,decoder,range(8401,8406),endpoint=endpoint)
                    b,_=group('femnist62',n,0. if method=='Local-only' else epsilon,method,decoder,range(8401,8406),
                        metric='balanced_accuracy',endpoint=endpoint)
                except StopIteration:
                    if partial:continue
                    raise
                title='Local-only' if method=='Local-only' else ('SDD-Match' if decoder=='matching' else method)
                if method=='DP-Head':title+=' local' if endpoint else ' global'
                coords=0 if method=='Local-only' else rr[0]['trace'][0]['diagnostics']['logical_report_coordinates']
                row=dict(n=n,epsilon=epsilon,method=title,accuracy=a,balanced_accuracy=b,seeds=list(range(8401,8406)),coordinates=coords)
                output['natural'].append(row)
                natural_rows.append([f'{n//1000}k',str(epsilon),title,str(coords),cell(a),cell(b)])
    table(paper/'tab-natural-full.tex',
        r'Official 62-class Federated EMNIST with distinct real writers and disjoint writer roles. '
        r'Five seeds, 180 steps per client; accuracy and macro recall (\%, mean$\pm$sample SD). '
        r'SDD uses fixed $K=62$ without public task labels; DP-Head clipping is publicly configured.',
        'tab:natural-full','rrlrrr',['Writers',r'$\varepsilon$','Method','Coords.','Accuracy','Macro recall'],natural_rows,True)
    table(paper/'tab-natural-1k.tex',
        r'Federated EMNIST, 1000 distinct private writers. Five paired seeds, 180 steps per client; '
        r'accuracy and macro recall (\%, mean$\pm$sample SD). Settings match Table~\ref{tab:natural-main}. '
        r'Both cohorts and all per-seed outcomes are retained in the artifact.',
        'tab:natural-1k','rrlrrr',['Writers',r'$\varepsilon$','Method','Coords.','Accuracy','Macro recall'],
        [r for r in natural_rows if r[0]=='1k'],True)
    if not partial:
        output['natural_paired']=[]
        for n in [1000,2000]:
            for epsilon in [.1,1.]:
                for metric in ['accuracy','balanced_accuracy']:
                    a=next(r for r in output['natural'] if r['n']==n and r['epsilon']==epsilon and r['method']=='SDD-Match')[metric]
                    b=next(r for r in output['natural'] if r['n']==n and r['epsilon']==epsilon and r['method']=='DP-FedCT')[metric]
                    output['natural_paired'].append(dict(n=n,epsilon=epsilon,metric=metric,matching_vs_per_example=paired(a['values'],b['values'])))
        methods=['Local-only','SDD','SDD-Match','DP-FedCT','DP-Head global','DP-Head local']
        for method in methods:
            rows=[next(r for r in output['natural'] if r['n']==2000 and r['epsilon']==e and r['method']==method) for e in [.1,1.]]
            natural_main.append([method,str(rows[0]['coordinates'])]+[cell(r[k]) for r in rows for k in ['accuracy','balanced_accuracy']])
        table(paper/'tab-natural-main.tex',
            r'Federated EMNIST: 2000 distinct private writers, 62 classes, five seeds and 180 steps. '
            r'Each budget shows accuracy / macro recall (\%, mean$\pm$sample SD). '
            r'No writer or source example is duplicated between clients; held-out writers are disjoint. '
            r'SDD fixes $K=C=62$; DP-Head clipping uses public development labels.',
            'tab:natural-main','lrrrrr',['Method','Coords.',r'$0.1$: acc.',r'$0.1$: macro',r'$1$: acc.',r'$1$: macro'],natural_main,True)
    qs=[json.loads(p.read_text()) for p in sorted((ROOT/'predictions').glob('quality_*.json'))]
    for mode in ['raw','center']:
        exact=[r for r in qs if 'exact_mean' in r['modes'][mode]]
        if exact:
            err=[abs(r['modes'][mode]['exact_mean']-r['modes'][mode]['predicted_accuracy']) for r in exact]
            output['quality'].append(dict(mode=mode,cells=len(exact),mae=float(np.mean(err)),max_error=float(np.max(err)),
                median_exact_sem=float(np.median([r['modes'][mode]['exact_sem'] for r in exact])),
                within_two_combined_mc_sem=int(sum(abs(r['modes'][mode]['exact_mean']-r['modes'][mode]['predicted_accuracy'])<=
                    2*np.hypot(r['modes'][mode]['exact_sem'],r['modes'][mode]['gaussian_mc_sem'])+1e-12 for r in exact))))
    design=json.loads((ROOT/'public-selection/cifar10_quality.json').read_text())
    for cfg in design['records']:
        for seed in [8301,8302,8303]:
            options=[r for r in qs if r['n']==cfg['n'] and r['epsilon']==cfg['epsilon'] and r['seed']==seed]
            if len(options)!=5:continue
            chosen=next(r for r in options if r['k']==cfg['selected_k'])
            oracle=max(options,key=lambda r:r['modes']['center']['predicted_accuracy'])
            output['selector'].append(dict(n=cfg['n'],epsilon=cfg['epsilon'],seed=seed,selected_k=cfg['selected_k'],
                offline_best_k=oracle['k'],expected_target_accuracy=chosen['modes']['center']['predicted_accuracy'],
                offline_best_accuracy=oracle['modes']['center']['predicted_accuracy'],
                expected_quality_regret=oracle['modes']['center']['predicted_accuracy']-chosen['modes']['center']['predicted_accuracy']))
    output['quality_cells']=len(qs)
    output['new_training_conditions']=len(records)
    (paper/'granularity-data.json').write_text(json.dumps(output,indent=2)+'\n')
    (ROOT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    if qs:plot(paper,qs,output)
    print(json.dumps({k:v for k,v in output.items() if k not in ['decoders','natural','selector']},indent=2))


def plot(paper,qs,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.25))
    colors={100:'#c23b3b',1000:'#d28e00',10000:'#2468a0'}
    for n in colors:
        rows=[r for r in qs if r['n']==n and 'exact_mean' in r['modes']['center']]
        ax[0].errorbar([r['modes']['center']['predicted_accuracy'] for r in rows],
            [r['modes']['center']['exact_mean'] for r in rows],
            yerr=[1.96*r['modes']['center']['exact_sem'] for r in rows],fmt='o',ms=3,color=colors[n],
            lw=.6,capsize=1.5,label=f'{n//1000}k' if n>=1000 else str(n))
    ax[0].plot([0,1],[0,1],':',color='.4',lw=1)
    ax[0].set(xlabel='Gaussian-predicted target accuracy',ylabel='Exact-DG target accuracy',title='(a) Independent noise validation',xlim=(0,1),ylim=(0,1))
    ax[0].legend(title='Clients',fontsize=6,title_fontsize=6,loc='upper left',frameon=False)
    for epsilon,color,marker in [(.025,'#c23b3b','o'),(.1,'#2468a0','s'),(1.,'#2b8a3e','^')]:
        rows=sorted([r for r in qs if r['n']==10000 and r['seed']==8301 and r['epsilon']==epsilon],key=lambda r:r['k'])
        ax[1].plot([r['k'] for r in rows],[r['modes']['center']['predicted_accuracy'] for r in rows],
            marker=marker,ms=3,color=color,label=rf'$\epsilon={epsilon:g}$')
    rows=sorted([r for r in qs if r['n']==10000 and r['seed']==8301 and r['epsilon']==.1],key=lambda r:r['k'])
    ax[1].plot([r['k'] for r in rows],[r['modes']['center']['purity'] for r in rows],'k--',lw=1,label='Regional purity')
    ax[1].set(xscale='log',xlabel='Public regions K',ylabel='Proxy-label accuracy',title='(b) Resolution at 10k clients',ylim=(0,1))
    ax[1].set_xticks([5,10,20,50,100],[5,10,20,50,100]);ax[1].legend(fontsize=6,frameon=False,loc='lower left')
    for decoder,color,label in [('center','#c23b3b','Centered hard'),('matching','#2468a0','Matching')]:
        row=next(r for r in summary['decoders'] if r['dataset']=='cifar10' and r['n']==10000 and r['decoder']==decoder)
        values=np.array(row['original']['values']+row['fresh']['values'])*100
        ax[2].plot(range(1,11),values,'o-',ms=3,lw=.8,color=color,label=label)
    ax[2].axvline(5.5,color='.6',lw=.6,ls=':')
    ax[2].set(xlabel='Paired seed (1–5 original; 6–10 fresh)',ylabel='Final client accuracy (%)',title='(c) Decoding at fixed privacy',ylim=(77,88),xticks=[1,3,5,6,8,10])
    ax[2].legend(fontsize=6,frameon=False,loc='lower right')
    for a in ax:a.grid(alpha=.15)
    fig.tight_layout(pad=.7,w_pad=1.)
    folder=paper/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'fig-granularity.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    (folder/'fig-granularity-data.json').write_text(json.dumps(dict(quality=qs,decoders=summary['decoders']),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paper',type=Path,required=True);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();export(args.paper,args.partial)
