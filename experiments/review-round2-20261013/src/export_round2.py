"""Generate attribution tables and paired effect sizes from recorded results."""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'review-evidence-20261012'


def metric(r):return r['balanced_accuracy'] if r['dataset']=='web' else r['accuracy']


def describe(rows):
    a=np.array([metric(r) for r in rows])
    return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'replicates':len(a)}


def cell(rows):
    q=describe(rows);return f"{100*q['mean']:.1f}\\,$\\pm$\\,{100*q['sd']:.1f}"


def paired(a,b):
    aa={r['seed']:r for r in a};bb={r['seed']:r for r in b}
    assert aa.keys()==bb.keys()
    d=np.array([metric(aa[s])-metric(bb[s]) for s in sorted(aa)])
    half=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
    return {'mean_gain':float(d.mean()),'ci_low':float(d.mean()-half),'ci_high':float(d.mean()+half),
        'replicates':len(d),'seed_differences':d.tolist()}


def table(path,caption,label,header,body,spec,wide=True):
    env='table*' if wide else 'table'
    path.write_text('\\begin{'+env+'}[t]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
        '\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}'+spec+'@{}}\\toprule\n'
        +header+r' \\\midrule'+'\n'+'\n'.join(body)+'\n'+r'\bottomrule\end{tabular}'+'\n\\end{'+env+'}\n')


def parent_rows(ds,n,e,m,seeds):
    a=[r for r in OLD if r.get('model')!='full ResNet-18' and r.get('dataset')==ds
        and r.get('actual_clients')==n and r.get('alpha')==(.1 if ds!='web' else None)
        and r.get('epsilon')==e and r.get('method')==m and r['seed'] in seeds
        and r.get('configuration_override')==('fixed-public-1k' if n==1000 or ds=='web' else None)]
    assert len(a)==len(seeds),(ds,n,e,m,len(a));return sorted(a,key=lambda r:r['seed'])


def new_rows(ds,n,e,m,seeds,label_free=False):
    a=[r for r in NEW if r['dataset']==ds and r['method']==m and r['seed'] in seeds
        and (r.get('n',r.get('actual_clients'))==n if not m.startswith('Public') else True)
        and (r.get('epsilon')==e if 'epsilon' in r else True)
        and ('configuration_rule' in r)==label_free]
    assert len(a)==len(seeds),(ds,n,e,m,label_free,len(a));return sorted(a,key=lambda r:r['seed'])


def local_head(rows):
    return [dict(r,accuracy=r['local_endpoint']['accuracy'],balanced_accuracy=r['local_endpoint']['balanced_accuracy'],
        per_class_recall=r['local_endpoint']['recall']) for r in rows]


def export(paper):
    summary={'attribution':[],'label_free':[],'web_per_class':{},'seed_reconciliation':[]}
    body=[];free_body=[];head_body=[]
    conditions=[('cifar10',10000,.1,list(range(801,806)),'CIFAR, 10k'),
        ('mnist',10000,.1,list(range(801,806)),'MNIST, 10k'),
        ('cifar10',1000,.1,list(range(8101,8111)),'CIFAR, 1k'),
        ('web',1000,.25,list(range(8101,8111)),r'Usenet, 1k, $0.25$'),
        ('web',3000,1.,list(range(8101,8111)),r'Usenet, 3k, $1$')]
    for ds,n,e,seeds,name in conditions:
        regional='representative' if ds=='web' else 'SDD'
        sdd=parent_rows(ds,n,e,regional,seeds)
        local=new_rows(ds,n,e,'Local-only',seeds)
        counter=new_rows(ds,n,e,'Local+Public',seeds)
        public=new_rows(ds,n,e,'Public-only-180',seeds)
        selected=new_rows(ds,n,e,'Public-only-selected',seeds)
        head=new_rows(ds,n,e,'DP-Head',seeds);adapt=local_head(head)
        item={'dataset':ds,'n':n,'epsilon':e,'sdd_variant':regional,'SDD':describe(sdd),'Local_only':describe(local),
            'Local_plus_public':describe(counter),'Public_only_180':describe(public),'Public_only_selected':describe(selected),
            'DP_Head_global':describe(head),'DP_Head_local':describe(adapt),
            'sdd_vs_local':paired(sdd,local),'sdd_vs_public_180':paired(sdd,public),
            'sdd_vs_public_selected':paired(sdd,selected),'sdd_vs_public_transfer':paired(sdd,counter),
            'sdd_vs_head_local':paired(sdd,adapt),'sdd_vs_head_global':paired(sdd,head)}
        summary['attribution'].append(item)
        body.append(name+' & '+' & '.join(cell(a) for a in [sdd,local,counter,public,selected])+r' \\')
        head_body.append(name+' & '+' & '.join(cell(a) for a in [sdd,head,adapt])+r' \\')
        fm=new_rows(ds,n,e,'SDD',seeds,True);ct=new_rows(ds,n,e,'DP-FedCT',seeds,True)
        item={'dataset':ds,'n':n,'epsilon':e,'k':10 if ds!='web' else 6,'SDD_M':describe(fm),
            'DP_FedCT':describe(ct),'Local_only':describe(local),'mean_vs_local':paired(fm,local),'mean_vs_ct':paired(fm,ct)}
        fr=new_rows(ds,n,e,'representative',seeds,True) if ds=='web' else None
        if fr:item.update(SDD_R=describe(fr),representative_vs_local=paired(fr,local),representative_vs_ct=paired(fr,ct))
        summary['label_free'].append(item)
        free_body.append(name+' & '+str(item['k'])+' & '+cell(fm)+' & '+(cell(fr) if fr else '--')+' & '+cell(ct)+' & '+cell(local)+r' \\')
    table(paper/'tab-round2-attribution.tex',r'Knowledge-source attribution, percent mean $\pm$ sample SD (Web: balanced accuracy). Images use $\alpha=0.1$, $\varepsilon=0.1$, five seeds at 10k and ten elsewhere; Web uses ten. SDD is M/R for images/Web. Local+Pub uses public-model proxy labels. Public-only uses all 5000 image/1500 Web training labels, with learning rate 0.3, and releases no private report.',
        'tab:attribution','task, cohort & SDD & Local & Local+Pub & Pub (180) & Pub (600)',body,'lrrrrr')
    table(paper/'tab-round2-label-free.tex',r'No-public-label check. Taxonomy resolution $K=C$ and centered consensus are fixed for every task without public training, validation or proxy labels. The feature maps and all 180 client steps are unchanged. Mean $\pm$ sample SD in percent; five seeds for 10k image cohorts and ten fresh seeds elsewhere. Both Web estimators are shown without routing.',
        'tab:label-free','task, cohort & $K$ & SDD-M & SDD-R & DP-FedCT & Local',free_body,'lrrrrr')
    table(paper/'tab-round2-head-local.tex',r'Matched DP-Head endpoints. Each method uses 180 steps. Head-local evaluates client models before the third aggregation, with the first two DP broadcasts and all three releases charged; no extra adaptation is added. Mean $\pm$ sample SD in percent, with the same replications as Table~\ref{tab:attribution}.',
        'tab:head-local','task, cohort & SDD & Head-global & Head-local',head_body,'lrrr',wide=False)
    web_body=[]
    seeds=list(range(8101,8111))
    for m,name in [('Local-only','Local'),('SDD','SDD-M'),('representative','SDD-R'),('DP-FedCT','DP-FedCT'),
        ('PATE-DI','PATE-DI'),('DP-Head','Head-global'),('DP-Head-local','Head-local'),
        ('Local+Public','Local+Pub'),('Public-only-selected','Public-only')]:
        a=new_rows('web',3000,1.,'DP-Head' if m=='DP-Head-local' else m,seeds)
        if m=='DP-Head-local':a=local_head(a)
        recalls=np.array([r['per_class_recall'] for r in a])
        summary['web_per_class'][m]={'mean':recalls.mean(0).tolist(),'sd':recalls.std(0,ddof=1).tolist()}
        web_body.append(name+' & '+' & '.join(f'{100*v:.1f}' for v in recalls.mean(0))+r' \\')
    table(paper/'tab-round2-web-recall.tex',r'Usenet per-family recall (percent), averaged over the same ten seeds at $n=3000$, $\varepsilon=1$. SDD uses the original fixed $K=5$ public configuration. Balanced accuracy is the arithmetic mean of these six recalls; all clients are evaluated on held-out authors.',
        'tab:web-recall','method & comp & rec & sci & talk & alt/soc & misc',web_body,'lrrrrrr')
    seed_body=[]
    for ds,n,e,name in [('cifar10',1000,.1,'CIFAR, 1k'),('web',1000,.25,'Usenet, 1k')]:
        initial=[r for r in OLD if r.get('dataset')==ds and r.get('actual_clients')==n and r.get('epsilon')==e
            and r.get('alpha')==(.1 if ds!='web' else None) and r.get('method')=='SDD' and r['seed'] in [801,802,803]
            and r.get('configuration_override') is None]
        fresh=parent_rows(ds,n,e,'SDD',list(range(8101,8111)))
        assert len(initial)==3
        assert {(r['k'],r['mode'],r['optimizer_steps_per_round'],r['batch_size']) for r in initial+fresh}.__len__()==1
        groups={'initial 3':initial,'fresh 10':fresh,'pooled 13':initial+fresh}
        summary['seed_reconciliation'].append({'dataset':ds,'n':n,'epsilon':e,'groups':{k:describe(v) for k,v in groups.items()},
            'k':initial[0]['k'],'mode':initial[0]['mode']})
        for group,a in groups.items():seed_body.append(name+' & '+group+' & '+cell(a)+r' \\')
    table(paper/'tab-round2-seeds.tex',r'SDD-M seed reconciliation. Within each task, initial and fresh groups have identical public $K$, consensus mode, feature map, local schedule and budget. The pooled estimate includes every run while retaining group identity. Corresponding conclusions use the independent ten-seed group.',
        'tab:seed-reconciliation',r'setting & seed group & accuracy (\%)',seed_body,'llr',wide=False)
    (ROOT/'round2-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);args=p.parse_args()
    NEW=[json.loads(f.read_text()) for f in (ROOT/'results').glob('*.json')]
    OLD=[json.loads(f.read_text()) for f in (PARENT/'results').glob('*.json')]
    export(args.paper)
