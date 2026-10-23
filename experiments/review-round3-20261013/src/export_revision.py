"""Create corrected table labels and protected-target coverage statistics."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'review-round2-20261013'
sys.path.insert(0,str(PARENT/'src'))
from export_round2 import table


def describe(a):
    a=np.asarray(a,dtype=float);n=len(a);half=t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n)
    return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)), 'replicates':n,
        'ci_low':float(a.mean()-half),'ci_high':float(a.mean()+half)}


def export(paper):
    source=PARENT.parent.parent/'paper-release-round2-20261013/sdd-www2027/paper'
    # Prefer the packaged pristine generated copies for portable reproduction.
    snapshot=ROOT/'table-source'
    if snapshot.exists():source=snapshot
    for name in ['tab-round2-attribution.tex','tab-round2-head-local.tex']:
        text=(source/name).read_text()
        text=text.replace('CIFAR,','CIFAR (M),').replace('MNIST,','MNIST (M),').replace('Usenet,','Usenet (R),')
        text=text.replace('task, cohort & SDD &','task / variant, cohort & SDD &')
        (paper/name).write_text(text)
    text=(source/'tab-round2-label-free.tex').read_text()
    text=text.replace('No-public-label check.','No-public-task-label check; image rows use $\\alpha=0.1$, $\\varepsilon=0.1$.')
    text=text.replace('without public training, validation or proxy labels.',
        'without using public task labels for training or configuration; benchmark labels are used for offline evaluation.')
    (paper/'tab-round2-label-free.tex').write_text(text)
    # Restore the exact primary CIFAR grid, while retaining all grid files.
    grid=(source/'tab-new-image-results.tex').read_text().split('\\end{table*}',1)[0]+'\\end{table*}\n'
    (paper/'tab-budget-cifar.tex').write_text(grid)
    records=[json.loads(p.read_text()) for p in (ROOT/'diagnostics').glob('*.json')]
    assert len(records)==60
    original=[json.loads(p.read_text()) for p in (PARENT/'results').glob('*.json')]
    summary={'protected_target_coverage':{},'minimum_recall':{},
        'minimum_recall_rule':'minimum over six classes within each seed, then summarize the ten seed values',
        'training_target_rule':'only releases 1 and 2 enter subsequent client training'}
    body=[]
    for method,label in [('SDD','SDD-M'),('representative','SDD-R'),('DP-FedCT','DP-FedCT'),('DP-Head','DP-Head')]:
        rows=[r for r in records if r['job']['method']==method and r['job']['kind']=='public-selected']
        rows.sort(key=lambda r:r['job']['seed']);assert len(rows)==10
        targets=np.array([[q['target_fraction'] for q in r['observations'][:2]] for r in rows])
        first=targets[:,:,:3].sum(2).mean(1);last=targets[:,:,3:].sum(2).mean(1)
        counts=np.array([[q['represented_target_classes'] for q in r['observations'][:2]] for r in rows]).mean(1)
        minimum=np.array([min(r['per_class_recall']) for r in rows])
        summary['protected_target_coverage'][method]={'first_three_fraction':describe(first),'last_three_fraction':describe(last),
            'mean_represented_classes':describe(counts),'target_fraction_by_class':targets.mean((0,1)).tolist()}
        summary['minimum_recall'][method]=describe(minimum)
        body.append(label+f" & {100*first.mean():.1f} & {100*last.mean():.1f} & {counts.mean():.1f} & "
            +f"{100*minimum.mean():.1f}\\,$\\pm$\\,{100*minimum.std(ddof=1):.1f}"+r' \\')
        if method=='DP-Head':
            summary['head_rounds']=[{'round':i+1,'comp_target_fraction':float(np.mean([r['observations'][i]['target_fraction'][0] for r in rows])),
                'global_balanced_accuracy':float(np.mean([r['observations'][i]['global_balanced_accuracy'] for r in rows])),
                'clipped_fraction':float(np.mean([r['observations'][i]['clipped_fraction'] for r in rows]))} for i in range(3)]
    for method in ['Local-only','Local+Public','Public-only-selected']:
        rows=[r for r in original if r['dataset']=='web' and r['method']==method
            and (r.get('actual_clients')==3000 if not method.startswith('Public') else True)]
        assert len(rows)==10
        summary['minimum_recall'][method]=describe([min(r['per_class_recall']) for r in rows])
    # Combine the existing recall table and new allocation metrics in one table.
    recall_body=[]
    for method,label in [('Local-only','Local'),('SDD','SDD-M'),('representative','SDD-R'),
        ('DP-FedCT','DP-FedCT'),('PATE-DI','PATE-DI'),('DP-Head','Head-global'),
        ('DP-Head-local','Head-local'),('Local+Public','Local+Pub'),('Public-only-selected','Public-only')]:
        source_method='DP-Head' if method=='DP-Head-local' else method
        rows=[r for r in original if r['dataset']=='web' and r['method']==source_method
            and r['seed'] in range(8101,8111) and 'configuration_rule' not in r
            and (r.get('actual_clients')==3000 if not method.startswith('Public') else True)
            and r.get('epsilon',1.)==1.]
        assert len(rows)==10
        recalls=np.array([r['local_endpoint']['recall'] if method=='DP-Head-local' else r['per_class_recall'] for r in rows])
        minimum=describe(recalls.min(1));summary['minimum_recall'][method]=minimum
        allocation=summary['protected_target_coverage'].get(method)
        fraction=f"{100*allocation['last_three_fraction']['mean']:.1f}" if allocation else '--'
        nc=f"{allocation['mean_represented_classes']['mean']:.1f}" if allocation else '--'
        recall_body.append(label+' & '+' & '.join(f'{100*v:.1f}' for v in recalls.mean(0))
            +' & '+fraction+' & '+nc+f" & {100*minimum['mean']:.1f}\\,$\\pm$\\,{100*minimum['sd']:.1f}"+r' \\')
    table(paper/'tab-round2-web-recall.tex',r'Usenet family recall and protected target allocation, 3k/$\varepsilon=1$, ten seeds, original public-selected configuration ($K=5$ for SDD). Target \% is the fraction assigned to talk/alt--soc/misc across releases 1/2; $N_c$ is represented target classes per release (available for the four replayed interfaces). All proxy samples have targets. Min recall is computed within each seed, then summarized (percent mean $\pm$ SD); other recall columns are seed means.',
        'tab:web-recall',r'method & comp & rec & sci & talk & alt/soc & misc & target \% & $N_c$ & min recall',recall_body,'lrrrrrrrrr')
    text=(paper/'tab-round2-web-recall.tex').read_text().replace('\\label{tab:web-recall}',
        '\\label{tab:web-recall}\\label{tab:target-coverage}')
    (paper/'tab-round2-web-recall.tex').write_text(text)
    table(paper/'tab-target-coverage.tex',r'Protected target allocation and per-run class coverage, Usenet 3k/$\varepsilon=1$, ten seeds, original public-selected settings. Target fractions average releases 1/2, which supervise subsequent training; all proxy items have targets. $N_c$ is represented target classes per release. Minimum recall is computed within each seed, then summarized (percent mean $\pm$ SD), not minimized after pooling seeds.',
        'tab:target-coverage',r'method & comp/rec/sci & talk/alt/misc & $N_c$ & min recall',body,'lrrrr')
    (ROOT/'coverage-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);a=p.parse_args()
    export(a.paper)
