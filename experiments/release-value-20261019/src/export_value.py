"""Generate the analysis manuscript from preserved and new protected records."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ROOT,PARENT,save,sha
sys.path.insert(0,str(PARENT/'src'))
from export_joint import export as parent_export
from export_frontier import stats,crossed,paired,table,cell,number

def export(paper):
    parent_export(paper)
    criterion=paper/'tab-bound-values.tex'
    criterion.write_text(criterion.read_text().replace('preceding conditional stability lemma','sufficient coupled-decoder criterion'))
    context=json.loads((paper/'joint-data.json').read_text())
    samples=sorted([json.loads(p.read_text()) for p in (PARENT/'analysis/joint-confirmation').glob('cifar10_random_*_m10.json')],key=lambda x:x['draw'])
    a=np.array([[h['ungated']['accuracy'] for h in sorted(x['hybrid'],key=lambda h:h['seed']) if h['n']==1000] for x in samples])
    public=np.array([x['baselines']['count-matching']['accuracy'] for x in samples]);delta=(a-public[:,None])*100;label_mean=delta.mean(1);cohort_mean=delta.mean(0)
    distribution=dict(setting='independent CIFAR1k, eps .1, m10; fixed raw-count public baseline',mean_points=float(delta.mean()),median_label_mean_points=float(np.median(label_mean)),p05_label_mean_points=float(np.quantile(label_mean,.05)),
        negative_label_mean_fraction=float((label_mean<-1e-8).mean()),negative_case_fraction=float((delta<-1e-8).mean()),label_mean_sd_points=float(label_mean.std(ddof=1)),cohort_mean_sd_points=float(cohort_mean.std(ddof=1)),
        label_mean_gain_points=label_mean.tolist(),cohort_mean_gain_points=cohort_mean.tolist(),crossed_gain_points=delta.tolist(),quantiles_points={str(q):float(np.quantile(label_mean,q)) for q in [0,.05,.1,.25,.5,.75,.9,.95,1]})
    can=json.loads((PARENT/'analysis/canonical-regional-readouts.json').read_text())['records'];pool=json.loads((ROOT/'analysis/zero-label-pooled-point.json').read_text());pooled=[]
    with np.load(ROOT/'inputs/cifar10.npz') as z:
        import hashlib
        key=hashlib.sha256(z['x_proxy'].tobytes()).hexdigest()[:12]
        with np.load(ROOT/'inputs/partitions'/f'{key}_K10_semantic.npz') as p:ids=p['ids']
        correct_names=np.array([np.bincount(z['y_proxy'][ids==k],minlength=10).argmax() for k in range(10)])
    for row in pool:
        values=[];recovered=[]
        for src in row['sources']:
            point=json.loads((ROOT.parents[1]/src['path']).read_text())
            name=src['path'].replace('_per-example_K0_-','_regional_K'+str(len(row['sizes']))+'_M')
            r=can[name]['releases']['1']['center'];values.append(r[row['metric']])
            if row['dataset']=='cifar10':
                source=json.loads((ROOT.parents[1]/name).read_text());recovered.append(np.array_equal(source['endpoints']['center']['labels'],correct_names))
        pooled.append(dict(**row,regional=stats(values),paired_regional_minus_pooled=paired(values,row['readouts']['center']['per_seed']),
            complete_proxy_majority_naming_rate=float(np.mean(recovered)) if recovered else None,
            naming_diagnostic_uses_full_proxy_labels_only_offline=True))
    nf=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('no_feedback*.json'))];assert len(nf)==10
    nofeedback={key:stats([r[key]['accuracy'] for r in nf]) for key in ['last','pooled','feedback_last','feedback_pooled']}
    nofeedback['pooled_minus_feedback_pooled']=paired([r['pooled']['accuracy'] for r in nf],[r['feedback_pooled']['accuracy'] for r in nf])
    linear={p.name:json.loads(p.read_text()) for p in sorted((ROOT/'analysis/linear-validation').glob('*.json'))};assert len(linear)==3000
    validation=[]
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'results/validation').glob('*.json'))];assert len(records)==3000
    for e in [.1,1.,3.]:
        rows=[r for r in records if r['epsilon_total']==e];assert len(rows)==1000
        lin=lambda r:linear[f"s{r['seed']}_e{e:g}_m10_d{r['draw']}.json"]
        matrix=lambda fn:np.array([[fn(next(r for r in rows if r['draw']==d and r['seed']==s)) for s in range(8901,8911)] for d in range(100)])
        item=dict(epsilon=e,n=1000,label_draws=100,cohort_seeds=10,
            private_selected_rate=float(np.mean([r['candidate_uses_private'] for r in rows])),
            global_acceptance=crossed(matrix(lambda r:r['policy']['global_gain_accepted']))[0],
            coarse_class_acceptance=crossed(matrix(lambda r:r['policy']['class_checked_accepted']))[0],
            class_acceptance=crossed(matrix(lambda r:lin(r)['class_checked_accepted']))[0],
            candidate_heldout=crossed(matrix(lambda r:r['research_diagnostics']['candidate_heldout_accuracy']))[0],
            public_heldout=crossed(matrix(lambda r:r['research_diagnostics']['public_heldout_accuracy']))[0],
            true_gain=crossed(matrix(lambda r:r['research_diagnostics']['true_gain']))[0],
            global_heldout=crossed(matrix(lambda r:r['research_diagnostics']['global_policy_heldout_accuracy']))[0],
            coarse_class_heldout=crossed(matrix(lambda r:r['research_diagnostics']['class_policy_heldout_accuracy']))[0],
            class_heldout=crossed(matrix(lambda r:lin(r)['policy_heldout_accuracy']))[0],
            candidate_test=crossed(matrix(lambda r:r['research_diagnostics']['candidate_test']['accuracy']))[0],
            public_test=crossed(matrix(lambda r:r['research_diagnostics']['public_test']['accuracy']))[0],
            global_test=crossed(matrix(lambda r:r['research_diagnostics']['global_policy_test']['accuracy']))[0],
            coarse_class_test=crossed(matrix(lambda r:r['research_diagnostics']['class_policy_test']['accuracy']))[0],
            class_test=crossed(matrix(lambda r:lin(r)['policy_test']['accuracy']))[0],
            mean_gain_lower=float(np.mean([r['protected_validation']['gain_lower'] for r in rows])),
            mean_class_minimum_lower=float(np.mean([lin(r)['bounds']['minimum_recall_lower'] for r in rows])),
            gain_lower_empirical_coverage=float(np.mean([r['research_diagnostics']['gain_lower_valid'] for r in rows])),
            class_band_empirical_simultaneous_coverage=float(np.mean([lin(r)['research_diagnostics']['simultaneous_bands_valid'] for r in rows])),
            class_false_accept_count=sum(lin(r)['research_diagnostics']['false_accept'] for r in rows),
            global_false_accept_count=sum(r['policy']['global_gain_accepted'] and r['research_diagnostics']['true_gain']<=0 for r in rows),
            per_seed=[dict(seed=s,global_acceptance=float(np.mean([r['policy']['global_gain_accepted'] for r in rows if r['seed']==s])),class_acceptance=float(np.mean([lin(r)['class_checked_accepted'] for r in rows if r['seed']==s]))) for s in range(8901,8911)])
        validation.append(item)
    out=dict(distribution=distribution,pooled=pooled,no_feedback=nofeedback,private_validation=validation,
        preserved_primary_records=3426,new_private_validation_cohorts=10,new_teachers=10,naming_release_conditions=30,validation_cases=3000,validation_vectors=6000,no_feedback_protocols=10,
        validation_scope='restricted regional family; fresh 480/120 source-disjoint cohorts and fresh public label draws; total rho .5 naming/.25 gain/.25 class statistics; finite held-out target')
    macros={'GainMedian':f"{distribution['median_label_mean_points']:.1f}",'GainFifth':f"{distribution['p05_label_mean_points']:.1f}",
        'GainNegativeDraws':f"{100*distribution['negative_label_mean_fraction']:.0f}",'GainNegativeCases':f"{100*distribution['negative_case_fraction']:.0f}",
        'LabelAxisSD':f"{distribution['label_mean_sd_points']:.2f}",'CohortAxisSD':f"{distribution['cohort_mean_sd_points']:.2f}"}
    for label,e in [('Tight',.1),('Middle',1.),('Larger',3.)]:
        row=next(r for r in validation if r['epsilon']==e)
        macros[label+'GlobalAccept']=f"{100*row['global_acceptance']['mean']:.1f}"
        macros[label+'ClassAccept']=f"{100*row['class_acceptance']['mean']:.1f}"
    (paper/'value-values.tex').write_text('% Generated analysis and private verification results.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    table(paper/'tab-resource-map.tex',r'Cohort phases and annotation resources. P: paired protocols; I: independent frontier confirmation; V: source-disjoint private-verification demonstration. $q\in\{1,3,5,10,20,50\}$. All low-label selection labels are inside m.',
        'tab:resource-map','lll',['Condition','Cohorts/draws','Labels'],[
        ['Protocol P','8501--8510','0'],['Frontier I','8801--8810','Cq'],
        ['Random acquisition','100 draws','Cq'],['Balanced (C/M)','50 draws','Cq'],['Feature-directed','1 fixed pool','Cq'],['Verification V','8901--8910; 100 draws','10']],False)
    vrows=[[f"{r['epsilon']:g}"]+[f"{100*r[k]['mean']:.1f}" for k in ['public_test','candidate_test','global_acceptance','global_test','class_acceptance','class_test']] for r in validation]
    table(paper/'tab-private-validation.tex',r'Budgeted private verification: fresh CIFAR-1k cohorts, 100 random ten-label draws crossed with ten actual cohorts. Each deployment spends one total $(\varepsilon,10^{-5})$ budget: $50\%$ of $\rho$ on naming, $25\%$ each on paired gain and class statistics. The demonstration fixes the public count-matching comparator and the regional candidate family; it is separate from the fourteen-learner frontier. All accuracies are independent-test results; policy utilities include every rejection at its public fallback. Class policy uses direct class-sum confidence bands; acceptance is \%.',
        'tab:private-validation','rrrrrrr',[r'Total $\varepsilon$','Public','Candidate','Gain accept','Gain policy','Class accept','Class policy'],vrows)
    prows=[]
    names={'cifar10':'CIFAR','mnist':'MNIST','web':'Usenet','femnist62':'FEMNIST'}
    for r in pooled:
        d=r['paired_regional_minus_pooled'];prows.append([('I' if r['phase']=='confirmation' else 'P')+f": {names[r['dataset']]} {r['n']//1000}k / {r['epsilon']:g}",cell(r['regional']),f"{100*r['readouts']['center']['mean']:.2f}$\\pm${100*r['readouts']['center']['sd']:.2f}",f"{100*d['mean']:.2f} [{100*d['ci_low']:.2f}, {100*d['ci_high']:.2f}]"])
    table(paper/'tab-zero-label-pool.tex',r'Zero-public-label readouts from existing protected T=1/180 matrices, $K=C$, centered decoding. P/I denote paired protocol/independent confirmation cohorts, ten seeds each. Point averages unprojected released scores within each public region before projection. Accuracy, \%; Web uses balanced accuracy. Differences use paired Student intervals over the ten source cohort/noise seeds.',
        'tab:zero-label-pool','lrrr',[r'Group: task, clients / $\varepsilon$','Regional NN','Pooled Point NN',r'Paired $\Delta$ [95\% CI]'],prows)
    nfrows=[[name,cell(nofeedback[key])] for name,key in [('Feedback, last','feedback_last'),('Feedback, pooled','feedback_pooled'),('Private-only, last','last'),('Private-only, pooled','pooled')]]
    table(paper/'tab-no-feedback.tex',r'CIFAR-1k, $\varepsilon=.1$, ten paired 600-record cohorts and 180 private/proxy training steps. Each T=3 protocol charges three releases. Private-only rounds contain 128 private examples/batch; feedback rounds 2/3 use 64 private and 64 proxy examples. Pooled averages the three unprojected matrices before centered decoding. Mean$\pm$sample SD accuracy, \%. The contrast changes teacher content as well as feedback exposure.',
        'tab:no-feedback','lr',['T=3 protocol/readout','Public NN'],nfrows,False)
    drows=[[name,f'{v:.2f}'] for name,v in [('Mean gain (points)',distribution['mean_points']),('Median label-mean gain',distribution['median_label_mean_points']),('5th percentile label mean',distribution['p05_label_mean_points']),(r'Negative label means (\%)',100*distribution['negative_label_mean_fraction']),(r'Negative crossed cases (\%)',100*distribution['negative_case_fraction']),('SD of label means (points)',distribution['label_mean_sd_points']),('SD of cohort means (points)',distribution['cohort_mean_sd_points'])]]
    table(paper/'tab-gain-distribution.tex',r'Independent CIFAR-1k, $\varepsilon=.1$, ten random public labels. Each label mean averages ten cohorts; each cohort mean averages 100 label draws. The baseline is fixed raw-count matching. Crossed cases are dependent.',
        'tab:gain-distribution','lr',['Statistic','Value'],drows,False)
    crows=[[f"{r['epsilon']:g}",f"{100*r['gain_lower_empirical_coverage']:.1f}",f"{100*r['class_band_empirical_simultaneous_coverage']:.1f}",str(r['global_false_accept_count']),str(r['class_false_accept_count'])] for r in validation]
    table(paper/'tab-validation-coverage.tex',r'Private-verification research diagnostics, 1000 crossed cases per budget. Coverage concerns the finite held-out cohort and validation noise, with 97.5\% one-sided gain coverage and 97.5\% simultaneous class coverage. False accepts count gain-policy acceptance with nonpositive true held-out gain. These diagnostics never enter a decision.',
        'tab:validation-coverage','rrrrr',[r'Total $\varepsilon$',r'Gain coverage (\%)',r'Class coverage (\%)','Gain false accepts','Class false accepts'],crows)
    figures(paper,out)
    save(paper/'value-data.json',out);save(ROOT/'analysis/summary.json',out)
    print('Exported value/verification analysis:',len(records),'validation cases;',len(pool),'zero-label settings.')

def figures(paper,out):
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.1),gridspec_kw={'width_ratios':[1,1.2]})
    rows=[next(r for r in out['pooled'] if r['dataset']=='cifar10' and r['n']==1000 and r['phase']==phase) for phase in ['paired','confirmation']]
    ax=axes[0]
    for j,r in enumerate(rows):
        d=r['paired_regional_minus_pooled'];ax.errorbar(100*d['mean'],1-j,xerr=[[100*(d['mean']-d['ci_low'])],[100*(d['ci_high']-d['mean'])]],fmt='o',color=['#3d73a1','#b34d36'][j],capsize=4)
    ax.axvline(0,color='#777777',ls='--',lw=1);ax.set_yticks([1,0],labels=['Paired cohorts','Independent cohorts']);ax.set_ylim(-.7,1.7);ax.set_xlim(-30,30);ax.set_xlabel('Regional minus pooled Point (points)');ax.grid(axis='x',alpha=.2);ax.set_title('(a) Zero-label pooling')
    ax=axes[1];values=np.sort(out['distribution']['label_mean_gain_points']);ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color='#46816a');ax.axvline(0,color='#777777',ls='--',lw=1)
    for q,label in [(.05,'5th percentile'),(.5,'Median')]:
        value=float(np.quantile(values,q));ax.plot(value,q,'o',color='#b34d36',markersize=4);ax.annotate(f'{label}: {value:.1f}',(value,q),xytext=(7,7),textcoords='offset points',fontsize=7)
    ax.set_ylim(0,1.05);ax.set_xlabel('Gain over public count match (points)');ax.set_ylabel('Fraction of draws');ax.grid(alpha=.2);ax.set_title('(b) Ten-label gain distribution')
    fig.tight_layout();fig.savefig(paper/'figures/fig-value-mechanisms.pdf',bbox_inches='tight');plt.close(fig)
    save(paper/'figures/value-figure-data.json',dict(distribution=out['distribution'],pooled=out['pooled']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);export(p.parse_args().paper)
