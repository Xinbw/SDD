"""EAB round 13 exporter: print-size figures, the combined split-confirmation table and macros.

Figures: the same data and drawing code as export_value.py (fig-value-mechanisms) and export_eab8.py
(fig-ceiling, fig-verification), drawn at the width they are printed so that every text element is at
least 7 pt; written as *-print.pdf (the originals are still regenerated and compared).
Table: ieee-tab-splitconf.tex merges the CIFAR budget-split confirmation (release-budgetconf) with the
FEMNIST replication (release-femnistconf) and its frozen planner predictions.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
STAGE=Path(__file__).resolve().parents[1];EXP=STAGE.parent
BC=EXP/'release-budgetconf-20261021';FC=EXP/'release-femnistconf-20261022';EAB2=EXP/'release-eab2-20261020'
sys.path.insert(0,str(EXP/'release-eab8-20261021/src'));import export_eab8 as e8
TEXTWIDTH=7.16
f1=lambda x:f'{x:.1f}';z=lambda x:0. if abs(x)<.05 else x;sg=lambda t:t.replace('-','$-$');ph=lambda p:f'{p:g}'.removeprefix('0')
RC={'font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42}

def fig_value(paper):
    d=json.loads((paper/'figures/value-figure-data.json').read_text());plt.rcParams.update(RC)
    fig,axes=plt.subplots(1,2,figsize=(.68*TEXTWIDTH,1.75),gridspec_kw={'width_ratios':[1,1.15]})
    rows=[next(r for r in d['pooled'] if r['dataset']=='cifar10' and r['n']==1000 and r['phase']==phase) for phase in ['paired','confirmation']]
    ax=axes[0]
    for j,r in enumerate(rows):
        x=r['paired_regional_minus_pooled'];ax.errorbar(100*x['mean'],1-j,xerr=[[100*(x['mean']-x['ci_low'])],[100*(x['ci_high']-x['mean'])]],fmt='o',color=['#3d73a1','#b34d36'][j],capsize=3,ms=4)
    ax.axvline(0,color='#777777',ls='--',lw=.8);ax.set_yticks([1,0],labels=['Paired','Independent']);ax.set_ylim(-.7,1.7);ax.set_xlim(-30,30)
    ax.set_xlabel('Regional minus pooled Point');ax.grid(axis='x',alpha=.2);ax.set_title('(a) Zero-label pooling')
    ax=axes[1];v=np.sort(d['distribution']['label_mean_gain_points']);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color='#46816a');ax.axvline(0,color='#777777',ls='--',lw=.8)
    for q,label in [(.05,'5th percentile'),(.5,'Median')]:
        x=float(np.quantile(v,q));ax.plot(x,q,'o',color='#b34d36',markersize=3.5);ax.annotate(f'{label}: {x:.1f}',(x,q),xytext=(5,5),textcoords='offset points',fontsize=7)
    ax.set_ylim(0,1.05);ax.set_xlabel('Gain over count match (points)');ax.set_ylabel('Fraction of draws');ax.grid(alpha=.2);ax.set_title('(b) Ten-label gain')
    fig.tight_layout(w_pad=.8);fig.savefig(paper/'figures/fig-value-mechanisms-print.pdf',bbox_inches='tight');plt.close(fig)

def fig_ceiling(paper):
    """Audit panel only (the decomposition panel duplicated Table VIII); single column, printed at full size."""
    e3=json.loads((paper/'eab3-data.json').read_text());audit=e3['audit'];plt.rcParams.update(RC)
    fig,b=plt.subplots(figsize=(3.45,2.15))
    for task in ['cifar10','mnist','web','femnist62']:
        T=[x for x in audit if x['dataset']==task];q=[x['labels_per_class'] for x in T]
        b.plot(q,[100*x['in_sample_estimate'] for x in T],'-',color=e8.COLORS[task],lw=1.1,label=e8.NAMES[task])
        b.plot(q,[100*x['heldout_naming'] for x in T],'--',color=e8.COLORS[task],lw=1.1)
        b.axhline(100*T[0]['true_one_to_one_ceiling'],color=e8.COLORS[task],lw=.6,ls=':')
    b.plot([],[],'k-',lw=1,label='ceiling estimate');b.plot([],[],'k--',lw=1,label='same naming, held out')
    b.set_xscale('log');b.set_xlabel('audit labels per class');b.set_ylabel('accuracy (%)')
    b.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.45,-.22),handlelength=1.5,columnspacing=.6)
    fig.tight_layout();fig.savefig(paper/'figures/fig-ceiling-print.pdf',bbox_inches='tight');plt.close(fig)

def fig_verification(paper):
    A=json.loads((BC/'analysis/budgetconf-analysis.json').read_text());S=json.loads((BC/'SELECTION.json').read_text())
    minrec=json.loads((EAB2/'analysis/published-minimum-recall.json').read_text())['by_epsilon'];plt.rcParams.update(RC)
    fig,ax=plt.subplots(1,3,figsize=(.9*TEXTWIDTH,2.45),gridspec_kw={'width_ratios':[1.3,1,1]})
    c=ax[0];bins=[(0,.01,'<1'),(.01,.1,'1-10'),(.1,.5,'10-50'),(.5,1.01,'$\\geq$50')];cols=['#d62728','#ff9896','#c7c7c7','#2ca02c']
    bars=[(e,rule) for rule in ['mean-gain rule','all-recall>0 rule'] for e in ['0.1','1.0','3.0']]
    for i,(e,rule) in enumerate(bars):
        v=np.array(minrec[e][rule]['values']);n=minrec[e][rule]['cases'];left=0
        for j,(lo,hi,lab) in enumerate(bins):
            w=100*np.sum((v>=lo)&(v<hi))/n;c.barh(len(bars)-1-i,w,left=left,color=cols[j],height=.7,label=lab+'%' if i==0 else None);left+=w
    c.set_yticks(range(len(bars)));c.set_yticklabels([f"{'gain' if r.startswith('mean') else 'recall'}, $\\varepsilon$={float(e):g}" for e,r in bars][::-1])
    c.set_xlabel('published, % of cases',labelpad=1);c.set_xlim(0,100);c.set_title('(a) fixed rules publish')
    c.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.35,-.27),title='test min. class recall',title_fontsize=7,handlelength=1,columnspacing=.5)
    for k,(cell,title) in enumerate([('all recalls>0|1','(b) recalls $>0$, $\\varepsilon=1$'),('all recalls>=.50|3','(c) recalls $\\geq.5$, $\\varepsilon=3$')]):
        g=ax[k+1];s=S['selection'][cell];cu=A['cells'][cell];phis=[float(p) for p in s['development_accept_curve']]
        g.plot(phis,[s['development_accept_curve'][f'{p:g}'] for p in phis],'--o',color='#7f7f7f',ms=2.5,lw=1,label='development')
        g.plot(phis,[cu['confirmation_curve'][f'{p:g}']['accept'] for p in phis],'-o',color='#1f77b4',ms=2.5,lw=1.1,label='confirmation')
        g.axhline(cu['fixed']['accept'],color='#d62728',ls=':',lw=1,label='fixed split')
        g.scatter([cu['primary']['phi']],[cu['primary']['accept']],marker='*',s=55,color='#1f77b4',zorder=3,label='frozen choice')
        g.set_xlabel('naming share $\\phi$');g.set_ylim(-3,100);g.set_title(title)
        if k==0:g.set_ylabel('acceptance (%)')
    h,l=ax[1].get_legend_handles_labels();fig.subplots_adjust(left=.13,right=.99,bottom=.4,top=.9,wspace=.35)
    fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.73,0),handlelength=1.6,columnspacing=.8)
    fig.savefig(paper/'figures/fig-verification-print.pdf',bbox_inches='tight');plt.close(fig)

def table(paper):
    A=json.loads((BC/'analysis/budgetconf-analysis.json').read_text());F=json.loads((FC/'analysis/femnistconf-analysis.json').read_text())
    rows=[]
    for t,label in e8.TARGETS:
        for i,e in enumerate(['0.1','1','3']):
            c=A['cells'][f'{t}|{e}'];p=c['primary'];d=p['utility_diff']
            rows.append(['CIFAR' if not rows else '',label if i==0 else '',ph(float(e)),ph(p['phi']),f"{f1(c['fixed']['accept'])}/{f1(c['fixed']['utility'])}",f"{f1(p['accept'])}/{f1(p['utility'])}",'--',
                sg(f"{z(d['mean']):+.1f} [{z(d['ci_low']):+.1f}, {z(d['ci_high']):+.1f}]"),str(p['false_accepts']),'--' if e8.flat(c) else ph(c['confirmation_best']['phi'])])
    first=True
    for t,label in e8.TARGETS:
        cells=[F['cells'][f'{t}|{e:g}'] for e in [1.,3.,10.]]
        if t!='gain>0':
            # all eps identical at display precision for class targets: one row, asserted
            key=lambda c:tuple(f1(x) for x in (c['fixed']['accept'],c['fixed']['utility'],c['frozen']['accept'],c['frozen']['utility'],c['fixed']['predicted'],c['frozen']['predicted'],c['frozen']['utility_diff']['mean']))+(c['fixed']['false_accepts'],c['frozen']['false_accepts'],c['frozen']['phi'])
            assert len({key(c) for c in cells})==1;cells=cells[:1];es=['1--10']
        else:es=['1','3','10']
        for i,(c,e) in enumerate(zip(cells,es)):
            f,p=c['fixed'],c['frozen'];d=p['utility_diff']
            flat=max(v['utility'] for v in c['curve'].values())-min(v['utility'] for v in c['curve'].values())<1e-9
            rows.append(['FEMNIST' if first else '',label if i==0 else '',e,ph(p['phi']),f"{f1(f['accept'])}/{f1(f['utility'])}",f"{f1(p['accept'])}/{f1(p['utility'])}",
                f"{f1(f['predicted'])}/{f1(p['predicted'])}",sg(f"{z(d['mean']):+.1f} [{z(d['ci_low']):+.1f}, {z(d['ci_high']):+.1f}]"),str(p['false_accepts']),'--' if flat else ph(float(c['confirmation_best']['config'][5:] if c['confirmation_best']['config'].startswith('class') else c['confirmation_best']['config'][4:]))])
            first=False
    body='\n'.join(' & '.join(r)+r' \\'+('\n\\midrule' if k==8 else '') for k,r in enumerate(rows))
    (paper/'ieee-tab-splitconf.tex').write_text('\\begin{table*}[t]\n\\caption{Budget Split Selected on Development Cohorts and Confirmed on New Cohorts}\n\\label{tab:budgetconf}\\small\\setlength{\\tabcolsep}{4pt}\n\\centering\n'
        '\\begin{tabular}{@{}lllrrrrrrr@{}}\\toprule\nTask & Target & $\\varepsilon$ & $\\phi$ & Fixed & Frozen & Planner & $\\Delta$ accuracy & FA & Best \\\\\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n\\end{table*}\n')

def main(paper):
    fig_value(paper);fig_ceiling(paper);fig_verification(paper);table(paper)
    L=json.loads((STAGE/'analysis/eab13-ledger.json').read_text());p=L['pilot'];fp=L['femnist_public']
    m=dict(PilotDevLabels=str(p['development_labels']),PilotValLabels=str(p['validation_labels']),PilotConfigs=str(p['configurations']),PilotReleases=str(p['protected_releases']),
        PilotRhoMult=f"{p['rho_multiple_of_deployment']:.1f}",PilotEpsOne=f"{p['epsilon_if_same_clients']['1']:.1f}",
        FCPublicPortfolio=f1(fp['portfolio']),FCPublicBest=f1(fp['strongest']['accuracy']),MinNoiseScale=str(L['noise_scale']['min_integer_noise_scale']))
    (paper/'eab13-values.tex').write_text(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k,v in m.items()))
    (paper/'eab13-data.json').write_text(json.dumps(dict(ledger=L,macros=m),indent=2)+'\n');print('wrote',len(m),'macros, 3 print figures, combined split table')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--paper',type=Path,required=True);main(a.parse_args().paper)
