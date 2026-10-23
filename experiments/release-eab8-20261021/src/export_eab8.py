"""Export eighth-EAB-round objects (runs after export_eab7.py; no hand-entered numbers).

Inputs: release-budgetconf-20261021 (frozen design, SELECTION.json, analysis), paper/eab3-data.json
(decomposition rows and development audit, as plotted in the superseded fig-evidence) and the
published-minimum-recall diagnostic of release-eab2-20261020.
Outputs: tab-budgetconf.tex, eab8-values.tex, eab8-data.json, figures/fig-ceiling.pdf (former
fig-evidence panels a-b) and figures/fig-verification.pdf (former panel c plus development versus
confirmation acceptance curves).
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;BC=EXP/'release-budgetconf-20261021';EAB2=EXP/'release-eab2-20261020'
NAMES={'cifar10':'CIFAR','mnist':'MNIST','web':'Usenet','femnist62':'FEMNIST'}
COLORS={'cifar10':'#1f77b4','mnist':'#d62728','web':'#2ca02c','femnist62':'#9467bd'}
TARGETS=[('gain>0','gain $>0$'),('all recalls>0','all recalls $>0$'),('all recalls>=.50','all recalls $\\ge.50$')]
f1=lambda x:f'{x:.1f}';sg=lambda t:t.replace('-','$-$');
flat=lambda c:max(v['utility'] for v in c['confirmation_curve'].values())-min(v['utility'] for v in c['confirmation_curve'].values())<1e-9
ph=lambda p:f'{p:g}'.removeprefix('0')

def table(paper,A,S):
    lines=[]
    for t,label in TARGETS:
        for i,e in enumerate(['0.1','1','3']):
            c=A['cells'][f'{t}|{e}'];p=c['primary'];d=p['utility_diff']
            lines.append([label if i==0 else '',ph(float(e)),ph(p['phi']),f"{f1(c['fixed']['accept'])}/{f1(c['fixed']['utility'])}",f"{f1(p['accept'])}/{f1(p['utility'])}",
                sg(f"{d['mean']:+.1f} [{d['ci_low']:+.1f}, {d['ci_high']:+.1f}]"),str(p['false_accepts']),'--' if flat(c) else ph(c['confirmation_best']['phi'])])
    body='\n'.join(' & '.join(r)+r' \\' for r in lines)
    (paper/'tab-budgetconf.tex').write_text('\\begin{table}[t]\n\\caption{Budget split selected on independent development cohorts, frozen, then confirmed on new cohorts and the test split (acceptance \\% / test accuracy \\%, label draws 50--99). $\\phi$: frozen naming share; $\\Delta$: frozen minus fixed accuracy with crossed-bootstrap 95\\% interval; FA: false accepts; Best: confirmation-best share, not used (--: all shares tie).}\n'
        '\\label{tab:budgetconf}\\small\\setlength{\\tabcolsep}{1.8pt}\n\\centering\n\\begin{tabular}{@{}llrrrrrr@{}}\\toprule\nTarget & $\\varepsilon$ & $\\phi$ & Fixed & Frozen & $\\Delta$ accuracy & FA & Best \\\\\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n\\end{table}\n')

def figures(paper,A,S):
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    e3=json.loads((paper/'eab3-data.json').read_text());rows=e3['rows'];audit=e3['audit']
    fig,ax=plt.subplots(1,2,figsize=(6.0,2.55),gridspec_kw={'width_ratios':[1.3,1]})
    a=ax[0];y=np.arange(len(rows))[::-1]
    marks=[('many','M:1 ceiling','k','^',None),('one','1:1 ceiling','k','v','none'),('clean','clean naming','#2ca02c','o',None),
           ('named','protected naming','#2ca02c','o','none'),('joint','joint LOO','#d62728','s','none'),('gated','gated','#d62728','x',None)]
    for i,r in zip(y,rows):
        lo=min(r[k] for k,*_ in marks);hi=max(r[k] for k,*_ in marks);a.plot([100*lo,100*hi],[i,i],color='.85',lw=.8,zorder=0)
        a.plot([100*r['public']]*2,[i-.38,i+.38],color='#ff7f0e',lw=1.6,zorder=1)
    for k,label,col,m,face in marks:
        a.scatter([100*r[k] for r in rows],y,marker=m,s=16 if m!='x' else 20,facecolors=face if face else col,edgecolors=None if m=='x' else col,linewidths=.8,label=label,zorder=2)
    a.plot([],[],color='#ff7f0e',lw=1.6,label='strongest public')
    a.set_yticks(y);a.set_yticklabels([f"{NAMES[r['dataset']]} {r['n']//1000}k/{r['epsilon']:g}" for r in rows]);a.set_xlabel('accuracy (%)')
    a.set_title('(a) loss decomposition',fontsize=8);a.legend(loc='upper center',bbox_to_anchor=(.42,-.25),ncol=3,frameon=False,handletextpad=.2,columnspacing=.6,fontsize=5.8);a.set_xlim(0,100)
    b=ax[1]
    for task in ['cifar10','mnist','web','femnist62']:
        T=[x for x in audit if x['dataset']==task];q=[x['labels_per_class'] for x in T]
        b.plot(q,[100*x['in_sample_estimate'] for x in T],'-',color=COLORS[task],lw=1.1,label=NAMES[task])
        b.plot(q,[100*x['heldout_naming'] for x in T],'--',color=COLORS[task],lw=1.1)
        b.axhline(100*T[0]['true_one_to_one_ceiling'],color=COLORS[task],lw=.6,ls=':')
    b.plot([],[],'k-',lw=1,label='ceiling estimate');b.plot([],[],'k--',lw=1,label='same naming, held out')
    b.set_xscale('log');b.set_xlabel('audit labels per class');b.set_ylabel('accuracy (%)');b.set_title('(b) auditing the ceiling',fontsize=8)
    b.legend(frameon=False,fontsize=5.8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.25),handlelength=1.6,columnspacing=.6)
    fig.tight_layout(w_pad=.6);(paper/'figures').mkdir(exist_ok=True);fig.savefig(paper/'figures/fig-ceiling.pdf',bbox_inches='tight');plt.close(fig)
    minrec=json.loads((EAB2/'analysis/published-minimum-recall.json').read_text())['by_epsilon']
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.3),gridspec_kw={'width_ratios':[1.25,1,1]})
    c=ax[0];bins=[(0,.01,'<1'),(.01,.1,'1-10'),(.1,.5,'10-50'),(.5,1.01,'$\\geq$50')];cols=['#d62728','#ff9896','#c7c7c7','#2ca02c']
    bars=[(e,rule) for rule in ['mean-gain rule','all-recall>0 rule'] for e in ['0.1','1.0','3.0']]
    for i,(e,rule) in enumerate(bars):
        v=np.array(minrec[e][rule]['values']);n=minrec[e][rule]['cases'];left=0
        for j,(lo,hi,lab) in enumerate(bins):
            w=100*np.sum((v>=lo)&(v<hi))/n;c.barh(len(bars)-1-i,w,left=left,color=cols[j],height=.7,label=lab+'%' if i==0 else None);left+=w
    c.set_yticks(range(len(bars)));c.set_yticklabels([f"{'gain' if r.startswith('mean') else 'recalls>0'}, $\\varepsilon$={float(e):g}" for e,r in bars][::-1])
    c.set_xlabel('published, % of 1000 cases',labelpad=1);c.set_xlim(0,100);c.set_title('(a) what the fixed rules publish',fontsize=8)
    c.legend(frameon=False,fontsize=5.8,ncol=4,loc='upper center',bbox_to_anchor=(.4,-.33),title='test minimum class recall',title_fontsize=5.8,handlelength=1,columnspacing=.6)
    for k,(cell,title) in enumerate([('all recalls>0|1','(b) all recalls $>0$, $\\varepsilon=1$'),('all recalls>=.50|3','(c) all recalls $\\geq.50$, $\\varepsilon=3$')]):
        g=ax[k+1];s=S['selection'][cell];cu=A['cells'][cell]
        phis=[float(p) for p in s['development_accept_curve']]
        g.plot(phis,[s['development_accept_curve'][f'{p:g}'] for p in phis],'--o',color='#7f7f7f',ms=2.5,lw=1,label='development')
        g.plot(phis,[cu['confirmation_curve'][f'{p:g}']['accept'] for p in phis],'-o',color='#1f77b4',ms=2.5,lw=1.1,label='confirmation')
        g.axhline(cu['fixed']['accept'],color='#d62728',ls=':',lw=1,label='fixed 50/25/25')
        g.scatter([cu['primary']['phi']],[cu['primary']['accept']],marker='*',s=60,color='#1f77b4',zorder=3,label='frozen choice')
        g.set_xlabel('naming share $\\phi$');g.set_ylim(-3,100);g.set_title(title,fontsize=8)
        if k==0:g.set_ylabel('acceptance (%)')
    h,l=ax[1].get_legend_handles_labels();fig.subplots_adjust(left=.155,right=.99,bottom=.36,top=.9,wspace=.32)
    fig.legend(h,l,loc='lower center',ncol=4,frameon=False,fontsize=5.8,bbox_to_anchor=(.71,0),handlelength=1.6,columnspacing=.8)
    fig.savefig(paper/'figures/fig-verification.pdf',bbox_inches='tight');plt.close(fig)

def main(paper):
    D=json.loads((BC/'BUDGETCONF-DESIGN.json').read_text())['design'];S=json.loads((BC/'SELECTION.json').read_text());A=json.loads((BC/'analysis/budgetconf-analysis.json').read_text())
    table(paper,A,S);figures(paper,A,S)
    r1=A['cells']['all recalls>0|1'];h3=A['cells']['all recalls>=.50|3'];g1=A['cells']['gain>0|1']
    cases=sum(len(json.loads(p.read_text())['rows']) for p in sorted((BC/'results').glob('*.json')))
    fa=sum(c['primary']['false_accepts'] for c in A['cells'].values());ci=lambda d:f"[{d['ci_low']:+.1f}, {d['ci_high']:+.1f}]"
    vals=dict(BCCases=str(cases),BCCohorts=str(len(D['development_cohort_seeds'])),BCRecTrain=str(D['training_records_per_client']),BCRecHeld=str(D['heldout_records_per_client']),
        BCPhiRecallOne=ph(r1['primary']['phi']),BCFixedRecallOne=f1(r1['fixed']['accept']),BCFrozenRecallOne=f1(r1['primary']['accept']),
        BCFixedRecallOneUtil=f1(r1['fixed']['utility']),BCFrozenRecallOneUtil=f1(r1['primary']['utility']),
        BCDiffRecallOneUtil=f"{r1['primary']['utility_diff']['mean']:+.1f}",BCDiffRecallOneUtilCI=ci(r1['primary']['utility_diff']),
        BCDiffRecallOneAcc=f"{100*r1['primary']['accept_diff']['mean']:+.1f}",
        BCDiffRecallOneAccCI=f"[{100*r1['primary']['accept_diff']['ci_low']:+.1f}, {100*r1['primary']['accept_diff']['ci_high']:+.1f}]",
        BCPhiHalfThree=ph(h3['primary']['phi']),BCFrozenHalfThree=f1(h3['primary']['accept']),BCFrozenHalfThreeUtil=f1(h3['primary']['utility']),BCFixedHalfThreeUtil=f1(h3['fixed']['utility']),
        BCDevRecallOne=f1(S['selection']['all recalls>0|1']['development_accept_curve'][f"{r1['primary']['phi']:g}"]),
        BCFalseAccepts=str(fa),BCGainMaxDiff=f"{max(abs(A['cells'][f'gain>0|{e}']['primary']['utility_diff']['mean']) for e in ['0.1','1','3']):.1f}",
        BCMaxRegret=f"{max(c['confirmation_best']['regret_points'] for c in A['cells'].values()):.1f}",
        BCPilotRecallOne=f1(A['pilot']['all recalls>0|1']['mean_confirmation_utility']),BCPilotHalfThree=f1(A['pilot']['all recalls>=.50|3']['mean_confirmation_utility']),
        BCPilotSubsamples=str(A['pilot']['all recalls>0|1']['subsamples']),BCPoolImages=str(45000//2))
    jd=json.loads((paper/'joint-data.json').read_text())['frontiers']
    anchors=[r for r in jd if r['scheme']=='random' and r['q']==1 and not (r['phase']=='development' and r['dataset'] in ['cifar10','mnist'])]
    gap=max(abs(r['joint']['mean']-r['controls']['per-example']['utility']['mean']) for r in anchors)
    mn=next(r for r in anchors if r['dataset']=='mnist')
    vals.update(PointRegionMaxGap=f"{100*gap:.1f}",PointRegionSettings=str(len(anchors)),HeadMnistTenK=f"{100*mn['controls']['head-fixed']['utility']['mean']:.1f}",RegionMnistTenK=f"{100*mn['joint']['mean']:.1f}")
    (paper/'eab8-values.tex').write_text('% Generated: eighth EAB round (export_eab8.py).\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    (paper/'eab8-data.json').write_text(json.dumps(dict(values=vals,primary=A['primary'],predictions=A['predictions'],selection={k:dict(primary_phi=v['primary_phi'],secondary_phi=v['secondary_phi']) for k,v in S['selection'].items()},
        selection_frozen_utc=S['frozen_utc'],design_frozen_utc=json.loads((BC/'BUDGETCONF-DESIGN.json').read_text())['frozen_utc'],cases=cases),indent=2)+'\n')
    print(json.dumps(vals,indent=1))

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--paper',type=Path,required=True);main(q.parse_args().paper.resolve())
