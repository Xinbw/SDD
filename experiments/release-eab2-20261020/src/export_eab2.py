"""Export second-EAB-round numerical objects from frozen records (no hand-entered numbers).

Inputs: paper/eab-data.json (partition-ceiling rows exported by export_eab.py), the clean-count
naming decomposition, the audit-circularity check, the frozen selection-rule study and the
published-minimum-recall diagnostic. Outputs: tab-decomposition.tex, eab2-values.tex,
eab2-data.json and figures/fig-evidence.pdf.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
NAMES={'cifar10':'CIFAR','mnist':'MNIST','web':'Usenet','femnist62':'FEMNIST'}
ORDER={'cifar10':0,'mnist':1,'web':2,'femnist62':3}
COLORS={'cifar10':'#1f77b4','mnist':'#d62728','web':'#2ca02c','femnist62':'#9467bd'}

def load(name):return json.loads((ROOT/'analysis'/name).read_text())

def main(paper):
    ceil=json.loads((paper/'eab-data.json').read_text())['ceiling_rows']
    dec={(r['dataset'],r['n'],r['epsilon']):r for r in load('naming-decomposition.json')['records']}
    sel=load('selection-rules.json');audit=load('audit-circularity.json')['records'];minrec=load('published-minimum-recall.json')['by_epsilon']
    rule=sel['selected_rule'];rows=[];out=dict(rows=[],selected_rule=rule,selection_scores=sel['selection_scores_draws0_49'],heldout=sel['heldout_draws50_99'],audit=audit,
        published_minimum_recall={e:{k:{x:y for x,y in v.items() if x!='values'} for k,v in r.items()} for e,r in minrec.items()})
    for c in sorted(ceil,key=lambda x:(ORDER[x['dataset']],x['n'],x['epsilon'])):
        key=(c['dataset'],c['n'],c['epsilon']);d=dec[key];tag=f"{c['dataset']}/{c['n']}/{c['epsilon']:g}"
        g=sel['fixed_rules_all_draws'][rule][tag];loo=sel['fixed_rules_all_draws']['loo'][tag]
        assert abs(loo['utility']-c['joint'])<5e-4 and abs(d['released_named']-c['fixed_private_naming'])<1e-9 and abs(g['public']-c['public'])<5e-4,tag
        r=dict(task=tag,dataset=c['dataset'],n=c['n'],epsilon=c['epsilon'],one=c['ceiling_one_to_one'],many=c['ceiling_many_to_one'],public=c['public'],
            clean=d['clean_named'],named=d['released_named'],joint=c['joint'],gated=g['utility'],gated_gain=g['mean'],gated_ci=[g['ci_low'],g['ci_high']],v=c['mean_noise_std'],head=c['head_joint'],
            teacher_loss=c['ceiling_many_to_one']-d['clean_named'],noise_loss=d['clean_named']-d['released_named'],selection_loss=d['released_named']-c['joint'])
        out['rows'].append(r)
        p=lambda x:f'{100*x:.1f}'
        rows.append([f"{NAMES[c['dataset']]} {c['n']//1000}k/{c['epsilon']:g}",f"{p(r['one'])}/{p(r['many'])}",p(r['public']),
            f"{100*(r['one']-r['public']):+.1f}/{100*(r['many']-r['public']):+.1f}",p(r['clean']),p(r['named']),f"{r['v']:.3f}",p(r['joint']),p(r['gated']),p(r['head'])])
    body='\n'.join(' & '.join(x)+r' \\' for x in rows)
    v0=rule.split('-')[-1]
    (paper/'tab-decomposition.tex').write_text('\\begin{table*}[t]\n\\caption{Where regional value is lost, at random $m=C$ labels. Ceil.: best one-to-one / many-to-one naming of the fixed public partition (offline, test labels). Room: ceiling minus the strongest public learner Pub. Clean: the same centered decoder applied to the stored noise-free votes (never released). Named: the protected zero-label naming. $v$: calibrated naming-noise standard deviation. Joint: ten-label LOO selection. Gated: LOO unless the labels miss a class and $v\\le'+v0+'$, in which case the protected naming is kept (rule frozen before analysis and selected on draws 0--49). Head: private-head joint control. \\%; Usenet uses balanced accuracy.}\n'
        '\\label{tab:decomposition}\\small\n\\centering\n\\begin{tabular}{@{}lrrrrrrrrr@{}}\\toprule\nTask/$\\varepsilon$ & Ceil. & Pub. & Room & Clean & Named & $v$ & Joint & Gated & Head \\\\\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n\\end{table*}\n')
    by={(r['dataset'],r['n'],r['epsilon']):r for r in out['rows']}
    c1=by[('cifar10',1000,.1)];c10=by[('cifar10',10000,.1)];mn=by[('mnist',10000,.1)];us=by[('web',3000,1.)];f1=by[('femnist62',2000,1.)];f0=by[('femnist62',2000,.1)];f01=by[('femnist62',1000,.1)]
    h=sel['heldout_draws50_99']
    def held(tag,which):return f"{100*h[tag][which]['mean']:+.1f}"
    def aud(task,mult):return next(a for a in audit if a['dataset']==task and a['labels_per_class']==mult)
    a1=aud('cifar10',1);a5=aud('cifar10',5);m20=aud('mnist',20);m50=aud('mnist',50)
    mr=minrec
    vals=dict(CleanUsenet=f"{100*us['clean']:.1f}",CleanMnist=f"{100*mn['clean']:.1f}",CleanFemnist=f"{100*f1['clean']:.1f}",
        TeacherFemnist=f"{100*f1['teacher_loss']:.1f}",NoiseLossFemnistTenth=f"{100*f0['noise_loss']:.1f}",NoiseLossFemnistOneTenth=f"{100*f01['noise_loss']:.1f}",
        RoomFemnistMany=f"{100*(f1['many']-f1['public']):.1f}",NoiseLossCifarOne=f"{100*c1['noise_loss']:.1f}",NoiseLossUsenet=f"{100*us['noise_loss']:.1f}",
        SelLossMnist=f"{100*mn['selection_loss']:.1f}",NamedCifarTen=f"{100*c10['named']:.1f}",SelLossCifarTen=f"{100*c10['selection_loss']:.1f}",
        GateVzero=v0,GatedMnistHeld=held('mnist/10000/0.1','selected'),LooMnistHeld=held('mnist/10000/0.1','reference'),
        GatedCifarTenHeld=held('cifar10/10000/0.1','selected'),LooCifarTenHeld=held('cifar10/10000/0.1','reference'),
        GatedUsenetHeld=held('web/3000/1','selected'),LooUsenetHeld=held('web/3000/1','reference'),
        GatedMnistGain=f"{100*mn['gated_gain']:+.1f}",GatedMnistCI=f"[{100*mn['gated_ci'][0]:.1f},{100*mn['gated_ci'][1]:.1f}]",
        CoverCifarOne=f"{100*sel['fixed_rules_all_draws']['cover']['cifar10/1000/0.1']['mean']:+.1f}",
        AuditCifarOneEst=f"{100*a1['in_sample_estimate']:.1f}",AuditCifarOneHeld=f"{100*a1['heldout_naming']:.1f}",
        AuditCifarFiveEst=f"{100*a5['in_sample_estimate']:.1f}",AuditCifarFiveHeld=f"{100*a5['heldout_naming']:.1f}",
        AuditMnistTwentyEst=f"{100*m20['in_sample_estimate']:.1f}",AuditMnistTwentyHeld=f"{100*m20['heldout_naming']:.1f}",
        AuditMnistFiftyEst=f"{100*m50['in_sample_estimate']:.1f}",AuditMnistFiftyHeld=f"{100*m50['heldout_naming']:.1f}",
        MinRecGainOnePub=str(mr['1.0']['mean-gain rule']['published']),MinRecGainOneZero=f"{100*mr['1.0']['mean-gain rule']['share_below']['0.01']:.1f}",
        MinRecGainOneTenth=f"{100*mr['1.0']['mean-gain rule']['share_below']['0.1']:.1f}",MinRecGainThreeTenth=f"{100*mr['3.0']['mean-gain rule']['share_below']['0.1']:.1f}",
        MinRecClassThreeMedian=f"{100*mr['3.0']['all-recall>0 rule']['minimum_recall_quantiles']['0.5']:.1f}",
        MinRecClassThreeLow=f"{100*mr['3.0']['all-recall>0 rule']['minimum_recall_quantiles']['0.05']:.1f}")
    (paper/'eab2-values.tex').write_text('% Generated: second EAB round (export_eab2.py).\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    (paper/'eab2-data.json').write_text(json.dumps(out,indent=2)+'\n')
    figure(paper,out['rows'],audit,minrec);print(json.dumps(vals,indent=1))

def figure(paper,rows,audit,minrec):
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.75),gridspec_kw={'width_ratios':[1.3,1,1.05]})
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
    a.set_title('(a) loss decomposition',fontsize=8);a.legend(loc='upper center',bbox_to_anchor=(.42,-.25),ncol=3,frameon=False,handletextpad=.2,columnspacing=.6,fontsize=5.8)
    a.set_xlim(0,100)
    b=ax[1]
    for task in ['cifar10','mnist','web','femnist62']:
        A=[x for x in audit if x['dataset']==task];q=[x['labels_per_class'] for x in A]
        b.plot(q,[100*x['in_sample_estimate'] for x in A],'-',color=COLORS[task],lw=1.1,label=NAMES[task])
        b.plot(q,[100*x['heldout_naming'] for x in A],'--',color=COLORS[task],lw=1.1)
        b.axhline(100*A[0]['true_one_to_one_ceiling'],color=COLORS[task],lw=.6,ls=':')
    b.plot([],[],'k-',lw=1,label='ceiling estimate');b.plot([],[],'k--',lw=1,label='same naming, held out')
    b.set_xscale('log');b.set_xlabel('audit labels per class');b.set_ylabel('accuracy (%)');b.set_title('(b) auditing the ceiling',fontsize=8)
    b.legend(frameon=False,fontsize=5.8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.25),handlelength=1.6,columnspacing=.6)
    c=ax[2];bins=[(0,.01,'<1'),(.01,.1,'1-10'),(.1,.5,'10-50'),(.5,1.01,'$\\geq$50')];cols=['#d62728','#ff9896','#c7c7c7','#2ca02c']
    bars=[(e,rule) for rule in ['mean-gain rule','all-recall>0 rule'] for e in ['0.1','1.0','3.0']]
    for i,(e,rule) in enumerate(bars):
        v=np.array(minrec[e][rule]['values']);n=minrec[e][rule]['cases'];left=0
        for j,(lo,hi,lab) in enumerate(bins):
            w=100*np.sum((v>=lo)&(v<hi))/n;c.barh(len(bars)-1-i,w,left=left,color=cols[j],height=.7,label=lab+'%' if i==0 else None);left+=w
    c.set_yticks(range(len(bars)));c.set_yticklabels([f"{'gain' if r.startswith('mean') else 'recalls>0'}, $\\varepsilon$={float(e):g}" for e,r in bars][::-1])
    c.set_xlabel('published, % of 1000 cases');c.set_xlim(0,100);c.set_title('(c) what the rules publish',fontsize=8)
    c.legend(frameon=False,fontsize=5.8,ncol=4,loc='upper center',bbox_to_anchor=(.4,-.25),title='test minimum class recall',title_fontsize=5.8,handlelength=1,columnspacing=.6)
    fig.tight_layout(w_pad=.6);(paper/'figures').mkdir(exist_ok=True);fig.savefig(paper/'figures/fig-evidence.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);main(p.parse_args().paper.resolve())
