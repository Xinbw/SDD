"""Export third-EAB-round numerical objects from frozen records (no hand-entered numbers).

Supersedes export_eab2.py (kept unchanged in release-eab2-20261020) in the manuscript.
Inputs: paper/eab-data.json (partition-ceiling rows exported by export_eab.py), the clean-count
naming decomposition, the selection-rule study and the published-minimum-recall diagnostic of
release-eab2-20261020; the development-split audit and the closed-form verification planning map
of this round; frozen input/design files for the setup table.
Outputs: tab-decomposition.tex, tab-setup.tex, eab3-values.tex, eab3-data.json,
figures/fig-evidence.pdf and figures/fig-planning.pdf.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;EAB2=EXP/'release-eab2-20261020';FRONTIER=EXP/'release-frontier-20261019';VALUE=EXP/'release-value-20261019'
NAMES={'cifar10':'CIFAR','mnist':'MNIST','web':'Usenet','femnist62':'FEMNIST'}
ORDER={'cifar10':0,'mnist':1,'web':2,'femnist62':3}
COLORS={'cifar10':'#1f77b4','mnist':'#d62728','web':'#2ca02c','femnist62':'#9467bd'}

def load(name):return json.loads(((ROOT if (ROOT/'analysis'/name).exists() else EAB2)/'analysis'/name).read_text())

def main(paper):
    ceil=json.loads((paper/'eab-data.json').read_text())['ceiling_rows']
    dec={(r['dataset'],r['n'],r['epsilon']):r for r in load('naming-decomposition.json')['records']}
    sel=load('selection-rules.json');audit=load('audit-circularity-dev.json')['records'];plan=load('verification-planning.json');minrec=load('published-minimum-recall.json')['by_epsilon']
    rule=sel['selected_rule'];rows=[];out=dict(rows=[],selected_rule=rule,planning={k:v for k,v in plan.items() if k not in ['by_n','by_C']},selection_scores=sel['selection_scores_draws0_49'],heldout=sel['heldout_draws50_99'],audit=audit,
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
    (paper/'tab-decomposition.tex').write_text('\\begin{table*}[t]\n\\caption{Where regional value is lost, at random $m=C$ labels. Ceil.: best one-to-one / many-to-one naming of the fixed public partition (offline, test labels). Room: ceiling minus the strongest public learner Pub. Clean: the same centered decoder applied to the stored noise-free votes (never released). Named: the protected zero-label naming. $v$: calibrated naming-noise standard deviation. Joint: ten-label LOO selection. Gated: LOO unless the labels miss a class and $v\\le'+v0+'$, in which case the protected naming is kept. The rule family was proposed after earlier rounds showed fixed naming above Joint on MNIST and CIFAR-10k; its threshold $v_0$ was registered before this analysis, selected on label draws 0--49 and is reported on draws 50--99 of the same cohorts (an exploratory estimate). Head: private-head joint control. \\%; Usenet uses balanced accuracy.}\n'
        '\\label{tab:decomposition}\\small\n\\centering\n\\begin{tabular}{@{}lrrrrrrrrr@{}}\\toprule\nTask/$\\varepsilon$ & Ceil. & Pub. & Room & Clean & Named & $v$ & Joint & Gated & Head \\\\\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n\\end{table*}\n')
    by={(r['dataset'],r['n'],r['epsilon']):r for r in out['rows']}
    c1=by[('cifar10',1000,.1)];c10=by[('cifar10',10000,.1)];mn=by[('mnist',10000,.1)];us=by[('web',3000,1.)];f1=by[('femnist62',2000,1.)];f0=by[('femnist62',2000,.1)];f01=by[('femnist62',1000,.1)]
    h=sel['heldout_draws50_99']
    def held(tag,which):return f"{100*h[tag][which]['mean']:+.1f}"
    def aud(task,mult):return next(a for a in audit if a['dataset']==task and a['labels_per_class']==mult)
    a1=aud('cifar10',1);a5=aud('cifar10',5);m20=aud('mnist',20);m50=aud('mnist',50);u50=aud('web',50);f20=aud('femnist62',20)
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
        AuditUsenetFiftyEst=f"{100*u50['in_sample_estimate']:.1f}",AuditUsenetFiftyHeld=f"{100*u50['heldout_naming']:.1f}",AuditUsenetTrue=f"{100*u50['true_one_to_one_ceiling']:.1f}",
        AuditFemnistTwentyEst=f"{100*f20['in_sample_estimate']:.1f}",AuditFemnistTwentyHeld=f"{100*f20['heldout_naming']:.1f}",AuditFemnistTrue=f"{100*f20['true_one_to_one_ceiling']:.1f}",
        GatedUsenetLoss=f"{100*(h['web/3000/1']['reference']['mean']-h['web/3000/1']['selected']['mean']):.1f}",
        MinRecGainOnePub=str(mr['1.0']['mean-gain rule']['published']),MinRecGainOneZero=f"{100*mr['1.0']['mean-gain rule']['share_below']['0.01']:.1f}",
        MinRecGainOneTenth=f"{100*mr['1.0']['mean-gain rule']['share_below']['0.1']:.1f}",MinRecGainThreeTenth=f"{100*mr['3.0']['mean-gain rule']['share_below']['0.1']:.1f}",
        MinRecClassThreeMedian=f"{100*mr['3.0']['all-recall>0 rule']['minimum_recall_quantiles']['0.5']:.1f}",
        MinRecClassThreeLow=f"{100*mr['3.0']['all-recall>0 rule']['minimum_recall_quantiles']['0.05']:.1f}")
    vals.update(planning_values(plan))
    (paper/'eab3-values.tex').write_text('% Generated: third EAB round (export_eab3.py).\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    (paper/'eab3-data.json').write_text(json.dumps(out,indent=2)+'\n')
    setup_table(paper);figure(paper,out['rows'],audit,minrec);planning_figure(paper,plan);print(json.dumps(vals,indent=1))

WORDS={'gain>=0.1':'gain $\\geq.10$','recall>0':'all recalls $>0$','recall>=0.25':'all recalls $\\geq.25$','recall>=0.5':'all recalls $\\geq.50$','nondeg eta=0.5':'non-degr. $\\eta=.50$','nondeg eta=0.25':'non-degr. $\\eta=.25$'}
LEGEND={'gain>=0.1':'gain$\\geq$0.10','recall>0':'recalls$>$0','recall>=0.25':'recalls$\\geq$0.25','recall>=0.5':'recalls$\\geq$0.50','nondeg eta=0.5':'non-degr. 0.50','nondeg eta=0.25':'non-degr. 0.25'}
MACRO={'gain>=0.1':'GainTenth','recall>0':'RecallZero','recall>=0.25':'RecallQuarter','recall>=0.5':'RecallHalf','nondeg eta=0.5':'NondegHalf','nondeg eta=0.25':'NondegQuarter'}

def planning_values(plan):
    gap=max(abs(c['predicted_accept']-c['recorded_accept']) for row in plan['decision_check'].values() for c in row.values())
    e=plan['effect_sizes'];v=dict(PlanMaxGap=f"{gap:.1f}",PlanCells=str(sum(len(r) for r in plan['decision_check'].values())),
        PlanGain=f"{100*e['gain']:.1f}",PlanMinRecall=f"{100*e['minimum_recall']:.1f}",PlanMinDiff=f"{100*e['minimum_recall_difference']:.1f}")
    for k,name in MACRO.items():v['PlanAnchor'+name]=f"{plan['anchor_n1000_C10'][k]:.1f}"
    c=plan['by_C']['recall>0'];v['PlanRecallZeroCSixtyTwo']=f"{c['epsilon'][c['C'].index(62)]:.1f}"
    n=plan['by_n']['recall>0'];v['PlanRecallZeroNTenK']=f"{n['epsilon'][n['n'].index(10000)]:.2f}"
    return v

def setup_table(paper):
    import sys
    sys.path.insert(0,str(FRONTIER/'src'));from common import CLASSES,SETTINGS
    shape={}
    for d in CLASSES:
        with np.load(FRONTIER/'inputs'/f'{d}.npz') as z:
            shape[d]=dict(dim=z['x_proxy'].shape[1],private=len(np.unique(z['owners_private'])) if 'owners_private' in z else z['x_private'].shape[0],
                proxy=z['x_proxy'].shape[0],dev=z['x_development'].shape[0],test=z['x_test'].shape[0])
    unit={'cifar10':'img.','mnist':'img.','web':'authors','femnist62':'writers'};feat={'cifar10':'DINOv2','mnist':'DINOv2','web':'TF-IDF','femnist62':'DINOv2'}
    k=lambda n:f'{n//1000}k';rows=[]
    for d in ['cifar10','mnist','web','femnist62']:
        S=[x for x in SETTINGS if x[0]==d];s=shape[d]
        rows.append([NAMES[d],str(CLASSES[d]),f"{feat[d]}/{s['dim']}",f"{s['private']} {unit[d]}",', '.join(dict.fromkeys(k(x[1]) for x in S)),
            ', '.join(dict.fromkeys(f'{x[2]:g}'.removeprefix('0') for x in S)),f"{s['proxy']}/{s['dev']}/{s['test']}"])
    v=json.loads((VALUE/'PRIVATE-VALIDATION-DESIGN.json').read_text())['design']
    rows.append(['Verif.',str(CLASSES[v['dataset']]),f"DINOv2/{shape[v['dataset']]['dim']}",f"{v['per_client_training_records']}/{v['per_client_heldout_records']} rec.",k(v['clients']),
        ', '.join(f'{e:g}'.removeprefix('0') for e in v['epsilon_total']),'as CIFAR'])
    body='\n'.join(' & '.join(r)+r' \\' for r in rows)
    (paper/'tab-setup.tex').write_text('\\begin{table}[t]\n\\caption{Evaluated settings. Features: frozen encoder / proxy-fitted dimension. Private: eligible private units (constructed image clients draw from the image pool with replacement). $n$: clients per cohort. Splits: proxy/development/test. Each setting crosses 10 cohorts with 100 random label draws. Verif.: CIFAR-10 verification cohorts with source-disjoint training/held-out records per client.}\n'
        '\\label{tab:setup}\\small\\setlength{\\tabcolsep}{2pt}\n\\centering\n\\begin{tabular}{@{}lrlllll@{}}\\toprule\nTask & $C$ & Features & Private & $n$ & $\\varepsilon$ & Splits \\\\\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n\\end{table}\n')

def planning_figure(paper,plan):
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(3.45,2.2),sharey=True)
    style={'gain>=0.1':('#1f77b4','-'),'recall>0':('#2ca02c','-'),'recall>=0.25':('#2ca02c','--'),'recall>=0.5':('#2ca02c',':'),'nondeg eta=0.5':('#d62728','-'),'nondeg eta=0.25':('#d62728','--')}
    for a,key,x in [(ax[0],'by_n','n'),(ax[1],'by_C','C')]:
        for name,(col,ls) in style.items():
            r=plan[key][name];xs=[u for u,e in zip(r[x],r['epsilon']) if e is not None];ys=[e for e in r['epsilon'] if e is not None]
            a.plot(xs,ys,color=col,ls=ls,lw=1.1,label=LEGEND[name])
        for e in [.1,1,3]:a.axhline(e,color='.8',lw=.5,zorder=0)
        a.axvline(1000 if x=='n' else 10,color='k',lw=.5,ls=':');a.set_xscale('log');a.set_yscale('log')
    ax[0].set_xlabel('clients $n$ ($C=10$)');ax[1].set_xlabel('classes $C$ ($n=1$k)');ax[0].set_ylabel('minimum total $\\varepsilon$');ax[0].set_ylim(.01,100)
    ax[1].set_xticks([2,10,62]);ax[1].set_xticklabels(['2','10','62'])
    ax[0].set_title('(a)',fontsize=8);ax[1].set_title('(b)',fontsize=8)
    h,l=ax[0].get_legend_handles_labels();fig.tight_layout(rect=(0,.13,1,1),w_pad=.4);fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,0),handlelength=1.8,columnspacing=.8);fig.savefig(paper/'figures/fig-planning.pdf',bbox_inches='tight');plt.close(fig)

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
