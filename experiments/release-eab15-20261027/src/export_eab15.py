"""EAB round 15 exporter: figure layout in the IEEE convention.

The four data figures of the manuscript are redrawn from the same data with
legends above the plots and the subfigure labels (a), (b), ... below each
panel, under its x-axis label. The drawing code is that of export_joint.py
(fig-joint-frontier) and export_eab13.py (the three *-print figures); only
legend and label placement change, and fig-joint-frontier is drawn at the
width it is printed (text >= 7 pt). Written as fig-*-ieee.pdf; the earlier
files are still regenerated and compared by reproduce.py.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
STAGE=Path(__file__).resolve().parents[1];EXP=STAGE.parent
sys.path.insert(0,str(EXP/'release-eab13-20261023/src'));import export_eab13 as e13
e8=e13.e8;RC=e13.RC;TEXTWIDTH=e13.TEXTWIDTH;BC=e13.BC;EAB2=e13.EAB2

def sublabel(fig,ax,text,pad=3):
    """Subfigure label centred on the axes, below its tick labels and x-axis label."""
    def anchor(r):
        b=ax.get_tightbbox(r);w=ax.get_window_extent(r);return type(b).from_extents(w.x0,b.y0,w.x1,b.y1)
    fig.add_artist(Annotation(text,xy=(.5,0),xycoords=anchor,xytext=(0,-pad),textcoords='offset points',ha='center',va='top',fontsize=RC['axes.titlesize']))

def fig_joint(paper):
    out=json.loads((paper/'joint-data.json').read_text());plt.rcParams.update(RC)
    colors={'public':'#3d73a1','joint':'#b34d36','head':'#a77b28','point':'#49866c'}
    fig,axes=plt.subplots(2,2,figsize=(.72*TEXTWIDTH,3.15))
    conditions=[('cifar10',1000,'random','confirmation'),('cifar10',10000,'random','confirmation'),('cifar10',10000,'stratified','development'),('mnist',10000,'random','confirmation')]
    for k,(ax,(d,n,s,phase)) in enumerate(zip(axes.flat,conditions)):
        rows=sorted([r for r in out['frontiers'] if (r['dataset'],r['n'],r['scheme'],r['phase'])==(d,n,s,phase)],key=lambda r:r['q']);x=[r['q'] for r in rows]
        ax.plot(x,[100*r['public']['mean'] for r in rows],'o-',color=colors['public'],markersize=2.5,lw=1.1,label='Public portfolio')
        ax.plot(x,[100*max(v['mean'] for v in r['fixed_public_baselines'].values()) for r in rows],'d--',color='#77669a',markersize=2.5,lw=1.1,label='Best frozen public mean')
        y=[100*r['joint']['mean'] for r in rows];ax.plot(x,y,'s-',color=colors['joint'],markersize=2.5,lw=1.1,label='Region joint');ax.fill_between(x,[100*r['joint']['ci_low'] for r in rows],[100*r['joint']['ci_high'] for r in rows],color=colors['joint'],alpha=.12)
        for method,label,color,mark in [('head-fixed','Head joint',colors['head'],'^:'),('per-example','Point joint (pooled)',colors['point'],'v:')]:ax.plot(x,[100*r['controls'][method]['utility']['mean'] for r in rows],mark,color=color,markersize=2.5,lw=1.1,label=label)
        ax.axhline(100*rows[0]['private_center']['mean'],ls='--',lw=.8,color='#777777',label='Region only, zero labels')
        ax.set_xscale('log');ax.set_xticks([1,3,5,10,20,50],labels=['1','3','5','10','20','50']);ax.minorticks_off();ax.set_ylim(35 if d=='cifar10' else 15,100);ax.grid(alpha=.2)
        ax.set_ylabel('test accuracy (%)')
        if k>1:ax.set_xlabel('public labels / class (mean if random)')
        sublabel(fig,ax,f'({"abcd"[k]}) '+('CIFAR-10' if d=='cifar10' else 'MNIST')+f' {n//1000}k, '+('balanced' if s=='stratified' else 'random'))
    h,l=axes[0,0].get_legend_handles_labels()
    fig.tight_layout(rect=[0,0,1,.88],h_pad=2.6,w_pad=1.2)
    fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.88),handlelength=2,columnspacing=1)
    fig.savefig(paper/'figures/fig-joint-frontier-ieee.pdf',bbox_inches='tight');plt.close(fig)

def fig_value(paper):
    d=json.loads((paper/'figures/value-figure-data.json').read_text());plt.rcParams.update(RC)
    fig,axes=plt.subplots(1,2,figsize=(.68*TEXTWIDTH,1.75),gridspec_kw={'width_ratios':[1,1.15]})
    rows=[next(r for r in d['pooled'] if r['dataset']=='cifar10' and r['n']==1000 and r['phase']==phase) for phase in ['paired','confirmation']]
    ax=axes[0]
    for j,r in enumerate(rows):
        x=r['paired_regional_minus_pooled'];ax.errorbar(100*x['mean'],1-j,xerr=[[100*(x['mean']-x['ci_low'])],[100*(x['ci_high']-x['mean'])]],fmt='o',color=['#3d73a1','#b34d36'][j],capsize=3,ms=4)
    ax.axvline(0,color='#777777',ls='--',lw=.8);ax.set_yticks([1,0],labels=['Paired','Independent']);ax.set_ylim(-.7,1.7);ax.set_xlim(-30,30)
    ax.set_xlabel('Regional minus pooled Point');ax.grid(axis='x',alpha=.2)
    ax=axes[1];v=np.sort(d['distribution']['label_mean_gain_points']);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color='#46816a');ax.axvline(0,color='#777777',ls='--',lw=.8)
    for q,label in [(.05,'5th percentile'),(.5,'Median')]:
        x=float(np.quantile(v,q));ax.plot(x,q,'o',color='#b34d36',markersize=3.5);ax.annotate(f'{label}: {x:.1f}',(x,q),xytext=(5,5),textcoords='offset points',fontsize=7)
    ax.set_ylim(0,1.05);ax.set_xlabel('Gain over count match (points)');ax.set_ylabel('Fraction of draws');ax.grid(alpha=.2)
    fig.tight_layout(w_pad=.8)
    sublabel(fig,axes[0],'(a) Zero-label pooling');sublabel(fig,axes[1],'(b) Ten-label gain')
    fig.savefig(paper/'figures/fig-value-mechanisms-ieee.pdf',bbox_inches='tight');plt.close(fig)

def fig_ceiling(paper):
    e3=json.loads((paper/'eab3-data.json').read_text());audit=e3['audit'];plt.rcParams.update(RC)
    fig,b=plt.subplots(figsize=(3.45,2.15))
    for task in ['cifar10','mnist','web','femnist62']:
        T=[x for x in audit if x['dataset']==task];q=[x['labels_per_class'] for x in T]
        b.plot(q,[100*x['in_sample_estimate'] for x in T],'-',color=e8.COLORS[task],lw=1.1,label=e8.NAMES[task])
        b.plot(q,[100*x['heldout_naming'] for x in T],'--',color=e8.COLORS[task],lw=1.1)
        b.axhline(100*T[0]['true_one_to_one_ceiling'],color=e8.COLORS[task],lw=.6,ls=':')
    b.plot([],[],'k-',lw=1,label='ceiling estimate');b.plot([],[],'k--',lw=1,label='same naming, held out')
    b.set_xscale('log');b.set_xlabel('audit labels per class');b.set_ylabel('accuracy (%)')
    b.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),handlelength=1.5,columnspacing=.6,borderaxespad=0)
    fig.tight_layout();fig.savefig(paper/'figures/fig-ceiling-ieee.pdf',bbox_inches='tight');plt.close(fig)

def fig_verification(paper):
    A=json.loads((BC/'analysis/budgetconf-analysis.json').read_text());S=json.loads((BC/'SELECTION.json').read_text())
    minrec=json.loads((EAB2/'analysis/published-minimum-recall.json').read_text())['by_epsilon'];plt.rcParams.update(RC)
    fig,ax=plt.subplots(1,3,figsize=(.9*TEXTWIDTH,2.3),gridspec_kw={'width_ratios':[1.3,1,1]})
    c=ax[0];bins=[(0,.01,'<1'),(.01,.1,'1-10'),(.1,.5,'10-50'),(.5,1.01,'$\\geq$50')];cols=['#d62728','#ff9896','#c7c7c7','#2ca02c']
    bars=[(e,rule) for rule in ['mean-gain rule','all-recall>0 rule'] for e in ['0.1','1.0','3.0']]
    for i,(e,rule) in enumerate(bars):
        v=np.array(minrec[e][rule]['values']);n=minrec[e][rule]['cases'];left=0
        for j,(lo,hi,lab) in enumerate(bins):
            w=100*np.sum((v>=lo)&(v<hi))/n;c.barh(len(bars)-1-i,w,left=left,color=cols[j],height=.7,label=lab+'%' if i==0 else None);left+=w
    c.set_yticks(range(len(bars)));c.set_yticklabels([f"{'gain' if r.startswith('mean') else 'recall'}, $\\varepsilon$={float(e):g}" for e,r in bars][::-1])
    c.set_xlabel('published, % of cases',labelpad=1);c.set_xlim(0,100)
    c.legend(frameon=False,ncol=4,loc='lower center',bbox_to_anchor=(.45,1.01),title='test min. class recall',title_fontsize=7,handlelength=1,columnspacing=.5,handletextpad=.4,borderaxespad=0)
    for k,(cell,title) in enumerate([('all recalls>0|1','(b) recalls $>0$, $\\varepsilon=1$'),('all recalls>=.50|3','(c) recalls $\\geq.5$, $\\varepsilon=3$')]):
        g=ax[k+1];s=S['selection'][cell];cu=A['cells'][cell];phis=[float(p) for p in s['development_accept_curve']]
        g.plot(phis,[s['development_accept_curve'][f'{p:g}'] for p in phis],'--o',color='#7f7f7f',ms=2.5,lw=1,label='development')
        g.plot(phis,[cu['confirmation_curve'][f'{p:g}']['accept'] for p in phis],'-o',color='#1f77b4',ms=2.5,lw=1.1,label='confirmation')
        g.axhline(cu['fixed']['accept'],color='#d62728',ls=':',lw=1,label='fixed split')
        g.scatter([cu['primary']['phi']],[cu['primary']['accept']],marker='*',s=55,color='#1f77b4',zorder=3,label='frozen choice')
        g.set_xlabel('naming share $\\phi$');g.set_ylim(-3,100)
        if k==0:g.set_ylabel('acceptance (%)')
    fig.subplots_adjust(left=.13,right=.99,bottom=.3,top=.78,wspace=.35)
    h,l=ax[1].get_legend_handles_labels();x0=ax[1].get_position().x0;x1=ax[2].get_position().x1
    fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=((x0+x1)/2,ax[1].get_position().y1+.01),handlelength=1.6,columnspacing=.8,borderaxespad=0)
    sublabel(fig,ax[0],'(a) fixed rules publish')
    for k,title in enumerate(['(b) recalls $>0$, $\\varepsilon=1$','(c) recalls $\\geq.5$, $\\varepsilon=3$']):sublabel(fig,ax[k+1],title)
    fig.savefig(paper/'figures/fig-verification-ieee.pdf',bbox_inches='tight');plt.close(fig)

def main(paper):
    fig_joint(paper);fig_value(paper);fig_ceiling(paper);fig_verification(paper);print('wrote 4 IEEE-layout figures')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--paper',type=Path,required=True);main(a.parse_args().paper)
