"""Export EAB-revision numerical objects from frozen records (no hand-entered numbers).

Inputs: paper/joint-data.json (exported frontier records), partition ceilings, the
threshold post-processing of the 3000 existing validation outputs, and the frozen
allocation feasibility study. Outputs: tab-ceiling.tex, tab-verify-ladder.tex,
tab-allocation.tex, eab-values.tex and eab-data.json.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
sys.path.insert(0,str(EXP/'release-frontier-20261019/src'))
import privacy

NAMES={'cifar10':'CIFAR','mnist':'MNIST','web':'Usenet','femnist62':'FEMNIST'}
CLASSES={'cifar10':10,'mnist':10,'web':6,'femnist62':62}

def table(path,caption,label,spec,header,rows,star=False,colsep=None):
    env='table*' if star else 'table'
    sep=f'\\setlength{{\\tabcolsep}}{{{colsep}}}\n' if colsep else ''
    body='\n'.join(' & '.join(r)+r' \\' for r in rows)
    path.write_text(f'\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\\small\n\\centering\n'+sep+f'\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\\toprule\n'
        +' & '.join(header)+r' \\'+'\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n'+f'\\end{{{env}}}\n')

def main(paper):
    joint=json.loads((paper/'joint-data.json').read_text())
    ceilings={(r['task'],r['K']):r for r in json.loads((ROOT/'analysis/partition-ceilings.json').read_text())['records']}
    thresholds=json.loads((ROOT/'analysis/threshold-policies.json').read_text())['policies']
    alloc=json.loads((EXP/'release-allocation-20261019/results/allocation-analysis.json').read_text())
    out=dict(ceiling_rows=[],ladder=thresholds,allocation=alloc['targets'],allocation_go=alloc['go'],allocation_reference_sanity=alloc['sanity_reference_all_draws'])
    rows=[];by={}
    for r in joint['frontiers']:
        if r['scheme']!='random' or r['q']!=1:continue
        if r['phase']=='development' and r['dataset'] in ['cifar10','mnist']:continue
        d=r['dataset'];C=CLASSES[d];ceil=ceilings[(d,C)]
        cert=privacy.certificate(r['epsilon'],1e-5,1,r['n'],clusters=C,dimension=C*C)
        best=max(r['fixed_public_baselines'].values(),key=lambda v:v['mean'])['mean']
        one=ceil['one_to_one_ceiling'];many=ceil['many_to_one_ceiling']
        rec=dict(dataset=d,n=r['n'],epsilon=r['epsilon'],group='I' if r['phase']=='confirmation' else 'P',ceiling_one_to_one=one,ceiling_many_to_one=many,
            public=best,fixed_private_naming=r['private_center']['mean'],joint=r['joint']['mean'],head_joint=r['controls']['head-fixed']['utility']['mean'],
            headroom=one-best,naming_loss=many-r['private_center']['mean'],mean_noise_std=cert['residual_std'],missing_class_probability=r['missing_probability'])
        out['ceiling_rows'].append(rec);by[(d,r['n'],r['epsilon'])]=rec
    order=sorted(out['ceiling_rows'],key=lambda x:({'cifar10':0,'mnist':1,'web':2,'femnist62':3}[x['dataset']],x['n'],x['epsilon']))
    for x in order:
        rows.append([f"{NAMES[x['dataset']]} {x['n']//1000}k/{x['epsilon']:g}",f"{100*x['ceiling_one_to_one']:.1f}/{100*x['ceiling_many_to_one']:.1f}",
            f"{100*x['public']:.1f}",f"{100*x['headroom']:+.1f}",f"{100*x['fixed_private_naming']:.1f}",f"{x['mean_noise_std']:.3f}",f"{100*x['joint']:.1f}",f"{100*x['head_joint']:.1f}"])
    table(paper/'tab-ceiling.tex',r'Partition ceilings and the marginal value of regional supervision at random $m=C$. Ceil.: best one-to-one / many-to-one regional naming of the fixed public partition, an offline diagnostic computed with test labels. Room: one-to-one ceiling minus the strongest public learner. Named: fixed zero-label private naming (centered). $v$: calibrated mean-noise standard deviation of the naming release. Joint and Head (joint) as in Table~\ref{tab:joint-controls}. \%; Usenet uses balanced accuracy.',
        'tab:ceiling','lrrrrrrr',[r'Task/$\varepsilon$','Ceil.','Pub.','Room','Named','$v$','Joint','Head'],rows,colsep='2.6pt')
    # verification-cost ladder on the existing releases (fixed 50/25/25 split)
    spec=[('gain>=0.0',r'gain $>0$'),('gain>=0.1',r'gain $\ge.10$'),('gain>=0.2',r'gain $\ge.20$'),('recall>=0.0',r'all recalls $>0$'),
          ('recall>=0.25',r'all recalls $\ge.25$'),('recall>=0.5',r'all recalls $\ge.50$'),('nondeg eta=0.25',r'non-degr. $\eta=.25$'),('nondeg eta=0.5',r'non-degr. $\eta=.50$')]
    lrows=[[name]+[f"{thresholds[e][k]['accept']:.1f} / {thresholds[e][k]['policy_test']:.1f}" for e in ['0.1','1.0','3.0']] for k,name in spec]
    table(paper/'tab-verify-ladder.tex',r'Verification-cost ladder on the same 3000 protected validation outputs (fixed $50/25/25$ split; no new query). Each row is a separately declared rule with its own $\alpha=.025$ class family. Entries: acceptance \% / unconditional test accuracy \% (public fallback 62.3). Non-degradation certifies $\mathrm{recall}_{f,c}-\mathrm{recall}_{b,c}\ge-\eta$ for every class on the finite held-out cohort.',
        'tab:verify-ladder','lrrr',['Publication target',r'$\varepsilon=.1$',r'$\varepsilon=1$',r'$\varepsilon=3$'],lrows)
    arows=[]
    for key,name in [('gain','gain $>0$, no histogram'),('class r0=0.0 psi=0.25','all recalls $>0$'),('class r0=0.5 psi=0.25',r'all recalls $\ge.50$')]:
        cells=[]
        for e in ['0.1','0.3','1.0','3.0']:
            x=alloc['targets'][key][e];curve=list(x['selection_curve_draws0_49'].values())
            share='flat' if max(curve)-min(curve)<1e-9 else f"{x['selected_phi_naming']:g}"
            cells.append(f"{x['heldout_reference']:.1f}$\\to${x['heldout_selected']:.1f} ({share})")
        arows.append([name]+cells)
    table(paper/'tab-allocation.tex',r'Joint allocation of one total budget between naming and verification: fresh exact re-releases of stored clean statistics, ten cohorts, 100 label draws. Entries: test accuracy of the fixed $50/25/25$ reference $\to$ the naming share selected on draws 0--49, both evaluated on draws 50--99 (selected naming share; flat: every share rejects every case). Class rules give the gain release $25\%$ of the verification budget.',
        'tab:allocation','lrrrr',['Target',r'$\varepsilon=.1$',r'$\varepsilon=.3$',r'$\varepsilon=1$',r'$\varepsilon=3$'],arows,star=True)
    c1=by[('cifar10',1000,.1)];c10=by[('cifar10',10000,.1)];mn=by[('mnist',10000,.1)];us=by[('web',3000,1.)];fe=by[('femnist62',2000,1.)]
    a1=alloc['targets']['class r0=0.0 psi=0.25']['1.0'];a3=alloc['targets']['class r0=0.5 psi=0.25']['3.0'];ag=alloc['targets']['gain']
    vals=dict(CeilCifar=f"{100*c1['ceiling_one_to_one']:.1f}",CeilMnist=f"{100*mn['ceiling_one_to_one']:.1f}",CeilUsenet=f"{100*us['ceiling_one_to_one']:.1f}",
        CeilFemnistOne=f"{100*fe['ceiling_one_to_one']:.1f}",CeilFemnistMany=f"{100*fe['ceiling_many_to_one']:.1f}",
        RoomCifar=f"{100*c1['headroom']:.1f}",RoomMnist=f"{100*mn['headroom']:.1f}",RoomUsenet=f"{100*us['headroom']:.1f}",RoomFemnist=f"{100*fe['headroom']:.1f}",
        NamedMnist=f"{100*mn['fixed_private_naming']:.1f}",NamedUsenet=f"{100*us['fixed_private_naming']:.1f}",NamedCifarOne=f"{100*c1['fixed_private_naming']:.1f}",
        NoiseCifarOne=f"{c1['mean_noise_std']:.3f}",NoiseCifarTen=f"{c10['mean_noise_std']:.3f}",NoiseUsenet=f"{us['mean_noise_std']:.3f}",
        JointMnist=f"{100*mn['joint']:.1f}",HeadMnist=f"{100*mn['head_joint']:.1f}",PublicMnist=f"{100*mn['public']:.1f}",
        LadderGainTwoTenth=f"{thresholds['1.0']['gain>=0.2']['accept']:.1f}",LadderRecallHalfThree=f"{thresholds['3.0']['recall>=0.5']['accept']:.1f}",
        LadderRecallQuarterThree=f"{thresholds['3.0']['recall>=0.25']['accept']:.1f}",LadderNondegHalf=f"{thresholds['3.0']['nondeg eta=0.5']['accept']:.1f}",
        LadderNondegQuarter=f"{thresholds['3.0']['nondeg eta=0.25']['accept']:.1f}",
        AllocClassOneRef=f"{a1['heldout_reference']:.1f}",AllocClassOneSel=f"{a1['heldout_selected']:.1f}",AllocClassOneGain=f"{a1['improvement']:.1f}",
        AllocClassOneCI=f"[{a1['ci95'][0]:.2f},{a1['ci95'][1]:.2f}]",AllocHalfThreeRef=f"{a3['heldout_reference']:.1f}",AllocHalfThreeSel=f"{a3['heldout_selected']:.1f}",
        AllocGainTenth=f"{ag['0.1']['improvement']:.1f}",AllocGainTenthCI=f"[{ag['0.1']['ci95'][0]:.2f},{ag['0.1']['ci95'][1]:.2f}]",
        AllocGainThird=f"{ag['0.3']['improvement']:.1f}",AllocGainThirdCI=f"[{ag['0.3']['ci95'][0]:.2f},{ag['0.3']['ci95'][1]:.2f}]")
    (paper/'eab-values.tex').write_text('% Generated: EAB revision analyses (export_eab.py).\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    (paper/'eab-data.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(vals,indent=1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);main(p.parse_args().paper.resolve())
