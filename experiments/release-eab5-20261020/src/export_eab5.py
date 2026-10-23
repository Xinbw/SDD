"""Export fifth-EAB-round objects from frozen records (no hand-entered numbers).

Runs after export_eab3.py and export_eab4.py. Overwrites tab-decomposition.tex (decoder-matched
ceilings, signed controlled differences) and tab-verify-teacher.tex (accuracy- and balanced-accuracy
verification side by side); writes tab-cost.tex, eab5-values.tex and eab5-data.json.
"""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
EAB2=EXP/'release-eab2-20261020';EAB=EXP/'release-eab-20261020';ALLOC=EXP/'release-allocation-20261019';VT=EXP/'release-verify-teacher-20261020';DJ=EXP/'release-disjoint-20261020'
NAMES={'cifar10':'CIFAR','mnist':'MNIST','web':'Usenet','femnist62':'FEMNIST'}
J=lambda p:json.loads(Path(p).read_text())
p1=lambda x:f'{100*x:.1f}';s1=lambda x:f'{100*x:+.1f}'

def decomposition(paper,vals):
    rows=J(paper/'eab3-data.json')['rows'];dec={(r['dataset'],r['n'],r['epsilon']):r for r in J(EAB2/'analysis/naming-decomposition.json')['records']}
    lines=[]
    for r in rows:
        d=dec[(r['dataset'],r['n'],r['epsilon'])];assert abs(d['clean_named']-r['clean'])<1e-12
        t,nz,sl=r['many']-r['clean'],r['clean']-r['named'],r['named']-r['joint'];assert abs(t+nz+sl-(r['many']-r['joint']))<1e-12
        better=sum(p['released']>p['clean'] for p in d['per_seed']);r.update(teach=t,noise=nz,sel=sl,noise_improves_seeds=better,seeds=len(d['per_seed']))
        lines.append([f"{NAMES[r['dataset']]} {r['n']//1000}k/{r['epsilon']:g}",f"{p1(r['one'])}/{p1(r['many'])}",p1(r['public']),s1(r['many']-r['public']),p1(r['clean']),p1(r['named']),p1(r['joint']),
            s1(t),s1(nz),s1(sl),f"{better}/{len(d['per_seed'])}",f"{r['v']:.3f}",p1(r['gated']),p1(r['head'])])
    body='\n'.join(' & '.join(x)+r' \\' for x in lines)
    (paper/'tab-decomposition.tex').write_text('\\begin{table*}[t]\n\\caption{Where regional value is lost, at random $m=C$ labels. Ceil.: best one-to-one / many-to-one naming of the fixed public partition (offline, test labels); the centered decoder is many-to-one, so all differences use the many-to-one ceiling. Room: that ceiling minus the strongest public learner Pub. Clean: the same centered decoder on the stored noise-free votes (never released). Named: protected zero-label naming. Joint: ten-label LOO selection. $\\Delta_T,\\Delta_N,\\Delta_S$: signed controlled differences Ceil$-$Clean, Clean$-$Named, Named$-$Joint on the same partition, decoder, data and seeds; they sum to Ceil$-$Joint and are not causal attributions (negative $\\Delta_S$: fusion corrects noise). N$\\uparrow$: seeds where noise improves naming. $v$: naming-noise std. Gated: LOO unless the labels miss a class and $v\\le0.03$ (rule family proposed after earlier rounds; threshold selected on draws 0--49, reported on draws 50--99 of the same cohorts). Head: private-head joint control. \\%; Usenet uses balanced accuracy.}\n'
        '\\label{tab:decomposition}\\small\\setlength{\\tabcolsep}{3.2pt}\n\\centering\n\\begin{tabular}{@{}lrrrrrrrrrrrrr@{}}\\toprule\nTask/$\\varepsilon$ & Ceil. & Pub. & Room & Clean & Named & Joint & $\\Delta_T$ & $\\Delta_N$ & $\\Delta_S$ & N$\\uparrow$ & $v$ & Gated & Head \\\\\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n\\end{table*}\n')
    by={(r['dataset'],r['n'],r['epsilon']):r for r in rows};mn=by[('mnist',10000,.1)];us=by[('web',3000,1.)];c1=by[('cifar10',1000,.1)];f1=by[('femnist62',2000,1.)];c10=by[('cifar10',10000,.1)]
    vals.update(CeilMnistMany=p1(mn['many']),RoomMnistMany=p1(mn['many']-mn['public']),RoomCifarMany=p1(c10['many']-c10['public']),RoomUsenetMany=p1(us['many']-us['public']),
        TeachMnist=s1(mn['teach']),NoiseMnist=s1(mn['noise']),SelMnist=s1(mn['sel']),TeachUsenet=p1(us['teach']),SelCifarOne=s1(c1['sel']),
        NoiseImprMnist=f"{mn['noise_improves_seeds']}/{mn['seeds']}",NoiseImprFemnist=f"{f1['noise_improves_seeds']}/{f1['seeds']}",NoiseFemnistOne=s1(f1['noise']))
    return [{k:v for k,v in r.items()} for r in rows]

def verify_teacher(paper,vals):
    a=J(VT/'analysis/verify-teacher-analysis.json')['results'];b=J(ROOT/'analysis/balanced-verification.json')['by_setting'];lines=[]
    for t,name in [('mnist','MNIST 10k'),('cifar10','CIFAR 10k'),('web','Usenet 3k')]:
        r=a[f'{t}/1/primary'];x=b[f'{t}/1'];m='balanced_accuracy' if t=='web' else 'accuracy';u=x['utility_balanced' if t=='web' else 'utility_accuracy']
        assert abs(u['accuracy_verified']-r['utility']['gated_gain_verified'])<1e-12 and x['private_candidate_cases']==r['private_candidate_cases']
        lines.append([name,str(r['private_candidate_cases']),p1(r['gain_acceptance']),p1(x['balanced_acceptance']),str(r['false_gain_accepts']),p1(u['public']),p1(u['loo']),p1(u['gated']),p1(u['accuracy_verified']),p1(u['balanced_verified'])])
    body='\n'.join(' & '.join(x)+r' \\' for x in lines)
    (paper/'tab-verify-teacher.tex').write_text('\\begin{table}[t]\n\\caption{Gated rule plus private verification on new independent cohorts at total $\\varepsilon=1$ ($50/25/25$ split; draws 50--99; design frozen before implementation). Priv.: cases whose gated candidate uses private votes. Acceptance (\\%) among them of the pre-registered client-mean accuracy gain rule (acc) and of a balanced-accuracy rule on the same released histogram, added after the outcome (bal). FA: false accepts (acc rule). Test utility (\\%, balanced accuracy for Usenet) of the public comparator, LOO, the gated rule and the gated rule verified by each target.}\n'
        '\\label{tab:verify-teacher}\\small\\setlength{\\tabcolsep}{2.2pt}\n\\centering\n\\begin{tabular}{@{}lrrrrrrrrr@{}}\\toprule\n & & \\multicolumn{2}{c}{Accept} & & & & & \\multicolumn{2}{c}{Verified} \\\\\nTask & Priv. & acc & bal & FA & Pub. & LOO & Gated & acc & bal \\\\\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n\\end{table}\n')
    w=b['web/1'];m=b['mnist/1']
    vals.update(BalUsenetAccept=p1(w['balanced_acceptance']),BalMnistAccept=p1(m['balanced_acceptance']),BalUsenetUtil=p1(w['utility_balanced']['balanced_verified']),
        BalMnistUtil=p1(m['utility_accuracy']['balanced_verified']),AccMnistUtil=p1(m['utility_accuracy']['accuracy_verified']),
        UsenetAccVerAcc=p1(w['utility_accuracy']['accuracy_verified']),UsenetAccPubAcc=p1(w['utility_accuracy']['public']))
    return b

def cost(paper,vals):
    lad=J(EAB/'analysis/threshold-policies.json')['policies'];tb=J(ROOT/'analysis/tight-bands.json')['by_epsilon'];al=J(ALLOC/'results/allocation-analysis.json')['targets'];dj=J(DJ/'analysis/disjoint-analysis.json')
    spec=[('gain $>0$','gain>=0.0',None,'gain','gain>0'),('all recalls $>0$','recall>=0.0','recall>=0 tight','class r0=0.0 psi=0.25','all recalls>0'),
('all recalls $\\ge.50$','recall>=0.5','recall>=0.5 tight','class r0=0.5 psi=0.25','all recalls>=.50')]
    lines=[]
    for name,lk,tk,ak,dk in spec:
        for e in ['0.1','1.0','3.0']:
            f=lad[e][lk]['policy_test'];t=tb[e][tk]['policy_test'] if tk else f
            o=f"{al[ak][e]['heldout_reference']:.1f}$\\to${al[ak][e]['heldout_selected']:.1f}" if ak else '--'
            d=dj['by_epsilon'][f'{float(e):g}']['rules'][dk];lines.append([name if e=='0.1' else '',f'{float(e):g}'.removeprefix('0'),f"{lad[e][lk]['accept']:.1f}/{f:.1f}",f'{t:.1f}',o,f"{d['accept']:.1f}/{d['policy_test']:.1f}"])
    body='\n'.join(' & '.join(x)+r' \\' for x in lines)
    (paper/'tab-cost.tex').write_text('\\begin{table}[t]\n\\caption{Protocol versus target cost (test accuracy \\%; public fallback 62.3). Fixed: the $50/25/25$ ladder of Table~\\ref{tab:verify-ladder} (acceptance/utility). Tight: same released histograms, one-sided lower bands only. Optimized: fresh exact re-releases of the stored clean statistics, fixed split $\\to$ the split selected for this target on draws 0--49 (gain: no histogram), both on draws 50--99. Disjoint: new cohorts in which no private source image is reused (acceptance/utility).}\n'
        '\\label{tab:cost}\\small\\setlength{\\tabcolsep}{2.5pt}\n\\centering\n\\begin{tabular}{@{}llrrrr@{}}\\toprule\nTarget & $\\varepsilon$ & Fixed & Tight & Optimized & Disjoint \\\\\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n\\end{table}\n')
    g=dj['primary']['eps1_gain_policy_improvement'];r1=dj['by_epsilon']['1']['rules'];r3=dj['by_epsilon']['3']['rules']
    fa3=max(r3[n]['false_accepts'] for n in r3)
    mx=max(abs(v['rules'][n]['accept']-v['rules'][n]['reused_source']['accept']) for v in dj['by_epsilon'].values() for n in v['rules'])
    vals.update(TightRecallOne=f"{tb['1.0']['recall>=0 tight']['accept']:.1f}",FixedRecallOne=f"{lad['1.0']['recall>=0.0']['accept']:.1f}",
        DisjGainImpr=f"{100*g['mean']:.1f}",DisjGainCI=f"[{100*g['ci_low']:.1f},{100*g['ci_high']:.1f}]",ReusedGainImpr=f"{dj['reused_source_eps1_gain_improvement_points']:.1f}",
        DisjGainAccOne=f"{r1['gain>0']['accept']:.1f}",DisjRecallThree=f"{r3['all recalls>0']['accept']:.1f}",DisjFAThree=str(fa3),DisjMaxDiff=f'{mx:.1f}',
        DisjRecords=str(J(DJ/'DISJOINT-DESIGN.json')['design']['training_records_per_client']),DisjHeld=str(J(DJ/'DISJOINT-DESIGN.json')['design']['heldout_records_per_client']),
        DisjSources=str(J(DJ/'DISJOINT-DESIGN.json')['design']['clients']*J(DJ/'DISJOINT-DESIGN.json')['design']['records_per_client']))
    return dj

def planning(vals):
    o=J(ROOT/'analysis/planning-out-of-sample.json');vals.update(PlanOOSGap=f"{o['max_gap_points']:.1f}",PlanOOSCells=str(o['cells']),
        UsenetMassMin=f"{o['by_setting']['web/1']['class_masses_min']:.3f}",UsenetMassMax=f"{o['by_setting']['web/1']['class_masses_max']:.3f}",
        MnistClassOne=f"{o['by_setting']['mnist/1']['recall']['recorded']:.1f}")
    return o

def main(paper):
    vals={};rows=decomposition(paper,vals);b=verify_teacher(paper,vals);dj=cost(paper,vals);pl=planning(vals)
    (paper/'eab5-values.tex').write_text('% Generated: fifth EAB round (export_eab5.py).\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    (paper/'eab5-data.json').write_text(json.dumps(dict(values=vals,decomposition=rows,balanced=b,disjoint_primary=dj['primary'],disjoint_predictions=dj['predictions'],planning_out_of_sample=pl),indent=2)+'\n')
    print(json.dumps(vals,indent=1))

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--paper',type=Path,required=True);main(q.parse_args().paper.resolve())
