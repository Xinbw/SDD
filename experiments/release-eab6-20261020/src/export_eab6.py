"""Export sixth-EAB-round objects (runs after export_eab5.py; no hand-entered numbers).

Overwrites tab-cost.tex with the single-subset (draws 50-99) comparison including acceptance and the
selected splits; writes eab6-values.tex and eab6-data.json.
"""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
J=lambda n:json.loads((ROOT/'analysis'/n).read_text())

def main(paper):
    ct=J('cost-table.json')['by_target'];c24=J('calculator-24.json');ht=J('disjoint-heterogeneity.json')
    cell=lambda x:f"{x['accept']:.1f}/{x['utility']:.1f}";lines=[]
    for name,label in [('gain>0','gain $>0$'),('all recalls>0','all recalls $>0$'),('all recalls>=.50','all recalls $\\ge.50$')]:
        for i,e in enumerate(['0.1','1.0','3.0']):
            x=ct[name][e];o=x['optimized'];phi='--' if o['flat_curve'] else f"{o['phi_naming']:g}".removeprefix('0')
            lines.append([label if i==0 else '',f'{float(e):g}'.removeprefix('0'),cell(x['fixed']),cell(x['tight']),cell(x['optimized_reference']),cell(o),phi,cell(x['disjoint'])])
    body='\n'.join(' & '.join(r)+r' \\' for r in lines)
    (paper/'tab-cost.tex').write_text('\\begin{table}[t]\n\\caption{Protocol versus target cost on label draws 50--99 (acceptance \\% / test accuracy \\%; public fallback 62.8). Fixed: recorded $50/25/25$ outputs. Tight: same histograms, one-sided lower bands. Re-rel.: fresh exact re-release of the stored clean statistics at $50/25/25$. Opt.: the naming share $\\phi$ chosen for this target on draws 0--49 by test utility (gain: $\\phi\\in\\{.1,\\dots,.9\\}$, rest to the gain report, no histogram; classes: $\\phi\\in\\{.1,\\dots,.8\\}$, the gain report takes $1/4$ of the rest; --: all shares tie). Disjoint: cohorts without source reuse. Non-degradation was not part of the allocation study.}\n'
        '\\label{tab:cost}\\small\\setlength{\\tabcolsep}{1.6pt}\n\\centering\n\\begin{tabular}{@{}llrrrrrr@{}}\\toprule\nTarget & $\\varepsilon$ & Fixed & Tight & Re-rel. & Opt. & $\\phi$ & Disjoint \\\\\n\\midrule\n'+body+'\n\\bottomrule\\end{tabular}\n\\end{table}\n')
    r1=ct['all recalls>0']['1.0'];h5=ct['all recalls>=.50']['3.0']
    vals=dict(CalcCells=str(c24['cells']),CalcGap=f"{c24['max_gap_points']:.1f}",
        CostFixedRecallOne=f"{r1['fixed']['accept']:.1f}",CostTightRecallOne=f"{r1['tight']['accept']:.1f}",CostRefRecallOne=f"{r1['optimized_reference']['accept']:.1f}",
        CostOptRecallOne=f"{r1['optimized']['accept']:.1f}",CostOptRecallOneUtil=f"{r1['optimized']['utility']:.1f}",CostRefRecallOneUtil=f"{r1['optimized_reference']['utility']:.1f}",
        CostOptRecallOnePhi=f"{r1['optimized']['phi_naming']:g}".removeprefix('0'),CostOptHalfThree=f"{h5['optimized']['accept']:.1f}",
        HetDisjoint=f"{ht['disjoint']['top_class_share']:.3f}",HetLast=f"{ht['disjoint_last100']['top_class_share']:.3f}",HetReused=f"{ht['reused_source_600']['top_class_share']:.3f}",
        HetFree=f"{ht['unconstrained_dirichlet_40']['top_class_share']:.3f}")
    (paper/'eab6-values.tex').write_text('% Generated: sixth EAB round (export_eab6.py).\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    (paper/'eab6-data.json').write_text(json.dumps(dict(values=vals,cost=ct,calculator_24=c24,heterogeneity=ht),indent=2)+'\n');print(json.dumps(vals,indent=1))

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--paper',type=Path,required=True);main(q.parse_args().paper.resolve())
