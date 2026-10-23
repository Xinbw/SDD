"""Twelfth EAB round exporter: macros for external baselines, the weak-encoder ablation, the FEMNIST
replication and the corrected-constant certificate check (eab12-values.tex, eab12-data.json)."""
import argparse,json
from pathlib import Path
import numpy as np
STAGE=Path(__file__).resolve().parents[1];EXP=STAGE.parent
RE=EXP/'review-evidence-20261012/results';ENC=EXP/'release-encoder-20261022';FC=EXP/'release-femnistconf-20261022'

def f1(x):return f'{x:.1f}'
def sgn(x):return ('+' if x>=0 else '$-$')+f'{abs(x):.1f}'

def external():
    out={}
    for d in ['cifar10','mnist']:
        out[d]={}
        for m in ['SDD','DP-FedCT','PATE-DI','DP-Head']:
            v=[json.loads((RE/f'{d}_n10000_a0.1_e0.1_s{s}_{m}.json').read_text())['accuracy'] for s in range(801,806)]
            out[d][m]=100*float(np.mean(v))
    return out

def main(paper):
    ext=external();enc=json.loads((ENC/'analysis/encoder-analysis.json').read_text());tau=json.loads((STAGE/'analysis/tau-recheck.json').read_text())
    fc=json.loads((FC/'analysis/femnistconf-analysis.json').read_text())
    r=enc['results']['resnet18'];g=r['random']['m10']['gain_vs_strongest']
    m={'TauCerts':f"{tau['certificates']}",'TauMaxLog':f"-{abs(tau['max_log10_tau'])/1e6:.1f}\\times10^{{6}}",
       'ExtSeeds':'5'}
    for d,D in [('cifar10','Cifar'),('mnist','Mnist')]:
        for k,K in [('SDD','SDD'),('DP-FedCT','FedCT'),('PATE-DI','PATE'),('DP-Head','Head')]:m[f'Ext{K}{D}']=f1(ext[d][k])
    m.update(EncCeilOne=f1(r['ceilings']['one_to_one']),EncCeilMany=f1(r['ceilings']['many_to_one']),EncPublic=f1(r['random']['m10']['strongest_public']['accuracy']),
        EncJoint=f1(r['random']['m10']['joint']),EncGain=sgn(g['mean']),EncGainCI=f"[{sgn(g['ci_low'])},{sgn(g['ci_high'])}]",
        EncDirected=f1(r['directed']['m10']['portfolio']),EncHead=f1(r['native_head']),EncPrivRate=f"{r['random']['m10']['private_selected_rate']:.0f}")
    c=fc['cells'];p=fc['primary']
    m.update(FCCases=f"{fc['cases']}",FCPlanGap=f1(p['max_planner_gap_points']),
        FCGainOne=f1(c['gain>0|1']['fixed']['accept']),FCGainOneTrue=f1(c['gain>0|1']['fixed']['target_true_rate']),
        FCClassMax=f1(max(c[f'{t}|{e:g}'][x]['accept'] for t in ['all recalls>0','all recalls>=.50'] for e in [1.,3.,10.] for x in ['fixed','frozen'])),
        FCTail=f1(c['gain>0|1']['fixed']['published_min_recall_below_10pct']),
        FCFA=f"{sum(v[x]['false_accepts'] for v in c.values() for x in ['fixed','frozen'])}",
        FCUtilFixed=f1(c['gain>0|1']['fixed']['utility']),FCPub=f1(c['gain>0|1']['fixed']['public_fallback']))
    (paper/'eab12-values.tex').write_text(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k,v in m.items()))
    (paper/'eab12-data.json').write_text(json.dumps(dict(external=ext,encoder=enc['primary'],encoder_predictions=enc['predictions'],femnist=fc['primary'],
        femnist_predictions=fc['registered_predictions'],tau=tau,macros=m),indent=2)+'\n')
    print('wrote',len(m),'macros')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--paper',type=Path,required=True);main(a.parse_args().paper)
