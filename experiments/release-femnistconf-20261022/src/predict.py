"""Planner prediction of confirmation acceptance from development effect sizes (writes PREDICTIONS.json).

Runs after SELECTION.json and before any confirmation case. For each development case it reads the
clean client-mean held-out statistics (offline effect sizes: gain, class masses, correct masses) and
evaluates the release-eab3 Gaussian-approximation acceptance probability with the configuration's rho
fractions (0 for public candidates). The mean over development cases is the predicted confirmation
acceptance of that configuration.
"""
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent;D=json.loads((ROOT/'FEMNISTCONF-DESIGN.json').read_text())['design']
sys.path.insert(0,str(EXP/'release-frontier-20261019/src'));sys.path.insert(0,str(EXP/'release-eab3-20261020/src'))
import privacy,planning
S=planning.S;A=planning.A;DELTA=planning.DELTA;C=62;N=D['clients']
TARGETS=[('gain>0','gain',0.),('all recalls>0','recall',0.),('all recalls>=.50','recall',.5)]
def fractions(config):
    if config=='fixed':return .25,.25
    p=float(config[4:] if config.startswith('gain') else config[5:])
    return (1-p,0.) if config.startswith('gain') else ((1-p)/4,3*(1-p)/4)
def noise(e,fg,fc):
    g=privacy.certificate(e,DELTA,1,N,integer_sensitivity=S,dimension=1,rho_fraction=fg)['residual_std']
    h=privacy.certificate(e,DELTA,1,N,integer_sensitivity_squared=2*S*S,dimension=4*C,rho_fraction=fc)['residual_std'] if fc>0 else None
    return g,h
def accept_probability(kind,level,vg,vh,gain,mass,correct):
    """planning.accept_probability with the noise scales passed in (identical formulas)."""
    p=float(planning.Phi((gain-2*vg*math.sqrt(2*math.log(1/A))-1/S-(level if kind=='gain' else 0))/(2*vg)))
    if kind=='gain':return p
    z=math.sqrt(2*math.log(4*C/A));rn=math.sqrt(2)*vh*z+2/S;rd=2*vh*z+4/S
    return p*float(np.prod(planning.Phi((correct-rn-level*(mass+rd))/(vh*math.sqrt(2*(1-level)**2+2*level**2)))))
def check_equivalence():
    rng=np.random.default_rng(0)
    for e in [1.,3.]:
        vg,vh=planning.noise(e,N,C)
        for kind,level in [('gain',0.),('recall',0.),('recall',.5)]:
            m=rng.dirichlet(np.ones(C));c=m*rng.uniform(.3,1,C);g=.2
            assert abs(accept_probability(kind,level,vg,vh,g,m,c)-planning.accept_probability(kind,level,e,N,C,g,m,c,np.zeros(C)))<1e-12
def main():
    assert not list((ROOT/'results').glob('confirm_*.json')),'confirmation results exist'
    assert (ROOT/'SELECTION.json').exists(),'freeze the selection first';check_equivalence()
    sel=json.loads((ROOT/'SELECTION.json').read_text())['selection'];cells={}
    files=sorted((ROOT/'results').glob('dev_s*.json'));assert len(files)==len(D['development_cohort_seeds']) and all('_clean' not in f.name for f in files)
    cases=[]
    for f in files:
        r=json.loads(f.read_text());z=np.load(ROOT/'results'/r['clean_file'])['mass_correct']
        for row in r['rows']:cases.append((row['e'],row['config'],row['private'],row['true_gain'],z[row['clean_index']]))
    configs=sorted({c[1] for c in cases})
    for t,kind,level in TARGETS:
        fam='gain' if kind=='gain' else 'class'
        for e in D['epsilon_total']:
            out={}
            for cfg in configs:
                if not (cfg=='fixed' or cfg.startswith(fam)):continue
                fg,fc=fractions(cfg);vg,vh=noise(e,fg,fc)
                P=[accept_probability(kind,level,vg,vh,g,mc[0],mc[1]) if priv else 0. for (ee,cc,priv,g,mc) in cases if ee==e and cc==cfg]
                out[cfg]=100*float(np.mean(P))
            s=sel[f'{t}|{e:g}'];frozen=f"{fam}{s['primary_phi']:g}"
            cells[f'{t}|{e:g}']=dict(target=t,epsilon=e,predicted_fixed=out['fixed'],frozen_config=frozen,predicted_frozen=out[frozen],predicted_by_config=out)
    rec=dict(predictions=cells,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selection_sha256=hashlib.sha256((ROOT/'SELECTION.json').read_bytes()).hexdigest(),
        development_results_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        planning_source_sha256=hashlib.sha256((EXP/'release-eab3-20261020/src/planning.py').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_read='development clean client-mean held-out statistics (offline effect sizes) and the frozen selection',inputs_not_read='any confirmation record or test label')
    path=ROOT/'PREDICTIONS.json';assert not path.exists(),'PREDICTIONS.json already frozen';path.write_text(json.dumps(rec,indent=2)+'\n')
    for k,v in cells.items():print(k,'fixed %.1f'%v['predicted_fixed'],v['frozen_config'],'%.1f'%v['predicted_frozen'])
if __name__=='__main__':main()
