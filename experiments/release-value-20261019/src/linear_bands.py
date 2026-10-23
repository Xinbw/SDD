"""Analytical histogram postprocessing: simultaneous linear-function bands.

No new protected query. Numerator and denominator sums use their direct
sub-Gaussian proxies, rather than summing coordinate-wise radii.
The initial coordinate-band protocol and records remain immutable.
"""
import argparse,json,math
import numpy as np
from study import ROOT,save,sha,freeze
C=10;S=1<<16;ALPHA=.025

def design():
    freeze('LINEAR-BANDS-DESIGN.json',dict(scope='Analytical postprocessing addition after initial coordinate-band experiment; initial outputs retained',
        protected_source='same existing C x 4 histogram and same gain release',new_private_queries=0,new_privacy_budget=0,
        fixed_targets='C candidate-correct masses (two coordinates each), C class masses (four each)',
        class_family_alpha=ALPHA,union='two-sided union over 2C fixed linear targets, 4C one-sided events',
        numerator_radius='sqrt(2)*v*sqrt(2 log(4C/alpha)) + 2/s',denominator_radius='2*v*sqrt(2 log(4C/alpha)) + 4/s',
        rule='same gain lower >0 and private candidate selected, plus every new recall lower >0',
        clean_private_or_test_statistics_choose_rule=False,no_weight_or_threshold_search=True))

def bands(released,cert):
    z=math.sqrt(2*math.log(4*C/ALPHA));v=cert['residual_std']
    rn=math.sqrt(2)*v*z+2/S;rd=2*v*z+4/S
    h=np.asarray(released).reshape(C,4);mass=h.sum(1);correct=h[:,2]+h[:,3]
    dl=np.clip(mass-rd,0,1);du=np.clip(mass+rd,0,1);nl=np.clip(correct-rn,0,1);nu=np.clip(correct+rn,0,1)
    lower=np.clip(np.divide(nl,du,out=np.zeros(C),where=du>0),0,1)
    upper=np.clip(np.divide(nu,dl,out=np.ones(C),where=dl>0),0,1)
    invalid=lower>upper;lower[invalid]=0;upper[invalid]=1
    return dict(numerator_radius=rn,denominator_radius=rd,recall_lower=lower.tolist(),recall_upper=upper.tolist(),minimum_recall_lower=float(lower.min()))

def main():
    design();folder=ROOT/'analysis/linear-validation';folder.mkdir(exist_ok=True)
    for path in sorted((ROOT/'results/validation').glob('*.json')):
        target=folder/path.name
        if target.exists():continue
        r=json.loads(path.read_text());diag=np.load(ROOT/r['diagnostic_file']);bound=bands(diag['released_class_histogram'],r['privacy_classes'])
        accepted=bool(r['candidate_uses_private'] and r['protected_validation']['gain_lower']>0 and bound['minimum_recall_lower']>0)
        d=r['research_diagnostics'];true=np.array(d['true_recall']);coverage=bool(np.all(np.array(bound['recall_lower'])<=true+1e-12)&np.all(np.array(bound['recall_upper'])>=true-1e-12))
        save(target,dict(source=path.name,source_sha256=sha(path),seed=r['seed'],draw=r['draw'],epsilon_total=r['epsilon_total'],bounds=bound,
            class_checked_accepted=accepted,uses_clean_statistics_for_decision=False,new_private_queries=0,
            policy_test=d['candidate_test'] if accepted else d['public_test'],policy_heldout_accuracy=d['candidate_heldout_accuracy'] if accepted else d['public_heldout_accuracy'],
            research_diagnostics=dict(simultaneous_bands_valid=coverage,false_accept=bool(accepted and (d['true_gain']<=0 or min(d['true_recall'])<=0)))))
    print('Direct class-sum bounds: 3000 protected-histogram postprocessing cases.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');a=p.parse_args();design()
    if not a.freeze_only:main()
