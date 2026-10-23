"""Freeze the design of the source-disjoint verification ladder (EAB round 5, step 2).

Every private CIFAR-10 source image belongs to at most one client and is used at most once
(training or held-out). Otherwise the protocol of the CIFAR-1k verification demonstration
(release-value-20261019/private_validation.py) is kept: candidate family, comparator, label draws,
budget split, bands and publication targets.
"""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent

def design():
    return dict(
        stage='EAB round 5, step 2: verification-cost ladder with source-disjoint image clients',
        question='Does the CIFAR verification ladder, and the private gain it certifies, survive when no private source image is reused across or within clients?',
        prior_knowledge_disclosed='Designers know the reused-source CIFAR-1k ladder (tab-verify-ladder) and its 26.6-point eps=1 gain-policy improvement over the public fallback.',
        dataset='cifar10',clients=1000,records_per_client=40,training_records_per_client=32,heldout_records_per_client=8,
        partition=('sequential: each client draws Dirichlet(alpha=.1) class proportions and 40 labels; images are taken WITHOUT replacement '
            'from per-class pools of the 45000 private images (renormalizing over non-exhausted classes); the first 32 are training, '
            'the last 8 held-out; 40000 images used, each exactly once; client seed = cohort seed'),
        source_reuse=0,cohort_seeds=list(range(9601,9611)),epsilon_total=[.1,1.,3.],delta_total=1e-5,
        zcdp_budget_fractions=dict(naming=.5,gain=.25,classes=.25),composition='sequential, one total budget per deployment',
        teacher=dict(model='linear softmax head on frozen 128-d DINOv2 features',private_steps=180,batch=128,learning_rate=.1,momentum=.9,weight_decay=5e-4),
        naming='K=C/M; one protected aggregate at rho fraction .5 (as release-value private_validation)',
        candidate_family='as release-value private_validation.select: public count match, eight joint weights, private centered and coverage; exact LOO on the 10 labels',
        comparator='fixed observed-count region matching (as release-value)',
        label_draws='release-value-20261019/public/labels_{0..99}.json (the same 100 ten-label draws as the reused-source demonstration)',
        validation=dict(gain_report='per-client held-out accuracy difference, s=2^16, integer sensitivity s',class_report='C x 4 joint correctness histogram per client, normalized by its 8 records, squared sensitivity 2 s^2',
            bands='gain lower and direct linear-sum class bands of release-value-20261019; non-degradation band of release-eab-20261020/thresholds.py; alpha=.025'),
        ladder_targets=['gain>0','gain>=.10','gain>=.20','all recalls>0','all recalls>=.25','all recalls>=.50','non-degradation eta=.25','non-degradation eta=.50'],
        primary=dict(S1='at eps=1, the gain-verified policy has unconditional test accuracy above the public fallback with crossed-bootstrap 95% lower bound > 0 (cohorts x draws, 2000 replicates, seed 20261025)',
            reported_against='the reused-source ladder at the same eps (paired label draws, different cohorts)'),
        predictions=['S1 holds','acceptance is monotone along the ladder at every eps (gain>0 >= gain>=.10 >= gain>=.20; recalls>0 >= >=.25 >= >=.50; eta=.50 >= eta=.25)',
            'no false accept at eps >= 1 for any ladder rule',
            'the eps=1 gain-policy improvement over the public fallback is smaller than in the reused-source demonstration (fewer training records per client)'],
        confound_disclosed='source-disjointness at n=1000 forces 32 training records per client instead of 480; the contrast mixes source reuse with per-client data size',
        reporting=dict(all_cases_retained=True,no_reruns_with_new_seeds=True,test_labels='loaded only after all protected decisions',
            amendments='implementation fixes after this freeze are logged in AMENDMENTS.json with time and reason'))

def main():
    path=ROOT/'DISJOINT-DESIGN.json';d=json.loads(json.dumps(design()))
    if path.exists():
        assert json.loads(path.read_text())['design']==d,'frozen design differs';print('design unchanged');return
    rec=dict(design=d,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))})
    path.write_text(json.dumps(rec,indent=2)+'\n');print('frozen',rec['frozen_utc'])

if __name__=='__main__':main()
