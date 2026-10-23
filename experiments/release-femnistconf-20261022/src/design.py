"""Freeze the FEMNIST verification replication (EAB round 12, reviewer W4).

Replicates the budget-split protocol of release-budgetconf-20261021 (CIFAR) on
official 62-class Federated EMNIST with natural writer clients and strongly
unequal class masses, and adds a planner test: before any confirmation case,
the Gaussian-approximation calculator (release-eab3 planning.accept_probability,
generalized to the configuration's rho fractions) predicts confirmation
acceptance from development-cohort effect sizes; the prediction is frozen with
the selection.
"""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PHI_GAIN=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
PHI_CLASS=[.1,.2,.3,.4,.5,.6,.7,.8]

def design():
    return dict(
        stage='EAB round 12: FEMNIST replication of independent split selection and planner-predicted verification',
        prior_knowledge_disclosed=('Designers know the CIFAR budgetconf results (H1 held) and the inherited FEMNIST joint-naming results '
            '(m=62 random labels: joint below the strongest public learner at n=1k; about 32 of 62 classes missing per label draw).'),
        dataset='femnist62 (release-granularity inputs, DINOv2/128 features, K=C=62 public regions)',
        writers=('the 2600 private writers are split by a seeded permutation (seed 20261022) into pool A (1300, development only) and '
                 'pool B (1300, confirmation only); no writer is in both'),
        clients=1000,client_sampling='1000 distinct writers per cohort without replacement from the phase pool',
        records=('each writer keeps all original records; a seeded per-writer permutation puts floor(0.8*count) records in training and the '
                 'rest in a held-out set used only for verification (at least one held-out record)'),
        development_cohort_seeds=list(range(9901,9911)),confirmation_cohort_seeds=list(range(9911,9921)),
        label_draws=dict(source='release-frontier public/femnist62-label-samples.json, scheme random, q=1 (m=62)',development=list(range(0,50)),confirmation=list(range(50,100))),
        evaluation_records=dict(development='public validation split (4685 records of disjoint writers)',confirmation='public test split (4549 records); never loaded in development'),
        epsilon_total=[1.,3.,10.],epsilon_note='eps=10 probes where class evidence becomes certifiable at C=62; eps=.1 is omitted (no target certifiable on CIFAR)',
        delta_total=1e-5,composition='sequential, one total budget per deployment',
        teacher=dict(model='linear softmax head on frozen 128-d features',private_steps=180,batch=128,learning_rate=.1,momentum=.9,weight_decay=5e-4),
        candidate_family='public count matching, eight joint weights, private centered and coverage; exact LOO accuracy on the 62 labels (as CIFAR)',
        comparator='fixed observed-count region matching',
        configurations=dict(fixed='.5/.25/.25',gain_grid=dict(phi_naming=PHI_GAIN,gain='1-phi',classes=0),
            class_grid=dict(phi_naming=PHI_CLASS,gain='(1-phi)/4',classes='3(1-phi)/4')),
        noise='sha256-derived keys with prefix FEMNISTCONF-20261022',
        targets={'gain>0':'private candidate and client-mean gain lower > 0','all recalls>0':'and every class recall lower bound > 0',
                 'all recalls>=.50':'and every class recall lower bound >= .5'},
        truth=('targets are judged on the clean client-mean held-out statistics of the cohort (the quantity the release certifies); '
               'false accept = accepted while the clean statistic misses the target'),
        bands='as budgetconf (gain lower bound, direct linear-sum class bands); alpha .025 each',
        selection='as budgetconf: per target and eps, grid phi maximizing mean validation accuracy of the published classifier; ties by |phi-.5| then phi; frozen in SELECTION.json before any confirmation case',
        planner=dict(inputs='development clean client-mean statistics per case (gain, class masses, correct masses) of private candidates; 0 for public candidates',
            model='release-eab3 planning.accept_probability with the configuration rho fractions',
            prediction='mean predicted acceptance over development cases = predicted confirmation acceptance, for fixed and frozen configurations of every target/eps',
            frozen_in='PREDICTIONS.json, written with SELECTION.json before any confirmation case'),
        primary=dict(F1='every fixed and frozen target/eps cell: |planner prediction - confirmation acceptance| <= 5 points',
                     F2='every fixed and frozen target/eps cell: false-accept rate <= 5% of accepted cases',success='F1 and F2'),
        predictions=['F1 holds','F2 holds','no frozen cell more than 1 point below fixed in confirmation test accuracy',
            'gain>0 is accepted more often than all recalls>0 at every eps under the fixed split',
            'no class target is accepted in more than 5% of cases at eps<=3'],
        descriptive=['class tail of classifiers published by the gain rule','regret against the confirmation-best phi','confirmation acceptance curves'],
        reporting=dict(all_cases_retained=True,no_reruns_with_new_seeds=True,amendments='AMENDMENTS.json with time and reason'))

def main():
    path=ROOT/'FEMNISTCONF-DESIGN.json';d=json.loads(json.dumps(design()))
    if path.exists():
        assert json.loads(path.read_text())['design']==d,'frozen design differs';print('design unchanged');return
    rec=dict(design=d,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))})
    path.write_text(json.dumps(rec,indent=2)+'\n');print('frozen',rec['frozen_utc'])

if __name__=='__main__':main()
