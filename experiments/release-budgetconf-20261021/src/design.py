"""Freeze the design of the budget-split confirmation study (EAB round 8).

The retrospective allocation analysis (release-allocation-20261019, Table IX "Opt.") chose the naming
share phi with offline TEST accuracy and scored it on the same test set. This study separates the two:
phi is selected on DEVELOPMENT cohorts whose private sources are disjoint from the CONFIRMATION cohorts,
scored on the labelled public validation split (never the test split), frozen in SELECTION.json, and
only then evaluated against the fixed 50/25/25 split on confirmation cohorts and the test split.
"""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

PHI_GAIN=[.1,.2,.3,.4,.5,.6,.7,.8,.9]
PHI_CLASS=[.1,.2,.3,.4,.5,.6,.7,.8]

def design():
    return dict(
        stage='EAB round 8: independent development selection of the budget split, frozen, then confirmation',
        question=('Does a naming share selected on independent development cohorts, without test labels, beat the fixed 50/25/25 '
                  'split on new confirmation cohorts at the same total budget?'),
        prior_knowledge_disclosed=('Designers know the retrospective allocation analysis (Table IX Opt.: at eps=1 all-recall acceptance '
            '7.8 -> 81.8 with phi=.1, recall>=.5 certifiable at eps=3) and the source-disjoint ladder (32/8 records per client).'),
        dataset='cifar10',
        source_split=('the 45000 private CIFAR images are split per class into two halves by a seeded permutation (seed 20261021): '
                      'pool A for development cohorts only, pool B for confirmation cohorts only; no image is in both'),
        clients=1000,records_per_client=20,training_records_per_client=16,heldout_records_per_client=4,
        partition=('as release-disjoint-20261020 inside the phase pool: Dirichlet(alpha=.1) class proportions per client, images '
                   'without replacement (renormalizing over non-exhausted classes); 20000 of 22500 pool images used once each'),
        development_cohort_seeds=list(range(9701,9711)),confirmation_cohort_seeds=list(range(9801,9811)),
        label_draws=dict(development='release-value-20261019/public/labels_{0..49}.json',confirmation='release-value-20261019/public/labels_{50..99}.json'),
        evaluation_records=dict(development='public validation split (3967 labelled records, never used before for CIFAR selection)',
                                confirmation='public test split (4033 records); test labels never loaded in the development phase'),
        epsilon_total=[.1,1.,3.],delta_total=1e-5,composition='sequential, one total budget per deployment',
        teacher=dict(model='linear softmax head on frozen 128-d DINOv2 features',private_steps=180,batch=128,learning_rate=.1,momentum=.9,weight_decay=5e-4),
        candidate_family='as release-value private_validation.select: public count match, eight joint weights, private centered and coverage; exact LOO on the 10 labels',
        comparator='fixed observed-count region matching',
        configurations=dict(
            fixed='rho fractions naming/gain/classes = .5/.25/.25 (the published protocol)',
            gain_grid=dict(phi_naming=PHI_GAIN,gain='1-phi',classes=0),
            class_grid=dict(phi_naming=PHI_CLASS,gain='(1-phi)/4',classes='3(1-phi)/4')),
        noise=('one fresh exact naming release per (cohort, eps, configuration), shared by its label draws; fresh exact gain and class '
               'releases per case; sha256-derived keys with prefix BUDGETCONF-20261021'),
        targets=dict(**{'gain>0':'private candidate and gain lower > 0 (fixed or gain grid)',
            'all recalls>0':'and every direct linear-band recall lower bound > 0 (fixed or class grid)',
            'all recalls>=.50':'and every recall lower bound >= .5 (fixed or class grid)'}),
        bands='gain lower and direct linear-sum class bands of release-verify-teacher-20261020 class_bands; alpha_g=alpha_h=.025',
        selection=dict(
            primary=('per target and eps, the grid phi maximizing mean unconditional VALIDATION accuracy of the published classifier '
                     '(candidate if accepted, else comparator) over the 10 development cohorts x 50 development draws'),
            secondary=('label-free: the grid phi maximizing development acceptance rate (protected decisions only)'),
            ties='largest criterion; ties within 1e-12 broken by smallest |phi-.5|, then smaller phi',
            inputs_used='development protected decisions, development candidate/comparator labels per region, validation labels',
            inputs_not_used='confirmation cohorts, test split, clean development statistics',
            frozen_in='SELECTION.json, written before any confirmation case is computed'),
        confirmation='all configurations are run on the confirmation cohorts; only fixed versus frozen selections are tested',
        primary=dict(H1=('eps=1, all recalls>0: frozen primary selection minus fixed, paired over (draw, cohort), has crossed-bootstrap '
                         '95% lower bounds > 0 for BOTH acceptance and unconditional test accuracy (50 draws x 10 cohorts, 2000 replicates, seed 20261026)')),
        predictions=['H1 holds',
            'eps=3, all recalls>=.50: frozen acceptance minus fixed has bootstrap lower bound > 0',
            'no target/eps cell: frozen unconditional test accuracy more than 1 point below fixed',
            'every frozen target/eps cell has false-accept rate <= 5% (alpha_g + alpha_h)',
            'the selected class-target phi is below .5 at eps <= 1'],
        descriptive=['regret of each frozen selection against the confirmation-best grid phi',
            'pilot size: how often a 1-cohort x 10-draw development subsample selects the same phi, and its confirmation accuracy',
            'class tail: share of published classifiers with some test-class recall below 10%',
            'label-free secondary selection versus fixed'],
        confound_disclosed='20 records per client (16/4) to make the two source pools disjoint at n=1000; settings differ from Table IX',
        reporting=dict(all_cases_retained=True,no_reruns_with_new_seeds=True,amendments='implementation fixes after this freeze are logged in AMENDMENTS.json with time and reason'))

def main():
    path=ROOT/'BUDGETCONF-DESIGN.json';d=json.loads(json.dumps(design()))
    if path.exists():
        assert json.loads(path.read_text())['design']==d,'frozen design differs';print('design unchanged');return
    rec=dict(design=d,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))})
    path.write_text(json.dumps(rec,indent=2)+'\n');print('frozen',rec['frozen_utc'])

if __name__=='__main__':main()
