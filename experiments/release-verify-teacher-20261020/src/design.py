"""Freeze the design of the gated-rule + private-verification confirmation (EAB round 3, step 2).

Run once before any implementation output exists; later runs assert the design is unchanged.
New independent actual cohorts on MNIST 10k, CIFAR-10 10k and Usenet 3k; one total budget per
deployment split 50/25/25 (naming / paired gain / class histogram), sequential composition.
"""
import hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent
sys.path.insert(0,str(EXP/'release-frontier-20261019/src'))
import privacy

TASKS=[dict(dataset='mnist',classes=10,naming_clients=10000,validation_clients=10000,label_draw_source='release-frontier-20261019/confirmation/samples (seed 11000001), q=1, m=C',public_records='release-frontier-20261019/analysis/joint-confirmation'),
       dict(dataset='cifar10',classes=10,naming_clients=10000,validation_clients=10000,label_draw_source='release-frontier-20261019/confirmation/samples (seed 11000001), q=1, m=C',public_records='release-frontier-20261019/analysis/joint-confirmation'),
       dict(dataset='web',classes=6,naming_clients=3000,validation_clients=1249,label_draw_source='release-frontier-20261019/public samples (development draws), q=1, m=C',public_records='release-frontier-20261019/analysis/joint-naming')]
EPSILONS=[.1,1.];DELTA=1e-5;FRACTIONS=dict(naming=.5,gain=.25,classes=.25);S=1<<16;ALPHA=.025;V0=.03
SEEDS={'mnist':list(range(9201,9211)),'cifar10':list(range(9301,9311)),'web':list(range(9401,9411))}

def expected_scales():
    out={}
    for t in TASKS:
        C=t['classes']
        for e in EPSILONS:
            vn=privacy.certificate(e,DELTA,1,t['naming_clients'],clusters=C,dimension=C*C,rho_fraction=FRACTIONS['naming'])['residual_std']
            vg=privacy.certificate(e,DELTA,1,t['validation_clients'],integer_sensitivity=S,dimension=1,rho_fraction=FRACTIONS['gain'])['residual_std']
            vh=privacy.certificate(e,DELTA,1,t['validation_clients'],integer_sensitivity_squared=2*S*S,dimension=4*C,rho_fraction=FRACTIONS['classes'])['residual_std']
            out[f"{t['dataset']}/{e:g}"]=dict(naming_noise_std=vn,gate_open=vn<=V0,gain_band_radius=2*vg*math.sqrt(2*math.log(1/ALPHA))+1/S,
                class_numerator_radius=math.sqrt(2)*vh*math.sqrt(2*math.log(4*C/ALPHA))+2/S,class_mass=1/C)
    return out

def design():
    return dict(
        stage='EAB round 3, step 2: does private verification separate good from poor teacher content? Confirmation of the gated selection rule on new independent cohorts.',
        question='With the frozen gated rule as candidate and the within-budget public portfolio as comparator, does total-budget private verification accept on MNIST (good teacher) and reject on Usenet (poor teacher content)?',
        prior_knowledge_disclosed=('The designers have seen all existing records: the old-cohort selection-rule study (release-eab2-20261020), '
            'and old-cohort test metrics showing protected centered naming above the LOO public portfolio on MNIST/CIFAR-10k and below it on Usenet '
            '(full naming budget). The predictions below are therefore informed; the new evidence is the protected verification on new cohorts.'),
        tasks=TASKS,epsilon_total=EPSILONS,delta_total=DELTA,zcdp_budget_fractions=FRACTIONS,
        composition='naming, gain and class releases composed sequentially into one total (epsilon, delta) per deployment, although naming and validation clients are disjoint; no parallel-composition credit is claimed',
        cohorts=dict(seeds=SEEDS,independent_of_all_previous_cohorts=True,
            images=('naming clients: 600 records with replacement under Dirichlet alpha=.1 from the TRAINING source pool; validation clients: a disjoint '
                'set of n constructed clients, 120 records each with replacement under their own Dirichlet alpha=.1 proportions from the HELD-OUT source '
                'pool; pools = stratified 80/20 split of the private pool, seed 2026101901 (the CIFAR-1k verification split rule, applied per task)'),
            usenet='naming clients: 3000 From-header authors drawn without replacement from the 4249 eligible authors (cohort seed); validation clients: the remaining 1249 authors with all their posts',
            client_unit='a complete client dataset; naming and validation clients are different clients',
            validation_population_seed='cohort seed + 500000'),
        teacher=dict(model='linear softmax head on the frozen 128-d features',private_steps=180,batch=128,learning_rate=.1,momentum=.9,weight_decay=5e-4,release_rounds=1,
            naming='K=C public partition, M votes, one protected aggregate at rho fraction .5, centered and coverage decoders as in release-frontier-20261019'),
        label_draws=dict(m='C (q=1)',draws='the 100 existing random draws per task listed in tasks.label_draw_source; public learner outputs reused and asserted equal on test',
            primary_draws=list(range(50,100)),secondary_draws=list(range(0,50)),
            reason='draws 50-99 were never used to select the gated rule; draws 0-49 selected it on old cohorts'),
        candidate=dict(rule=f'cover-noise-{V0:g} (frozen in release-eab2-20261020): if the m=C labels miss a class and the naming noise std v <= {V0:g}, the protected centered naming; otherwise exact LOO over the frozen family (14 public learners, 8 joint weights, private centered and coverage)',
            v='residual_std of the naming certificate at rho fraction .5 (public function of n, epsilon, K)',
            gate_status_now=None),
        comparator='the LOO-selected public portfolio on the same m=C labels (deployable public fallback); its predictions on validation records are the same public models queried inductively',
        validation=dict(gain_report='per-client held-out accuracy difference d_i=u_i(f)-u_i(b) in [-1,1], report round half-up s(1+d_i)/2, s=2^16, integer sensitivity s',
            class_report='C x 4 joint correctness histogram per client normalized by its record count, exact largest-remainder integer simplex, squared sensitivity 2 s^2',
            gain_lower='released gain - 2 v_g sqrt(2 log(1/alpha)) - 1/s, alpha=.025',
            class_bands='direct linear-sum bands of release-value-20261019/linear_bands.py, alpha=.025',
            noise='independent exact distributed discrete Gaussian shares; deterministic research keys',
            releases_always_made=True),
        policies=dict(public='publish the comparator b',loo='publish the LOO choice (no verification)',gated='publish the gated candidate (no verification)',
            gated_gain_verified='publish the gated candidate iff it uses private votes and gain lower > 0, else b',
            gated_class_verified='additionally every class recall lower > 0, else b'),
        primary=dict(epsilon=1.,draws='50-99',cases='cases whose gated candidate uses private votes',
            C1='MNIST: gain-verified acceptance rate >= 50%',C2='Usenet: gain-verified acceptance rate <= 10%',
            success='C1 and C2',informativeness='expected Usenet gain band radius at epsilon=1 is recorded below (must be < .10 for the test to be informative)'),
        predictions=['C1 holds','C2 holds','CIFAR-10k at epsilon=1: acceptance >= 50%',
            'test utility of gated_gain_verified minus LOO > 0 on MNIST and >= 0 on Usenet at epsilon=1 (verification removes the gate loss on Usenet)',
            'new cohorts reproduce the gate direction at epsilon=1: gated above LOO on MNIST, below LOO on Usenet',
            'at epsilon=.1 the gate is closed on all three tasks (v > .03), so the candidate is the LOO choice'],
        reporting=dict(utility='task metric on test (balanced accuracy for Usenet, accuracy otherwise) and both metrics for all tasks; mean over cohorts x draws',
            intervals='crossed bootstrap over cohorts x label draws, 2000 replicates, seed 20261022',
            offline_diagnostics='clean held-out gain and recalls from unreleased counts; false accepts; test labels loaded only after all decisions',
            all_cases_retained=True,no_reruns_with_new_seeds=True,
            amendments='any implementation fix made after this freeze is logged in AMENDMENTS.json with time and reason; none may depend on primary outcomes'),
        expected_noise_scales=expected_scales())

def main():
    path=ROOT/'VERIFY-TEACHER-DESIGN.json';d=json.loads(json.dumps(design()))
    for k,v in d['expected_noise_scales'].items():
        if k.endswith('/1'):assert v['gate_open'],k
        else:assert not v['gate_open'],k
    d['candidate']['gate_status_now']={k:v['gate_open'] for k,v in d['expected_noise_scales'].items()}
    assert d['expected_noise_scales']['web/1']['gain_band_radius']<.1
    if path.exists():
        assert json.loads(path.read_text())['design']==d,'frozen design differs';print('design unchanged');return
    record=dict(design=d,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))},
        selection_rule_design_sha256=hashlib.sha256((EXP/'release-eab2-20261020/SELECTION-RULE-DESIGN.json').read_bytes()).hexdigest())
    path.write_text(json.dumps(record,indent=2)+'\n');print('frozen',record['frozen_utc'])

if __name__=='__main__':main()
