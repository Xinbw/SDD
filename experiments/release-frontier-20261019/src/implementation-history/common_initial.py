"""Frozen label-budget and release-count study; preceding evidence is immutable."""
import hashlib
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'release-selector-20261019'
CLASSES={'cifar10':10,'mnist':10,'web':6,'femnist62':62}
SETTINGS=[('cifar10',1000,.1),('cifar10',10000,.1),('mnist',10000,.1),('web',3000,1.),
          ('femnist62',1000,.1),('femnist62',1000,1.),('femnist62',2000,.1),('femnist62',2000,1.)]
SEEDS=list(range(8501,8511))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)

def freeze(name,design):
    path=ROOT/name
    if path.exists():assert json.loads(path.read_text())['design']==design,(name,'frozen design differs')
    else:save(path,dict(design=design,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256={p.name:sha(p) for p in (ROOT/'src').glob('*') if p.suffix in ['.py','.cpp','.so']}))

def release_key(dataset,n,epsilon,seed,steps,method,k,estimator):
    return f'{dataset}_{n}_{epsilon:g}_{seed}_T1_s{steps}_{method}_K{k}_{estimator}'

def frozen_design():
    return dict(settings=[dict(dataset=d,n=n,epsilon=e) for d,n,e in SETTINGS],
        private_seeds=SEEDS,seed_role='paired with preceding immutable three-release cohorts, not a fresh confirmation',
        single_release_teacher_steps=[60,180],all_steps_before_release_are_private_only=True,
        delta=1e-5,accountant='same distributed discrete Gaussian; rounds=1; exact integer recalibration',
        region_estimators=['M','R'],base_resolution='K=C',
        additional_regions='preceding public-configured K/estimator; Web K=12/M; FEMNIST K=124/M',
        per_example_and_compact_head_single_release_controls=True,
        decoders=['raw','center','coverage'],coverage='K>=C; maximize weighted projected scores subject to at least one region per class',
        no_label_primary_decoder='center; coverage retained as a separate fixed postprocessing rule',
        public_label_multipliers=[1,3,5,10,20,50],zero_label_reference=True,
        random_label_draws=100,stratified_label_draws=50,stratified_tasks=['cifar10','mnist'],
        label_sampling_seed=970001,label_pool='development only; all its feature vectors are public to every arm',
        label_budget_includes_all_model_selection_labels=True,no_external_validation_labels=True,
        public_free_information=['class count','class taxonomy','frozen representations','unlabeled development/proxy features'],
        class_frequencies_not_free=True,
        public_baselines=['regional majority with nearest-named-region fallback','coverage-constrained regional naming',
            'nearest class centroid','ridge linear head with exact leave-one-out selection',
            'label spreading, ordinary and class-mass-normalized, leave-one-out selection'],
        public_graph=dict(nodes='development plus proxy; test is inductive query only',neighbors=20,
            affinity='symmetric Gaussian kNN; bandwidth median squared twentieth-neighbor distance',alpha=[.8,.95]),
        ridge_regularization=[.001,.01,.1,1.,10.],
        public_portfolio_selection='leave-one-out accuracy, balanced accuracy on Web; fixed candidate-order tie break',
        missing_region_fallback='nearest nonempty named public region; ties lowest region/class index',
        private_frontier='T1, 180 private steps, fixed K=C/M; no public teacher or public validation dataset',
        hybrid=dict(private_release='same single K=C/M matrix',
            private_decoders=['center','coverage','raw'],
            public_component='public portfolio chosen using labels within m',
            candidates=['score mixture with private hard regional labels','public-evidence regional assignment'],
            private_weights=[0.,.1,.3,.5,.7,.9,1.],
            selection='leave-one-out on the same m labels; no test or clean private counts'),
        uncertainty='fully crossed label draws and paired cohort/noise seeds; two-way bootstrap, 2000 replicates',
        safety=dict(publication_gate='LOO hybrid gain lower bound over selected public portfolio >0, plus all-class coverage',
            min_recall='LOO Wilson 95% lower bound >0 for every class; unobserved class fails',
            local_models='remain private; public proxy does not certify private local benefit',
            report=['publication frequency','unconditional gated utility','per-class test recall','ungated utility']),
        bound_diagnostic='unprojected sup shift; first-release centered/matching bound; raw has separate gap bound',
        no_private_test_routing=True,all_label_budgets_and_adverse_seeds_retained=True)

if __name__=='__main__':
    freeze('DESIGN-FROZEN.json',frozen_design());print('Frozen label-budget, decoder, gate and single-release study.')
