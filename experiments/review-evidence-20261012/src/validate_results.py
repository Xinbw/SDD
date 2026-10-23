"""Validate the complete scientific record and integer release certificates."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
expected={r['id'] for r in json.loads((ROOT/'schedule.json').read_text())['jobs']}
for name in ['WEB-CONFIRMATION-SCHEDULE.json','RANDOM-REPRESENTATIVE-SCHEDULE.json']:
    if (ROOT/name).exists():expected.update(r['id'] for r in json.loads((ROOT/name).read_text())['jobs'])
files={p.stem:p for p in (ROOT/'results').glob('*.json')}
missing=sorted(expected-files.keys());assert not missing,missing
stages=0;max_epsilon_excess=-math.inf;largest_tau=0.;worst_log_wrap=-math.inf;guard_max=0.

def certificate(cert,epsilon):
    global stages,max_epsilon_excess,largest_tau,worst_log_wrap
    n=cert['actual_clients'];h=cert['honest_clients'];b=cert['sigma_share_integer'];d=cert['dimension']
    assert n==cert['calibrated_clients'] and h==n-cert['colluders'] and 1<=h<=n
    assert b>=1 and b==int(b)
    tau=cert['tau_certified_upper'];assert tau<=1e-30
    if h>1:assert cert['log_tau_upper']<=math.log(tau)
    else:assert tau==0.
    rho=cert['sensitivity_integer_squared']/(2*h*b*b)+d*tau/4
    assert math.isclose(rho,cert['rho_round_upper'],rel_tol=1e-12)
    assert math.isclose(cert['rho_total_upper'],rho*cert['rounds'],rel_tol=1e-12)
    assert cert['epsilon_upper']<=epsilon
    max_epsilon_excess=max(max_epsilon_excess,cert['epsilon_upper']-epsilon)
    largest_tau=max(largest_tau,tau);stages+=1
    # Conservative per-coordinate integer signal bound; valid for every interface.
    signal_bound=n*math.ceil(math.sqrt(cert['sensitivity_integer_squared']))
    a=2**63-signal_bound;assert a>0
    log_wrap=math.log(2*d*cert['rounds'])-a*a/(2*n*b*b)
    worst_log_wrap=max(worst_log_wrap,log_wrap)

for key,path in files.items():
    row=json.loads(path.read_text());assert row['actual_clients']==row['calibrated_clients']
    assert 0<=row['accuracy']<=1
    if row.get('model')=='full ResNet-18':certificate(row['privacy'],row['epsilon']);continue
    assert row['population']['actual_clients']==row['actual_clients']
    assert row['population']['unique_client_identifiers']==row['actual_clients']
    assert row['evaluation_split']=='test' and row['public_development'] is False
    assert len(row['trace'])==3 and row['optimizer_steps_per_round']==60
    for stage in row['trace']:
        cert=stage['privacy']
        if row['method']=='PATE-DI':
            certificate(cert['screen'],row['epsilon']);certificate(cert['answers'],row['epsilon'])
            assert cert['epsilon_upper']<=row['epsilon']
        else:
            certificate(cert,row['epsilon'])
            if row['method']=='DP-Head':
                radius=cert['integer_report_radius'];assert radius**2<2**53
                assert cert['sensitivity_integer_squared']==(2*radius)**2
                guard_max=max(guard_max,stage['diagnostics']['integer_ball_guard_fraction'])
            elif row['method']!='DP-FedCT':
                assert cert['sensitivity_integer_squared']==2*row['k']*(1<<16)**2
    if row['dataset']=='web':
        assert row['population']['distinct_author_hash'] is not None

frozen=json.loads((ROOT/'public-selection/FROZEN.json').read_text())
for name,digest in frozen['files'].items():
    assert hashlib.sha256((ROOT/'public-selection'/name).read_bytes()).hexdigest()==digest
    entry=json.loads((ROOT/'public-selection'/name).read_text())
    assert len(entry['budgets'])==7 and entry['configuration_frozen'] and entry['test_access'] is False
start=json.loads((ROOT/'FINAL-TEST-START.json').read_text())
assert start['schedule_sha256']==hashlib.sha256((ROOT/'schedule.json').read_bytes()).hexdigest()
assert start['public_freeze_sha256']==hashlib.sha256((ROOT/'public-selection/FROZEN.json').read_bytes()).hexdigest()
report={'matched_population_jobs_expected':len(expected),'matched_population_jobs_verified':len(expected),
    'legacy_runs':sum(p.stem.startswith('resnet_') for p in files.values()),'privacy_stage_certificates_verified':stages,
    'maximum_epsilon_excess':max_epsilon_excess,'largest_tau_upper':largest_tau,
    'worst_log_union_wrap_probability':worst_log_wrap,'integer_parameter_guard_max_fraction':guard_max,
    'configuration_hashes_verified':len(frozen['files']),'all_budget_bounds_pass':True,
    'source_feature_bundle_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'inputs').glob('*.npz')
        if 'initial' not in p.name},'final_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}}
(ROOT/'scientific-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
