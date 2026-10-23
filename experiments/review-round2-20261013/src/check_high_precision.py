"""Independently verify exact integer sensitivities and composed DP conversion."""
import json
import math
from pathlib import Path
import mpmath as mp
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
mp.mp.dps=90
cache={};runs=0;smallest_slack=mp.inf

def rho(cert):
    n=cert['honest_clients'];b=cert['sigma_share_integer'];d=cert['dimension']
    sensitivity=cert['sensitivity_integer_squared']
    assert int(sensitivity)==sensitivity
    if n>1:
        # pi>3 gives a rigorous, very loose check of the retained tau bound.
        log_upper=mp.log(10*(n-1))-9*mp.mpf(b)**2
        assert log_upper<=mp.log(mp.mpf('1e-30'))
        tau=mp.mpf('1e-30')
    else:tau=mp.mpf(0)
    return mp.mpf(int(sensitivity))/(2*n*b*b)+d*tau/4

for path in (ROOT/'results').glob('*.json'):
    row=json.loads(path.read_text())
    if 'trace' not in row:continue
    delta=mp.mpf(str(row.get('delta',1e-5)))
    epsilon=mp.mpf(str(row['epsilon']))
    if row.get('model')=='full ResNet-18':
        certificates=[row['privacy']];total=rho(certificates[0])*certificates[0]['rounds'];accounting='zcdp'
    else:
        certificates=[]
        for stage in row['trace']:
            cert=stage['privacy']
            certificates.extend([cert['screen'],cert['answers']] if row['method']=='PATE-DI' else [cert])
        total=sum(rho(c) for c in certificates);accounting='rdp' if row['method']=='PATE-DI' else 'zcdp'
    key=(str(total),str(delta),accounting)
    if key not in cache:
        if accounting=='zcdp':bound=total+2*mp.sqrt(total*mp.log(1/delta))
        else:
            def f(z):
                a=1+math.exp(z)
                return float(total)*a+math.log(1/(a*float(delta)))/(a-1)+math.log1p(-1/a)
            fitted=minimize_scalar(f,bounds=(-8,18),method='bounded')
            # Use one explicit valid order, then evaluate its bound at high precision.
            a=1+mp.exp(mp.mpf(str(fitted.x)))
            bound=a*total+mp.log(1/(a*delta))/(a-1)+mp.log(1-1/a)
        cache[key]=bound
    bound=cache[key];assert bound<=epsilon,(path,str(bound),str(epsilon))
    smallest_slack=min(smallest_slack,epsilon-bound);runs+=1
report={'runs_checked':runs,'decimal_digits':90,'unique_composed_bounds':len(cache),
    'all_integer_sensitivities_exact':True,'convolution_upper_bounds_proved_using_pi_lower_bound_3':True,
    'all_recomposed_budgets_pass':True,'smallest_epsilon_slack':str(smallest_slack)}
(ROOT/'privacy/high-precision-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
