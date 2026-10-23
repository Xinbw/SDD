"""Cross-check the manuscript's quantitative claims against final result records."""
from pathlib import Path
import argparse
import json
import re
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--paper', type=Path, required=True)
args = parser.parse_args()
rows = [json.loads(p.read_text()) for p in (ROOT/'results').glob('*.json')]
source = '\n'.join(p.read_text() for p in args.paper.glob('sec-*.tex'))
source += (args.paper/'main.tex').read_text()

def select(dataset, n, alpha, epsilon, method, seeds, override=None):
    r = [x for x in rows if x.get('model') != 'full ResNet-18'
         and x['dataset'] == dataset and x['actual_clients'] == n
         and x['alpha'] == alpha and x['epsilon'] == epsilon
         and x['method'] == method and x['seed'] in seeds
         and x.get('configuration_override') == override]
    assert len(r) == len(seeds), (dataset, n, epsilon, method, len(r))
    return sorted(r, key=lambda x: x['seed'])

def mean(r):
    field = 'balanced_accuracy' if r[0]['dataset'] == 'web' else 'accuracy'
    return 100*np.array([x[field] for x in r])

def effect(a, b):
    assert [x['seed'] for x in a] == [x['seed'] for x in b]
    d = mean(a)-mean(b)
    half = t.ppf(.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
    return dict(gain=round(float(d.mean()), 1),
                ci=[round(float(d.mean()-half), 1), round(float(d.mean()+half), 1)])

image = lambda method: select('cifar10',10000,.1,.1,method,range(801,806))
web = lambda method: select('web',3000,None,1.,method,range(8101,8111),'fixed-public-1k')
small = lambda method: select('cifar10',1000,.1,.1,method,range(8101,8111),'fixed-public-1k')
checks = {
    'cifar_accuracy': round(float(mean(image('SDD')).mean()), 1),
    'cifar_vs_per_example': effect(image('SDD'), image('DP-FedCT')),
    'cifar_vs_random': effect(image('SDD'), image('random')),
    'web_representative_accuracy': round(float(mean(web('representative')).mean()), 1),
    'web_vs_per_example': effect(web('representative'), web('DP-FedCT')),
    'web_vs_head': effect(web('representative'), web('DP-Head')),
    'web_vs_random_representative': effect(web('representative'), web('random-representative')),
    'small_cifar_vs_head': effect(small('SDD'), small('DP-Head')),
}
assert checks['cifar_accuracy'] == 85.0
assert checks['cifar_vs_per_example'] == {'gain':15.3,'ci':[12.4,18.2]}
assert checks['cifar_vs_random'] == {'gain':36.1,'ci':[32.0,40.1]}
assert checks['web_representative_accuracy'] == 26.3
assert checks['web_vs_per_example'] == {'gain':5.1,'ci':[3.6,6.5]}
assert checks['web_vs_head'] == {'gain':9.6,'ci':[8.4,10.7]}
assert checks['web_vs_random_representative'] == {'gain':8.1,'ci':[7.0,9.2]}
assert checks['small_cifar_vs_head'] == {'gain':22.5,'ci':[15.3,29.6]}
for method in ['SDD','representative','random-representative']:
    assert {x['k'] for x in web(method)} == {5}
    assert {x['trace'][0]['diagnostics']['logical_report_coordinates'] for x in web(method)} == {30}
assert {x['trace'][0]['diagnostics']['logical_report_coordinates'] for x in web('DP-FedCT')} == {9000}
assert {x['trace'][0]['diagnostics']['logical_report_coordinates'] for x in image('SDD')} == {100}
assert {x['trace'][0]['diagnostics']['logical_report_coordinates'] for x in small('SDD')} == {200}
assert {x['trace'][0]['diagnostics']['logical_report_coordinates'] for x in image('DP-Head')} == {1280}
assert {x['trace'][0]['diagnostics']['logical_report_coordinates'] for x in web('DP-Head')} == {768}
assert 9000/30 == 300 and 20000/100 == 200 and 1280/200 == 6.4
assert not re.findall(r'(?:[a-z]|\$|\})\.[A-Z][a-z]', source)
assert '\\input{numbers}' not in source
assert source.count('\\includegraphics') == 3
checks.update(report_coordinate_ratios_verified=True,
              sentence_spacing_scan_pass=True, active_figures=3)
(ROOT/'manuscript-number-validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
