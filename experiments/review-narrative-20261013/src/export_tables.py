"""Regenerate existing tables and publish the fixed no-task-label core comparison."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
spec = importlib.util.spec_from_file_location('preserved_fourth_round_tables',
    ROOT.parent / 'review-round4-20261013/src/export_tables.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def describe(rows):
    values = np.array([r['accuracy'] for r in rows])
    return {'mean': float(values.mean()), 'sd': float(values.std(ddof=1))}


def export(paper):
    previous.export(paper)
    def rows(path):
        return [line[:-3].split(' & ') for line in path.read_text().splitlines()
                if ' & ' in line and line.endswith(r' \\')]
    sources = rows(paper / 'tab-round2-attribution.tex')
    heads = rows(paper / 'tab-round2-head-local.tex')
    assert len(sources) == len(heads) == 5
    resources = []
    for source, head in zip(sources, heads):
        assert source[:2] == head[:2]
        resources.append(' & '.join(source + head[2:]) + r' \\')
    resource_caption = (
        r'Public-configured supervision-source and parameter-endpoint references, percent mean $\pm$ sample SD '
        r'(Web: balanced accuracy). Images use $\alpha=0.1$, $\varepsilon=0.1$, five seeds '
        r'at 10k and ten elsewhere; Web uses ten. M/R denote image/Web SDD. All share '
        r'public features. Pub heads use all 5000 image/1500 Web development labels, '
        r'learning rate 0.3 and zero private report coordinates; Local+Pub uses their '
        r'proxy targets. Client methods use 180 steps. Head-local precedes the third '
        r'aggregation, retaining two broadcasts with all three releases charged.')
    previous.export_revision.table(paper / 'tab-narrative-resources.tex', resource_caption,
        'tab:attribution',
        r'\multicolumn{1}{c}{} & \multicolumn{5}{c}{supervision sources} & '
        r'\multicolumn{2}{c}{parameter endpoints} \\ \cmidrule(lr){2-6}\cmidrule(lr){7-8}'
        '\n' + r'task / variant, cohort & SDD & Local & Local+Pub & Pub (180) & Pub (600) & Head-global & Head-local',
        resources, 'lrrrrrrr')
    path = paper / 'tab-narrative-resources.tex'
    path.write_text(path.read_text().replace(r'\label{tab:attribution}',
        r'\label{tab:attribution}\label{tab:head-local}'))
    records = [json.loads(p.read_text()) for p in
               (ROOT.parent / 'review-round2-20261013/results').glob('*.json')]
    summary, body = [], []
    for dataset, label in [('cifar10', 'CIFAR-10'), ('mnist', 'MNIST')]:
        groups = {}
        for method in ['Local-only', 'DP-FedCT', 'SDD']:
            rows = [r for r in records if r['dataset'] == dataset and r['method'] == method
                    and r.get('actual_clients') == 10000 and r.get('alpha') == .1
                    and r['seed'] in range(801, 806)
                    and (method == 'Local-only' or 'configuration_rule' in r)]
            rows.sort(key=lambda r: r['seed'])
            assert len(rows) == 5
            groups[method] = rows
        paired = np.array([a['accuracy'] - b['accuracy']
                           for a, b in zip(groups['SDD'], groups['DP-FedCT'])])
        half = float(t.ppf(.975, 4) * paired.std(ddof=1) / np.sqrt(5))
        values = {method: describe(rows) for method, rows in groups.items()}
        cell = lambda q: f"{100 * q['mean']:.1f}\\,$\\pm$\\,{100 * q['sd']:.1f}"
        body.append(label + ' & ' + cell(values['Local-only']) + ' & ' + cell(values['DP-FedCT'])
                    + r' & \textbf{' + cell(values['SDD']) + '} & '
                    + f"{100 * paired.mean():.1f} & [{100 * (paired.mean() - half):.1f},{100 * (paired.mean() + half):.1f}]"
                    + r' \\')
        summary.append({'dataset': dataset, 'actual_clients': 10000, 'alpha': .1,
                        'epsilon': .1, 'delta': 1e-5, 'k': 10, 'mode': 'center',
                        'public_task_labels_for_training_or_configuration': 0,
                        'seeds': list(range(801, 806)), 'methods': values,
                        'sdd_gain_vs_per_example': float(paired.mean()),
                        'paired_ci_low': float(paired.mean() - half),
                        'paired_ci_high': float(paired.mean() + half),
                        'report_coordinates': {'Local-only': 0, 'SDD': 100, 'DP-FedCT': 20000}})
    caption = (r'Client-sourced supervision without public task labels, actual $10^4$ clients, '
               r'$\alpha=0.1$, $\varepsilon=0.1$, $\delta=10^{-5}$, five seeds. '
               r'Fixed $K=C=10$ and centered consensus; 180 client steps. Accuracy is percent '
               r'mean $\pm$ sample SD. SDD uses 100 report coordinates per client per round, '
               r'DP-FedCT 20000, Local-only zero. Gain and paired $95\%$ interval are '
               r'percentage points relative to DP-FedCT.')
    previous.export_revision.table(paper / 'tab-narrative-core.tex', caption,
        'tab:core-supervision', r'task & Local-only & DP-FedCT & SDD-M & gain & paired $95\%$ interval',
        body, 'lrrrrr')
    (paper / 'core-supervision-data.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', type=Path, required=True)
    export(parser.parse_args().paper)
