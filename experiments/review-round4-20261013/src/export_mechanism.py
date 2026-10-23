"""Correct Figure 3 labels while preserving all plotted scientific values."""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / 'review-round2-20261013'
spec = importlib.util.spec_from_file_location('preserved_curve_tools', LEGACY / 'src/export_curves.py')
curves = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curves)
curves.ROWS = [json.loads(p.read_text()) for p in sorted((curves.ROOT / 'results').glob('*.json'))]


def export(paper):
    (paper / 'figures').mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharey=True)
    data = {'panel_a': [], 'panel_b': [], 'panel_a_budgets': {'solid': .1, 'dashed': .025},
            'panel_b_budgets': {'solid': .1, 'dashed': 1.}, 'legend': 'methods only'}
    summary = json.loads((LEGACY / 'round2-summary.json').read_text())
    for method in curves.METHODS[:3]:
        for epsilon, style in [(.1, '-'), (1., '--')]:
            records = [curves.subset('cifar10', n, .1, epsilon, method,
                range(8101, 8111) if n == 1000 and epsilon == .1 else (801, 802, 803),
                'fixed-public-1k' if n == 1000 and epsilon == .1 else None)
                for n in [100, 1000, 10000]]
            values = [curves.describe(rows) for rows in records]
            axes[1].errorbar([100, 1000, 10000], [q['mean'] for q in values],
                yerr=[q['sd'] for q in values], lw=1.3, ls=style,
                marker=curves.MARKERS[method], ms=4, capsize=2, color=curves.COLORS[method])
            data['panel_b'].append({'method': method, 'epsilon': epsilon,
                'cohorts': [100, 1000, 10000], 'values': values,
                'source_records': [[r['seed'] for r in rows] for rows in records]})
    endpoint = next(q for q in summary['attribution'] if q['dataset'] == 'cifar10' and q['n'] == 1000)
    local = endpoint['DP_Head_local']
    axes[1].errorbar(1000, local['mean'], yerr=local['sd'], marker='^', ms=5, mfc='white',
        mec=curves.COLORS['DP-Head'], color=curves.COLORS['DP-Head'], capsize=2, ls='none')
    data['head_local'] = local
    axes[1].set_xscale('log')
    axes[1].set_xticks([100, 1000, 10000], ['100', '1k', '10k'])
    axes[1].set_xlabel('Actual trained and reporting clients')
    axes[1].set_title('(b) Actual participation, CIFAR-10')
    for epsilon, style in [(.025, '--'), (.1, '-')]:
        points = []
        for method in curves.METHODS[:3]:
            rows = curves.subset('cifar10', 10000, .1, epsilon, method)
            coordinates = rows[0]['trace'][0]['diagnostics']['logical_report_coordinates']
            values = curves.describe(rows)
            points.append((coordinates, values['mean']))
            axes[0].errorbar(coordinates, values['mean'], yerr=values['sd'],
                color=curves.COLORS[method], marker=curves.MARKERS[method], ms=5, capsize=2)
            data['panel_a'].append({'method': method, 'epsilon': epsilon,
                'coordinates': coordinates, **values, 'seeds': [r['seed'] for r in rows]})
        points.sort()
        axes[0].plot([q[0] for q in points], [q[1] for q in points],
                     color='#aaa', ls=style, lw=.8, zorder=0)
    axes[0].set_xscale('log')
    axes[0].set_xticks([100, 1280, 20000], ['100', '1280', '20k'])
    axes[0].set_xlabel('Integer report coordinates per client')
    axes[0].set_ylabel('Mean client accuracy')
    axes[0].set_title('(a) Accuracy and report size')
    endpoint = next(q for q in summary['attribution'] if q['dataset'] == 'cifar10' and q['n'] == 10000)
    public = endpoint['Public_only_selected']['mean']
    data['public_only_reference'] = public
    axes[0].axhline(public, color='#777', lw=.9, ls=':')
    axes[0].text(.03, .995, f'Public-only: {100 * public:.1f}% (0 private coordinates)',
        transform=axes[0].transAxes, fontsize=6.8, va='top',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1, 'alpha': .85})
    axes[0].text(.03, .12, r'$\varepsilon=0.1$: solid; $0.025$: dashed',
                 transform=axes[0].transAxes, fontsize=7, va='top')
    axes[1].text(.03, .12, r'$\varepsilon=0.1$: solid; $1$: dashed',
                 transform=axes[1].transAxes, fontsize=7, va='top')
    handles = [Line2D([], [], color=curves.COLORS[m], marker=curves.MARKERS[m],
                      linestyle='none', markersize=5, label='SDD-M' if m == 'SDD' else m)
               for m in curves.METHODS[:3]]
    handles.append(Line2D([], [], color=curves.COLORS['DP-Head'], marker='^',
        markerfacecolor='white', linestyle='none', markersize=5, label=r'Head-local (b, $\varepsilon=0.1$)'))
    fig.legend(handles=handles, ncol=4, loc='lower center', fontsize=7, frameon=False)
    for ax in axes:
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=.16)
    fig.tight_layout(rect=(0, .18, 1, 1))
    fig.savefig(paper / 'figures/fig-mechanism.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(paper / 'figures/fig-mechanism.png', dpi=180)
    plt.close(fig)
    (paper / 'figures/fig-mechanism-data.json').write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', type=Path, required=True)
    export(parser.parse_args().paper)
