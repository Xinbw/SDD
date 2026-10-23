"""Regenerate current tables from unchanged records and clarify target statistics."""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent / 'review-round3-20261013'
sys.path.insert(0, str(PREVIOUS / 'src'))
import export_revision


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

CAPTION = (
    r'Usenet family recall and protected target allocation, 3k/$\varepsilon=1$, '
    r'ten seeds, public-selected configuration ($K=5$ for SDD). '
    r'Target \% and $N_c$ apply to SDD-M/R, DP-FedCT and DP-Head, which label all '
    r'1500 proxy items. Target \% divides talk/alt--soc/misc target counts by 1500; '
    r'$N_c$ counts distinct target classes. Both average releases 1/2 within '
    r'each seed, then ten seeds. Recalls first average evaluated models within '
    r'each seed; min recall takes the six-class minimum before summarizing '
    r'seeds (percent mean $\pm$ SD). Other recall columns are seed means.'
)


def export(paper):
    paper.mkdir(parents=True, exist_ok=True)
    original = ROOT.parent / 'review-evidence-20261012'
    second = ROOT.parent / 'review-round2-20261013'
    base = load_module('original_table_export', original / 'src/export_results.py')
    base.ROWS = [json.loads(p.read_text()) for p in sorted((original / 'results').glob('*.json'))]
    base.figures = lambda _: None
    revised = load_module('second_table_export', second / 'src/export_round2.py')
    revised.OLD = base.ROWS
    revised.NEW = [json.loads(p.read_text()) for p in sorted((second / 'results').glob('*.json'))]
    # Keep exporter-generated audit files outside the immutable evidence trees.
    with tempfile.TemporaryDirectory(prefix='sdd-table-audit-') as folder:
        temporary = Path(folder)
        (temporary / 'results').symlink_to((original / 'results').resolve(), target_is_directory=True)
        base.ROOT = temporary
        revised.ROOT = temporary
        base.export(paper)
        revised.export(paper)
    export_revision.export(paper)
    path = paper / 'tab-round2-web-recall.tex'
    lines = path.read_text().splitlines()
    assert lines[2].startswith(r'\caption{')
    lines[2] = r'\caption{' + CAPTION + '}'
    path.write_text('\n'.join(lines) + '\n')
    purity = [json.loads((original / 'results' / f'cifar10_n10000_a0.1_e0.1_s{seed}_random.json').read_text())
              ['trace'][0]['diagnostics']['public_cluster_purity'] for seed in range(801, 806)]
    assert f'{sum(purity) / len(purity):.3f}' == '0.139'
    assert 'CIFAR & random & 10 & 0.139' in (paper / 'tab-new-controls.tex').read_text()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', type=Path, required=True)
    export(parser.parse_args().paper)
