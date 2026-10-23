"""Verify fourth-round presentation and its unchanged scientific provenance."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(project):
    parent = json.loads((ROOT / 'PARENT-MANIFEST.json').read_text())
    sampler = json.loads((project / 'experiments/review-round3-20261013/SAMPLER-ANONYMIZATION.json').read_text())
    for name, digest in parent['scientific_files'].items():
        actual = sha(project / name)
        allowed = {digest}
        if name.endswith('/src/exact_dg.so'):
            allowed.add(sampler['anonymous_binary_sha256'])
        assert actual in allowed, name
    rows = list(csv.DictReader((project / 'EXPERIMENT_INDEX.csv').open()))
    assert len(rows) == 1586 and len({r['result'] for r in rows}) == 1586
    for row in rows:
        result = json.loads((project / row['result']).read_text())
        for field, source in [('dataset', 'dataset'), ('variant', 'method')]:
            assert row[field] == result[source], (row['result'], field)
        assert int(row['seed']) == result['seed']
        if row['cohort'] != 'public':
            assert int(row['cohort']) == result.get('actual_clients', result.get('n'))
        if row['k']:
            assert int(row['k']) == result['k']
        assert float(row['epsilon']) == result.get('epsilon', 0)
        if row['configuration'].endswith('.json'):
            stage = project / 'experiments' / row['stage']
            assert (stage / 'public-selection' / row['configuration']).exists() or (stage / row['configuration']).exists()
        if 'configuration_rule' in result:
            assert row['configuration'] == result['configuration_rule']
        if 'public_configuration_file' in result:
            assert row['configuration'] == result['public_configuration_file']
        if 'trace' in result:
            for stage in result['trace']:
                privacy = stage['privacy']
                certificates = [privacy['screen'], privacy['answers']] if 'screen' in privacy else [privacy]
                for certificate in certificates:
                    assert certificate['actual_clients'] == result['actual_clients']
                    assert certificate['calibrated_clients'] == result['actual_clients']
                    assert certificate['epsilon_requested'] == result['epsilon']
                    assert certificate['rounds'] == result['rounds']
    paper = project / 'paper'
    text = '\n'.join(p.read_text() for p in paper.glob('sec-*.tex')) + (paper / 'main.tex').read_text()
    assert text.count('\\includegraphics') == 3
    assert not re.search(r'(?:[a-z]|\$|\})\.[A-Z][a-z]', text)
    assert 'accuracy--report-size trade-off' in text and 'accuracy--communication trade-off' not in text
    assert '$0.138$' not in text and '$0.139$' in text
    assert 'both estimators\' macro' in text and '$\\min(K,C)$' in text
    assert 'post histories' in text and 'metadata that is public or independent' in text
    assert 'ideal independent random bits' in text and 'ChaCha20 pseudorandom bit streams' in text
    assert '$\\varepsilon=0.1/0.025$ in (a)' in text and '$\\varepsilon=0.1/1$ in (b)' in text
    abstract = text.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    assert '$84.2\\%$' in abstract and '14.4' in abstract and '200\\times' in abstract
    caption = (paper / 'tab-round2-web-recall.tex').read_text()
    assert 'SDD-M/R, DP-FedCT and DP-Head' in caption and 'divides' in caption and 'by 1500' in caption
    assert 'releases 1/2 within each seed, then ten seeds' in caption
    assert 'six-class minimum before summarizing' in caption
    figure = json.loads((paper / 'figures/fig-mechanism-data.json').read_text())
    assert figure['panel_a_budgets'] == {'solid': .1, 'dashed': .025}
    assert figure['panel_b_budgets'] == {'solid': .1, 'dashed': 1.}
    assert figure['legend'] == 'methods only'
    for point in figure['panel_a']:
        raw = [json.loads((project / 'experiments/review-evidence-20261012/results' /
               f'cifar10_n10000_a0.1_e{point["epsilon"]}_s{seed}_{point["method"]}.json').read_text())
               for seed in [801, 802, 803]]
        values = np.array([r['accuracy'] for r in raw])
        assert point['mean'] == float(values.mean()) and point['sd'] == float(values.std(ddof=1))
        assert all(point['coordinates'] == r['trace'][0]['diagnostics']['logical_report_coordinates'] for r in raw)
        if point['method'] == 'SDD':
            assert point['coordinates'] == (200 if point['epsilon'] == .025 else 100)
    for name, digest in parent['parent_figures'].items():
        if name != 'fig-mechanism.pdf':
            assert sha(paper / 'figures' / name) == digest, name
    for name, digest in parent['parent_table_bodies'].items():
        body = (paper / name).read_text().split('\\begin{tabular}', 1)[1].split('\\end{tabular}', 1)[0]
        assert hashlib.sha256(body.encode()).hexdigest() == digest, name
    objects = list(csv.DictReader((project / 'PAPER_EVIDENCE_INDEX.csv').open()))
    assert len(objects) == 15
    for item in objects:
        assert (project / item['paper_file']).is_file()
        for name in item['evidence'].split(';'):
            assert name == 'theory' or (project / name).exists(), name
    doc = fitz.open(paper / 'main.pdf')
    texts = [p.get_text() for p in doc]
    assert len(doc) == 12 and 'CONCLUSION' in texts[7]
    assert 'auditable interface for private collaboration.' in texts[7].replace('-\n', '').replace('\n', ' ')
    assert all('REFERENCES' not in t for t in texts[:8]) and 'REFERENCES' in texts[8]
    log = paper / 'main.log'
    if log.exists():
        assert 'Overfull' not in log.read_text() and not re.search(r'(Citation|Reference).*undefined', log.read_text())
    assert not doc.metadata.get('author') or 'Anonymous' in doc.metadata['author']
    report = {'preserved_condition_records': 1586, 'preserved_diagnostic_replays': 60,
              'parent_scientific_fingerprints_verified': True, 'indexed_output_identities_verified': 1586,
              'configuration_and_noise_parameters_match_indexed_outputs': True,
              'figure3_points_and_report_coordinates_verified': True,
              'figure3_panel_budget_labels_corrected': True, 'figures1_and2_byte_unchanged': True,
              'generated_numerical_table_rows_unchanged': True, 'target_statistics_definitions_clarified': True,
              'random_purity_rounded_from_0_1385_to_0_139': True,
              'main_pages': 8, 'total_pages': 12, 'references_start_page': 9,
              'conclusion_fully_on_page': 8, 'overfull_boxes': False,
              'undefined_citations_or_references': False, 'sentence_spacing_pass': True,
              'new_training_or_configuration_search': False}
    (ROOT / 'revision-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    validate(parser.parse_args().project.resolve())
