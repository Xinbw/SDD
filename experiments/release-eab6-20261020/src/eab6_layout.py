"""ICDE 2027 EAB (sixth round) layout audit and paper-evidence index for the IEEE manuscript.

ICDE 2027 research papers: IEEE format, at most 12 pages excluding references
and the AI-generated content acknowledgement; no appendix.
"""
import argparse,csv,hashlib,json,re
from pathlib import Path
import fitz

MAX_BODY_PAGES=12
OBJECTS=[
    ('Figure','fig:scenario','fig-scenario.tex','Data-platform decision scenario','schematic; no data','manuscript (TikZ)'),
    ('Figure','fig:pipeline','fig-pipeline.tex','Evaluated pipeline and budget accounting','schematic; no data','manuscript (TikZ)'),
    ('Table','tab:setup','tab-setup.tex','Evaluated settings, private units and splits','frozen input and design files','export_eab3.py'),
    ('Table','tab:release-interfaces','sec-threat-model.tex','Release interface dimensions and whole-client sensitivity','bounded reports','manuscript'),
    ('Table','tab:fewshot','tab-fewshot.tex','Public naming and semi-supervision under balanced labels','frontier joint records','export_value.py'),
    ('Table','tab:joint-controls','tab-joint-controls.tex','Same joint postprocessing for three protected interfaces','frontier joint controls','export_value.py'),
    ('Table','tab:private-validation','tab-private-validation.tex','Total-budget verification with unconditional public fallback','new validation and linear bands','export_value.py'),
    ('Table','tab:decomposition','tab-decomposition.tex','Ceilings, noise-free naming, protected naming, selection and the gated rule','partition ceilings; clean-count decomposition (signed, decoder-matched); exploratory selection-rule study','export_eab5.py'),
    ('Table','tab:verify-teacher','tab-verify-teacher.tex','Gated rule plus private verification on new independent cohorts (pre-registered; primary criterion C2 not met)','release-verify-teacher-20261020: 6000 cases, frozen design; balanced-accuracy rule (post hoc)','export_eab5.py'),
    ('Table','tab:cost','tab-cost.tex','Protocol versus target cost: fixed, tighter bands, per-target allocation, source-disjoint clients','threshold policies; tight bands; allocation study with acceptance and selected shares; release-disjoint-20261020; draws 50-99','export_eab6.py'),
    ('Table','tab:verify-ladder','tab-verify-ladder.tex','Verification-cost ladder of publication targets','3000 existing protected validation outputs','export_eab.py'),
    ('Figure','fig:label-frontier','figures/fig-joint-frontier.pdf','Annotation acquisition and marginal private information','frontier joint records','export_value.py'),
    ('Figure','fig:value-mechanisms','figures/fig-value-mechanisms.pdf','Pooling comparison and typical/lower-tail gains','existing protected matrices; confirmation draws','export_value.py'),
    ('Figure','fig:natural-frontier','figures/fig-natural-frontier.pdf','Natural writer geometry versus release noise','frontier and selector resolution grids','export_value.py'),
    ('Figure','fig:evidence','figures/fig-evidence.pdf','Loss decomposition, development-split audit circularity and minimum recall of published classifiers','clean-count decomposition; development audit; published-minimum-recall','export_eab3.py'),
    ('Figure','fig:planning','figures/fig-planning.pdf','Closed-form verification planning map over clients and classes','closed-form bands; decision check on 3000 validation outputs','export_eab3.py')]
ARTIFACT_ONLY=[('tab:allocation','tab-allocation.tex','Naming versus verification budget allocation (summarized in Section VI-E)','frozen allocation study','export_eab.py'),
    ('tab:bound-values','tab-bound-values.tex','Numerical scope of coupled stability diagnostics','frontier bound records','export_value.py'),
    ('tab:publication-gate','tab-publication-gate.tex','Public class evidence and abstention decisions','frontier joint records','export_value.py'),
    ('tab:resource-map','tab-resource-map.tex','Cohort phases and counted annotation access','frozen phase designs','export_value.py'),
    ('tab:validation-coverage','tab-validation-coverage.tex','Finite-heldout noise coverage diagnostics','new validation records','export_value.py'),
    ('tab:ceiling','tab-ceiling.tex','First-round partition-ceiling table (superseded in the body by tab:decomposition)','frontier joint records; partition ceilings','export_eab.py'),
    ('tab:zero-label-pool','tab-zero-label-pool.tex','Zero-label pooling of all existing point matrices','110 protected point sources','export_value.py'),
    ('tab:rounds','tab-rounds.tex','Fixed-teacher/equal-step release-count contrasts','frontier paired protocols','export_value.py'),
    ('tab:no-feedback','tab-no-feedback.tex','Time-varying private teachers and protected averaging','new no-feedback protocols','export_value.py')]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(project):
    paper=project/'paper';log=(paper/'main.log').read_text() if (paper/'main.log').exists() else None
    if log is not None:
        assert 'Overfull' not in log,'overfull box'
        assert not re.search(r'(Citation|Reference).*undefined',log),'undefined reference'
    source='\n'.join(p.read_text() for p in sorted(paper.glob('sec-*.tex'))+sorted(paper.glob('fig-*.tex')))+(paper/'main.tex').read_text()
    assert source.count('[Experiment, Analysis, and Benchmark]')>=1,'EAB title tag'
    assert '\\appendix' not in source,'ICDE forbids appendices'
    assert '\\documentclass[conference]{IEEEtran}' in source
    pdf=fitz.open(paper/'main.pdf');texts=[p.get_text() for p in pdf]
    def page(marker):
        hits=[i+1 for i,t in enumerate(texts) if marker in t];assert hits,marker;return hits[0]
    conclusion=page('CONCLUSION');ack=page('ACKNOWLEDGMENT OF AI-GENERATED CONTENT');refs=page('REFERENCES')
    assert conclusion<=ack<=refs
    # The body ends where the AI acknowledgement begins; everything before it counts.
    body_text=texts[ack-1].split('ACKNOWLEDGMENT OF AI-GENERATED CONTENT')[0].strip()
    body_end=ack if body_text else ack-1
    assert body_end<=MAX_BODY_PAGES,('body pages',body_end)
    return dict(pdf_sha256=sha(paper/'main.pdf'),venue='ICDE 2027 research track, Experiment/Analysis/Benchmark (EAB) paper, data security and privacy',format='IEEEtran conference',
        body_pages=body_end,body_page_limit=MAX_BODY_PAGES,conclusion_page=conclusion,ai_acknowledgement_page=ack,references_start_page=refs,total_pages=len(pdf),
        appendix=False,overfull_boxes=False,undefined_references=False,main_figures=source.count('\\begin{figure'),data_figures=source.count('\\includegraphics'),title_tag='[Experiment, Analysis, and Benchmark]',
        tex_source_sha256={p.name:sha(p) for p in sorted(paper.glob('*.tex'))},bibliography_sha256=sha(paper/'refs.bib'))

def evidence(project):
    aux=(project/'paper/main.aux').read_text();rows=[]
    for kind,label,file,role,source,generator in OBJECTS:
        m=re.search(r'\\newlabel\{'+re.escape(label)+r'\}\{\{([^{}]+)\}\{(\d+)\}',aux);assert m,label
        number,page=m.groups();assert int(page)<=MAX_BODY_PAGES,label
        rows.append(dict(object=kind+' '+number,latex_label=label,pdf_page=page,paper_region='main',paper_file='paper/'+file,argument_role=role,evidence=source,generator=generator))
    for label,file,role,source,generator in ARTIFACT_ONLY:
        assert (project/'paper'/file).exists(),file
        rows.append(dict(object='Table (artifact only)',latex_label=label,pdf_page='',paper_region='artifact; summarized in text',paper_file='paper/'+file,argument_role=role,evidence=source,generator=generator))
    with (project/'PAPER_EVIDENCE_INDEX.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--write',action='store_true');a=p.parse_args()
    project=a.project.resolve();report=audit(project)
    if a.write:
        evidence(project);(project/'paper/BUILD_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    else:
        build=json.loads((project/'paper/BUILD_REPORT.json').read_text());assert build['pdf_sha256']==report['pdf_sha256']
    print(json.dumps({k:v for k,v in report.items() if k!='tex_source_sha256'},indent=2))
