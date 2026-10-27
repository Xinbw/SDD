"""Reproduce private-supervision resource and verification analysis."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'experiments/release-value-20261019'
PARENT=ROOT/'experiments/release-frontier-20261019'
ICDE=ROOT/'experiments/release-icde-20261019'
ALLOC=ROOT/'experiments/release-allocation-20261019'
EAB=ROOT/'experiments/release-eab-20261020'
EAB2=ROOT/'experiments/release-eab2-20261020'
EAB3=ROOT/'experiments/release-eab3-20261020'
VT=ROOT/'experiments/release-verify-teacher-20261020'
EAB4=ROOT/'experiments/release-eab4-20261020'
DJ=ROOT/'experiments/release-disjoint-20261020'
EAB5=ROOT/'experiments/release-eab5-20261020'
EAB6=ROOT/'experiments/release-eab6-20261020'
EAB7=ROOT/'experiments/release-eab7-20261020'
EAB8=ROOT/'experiments/release-eab8-20261021'
EAB9=ROOT/'experiments/release-eab9-20261021'
EAB10=ROOT/'experiments/release-eab10-20261022'
EAB11=ROOT/'experiments/release-eab11-20261022'
EAB12=ROOT/'experiments/release-eab12-20261022'
ENC=ROOT/'experiments/release-encoder-20261022'
FCONF=ROOT/'experiments/release-femnistconf-20261022'
EAB13=ROOT/'experiments/release-eab13-20261023'
EAB14=ROOT/'experiments/release-eab14-20261023'
EAB15=ROOT/'experiments/release-eab15-20261027'
BC=ROOT/'experiments/release-budgetconf-20261021'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def same_json(a,b,tol=1e-12):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_json(a[k],b[k],tol) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same_json(x,y,tol) for x,y in zip(a,b))
    if isinstance(a,bool) or isinstance(b,bool) or not isinstance(a,(int,float)) or not isinstance(b,(int,float)):return a==b
    return abs(a-b)<=tol*max(1.,abs(a),abs(b))
def same_record(path,before):
    """A re-derived analysis must equal its frozen record: byte-identical, or (some frozen scripts sum over
    unsorted directory listings, so a fresh clone may reorder floating-point sums) identical JSON with
    every number within 1e-12 relative. The frozen bytes are restored either way."""
    after=path.read_bytes()
    try:
        if after!=before:
            assert same_json(json.loads(before),json.loads(after)),path.name
            print('re-derived',path.name,'equal within 1e-12 (summation order)',flush=True)
    finally:path.write_bytes(before)
def join_split_files():
    """The public repository stores files above GitHub's size limit as <name>.part00, .part01, ...;
    join them once (RECORD_MANIFEST then checks the joined file's SHA-256)."""
    for first in sorted(ROOT.rglob('*.part00')):
        target=first.with_suffix('')
        if target.exists():continue
        parts=sorted(first.parent.glob(target.name+'.part[0-9][0-9]'))
        with target.open('wb') as out:
            for part in parts:out.write(part.read_bytes())
        print('joined',target.relative_to(ROOT),'from',len(parts),'parts')
def prepare():
    join_split_files()
    subprocess.run([sys.executable,str(ROOT/'experiments/review-round2-20261013/src/prepare_replay.py')],check=True)
    for name in ['review-round2-20261013','release-granularity-20261019','release-selector-20261019','release-frontier-20261019','release-value-20261019']:
        stage=ROOT/'experiments'/name;(stage/'inputs/partitions').mkdir(parents=True,exist_ok=True);(stage/'checkpoints').mkdir(exist_ok=True)
        for dataset in ['cifar10','mnist','web']:
            target=stage/'inputs'/(dataset+'.npz')
            if not target.exists():target.symlink_to('../../review-evidence-20261012/inputs/'+dataset+'.npz')
        if name in ['release-selector-20261019','release-frontier-20261019','release-value-20261019']:
            target=stage/'inputs/femnist62.npz'
            if not target.exists():target.symlink_to('../../release-granularity-20261019/inputs/femnist62.npz')
        if name!='release-value-20261019':
            target=stage/'src/vendor'
            if not target.exists():target.symlink_to('../../review-evidence-20261012/src/vendor',target_is_directory=True)
def run(path,*args,log=None):
    command=[sys.executable,str(path),*map(str,args)]
    if log:
        with log.open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    else:subprocess.run(command,check=True)
def fingerprints():
    manifest=json.loads((ROOT/'RECORD_MANIFEST.json').read_text())
    for group in ['scientific_files','presentation_files']:
        for name,digest in manifest[group].items():assert sha(ROOT/name)==digest,name
    print('PASS: fingerprints of',sum(len(manifest[g]) for g in ['scientific_files','presentation_files']),'record and presentation files')
def verify(quick=False):
    prepare();fingerprints()
    if not quick:
        run(PARENT/'src/validate_frontier.py','--project',ROOT,'--no-layout')
        # Scientific audit is unchanged; the WWW eight-page layout check is replaced
        # by the ICDE 2027 audit (IEEE format, 12 body pages, no appendix).
        run(STAGE/'src/validate_value.py','--project',ROOT,'--no-layout')
    # Eighth round: budget-split confirmation (certificates, decisions, frozen selection, analysis).
    (ROOT/'reproduction-output').mkdir(exist_ok=True)
    run(EAB8/'src/check_budgetconf.py')
    path=BC/'analysis/budgetconf-analysis.json';before=path.read_bytes();run(BC/'src/analyze.py',log=ROOT/'reproduction-output/budgetconf-analysis.log');same_record(path,before)
    # EAB revision: deterministic re-derivation of the new analyses, then layout.
    (ROOT/'reproduction-output').mkdir(exist_ok=True)
    for path in [EAB/'analysis/threshold-policies.json',ALLOC/'results/allocation-analysis.json']:
        before=path.read_bytes();run(EAB/'src/thresholds.py' if 'threshold' in path.name else ALLOC/'src/analyze.py',log=ROOT/'reproduction-output'/(path.stem+'.log'))
        same_record(path,before)
    # Second EAB round: frozen selection-rule study and published-minimum-recall diagnostic.
    for path,script in [(EAB2/'analysis/selection-rules.json',EAB2/'src/selection.py'),(EAB2/'analysis/published-minimum-recall.json',EAB2/'src/minrecall.py')]:
        before=path.read_bytes();run(script,log=ROOT/'reproduction-output'/(path.stem+'.log'));same_record(path,before)
    # Third EAB round: closed-form verification planning map (CPU, deterministic).
    path=EAB3/'analysis/verification-planning.json';before=path.read_bytes();run(EAB3/'src/planning.py',log=ROOT/'reproduction-output/verification-planning.log');same_record(path,before)
    # Fourth round: pre-registered verify-teacher analysis and its post-hoc split (CPU).
    for path,script in [(VT/'analysis/verify-teacher-analysis.json',VT/'src/analyze.py'),(VT/'analysis/post-hoc-decision-split.json',VT/'src/post_hoc.py')]:
        before=path.read_bytes();run(script,log=ROOT/'reproduction-output'/(path.stem+'.log'));same_record(path,before)
    # Fifth round: tighter bands, balanced-accuracy control, planning out-of-sample check, source-disjoint ladder.
    for path,script in [(EAB5/'analysis/tight-bands.json',EAB5/'src/tight_bands.py'),(EAB5/'analysis/balanced-verification.json',EAB5/'src/balanced.py'),
                        (EAB5/'analysis/planning-out-of-sample.json',EAB5/'src/planning_oos.py'),(DJ/'analysis/disjoint-analysis.json',DJ/'src/analyze.py')]:
        before=path.read_bytes();run(script,log=ROOT/'reproduction-output'/(path.stem+'.log'));same_record(path,before)
    # Sixth round: single-subset cost table with acceptance and splits, 24-cell calculator check, disjoint heterogeneity.
    for path,script in [(EAB6/'analysis/cost-table.json',EAB6/'src/cost_table.py'),(EAB6/'analysis/calculator-24.json',EAB6/'src/calc24.py'),(EAB6/'analysis/disjoint-heterogeneity.json',EAB6/'src/hetero.py')]:
        before=path.read_bytes();run(script,log=ROOT/'reproduction-output'/(path.stem+'.log'));same_record(path,before)
    # Twelfth round: weaker-encoder ablation, FEMNIST replication and the corrected-constant certificate check.
    for path,script in [(ENC/'analysis/encoder-analysis.json',ENC/'src/analyze.py'),(FCONF/'analysis/femnistconf-analysis.json',FCONF/'src/analyze.py'),(EAB12/'analysis/tau-recheck.json',EAB12/'src/recheck_tau.py'),(EAB13/'analysis/eab13-ledger.json',EAB13/'src/ledger.py')]:
        before=path.read_bytes();run(script,log=ROOT/'reproduction-output'/(path.stem+'.log'));same_record(path,before)
    run(EAB15/'src/eab15_layout.py','--project',ROOT)
    out=ROOT/'reproduction-output/value-regeneration';out.mkdir(parents=True,exist_ok=True)
    run(STAGE/'src/export_value.py','--paper',out,log=out/'regeneration.log')
    run(EAB/'src/export_eab.py','--paper',out)
    # export_eab2.py is superseded in the manuscript by export_eab3.py (development-split audit, setup table, planning map).
    run(EAB3/'src/export_eab3.py','--paper',out,log=out/'eab3-regeneration.log')
    run(EAB4/'src/export_eab4.py','--paper',out,log=out/'eab4-regeneration.log')
    run(EAB5/'src/export_eab5.py','--paper',out,log=out/'eab5-regeneration.log')
    run(EAB6/'src/export_eab6.py','--paper',out,log=out/'eab6-regeneration.log')
    run(EAB7/'src/export_eab7.py','--paper',out)
    run(EAB8/'src/export_eab8.py','--paper',out,log=out/'eab8-regeneration.log')
    # Eleventh round: IEEE table style (short titles; definitions in the text) derived from the generated tables.
    run(EAB11/'src/export_eab11.py','--paper',out,log=out/'eab11-regeneration.log')
    run(EAB12/'src/export_eab12.py','--paper',out,log=out/'eab12-regeneration.log')
    run(EAB13/'src/export_eab13.py','--paper',out,log=out/'eab13-regeneration.log')
    # Fifteenth round: the same figures with legends above the plots and subfigure labels below the panels.
    run(EAB15/'src/export_eab15.py','--paper',out,log=out/'eab15-regeneration.log')
    for name in ['tab-ceiling.tex','tab-verify-ladder.tex','tab-allocation.tex','eab-values.tex','eab-data.json','tab-decomposition.tex','tab-setup.tex','eab3-values.tex','eab3-data.json','tab-verify-teacher.tex','eab4-values.tex','eab4-data.json','tab-cost.tex','eab5-values.tex','eab5-data.json','eab6-values.tex','eab6-data.json','tab-budgetconf.tex','eab8-values.tex','eab8-data.json','eab12-values.tex','eab12-data.json','eab13-values.tex','eab13-data.json']:
        assert (out/name).read_bytes()==(ROOT/'paper'/name).read_bytes(),name
    for path in [*out.glob('tab-*.tex'),*out.glob('ieee-tab-*.tex')]:assert path.read_bytes()==(ROOT/'paper'/path.name).read_bytes(),path.name
    for name in ['frontier-values.tex','frontier-data.json','figures/frontier-figure-data.json','joint-values.tex','joint-data.json','figures/joint-figure-data.json','value-values.tex','value-data.json','figures/value-figure-data.json']:
        assert (out/name).read_bytes()==(ROOT/'paper'/name).read_bytes(),name
    import fitz
    def text(path):
        with fitz.open(path) as doc:return ' '.join(doc[0].get_text().split())
    for name in ['fig-joint-frontier.pdf','fig-value-mechanisms.pdf','fig-natural-frontier.pdf','fig-evidence.pdf','fig-planning.pdf','fig-ceiling.pdf','fig-verification.pdf','fig-value-mechanisms-print.pdf','fig-ceiling-print.pdf','fig-verification-print.pdf','fig-joint-frontier-ieee.pdf','fig-value-mechanisms-ieee.pdf','fig-ceiling-ieee.pdf','fig-verification-ieee.pdf']:
        assert text(out/'figures'/name)==text(ROOT/'paper/figures'/name),name
    print('PASS (quick: inherited validators skipped):' if quick else 'PASS:','preserved 3426 primary records; 3000 composed private-verification cases; 110 existing zero-label pooled readouts; exact reports, confidence bands and decisions; regenerated manuscript; partition ceilings, verification-cost ladder, allocation analysis, exploratory selection-rule study, published-minimum-recall and closed-form planning map and the pre-registered verify-teacher (6000 cases) and source-disjoint (3000 cases) analyses, tighter bands, balanced-accuracy control, planning out-of-sample check, single-subset cost table, 24-cell calculator check, disjoint heterogeneity and the budget-split confirmation (54000 cases), the weaker-encoder ablation, the FEMNIST replication and the corrected-constant check of every certificate re-derived; ICDE 2027 EAB layout with 12 body pages and no appendix.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['quick','verify','smoke','private-validation','no-feedback','export','compile']);a=p.parse_args()
    if a.mode in ['verify','quick']:verify(a.mode=='quick')
    elif a.mode=='compile':
        binary=shutil.which('tectonic');assert binary
        subprocess.run([binary,'--keep-logs','--keep-intermediates','main.tex'],cwd=ROOT/'paper',check=True)
    else:
        prepare()
        if a.mode=='smoke':
            run(STAGE/'src/replay_smoke.py');run(ALLOC/'src/reconstruct_check.py')
            outputs=[EAB2/'analysis/naming-decomposition.json',EAB2/'analysis/audit-circularity.json'];before=[sha(p) for p in outputs]
            run(EAB2/'src/decomposition.py');assert [sha(p) for p in outputs]==before,'clean-count decomposition changed'
            run(EAB4/'src/replay_vt.py');run(EAB5/'src/replay_disjoint.py');run(EAB8/'src/replay_budgetconf.py')
            path=EAB3/'analysis/audit-circularity-dev.json';before=path.read_bytes();run(EAB3/'src/audit_dev.py');same_record(path,before)
        elif a.mode=='no-feedback':run(STAGE/'src/no_feedback.py')
        elif a.mode=='private-validation':run(STAGE/'src/private_validation.py');run(STAGE/'src/linear_bands.py')
        else:run(STAGE/'src/export_value.py','--paper',ROOT/'paper')
