"""Index protected releases, counted label sets and all crossed decisions."""
import argparse
import csv
import json
import re
from pathlib import Path
from common import ROOT

def write(path,rows,fields=None):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)

def build(project):
    stage=project/'experiments'/ROOT.name
    existing=list(csv.DictReader((project/'EXPERIMENT_INDEX.csv').open()));fields=list(existing[0])
    rows=[r for r in existing if r['stage']!=ROOT.name];assert len(rows)==2336
    for path in sorted((stage/'results').glob('*.json')):
        r=json.loads(path.read_text());rows.append(dict(stage=ROOT.name,role='single-release, private-only pretraining',dataset=r['dataset'],
            cohort=r['n'],epsilon=r['epsilon'],variant=r['method'],seed=r['seed'],k=r['k'],
            configuration='FIXED-HEAD-DESIGN.json' if r['method']=='head-fixed' else 'PRIVATE-SCHEDULE.json',result=str(path.relative_to(project))))
    for path in sorted((stage/'confirmation/results').glob('*.json')):
        r=json.loads(path.read_text());rows.append(dict(stage=ROOT.name,role='independent joint-naming confirmation',dataset=r['dataset'],cohort=r['n'],epsilon=r['epsilon'],variant=r['method'],seed=r['seed'],k=r['k'],configuration='JOINT-CONFIRMATION-DESIGN.json',result=str(path.relative_to(project))))
    assert len(rows)==3426
    write(project/'EXPERIMENT_INDEX.csv',rows,fields)
    labels=[];pairs=[];controls=[]
    for phase,folder,controlfolder in [('development','joint-naming','joint-controls'),('confirmation','joint-confirmation','joint-controls-confirmation')]:
        for path in sorted((stage/'analysis'/folder).glob('*.json')):
            r=json.loads(path.read_text());result=str(path.relative_to(project))
            labels.append(dict(phase=phase,dataset=r['dataset'],acquisition=r['scheme'],draw=r['draw'],m=r['m'],labels_per_class_average=r['q'],missing_classes=r['missing_classes'],selected_public=r['selected_public'],extra_validation_labels=0,crossed_cohort_cases=len(r['hybrid']),sample_sha256=r['sample_sha256'],result=result))
            for h in r['hybrid']:
                pairs.append(dict(phase=phase,dataset=r['dataset'],acquisition=r['scheme'],draw=r['draw'],m=r['m'],cohort=h['n'],epsilon=h['epsilon'],seed=h['seed'],public_model=r['selected_public'],joint_rule=h['gate']['selected'],private_augmentation_published=h['gate']['private_augmentation_published'],public_classifier_published=h['gate']['public_classifier_published'],ungated_accuracy=h['ungated']['accuracy'],gated_accuracy=h['gated']['accuracy'],ungated_balanced_accuracy=h['ungated']['balanced_accuracy'],gated_balanced_accuracy=h['gated']['balanced_accuracy'],abstention_client_accuracy=h['abstention_client_utility']['accuracy'],result=result,protected_source='experiments/'+ROOT.name+('/confirmation' if phase=='confirmation' else '')+'/results/'+h['protected_source']+'.json'))
            cp=stage/'analysis'/controlfolder/path.name
            for h in json.loads(cp.read_text())['controls']:
                controls.append(dict(phase=phase,dataset=r['dataset'],acquisition=r['scheme'],draw=r['draw'],m=r['m'],cohort=h['n'],epsilon=h['epsilon'],seed=h['seed'],method=h['method'],selected=h['selected'],accuracy=h['metric']['accuracy'],balanced_accuracy=h['metric']['balanced_accuracy'],result=str(cp.relative_to(project)),protected_source='experiments/'+ROOT.name+('/confirmation' if phase=='confirmation' else '')+'/results/'+h['source']+'.json'))
    assert len(labels)==4224 and len(pairs)==75480 and len(controls)==150960
    write(project/'LABEL_BUDGET_INDEX.csv',labels);write(project/'LABEL_COHORT_INDEX.csv',pairs);write(project/'JOINT_RELEASE_CONTROL_INDEX.csv',controls)
    readers=[]
    for path in sorted((stage/'analysis/students').glob('*.json')):
        r=json.loads(path.read_text());readers.append(dict(dataset=r['dataset'],cohort=r['n'],epsilon=r['epsilon'],seed=r['seed'],
            private_teacher_steps=r['steps'],decoder=r['decoder'],public_steps=180,extra_private_queries=0,
            source='experiments/'+ROOT.name+'/results/'+r['source']+'.json',result=str(path.relative_to(project))))
    write(project/'SINGLE_RELEASE_READOUT_INDEX.csv',readers)
    data=json.loads((project/'paper/frontier-data.json').read_text());seedrows=[]
    for r in data['single_release_per_seed']:
        item=dict(r);item['T1_180_class_recall']=json.dumps(item['T1_180_class_recall']);seedrows.append(item)
    write(project/'SINGLE_RELEASE_SEED_RESULTS.csv',seedrows)
    objects=[
        ('Table','tab:release-interfaces','sec-method.tex','Release dimensions and exact sensitivity','theory','manuscript'),
        ('Table','tab:resource-map','tab-resource-map.tex','Label access and independent replication units','JOINT-NAMING-DESIGN.json;JOINT-CONFIRMATION-DESIGN.json','export_joint.py'),
        ('Table','tab:fewshot','tab-fewshot.tex','Public naming, linear and graph learning at 1/3/5 supplied labels per class','analysis/joint-naming/*stratified*','export_joint.py'),
        ('Table','tab:joint-controls','tab-joint-controls.tex','Equal-label comparison with identically decoded head and pooled point releases','analysis/joint*','export_joint.py'),
        ('Table','tab:publication-gate','tab-publication-gate.tex','Unconditional fallback and abstention with actual local utility','analysis/joint*','export_joint.py'),
        ('Table','tab:rounds','tab-rounds.tex','Fixed-teacher and equal-step single-release contrasts','results; preceding fixed-center records','export_frontier.py'),
        ('Table','tab:round-students','tab-round-students.tex','Same-capacity public students and tuned-head resources','analysis/students;results','export_frontier.py'),
        ('Table','tab:bound-values','tab-bound-values.tex','Actual pre-projection shift and applicable bound values','analysis/bound-diagnostics.json','export_frontier.py'),
        ('Figure','fig:label-frontier','figures/fig-joint-frontier.pdf','Equal-label independent confirmation and matched private-release controls','analysis/joint-confirmation/*;analysis/joint-naming/*stratified*','export_joint.py'),
        ('Figure','fig:release-count','figures/fig-release-count.pdf','Release count and Web class allocation','results;analysis/joint-naming/*web*','export_joint.py'),
        ('Figure','fig:natural-frontier','figures/fig-natural-frontier.pdf','Natural-writer geometry and noise resolution','results; preceding natural resolution grid','export_frontier.py')]
    aux=(project/'paper/main.aux').read_text();evidence=[]
    for kind,label,file,role,source,generator in objects:
        match=re.search(r'\\newlabel\{'+re.escape(label)+r'\}\{\{([^{}]+)\}\{(\d+)\}',aux);assert match,label
        number,page=match.groups();evidence.append(dict(object=kind+' '+number,latex_label=label,pdf_page=page,
            paper_region='main' if int(page)<=8 else 'appendix',paper_file='paper/'+file,argument_role=role,
            evidence=source,generator=generator))
    write(project/'PAPER_EVIDENCE_INDEX.csv',evidence)
    print('Indexed',len(rows),'primary records;',len(labels),'label sets;',len(pairs),'crossed cases;',len(readers),'students.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);build(p.parse_args().project.resolve())
