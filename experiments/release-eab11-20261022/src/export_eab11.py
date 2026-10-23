"""Eleventh ICDE 2027 EAB round: IEEE table style with short titles only.

IEEEtran sets table captions in small capitals, suited to a short title. This
script reads each generated body table tab-<name>.tex (unchanged; still
byte-compared by reproduce.py), replaces its caption by a short title and
writes ieee-tab-<name>.tex. The definitions formerly in the caption are stated,
condensed, in the manuscript paragraph that first cites the table.

Guards: everything except the caption is copied verbatim; titles contain no
digits.
"""
import argparse,re
from pathlib import Path

TITLES={
    'setup':'Evaluated Settings',
    'fewshot':'Public-Only Naming and Learning with Few Labels',
    'joint-controls':'Equal-Label Comparison of Release Interfaces',
    'decomposition':'Where Regional Value Is Lost',
    'verify-ladder':'Verification-Cost Ladder of Publication Targets',
    'cost':'Verification Cost under Four Protocols',
    'budgetconf':'Frozen Budget Split: Development Selection and Confirmation',
    'verify-teacher':'Gated Rule plus Private Verification on New Cohorts',
}

def caption_span(text):
    start=text.index('\\caption{');i=start+len('\\caption{');depth=1
    while depth:
        depth+={'{':1,'}':-1}.get(text[i],0);i+=1
    return start,i

def convert(name,text):
    title=TITLES[name];assert not re.search(r'\d',title),title
    assert text.count('\\caption{')==1,name
    start,end=caption_span(text)
    return text[:start]+'\\caption{'+title+'}'+text[end:]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);paper=p.parse_args().paper
    for name in TITLES:
        (paper/('ieee-tab-'+name+'.tex')).write_text(convert(name,(paper/('tab-'+name+'.tex')).read_text()))
    print('wrote',len(TITLES),'IEEE-style tables')
