"""Tenth ICDE 2027 EAB round: IEEE table style (short title above, note below).

IEEEtran sets table captions in small capitals, which suits a short title but
not a paragraph of definitions. For each generated body table this script reads
the generated file tab-<name>.tex (unchanged; still byte-compared by
reproduce.py), keeps a short title as the caption and moves the remaining
definitions to a \\tablenote below the tabular, writing ieee-tab-<name>.tex.

Guards: the tabular body is copied verbatim; titles and rewording contain no
digits; the multiset of numbers in title plus note equals that of the original
caption, so no generated number can be dropped, added or altered.
"""
import argparse,re
from collections import Counter
from pathlib import Path

SPEC={
    'setup':('Evaluated Settings',[('Evaluated settings. ','')]),
    'fewshot':('Public-Only Naming and Learning with Few Labels',
        [('Public-only naming and learning with 1/3/5 supplied labels per class, 50 independent draws. ','1/3/5 supplied labels per class; 50 independent draws. ')]),
    'joint-controls':('Equal-Label Comparison of Release Interfaces',
        [('Equal-label comparison at random $m=C$: 100 public label draws','Random $m=C$ labels; 100 public label draws')]),
    'decomposition':('Where Regional Value Is Lost',
        [('Where regional value is lost at random $m=C$ labels (\\%; Usenet: balanced accuracy). ','Random $m=C$ labels; \\%, Usenet balanced accuracy. ')]),
    'verify-ladder':('Verification-Cost Ladder of Publication Targets',
        [('Verification-cost ladder on the same 3000 protected validation outputs (fixed $50/25/25$ split; no new query). ','Same 3000 protected validation outputs, fixed $50/25/25$ split, no new query. ')]),
    'cost':('Verification Cost under Four Protocols',
        [('Verification cost under four protocols on label draws 50--99 (acceptance \\% / test accuracy \\%; public fallback 62.8). ','Label draws 50--99. Entries: acceptance \\% / test accuracy \\%; public fallback 62.8. ')]),
    'budgetconf':('Frozen Budget Split: Development Selection and Confirmation',
        [('Budget split selected on independent development cohorts, frozen, then confirmed on new cohorts and the test split (acceptance \\% / test accuracy \\%, label draws 50--99). ',
          'Development cohorts select $\\phi$; new cohorts and the test split confirm it. Entries: acceptance \\% / test accuracy \\%; label draws 50--99. ')]),
    'verify-teacher':('Gated Rule plus Private Verification on New Cohorts',
        [('Gated rule plus private verification on new independent cohorts at total $\\varepsilon=1$ ($50/25/25$ split; draws 50--99; design frozen before implementation). ',
          'New independent cohorts; total $\\varepsilon=1$, $50/25/25$ split, label draws 50--99; design frozen before implementation. ')]),
}

def numbers(text):return Counter(re.findall(r'\d+',text))

def caption_span(text):
    start=text.index('\\caption{');i=start+len('\\caption{');depth=1
    while depth:
        depth+={'{':1,'}':-1}.get(text[i],0);i+=1
    return start,i,text[start+len('\\caption{'):i-1]

def convert(name,text):
    title,edits=SPEC[name];assert not re.search(r'\d',title),title
    start,end,caption=caption_span(text);note=caption
    for old,new in edits:
        assert note.count(old)==1,(name,old);assert numbers(new)<=numbers(old),(name,new);note=note.replace(old,new)
    assert numbers(note)==numbers(caption),(name,numbers(caption)-numbers(note))
    out=text[:start]+'\\caption{'+title+'}'+text[end:]
    assert out.count('\\end{tabular}')==1
    return out.replace('\\end{tabular}','\\end{tabular}\n\\tablenote{'+note+'}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);paper=p.parse_args().paper
    for name in SPEC:
        source=(paper/('tab-'+name+'.tex')).read_text()
        (paper/('ieee-tab-'+name+'.tex')).write_text(convert(name,source))
    print('wrote',len(SPEC),'IEEE-style tables')
