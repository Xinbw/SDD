"""Seventh EAB round: shorter captions for the two densest generated tables (runs after export_eab6.py).

Only the caption argument is replaced; the generated table body is asserted unchanged, so no number
is touched. Definitions moved into Sections VI-C and VI-E of the manuscript.
"""
import argparse
from pathlib import Path
CAPTIONS={
 'tab-decomposition.tex':r'Where regional value is lost at random $m=C$ labels (\%; Usenet: balanced accuracy). Ceil.: one-to-one / many-to-one partition ceiling; Room: many-to-one ceiling minus Pub.; Clean / Named: noise-free / protected centered naming; Joint: ten-label LOO; $\Delta_T,\Delta_N,\Delta_S$: Eq.~\eqref{eq:decomposition}; N$\uparrow$: seeds where noise improves naming; $v$: naming-noise std; Gated, Head: Section~\ref{sec:eval:ceiling}.',
 'tab-cost.tex':r'Verification cost under four protocols on label draws 50--99 (acceptance \% / test accuracy \%; public fallback 62.8). Columns are defined in Section~\ref{sec:eval:verification}; Opt.\ is retrospective ($\phi$ chosen with offline test accuracy). Non-degradation was not part of the allocation study.'}

def replace_caption(text,new):
    i=text.index('\\caption{')+len('\\caption{');depth=1;j=i
    while depth:
        depth+={'{':1,'}':-1}.get(text[j],0);j+=1
    return text[:i]+new+text[j-1:],text[j:]

def main(paper):
    for name,cap in CAPTIONS.items():
        p=paper/name;old=p.read_text();new,body=replace_caption(old,cap)
        assert new.endswith(body) and old.endswith(body),name
        p.write_text(new)
    print('captions replaced:',', '.join(CAPTIONS))

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--paper',type=Path,required=True);main(q.parse_args().paper.resolve())
