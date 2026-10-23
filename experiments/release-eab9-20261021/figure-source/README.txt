Redesigned figures for the EAB8 manuscript

Fig1_Decision_Map: conceptual overview of the buy / release / verify decisions.
Intended width: 3.5 inches (single column). The two supervision sources can be combined.
Fig2_Privacy_Pipeline: public/private operational lanes and a composed privacy ledger.
Intended width: 7.2 inches (double column). The bar shows the reference 50/25/25
allocation, not a claim that this is the frozen allocation in every experiment.

Each figure is supplied as a PDF, an editable-text SVG, and a PNG preview.
Figures_1_2_redesign.pdf combines both figures for review.
redraw_figures.py reproduces all exports using matplotlib and PyMuPDF.
SVGs keep text as text; DejaVu Sans is used. Figures contain no raster assets.

Suggested caption for Figure 1:
Three platform decisions under a shared resource contract: the acquisition
contract for public labels and private supervision, the protected release object,
and the evidence required before publication.

Suggested caption for Figure 2:
A deployment across public and private operations. Protected naming and
verification consume the same whole-client privacy budget; all operations on
released statistics are public post-processing. The reference 50/25/25 split is
shown; alternative allocations change the shares, not the composition rule.
