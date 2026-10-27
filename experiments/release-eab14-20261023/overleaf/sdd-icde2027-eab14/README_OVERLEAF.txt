SDD / ICDE 2027 EAB submission (EAB14), minimal Overleaf package.

1. Upload this zip to Overleaf (New Project > Upload Project).
2. Menu > Settings: Compiler = XeLaTeX, Main document = main.tex.
   (XeLaTeX reproduces the delivered PDF: TeX Gyre Termes fonts, 12 body pages.
    pdfLaTeX also compiles, with standard Times, but line breaks may shift.)
3. Recompile. BibTeX runs automatically; main.bbl is included as a fallback.

IEEEtran.cls / IEEEtran.bst are the copies used for the delivered PDF.
All numbers in tables and *-values.tex are generated from the experiment
records (https://github.com/Xinbw/SDD); do not edit them by hand.
Author block (main.tex, \author{...}) is still a placeholder.
