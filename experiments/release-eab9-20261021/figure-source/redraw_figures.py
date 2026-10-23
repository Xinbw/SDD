from pathlib import Path
import zipfile
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch
from matplotlib.path import Path as MPath
import fitz

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'mathtext.fontset': 'dejavusans',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})
INK = '#24364B'
NAVY = '#183753'
BLUE = '#376FA3'
BLUE_FILL = '#EFF4F9'
BLUE_LINE = '#C7D7E6'
AMBER = '#A87334'
AMBER_FILL = '#FAF4EB'
AMBER_LINE = '#E3CFB1'
GREEN = '#2C7562'
GREEN_FILL = '#EAF4EF'
MUTED = '#627286'
LINE = '#D6DEE6'

def setup(w, h):
    fig = plt.figure(figsize=(w, h), dpi=240)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0,w), ylim=(0,h))
    ax.set_axis_off()
    return fig, ax

def txt(ax, x, y, s, size=8, color=INK, weight='normal', ha='center', va='center', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, linespacing=1.25, **kw)

def rect(ax, x, y, w, h, fill='white', edge=LINE, radius=.045, lw=.65):
    p = FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                       facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(p)
    return p

def line(ax, x1,y1,x2,y2,color=LINE,lw=.7):
    ax.plot([x1,x2],[y1,y2],color=color,lw=lw,solid_capstyle='round')

def badge(ax, x,y,n,color=NAVY,r=.10):
    ax.add_patch(Circle((x,y),r,facecolor=color,edgecolor='none'))
    txt(ax,x,y,str(n),7.0,'white','bold')

def arrow_path(ax, coords, color=MUTED):
    path=MPath(coords,[MPath.MOVETO]+[MPath.LINETO]*(len(coords)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=7,
                                lw=.8,color=color,joinstyle='round'))

def matrix(ax, cx,cy,cell=.060,rows=3,cols=4,color=BLUE):
    w=cols*cell;h=rows*cell
    for r in range(rows):
        for c in range(cols):
            active=c==(r*2)%cols
            ax.add_patch(Rectangle((cx-w/2+c*cell,cy-h/2+r*cell),cell*.76,cell*.76,
                                  facecolor=color if active else '#DCE5EE',edgecolor='none'))

def points(ax,cx,cy,color=BLUE):
    for r in range(3):
        for c in range(4):
            ax.add_patch(Circle((cx+(c-1.5)*.060,cy+(r-1)*.060),.013,
                                facecolor=color,edgecolor='none'))

def regions(ax,cx,cy,scale=1):
    colors=[BLUE,'#6F9CBA','#9CB6CA']
    for offx,offy,col in [(-.12,-.01,colors[0]),(.05,.055,colors[1]),(.13,-.06,colors[2])]:
        ax.add_patch(Circle((cx+offx*scale,cy+offy*scale),.080*scale,
                            facecolor=col,alpha=.15,edgecolor=col,lw=.6))
        for dx,dy in [(-.026,-.017),(.023,-.020),(.001,.024)]:
            ax.add_patch(Circle((cx+(offx+dx)*scale,cy+(offy+dy)*scale),.010*scale,
                                facecolor=col,edgecolor='none'))

def export(fig,stem):
    for ext in ('pdf','svg','png'):
        fig.savefig(ROOT/f'{stem}.{ext}',dpi=320,transparent=False)
    plt.close(fig)

# Figure 1: a conceptual decision map, at single-column width.
fig,ax=setup(3.5,3.12)
txt(ax,.14,2.965,'Three decisions, one resource contract',9.4,NAVY,'bold',ha='left')
txt(ax,.14,2.75,'Frozen features + a public proxy pool',7.4,MUTED,ha='left')
line(ax,.14,2.61,3.36,2.61)

badge(ax,.25,2.28,1)
txt(ax,.45,2.28,'BUY',8.2,NAVY,'bold',ha='left')
rect(ax,1.03,2.035,1.04,.49,BLUE_FILL,BLUE_LINE)
rect(ax,2.32,2.035,1.04,.49,AMBER_FILL,AMBER_LINE)
txt(ax,1.55,2.355,r'$m$ public labels',7.7,NAVY,'bold')
txt(ax,1.55,2.17,'random / directed',7.0,MUTED)
txt(ax,2.84,2.355,'Private reports',7.7,AMBER,'bold')
txt(ax,2.84,2.17,r'$(\varepsilon,\delta)$ · aggregates',7.0,MUTED)
txt(ax,2.195,2.28,'+',11,MUTED)

line(ax,.14,1.92,3.36,1.92)
badge(ax,.25,1.57,2)
txt(ax,.45,1.57,'RELEASE',7.5,NAVY,'bold',ha='left')
for x in (1.03,1.85,2.67): rect(ax,x,1.245,.69,.60,'white',BLUE_LINE)
matrix(ax,1.375,1.66,.052,3,4)
points(ax,2.195,1.66)
regions(ax,3.015,1.66,.74)
for cx,label,dim in [(1.375,'Head',r'$d_{\rm head}$'),(2.195,'Point',r'$|P|C$'),(3.015,'Region',r'$KC$')]:
    txt(ax,cx,1.455,label,7.8,NAVY,'bold')
    txt(ax,cx,1.315,dim,7.4,MUTED)

line(ax,.14,1.13,3.36,1.13)
badge(ax,.25,.82,3)
txt(ax,.45,.82,'VERIFY',7.8,NAVY,'bold',ha='left')
rect(ax,1.03,.565,1.04,.49,'white',BLUE_LINE)
rect(ax,2.32,.565,1.04,.49,'white',BLUE_LINE)
txt(ax,1.55,.885,'Mean gain',7.7,NAVY,'bold')
txt(ax,1.55,.69,r'$G>0$',8.2,MUTED)
txt(ax,2.84,.885,'Gain + recall',7.4,NAVY,'bold')
txt(ax,2.84,.69,r'$\min_c r_c\geq r_0$',8.0,MUTED)

rect(ax,.14,.175,3.22,.27,GREEN_FILL,'#C8DED3',radius=.045)
txt(ax,1.75,.31,r'Publish $f$ or return public $b$',8.0,GREEN,'bold')
txt(ax,1.75,.070,r'$m$ & access · $(\varepsilon,\delta)$ · payload · public compute',7.0,MUTED)
export(fig,'Fig1_Decision_Map')

# Figure 2: dataflow through two operational lanes, with a separate budget ledger.
fig,ax=setup(7.2,3.12)
txt(ax,.15,2.96,'One deployment, one composed privacy budget',10.0,NAVY,'bold',ha='left')
rect(ax,.14,1.74,6.92,.96,'#F1F5F9','#E0E7EE',radius=.05)
rect(ax,.14,.75,6.92,.91,'#FCF7F0','#EEE2D1',radius=.05)
txt(ax,.285,2.18,'PUBLIC',7.1,BLUE,'bold',rotation=90)
txt(ax,.285,1.20,'PRIVATE',7.1,AMBER,'bold',rotation=90)

xs=[.98,2.28,3.59,4.90,6.22]
for x,n,y,col in [(xs[0],1,2.555,BLUE),(xs[1],2,1.515,AMBER),
                   (xs[2],3,2.555,BLUE),(xs[3],4,1.515,AMBER),(xs[4],5,2.555,GREEN)]:
    badge(ax,x,y,n,col,r=.080)

txt(ax,xs[0],2.355,'Public partition',8.1,NAVY,'bold')
regions(ax,xs[0],2.13,.95)
txt(ax,xs[0],1.945,'Frozen features',7.1,MUTED)
txt(ax,xs[0],1.81,r'$K$ public regions',7.1,MUTED)

txt(ax,xs[1],1.325,'Protected naming',8.0,NAVY,'bold')
matrix(ax,xs[1],1.105,.061,3,5,AMBER)
txt(ax,xs[1],.935,r'Bounded $K\times C$ votes',7.1,MUTED)
txt(ax,xs[1],.825,'SecAgg + DDG',7.0,AMBER)

txt(ax,xs[2],2.355,'Joint naming',8.1,NAVY,'bold')
matrix(ax,xs[2]-.13,2.13,.052,3,4,BLUE)
txt(ax,xs[2]+.065,2.13,'+',10,BLUE)
for i in range(3):
    rect(ax,xs[2]+.16+i*.035,2.05+i*.035,.12,.16,'white',BLUE_LINE,radius=.01,lw=.6)
txt(ax,xs[2],1.945,r'$A$ + $m$ public labels',7.1,MUTED)
txt(ax,xs[2],1.81,'Public selection',7.1,MUTED)

txt(ax,xs[3],1.325,'Private verification',7.5,NAVY,'bold')
for i,h in enumerate([.06,.16,.105,.20]):
    ax.add_patch(Rectangle((xs[3]-.17+i*.085,1.025),.048,h,facecolor=AMBER,edgecolor='none'))
line(ax,xs[3]-.21,1.015,xs[3]+.18,1.015,AMBER_LINE,.7)
txt(ax,xs[3],.935,'Paired gain + class counts',7.0,MUTED)
txt(ax,xs[3],.825,'SecAgg + DDG',7.0,AMBER)

txt(ax,xs[4],2.355,'Publication rule',8.1,NAVY,'bold')
rect(ax,5.65,2.035,1.14,.235,GREEN_FILL,'#C8DED3',radius=.035)
txt(ax,xs[4],2.15,r'Pass: publish $f$',7.5,GREEN,'bold')
rect(ax,5.65,1.795,1.14,.195,'white','#D6DEE6',radius=.035)
txt(ax,xs[4],1.892,r'Else: public $b$',7.2,MUTED)

arrow_path(ax,[(1.39,2.14),(1.60,2.14),(1.60,1.105),(1.92,1.105)])
arrow_path(ax,[(2.61,1.105),(2.94,1.105),(2.94,2.14),(3.16,2.14)])
arrow_path(ax,[(4.02,2.14),(4.25,2.14),(4.25,1.105),(4.52,1.105)])
arrow_path(ax,[(5.22,1.105),(5.48,1.105),(5.48,2.15),(5.64,2.15)])
for x,label in [(1.60,'regions'),(2.94,r'$A$'),(4.25,r'$f,b$'),(5.48,'bounds')]:
    txt(ax,x,1.69,label,7.0,MUTED,bbox={'facecolor':'white','edgecolor':'none','pad':.5})

txt(ax,.15,.61,'BUDGET',7.5,AMBER,'bold',ha='left')
txt(ax,3.83,.61,r'$\rho_{\rm name}+\rho_{\rm gain}+\rho_{\rm class}\leq\rho_{\rm total}$',8.3,NAVY)
bar_x=.85;bar_y=.265;bar_w=6.20;bar_h=.23
for start,width,col,label in [(0,.5,'#EAD4B3',r'$\rho_{\rm name}$  50%'),
                               (.5,.25,'#D5E2EF',r'$\rho_{\rm gain}$  25%'),
                               (.75,.25,'#C9DDD4',r'$\rho_{\rm class}$  25%')]:
    ax.add_patch(Rectangle((bar_x+bar_w*start,bar_y),bar_w*width,bar_h,
                          facecolor=col,edgecolor='white',lw=1))
    txt(ax,bar_x+bar_w*(start+width/2),bar_y+bar_h/2,label,7.5,NAVY)
txt(ax,.15,.09,r'Reference split shown. Every private release is charged to the same whole-client $(\varepsilon,\delta)$ budget.',7.1,MUTED,ha='left')
export(fig,'Fig2_Privacy_Pipeline')

# A two-page preview; the individual PDFs retain the intended paper widths.
combined=fitz.open()
for stem in ('Fig1_Decision_Map','Fig2_Privacy_Pipeline'):
    with fitz.open(ROOT/f'{stem}.pdf') as d: combined.insert_pdf(d)
combined.save(ROOT/'Figures_1_2_redesign.pdf')
combined.close()

readme='''Redesigned figures for the EAB8 manuscript\n\nFig1_Decision_Map: conceptual overview of the buy / release / verify decisions.\nIntended width: 3.5 inches (single column). The two supervision sources can be combined.\nFig2_Privacy_Pipeline: public/private operational lanes and a composed privacy ledger.\nIntended width: 7.2 inches (double column). The bar shows the reference 50/25/25\nallocation, not a claim that this is the frozen allocation in every experiment.\n\nEach figure is supplied as a PDF, an editable-text SVG, and a PNG preview.\nFigures_1_2_redesign.pdf combines both figures for review.\nredraw_figures.py reproduces all exports using matplotlib and PyMuPDF.\nSVGs keep text as text; DejaVu Sans is used. Figures contain no raster assets.\n\nSuggested caption for Figure 1:\nThree platform decisions under a shared resource contract: the acquisition\ncontract for public labels and private supervision, the protected release object,\nand the evidence required before publication.\n\nSuggested caption for Figure 2:\nA deployment across public and private operations. Protected naming and\nverification consume the same whole-client privacy budget; all operations on\nreleased statistics are public post-processing. The reference 50/25/25 split is\nshown; alternative allocations change the shares, not the composition rule.\n'''
(ROOT/'README.txt').write_text(readme,encoding='utf-8')
package_files=[ROOT/'README.txt',Path(__file__),ROOT/'Figures_1_2_redesign.pdf']
for stem in ('Fig1_Decision_Map','Fig2_Privacy_Pipeline'):
    package_files += [ROOT/f'{stem}.{ext}' for ext in ('pdf','svg','png')]
with zipfile.ZipFile(ROOT/'Figures_1_2_vector.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in package_files: z.write(p,p.name)
for p in package_files+[ROOT/'Figures_1_2_vector.zip']:
    print(p.name,p.stat().st_size)
