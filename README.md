# Private supervision under strong public representations

This artifact accompanies an experiment and analysis (EAB) paper for the
ICDE 2027 research track (data security and privacy): IEEE conference
format, twelve body pages excluding references and the AI-content
acknowledgement, no appendix, six figures (two vector schematics, four data
figures). New-round figures live in `paper/fig-*.tex`; the inherited
validators still count the three inherited data figures in `sec-*.tex`, and
`eab13_layout.py` audits all six. The title carries the required EAB tag. Inherited scientific
records are immutable.

Public repository: <https://github.com/Xinbw/SDD>. After `git clone`, run
`python reproduce.py quick`; files above GitHub's 100 MB limit are stored as
`<name>.part00`, `.part01`, ... and joined automatically (their SHA-256 is
checked against `RECORD_MANIFEST.json`).

**Reviewers: start with `REVIEWER_GUIDE.md` (`python reproduce.py quick`
checks everything from the records in a few minutes on a CPU) and
`CLAIMS.csv` (headline number -> record -> script -> check).**

New in this version:

- `experiments/release-eab13-20261023` (thirteenth round, post-processing only):
  `ledger.py` writes the pilot resource ledger of the CIFAR split selection
  (100 development + 3967 validation labels, 267 protected releases over 17
  configurations; 97.1 deployment budgets if one population answered them),
  the strongest public learners on the FEMNIST confirmation label draws, and
  the smallest integer noise scale over all certificates (applicability of the
  discrete-Gaussian sum bound). `export_eab13.py` redraws Figures 4-6 at their
  printed width (text >= 7 pt; Figure 6 keeps the audit panel, the
  decomposition panel duplicated Table VIII) and writes the combined CIFAR /
  FEMNIST split-confirmation table with the frozen planner predictions.
  Class targets are stated as client-weighted recall, the quantity the
  releases certify (equal to pooled recall when held-out sizes are equal).

- Twelfth round (reviewer W1-W5, M1-M8):
  - `experiments/release-encoder-20261022`: NEW frozen weak-encoder ablation
    (ResNet-18 ImageNet weights; same images, cohorts 8801-8810, label draws,
    learners and release as the CIFAR-1k confirmation). Registered E1
    (positive random-label joint gain) fails (-4.0 points), E2 (directed labels
    beat random labels plus private reports) holds. Weight-source amendment in
    `AMENDMENTS.json` (bit-identical to the official torchvision file).
  - `experiments/release-femnistconf-20261022`: NEW frozen FEMNIST replication
    of the budget-split protocol with a planner prediction frozen before
    confirmation (27000 cases). F1 (planner within 5 points) and F2 (false
    accepts <= 5%) hold: max gap 0.7 points, 0 false accepts.
  - `experiments/release-eab12-20261022`: `recheck_tau.py` re-checks every
    recorded certificate with the corrected convolution constant of Kairouz
    et al. (rho = Delta^2/(2hb^2) + d tau; the frozen accountant wrote d tau/4).
    tau < 10^(-6.6e6) everywhere, below the 10^-30/4 already charged, so every
    recorded budget still holds. `export_eab12.py` writes the macros for the
    external protocols, the encoder ablation, the FEMNIST replication and this
    check; `eab12_layout.py` audits the layout (6 body figures: the planning
    map and the few-shot table moved to the artifact).

- `experiments/release-eab11-20261022` (eleventh round, presentation only):
  `export_eab11.py` reads each generated `paper/tab-*.tex` (unchanged and
  still byte-compared) and writes `paper/ieee-tab-*.tex` with a short
  small-caps title; the definitions formerly in the caption are stated in the
  paragraph that cites the table. It supersedes the tenth round's
  table-note layout (`export_eab10.py`, kept for the record). No number,
  record or experiment changed.

- `experiments/release-eab10-20261022` (tenth round, presentation only):
  body tables follow the IEEE style. `export_eab10.py` reads each generated
  `paper/tab-*.tex` (unchanged and still byte-compared) and writes
  `paper/ieee-tab-*.tex` with a short small-caps title above the table and the
  definitions as a note below it. The tabular is copied verbatim and the
  script asserts that the numbers in title plus note equal those of the
  original caption. No number, record or experiment changed.

- `experiments/release-eab9-20261021` (ninth round, presentation only):
  Figures 1 and 2 are replaced by redesigned vector figures
  (`paper/figures/fig-decision-map.pdf`, `paper/figures/fig-privacy-pipeline.pdf`;
  sources, SVGs and the drawing script in `figure-source/`); the in-figure
  title lines duplicated by the captions are trimmed in LaTeX. No number,
  record or experiment changed.

- `experiments/release-budgetconf-20261021` (eighth round): a NEW private
  experiment frozen in `BUDGETCONF-DESIGN.json` before implementation. The
  45000 private CIFAR images are split per class into two disjoint pools; ten
  development cohorts (pool A, label draws 0-49) select the naming share per
  target and budget by accuracy on the public validation split (never the
  test split), frozen in `SELECTION.json` before any confirmation case; ten
  confirmation cohorts (pool B, draws 50-99, test split) compare it with the
  fixed 50/25/25 split (54000 cases). Primary H1 met; the prediction "class
  shares below .5 at eps<=1" not met (flat cells tie to .5).
  `AMENDMENTS.json` logs a wording correction and a file rename (neither
  changes the procedure).
- `experiments/release-eab8-20261021`: `export_eab8.py` (`tab-budgetconf.tex`,
  `eab8-values.tex`, `figures/fig-ceiling.pdf`, `figures/fig-verification.pdf`;
  the former `fig-evidence.pdf` is split in two and kept as artifact-only),
  `check_budgetconf.py` (certificates, decisions and frozen selection),
  `replay_budgetconf.py` (GPU replay), layout audit and finalizer.

- `experiments/release-allocation-20261019`: frozen feasibility study of
  naming-versus-verification budget allocation (fresh exact re-releases of
  stored clean statistics, 3000/3000 bit-exact reconstruction gate,
  pre-registered go criterion recorded in `ALLOCATION-DESIGN.json`).
- `experiments/release-eab-20261020`: partition ceilings (offline diagnostic,
  hash-gated public assignments), publication-threshold and per-class
  non-degradation post-processing of the 3000 existing protected validation
  outputs (no new query), and the exporter of the new tables.
- `experiments/release-eab2-20261020` (second EAB round): exact loss
  decomposition from stored clean vote counts (never released; offline),
  the test-split audit-circularity check (superseded in the manuscript), a
  selection-rule study whose thresholds were registered in
  `SELECTION-RULE-DESIGN.json` before it ran (rule selected on label draws
  0--49, evaluated on 50--99 of the same cohorts; the rule family itself was
  proposed after earlier rounds, so its results are exploratory), and the
  test minimum class recall of classifiers published by each verification
  rule. No new private query and no new privacy budget.
- `experiments/release-eab3-20261020` (third EAB round): the audit-circularity
  check with audits drawn from the public development split (test labels only
  score; `audit_dev.py`, GPU, re-run by `smoke`), the closed-form verification
  planning map with its case-by-case check against the 3000 recorded decisions
  (`planning.py`, CPU, re-run by `verify`), the exporter `export_eab3.py`
  (`tab-decomposition.tex`, `tab-setup.tex`, `eab3-values.tex`,
  `figures/fig-evidence.pdf`, `figures/fig-planning.pdf`; it supersedes
  `export_eab2.py`, which is kept unchanged), and the layout audit/finalizer.
  No new private query and no new privacy budget.
- `experiments/release-verify-teacher-20261020` (fourth EAB round): a NEW
  private experiment designed and frozen in `VERIFY-TEACHER-DESIGN.json`
  before implementation: the gated selection rule plus total-budget private
  verification on new independent cohorts (MNIST 10k, CIFAR-10 10k, Usenet
  3k; 10 cohorts each; epsilon in {.1, 1}; 50/25/25 split; disjoint holdout
  clients; 6000 cases). `src/run.py` runs it, `src/analyze.py` is the
  pre-registered analysis, `src/post_hoc.py` a post-hoc split written after
  the primary outcome was known, `AMENDMENTS.json` logs the one implementation
  fix (before any case existed) and an operational restart. Primary outcome:
  C1 (MNIST acceptance >= 50%) met, C2 (Usenet acceptance <= 10%) not met, so
  the pre-registered criterion failed; zero false accepts. Cohort checkpoints
  are not shipped; `smoke` retrains cohorts from scratch and reproduces stored
  cases byte for byte (`experiments/release-eab4-20261020/src/replay_vt.py`).
- `experiments/release-eab4-20261020`: exporter of `tab-verify-teacher.tex`
  and `eab4-values.tex`, layout audit and finalizer.
- `experiments/release-disjoint-20261020` (fifth round): a NEW private
  experiment frozen in `DISJOINT-DESIGN.json` before implementation: the CIFAR
  verification ladder on ten cohorts in which each of 40000 private source
  images belongs to exactly one client (32 training / 8 held-out records),
  same candidates, labels, split and bands (3000 cases). Primary S1 met;
  predictions 'no false accept at eps>=1' not met (two cases with zero true
  gain at eps=3) and 'smaller improvement than with reuse' met only nominally.
- `experiments/release-eab7-20261020` (seventh round): `export_eab7.py`
  shortens the captions of `tab-decomposition.tex` and `tab-cost.tex` (table
  bodies asserted unchanged), layout audit and finalizer. The optimized
  column of `tab-cost.tex` is a retrospective sensitivity analysis: the
  allocation study chose the naming share with offline test utility on draws
  0-49 and the same test set scores draws 50-99.
- `experiments/release-eab6-20261020` (sixth round): post-processing without
  new queries: `cost_table.py` (fixed, tight, re-released, optimized and
  source-disjoint costs on draws 50-99 with acceptance and the selected
  shares of the pre-registered allocation study; supersedes the eab5
  `tab-cost.tex`), `calc24.py` (calculator on all 24 ladder cells),
  `hetero.py` (class heterogeneity of the source-disjoint partition),
  `export_eab6.py`, layout audit and finalizer.
- `experiments/release-eab5-20261020`: post-processing without new queries:
  one-sided tighter class bands (`tight_bands.py`), a balanced-accuracy
  verification rule on the released histograms of the verify-teacher cases
  (`balanced.py`, added after that outcome), the out-of-sample check of the
  frozen planning calculator (`planning_oos.py`), the exporter
  `export_eab5.py` (decoder-matched signed decomposition, `tab-cost.tex`,
  verify-teacher table with both metrics), `replay_disjoint.py`, layout audit
  and finalizer.

Tables omitted from the body for space (`tab-allocation.tex`, `tab-bound-values.tex`,
`tab-publication-gate.tex`, `tab-resource-map.tex`,
`tab-validation-coverage.tex`, `tab-ceiling.tex`, `tab-zero-label-pool.tex`,
`tab-rounds.tex`, `tab-no-feedback.tex`) are retained, still regenerated by
`verify`, and listed as artifact-only in `PAPER_EVIDENCE_INDEX.csv`.

## Reproduce

```bash
python reproduce.py verify
python reproduce.py smoke
python reproduce.py compile
```

`verify` needs Python, NumPy, SciPy, Matplotlib, mpmath, PyMuPDF and PyTorch;
CUDA and TeX are unnecessary for numerical verification. It checks inherited
fingerprints, the ICDE layout (`experiments/release-icde-20261019`), exact budget composition, quantized bounded reports, confidence
bands, selection/publication decisions, and every generated numerical table,
macro and active figure. `smoke` additionally needs CUDA and scikit-learn.
It reconstructs two actual cohorts and their exact protected releases from
frozen features, and re-derives the clean-count decomposition and audit check. Output is isolated under `reproduction-output`. `compile`
needs Tectonic; the bundled PDF is the manuscript of record.

Resume the new frozen experiments with `private-validation` or `no-feedback`.
Existing records are preserved. `export` regenerates the manuscript numerical
objects. Four included feature bundles avoid raw-data/checkpoint downloads.
Execution versions are in `experiments/release-value-20261019/ENVIRONMENT.json`.

## Scientific phases and units

The inherited frontier contains 3426 primary records, 90 actual cohorts,
4224 public label sets and 75480 crossed naming cases, with 150960 matched
head/point controls. Its scientific fingerprints are retained in the new
stage's `PARENT-MANIFEST.json`, including anonymous-parent alternatives.
The new work adds:

- A postprocessing audit of 110 existing protected point matrices at zero
  public labels, with centered and coverage readouts. No new private query.
- Ten actual source-disjoint CIFAR-1k validation cohorts, seeds 8901--8910.
  A fixed stratified 80/20 split of private source images prevents any
  training/held-out source overlap across the entire population. Each client
  samples 480 training and 120 held-out records under the same Dirichlet .1
  class proportions; training takes 180 private-only SGD steps.
- One hundred fresh random ten-label sets, seed tape 12000001. These are
  separate from the frontier's label draws. All selection labels count in m.
- Thirty naming conditions (ten cohorts at total epsilon .1/1/3) and 3000
  verification cases. Each case has two protected validation vectors. They
  share teachers, labels and naming matrices; they are not independent trials.
- Ten three-release private-only controls on the paired 600-record cohorts
  8501--8510. Their first/final clean votes equal the original T=1 votes;
  their first protected matrices equal the original feedback T=3 matrices.

The experiment index has 6466 JSON records: 3426 inherited, 30 new naming,
3000 two-vector verification records and ten three-release controls. A JSON
record is therefore not synonymous with one independent cohort or one release.
The total public-label index has 4324 sets (4224 inherited plus 100 new).

## One total privacy budget for private verification

For each counterfactual deployment, the total zCDP allowance is split
50% naming, 25% paired gain, 25% class statistics. The scalar client report
is rounded s*(1+candidate-minus-public gain)/2, with integer sensitivity s.
The C by 4 joint correctness histogram is normalized per client and rounded
onto an exact integer simplex; squared sensitivity is 2*s*s. Every coordinate
receives independent exact distributed discrete-Gaussian noise shares.
All three conditional charges compose under whole-client replacement.
Disjoint training/held-out source records do not give parallel client privacy.

The candidate is chosen before private verification from a restricted regional
family: public count matching, the existing eight joint weights, and private
centered/coverage rules. The public comparator is fixed count matching.
LOO uses only the ten public labels, with public-first ties. This demonstration
isolates validation and is separate from the fourteen-learner frontier.
No test labels, clean private counts or local-model utility select a candidate
or a publication decision.

The gain lower bound has one-sided failure probability .025. Simultaneous
class-recall bounds have family failure probability .025. Together their
noise coverage is at least .95 apart from the certified negligible wrap term.
The target is utility on the finite actual private holdout, not future-client
performance, selective false-discovery rate, or benefit over each local model.
Two policies evaluate private replacement: gain only, or gain plus positive
lower recall in every class. Rejection uses the already-available public
comparator; it adds no class guarantee to that fallback and refunds no privacy.

The initial coordinate-union class bands are preserved in raw records.
`linear_bands.py` applies a sharper fixed analytic postprocessing to the same
protected histogram: it bounds the two-coordinate correct mass and four-
coordinate class mass directly. It uses no new query, privacy or test tuning.
Both initial and direct-sum decisions remain reproducible and indexed.

At total epsilon .1/1/3, gain acceptance is 4.6/86.8/87.0%; direct class
acceptance is 0/6.2/75.9%. Test utility is unconditional, including rejections
at the public fallback. Different budgets are distinct counterfactuals, not
an uncharged collection of releases in one deployed transcript.

## Locate evidence

- `EXPERIMENT_INDEX.csv`: all 6466 primary JSON records and roles.
- `PRIVATE_VERIFICATION_INDEX.csv`: 3000 new gain/class/coarse-policy cases.
- `PRIVATE_COHORT_INDEX.csv`: ten actual globally source-disjoint cohorts.
- `ZERO_LABEL_POOL_INDEX.csv`: 110 existing protected sources and readouts.
- `NO_FEEDBACK_INDEX.csv`: all ten last/pooled feedback/private-only contrasts.
- `LABEL_BUDGET_INDEX.csv`: all 4324 counted public label sets.
- `LABEL_COHORT_INDEX.csv`: 75480 inherited frontier decisions.
- `JOINT_RELEASE_CONTROL_INDEX.csv`: 150960 inherited matched release controls.
- `PAPER_EVIDENCE_INDEX.csv`: every active table/figure and argument role.
- `paper/value-data.json`: pooling, gain distributions and private verification.
- `paper/joint-data.json`: fourteen-learner frontier and matched private controls.
- `paper/frontier-data.json`: paired release-count and legacy-mixture evidence.

New experiment paths are inside `experiments/release-value-20261019`.
The original independent ten-label gain has mean13.0, median13.6 and fifth
percentile -3.8 points; 9% of label means and16% of crossed cases are negative.
Label means average ten cohorts; cases are dependent. Cohort and label axes
remain separate, including the ten-cohort uncertainty limitation.

## Scope and execution provenance

All task labels in this study can be supplied by public annotators. Random,
supplied-balanced and feature-directed annotation are distinct resource
contracts; the benchmarks do not establish that targeted annotation is
unavailable or that private Web semantics are irreplaceable. Zero-label
pooling and time-varying teachers prevent reading sensitivity ratios as final
accuracy advantages or assigning every T=1 gain to feedback echo.

Whole-client adjacency protects one constructed reporting dataset. Source
copies at other image clients remain in adjacent databases. Author/writer
benchmarks retain their natural disjoint client roles. Clean statistics,
benchmark test labels and public deterministic replay keys are research
diagnostics outside the deployment transcript. Production noise uses fresh
private keys. Private model checkpoints are excluded from deliveries.

Exact source snapshots reproduce the frozen execution hashes under
`src/implementation-history`. `EXECUTION-SOURCE-SNAPSHOTS.json` indexes them.
`SOURCE-NOTES.txt` distinguishes execution from subsequent analytical
postprocessing. The exact sampler and frozen features are inherited unchanged.
Learning payloads are distinct from measured secure-aggregation runtime.
The full local archive retains earlier drafts and delivery provenance.
