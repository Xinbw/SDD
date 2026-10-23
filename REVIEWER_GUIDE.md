# Reviewer guide

Start here. This file says how to check the paper's numbers quickly, and
which checks re-aggregate stored records and which retrain from data.

## 1. Five-minute check (CPU only)

```bash
git clone https://github.com/Xinbw/SDD.git && cd SDD
pip install numpy scipy matplotlib mpmath pymupdf torch scikit-learn
python reproduce.py quick
```

`quick` (a few minutes, no GPU):

1. checks the SHA-256 fingerprint of every record and presentation file
   (`RECORD_MANIFEST.json`);
2. recomputes every privacy certificate of the budget-split confirmation
   study from its declared parameters and checks that each deployment's
   composed zCDP budget stays within its total epsilon;
3. re-derives every protected publication decision from the released bounds
   stored in the records;
4. re-derives the frozen budget split (`SELECTION.json`) from the
   development records alone;
5. re-runs every frozen analysis (allocation, verify-teacher, source-disjoint,
   budget-split confirmation, ladder, calculator, weaker encoder, FEMNIST
   replication), checking hash equality, and re-checks every recorded
   certificate with the corrected convolution constant (`recheck_tau.py`);
6. regenerates every table, macro and data figure of the paper from the
   records into `reproduction-output/value-regeneration` and compares them
   byte for byte (figures by text layer) with `paper/`.

`python reproduce.py verify` does the same and also runs the inherited
validators. They recompute all 3426 inherited records' certificates at
90-digit precision, the exact reports, the confidence bands and the decisions
of the 3000 verification cases.

## 2. Retraining check (one GPU, a few minutes)

```bash
python reproduce.py smoke
```

`smoke` retrains private cohorts from the bundled frozen features in a
temporary directory: two CIFAR cohorts of the verification demonstration, one
cohort of each new frozen study (verify-teacher, source-disjoint, and the
development and confirmation sides of the budget-split study). It then
re-releases with the recorded noise keys and requires the protected outputs
and publication decisions to equal the stored records exactly.

`python reproduce.py compile` rebuilds the PDF with Tectonic.

## 3. Two kinds of reproduction

| Check | What is recomputed | Mode |
|---|---|---|
| Re-aggregation | tables, figures, macros, intervals, decisions and certificates from immutable JSON/NPZ records | `quick`, `verify` |
| Retraining | private client models, protected sums with exact discrete-Gaussian noise, candidates and decisions, from features | `smoke` |
| Full rerun | every cohort of a study (`experiments/<stage>/src/run.py`); hours on one GPU | manual |

## 4. Where each object comes from

`CLAIMS.csv` maps each headline number of the abstract and introduction to
its record, generating script and check. `PAPER_EVIDENCE_INDEX.csv` maps
every table and figure of the PDF (and the artifact-only tables) to its
generator and evidence. Each frozen study contains its design file (with UTC
freeze time and source hashes), an `AMENDMENTS.json` with every
post-freeze change, its records and an analysis script:

| Study | Directory | Frozen design |
|---|---|---|
| Budget-split confirmation (Table VII, Fig. 5b-c) | `experiments/release-budgetconf-20261021` | `BUDGETCONF-DESIGN.json`, `SELECTION.json` |
| Source-disjoint ladder (Table VI, Disjoint) | `experiments/release-disjoint-20261020` | `DISJOINT-DESIGN.json` |
| Gated rule plus verification (Table IX) | `experiments/release-verify-teacher-20261020` | `VERIFY-TEACHER-DESIGN.json` |
| Retrospective allocation (Table VI, Opt.) | `experiments/release-allocation-20261019` | `ALLOCATION-DESIGN.json` |
| Verification demonstration (Table V) | `experiments/release-value-20261019` | inherited |

Privacy certificates are stored with every record (`certificates` /
`privacy` fields: actual and honest clients, integer sensitivity, noise scale,
per-release and composed rho). Clean statistics and test labels in the
records are research diagnostics outside the deployment transcript. They are
loaded only after the protected decisions are fixed.
