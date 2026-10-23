Table 4 numerical provenance

These six JSON files are unchanged copies of seed-42 round-1 aggregate-audit outputs from logs/dp_v2/mia_agg/ in the experiment repository. They contain all seven epsilon rows and actual report-distance statistics; this revision adds no training or audit simulation.

CIFAR-10 alpha=0.1: K=10 at every budget.
MNIST alpha=1.0: K=5,20,100,100,2000,500,500 for epsilon=0.01,0.025,0.05,0.1,0.25,0.5,1.0 respectively.

The table exporter reads the K appropriate to each budget and uses all 4950 report pairs, seed 42, T=3 independent fixed-report releases. Historical frozen summaries under results/ are retained unchanged; the expanded active table is generated from these complete JSON inputs.
