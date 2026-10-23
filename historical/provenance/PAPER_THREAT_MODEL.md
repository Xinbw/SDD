# Threat model and formal privacy guarantee (paper-ready)

Target venue: TDSC. This file is the source of truth for §"Threat model" and
§"Privacy analysis". Every constant below was computed from the code in this
repo (`dp_utils.py`, `sdd_core.py`, `rdp_accounting.py`), not from memory —
regenerate with `python -c` snippets in §8 if the mechanism changes.

Companion files: `SDD_PHASE1_NOTES.md` (experiments), memory notes
`sdd-privacy-mechanism`, `sdd-known-issues`.

---

## 1. Setting and notation

| symbol | meaning | value in main experiments |
|---|---|---|
| $N$ | client population | 100 (real) / 10³–10⁴ (characterised) |
| $n$ | clients reporting in a round (cohort) | $qN$ |
| $q$ | participation rate | 1.0 (0.1 in the RDP variant) |
| $T$ | rounds | 3 |
| $K$ | semantic clusters over the proxy set | 20 |
| $C$ | classes | 10 |
| $d = KC$ | report dimension | 200 |
| $s = 2^b$ | quantisation grid | $2^{16}$ |
| $\sigma$ | noise std placed on the **aggregate sum**, probability units | 53.68 (zCDP) / 44.32 (RDP) |
| $t$ | colluding clients tolerated | 0 in current code (see §5) |

Each client $i$ holds a private dataset $D_i$ and a locally trained model
$f_i$. The proxy set $P$ ($|P| = 2000$, public, unlabelled) is partitioned once
into $K$ clusters $P_1, \dots, P_K$ by a **pre-computed, data-independent**
clustering (`cluster.py` forbids online SimCLR/K-Means, so the clustering is not
a function of any $D_i$ and consumes no privacy budget).

Client $i$'s round-$r$ report is the per-cluster mean posterior

$$R_i \in [0,1]^{K \times C}, \qquad R_i[k, \cdot] \;=\; \frac{1}{|P_k|}\sum_{x \in P_k} f_i(x) \;\in\; \Delta^{C-1},$$

i.e. **each of the $K$ rows is a probability vector.** This row-simplex property
is what the sensitivity bound rests on, and it is why clients must *not*
individually project or threshold (a non-linear per-client map breaks the
additivity of the noise shares — enforced in `sdd_core.py`, which projects once
on the aggregate).

## 2. Protocol (what the server sees)

```
Round r:
  client i:  R_i        <- per-cluster mean posterior          (K x C floats)
             Z_i        <- quantize(R_i, s)                    (K x C ints)
             Y_i        <- Z_i + DiscreteGaussian(0, sigma_i^2 * s^2)  (int)
             split Y_i into m = 4 uniform additive shares over Z_M,
             submit shares through the anonymous shuffler
  server:    S          <- sum_i Y_i        (all shares summed; ONLY this)
             A          <- dequantize(S, s) / n
             A          <- project_rows_to_simplex(A)          (post-processing)
             broadcast A as soft pseudo-labels; clients distil with KL
```

with $\sigma_i = \sigma/\sqrt{n}$, so the $n$ independent shares sum to variance
$\sigma^2$ on $S$ (`DPConfig.client_noise_std`).

The split-and-mix layer is **not implemented in code**: summing $m$ uniform
additive shares is bit-for-bit identical to summing $Y_i$ directly, so it cannot
change any number we report. It is part of the *claimed* protocol because it is
what justifies assumption A1. State this explicitly in the paper — do not let a
reviewer discover it.

## 3. Trust assumptions

- **A1 (aggregate-only view).** The server observes only $S = \sum_i Y_i$, never
  an individual $Y_i$. Realised by the anonymous multi-message shuffler
  (split-and-mix / IKOS-style): each client's contribution is split into $m$
  uniform shares that are individually independent of $Z_i$.
- **A2 ($t$-collusion).** At most $t$ of the $n$ reporting clients collude with
  the server, i.e. reveal their own noise shares. Current code assumes $t = 0$.
- **A3 (shuffler non-collusion).** The shuffler does not collude with the
  server. If it does, the guarantee degrades to per-client local DP with
  $\sigma_i = \sigma/\sqrt n$ — which is vacuous. This is the mechanism's
  single hardest assumption and belongs in the abstract's scope sentence, not
  buried in §3.
- **A4 (honest-but-curious clients).** Clients follow the protocol: they add
  their noise share and do not lie about $R_i$. We make **no robustness claim**;
  a single client omitting its share reduces the realised $\sigma$ by
  $\sqrt{(n-1)/n}$, and a malicious client can poison pseudo-labels. Robustness
  is explicitly out of scope.

**Adjacency.** Client-level replace-one: $D \simeq D'$ iff they differ in the
full local dataset of one client (hence in one entire report $R_i$). We protect
*a client's whole contribution to a round*, not a single sample. Under
`--dp-accounting rdp` with $q < 1$ the analysis is Poisson-subsampled
add/remove adjacency and additionally requires **secrecy of the sample**
(clients self-select by private coin, and the shuffler hides who participated);
say so wherever $q < 1$ numbers appear.

## 4. Sensitivity

**Lemma 1 (probability-unit sensitivity).** For adjacent $D \simeq D'$ differing
in client $i$'s report, $\lVert R_i - R_i' \rVert_2 \le \sqrt{2K}$, and this is
tight.

*Proof.* Rows are independent across $k$, so
$\lVert R_i - R_i'\rVert_2^2 = \sum_{k=1}^K \lVert R_i[k,\cdot] - R_i'[k,\cdot]\rVert_2^2$.
For two probability vectors $p, p' \in \Delta^{C-1}$,
$\lVert p - p'\rVert_2 \le \sqrt 2$, with equality iff $p, p'$ are distinct
one-hot vectors. Summing $K$ rows gives $\sqrt{2K}$, attained when every cluster
flips between two distinct one-hot labels. $\square$

Two consequences worth a sentence each in the paper:

- $\Delta_2 = \sqrt{2K}$ scales as $\sqrt K$, not the $2K$ an $L_1$/Laplace
  route would give — this is the whole reason the Gaussian mechanism is used.
- **$\Delta_2$ is independent of $|P|$.** This is the sharp distinction from
  PATE-style schemes, whose sensitivity unit is *per query*: releasing labels
  for 2000 proxy items costs us one release of $\sqrt{2K}$, not 2000 queries.
  The one-hot vote encoding (`diagnose_round1_noise.py --encodings`) leaves
  $\Delta_2$ and hence $\sigma$ unchanged while raising the clean ceiling
  0.6355 → 0.6910, precisely because sensitivity is already maximal over
  one-hots.

**Lemma 2 (integer-grid sensitivity).** On the quantised grid,
$\Delta_2^{\text{int}} \le s\sqrt{2K} + \sqrt d$.

*Proof.* Write $\mathrm{quantize}(R) = sR + e$ with coordinate-wise rounding
error $\lvert e_j \rvert \le 1/2$. Then
$\mathrm{quantize}(R) - \mathrm{quantize}(R') = s(R - R') + (e - e')$, and each
coordinate of $e - e'$ lies in $[-1, 1]$, so $\lVert e - e'\rVert_2 \le \sqrt d$.
Triangle inequality with Lemma 1 gives the claim. $\square$

The code accounts with $\Delta_2 = \sqrt{2K}$ in probability units, i.e. it
omits the $\sqrt d$ term. **Quantified slack:** with $K=20, C=10, s=2^{16}$ the
inflation factor is $1 + 3.41\times 10^{-5}$, so a nominal $\varepsilon = 1$ is
really $\varepsilon \le 1.0000349$. **Report the corrected number in the paper
and cite this line** — it costs nothing and removes an easy reviewer kill. Do
*not* "fix" it in code: changing $\Delta_2$ shifts every RNG stream and would
invalidate the logged results for a $3.4\times10^{-5}$ effect. The slack shrinks
linearly in $s$, so raising `--dp-quantization-bits` makes it disappear entirely
if a reviewer insists.

## 5. Main guarantee

**Theorem 1 (privacy of one round).** Under A1, A3, A4 with $t = 0$, the
server's view $S$ of one round satisfies $\rho$-zCDP with
$\rho = \Delta_2^2 / (2\sigma^2)$, up to the additive discrete-Gaussian
correction of Kairouz et al. (2021, Thm. 1) which is $O(e^{-2\pi^2 \sigma_i^2})$
in our parameter regime and hence numerically negligible: the code enforces
$\sigma_i s \ge 10$ (`MIN_INTEGER_SIGMA`), giving a correction below $10^{-800}$.

**Theorem 2 (whole run).** $T$ rounds compose to $\rho_{\text{total}} = T\rho$,
and $(\varepsilon, \delta)$-DP with
$\varepsilon = \rho_{\text{total}} + 2\sqrt{\rho_{\text{total}}\ln(1/\delta)}$.
Inverting at $\varepsilon = 1, \delta = 10^{-5}, T = 3, K = 20$ gives
$\rho_{\text{total}} = 0.02082$ and $\sigma = 53.68$. The Rényi accountant
(`--dp-accounting rdp`, integer orders 2..512, CKS conversion) is tighter on the
same mechanism and same assumptions: $\sigma = 44.32$ at the optimal order
$\alpha = 18$, i.e. **17.5 % less noise for free**.

**Corollary 1 (residual noise, the utility-relevant quantity).** The noise on the
broadcast pseudo-label matrix $A$ has per-coordinate std $\sigma / n$ — it decays
as $1/n$, not $1/\sqrt n$, because the noise is added once to the sum rather than
once per client.

| $n$ | residual std (zCDP) | residual std (RDP) |
|---|---|---|
| 100 | 0.5368 | 0.4431 |
| 1 000 | 0.0537 | 0.0443 |
| 10 000 | 0.00537 | 0.00443 |

Pair this table with the measured per-cluster top1−top2 margin (median 0.0181,
`diagnose_round1_noise.py`) to get the deployment-scale requirement: usable
pseudo-labels need residual std $\lesssim 0.02$, hence $n \gtrsim 2685$ at
$K{=}20, T{=}3$. That inequality — not a tuning failure — is the paper's
central negative result, and it is why the claim is reframed as cross-device
scale characterisation.

**Corollary 2 ($t$-collusion, to implement).** If $t$ clients reveal their
shares, the adversary subtracts them and the residual noise variance on $S$
drops to $(n-t)\sigma^2/n$. To hold the guarantee under A2, clients must use
$\sigma_i = \sigma/\sqrt{n-t}$; the cost is a noise inflation of
$\sqrt{n/(n-t)}$ on the useful signal:

| $t$ | noise penalty |
|---|---|
| 1 % of $n$ | ×1.0050 |
| 5 % of $n$ | ×1.0260 |
| 10 % of $n$ | ×1.0541 |
| 25 % of $n$ | ×1.1547 |
| 50 % of $n$ | ×1.4142 |

$t$-collusion robustness is therefore cheap (10 % collusion costs 5 % noise) and
should be claimed rather than assumed away.

**Implemented** (2026-07-30): `DPConfig.collusion_t` / `--dp-collusion-t`,
default 0. Shares become $\sigma/\sqrt{n-t}$ and `noise_std_on_average` carries
the $\sqrt{n/(n-t)}$ factor; $t \ge n$ raises rather than returning a vacuous
guarantee. `tests/test_dp_utils.py` pins that $t = 0$ is bit-identical to the
pre-change code, so every logged result stands unchanged.

## 6. What we explicitly do not claim

Each of these is a reviewer trap; the paper states them in a "scope and
limitations" subsection rather than leaving them to be found.

1. **No amplification-by-shuffling on top.** The mechanism already stands in the
   central model — A1 gives the server only the sum, with residual noise
   $\sigma/n$. Applying shuffling amplification as well would count the same
   anonymity twice. Write this sentence affirmatively.
2. **Subsampling is not a free win here.** Residual noise is $\sigma/n$ with
   $n = qN$; shrinking $q$ shrinks $\sigma$ and $n$ together. At fixed
   population 100, $q = 0.1$ is **2.29× worse**. Only "population ≫ needed
   cohort" pays (fixed $n{=}1000$, $N{=}10^4$, $q{=}0.1$ → 0.23× noise). The
   correct narrative is "10 000 population, $q{=}0.1$, 1000 report per round",
   never "10 of 100 clients".
3. **The LDP + shuffling + subsampling route is quantitatively dead.** The clones
   bound needs $\varepsilon_0 \le \log(n / 16\log(2/\delta))$: at $n{=}100$ the
   theorem does not apply at all; at $n{=}1000$ the best case
   $\varepsilon_0 = 1.63$ yields residual noise 0.649 = **15× ours**. Corroborated
   by BennettExact measurements. Our label is **distributed DP**, not LDP.
4. **No modular arithmetic / secure aggregation in the implementation.** Sums are
   int64; a real deployment doing modular secure aggregation must size the
   modulus $M$ against overflow, which we do not evaluate.
5. **No robustness to malicious clients** (A4), and **no claim that 100 clients
   at $\varepsilon = 1$ is viable** — that configuration needs $\varepsilon
   \approx 41$ at $K{=}20, T{=}3$, or $\approx 3.6$ with every lever pulled
   ($B{=}1$ cluster per round, $T{=}1$). Structural, not a hyperparameter issue.

## 7. Remaining gaps before submission

- [x] `collusion_t` (Corollary 2), default 0, with tests — done 2026-07-30.
- [x] Choose the $t$ actually claimed in the paper and re-run at that setting —
      done 2026-09-17 (notes §45, `run_collusion_t10.sh`): $t = 0.1n$
      (`--dp-collusion-t 10`, ×1.054 residual noise), CIFAR-10 α=0.1, K=10,
      3 seeds paired against the $t=0$ runs. ε=0.1: all three seeds
      bit-identical (no cluster label flips). ε=0.025: two seeds identical,
      one seed −3.14 pt (one cluster flipped), mean −1.05 ± 1.81 pt, p=0.42.
      Claim $t = 0.1n$ in the paper; the main tables can stay at $t=0$ with
      this sentence in the caption. Do not write "zero cost".
- [~] MIA on existing checkpoints — `mia_attack.py` written and smoke-tested;
      `run_mia.sh` queued behind the FedCT runs. Two setup decisions that make or
      break the numbers: non-members are **label-matched per client** (otherwise
      Dirichlet(0.1) skew alone scores a high AUC while inferring nothing), and
      the 2000 proxy images are **excluded from every non-member pool** (clients
      distil on them each round, so they are trained-on). Smoke test on 3 clients
      of `cohort10000`: macro AUC 0.559 / TPR@1%FPR 0.015, and members drawn once
      leak less (AUC 0.523) than the overall set — the with-replacement sampling
      creates a duplication-driven leakage axis the DP analysis does not cover.
      Note `loss` and `confidence` are monotone transforms of each other and give
      identical ROC metrics by construction: report them as one attack, not two.
- [ ] DP-FedCT baseline numbers (queue running; §11 of the notes).
- [ ] Second dataset; 3 seeds on the main line.
- [ ] Decide whether the paper's headline accountant is zCDP (all logged results)
      or RDP (17.5 % tighter, currently one seed per cohort). Recommend
      presenting zCDP as the frozen main table and RDP as an accounting ablation,
      since the RDP pairing was measured at +1.2 pt / +0.5 pt / ≈0 for cohorts
      1000 / 3000 / saturated — consistent with "same pseudo-label quality needs
      15–20 % smaller cohort", which is the honest phrasing.

## 8. Regenerating every constant here

```bash
python - <<'PY'
import math, dp_utils as D
K,C,T,eps,delta,b = 20,10,3,1.0,1e-5,16
s = 1 << b
for acct in ("zcdp","rdp"):
    cfg = D.DPConfig(epsilon=eps, delta=delta, q=1.0, num_clusters=K,
                     num_rounds=T, quantization_bits=b, accounting=acct)
    print(acct, "Delta2=%.4f sigma=%.4f rho=%.6f"
          % (cfg.l2_sensitivity, cfg.sigma_aggregate, cfg.rho_total))
    for n in (100, 1000, 10000):
        print("   n=%-6d residual=%.5f share=%.4f"
              % (n, cfg.sigma_aggregate/n, cfg.client_noise_std(n)))
D2 = math.sqrt(2*K)
infl = (s*D2 + math.sqrt(K*C)) / (s*D2)
rho = D.eps_to_zcdp(eps, delta)
sig = D2 / math.sqrt(2*rho/T)
print("integer-grid inflation %.10f -> corrected eps %.8f"
      % (infl, D.zcdp_to_eps(T*(D2*infl)**2/(2*sig**2), delta)))
PY
```
