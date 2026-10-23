"""
privacy_accounting.py
(epsilon, delta)-DP accounting for the SDD distributed discrete Gaussian
mechanism under the anonymous multi-message shuffle model.

System model
------------
Clients submit their (quantised) reports as m uniformly random additive shares
over Z_q through an anonymous shuffler ("split-and-mix" / IKOS; analysed for DP
by Balle-Bell-Gascon-Nissim CCS 2020 and Ghazi et al.).  Each individual message
is a one-time-pad share, so the ONLY quantity the server can recover is the sum
of all messages.  The server's view is therefore the aggregate, and the analysis
below is a central-model one -- there is no local-DP noise penalty and no need
for shuffle *amplification* (the shuffler is used to build secure aggregation,
not to amplify a local randomizer).

Mechanism
---------
Neighbouring datasets differ in one client's whole private dataset, so that
client's entire K x C report may change:

    L2 sensitivity      Delta_2 = sqrt(2K)

Noise of std sigma is placed on the AGGREGATE once, realised distributively:
each of the n reporting clients adds an independent discrete Gaussian share of
variance sigma^2 / n.  The residual noise on the server's averaged pseudo-label
is therefore sigma / n -- a factor n better than per-client (local-DP) noise,
which would leave sigma / sqrt(n).

Accounting
----------
Two analyses of the SAME mechanism are available; `accounting` selects one.

  "zcdp" (default, and what every result so far was produced with)

    rho_round = Delta_2^2 / (2 sigma^2)
    rho_total = T * rho_round                       (composition over T rounds)
    epsilon   = rho_total + 2 sqrt(rho_total ln(1/delta))

    Subsampling (q < 1) is NOT credited: it is treated conservatively as merely
    reducing the number of reporting clients.

  "rdp"  (tighter; see rdp_accounting.py)

    Renyi DP composition with the order optimised after composition.  At q = 1
    this is the identical mechanism analysed better -- roughly 20% less noise at
    the same epsilon, no new assumption.  At q < 1 it additionally credits
    Poisson-subsampling amplification, which DOES add an assumption: secrecy of
    the sample.  See the warning in rdp_accounting.py.

Reference: Kairouz, Liu & Steinke, "The Distributed Discrete Gaussian Mechanism
for Federated Learning with Secure Aggregation", ICML 2021.

CLI
---
    # forward: budget -> noise level
    python privacy_accounting.py --epsilon-total 1.0 --rounds 6 --clusters 20 \\
        --cohort-size 10000

    # same budget, tighter accountant
    python privacy_accounting.py --epsilon-total 1.0 --rounds 6 --clusters 20 \\
        --cohort-size 10000 --accounting rdp

    # inverse: required utility -> budget
    python privacy_accounting.py --target-noise-std 0.03 --rounds 6 \\
        --clusters 20 --cohort-size 10000
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Optional

import rdp_accounting
from dp_utils import ACCOUNTING_MODES, eps_to_zcdp, zcdp_to_eps


ACCOUNTING_MODE = "distributed_discrete_gaussian_multimessage_shuffle"


def _check_mode(accounting: str) -> str:
    if accounting not in ACCOUNTING_MODES:
        raise ValueError(
            f"accounting must be one of {sorted(ACCOUNTING_MODES)}, got {accounting!r}"
        )
    return accounting


def l2_sensitivity(num_clusters: int) -> float:
    """L2 sensitivity of one client's complete K x C report: sqrt(2K)."""
    if num_clusters < 1:
        raise ValueError(f"num_clusters must be >= 1, got {num_clusters}")
    return math.sqrt(2.0 * num_clusters)


def sigma_from_epsilon(
    epsilon_total: float,
    delta: float,
    num_rounds: int,
    num_clusters: int,
    accounting: str = "zcdp",
    q: float = 1.0,
) -> float:
    """
    Noise std placed on the aggregate each round, in probability units.

    `q` is only read in "rdp" mode, where it credits Poisson-subsampling
    amplification; "zcdp" ignores it (conservative).
    """
    if _check_mode(accounting) == "rdp":
        return rdp_accounting.sigma_from_epsilon(
            epsilon_total, delta, num_rounds, l2_sensitivity(num_clusters), q=q
        )
    rho_round = eps_to_zcdp(epsilon_total, delta) / num_rounds
    return l2_sensitivity(num_clusters) / math.sqrt(2.0 * rho_round)


def epsilon_from_sigma(
    sigma: float,
    delta: float,
    num_rounds: int,
    num_clusters: int,
    accounting: str = "zcdp",
    q: float = 1.0,
) -> float:
    """Inverse of sigma_from_epsilon."""
    if not (sigma > 0):
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if _check_mode(accounting) == "rdp":
        return rdp_accounting.epsilon_from_sigma(
            sigma, delta, num_rounds, l2_sensitivity(num_clusters), q=q
        )[0]
    rho_round = l2_sensitivity(num_clusters) ** 2 / (2.0 * sigma ** 2)
    return zcdp_to_eps(num_rounds * rho_round, delta)


def compute_total_privacy(
    epsilon_total: Optional[float],
    num_rounds: int,
    num_clusters: int,
    cohort_size: int,
    delta: float = 1e-5,
    target_noise_std: Optional[float] = None,
    subsampling_rate: float = 1.0,
    accounting: str = "zcdp",
) -> Dict[str, object]:
    """
    Either direction of the accounting, depending on which of `epsilon_total` /
    `target_noise_std` is supplied.

    Args:
        epsilon_total:    (eps, delta)-DP budget for the whole run. Give this OR
                          target_noise_std.
        num_rounds:       T.
        num_clusters:     K.
        cohort_size:      n, the number of clients reporting per round. This is
                          what turns sigma on the aggregate into the residual
                          noise sigma / n on the pseudo-label.
        delta:            delta of the guarantee.
        target_noise_std: Required per-coordinate noise std on the aggregated
                          pseudo-label. Give this to solve for epsilon.
        subsampling_rate: q. Recorded only in "zcdp" mode; credited as
                          Poisson-subsampling amplification in "rdp" mode.
        accounting:       "zcdp" (default, matches every published run) or
                          "rdp" (tighter; see rdp_accounting.py).
    """
    _check_mode(accounting)
    if (epsilon_total is None) == (target_noise_std is None):
        raise ValueError(
            "Give exactly one of epsilon_total / target_noise_std."
        )
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")
    if cohort_size < 1:
        raise ValueError(f"cohort_size must be >= 1, got {cohort_size}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    q_credited = subsampling_rate if accounting == "rdp" else 1.0
    if epsilon_total is not None:
        sigma = sigma_from_epsilon(
            epsilon_total, delta, num_rounds, num_clusters, accounting, q_credited
        )
        eps = epsilon_total
        direction = "epsilon -> noise"
    else:
        sigma = target_noise_std * cohort_size
        eps = epsilon_from_sigma(
            sigma, delta, num_rounds, num_clusters, accounting, q_credited
        )
        direction = "noise -> epsilon"

    rho_round = l2_sensitivity(num_clusters) ** 2 / (2.0 * sigma ** 2)
    if accounting == "rdp":
        _, best_order = rdp_accounting.epsilon_from_sigma(
            sigma, delta, num_rounds, l2_sensitivity(num_clusters), q=q_credited
        )
        # What the same sigma would have been reported as by the loose route,
        # so the gain from the tighter analysis is visible in the report.
        sigma_zcdp = sigma_from_epsilon(eps, delta, num_rounds, num_clusters, "zcdp")
    else:
        best_order = None
        sigma_zcdp = sigma
    return {
        "accounting_mode":        ACCOUNTING_MODE,
        "accounting":             accounting,
        "rdp_order":              best_order,
        "sigma_zcdp_equivalent":  sigma_zcdp,
        "direction":              direction,
        "num_clusters":           num_clusters,
        "l2_sensitivity":         l2_sensitivity(num_clusters),
        "num_rounds":             num_rounds,
        "cohort_size":            cohort_size,
        "subsampling_rate":       subsampling_rate,
        "delta":                  delta,
        "rho_per_round":          rho_round,
        "rho_total":              num_rounds * rho_round,
        "sigma_on_aggregate":     sigma,
        "client_share_std":       sigma / math.sqrt(cohort_size),
        "noise_std_on_average":   sigma / cohort_size,
        "total_epsilon":          eps,
        "total_delta":            delta,
        "shuffle_role":           "secure aggregation via split-and-mix "
                                  "(NOT amplification of a local randomizer)",
        "subsampling_amplification": (
            "credited (Poisson RDP; requires secrecy of the sample)"
            if q_credited < 1.0 else "not_credited"
        ),
    }


def print_privacy_report(result: Dict[str, object]) -> None:
    sep = "=" * 70
    print(sep)
    print("  Privacy Accounting  (distributed discrete Gaussian, multi-message shuffle)")
    print(sep)
    print(f"  direction                 : {result['direction']}")
    print(f"  accounting                : {result['accounting']}")
    print(f"  clusters K                : {result['num_clusters']}")
    print(f"  L2 sensitivity sqrt(2K)   : {result['l2_sensitivity']:.6g}")
    print(f"  rounds T                  : {result['num_rounds']}")
    print(f"  cohort size n             : {result['cohort_size']}")
    print(f"  subsampling rate q        : {result['subsampling_rate']}")
    print()
    if result["accounting"] == "rdp":
        # The zCDP rho of this sigma is deliberately NOT printed here: it would
        # sit next to a tighter epsilon and read as if the two were consistent.
        print(f"  optimal RDP order alpha   : {result['rdp_order']}")
        zc = result["sigma_zcdp_equivalent"]
        sg = result["sigma_on_aggregate"]
        print(f"  sigma the zCDP route needs: {zc:.6g}  "
              f"({100.0 * (1.0 - sg / zc):.1f}% more noise)")
    else:
        print(f"  rho per round (zCDP)      : {result['rho_per_round']:.6g}")
        print(f"  rho total (zCDP)          : {result['rho_total']:.6g}")
    print(f"  sigma on aggregate        : {result['sigma_on_aggregate']:.6g}")
    print(f"  per-client share std      : {result['client_share_std']:.6g}")
    print(f"  noise std on pseudo-label : {result['noise_std_on_average']:.6g}")
    print()
    print(f"  TOTAL epsilon             : {result['total_epsilon']:.6g}")
    print(f"  TOTAL delta               : {result['total_delta']:.2e}")
    print(f"  shuffler's role           : {result['shuffle_role']}")
    print(f"  subsampling amplification : {result['subsampling_amplification']}")
    print(sep)


def print_scaling_table(
    num_rounds: int,
    num_clusters: int,
    delta: float,
    target_noise_std: float,
    accounting: str = "zcdp",
    subsampling_rate: float = 1.0,
) -> None:
    """Total epsilon as a function of the deployment cohort size."""
    print()
    print(f"  Scaling with cohort size  (K={num_clusters}, T={num_rounds}, "
          f"delta={delta:.0e}, target noise std={target_noise_std}, "
          f"accounting={accounting})")
    print(f"  {'cohort n':>10s} {'sigma on agg':>14s} {'TOTAL epsilon':>14s}")
    for n in (100, 500, 1_000, 5_000, 10_000, 50_000, 100_000):
        r = compute_total_privacy(
            None, num_rounds, num_clusters, n, delta,
            target_noise_std=target_noise_std,
            subsampling_rate=subsampling_rate,
            accounting=accounting,
        )
        print(f"  {n:10d} {r['sigma_on_aggregate']:14.2f} {r['total_epsilon']:14.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Privacy accounting: distributed discrete Gaussian under the "
            "multi-message shuffle (aggregate-only) model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epsilon-total", type=float, default=None,
                        help="Budget for the whole run. Give this or --target-noise-std.")
    parser.add_argument("--target-noise-std", type=float, default=None,
                        help="Required noise std on the aggregated pseudo-label.")
    parser.add_argument("--rounds", type=int, required=True, help="T.")
    parser.add_argument("--clusters", type=int, default=20, help="K.")
    parser.add_argument("--cohort-size", type=int, required=True,
                        help="n: clients reporting per round.")
    parser.add_argument("--q", type=float, default=1.0,
                        help="Client subsampling rate. Recorded but NOT credited "
                             "under --accounting zcdp; credited as Poisson "
                             "amplification under --accounting rdp.")
    parser.add_argument("--accounting", choices=sorted(ACCOUNTING_MODES),
                        default="zcdp",
                        help="zcdp: closed-form composition, matches every run "
                             "produced so far. rdp: same mechanism, tighter "
                             "analysis (~20%% less noise at fixed epsilon).")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--scaling-table", action="store_true", default=False,
                        help="Also print total epsilon vs cohort size.")
    args = parser.parse_args()

    if (args.epsilon_total is None) == (args.target_noise_std is None):
        parser.error("give exactly one of --epsilon-total / --target-noise-std")

    result = compute_total_privacy(
        epsilon_total=args.epsilon_total,
        num_rounds=args.rounds,
        num_clusters=args.clusters,
        cohort_size=args.cohort_size,
        delta=args.delta,
        target_noise_std=args.target_noise_std,
        subsampling_rate=args.q,
        accounting=args.accounting,
    )
    print_privacy_report(result)
    if args.scaling_table:
        print_scaling_table(
            args.rounds, args.clusters, args.delta,
            args.target_noise_std or result["noise_std_on_average"],
            accounting=args.accounting,
            subsampling_rate=args.q,
        )


if __name__ == "__main__":
    main()
