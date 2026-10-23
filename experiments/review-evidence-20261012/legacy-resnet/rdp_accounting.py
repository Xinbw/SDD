"""
rdp_accounting.py
Renyi DP accountant for the SDD distributed discrete Gaussian, with optional
credit for Poisson client subsampling.

Why this exists
---------------
The zCDP route in `dp_utils` converts the whole run through the closed form

    epsilon = rho + 2 sqrt(rho ln(1/delta))

which is convenient but loose: it fixes one implicit Renyi order instead of
optimising over them.  Accounting the SAME mechanism in RDP and taking the best
order gives a strictly smaller epsilon -- equivalently, ~20% less noise at a
fixed epsilon.  Nothing about the mechanism, the trust model or the sensitivity
changes; only the analysis is tightened.

Two regimes are supported:

  q = 1  : plain (unsubsampled) Gaussian on the aggregate.
           RDP(alpha) = alpha / (2 z^2),  z = sigma / Delta_2.

  q < 1  : Poisson-subsampled Gaussian (Mironov, Talwar & Zhang 2019), integer
           orders only, via the binomial expansion

               RDP(alpha) = 1/(alpha-1) *
                   log sum_{k=0}^{alpha} C(alpha,k) (1-q)^{alpha-k} q^k
                              exp(k(k-1) / (2 z^2))

           At q = 1 the sum collapses to its k = alpha term and the expression
           reduces exactly to alpha / (2 z^2), so the two regimes agree.

RDP -> (eps, delta) uses the improved conversion of Canonne, Kamath & Steinke
(also in Balle et al. 2020):

    eps(alpha) = RDP(alpha) + log((alpha-1)/alpha)
                 - (log delta + log alpha) / (alpha - 1)

minimised over the order grid.

WARNING -- crediting q < 1 is NOT free
--------------------------------------
Subsampling amplification requires SECRECY OF THE SAMPLE: the adversary must not
learn which clients were selected.  A server that picks the cohort itself knows
the sample and gets no amplification.  It is claimable here only because clients
flip their own Poisson coin and report through the anonymous shuffler, so the
server sees a bag of messages and not an identified participant list.  That is a
real extra assumption and must be stated wherever a q < 1 epsilon is reported.

Note also that amplification is close to a wash for SDD's utility: the residual
noise on the pseudo-label is sigma / n with n = q * N, so shrinking q shrinks the
required sigma and the divisor n by roughly the same factor.  Subsampling pays
off only when the population N is much larger than the cohort you need.

Reference: Mironov, Talwar & Zhang, "Renyi Differential Privacy of the Sampled
Gaussian Mechanism", 2019.  Canonne, Kamath & Steinke, "The Discrete Gaussian
for Differential Privacy", NeurIPS 2020 (conversion).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence, Tuple

# Integer orders only: the subsampled-Gaussian bound above is a binomial
# expansion and is defined for integer alpha.  The grid is wide because the
# optimal order grows as the budget tightens (small epsilon -> large alpha).
DEFAULT_ORDERS: Tuple[int, ...] = tuple(range(2, 513))


def _log_binom(n: int, k: int) -> float:
    """log C(n, k) via lgamma, so alpha = 512 does not build huge integers."""
    return (
        math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)
    )


def rdp_gaussian(alpha: float, z: float) -> float:
    """RDP of the unsubsampled Gaussian at order alpha. z = sigma / Delta_2."""
    if not (z > 0):
        raise ValueError(f"noise multiplier z must be > 0, got {z}")
    if alpha <= 1:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    return alpha / (2.0 * z * z)


def rdp_sampled_gaussian(alpha: int, q: float, z: float) -> float:
    """
    RDP of the Poisson-subsampled Gaussian at INTEGER order alpha.

    Args:
        alpha: integer order > 1.
        q:     Poisson sampling rate in (0, 1].
        z:     noise multiplier sigma / Delta_2, where sigma is the std placed on
               the aggregate of the SAMPLED clients.
    """
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")
    if not (z > 0):
        raise ValueError(f"noise multiplier z must be > 0, got {z}")
    if int(alpha) != alpha or alpha < 2:
        raise ValueError(f"alpha must be an integer >= 2, got {alpha}")
    alpha = int(alpha)
    if q == 1.0:
        # The k = alpha term is the only survivor; kept explicit because
        # log(1 - q) is -inf.
        return rdp_gaussian(alpha, z)

    log_terms = []
    for k in range(alpha + 1):
        log_term = _log_binom(alpha, k) + (alpha - k) * math.log1p(-q)
        if k:
            log_term += k * math.log(q) + k * (k - 1) / (2.0 * z * z)
        log_terms.append(log_term)
    top = max(log_terms)
    log_sum = top + math.log(sum(math.exp(t - top) for t in log_terms))
    return log_sum / (alpha - 1)


def rdp_to_epsilon(rdp: float, alpha: float, delta: float) -> float:
    """
    Convert an RDP guarantee at one order into (eps, delta)-DP.

    Canonne-Kamath-Steinke conversion, which is never worse than the classical
    rdp + log(1/delta)/(alpha-1).
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if alpha <= 1:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    eps = (
        rdp
        + math.log((alpha - 1.0) / alpha)
        - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
    )
    return max(eps, 0.0)


def epsilon_from_noise_multiplier(
    z: float,
    q: float,
    num_rounds: int,
    delta: float,
    orders: Sequence[int] = DEFAULT_ORDERS,
) -> Tuple[float, int]:
    """
    Total epsilon of `num_rounds` compositions of the (subsampled) Gaussian.

    RDP composes by addition at each order; the best order is chosen after
    composition, which is where the tightening over zCDP comes from.

    Returns:
        (epsilon, optimal order).
    """
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")
    best_eps, best_alpha = math.inf, orders[0]
    for alpha in orders:
        eps = rdp_to_epsilon(num_rounds * rdp_sampled_gaussian(alpha, q, z),
                             alpha, delta)
        if eps < best_eps:
            best_eps, best_alpha = eps, alpha
    return best_eps, best_alpha


@lru_cache(maxsize=256)
def noise_multiplier_from_epsilon(
    epsilon: float,
    q: float,
    num_rounds: int,
    delta: float,
    orders: Sequence[int] = DEFAULT_ORDERS,
    tol: float = 1e-9,
) -> float:
    """
    Smallest noise multiplier z = sigma / Delta_2 meeting (epsilon, delta) over
    `num_rounds` rounds.  epsilon is decreasing in z, so bisect in log space.

    Cached: the q < 1 search is ~1e7 float ops, and DPConfig.sigma_aggregate is a
    property that gets read once per report.
    """
    if not (epsilon > 0):
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    lo, hi = 1e-3, 1e3
    while epsilon_from_noise_multiplier(hi, q, num_rounds, delta, orders)[0] > epsilon:
        hi *= 4.0
        if hi > 1e12:
            raise RuntimeError(
                f"no noise multiplier below 1e12 reaches epsilon={epsilon} "
                f"(q={q}, T={num_rounds}, delta={delta})"
            )
    while hi / lo > 1.0 + tol:
        mid = math.sqrt(lo * hi)
        if epsilon_from_noise_multiplier(mid, q, num_rounds, delta, orders)[0] > epsilon:
            lo = mid
        else:
            hi = mid
    return hi


def sigma_from_epsilon(
    epsilon: float,
    delta: float,
    num_rounds: int,
    l2_sensitivity: float,
    q: float = 1.0,
    orders: Sequence[int] = DEFAULT_ORDERS,
) -> float:
    """Noise std on the aggregate, in probability units (z * Delta_2)."""
    if not (l2_sensitivity > 0):
        raise ValueError(f"l2_sensitivity must be > 0, got {l2_sensitivity}")
    z = noise_multiplier_from_epsilon(epsilon, q, num_rounds, delta, orders)
    return z * l2_sensitivity


def epsilon_from_sigma(
    sigma: float,
    delta: float,
    num_rounds: int,
    l2_sensitivity: float,
    q: float = 1.0,
    orders: Sequence[int] = DEFAULT_ORDERS,
) -> Tuple[float, int]:
    """Inverse of `sigma_from_epsilon`; returns (epsilon, optimal order)."""
    if not (sigma > 0):
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not (l2_sensitivity > 0):
        raise ValueError(f"l2_sensitivity must be > 0, got {l2_sensitivity}")
    return epsilon_from_noise_multiplier(
        sigma / l2_sensitivity, q, num_rounds, delta, orders
    )
