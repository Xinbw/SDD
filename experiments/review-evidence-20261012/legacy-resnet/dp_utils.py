# dp_utils.py
"""
Distributed differential privacy primitives for SDD.

System model
------------
Clients send their reports through an ANONYMOUS MULTI-MESSAGE SHUFFLER.  Each
client splits its (quantised) report into m uniformly random additive shares
over Z_q and submits them as separate anonymous messages (the "split-and-mix" /
IKOS construction, analysed for DP by Balle-Bell-Gascon-Nissim, CCS 2020, and
Ghazi et al.).  Every individual message is a one-time-pad share and reveals
nothing; the only thing the server can compute is the SUM of all messages.

Consequently the server's view is exactly the aggregate, and the guarantee is a
*central*-model one.  Like the shuffle itself, split-and-mix is a
communication-layer construction: the shares recombine to exactly the same sum,
so it changes no number in the training pipeline and is not simulated here.

Mechanism: distributed discrete Gaussian
----------------------------------------
Neighbouring datasets differ in one client's entire private dataset, so that
client's whole K x C report may change.  Each row is a probability vector, so:

    L2 sensitivity of the report   Delta_2 = sqrt(2K)

(one row can move by at most sqrt(2) in L2; K rows -> sqrt(2K)).  Note this is
sqrt(K)-scaling rather than the 2K of the L1/Laplace route.

Noise is added ONCE to the aggregate, realised in a distributed way: client i
adds an independent share with variance sigma^2 / n, so the n shares sum to the
required sigma^2 on the aggregate.  Because the server only ever sees the sum,
the per-client 1/n share is sufficient — this is what buys the factor n over
per-client (local-DP) noise.

Accounting is via zero-concentrated DP:

    rho_round = Delta_2^2 / (2 * sigma^2)
    rho_total = T * rho_round
    epsilon   = rho_total + 2 * sqrt(rho_total * ln(1/delta))

Discrete (integer) noise is used because the shuffle shares live in Z_q.  Values
are quantised on a 1/quantization_scale grid before noising and de-quantised
after aggregation.

Reference: Kairouz, Liu & Steinke, "The Distributed Discrete Gaussian Mechanism
for Federated Learning with Secure Aggregation", ICML 2021.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

import rdp_accounting

# Which analysis of the mechanism is used to turn (epsilon, delta) into sigma.
# "zcdp" is the default and reproduces every result produced so far; "rdp" is the
# same mechanism analysed with Renyi composition (tighter, and the only mode that
# can credit q < 1).  See rdp_accounting.py.
ACCOUNTING_MODES = ("zcdp", "rdp")

# Above this sigma (in INTEGER units) a rounded continuous Gaussian and the true
# discrete Gaussian N_Z(0, sigma^2) differ in total variation by O(e^{-2 pi^2
# sigma^2}) — below 1e-800 already at sigma = 10 — so rounding is used as the
# sampler. Smaller sigma is rejected rather than silently approximated; the fix
# is to raise --dp-quantization-bits, which scales sigma up linearly.
MIN_INTEGER_SIGMA = 10.0


# ---------------------------------------------------------------------------
# zCDP <-> (epsilon, delta) conversion
# ---------------------------------------------------------------------------

def zcdp_to_eps(rho: float, delta: float) -> float:
    """epsilon of a rho-zCDP mechanism at a given delta."""
    if rho <= 0:
        return 0.0
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def eps_to_zcdp(epsilon: float, delta: float) -> float:
    """Inverse of zcdp_to_eps: the rho whose (epsilon, delta) is the one given."""
    if not (epsilon > 0):
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    a = math.log(1.0 / delta)
    return (math.sqrt(a + epsilon) - math.sqrt(a)) ** 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPConfig:
    """
    epsilon / delta are the (epsilon, delta)-DP guarantee of the ENTIRE training
    run (all num_rounds rounds together), not a per-round or per-report budget.
    """

    epsilon: float
    delta: float
    q: float
    num_clusters: int
    num_rounds: int
    cohort_size: Optional[int] = None      # n used in the privacy/utility calculus
    quantization_bits: int = 16
    accounting: str = "zcdp"               # "zcdp" (frozen default) or "rdp"
    collusion_t: int = 0                   # colluding clients tolerated (A2)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not (self.epsilon > 0):
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not (0.0 < self.delta < 1.0):
            raise ValueError(
                f"delta must be in (0, 1), got {self.delta}. The distributed "
                f"discrete Gaussian is (eps, delta)-DP; delta must be a real value."
            )
        if not (0.0 < self.q <= 1.0):
            raise ValueError(f"q must be in (0, 1], got {self.q}")
        if self.num_clusters < 1:
            raise ValueError(f"num_clusters must be >= 1, got {self.num_clusters}")
        if self.num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {self.num_rounds}")
        if self.cohort_size is not None and self.cohort_size < 1:
            raise ValueError(f"cohort_size must be >= 1, got {self.cohort_size}")
        if not (1 <= self.quantization_bits <= 30):
            raise ValueError(
                f"quantization_bits must be in [1, 30], got {self.quantization_bits}"
            )
        if self.collusion_t < 0:
            raise ValueError(f"collusion_t must be >= 0, got {self.collusion_t}")
        if self.accounting not in ACCOUNTING_MODES:
            raise ValueError(
                f"accounting must be one of {list(ACCOUNTING_MODES)}, "
                f"got {self.accounting!r}"
            )

    # -- mechanism parameters ------------------------------------------------

    @property
    def l2_sensitivity(self) -> float:
        """L2 sensitivity of one client's complete K x C report: sqrt(2K)."""
        return math.sqrt(2.0 * self.num_clusters)

    @property
    def quantization_scale(self) -> int:
        """Values are represented on a 1/scale integer grid."""
        return 1 << self.quantization_bits

    @property
    def rho_total(self) -> float:
        """zCDP budget for the whole run, derived from (epsilon, delta).

        Informational when accounting == "rdp": that mode composes in Renyi DP
        and does not go through rho.
        """
        return eps_to_zcdp(self.epsilon, self.delta)

    @property
    def rho_per_round(self) -> float:
        """zCDP spent per round (uniform split; no subsampling credit taken)."""
        return self.rho_total / self.num_rounds

    @property
    def rdp_order(self) -> Optional[int]:
        """Renyi order the budget is realised at; None outside "rdp" mode."""
        if self.accounting != "rdp":
            return None
        return rdp_accounting.epsilon_from_sigma(
            self.sigma_aggregate, self.delta, self.num_rounds,
            self.l2_sensitivity, q=self.q_credited,
        )[1]

    @property
    def q_credited(self) -> float:
        """Sampling rate the accountant is allowed to credit.

        Only "rdp" can credit it; "zcdp" treats q as merely thinning the cohort.
        """
        return self.q if self.accounting == "rdp" else 1.0

    @property
    def sigma_aggregate(self) -> float:
        """
        Noise std placed on the AGGREGATE SUM each round, in probability units.

        zcdp:  rho_round = Delta_2^2 / (2 sigma^2)
               => sigma = Delta_2 / sqrt(2 rho_round)
        rdp:   smallest sigma whose T-fold (subsampled) Gaussian composition
               meets (epsilon, delta) at the best Renyi order.
        """
        if self.accounting == "rdp":
            return rdp_accounting.sigma_from_epsilon(
                self.epsilon, self.delta, self.num_rounds,
                self.l2_sensitivity, q=self.q_credited,
            )
        return self.l2_sensitivity / math.sqrt(2.0 * self.rho_per_round)

    def effective_cohort(self, num_participants: int) -> int:
        """n used in the privacy/utility calculus for this round."""
        if num_participants < 1:
            raise ValueError("num_participants must be >= 1")
        return self.cohort_size if self.cohort_size is not None else num_participants

    def client_noise_std(self, num_participants: int) -> float:
        """
        Std (probability units) of the discrete-Gaussian share ONE client adds.

        The residual noise on the server's averaged output is
        sigma_aggregate / effective_cohort.  With `num_participants` clients
        actually reporting, the total noise to place on their sum is therefore
        sigma_aggregate * num_participants / effective_cohort, split into
        num_participants independent shares.

        With collusion_t > 0 the shares are inflated so that the noise SURVIVING
        the removal of t disclosed shares still equals the required total: t
        colluding clients subtract their own contributions, leaving
        (num_participants - t) shares.
        """
        n_dp = self.effective_cohort(num_participants)
        sigma_on_sum = self.sigma_aggregate * num_participants / n_dp
        return sigma_on_sum / math.sqrt(self.honest_participants(num_participants))

    def honest_participants(self, num_participants: int) -> int:
        """Shares that survive t-collusion, i.e. the divisor of the share variance."""
        if num_participants < 1:
            raise ValueError("num_participants must be >= 1")
        honest = num_participants - self.collusion_t
        if honest < 1:
            raise ValueError(
                f"collusion_t={self.collusion_t} leaves {honest} honest clients out "
                f"of {num_participants}; the guarantee is vacuous. Require "
                f"collusion_t < num_participants."
            )
        return honest

    def collusion_noise_penalty(self, num_participants: int) -> float:
        """Factor by which t-collusion robustness inflates the residual noise."""
        return math.sqrt(
            num_participants / self.honest_participants(num_participants)
        )

    def noise_std_on_average(self, num_participants: int) -> float:
        """Residual per-coordinate noise std on the aggregated pseudo-label."""
        base = self.sigma_aggregate / self.effective_cohort(num_participants)
        return base * self.collusion_noise_penalty(num_participants)

    def is_simulated_cohort(self, num_participants: int) -> bool:
        return (
            self.cohort_size is not None and self.cohort_size != num_participants
        )

    def summary(self, num_participants: Optional[int] = None) -> str:
        lines = [
            "DP config (multi-message shuffle + distributed discrete Gaussian)",
            "  system model          : anonymous multi-message shuffler; the server's",
            "                          view is the AGGREGATE only (split-and-mix shares)",
            f"  epsilon (whole run)   : {self.epsilon:.6f}",
            f"  delta                 : {self.delta:.2e}",
            f"  rounds T              : {self.num_rounds}",
            f"  clusters K            : {self.num_clusters}",
            f"  L2 sensitivity        : sqrt(2K) = {self.l2_sensitivity:.4f}",
            f"  accounting            : {self.accounting}",
        ]
        if self.accounting == "rdp":
            # rho is not how this mode reaches sigma; printing it next to a
            # tighter epsilon would read as if the two agreed.
            lines.append(f"  optimal RDP order     : {self.rdp_order}")
        else:
            lines += [
                f"  rho (zCDP) total      : {self.rho_total:.6g}",
                f"  rho per round         : {self.rho_per_round:.6g}",
            ]
        lines += [
            f"  sigma on aggregate    : {self.sigma_aggregate:.4f}  (probability units)",
            f"  quantization          : {self.quantization_bits} bits "
            f"(grid 1/{self.quantization_scale})",
            f"  q                     : {self.q}",
            f"  collusion tolerance t : {self.collusion_t}",
        ]
        if self.collusion_t > 0 and num_participants is not None:
            lines.append(
                f"  collusion noise cost  : x"
                f"{self.collusion_noise_penalty(num_participants):.4f} "
                f"(shares sized for {self.honest_participants(num_participants)} "
                f"surviving contributors)"
            )
        if self.accounting == "rdp":
            sigma_zcdp = self.l2_sensitivity / math.sqrt(2.0 * self.rho_per_round)
            gain = 100.0 * (1.0 - self.sigma_aggregate / sigma_zcdp)
            source = (
                "tighter composition alone"
                if self.q_credited >= 1.0
                else f"tighter composition AND the q={self.q} subsampling credit"
            )
            lines.append(
                f"  NOTE: Renyi composition; the zCDP route would demand "
                f"sigma={sigma_zcdp:.4f} for the same epsilon ({gain:.1f}% more "
                f"noise). Gain comes from {source}."
            )
        if self.q < 1.0 and self.accounting != "rdp":
            lines.append(
                "  NOTE: subsampling amplification is NOT credited (conservative); "
                "q only reduces the number of reporting clients."
            )
        if self.q < 1.0 and self.accounting == "rdp":
            lines += [
                "",
                "  *** SUBSAMPLING AMPLIFICATION CREDITED ***",
                f"  The reported epsilon={self.epsilon:g} credits Poisson sampling at "
                f"q={self.q}, which is",
                "  claimable ONLY under secrecy of the sample: clients must flip their",
                "  own coin and report through the anonymous shuffler, so the server",
                "  never learns who was selected. State this assumption when reporting.",
            ]
        if num_participants is not None:
            n_dp = self.effective_cohort(num_participants)
            lines += [
                f"  participants this run : {num_participants}",
                f"  cohort used for DP    : {n_dp}",
                f"  per-client share std  : {self.client_noise_std(num_participants):.6f}",
                f"  noise std on average  : "
                f"{self.noise_std_on_average(num_participants):.6f}",
            ]
            if self.is_simulated_cohort(num_participants):
                lines += [
                    "",
                    "  *** SIMULATED COHORT ***",
                    f"  The reported epsilon={self.epsilon:g} holds for a deployment with "
                    f"{n_dp} reporting clients.",
                    f"  This run has only {num_participants}; the injected noise matches "
                    f"what a {n_dp}-client",
                    "  cohort would leave after averaging, but the aggregate itself is "
                    "still estimated",
                    f"  from {num_participants} clients. Report this as a UTILITY "
                    "SIMULATION of that deployment.",
                ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discrete Gaussian noise
# ---------------------------------------------------------------------------

def sample_discrete_gaussian(
    shape,
    sigma: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample integer noise from the discrete Gaussian N_Z(0, sigma^2).

    `sigma` is in INTEGER units and must be >= MIN_INTEGER_SIGMA, where rounding a
    continuous Gaussian is within total variation O(exp(-2 pi^2 sigma^2)) of the
    exact discrete Gaussian (Poisson-summation / theta-function identity) — far
    below floating-point resolution.  For smaller sigma the caller should raise
    the quantisation scale instead of accepting an approximation.

    Returns:
        int64 tensor of the requested shape.
    """
    if not (sigma > 0):
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if sigma < MIN_INTEGER_SIGMA:
        raise ValueError(
            f"integer sigma={sigma:.4f} is below MIN_INTEGER_SIGMA="
            f"{MIN_INTEGER_SIGMA}; increase --dp-quantization-bits (sigma scales "
            f"linearly with the quantisation scale) instead of approximating."
        )
    normal = torch.randn(shape, device=device, dtype=torch.float64, generator=generator)
    return torch.round(normal * sigma).to(torch.int64)


def quantize(values: torch.Tensor, scale: int) -> torch.Tensor:
    """Map float values onto the 1/scale integer grid."""
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    return torch.round(values.to(torch.float64) * scale).to(torch.int64)


def dequantize(values: torch.Tensor, scale: int, dtype=torch.float32) -> torch.Tensor:
    """Inverse of quantize()."""
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    return (values.to(torch.float64) / scale).to(dtype)


def add_distributed_discrete_gaussian(
    values: torch.Tensor,
    client_noise_std: float,
    quantization_scale: int,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    One client's contribution: quantise its report and add its 1/n noise share.

    The result is an INTEGER tensor meant to be summed with the other clients'
    contributions (in the real protocol, via anonymous split-and-mix shares).
    An individual output is NOT differentially private on its own — only the
    aggregate is, which is why the server must never see it.

    Args:
        values:             (K, C) float tensor of cluster probabilities.
        client_noise_std:   Share std in PROBABILITY units (DPConfig.client_noise_std).
        quantization_scale: Integer grid resolution.
    Returns:
        int64 tensor of the same shape.
    """
    if not torch.is_tensor(values):
        raise TypeError(f"values must be a torch.Tensor, got {type(values)!r}")
    if not values.is_floating_point():
        raise TypeError(f"values must be floating point, got {values.dtype}")
    if not (client_noise_std > 0):
        raise ValueError(f"client_noise_std must be > 0, got {client_noise_std}")

    quantized = quantize(values, quantization_scale)
    sigma_int = client_noise_std * quantization_scale
    if sigma_int < MIN_INTEGER_SIGMA:
        needed = math.ceil(math.log2(MIN_INTEGER_SIGMA / client_noise_std))
        raise ValueError(
            f"The per-client noise share is too small to represent on the current "
            f"integer grid: sigma={client_noise_std:.3g} x 2^"
            f"{int(round(math.log2(quantization_scale)))} = {sigma_int:.3g} < "
            f"{MIN_INTEGER_SIGMA}. Set --dp-quantization-bits to at least {needed}. "
            f"(This happens when epsilon is very large or the cohort is very big — "
            f"the noise is then finer than the quantisation grid.)"
        )
    noise = sample_discrete_gaussian(
        tuple(values.shape),
        sigma_int,
        device=device if device is not None else values.device,
        generator=generator,
    )
    return quantized + noise.to(quantized.device)


def required_modulus_bits(
    num_participants: int,
    client_noise_std: float,
    quantization_scale: int,
    num_sigmas: float = 8.0,
) -> int:
    """
    Bits needed for the Z_q shares so the aggregate never wraps around.

    Reported for the protocol description; the training code sums in int64 and
    does not perform modular reduction.
    """
    max_signal = num_participants * quantization_scale
    noise_span = num_sigmas * client_noise_std * quantization_scale * math.sqrt(
        max(num_participants, 1)
    )
    total = 2.0 * (max_signal + noise_span)          # signed range
    return max(1, int(math.ceil(math.log2(total))))


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def project_rows_to_simplex(values: torch.Tensor) -> torch.Tensor:
    """
    Euclidean projection of every row of a (N, C) tensor onto the probability simplex.

    Standard sorting-based algorithm (Duchi et al., 2008):
        1. sort each row descending      -> u
        2. cssv = cumsum(u) - 1
        3. rho  = max { j : u_j - cssv_j / (j+1) > 0 }
        4. theta = cssv[rho] / (rho + 1)
        5. return clamp(values - theta, min=0)

    Applied by the SERVER once, after aggregation — projecting per client would be
    a non-linear step that breaks the additivity the noise shares rely on.
    The input tensor is never modified in place.  Works on CPU and CUDA.
    """
    if not torch.is_tensor(values):
        raise TypeError(f"values must be a torch.Tensor, got {type(values)!r}")
    if values.dim() != 2:
        raise ValueError(f"values must be 2-D (N, C), got shape {tuple(values.shape)}")
    if not values.is_floating_point():
        raise TypeError(f"values must be a floating-point tensor, got {values.dtype}")
    n_rows, n_cols = values.shape
    if n_rows == 0:
        return values.clone()
    if n_cols == 0:
        raise ValueError("Cannot project onto a 0-dimensional simplex.")

    work = values.detach().to(torch.float64)
    u, _ = torch.sort(work, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - 1.0
    arange = torch.arange(
        1, n_cols + 1, device=values.device, dtype=torch.float64
    ).unsqueeze(0)
    cond = (u - cssv / arange) > 0
    # rho = index of the last True in each row (there is always at least one:
    # j=0 gives u_0 - (u_0 - 1) = 1 > 0).
    rho = n_cols - 1 - cond.flip(dims=[1]).to(torch.int64).argmax(dim=1)
    theta = cssv.gather(1, rho.unsqueeze(1)) / (rho.unsqueeze(1).to(torch.float64) + 1.0)
    projected = (work - theta).clamp(min=0.0)
    return projected.to(values.dtype)


# ---------------------------------------------------------------------------
# Legacy local-DP primitive
# ---------------------------------------------------------------------------

def apply_laplace_noise(
    values: torch.Tensor,
    epsilon_local: float,
    l1_sensitivity: float,
) -> torch.Tensor:
    """
    Local (per-client) Laplace mechanism — retained for the ablation entry point,
    which still runs the per-sample local-DP path.  The main training pipeline
    uses the distributed discrete Gaussian above.

    Adds i.i.d. Laplace(0, l1_sensitivity / epsilon_local) to every coordinate.
    """
    if not torch.is_tensor(values):
        raise TypeError(f"values must be a torch.Tensor, got {type(values)!r}")
    if not values.is_floating_point():
        raise TypeError(f"values must be a floating-point tensor, got {values.dtype}")
    if not (epsilon_local > 0):
        raise ValueError(f"epsilon_local must be > 0, got {epsilon_local}")
    if not (l1_sensitivity > 0):
        raise ValueError(f"l1_sensitivity must be > 0, got {l1_sensitivity}")

    scale = float(l1_sensitivity) / float(epsilon_local)
    uniform = torch.rand(values.shape, dtype=values.dtype, device=values.device)
    uniform = uniform.sub_(0.5).clamp_(-0.5 + 1e-7, 0.5 - 1e-7)
    noise = -scale * torch.sign(uniform) * torch.log1p(-2.0 * uniform.abs())

    noisy = values + noise
    if not torch.isfinite(noisy).all():
        raise FloatingPointError(
            "apply_laplace_noise produced non-finite values "
            f"(epsilon_local={epsilon_local}, l1_sensitivity={l1_sensitivity})."
        )
    return noisy


def log_dp_settings(dp_config: DPConfig, num_participants: Optional[int] = None):
    print(dp_config.summary(num_participants))
