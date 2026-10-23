"""Exact discrete Gaussian shares and explicit distributed zCDP certificate.

References: Canonne et al. (2020), Algorithm 3; Kairouz et al. (2021),
Proposition 10. Experimental replay keys are deterministic; secret keys are
drawn from the OS when no replay seed is provided.
"""
from pathlib import Path
import ctypes
import hashlib
import math
import secrets
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
LIB = ROOT / 'exact_dg.so'


def compile_sampler():
    source = ROOT / 'exact_dg.cpp'
    if not LIB.exists() or LIB.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(['g++', '-O3', '-std=c++17', '-fopenmp', '-shared',
                        '-fPIC', '-I'+str(ROOT/'vendor/usr/include'), str(source),
                        '-lcrypto', '-o', str(LIB)], check=True)


def draw(shape, sigma_integer, seed=None, threads=16):
    compile_sampler()
    lib = ctypes.CDLL(str(LIB))
    lib.dg_fill.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                           ctypes.c_void_p, ctypes.c_int]
    lib.dg_fill.restype = ctypes.c_int
    key = secrets.token_bytes(32) if seed is None else hashlib.sha256(
        f'SDD-exact-DG-replay-v1:{seed}'.encode()).digest()
    key_buffer = ctypes.create_string_buffer(key)
    output = np.empty(shape, dtype=np.int64)
    code = lib.dg_fill(output.ctypes.data, output.size, int(sigma_integer),
                       ctypes.cast(key_buffer, ctypes.c_void_p), threads)
    if code:
        raise RuntimeError(f'Exact Gaussian sampler rejected parameters: code={code}')
    return output


def certificate(epsilon, delta, rounds, n, clusters=None, dimension=None,
                clip=None, colluders=0, bits=16, l2_sensitivity=None,
                rho_fraction=1.0, accounting='zcdp'):
    if not (0 <= colluders < n):
        raise ValueError('Colluders must leave an honest contributor.')
    scale = 1 << bits
    if l2_sensitivity is not None:
        sensitivity_squared = (l2_sensitivity*scale)**2
    elif clusters is not None:
        sensitivity_squared = 2*clusters*scale*scale
    else:
        if clip is None or dimension is None:
            raise ValueError('Parameter reports require a clipping norm and dimension.')
        sensitivity_squared = (2*clip*scale + math.sqrt(dimension))**2
    L = math.log(1/delta)
    rho_total_target = rho_from_epsilon(epsilon, delta, accounting)
    rho_round_target = rho_total_target/rounds*rho_fraction
    h = n-colluders
    desired_share = math.sqrt(sensitivity_squared/(2*h*rho_round_target))
    sigma_integer = math.floor(desired_share)+1
    if sigma_integer > 1000000000:
        raise ValueError('Raise the replay sampler scale limit or reduce excessive precision.')
    log_tau = (math.log(10*(h-1))-math.pi**2*sigma_integer**2) if h > 1 else -math.inf
    # A certified, deliberately loose bound is retained without underflowing.
    # tau <= 10(h-1) exp(-pi^2 b^2). All production configurations have tau < 1e-30.
    tau_upper = 1e-30 if log_tau < math.log(1e-30) else math.exp(log_tau)
    if h == 1:
        tau_upper = 0.0
    dimension = dimension if dimension is not None else 10*clusters
    rho_round = sensitivity_squared/(2*h*sigma_integer**2)+dimension*tau_upper/4
    if rho_round > rho_round_target:
        raise ValueError('Rounded scale leaves insufficient slack for the explicit correction.')
    rho_total = rounds*rho_round
    epsilon_bound = epsilon_from_rho(rho_total,delta,accounting)
    return dict(epsilon_requested=epsilon, delta=delta, rounds=rounds,
                actual_clients=n, calibrated_clients=n, honest_clients=h,
                colluders=colluders, quantization_scale=scale,
                sigma_share_integer=sigma_integer,
                sensitivity_integer_squared=sensitivity_squared,
                dimension=dimension, log_tau_upper=log_tau,
                tau_certified_upper=tau_upper, rho_round_upper=rho_round,
                rho_total_upper=rho_total, epsilon_upper=epsilon_bound,
                residual_std=math.sqrt(n)*sigma_integer/(scale*n),
                ring_bits=64, accounting=accounting, rho_fraction=rho_fraction,
                sampler='CKS exact rational rejection, ChaCha20 bits')


def epsilon_from_rho(rho, delta, accounting='zcdp'):
    if accounting=='zcdp':
        return rho+2*math.sqrt(rho*math.log(1/delta))
    from scipy.optimize import minimize_scalar
    # Improved RDP-to-DP conversion, evaluated at an explicit valid order.
    def objective(log_alpha_minus_one):
        alpha=1+math.exp(log_alpha_minus_one)
        return rho*alpha+math.log(1/(alpha*delta))/(alpha-1)+math.log1p(-1/alpha)
    result=minimize_scalar(objective,bounds=(-8,18),method='bounded')
    return max(0.0,float(result.fun))


def rho_from_epsilon(epsilon,delta,accounting='zcdp'):
    if accounting=='zcdp':
        L=math.log(1/delta)
        return (math.sqrt(L+epsilon)-math.sqrt(L))**2
    from scipy.optimize import brentq
    return brentq(lambda rho:epsilon_from_rho(rho,delta,accounting)-epsilon,
                  1e-16,max(1.,epsilon),xtol=1e-15)


def noise_sum(cert, dimension, seed, threads=16):
    shares = draw((cert['actual_clients'], dimension), cert['sigma_share_integer'],
                  seed=seed, threads=threads)
    return shares.sum(axis=0, dtype=np.int64)
