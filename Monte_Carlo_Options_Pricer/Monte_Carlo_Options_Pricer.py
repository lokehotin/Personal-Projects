"""
Monte Carlo Option Pricer
=========================

USAGE (examples)
----------------

# Basic run: European call, default 200k paths
python mc_pricer.py \
    --S0 100 --K 100 --r 0.02 --sigma 0.2 --T 1.0 --opt_type call

# European put, with explicit path count and random seed
python mc_pricer.py \
    --S0 100 --K 95 --r 0.03 --sigma 0.25 --T 0.5 --opt_type put \
    --n_paths 300000 --seed 123

# Disable variance reduction (no antithetic, no control variate)
python mc_pricer.py \
    --S0 120 --K 110 --r 0.01 --sigma 0.3 --T 2.0 --opt_type call \
    --antithetic False --control_variate False

# Print Delta and Vega estimates
python mc_pricer.py \
    --S0 100 --K 100 --r 0.02 --sigma 0.2 --T 1.0 --opt_type call \
    --greeks

Notes
-----
--S0              : initial spot price of underlying asset
--K               : strike price
--r               : risk-free interest rate (continuous compounding)
--sigma           : volatility (annualised, decimal form)
--T               : time to maturity (in years)
--opt_type        : option type ("call" or "put")
--n_paths         : number of Monte Carlo paths (default 200,000)
--antithetic      : toggle antithetic variates (default True)
--control_variate : toggle control variate adjustment (default True)
--seed            : random seed for reproducibility
--greeks          : output Delta and Vega estimates as well

Outputs include:
* Monte Carlo price ± confidence interval
* Black–Scholes benchmark price (for comparison)
* Optional Greeks: Delta, Vega
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class OptType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class EuroOption:
    S0: float  # spot
    K: float  # strike
    r: float  # risk-free (cont. comp.)
    sigma: float  # volatility
    T: float  # maturity in years
    opt_type: OptType


# ======== Closed-form (for validation & control variate) ========

from math import log, sqrt, exp
from scipy.stats import norm


def bs_price(opt):
    S0, K, r, sigma, T = opt.S0, opt.K, opt.r, opt.sigma, opt.T
    if T <= 0 or sigma <= 0:
        # immediate expiry / zero vol edge cases
        intrinsic = max(0.0, (S0 - K) if opt.opt_type == OptType.CALL else (K - S0))
        return intrinsic * np.exp(-r * T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if opt.opt_type == OptType.CALL:
        return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# ======== Monte Carlo core ========


def _simulate_terminal_prices(S0, r, sigma, T, n_paths, antithetic, seed):
    rng = np.random.default_rng(seed)
    if antithetic:
        m = (n_paths + 1) // 2
        Z = rng.standard_normal(m)
        Z = np.concatenate([Z, -Z])[:n_paths]
    else:
        Z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S0 * np.exp(drift + diffusion)
    return ST


def _payoff(ST, K, opt_type):
    if opt_type == OptType.CALL:
        return np.maximum(ST - K, 0.0)
    else:
        return np.maximum(K - ST, 0.0)


def mc_euro_price(opt, n_paths=200_000, antithetic=True, control_variate=True, seed=42):
    """
    Returns: (price, std_error, 95% CI tuple)
    """
    ST = _simulate_terminal_prices(
        opt.S0, opt.r, opt.sigma, opt.T, n_paths, antithetic, seed
    )
    disc = np.exp(-opt.r * opt.T)
    payoff = _payoff(ST, opt.K, opt.opt_type)
    Y = disc * payoff

    if control_variate:
        # Use ST as control with known mean E[ST]=S0*e^{rT}
        X = ST
        EX = opt.S0 * np.exp(opt.r * opt.T)
        b = np.cov(Y, X, bias=True)[0, 1] / np.var(X)
        Y_cv = Y - b * (X - EX)
        price = float(np.mean(Y_cv))
        se = float(np.std(Y_cv, ddof=0) / np.sqrt(len(Y_cv)))
    else:
        price = float(np.mean(Y))
        se = float(np.std(Y, ddof=0) / np.sqrt(len(Y)))

    ci = (price - 1.96 * se, price + 1.96 * se)
    return price, se, ci


# ======== Greeks (pathwise / LR) ========


def mc_delta_vega(opt, n_paths=200_000, seed=123):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    drift = (opt.r - 0.5 * opt.sigma**2) * opt.T
    diffusion = opt.sigma * np.sqrt(opt.T) * Z
    ST = opt.S0 * np.exp(drift + diffusion)
    disc = np.exp(-opt.r * opt.T)
    payoff = _payoff(ST, opt.K, opt.opt_type)

    # Pathwise Delta
    if opt.opt_type == OptType.CALL:
        indicator = (ST > opt.K).astype(float)
        delta_paths = disc * indicator * (ST / opt.S0)
    else:
        indicator = (ST < opt.K).astype(float)
        delta_paths = -disc * indicator * (ST / opt.S0)
    delta = float(np.mean(delta_paths))

    # LR Vega
    vega_paths = disc * payoff * (Z * np.sqrt(opt.T) - opt.sigma * opt.T)
    vega = float(np.mean(vega_paths) * opt.S0)

    return delta, vega


if __name__ == "__main__":
    opt = EuroOption(S0=100, K=100, r=0.02, sigma=0.2, T=1.0, opt_type=OptType.CALL)
    mc_p, mc_se, mc_ci = mc_euro_price(
        opt, n_paths=200_000, antithetic=True, control_variate=True, seed=7
    )
    bs_p = bs_price(opt)
    print(f"MC: {mc_p:.6f} ± {1.96*mc_se:.6f} (95% CI {mc_ci[0]:.6f}, {mc_ci[1]:.6f})")
    print(f"BS: {bs_p:.6f}")
    d, v = mc_delta_vega(opt, n_paths=300_000, seed=9)
    print(f"Delta ~= {d:.5f}, Vega ~= {v:.5f}")
