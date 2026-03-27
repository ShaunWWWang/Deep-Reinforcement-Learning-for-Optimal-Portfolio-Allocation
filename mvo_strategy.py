"""
mvo_strategy.py
===============
Daily-rebalanced Mean-Variance Optimization baseline.

Matches the paper's MVO setup:
    • 60-day rolling lookback window (same as DRL)
    • Ledoit-Wolf shrinkage covariance (Ledoit & Wolf 2004)
    • PSD correction — zero-out negative eigenvalues
    • Sharpe-ratio maximisation via PyPortfolioOpt (risk-free rate = 0)
    • Long-only, fully invested (∑ w_i = 1, 0 ≤ w_i ≤ 1)
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from pypfopt import EfficientFrontier, CovarianceShrinkage
    _PYPFOPT_AVAILABLE = True
except ImportError:
    _PYPFOPT_AVAILABLE = False
    print("[mvo_strategy] WARNING: PyPortfolioOpt not installed. "
          "Falling back to equal-weight portfolio.")


# ============================================================
#  Helpers
# ============================================================

def _psd_fix(matrix: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Enforce positive semi-definiteness by zeroing negative eigenvalues.

    Rebuilds the matrix as V · diag(max(λ, tol)) · Vᵀ.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, tol, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _ledoit_wolf_cov(prices_window: pd.DataFrame) -> np.ndarray:
    """
    Compute the Ledoit-Wolf shrinkage covariance of *annualised* returns.

    Falls back to the sample covariance if PyPortfolioOpt is unavailable.
    """
    if _PYPFOPT_AVAILABLE:
        shrink = CovarianceShrinkage(prices_window)
        S = shrink.ledoit_wolf().values          # annualised by default
    else:
        log_rets = np.log(prices_window / prices_window.shift(1)).dropna()
        S = np.cov(log_rets.values.T) * 252     # annualise manually

    return _psd_fix(S)


def _sample_mu(prices_window: pd.DataFrame) -> np.ndarray:
    """
    Annualised mean log return over the lookback window.

    This is the sample mean as used by the paper (no fancy estimation).
    """
    log_rets = np.log(prices_window / prices_window.shift(1)).dropna()
    return log_rets.mean().values * 252          # annualise


def _max_sharpe_weights(
    mu: np.ndarray,
    S: np.ndarray,
    tickers: List[str],
    rf: float = 0.0,
) -> np.ndarray:
    """
    Maximise Sharpe ratio via PyPortfolioOpt.
    Falls back to equal-weight if optimisation fails.
    """
    n = len(tickers)
    equal_w = np.full(n, 1.0 / n, dtype=np.float32)

    if not _PYPFOPT_AVAILABLE:
        return equal_w

    mu_series = pd.Series(mu, index=tickers)
    S_df      = pd.DataFrame(S, index=tickers, columns=tickers)

    try:
        ef = EfficientFrontier(mu_series, S_df, weight_bounds=(0, 1))
        ef.max_sharpe(risk_free_rate=rf)
        cleaned = ef.clean_weights()
        w = np.array([cleaned.get(t, 0.0) for t in tickers], dtype=np.float32)
        w = np.clip(w, 0.0, 1.0)
        total = w.sum()
        return w / total if total > 1e-8 else equal_w
    except Exception:
        # Numerical failure (e.g. singular covariance, no feasible solution)
        return equal_w


# ============================================================
#  Main class
# ============================================================

class MVOStrategy:
    """
    Mean-Variance Optimization strategy for the backtest.

    Usage
    -----
    mvo = MVOStrategy(bundle.prices, bundle.asset_names, lookback=60)
    for t in range(test_start, test_end):
        weights = mvo.get_weights(t)      # uses data[:t], shape (n_assets,)
        port_return = weights @ simple_rets[t]
    """

    def __init__(
        self,
        prices:      pd.DataFrame,
        asset_names: List[str],
        lookback:    int   = 60,
        rf:          float = 0.0,
    ) -> None:
        self.prices      = prices           # full history; indexed by date
        self.asset_names = asset_names
        self.lookback    = lookback
        self.rf          = rf
        self.n_assets    = len(asset_names)

        self._equal_w = np.full(self.n_assets, 1.0 / self.n_assets,
                                dtype=np.float32)

    def get_weights(self, t_idx: int) -> np.ndarray:
        """
        Compute optimal weights using the `lookback`-day window ending at t_idx-1.

        Parameters
        ----------
        t_idx : int   Current time index (today); uses *only* past data.

        Returns
        -------
        np.ndarray, shape (n_assets,), sums to 1.
        """
        if t_idx < self.lookback + 1:
            return self._equal_w.copy()

        # Price slice: t-lookback … t-1  (lookback days of prices → lookback-1 returns)
        # Need one extra row for the shift, so take lookback+1 prices
        price_slice = self.prices.iloc[t_idx - self.lookback - 1 : t_idx]
        if len(price_slice) < self.lookback:
            return self._equal_w.copy()

        try:
            mu = _sample_mu(price_slice)
            S  = _ledoit_wolf_cov(price_slice)
            return _max_sharpe_weights(mu, S, self.asset_names, self.rf)
        except Exception:
            return self._equal_w.copy()

    def run_backtest(
        self,
        log_returns: np.ndarray,
        test_start:  int,
        test_end:    int,
    ) -> np.ndarray:
        """
        Run the MVO strategy over [test_start, test_end) and return a
        daily simple-return array of shape (test_end - test_start,).
        """
        returns  = []
        for t in range(test_start, test_end):
            w            = self.get_weights(t)
            simple_rets  = np.exp(log_returns[t]) - 1.0
            port_ret     = float(w @ simple_rets)
            returns.append(port_ret)
        return np.array(returns, dtype=np.float64)
