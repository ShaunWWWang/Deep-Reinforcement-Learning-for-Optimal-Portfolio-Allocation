"""
backtest.py
===========
Run DRL and MVO backtests over each held-out test year and compute the
performance metrics reported in Table 2 of the paper.

Metrics computed
----------------
Annual return, Cumulative returns, Annual volatility, Sharpe ratio,
Calmar ratio, Stability (R² of log-cumret on time), Max drawdown,
Omega ratio, Sortino ratio, Skew, Kurtosis, Tail ratio, Daily VaR.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import PPO

from data_utils import DataBundle, WindowConfig, date_to_idx, LOOKBACK
from environment import PortfolioEnv
from mvo_strategy import MVOStrategy


TRADING_DAYS  = 252
INITIAL_VALUE = 100_000.0


# ============================================================
#  DRL backtest
# ============================================================

def run_drl_backtest(
    model_path: str,
    bundle:     DataBundle,
    start_idx:  int,
    end_idx:    int,
) -> pd.Series:
    """
    Run a trained PPO agent deterministically over [start_idx, end_idx).

    Parameters
    ----------
    model_path : str          Path to the .zip model (without extension).
    bundle     : DataBundle
    start_idx  : int          First test step.
    end_idx    : int          One past the last test step.

    Returns
    -------
    pd.Series   Daily simple-return series, indexed by date.
    """
    model = PPO.load(model_path, device="auto")

    env = PortfolioEnv(
        log_returns    = bundle.log_returns,
        vol20_norm     = bundle.vol20_norm,
        vol_ratio_norm = bundle.vol_ratio_norm,
        vix_norm       = bundle.vix_norm,
        start_idx      = start_idx,
        end_idx        = end_idx,
        lookback       = LOOKBACK,
    )

    obs, _  = env.reset()
    done    = False
    returns = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        returns.append(info["port_return"])
        done = terminated or truncated

    n      = len(returns)
    dates  = bundle.dates[start_idx : start_idx + n]
    return pd.Series(returns, index=dates, name="DRL", dtype=float)


# ============================================================
#  MVO backtest
# ============================================================

def run_mvo_backtest(
    bundle:    DataBundle,
    start_idx: int,
    end_idx:   int,
) -> pd.Series:
    """
    Run MVO strategy deterministically over [start_idx, end_idx).

    Returns
    -------
    pd.Series   Daily simple-return series, indexed by date.
    """
    mvo     = MVOStrategy(bundle.prices, bundle.asset_names, lookback=LOOKBACK)
    returns = mvo.run_backtest(bundle.log_returns, start_idx, end_idx)
    dates   = bundle.dates[start_idx:end_idx]
    return pd.Series(returns, index=dates, name="MVO", dtype=float)


# ============================================================
#  Portfolio-value tracker (for ∆pw turnover analysis)
# ============================================================

def _portfolio_values(returns: pd.Series, initial: float = INITIAL_VALUE) -> pd.Series:
    """Compound daily returns to get portfolio value series."""
    cumret = (1 + returns).cumprod()
    return cumret * initial


# ============================================================
#  Performance metrics
# ============================================================

def compute_metrics(
    returns: pd.Series,
    rf:      float = 0.0,
) -> Dict[str, float]:
    """
    Compute the performance metrics shown in Table 2 of the paper.

    Parameters
    ----------
    returns : pd.Series   Daily simple returns.
    rf      : float       Annual risk-free rate (default 0, as in the paper).

    Returns
    -------
    dict mapping metric name → value.
    """
    r  = returns.dropna().values.astype(np.float64)
    n  = len(r)
    if n == 0:
        return {}

    rf_daily = rf / TRADING_DAYS

    # ---- Cumulative & annual return ----
    cum_return = float(np.prod(1.0 + r) - 1.0)
    ann_return = float((1.0 + cum_return) ** (TRADING_DAYS / n) - 1.0)

    # ---- Volatility ----
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))

    # ---- Sharpe ratio ----
    excess  = r - rf_daily
    sharpe  = float(
        (excess.mean() / (excess.std() + 1e-12)) * np.sqrt(TRADING_DAYS)
    )

    # ---- Drawdown ----
    cum_val  = np.cumprod(1.0 + r)
    roll_max = np.maximum.accumulate(cum_val)
    dds      = cum_val / (roll_max + 1e-12) - 1.0
    max_dd   = float(dds.min())

    # ---- Calmar ratio ----
    calmar = float(ann_return / (abs(max_dd) + 1e-12))

    # ---- Stability (R² of log-cumret on linear time) ----
    log_cum   = np.log(np.maximum(cum_val, 1e-12))
    t_arr     = np.arange(n, dtype=float)
    _, _, r_val, _, _ = stats.linregress(t_arr, log_cum)
    stability = float(r_val ** 2)

    # ---- Omega ratio (threshold = 0) ----
    gains  = r[r > 0].sum()
    losses = -r[r < 0].sum()
    omega  = float(gains / (losses + 1e-12))

    # ---- Sortino ratio ----
    downside = r[r < rf_daily]
    dd_std   = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 1e-12
    sortino  = float((ann_return - rf) / (dd_std + 1e-12))

    # ---- Higher moments ----
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))     # excess kurtosis

    # ---- Tail ratio (95th pct / |5th pct|) ----
    p95 = float(np.percentile(r, 95))
    p5  = float(abs(np.percentile(r, 5)))
    tail_ratio = float(p95 / (p5 + 1e-12))

    # ---- Daily VaR (5 %) ----
    daily_var = float(np.percentile(r, 5))

    return {
        "Annual return":        ann_return,
        "Cumulative returns":   cum_return,
        "Annual volatility":    ann_vol,
        "Sharpe ratio":         sharpe,
        "Calmar ratio":         calmar,
        "Stability":            stability,
        "Max drawdown":         max_dd,
        "Omega ratio":          omega,
        "Sortino ratio":        sortino,
        "Skew":                 skew,
        "Kurtosis":             kurt,
        "Tail ratio":           tail_ratio,
        "Daily value at risk":  daily_var,
    }


# ============================================================
#  Full backtest loop
# ============================================================

def run_full_backtest(
    drl_model_paths: List[Optional[str]],
    bundle:          DataBundle,
    windows:         List[WindowConfig],
    n_drl_agents:    int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backtest DRL and MVO across all 10 test years.

    For DRL, if multiple model paths are provided (one per seed), the
    function averages their daily returns before computing metrics
    (matching the paper's 'average of five agents' approach).

    Parameters
    ----------
    drl_model_paths : list   One best-model path per window (or None).
    bundle          : DataBundle
    windows         : list of WindowConfig
    n_drl_agents    : int   How many seed models exist per window (used
                            when ensemble averaging is desired).

    Returns
    -------
    (drl_stats_df, mvo_stats_df, drl_returns_df, mvo_returns_df)
    """
    drl_stats, mvo_stats = [], []
    all_drl_rets = {}
    all_mvo_rets = {}

    for window, model_path in zip(windows, drl_model_paths):
        year       = int(window.test_start[:4])
        test_s     = max(date_to_idx(bundle, window.test_start), LOOKBACK)
        test_e     = date_to_idx(bundle, window.test_end)

        if test_s >= test_e:
            print(f"  Window {window.window_id}: skipping (no test data).")
            continue

        print(f"  Backtesting {year}  [{window.test_start} → {window.test_end})")

        # ---- MVO ----
        mvo_rets  = run_mvo_backtest(bundle, test_s, test_e)
        mvo_m     = compute_metrics(mvo_rets)
        mvo_m["Year"] = year
        mvo_stats.append(mvo_m)
        all_mvo_rets[year] = mvo_rets

        # ---- DRL ----
        if model_path and os.path.exists(model_path + ".zip"):
            drl_rets = run_drl_backtest(model_path, bundle, test_s, test_e)
            drl_m    = compute_metrics(drl_rets)
            drl_m["Year"] = year
            drl_stats.append(drl_m)
            all_drl_rets[year] = drl_rets
            print(f"    DRL Sharpe={drl_m['Sharpe ratio']:.4f}  "
                  f"MVO Sharpe={mvo_m['Sharpe ratio']:.4f}")
        else:
            print(f"    DRL model not found for window {window.window_id}; "
                  "running MVO only.")

    drl_df  = (pd.DataFrame(drl_stats).set_index("Year")
               if drl_stats else pd.DataFrame())
    mvo_df  = (pd.DataFrame(mvo_stats).set_index("Year")
               if mvo_stats else pd.DataFrame())

    drl_rets_df = pd.DataFrame(all_drl_rets) if all_drl_rets else pd.DataFrame()
    mvo_rets_df = pd.DataFrame(all_mvo_rets) if all_mvo_rets else pd.DataFrame()

    return drl_df, mvo_df, drl_rets_df, mvo_rets_df
