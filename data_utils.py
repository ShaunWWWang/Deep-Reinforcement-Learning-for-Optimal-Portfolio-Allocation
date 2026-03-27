"""
data_utils.py
=============
Download S&P 500 sector ETF prices, compute features, and define the
10 sliding-window training/validation/test splits used in the paper.

Sector ETFs used (all listed since before 2006):
    XLB  Materials        XLI  Industrials        XLY  Consumer Discretionary
    XLP  Consumer Staples XLV  Health Care        XLF  Financials
    XLK  Information Tech XLU  Utilities          XLE  Energy

Note: XLC (Communication Services, IPO 2018) and XLRE (Real Estate, IPO 2015)
are excluded; they don't cover the full 2006-2021 backtest window.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ============================================================
#  Constants
# ============================================================

SECTOR_TICKERS: List[str] = [
    "XLB",  # Materials
    "XLI",  # Industrials
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLV",  # Health Care
    "XLF",  # Financials
    "XLK",  # Information Technology
    "XLU",  # Utilities
    "XLE",  # Energy
]

LOOKBACK = 60      # T in the paper
INITIAL_CASH = 100_000.0


# ============================================================
#  Data containers
# ============================================================

@dataclass
class DataBundle:
    """All preprocessed arrays aligned on a common DatetimeIndex."""
    dates:          pd.DatetimeIndex
    log_returns:    np.ndarray        # (T, n_assets)  float32
    vol20_norm:     np.ndarray        # (T,)            float32
    vol_ratio_norm: np.ndarray        # (T,)            float32
    vix_norm:       np.ndarray        # (T,)            float32
    asset_names:    List[str]
    prices:         pd.DataFrame      # raw adjusted-close prices (T, n_assets)


@dataclass
class WindowConfig:
    """Date ranges for a single sliding window."""
    window_id:   int
    train_start: str   # inclusive
    train_end:   str   # exclusive
    val_start:   str   # inclusive
    val_end:     str   # exclusive
    test_start:  str   # inclusive
    test_end:    str   # exclusive


# ============================================================
#  Data download & preprocessing
# ============================================================

def download_data(
    start: str = "2005-07-01",   # extra history for warm-up before 2006
    end:   str = "2022-01-01",
) -> DataBundle:
    """
    Download sector ETF prices, S&P 500 index and VIX from Yahoo Finance,
    then compute and align all features.

    Returns
    -------
    DataBundle
    """
    tickers = SECTOR_TICKERS + ["^GSPC", "^VIX"]
    print(f"Downloading {len(tickers)} tickers ({start} → {end}) …")

    raw   = yf.download(tickers, start=start, end=end,
                        auto_adjust=True, progress=False)

    # yfinance may return MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw

    close = close.dropna(how="any")

    sp500  = close["^GSPC"]
    vix    = close["^VIX"]
    prices = close[SECTOR_TICKERS].copy()

    # --- Log returns (daily) ---
    log_ret = np.log(prices / prices.shift(1)).dropna()

    # Align all series to the log-return index
    idx       = log_ret.index
    sp500     = sp500.reindex(idx).ffill().dropna()
    vix_s     = vix.reindex(idx).ffill().dropna()
    idx       = idx.intersection(sp500.index).intersection(vix_s.index)
    log_ret   = log_ret.loc[idx]
    prices    = prices.loc[idx]
    sp500     = sp500.loc[idx]
    vix_s     = vix_s.loc[idx]

    # --- S&P 500 log returns for volatility ---
    sp_log = np.log(sp500 / sp500.shift(1)).fillna(0.0)

    vol20_raw  = sp_log.rolling(20).std().fillna(method="bfill")
    vol60_raw  = sp_log.rolling(60).std().fillna(method="bfill")
    vol_ratio  = (vol20_raw / vol60_raw.replace(0.0, np.nan)).fillna(1.0)

    # --- Expanding-window z-score (no look-ahead) ---
    vol20_norm     = _expanding_zscore(vol20_raw.values)
    vol_ratio_norm = _expanding_zscore(vol_ratio.values)
    vix_norm       = _expanding_zscore(vix_s.values)

    print(f"  Trading days: {len(idx)}  "
          f"({idx[0].date()} → {idx[-1].date()})")
    print(f"  Assets: {SECTOR_TICKERS}")

    return DataBundle(
        dates          = idx,
        log_returns    = log_ret.values.astype(np.float32),
        vol20_norm     = vol20_norm.astype(np.float32),
        vol_ratio_norm = vol_ratio_norm.astype(np.float32),
        vix_norm       = vix_norm.astype(np.float32),
        asset_names    = SECTOR_TICKERS,
        prices         = prices,
    )


def _expanding_zscore(arr: np.ndarray, min_periods: int = 60) -> np.ndarray:
    """
    Causal z-score using an expanding (never look-ahead) mean and std.

    For t < min_periods the output is 0.
    """
    out = np.zeros(len(arr), dtype=np.float32)
    for t in range(min_periods, len(arr)):
        window = arr[:t]
        mu     = window.mean()
        sigma  = window.std() + 1e-8
        out[t] = (arr[t] - mu) / sigma
    return out


# ============================================================
#  Sliding windows
# ============================================================

def get_sliding_windows() -> List[WindowConfig]:
    """
    Return the 10 sliding-window configurations from the paper.

    Each window:
        5 years training  |  1 year validation (burn)  |  1 year test
    Windows shift by one year; backtests cover 2012–2021.

    Window 1 : train [2006, 2011), val 2011, test 2012
    Window 2 : train [2007, 2012), val 2012, test 2013
    …
    Window 10: train [2015, 2020), val 2020, test 2021
    """
    windows = []
    for i in range(10):
        base = 2006 + i
        windows.append(WindowConfig(
            window_id   = i + 1,
            train_start = f"{base}-01-01",
            train_end   = f"{base + 5}-01-01",
            val_start   = f"{base + 5}-01-01",
            val_end     = f"{base + 6}-01-01",
            test_start  = f"{base + 6}-01-01",
            test_end    = f"{base + 7}-01-01",
        ))
    return windows


def date_to_idx(bundle: DataBundle, date_str: str, side: str = "left") -> int:
    """
    Convert a date string to the nearest integer index into bundle.dates.

    Parameters
    ----------
    bundle    : DataBundle
    date_str  : str   e.g. "2012-01-01"
    side      : "left" or "right" — passed to searchsorted

    Returns
    -------
    int   Clipped to [0, len(bundle.dates)-1].
    """
    ts  = pd.Timestamp(date_str)
    idx = int(bundle.dates.searchsorted(ts, side=side))
    return int(np.clip(idx, 0, len(bundle.dates) - 1))
