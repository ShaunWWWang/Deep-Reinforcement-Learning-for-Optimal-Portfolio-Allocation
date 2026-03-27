"""
main.py
=======
Top-level script for the FinPlan23 replication.

Usage
-----
# Run full pipeline (download → train → backtest)
python main.py --mode all

# MVO baseline only (no GPU / long training needed)
python main.py --mode mvo_only

# Skip training, backtest previously saved models
python main.py --mode backtest --models_dir models

# Train only
python main.py --mode train --output_dir results
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import download_data, get_sliding_windows
from train_pipeline import run_training_pipeline, LOOKBACK
from backtest import (
    run_full_backtest,
    run_mvo_backtest,
    compute_metrics,
    TRADING_DAYS,
)
from data_utils import date_to_idx


# ============================================================
#  Printing
# ============================================================

METRIC_ORDER = [
    "Annual return",
    "Cumulative returns",
    "Annual volatility",
    "Sharpe ratio",
    "Calmar ratio",
    "Stability",
    "Max drawdown",
    "Omega ratio",
    "Sortino ratio",
    "Skew",
    "Kurtosis",
    "Tail ratio",
    "Daily value at risk",
]


def print_results_table(drl_df: pd.DataFrame, mvo_df: pd.DataFrame) -> None:
    """Print Table-2-style averaged statistics."""
    print("\n" + "=" * 58)
    print(f"  {'Metric':<30}  {'DRL':>10}  {'MVO':>10}")
    print("=" * 58)

    for m in METRIC_ORDER:
        drl_val = mvo_val = float("nan")
        if not drl_df.empty and m in drl_df.columns:
            drl_val = (drl_df[m].min() if "drawdown" in m.lower()
                       else drl_df[m].mean())
        if not mvo_df.empty and m in mvo_df.columns:
            mvo_val = (mvo_df[m].min() if "drawdown" in m.lower()
                       else mvo_df[m].mean())
        print(f"  {m:<30}  {drl_val:>10.4f}  {mvo_val:>10.4f}")

    print("=" * 58)
    print("  (Averaged across all backtest years; "
          "Max drawdown = worst year)")


# ============================================================
#  Plotting  (mirrors Figure 2)
# ============================================================

def _plot_metric_comparison(
    ax:        plt.Axes,
    drl_df:    pd.DataFrame,
    mvo_df:    pd.DataFrame,
    metric:    str,
    title:     str,
    ylabel:    str = "",
) -> None:
    if not mvo_df.empty and metric in mvo_df.columns:
        ax.plot(mvo_df.index.astype(str), mvo_df[metric],
                "s--", color="steelblue", label="MVO", linewidth=1.5)
    if not drl_df.empty and metric in drl_df.columns:
        ax.plot(drl_df.index.astype(str), drl_df[metric],
                "o-", color="darkorange", label="DRL", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Backtest Year")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45)


def plot_backtest_comparison(
    drl_df:    pd.DataFrame,
    mvo_df:    pd.DataFrame,
    output_dir: str,
) -> None:
    """Reproduce Figure 2: Sharpe, Max Drawdown, Annual Return per year."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Backtest Performance: MVO vs Deep RL", fontsize=14, y=1.01)

    _plot_metric_comparison(axes[0], drl_df, mvo_df, "Sharpe ratio",
                            "Sharpe Ratio", "Sharpe")
    _plot_metric_comparison(axes[1], drl_df, mvo_df, "Max drawdown",
                            "Maximum Drawdown", "Drawdown")
    _plot_metric_comparison(axes[2], drl_df, mvo_df, "Annual return",
                            "Annual Return", "Return")

    plt.tight_layout()
    path = os.path.join(output_dir, "backtest_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {path}")
    plt.close()


def plot_return_distributions(
    drl_rets_df: pd.DataFrame,
    mvo_rets_df: pd.DataFrame,
    output_dir:  str,
) -> None:
    """Plot monthly return distribution (Figure 3c / 4c style)."""
    if drl_rets_df.empty and mvo_rets_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, df, name, color in zip(
        axes,
        [drl_rets_df, mvo_rets_df],
        ["DRL", "MVO"],
        ["darkorange", "steelblue"],
    ):
        if df.empty:
            continue
        # Aggregate all annual columns into one series
        all_rets = df.stack().values.astype(float)
        ax.hist(all_rets * 100, bins=50, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(all_rets.mean() * 100, color="black",
                   linestyle="--", linewidth=1.5, label="Mean")
        ax.set_title(f"{name} — Daily Return Distribution", fontsize=11)
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "return_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def plot_cumulative_returns(
    drl_rets_df: pd.DataFrame,
    mvo_rets_df: pd.DataFrame,
    output_dir:  str,
) -> None:
    """Plot cumulative portfolio value over the full backtest period."""
    if drl_rets_df.empty and mvo_rets_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    for df, label, color in [
        (drl_rets_df, "DRL", "darkorange"),
        (mvo_rets_df, "MVO", "steelblue"),
    ]:
        if df.empty:
            continue
        combined = df.stack()
        combined.index = combined.index.droplevel(1)
        combined       = combined.sort_index()
        cum_val        = 100_000 * (1 + combined).cumprod()
        ax.plot(cum_val.index, cum_val.values, label=label, color=color)

    ax.set_title("Cumulative Portfolio Value — DRL vs MVO (2012–2021)",
                 fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "cumulative_returns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


# ============================================================
#  Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinPlan23 replication — DRL vs MVO portfolio optimisation"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "mvo_only", "all"],
        default="all",
        help=(
            "train      → train DRL agents only\n"
            "backtest   → run backtests using saved models\n"
            "mvo_only   → run MVO baseline only (fast, no GPU)\n"
            "all        → full pipeline"
        ),
    )
    parser.add_argument("--output_dir", default="results",
                        help="Directory for plots and CSV results")
    parser.add_argument("--models_dir", default="models",
                        help="Directory for model checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Data
    # ----------------------------------------------------------
    print("\n" + "─" * 65)
    print("  STEP 1 — Download & preprocess data")
    print("─" * 65)
    bundle  = download_data(start="2005-07-01", end="2022-01-01")
    windows = get_sliding_windows()

    # ----------------------------------------------------------
    # 2. Train DRL agents
    # ----------------------------------------------------------
    drl_model_paths: List[Optional[str]] = [None] * len(windows)

    if args.mode in ("train", "all"):
        print("\n" + "─" * 65)
        print("  STEP 2 — Train PPO agents (10 windows × 5 seeds)")
        print("─" * 65)
        drl_model_paths = run_training_pipeline(
            bundle, windows, args.models_dir
        )

    elif args.mode == "backtest":
        # Attempt to locate previously saved models
        for i, w in enumerate(windows):
            candidate = os.path.join(
                args.models_dir,
                f"window_{w.window_id:02d}",
                f"best_window_{w.window_id:02d}",
            )
            drl_model_paths[i] = candidate if os.path.exists(
                candidate + ".zip"
            ) else None

    # ----------------------------------------------------------
    # 3. Backtests
    # ----------------------------------------------------------
    print("\n" + "─" * 65)
    print("  STEP 3 — Backtesting")
    print("─" * 65)

    if args.mode == "mvo_only":
        # Run MVO only (no DRL models required)
        mvo_stats, mvo_rets_by_year = [], {}
        for w in windows:
            test_s = max(date_to_idx(bundle, w.test_start), LOOKBACK)
            test_e = date_to_idx(bundle, w.test_end)
            if test_s >= test_e:
                continue
            year = int(w.test_start[:4])
            print(f"  MVO backtest {year} …")
            rets = run_mvo_backtest(bundle, test_s, test_e)
            m    = compute_metrics(rets)
            m["Year"] = year
            mvo_stats.append(m)
            mvo_rets_by_year[year] = rets

        mvo_df      = pd.DataFrame(mvo_stats).set_index("Year")
        drl_df      = pd.DataFrame()
        drl_rets_df = pd.DataFrame()
        mvo_rets_df = pd.DataFrame(mvo_rets_by_year)

    else:
        drl_df, mvo_df, drl_rets_df, mvo_rets_df = run_full_backtest(
            drl_model_paths, bundle, windows
        )

    # ----------------------------------------------------------
    # 4. Report
    # ----------------------------------------------------------
    print("\n" + "─" * 65)
    print("  RESULTS  (averaged across all 10 backtest years)")
    print("─" * 65)
    print_results_table(drl_df, mvo_df)

    # Save CSVs
    if not drl_df.empty:
        drl_df.to_csv(os.path.join(args.output_dir, "drl_stats.csv"))
        print(f"\n  DRL stats saved → {args.output_dir}/drl_stats.csv")
    if not mvo_df.empty:
        mvo_df.to_csv(os.path.join(args.output_dir, "mvo_stats.csv"))
        print(f"  MVO stats saved → {args.output_dir}/mvo_stats.csv")

    # Plots
    plot_backtest_comparison(drl_df, mvo_df, args.output_dir)
    plot_return_distributions(drl_rets_df, mvo_rets_df, args.output_dir)
    plot_cumulative_returns(drl_rets_df, mvo_rets_df, args.output_dir)

    print(f"\n  All outputs saved to '{args.output_dir}/'")
    print("  Done.\n")


if __name__ == "__main__":
    main()