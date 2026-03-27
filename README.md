# Deep RL vs MVO Portfolio Optimisation — FinPlan23 Replication

Replication of **"Deep Reinforcement Learning for Optimal Portfolio Allocation:
A Comparative Study with Mean-Variance Optimization"** (Sood et al., AAAI FinPlan Workshop 2023, J.P. Morgan AI Research).

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. MVO baseline only (fast, no GPU needed — good sanity check)
python main.py --mode mvo_only

# 3. Full pipeline: download → train DRL → backtest both
python main.py --mode all

# 4. Re-run backtests using previously trained models
python main.py --mode backtest --models_dir models
```

Outputs are written to `results/` (plots + CSV tables).

---

## Repository layout

```
portfolio_rl/
├── environment.py        # Gym environment + Differential Sharpe Ratio reward
├── data_utils.py         # Data download, feature engineering, sliding windows
├── mvo_strategy.py       # MVO baseline (Ledoit-Wolf + Sharpe maximisation)
├── train_pipeline.py     # PPO training loop (10 windows × 5 seeds)
├── backtest.py           # Backtest execution + performance metrics
├── main.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## Method overview

### Data

| Source | Tickers | Period |
|--------|---------|--------|
| Yahoo Finance | XLB, XLI, XLY, XLP, XLV, XLF, XLK, XLU, XLE | 2005–2022 |
| Yahoo Finance | ^GSPC (S&P 500), ^VIX | 2005–2022 |

> **Note on sectors**: The paper uses 11 S&P 500 GICS sectors.
> XLC (Communication Services) launched in 2018 and XLRE (Real Estate) in 2015,
> so this replication uses the 9 ETFs available from 2006 onward.

---

### Environment (`environment.py`)

**State matrix S_t — shape (n+1) × T, then flattened:**

```
Row i (security i):  [ w_i,  r_{i,t-1},  r_{i,t-2},  …,  r_{i,t-T+1} ]
Row n (cash):        [ w_c,  vol20_norm,  vol_ratio_norm,  VIX_norm,  0, … ]
```

Where:
- `w_i` = portfolio weight entering timestep t
- `r_{i,t-lag}` = log return of asset i at time t−lag
- `vol20_norm` = expanding-window z-scored 20-day S&P 500 rolling volatility
- `vol_ratio_norm` = expanding-window z-scored vol₂₀/vol₆₀ ratio
- `VIX_norm` = expanding-window z-scored VIX index value
- T = 60 (lookback days)

**Action:** raw logits ∈ ℝ^(n+1) → softmax → portfolio weights ∈ Δ^(n+1)  
(long-only, fully invested, no leverage, no short-selling)

**Reward: Differential Sharpe Ratio (Moody et al., 1998)**

Standard Sharpe ratio is defined over an entire period — unsuitable as
a per-step reward. The DSR approximates its gradient w.r.t. η at η→0:

```
A_t = A_{t-1} + η · (R_t − A_{t-1})          ← EMA of returns
B_t = B_{t-1} + η · (R_t² − B_{t-1})         ← EMA of squared returns

D_t = [ B_{t-1}·ΔA_t − ½·A_{t-1}·ΔB_t ] / (B_{t-1} − A_{t-1}²)^{3/2}
```

with η = 1/252, A₀ = B₀ = 0.

---

### DRL agent (`train_pipeline.py`)

**Algorithm:** Proximal Policy Optimisation (PPO, Schulman et al., 2017)  
**Implementation:** Stable-Baselines3

| Hyperparameter | Value |
|----------------|-------|
| Total timesteps per window | 7.5 M |
| Parallel environments (n_envs) | 10 |
| Rollout steps per env (n_steps) | 756 (= 252 × 3) |
| Batch size | 1 260 (= 252 × 5) |
| Epochs per update | 16 |
| Discount factor γ | 0.9 |
| GAE λ | 0.9 |
| Clip range ε | 0.25 |
| Learning rate | 3×10⁻⁴ → 1×10⁻⁵ (linear) |
| Policy network | [64, 64] FC, tanh, log_std_init = −1 |

**Sliding-window training** (10 windows, 1-year shift):

```
Window 1: train [2006,2011)  val 2011  test 2012
Window 2: train [2007,2012)  val 2012  test 2013
…
Window 10: train [2015,2020)  val 2020  test 2021
```

For each window, 5 agents are trained with different seeds.  
The best agent (highest mean validation reward) is:
1. Saved as the checkpoint for this window.
2. Used as the **seed policy** for the next window's 5 agents (policy transfer).

---

### MVO baseline (`mvo_strategy.py`)

At each day t:
1. Take the 60-day price window ending at t−1.
2. Estimate μ = annualised sample mean log return.
3. Estimate Σ = Ledoit-Wolf shrinkage covariance (PyPortfolioOpt).
4. Apply PSD correction (zero-out negative eigenvalues).
5. Solve max-Sharpe via `EfficientFrontier.max_sharpe(risk_free_rate=0)`.
6. Rebalance the portfolio.

---

### Performance metrics (`backtest.py`)

All metrics from Table 2 of the paper are computed from the daily return series:

| Metric | Formula |
|--------|---------|
| Annual return | (1 + cumret)^(252/n) − 1 |
| Annual volatility | σ_daily × √252 |
| Sharpe ratio | (mean_excess / std_excess) × √252 |
| Calmar ratio | ann_return / |max_drawdown| |
| Stability | R² of log-cumret on linear time |
| Max drawdown | min(cumval / rolling_max − 1) |
| Omega ratio | Σ gains / Σ |losses| |
| Sortino ratio | ann_return / downside_std |
| Tail ratio | P₉₅(r) / |P₅(r)| |
| Daily VaR | P₅(r) |

---

## Expected results (paper, Table 2)

| Metric | DRL | MVO |
|--------|-----|-----|
| Annual return | 0.1211 | 0.0653 |
| Sharpe ratio | 1.1662 | 0.6776 |
| Calmar ratio | 2.3133 | 1.1608 |
| Max drawdown | −0.3296 | −0.3303 |
| Sortino ratio | 1.7208 | 1.0060 |

> Results may differ slightly due to:
> - Using 9 sector ETFs (paper uses 11)
> - Randomness in PPO training
> - yfinance data adjustments that may have changed since publication

---

## Tips

**Fast iteration / debugging**
```python
# Reduce training timesteps for quick testing
TOTAL_TIMESTEPS = 100_000   # in train_pipeline.py
N_SEEDS = 2
```

**GPU acceleration**  
SB3 automatically uses CUDA if available. Set `device="cpu"` in
`train_pipeline.py` to force CPU.

**Transaction costs**  
The paper assumes no transaction costs. To add them, modify `PortfolioEnv.step()`:
```python
# Penalise turnover
turnover = np.abs(self._weights - old_weights).sum()
reward -= cost_per_unit * turnover
```

**Custom assets**  
Replace `SECTOR_TICKERS` in `data_utils.py` with any Yahoo Finance tickers.

---

## Citation

```bibtex
@inproceedings{sood2023deep,
  title={Deep Reinforcement Learning for Optimal Portfolio Allocation:
         A Comparative Study with Mean-Variance Optimization},
  author={Sood, Srijan and Papasotiriou, Kassiani and
          Vaiciulis, Marius and Balch, Tucker},
  booktitle={AAAI Workshop on AI in Finance (FinPlan)},
  year={2023}
}
```
