"""
environment.py
==============
Portfolio allocation Gym environment (Sood et al., FinPlan23).

Key classes
-----------
DifferentialSharpeRatio  — per-step risk-adjusted reward (Moody et al., 1998)
PortfolioEnv             — market-replay Gymnasium environment
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ============================================================
#  Differential Sharpe Ratio
# ============================================================

class DifferentialSharpeRatio:
    """
    Differential Sharpe Ratio (DSR) — Moody et al. (1998).

    The standard Sharpe ratio is defined over an entire period, making it
    unsuitable as a per-step RL reward. DSR approximates dS/dη|_{η→0},
    giving an instantaneous, causal signal that drives the policy toward
    higher risk-adjusted returns.

    Update equations (exponential moving averages):
        A_t  = A_{t-1}  + η · ΔA_t       where ΔA_t = R_t - A_{t-1}
        B_t  = B_{t-1}  + η · ΔB_t       where ΔB_t = R_t² - B_{t-1}

        D_t  = (B_{t-1}·ΔA_t − ½·A_{t-1}·ΔB_t) / (B_{t-1} − A_{t-1}²)^{3/2}

    Parameters
    ----------
    eta : float
        Adaptation rate.  The paper sets η ≈ 1/252 (one trading year).
    """

    def __init__(self, eta: float = 1.0 / 252.0) -> None:
        self.eta = eta
        self.A: float = 0.0   # EMA of returns
        self.B: float = 0.0   # EMA of squared returns

    def reset(self) -> None:
        self.A = 0.0
        self.B = 0.0

    def update(self, R: float) -> float:
        """
        Incorporate one return observation and return D_t.

        Parameters
        ----------
        R : float   Portfolio simple return at the current timestep.

        Returns
        -------
        float   D_t clipped to [-10, 10] for numerical stability.
        """
        delta_A = R - self.A
        delta_B = R ** 2 - self.B

        variance = self.B - self.A ** 2
        D = 0.0
        if variance > 1e-12:
            D = (self.B * delta_A - 0.5 * self.A * delta_B) / (variance ** 1.5)

        # Update EMAs *after* computing D (uses previous A_{t-1}, B_{t-1})
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return float(np.clip(D, -10.0, 10.0))


# ============================================================
#  Portfolio environment
# ============================================================

class PortfolioEnv(gym.Env):
    """
    Portfolio Allocation Environment — daily market replay.

    Follows the formulation in Sood et al. (FinPlan23).

    Observation (state) S_t — shape (n+1, T), then flattened:
    ─────────────────────────────────────────────────────────
    Row i ∈ [0, n-1] (security i):
        [  w_i,  r_{i,t-1},  r_{i,t-2},  …,  r_{i,t-T+1}  ]
         ^col0   ^col1        ^col2             ^col T-1

    Row n (cash):
        [  w_c,  vol20_norm,  vol_ratio_norm,  VIX_norm,  0, …, 0  ]

    Where:
        w_i          = portfolio weight of asset i entering timestep t
        r_{i,t-lag}  = log return of asset i at t - lag
        vol20_norm   = standardised 20-day rolling S&P500 volatility
        vol_ratio_norm = standardised vol20/vol60 ratio
        VIX_norm     = standardised VIX index value

    Action
    ------
    Raw logits ∈ ℝ^(n+1)  →  softmax  →  w ∈ Δ^(n+1)   (long-only simplex)

    Reward
    ------
    Differential Sharpe Ratio D_t.

    Parameters
    ----------
    log_returns   : np.ndarray, shape (T_total, n_assets)
        Daily log returns for n_assets securities.
    vol20_norm    : np.ndarray, shape (T_total,)
    vol_ratio_norm: np.ndarray, shape (T_total,)
    vix_norm      : np.ndarray, shape (T_total,)
        Market-regime indicators (expanding-window z-scored).
    start_idx     : int   First valid step (must be ≥ lookback).
    end_idx       : int   One past the last step (exclusive).
    lookback      : int   T — history window length (default 60 days).
    eta           : float DSR adaptation rate (default 1/252).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        log_returns: np.ndarray,
        vol20_norm: np.ndarray,
        vol_ratio_norm: np.ndarray,
        vix_norm: np.ndarray,
        start_idx: int,
        end_idx: int,
        lookback: int = 60,
        eta: float = 1.0 / 252.0,
    ) -> None:
        super().__init__()

        # ---- Validate ----
        assert log_returns.ndim == 2, "log_returns must be 2-D (T, n_assets)"
        assert start_idx >= lookback, (
            f"start_idx ({start_idx}) must be ≥ lookback ({lookback}) "
            "so the first observation has a full history."
        )
        assert start_idx < end_idx, "start_idx must be < end_idx"

        # ---- Store arrays ----
        self.log_returns    = log_returns.astype(np.float32)
        self.vol20_norm     = vol20_norm.astype(np.float32)
        self.vol_ratio_norm = vol_ratio_norm.astype(np.float32)
        self.vix_norm       = vix_norm.astype(np.float32)

        self.n_assets  = log_returns.shape[1]
        self.n_total   = self.n_assets + 1          # securities + cash
        self.lookback  = lookback
        self.start_idx = start_idx
        self.end_idx   = end_idx

        # ---- Spaces ----
        obs_dim = self.n_total * self.lookback
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Raw logits; softmax is applied inside step()
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_total,), dtype=np.float32
        )

        self.dsr = DifferentialSharpeRatio(eta=eta)

        # ---- State ----
        self._weights: np.ndarray = np.zeros(self.n_total, dtype=np.float32)
        self.current_step: int = start_idx

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax → valid portfolio weight vector."""
        x = np.asarray(x, dtype=np.float64)
        x = x - x.max()
        e = np.exp(x)
        w = (e / e.sum()).astype(np.float32)
        # Hard clip + renormalise to guard against floating-point drift
        w = np.clip(w, 0.0, 1.0)
        w /= w.sum()
        return w

    def _build_obs(self) -> np.ndarray:
        """Construct the (n+1, T) state matrix and return it flattened."""
        t     = self.current_step
        state = np.zeros((self.n_total, self.lookback), dtype=np.float32)

        # Column 0 — current portfolio weights
        state[:, 0] = self._weights

        # Columns 1 … lookback-1 — lagged log returns
        #   state[i, lag] = r_{i, t-lag}   for lag = 1, …, lookback-1
        for lag in range(1, self.lookback):
            hist_t = t - lag
            if hist_t >= 0:
                state[: self.n_assets, lag] = self.log_returns[hist_t]

        # Cash row — market-regime indicators
        state[self.n_assets, 1] = self.vol20_norm[t]
        state[self.n_assets, 2] = self.vol_ratio_norm[t]
        state[self.n_assets, 3] = self.vix_norm[t]

        return state.flatten()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step        = self.start_idx
        self._weights            = np.zeros(self.n_total, dtype=np.float32)
        self._weights[-1]        = 1.0      # start all-cash
        self.dsr.reset()
        return self._build_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Translate logits → valid portfolio weights
        self._weights = self._softmax(action)

        # 2. Realise portfolio return (cash earns 0 %)
        simple_rets  = np.exp(self.log_returns[self.current_step]) - 1.0
        port_return  = float(self._weights[: self.n_assets] @ simple_rets)

        # 3. Differential Sharpe Ratio reward
        reward = self.dsr.update(port_return)

        # 4. Advance timestep
        self.current_step += 1
        terminated = self.current_step >= self.end_idx

        obs = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._build_obs()
        )

        return (
            obs,
            reward,
            terminated,
            False,              # truncated — never truncated in this env
            {
                "port_return": port_return,
                "weights":     self._weights.copy(),
            },
        )

    def render(self) -> None:  # noqa: D401
        """Not implemented."""
