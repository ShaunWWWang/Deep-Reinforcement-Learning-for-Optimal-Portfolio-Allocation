"""
train_pipeline.py
=================
Training pipeline for the DRL portfolio agent.

Implements the paper's experimental setup:
    • 10 sliding windows (shifted by 1 year)
    • 5 PPO agents per window (different random seeds)
    • Best agent selected by validation-period mean reward
    • Best agent used as seed policy for the next window
    • PPO hyperparameters from Table 1

Hyperparameters (Table 1)
--------------------------
training_timesteps  7.5 M
n_envs              10
n_steps             756       (= 252 × 3 per env; total rollout = 7 560)
batch_size          1 260     (= 252 × 5)
n_epochs            16
gamma               0.9
gae_lambda          0.9
clip_range          0.25
learning_rate       3e-4 annealed linearly to 1e-5
Policy net          [64, 64] FC, tanh, log_std_init = -1
"""

from __future__ import annotations

import os
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from data_utils import DataBundle, WindowConfig, date_to_idx, LOOKBACK
from environment import PortfolioEnv


# ============================================================
#  Hyperparameters (Table 1)
# ============================================================

TOTAL_TIMESTEPS = 100_000
N_ENVS          = 10
N_STEPS         = 756          # per environment; total rollout = 756 * 10 = 7 560
BATCH_SIZE      = 1_260        # = 252 * 5
N_EPOCHS        = 16
GAMMA           = 0.9
GAE_LAMBDA      = 0.9
CLIP_RANGE      = 0.25
LR_START        = 3e-4
LR_END          = 1e-5
N_SEEDS         = 1            # agents trained per window

POLICY_KWARGS = dict(
    net_arch       = dict(pi=[64, 64], vf=[64, 64]),
    activation_fn  = th.nn.Tanh,
    log_std_init   = -1.0,
)

# Frequency (in env steps) at which we evaluate on the validation set
EVAL_FREQ = max(TOTAL_TIMESTEPS // (N_ENVS * 20), 1)   # ~20 evaluations


# ============================================================
#  Learning-rate schedule
# ============================================================

def _linear_schedule(initial: float, final: float):
    """
    Returns a callable accepted by SB3 as a learning_rate.
    progress_remaining: 1.0 at start → 0.0 at end of training.
    """
    def schedule(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)
    return schedule


# ============================================================
#  Environment factories
# ============================================================

def _make_env(
    bundle:     DataBundle,
    start_idx:  int,
    end_idx:    int,
    seed:       int,
) -> PortfolioEnv:
    """Create and wrap a single monitored PortfolioEnv."""
    env = PortfolioEnv(
        log_returns    = bundle.log_returns,
        vol20_norm     = bundle.vol20_norm,
        vol_ratio_norm = bundle.vol_ratio_norm,
        vix_norm       = bundle.vix_norm,
        start_idx      = start_idx,
        end_idx        = end_idx,
        lookback       = LOOKBACK,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def _create_train_vec_env(
    bundle:          DataBundle,
    train_start_idx: int,
    train_end_idx:   int,
    base_seed:       int = 0,
    use_subproc:     bool = True,
) -> SubprocVecEnv | DummyVecEnv:
    """
    Create N_ENVS parallel training environments.

    Falls back to DummyVecEnv (single-process) if SubprocVecEnv fails
    (e.g., on Windows or in notebooks).
    """
    fns = [
        partial(_make_env, bundle, train_start_idx, train_end_idx,
                seed=base_seed + i)
        for i in range(N_ENVS)
    ]
    if use_subproc:
        try:
            return SubprocVecEnv(fns)
        except Exception as exc:
            print(f"  [train] SubprocVecEnv failed ({exc}); "
                  "falling back to DummyVecEnv.")
    return DummyVecEnv(fns)


def _create_eval_env(
    bundle:       DataBundle,
    eval_start:   int,
    eval_end:     int,
    seed:         int = 9999,
) -> PortfolioEnv:
    """Create a single monitored evaluation environment."""
    env = PortfolioEnv(
        log_returns    = bundle.log_returns,
        vol20_norm     = bundle.vol20_norm,
        vol_ratio_norm = bundle.vol_ratio_norm,
        vix_norm       = bundle.vix_norm,
        start_idx      = eval_start,
        end_idx        = eval_end,
        lookback       = LOOKBACK,
    )
    return Monitor(env)


# ============================================================
#  Single-agent training
# ============================================================

def train_single_agent(
    bundle:          DataBundle,
    train_start_idx: int,
    train_end_idx:   int,
    val_start_idx:   int,
    val_end_idx:     int,
    seed:            int,
    log_dir:         str,
    seed_model_path: Optional[str] = None,
) -> Tuple[PPO, float]:
    """
    Train one PPO agent and return (model, mean_val_reward).

    If `seed_model_path` is provided and the file exists, the model is
    warm-started from those weights (policy transfer across windows).

    Parameters
    ----------
    bundle          : DataBundle
    train_start_idx : int   First training step.
    train_end_idx   : int   One past last training step.
    val_start_idx   : int   First validation step.
    val_end_idx     : int   One past last validation step.
    seed            : int   Random seed for reproducibility.
    log_dir         : str   Directory for tensorboard logs and checkpoints.
    seed_model_path : str | None   Path (without .zip) to seed model.

    Returns
    -------
    (PPO, float)   Trained model and its mean validation reward.
    """
    os.makedirs(log_dir, exist_ok=True)

    # ---- Environments ----
    vec_env  = _create_train_vec_env(bundle, train_start_idx, train_end_idx,
                                     base_seed=seed * 100)
    eval_env = _create_eval_env(bundle, val_start_idx, val_end_idx, seed=seed + 9999)

    # ---- Model ----
    lr_schedule = _linear_schedule(LR_START, LR_END)

    if seed_model_path and os.path.exists(seed_model_path + ".zip"):
        print(f"      Loading seed policy: {seed_model_path}")
        model = PPO.load(
            seed_model_path,
            env     = vec_env,
            device  = "auto",
            # Override schedule so annealing restarts correctly
            custom_objects = {"learning_rate": lr_schedule,
                              "clip_range": CLIP_RANGE},
        )
        model.set_env(vec_env)
    else:
        model = PPO(
            policy        = "MlpPolicy",
            env           = vec_env,
            learning_rate = lr_schedule,
            n_steps       = N_STEPS,
            batch_size    = BATCH_SIZE,
            n_epochs      = N_EPOCHS,
            gamma         = GAMMA,
            gae_lambda    = GAE_LAMBDA,
            clip_range    = CLIP_RANGE,
            policy_kwargs = POLICY_KWARGS,
            verbose       = 0,
            seed          = seed,
            device        = "auto",
        )

    # ---- Callbacks ----
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = log_dir,
        log_path             = log_dir,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = 1,
        deterministic        = True,
        verbose              = 0,
    )

    # ---- Train ----
    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = eval_cb,
        reset_num_timesteps = True,
        progress_bar        = True,
    )

    # ---- Load best checkpoint ----
    best_ckpt = os.path.join(log_dir, "best_model")
    if os.path.exists(best_ckpt + ".zip"):
        model = PPO.load(best_ckpt, env=vec_env, device="auto")

    # ---- Final validation score ----
    val_reward = _evaluate_agent(model, eval_env)

    vec_env.close()

    return model, val_reward


# ============================================================
#  Evaluation helper
# ============================================================

def _evaluate_agent(
    model:      PPO,
    env:        PortfolioEnv,
    n_episodes: int = 1,
) -> float:
    """
    Run the model deterministically on `env` and return mean episode reward.
    """
    total = 0.0
    for _ in range(n_episodes):
        obs, _  = env.reset()
        done    = False
        ep_rew  = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_rew   += rew
            done      = terminated or truncated
        total += ep_rew
    return total / n_episodes


# ============================================================
#  Full pipeline: 10 windows × 5 seeds
# ============================================================

def run_training_pipeline(
    bundle:     DataBundle,
    windows:    List[WindowConfig],
    output_dir: str = "models",
) -> List[Optional[str]]:
    """
    Train agents across all 10 windows.

    Algorithm
    ---------
    For each window:
        1. Train N_SEEDS agents (possibly warm-started from prior best).
        2. Evaluate each on the validation period.
        3. Keep the best-performing agent.
        4. Save it and pass its path as seed policy to the next window.

    Parameters
    ----------
    bundle      : DataBundle
    windows     : list of WindowConfig
    output_dir  : str   Root directory for model artefacts.

    Returns
    -------
    List of model paths (one per window, None if training was skipped).
    """
    os.makedirs(output_dir, exist_ok=True)
    best_model_paths: List[Optional[str]] = []
    seed_policy_path: Optional[str]       = None

    for window in windows:
        print(f"\n{'='*65}")
        print(f"  Window {window.window_id:2d}  |  "
              f"train [{window.train_start}, {window.train_end})  |  "
              f"val {window.val_start[:4]}  |  test {window.test_start[:4]}")
        print(f"{'='*65}")

        # Integer indices into bundle arrays
        train_s = max(date_to_idx(bundle, window.train_start), LOOKBACK)
        train_e =     date_to_idx(bundle, window.train_end)
        val_s   = max(date_to_idx(bundle, window.val_start),   LOOKBACK)
        val_e   =     date_to_idx(bundle, window.val_end)

        if train_s >= train_e or val_s >= val_e:
            print(f"  Skipping: insufficient data in bundle.")
            best_model_paths.append(None)
            continue

        print(f"  Train indices: [{train_s}, {train_e})  "
              f"({train_e - train_s} steps ≈ "
              f"{(train_e - train_s) / 252:.1f} yrs)")

        best_model:  Optional[PPO] = None
        best_reward: float         = -np.inf
        best_path:   Optional[str] = None

        for seed_idx in range(N_SEEDS):
            seed    = window.window_id * 100 + seed_idx
            log_dir = os.path.join(
                output_dir,
                f"window_{window.window_id:02d}",
                f"seed_{seed_idx}",
            )
            save_path = os.path.join(log_dir, f"final_w{window.window_id:02d}_s{seed_idx}")

            print(f"\n  Agent {seed_idx + 1}/{N_SEEDS}  (seed={seed})")

            # Skip if already trained
            if os.path.exists(save_path + ".zip"):
                print(f"    Found existing model — skipping training.")
                existing = PPO.load(save_path, device="auto")
                eval_env = _create_eval_env(bundle, val_s, val_e, seed=seed + 9999)
                val_reward = _evaluate_agent(existing, eval_env)
                print(f"    Loaded val reward: {val_reward:.4f}")
                model = existing
            else:
                model, val_reward = train_single_agent(
                    bundle          = bundle,
                    train_start_idx = train_s,
                    train_end_idx   = train_e,
                    val_start_idx   = val_s,
                    val_end_idx     = val_e,
                    seed            = seed,
                    log_dir         = log_dir,
                    seed_model_path = seed_policy_path,
                )
                model.save(save_path)
                print(f"    Saved: {save_path}")

            print(f"    Val reward: {val_reward:.4f}")

            if val_reward > best_reward:
                best_reward = val_reward
                best_model  = model
                best_path   = save_path

        # Save best model for this window
        if best_model is not None:
            window_best = os.path.join(
                output_dir,
                f"window_{window.window_id:02d}",
                f"best_window_{window.window_id:02d}",
            )
            best_model.save(window_best)
            print(f"\n  ★ Best agent reward={best_reward:.4f} → {window_best}")
            best_model_paths.append(window_best)
            seed_policy_path = window_best          # warm-start next window
        else:
            best_model_paths.append(None)

    return best_model_paths
