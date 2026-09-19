"""
=============================================================================
SBCA Complete Experiment Suite — Merged & Consolidated
=============================================================================
Sections:
  1. Imports & Global Config
  2. Data Loading
  3. Environment
  4. Model Definitions (SBCA 4 variants only)
  5. Metrics & Helper Functions
  6. Training Functions (SBCA variants)
  7. Baselines
     ├── EW / BH / DJ (+ daily log returns)
     ├── MVO (γ=2) (+ daily log returns)
     └── PPO (agent + training + PV)
  8. Bootstrap Significance Testing
  9. Sensitivity Tests (window, EMA, multi-seed)
  10. Pure Cost Sensitivity
  11. Plotting Helpers
  12. Main Orchestrator
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os, copy, warnings, random
import logging
from datetime import datetime
from tqdm import tqdm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ===========================================================================
# SECTION 1: Global Config
# ===========================================================================
os.makedirs("./logs", exist_ok=True)
log_filename = f"./logs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()]
)
logging.info("Experiment started")

os.makedirs("./result", exist_ok=True)
os.makedirs("./Plots", exist_ok=True)
os.makedirs("./Plots/individual_stocks", exist_ok=True)
os.makedirs("./Plots/learning_curves", exist_ok=True)
os.makedirs("./models", exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 20

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters ---
WINDOW_SIZE     = 30
COMMISSION      = 0.0025
RISK_FREE       = 0.02
GAMMA           = 0.99
LAMBDA_RISK     = 0.1
EMA_ALPHA       = 0.4
EPOCHS          = 30
PATIENCE        = 5
LR              = 3e-4
ENTROPY_COEF       = 0.1
PPO_ENTROPY_COEF   = 0.01
TURNOVER_PENALTY   = 0.005
# Bootstrap
N_BOOT          = 30000
BLOCK_SIZE      = 60

# PPO-specific
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
PPO_EPOCHS_INNER = 4
BATCH_SIZE      = 64

# Classical MVO
REBALANCE_FREQ  = 22
LOOKBACK_DAYS   = 60 * 21

# ===========================================================================
# SECTION 2: Data Loading
# ===========================================================================
csv_path = "bert_pred_for_SARL.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=["close", "delta_bert"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Stock_symbol", "Date"]).reset_index(drop=True)
df_close = df.pivot_table(index="Date", columns="Stock_symbol", values="close").ffill().bfill()
df_delta = df.pivot_table(index="Date", columns="Stock_symbol", values="delta_bert").fillna(0.5)
all_stocks = df_close.columns.tolist()
date_index = df_close.index
DAYS = len(df_close)

train_end_date = pd.to_datetime("2018-12-31")
val_end_date   = pd.to_datetime("2020-12-31")
train_end = date_index.get_indexer([train_end_date], method="pad")[0]
val_end   = date_index.get_indexer([val_end_date], method="pad")[0]

# ===========================================================================
# SECTION 3: Environment
# ===========================================================================
class StockEnv:
    def __init__(self, close_arr: np.ndarray, delta_arr: np.ndarray, window_size: int = WINDOW_SIZE):
        self.close = close_arr
        self.delta = delta_arr
        self.n_assets = close_arr.shape[1]
        self.days = close_arr.shape[0]
        self.window_size = window_size
        self.price_mean, self.price_std = self._calc_price_stats()

    def _calc_price_stats(self):
        rets = []
        w = self.window_size
        for t in range(w, train_end):
            pw = self.close[t - w:t]
            ret = np.diff(pw, axis=0) / (pw[:-1] + 1e-8)
            rets.append(ret.flatten())
        rets = np.concatenate(rets)
        return np.mean(rets), np.std(rets) + 1e-8

    def get_state(self, t: int):
        w = self.window_size
        if t < w:
            pf = np.zeros(w * self.n_assets, dtype=np.float32)
        else:
            pw = self.close[t - w:t]
            ret = np.diff(pw, axis=0) / (pw[:-1] + 1e-8)
            ret = np.concatenate([np.zeros((1, self.n_assets)), ret], axis=0)
            pf = ((ret.flatten() - self.price_mean) / self.price_std).astype(np.float32)
        tf = ((self.delta[t].copy() - 0.5) * 2).astype(np.float32)
        return pf, tf


# ===========================================================================
# SECTION 4: Model Definitions
# ===========================================================================
def _trim_w(w: torch.Tensor, n_assets: int) -> torch.Tensor:
    return w[..., :n_assets]

class CrossModalFusion(nn.Module):
    """Cross-modal gated fusion: pf * (1 + tanh(gate(tf))) + tf"""
    def __init__(self, price_dim: int, text_dim: int, hidden: int = 64):
        super().__init__()
        self.price_enc = nn.Sequential(nn.Linear(price_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.text_enc  = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.scale = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.fuse  = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pf = self.price_enc(p)
        tf = self.text_enc(t)
        scaled = pf * (1 + self.scale(tf))
        return self.fuse(scaled + tf)


# ---- SBCA Variants (4 combinations: ±CM × ±AC) ----
class Policy_BERT(nn.Module):
    """No CM, No AC: simple MLP → softmax"""
    def __init__(self, state_dim, n_assets):
        super().__init__()
        self.n_assets = n_assets
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_assets), nn.Softmax(-1),
        )

    def forward(self, x):
        return self.net(x)


class Policy_BERT_AC(nn.Module):
    """No CM, With AC: shared backbone + actor + critic"""
    def __init__(self, state_dim, n_assets):
        super().__init__()
        self.n_assets = n_assets
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.actor  = nn.Sequential(nn.Linear(64, n_assets), nn.Softmax(-1))
        self.critic = nn.Sequential(nn.Linear(64, 1))

    def forward(self, x):
        f = self.backbone(x)
        return self.actor(f), self.critic(f)


class Policy_BERT_CM(nn.Module):
    """With CM, No AC: CM → head → softmax"""
    def __init__(self, win_size, n_assets):
        super().__init__()
        self.n_assets = n_assets
        self.cm   = CrossModalFusion(win_size * n_assets, n_assets)
        self.head = nn.Sequential(nn.Linear(64, n_assets), nn.Softmax(-1))

    def forward(self, p, t):
        return self.head(self.cm(p, t))


class Policy_BERT_CM_AC(nn.Module):
    """With CM, With AC (SBCA Full Model): CM → actor + critic"""
    def __init__(self, win_size, n_assets):
        super().__init__()
        self.n_assets = n_assets
        self.cm     = CrossModalFusion(win_size * n_assets, n_assets)
        self.actor  = nn.Sequential(nn.Linear(64, n_assets), nn.Softmax(-1))
        self.critic = nn.Sequential(nn.Linear(64, 1))

    def forward(self, p, t):
        f = self.cm(p, t)
        return self.actor(f), self.critic(f)


# Alias for the full SBCA model (CM+AC). Used by: bootstrap (§8), sensitivity (§9), cost_sensitivity (§10)
SBCA = Policy_BERT_CM_AC


# ===========================================================================
# SECTION 5: Metrics & Helper Functions
# ===========================================================================
def calc_log_returns(pv):
    return np.diff(np.log(pv))

def calc_sharpe(pv, commission=COMMISSION):
    ret = calc_log_returns(pv)
    daily_rf = RISK_FREE / 252
    return (np.mean(ret - daily_rf) / (np.std(ret) + 1e-8)) * np.sqrt(252)

def sortino_ratio(pv):
    ret = calc_log_returns(pv)
    down_ret = ret[ret < 0]
    if len(down_ret) == 0:
        return 10.0
    downside_std = np.std(down_ret) + 1e-8
    return np.mean(ret) / downside_std * np.sqrt(252)

def max_drawdown(pv):
    peak = np.maximum.accumulate(pv)
    return np.min((pv - peak) / (peak + 1e-8))

def annual_return(pv):
    return pv[-1] ** (252 / len(pv)) - 1

def calmar_ratio(pv):
    mdd = abs(max_drawdown(pv))
    return annual_return(pv) / (mdd + 1e-8)

def evaluate(pv):
    pv = np.array(pv)
    return dict(
        PV      = round(float(pv[-1]), 4),
        AR      = round(annual_return(pv), 4),
        SR      = round(calc_sharpe(pv), 4),
        Sortino = round(sortino_ratio(pv), 4),
        MDD     = round(max_drawdown(pv), 4),
        Calmar  = round(calmar_ratio(pv), 4),
    )

def compute_metrics_from_log_rets(log_rets):
    """Compute AR, SR, Sortino, MDD from daily log returns (bootstrap version)."""
    daily_rf = RISK_FREE / 252
    excess = log_rets - daily_rf
    ar = np.exp(np.mean(log_rets) * 252) - 1
    sr = np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252)
    downside_ret = log_rets[log_rets < 0]
    if len(downside_ret) == 0:
        sortino = 10.0
    else:
        downside_std = np.std(downside_ret) + 1e-8
        sortino = np.mean(log_rets) / downside_std * np.sqrt(252)
    cum = np.exp(np.cumsum(log_rets))
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    mdd = np.min(drawdown)
    return {"AR": ar, "SR": sr, "Sortino": sortino, "MDD": mdd}


def norm_adv(adv: torch.Tensor) -> torch.Tensor:
    if adv.numel() <= 1:
        return adv
    return (adv - adv.mean()) / (adv.std() + 1e-8)

def risk_adjusted_reward(w: torch.Tensor, y: torch.Tensor, w_old: torch.Tensor, commission: float):
    to = torch.sum(torch.abs(w - w_old)) / 2.0
    port_gross = torch.sum(w * y)
    net_return = port_gross - commission * to
    logr = torch.log(net_return.clamp(min=1e-4))
    downside = torch.clamp(-logr, min=0.0)
    risk = downside ** 2
    return logr, risk, to


# ===========================================================================
# SECTION 6: Training Functions (SBCA Variants)
# ===========================================================================
def _forward(model, pf, tf, is_cm, is_ac, device):
    """Unified forward for all 4 SBCA variants."""
    if is_cm:
        pt = torch.from_numpy(pf).unsqueeze(0).to(device)
        tt = torch.from_numpy(tf).unsqueeze(0).to(device)
        out = model(pt, tt)
    else:
        s = torch.from_numpy(np.concatenate([pf, tf])).unsqueeze(0).to(device)
        out = model(s)
    if is_ac:
        w_raw, v = out[0].squeeze(0), out[1]
    else:
        w_raw, v = out.squeeze(0), None
    w = torch.softmax(w_raw, dim=-1)
    w = _trim_w(w, model.n_assets)
    return w, v


def _val_pv(env, model, val_range, is_cm, is_ac, commission=COMMISSION):
    """Validation PV for SBCA variants."""
    model.eval()
    pv = 1.0
    w_old = np.ones(env.n_assets) / env.n_assets
    with torch.no_grad():
        for t in val_range:
            if t + 1 >= env.days: break
            pf, tf = env.get_state(t)
            w, _ = _forward(model, pf, tf, is_cm, is_ac, DEVICE)
            wn = w.cpu().numpy()
            w_smooth = EMA_ALPHA * wn + (1 - EMA_ALPHA) * w_old
            to = np.sum(np.abs(w_smooth - w_old)) / 2.0
            y_arr = env.close[t + 1] / env.close[t]
            gr = np.sum(w_smooth * y_arr)
            pv = max(pv * (gr - commission * to), 1e-4)
            w_old = w_smooth
    return pv


def _train_loop(env, model, opt, train_range, val_range, is_cm, is_ac, desc, name, suffix,
                commission=COMMISSION, scheduler=None, ema_alpha=EMA_ALPHA, lean=False):
    """Training loop for all SBCA variants.

    When lean=True: skip step logging, plots, and model saving (for bootstrap/sensitivity)."""
    best_val_pv = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_count = 0
    train_losses, val_pvs = [], []
    step_loss_buf, step_loss_list, step_val_list = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        steps = 0
        w_old = torch.ones(env.n_assets).to(DEVICE) / env.n_assets
        for t in tqdm(train_range, desc=f"{desc} E{epoch+1}/{EPOCHS}", leave=False):
            if t + 1 >= env.days: break
            pf, tf = env.get_state(t)
            w, v = _forward(model, pf, tf, is_cm, is_ac, DEVICE)
            y = torch.from_numpy(env.close[t + 1] / env.close[t]).float().to(DEVICE)
            logr, risk, to = risk_adjusted_reward(w, y, w_old, commission)
            entropy = -torch.sum(w * torch.log(w + 1e-8))
            if is_ac:
                pf_n, tf_n = env.get_state(min(t + 1, env.days - 1))
                with torch.no_grad():
                    _, v_next = _forward(model, pf_n, tf_n, is_cm, is_ac, DEVICE)
                target = logr + GAMMA * v_next.squeeze()
                adv = norm_adv((target - v.squeeze()).detach())
                loss = (-adv * logr + 0.5 * nn.functional.mse_loss(v.squeeze(), target.detach())
                        + TURNOVER_PENALTY * to + LAMBDA_RISK * risk - ENTROPY_COEF * entropy)
            else:
                loss = -logr + TURNOVER_PENALTY * to + LAMBDA_RISK * risk - ENTROPY_COEF * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            w_old = ema_alpha * w.detach() + (1 - ema_alpha) * w_old
            epoch_loss += loss.item()
            steps += 1
            if not lean:
                step_loss_buf.append(loss.item())
                if len(step_loss_buf) >= 100:
                    avg_loss = np.mean(step_loss_buf)
                    step_loss_list.append(avg_loss)
                    fast_val = _val_pv(env, model, val_range, is_cm, is_ac, commission)
                    step_val_list.append(fast_val)
                    step_loss_buf.clear()

        avg_loss = epoch_loss / max(steps, 1)
        val_pv = _val_pv(env, model, val_range, is_cm, is_ac, commission)
        train_losses.append(avg_loss)
        val_pvs.append(val_pv)
        if scheduler:
            scheduler.step(val_pv)
        if not lean:
            print(f"[{desc}] E{epoch+1} loss={avg_loss:.5f} val_pv={val_pv:.4f}")
        if val_pv > best_val_pv:
            best_val_pv = val_pv
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
            if not lean:
                torch.save(model.state_dict(), f"./models/best_{name}{suffix}.pth")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                if not lean:
                    print(f"Early stop at E{epoch+1}")
                break

    if not lean:
        _plot_learning_curve(name, suffix, train_losses, val_pvs, best_epoch)
        _plot_step_curve(name, suffix, step_loss_list, step_val_list)
    model.load_state_dict(best_state)
    return model


def _backtest(env, model, test_range, is_cm, is_ac, commission=COMMISSION, ema_alpha=EMA_ALPHA):
    """Backtest for SBCA variants, returns (pv_array, daily_returns, weight_df)."""
    pv, rets, ws = [1.0], [], []
    w_old = np.ones(env.n_assets) / env.n_assets
    model.eval()
    with torch.no_grad():
        for t in test_range:
            if t + 1 >= env.days: break
            pf, tf = env.get_state(t)
            w, _ = _forward(model, pf, tf, is_cm, is_ac, DEVICE)
            wn = w.cpu().numpy()
            w_smooth = ema_alpha * wn + (1 - ema_alpha) * w_old
            to = np.sum(np.abs(w_smooth - w_old)) / 2.0
            y_arr = env.close[t + 1] / env.close[t]
            gr = np.sum(w_smooth * y_arr)
            net = max(pv[-1] * (gr - commission * to), 1e-4)
            pv.append(net)
            rets.append(net / pv[-2] - 1)
            ws.append(w_smooth.copy())
            w_old = w_smooth
    idx = date_index[list(test_range)[:len(ws)]]
    wdf = pd.DataFrame(ws, columns=all_stocks[:env.n_assets], index=idx)
    return np.array(pv), np.array(rets), wdf


def _make_and_train(env, model_cls, model_kwargs, is_cm, is_ac, date_range, desc, name, suffix,
                    commission=COMMISSION):
    """Factory: build, train, and backtest one SBCA variant."""
    set_seed(42)
    model = model_cls(**model_kwargs).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 1, verbose=False)
    tr, vr, te = date_range
    model = _train_loop(env, model, opt, tr, vr, is_cm, is_ac, desc, name, suffix, commission, sched)
    return _backtest(env, model, te, is_cm, is_ac, commission)


# Convenience wrappers
def run_SARL_BERT(env, dr, sf="", c=COMMISSION):
    sd = WINDOW_SIZE * env.n_assets + env.n_assets
    return _make_and_train(env, Policy_BERT, {"state_dim": sd, "n_assets": env.n_assets},
                           False, False, dr, "SARL_BERT", "SARL_BERT", sf, c)

def run_SARL_BERT_AC(env, dr, sf="", c=COMMISSION):
    sd = WINDOW_SIZE * env.n_assets + env.n_assets
    return _make_and_train(env, Policy_BERT_AC, {"state_dim": sd, "n_assets": env.n_assets},
                           False, True, dr, "SARL_BERT_AC", "SARL_BERT_AC", sf, c)

def run_SARL_BERT_CM(env, dr, sf="", c=COMMISSION):
    return _make_and_train(env, Policy_BERT_CM, {"win_size": WINDOW_SIZE, "n_assets": env.n_assets},
                           True, False, dr, "SARL_BERT_CM", "SARL_BERT_CM", sf, c)

def run_SARL_BERT_CM_AC(env, dr, sf="", c=COMMISSION):
    return _make_and_train(env, Policy_BERT_CM_AC, {"win_size": WINDOW_SIZE, "n_assets": env.n_assets},
                           True, True, dr, "SARL_BERT_CM_AC", "SARL_BERT_CM_AC", sf, c)


# ===========================================================================
# SECTION 7: Baselines (EW, BH, DJ, MVO γ=2, PPO)
# ===========================================================================
# ---- EW / BH / DJ ----
def equal_weight(env, test_range, c=COMMISSION):
    """Equal-weight with monthly rebalancing."""
    pv = [1.0]
    w = np.ones(env.n_assets) / env.n_assets
    for t in test_range:
        if t + 1 >= env.days: break
        y = env.close[t + 1] / env.close[t]
        gr = np.sum(w * y)
        pv.append(pv[-1] * gr)
        w = w * y / np.sum(w * y)
        if (t - test_range.start) % 22 == 0:
            w_new = np.ones(env.n_assets) / env.n_assets
            turnover = np.sum(np.abs(w_new - w)) / 2.0
            pv[-1] = pv[-1] * (1 - c * turnover)
            w = w_new.copy()
    return np.array(pv)


def buy_hold(env, test_range, c=COMMISSION):
    """Buy-and-hold with one-time initial cost."""
    pv = [1.0]
    w = np.ones(env.n_assets) / env.n_assets
    initial_turnover = np.sum(w) / 2.0
    pv[-1] = pv[-1] * (1 - c * initial_turnover)
    for t in test_range:
        if t + 1 >= env.days: break
        y = env.close[t + 1] / env.close[t]
        gr = np.sum(w * y)
        pv.append(pv[-1] * gr)
        w = w * y / np.sum(w * y)
    return np.array(pv)


def dow_jones_market(env, test_range, c=COMMISSION):
    """Price-weighted (Dow Jones) portfolio."""
    pv = [1.0]
    n = env.n_assets
    initial_prices = env.close[test_range[0]]
    w = initial_prices / np.sum(initial_prices)
    initial_turnover = np.sum(w) / 2.0
    pv[-1] = pv[-1] * (1 - c * initial_turnover)
    for t in test_range:
        if t + 1 >= env.days: break
        y = env.close[t + 1] / env.close[t]
        gr = np.sum(w * y)
        pv.append(pv[-1] * gr)
        w = w * y / np.sum(w * y)
    return np.array(pv)


def baseline_daily_log_rets(close_arr, test_range, strategy):
    """Daily log returns for EW/BH/DJ strategies."""
    daily = []
    n = close_arr.shape[1]
    w = np.ones(n) / n
    if strategy == "ew":
        for t in test_range:
            if t + 1 >= close_arr.shape[0]: break
            y = close_arr[t + 1] / close_arr[t]
            gr = np.sum(w * y)
            daily.append(np.log(max(gr, 1e-4)))
            w = w * y / np.sum(w * y + 1e-8)
            if (t - test_range.start) % 22 == 0:
                w_new = np.ones(n) / n
                to = np.sum(np.abs(w_new - w)) / 2.0
                daily[-1] = np.log(max(gr - COMMISSION * to, 1e-4))
                w = w_new.copy()
    elif strategy == "bh":
        ito = np.sum(np.abs(np.ones(n) / n)) / 2.0
        first = True
        for t in test_range:
            if t + 1 >= close_arr.shape[0]: break
            y = close_arr[t + 1] / close_arr[t]
            gr = np.sum(w * y)
            if first:
                daily.append(np.log(max(gr - COMMISSION * ito, 1e-4)))
                first = False
            else:
                daily.append(np.log(max(gr, 1e-4)))
            w = w * y / np.sum(w * y + 1e-8)
    elif strategy == "dj":
        w = close_arr[test_range[0]] / np.sum(close_arr[test_range[0]])
        ito = np.sum(np.abs(w)) / 2.0
        first = True
        for t in test_range:
            if t + 1 >= close_arr.shape[0]: break
            y = close_arr[t + 1] / close_arr[t]
            gr = np.sum(w * y)
            if first:
                daily.append(np.log(max(gr - COMMISSION * ito, 1e-4)))
                first = False
            else:
                daily.append(np.log(max(gr, 1e-4)))
            w = w * y / np.sum(w * y + 1e-8)
    return np.array(daily)


# ---- MVO (γ=2) ----
def _solve_mvo(mu, Sigma, gamma=2.0):
    """Mean-variance optimization with L2 regularization on covariance."""
    n = len(mu)
    Sigma_reg = Sigma + 1e-4 * np.eye(n)

    def obj(w):
        return -(mu.dot(w) - gamma / 2 * w.dot(Sigma_reg).dot(w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        return np.ones(n) / n
    w = np.maximum(res.x, 0)
    return w / (w.sum() + 1e-8)


def mvo_backtest(env, test_range, gamma=2.0, c=COMMISSION):
    """MVO with monthly rebalancing, γ=2. Returns PV array."""
    pv = [1.0]
    close_arr = env.close
    n_assets = env.n_assets
    w_prev = np.ones(n_assets) / n_assets
    for idx, t in enumerate(test_range):
        if t + 1 >= close_arr.shape[0]:
            break
        reb = (idx == 0) or (idx % REBALANCE_FREQ == 0)
        if reb:
            est_start = max(0, t - LOOKBACK_DAYS)
            if t - est_start >= 252:
                prices_est = close_arr[est_start:t]
                rets_est = pd.DataFrame(np.diff(prices_est, axis=0) / (prices_est[:-1] + 1e-8))
                mu = rets_est.mean().values * 252
                Sigma = rets_est.cov().values * 252
                w_opt = _solve_mvo(mu, Sigma, gamma)
            else:
                w_opt = np.ones(n_assets) / n_assets
            to = np.sum(np.abs(w_opt - w_prev)) / 2.0
            w_prev = w_opt.copy()
        y = close_arr[t + 1] / close_arr[t]
        gr = np.sum(w_prev * y)
        pv.append(pv[-1] * max(gr - c * to if reb else gr, 1e-4))
        w_prev = w_prev * y / np.sum(w_prev * y + 1e-8)
    return np.array(pv)


def classical_daily_log_rets(close_arr, test_range, gamma=2.0):
    """Daily log returns for MVO (γ=2)."""
    daily = []
    n_assets = close_arr.shape[1]
    w_prev = np.ones(n_assets) / n_assets
    for idx, t in enumerate(test_range):
        if t + 1 >= close_arr.shape[0]: continue
        reb = (idx == 0) or (idx % REBALANCE_FREQ == 0)
        if reb:
            est_start = max(0, t - LOOKBACK_DAYS)
            if t - est_start >= 252:
                prices_est = close_arr[est_start:t]
                rets_est = pd.DataFrame(np.diff(prices_est, axis=0) / (prices_est[:-1] + 1e-8))
                mu = rets_est.mean().values * 252
                Sigma = rets_est.cov().values * 252
                w_opt = _solve_mvo(mu, Sigma, gamma)
            else:
                w_opt = np.ones(n_assets) / n_assets
            to = np.sum(np.abs(w_opt - w_prev)) / 2.0
            w_prev = w_opt.copy()
        y = close_arr[t + 1] / close_arr[t]
        gr = np.sum(w_prev * y)
        daily.append(np.log(max(gr - COMMISSION * to if reb else gr, 1e-4)))
        w_prev = w_prev * y / np.sum(w_prev * y + 1e-8)
    return np.array(daily)


# ---- PPO Agent ----
class PPOActorCritic(nn.Module):
    """PPO with shared backbone + actor (logits) + critic + learnable log_std"""
    def __init__(self, state_dim, n_assets):
        super().__init__()
        self.n_assets = n_assets
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.actor  = nn.Linear(64, n_assets)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.ones(n_assets) * -1.0)

    def forward(self, x):
        f = self.backbone(x)
        return self.actor(f), self.critic(f)

    def get_action(self, x, deterministic=False):
        logits, value = self.forward(x)
        std = torch.exp(self.log_std)
        al = logits if deterministic else logits + torch.randn_like(logits) * std
        w = torch.softmax(al, dim=-1)
        lp = -0.5 * (((al - logits) / (std + 1e-8)) ** 2 + 2 * np.log(2 * np.pi)) - self.log_std
        return w, value.squeeze(-1), lp.sum(dim=-1), al

    def evaluate_action(self, x, old_logits):
        logits, value = self.forward(x)
        std = torch.exp(self.log_std)
        lp = -0.5 * (((old_logits - logits) / (std + 1e-8)) ** 2 + 2 * np.log(2 * np.pi)) - self.log_std
        ent = self.log_std.sum() + 0.5 * self.n_assets * (1 + np.log(2 * np.pi))
        return lp.sum(dim=-1), ent, value.squeeze(-1)


def _forward_ppo(model, pf, tf, device):
    """Dedicated forward for PPO."""
    s_np = np.concatenate([pf, tf])
    s = torch.from_numpy(s_np).unsqueeze(0).to(device)
    logits, v = model(s)
    w = torch.softmax(logits, dim=-1)
    return w, v


def _val_pv_ppo(env, model, val_range):
    """Validation PV for PPO."""
    model.eval()
    pv = 1.0
    w_old = np.ones(env.n_assets) / env.n_assets
    with torch.no_grad():
        for t in val_range:
            if t + 1 >= env.days: break
            pf, tf = env.get_state(t)
            w, _ = _forward_ppo(model, pf, tf, DEVICE)
            wn = w.cpu().numpy()
            w_smooth = EMA_ALPHA * wn + (1 - EMA_ALPHA) * w_old
            to = np.sum(np.abs(w_smooth - w_old)) / 2.0
            y_arr = env.close[t + 1] / env.close[t]
            gr = np.sum(w_smooth * y_arr)
            pv = max(pv * (gr - COMMISSION * to), 1e-4)
            w_old = w_smooth
    return pv


def compute_gae(rewards, values, next_value, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages, [adv + val for adv, val in zip(advantages, values[:-1])]


def _ppo_update_step(model, opt, buf_s, buf_a, buf_lp, buf_r, buf_v, buf_d):
    """Single PPO update from collected buffer."""
    s = torch.cat(buf_s).to(DEVICE)
    a = torch.cat(buf_a).to(DEVICE)
    olp = torch.stack(buf_lp).detach().to(DEVICE)
    r_np = [x.item() for x in buf_r]
    v_np = [x.item() for x in buf_v]
    next_value = 0.0
    adv, ret = compute_gae(r_np, v_np, next_value, buf_d)
    adv = torch.tensor(adv).to(DEVICE)
    ret = torch.tensor(ret).to(DEVICE)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(s)
    for _ in range(PPO_EPOCHS_INNER):
        idx = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            ii = idx[start:start + BATCH_SIZE]
            nlp, ent, nv = model.evaluate_action(s[ii], a[ii])
            ratio = torch.exp(nlp - olp[ii])
            s1 = ratio * adv[ii]
            s2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[ii]
            aloss = -torch.min(s1, s2).mean()
            vloss = 0.5 * ((nv - ret[ii]) ** 2).mean()
            loss = aloss + 0.5 * vloss - PPO_ENTROPY_COEF * ent.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
    return loss.item()


def train_ppo_and_get_rets(env, seed):
    """Train PPO and return daily log returns (for bootstrap)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    n_assets = env.close.shape[1]
    state_dim = WINDOW_SIZE * n_assets + n_assets
    TR = range(WINDOW_SIZE, train_end - 1)
    TE = range(val_end, len(env.close) - 1)
    VR = range(train_end, val_end - 1)

    model = PPOActorCritic(state_dim, n_assets).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    best_val_pv = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_count = 0
    collect_steps = 252

    for epoch in range(EPOCHS):
        model.train()
        w_old = torch.ones(n_assets).to(DEVICE) / n_assets
        buf_s, buf_a, buf_lp, buf_r, buf_v, buf_d = [], [], [], [], [], []
        for t in tqdm(TR, desc=f"PPO s{seed} E{epoch+1}", leave=False):
            if t + 1 >= env.days: break
            pf, tf = env.get_state(t)
            s = torch.from_numpy(np.concatenate([pf, tf])).unsqueeze(0).to(DEVICE)
            w, v, lp, al = model.get_action(s)
            w = w.squeeze(0)[:n_assets]
            y = torch.from_numpy(env.close[t + 1] / env.close[t]).float().to(DEVICE)
            logr, risk, to = risk_adjusted_reward(w, y, w_old, COMMISSION)
            reward = logr - LAMBDA_RISK * risk - TURNOVER_PENALTY * to
            buf_s.append(s.cpu()); buf_a.append(al.detach().cpu()); buf_lp.append(lp.detach().cpu())
            buf_r.append(reward.detach().cpu()); buf_v.append(v.detach().cpu()); buf_d.append(0.0)
            w_old = EMA_ALPHA * w.detach() + (1 - EMA_ALPHA) * w_old
            if len(buf_s) >= collect_steps:
                _ppo_update_step(model, opt, buf_s, buf_a, buf_lp, buf_r, buf_v, buf_d)
                buf_s.clear(); buf_a.clear(); buf_lp.clear(); buf_r.clear(); buf_v.clear(); buf_d.clear()
        if len(buf_s) > 0:
            _ppo_update_step(model, opt, buf_s, buf_a, buf_lp, buf_r, buf_v, buf_d)
        val_pv = _val_pv_ppo(env, model, VR)
        if val_pv > best_val_pv:
            best_val_pv = val_pv; best_state = copy.deepcopy(model.state_dict()); patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE: break

    model.load_state_dict(best_state)
    model.eval()
    daily = []
    w_old = np.ones(n_assets) / n_assets
    with torch.no_grad():
        for t in TE:
            if t + 1 >= env.days: break
            pf, tf = env.get_state(t)
            s = torch.from_numpy(np.concatenate([pf, tf])).unsqueeze(0).to(DEVICE)
            w, _, _, _ = model.get_action(s, deterministic=True)
            wn = w.squeeze(0).cpu().numpy()[:n_assets]
            w_smooth = EMA_ALPHA * wn + (1 - EMA_ALPHA) * w_old
            to = np.sum(np.abs(w_smooth - w_old)) / 2.0
            y_arr = env.close[t + 1] / env.close[t]
            gr = np.sum(w_smooth * y_arr)
            daily.append(np.log(max(gr - COMMISSION * to, 1e-4)))
            w_old = w_smooth
    return np.array(daily)


def run_ppo_pv(env, seed=42):
    """Train PPO and return PV array (for ablation comparison)."""
    daily = train_ppo_and_get_rets(env, seed)
    pv = [1.0]
    for r in daily:
        pv.append(pv[-1] * np.exp(r))
    return np.array(pv)


# ===========================================================================
# SECTION 8: Bootstrap Significance Testing
# ===========================================================================

# ---- Bootstrap helpers: train models, return daily log returns ----
def train_sbca_and_get_rets(env, seed):
    """Train SBCA (CM+AC) and return daily log returns."""
    set_seed(seed)
    n_assets = env.close.shape[1]
    TR = range(WINDOW_SIZE, train_end - 1)
    TE = range(val_end, len(env.close) - 1)
    VR = range(train_end, val_end - 1)

    model = SBCA(WINDOW_SIZE, n_assets).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    model = _train_loop(env, model, opt, TR, VR, True, True,
                        f"SBCA s{seed}", "SBCA", f"_bs_s{seed}",
                        COMMISSION, None, EMA_ALPHA, lean=True)

    _, simple_rets, _ = _backtest(env, model, TE, True, True, COMMISSION)
    return np.log(1 + np.array(simple_rets))


# ---- Block bootstrap test ----
def block_bootstrap_one_sided_test(rets_a, rets_b, n_boot=N_BOOT):
    """
    Circular block bootstrap paired-difference test.
    H0: metric(A) - metric(B) <= 0  (one-sided, A = SBCA, B = baseline).
    Uses H0 centering: bootstrap distribution centered at 0.
    """
    assert len(rets_a) == len(rets_b), "Return series length mismatch!"
    n = len(rets_a)
    ma_obs = compute_metrics_from_log_rets(rets_a)
    mb_obs = compute_metrics_from_log_rets(rets_b)
    obs_diff = {k: ma_obs[k] - mb_obs[k] for k in ma_obs}
    boot_raw_diff = {k: [] for k in ma_obs}

    for _ in tqdm(range(n_boot), desc="Bootstrap sampling"):
        idx = []
        while len(idx) < n:
            start = np.random.randint(0, n)
            blk = [(start + i) % n for i in range(BLOCK_SIZE)]
            idx.extend(blk)
        idx = np.array(idx[:n])
        sample_a = rets_a[idx]
        sample_b = rets_b[idx]
        ma_star = compute_metrics_from_log_rets(sample_a)
        mb_star = compute_metrics_from_log_rets(sample_b)
        for k in ma_obs:
            boot_raw_diff[k].append(ma_star[k] - mb_star[k])

    res = {}
    for metric_name in ma_obs:
        raw_arr = np.array(boot_raw_diff[metric_name])
        od = obs_diff[metric_name]
        boot_h0_dist = raw_arr - np.mean(raw_arr)  # center under H0
        p_val = np.mean(boot_h0_dist >= od)
        res[metric_name] = {"diff_true": round(od, 4), "p_value": round(p_val, 6)}
    return res


def run_bootstrap_all():
    """
    Bootstrap significance test: SBCA vs 5 baselines (EW, BH, DJ, MVO γ=2, PPO)
    across 3 asset groups × 4 metrics (AR, SR, Sortino, MDD).
    """
    print("\n" + "=" * 60)
    print("  BOOTSTRAP SIGNIFICANCE TEST")
    print("=" * 60)

    GLOBAL_RNG_SEED = 42
    random.seed(GLOBAL_RNG_SEED)
    np.random.seed(GLOBAL_RNG_SEED)

    stock_groups = [
        (all_stocks[:2], "2assets"),
        (all_stocks[:4], "4assets"),
        (all_stocks[:6], "6assets"),
    ]
    seeds = [42, 123, 456, 789, 1024]
    bootstrap_seed_for_test = seeds[0]

    baseline_defs = [
        ("ew", "Equal Weight", "simple"),
        ("bh", "Buy & Hold", "simple"),
        ("dj", "Dow Jones", "simple"),
        ("mvo", "MVO (γ=2)", "classical"),
        ("ppo", "PPO", "ppo"),
    ]

    all_rows = []
    seed_summary_rows = []

    for stocks, suffix in stock_groups:
        close_arr = df_close[stocks].values
        delta_arr = df_delta[stocks].values
        env = StockEnv(close_arr, delta_arr)
        TE = range(val_end, len(close_arr) - 1)

        print(f"\n  Group: {suffix}")

        # Train SBCA with 5 seeds
        sbca_all_seed_rets = []
        sbca_seed_metrics = []
        for seed in seeds:
            print(f"    Training SBCA seed={seed}...")
            ret_series = train_sbca_and_get_rets(env, seed)
            sbca_all_seed_rets.append(ret_series)
            met = compute_metrics_from_log_rets(ret_series)
            sbca_seed_metrics.append(met)

        df_seed_metric = pd.DataFrame(sbca_seed_metrics)
        for metric in ["AR", "SR", "Sortino", "MDD"]:
            seed_summary_rows.append({
                "Group": suffix,
                "Metric": f"{metric}_mean",
                "Value": df_seed_metric[metric].mean(),
                "Std": df_seed_metric[metric].std()
            })

        sbca_test_rets = sbca_all_seed_rets[seeds.index(bootstrap_seed_for_test)]

        for bkey, bname, btype in baseline_defs:
            print(f"    Computing baseline: {bname}...")
            if btype == "simple":
                bl_rets = baseline_daily_log_rets(close_arr, TE, bkey)
            elif btype == "classical":
                bl_rets = classical_daily_log_rets(close_arr, TE, gamma=2.0)
            elif btype == "ppo":
                ppo_all = []
                for seed in seeds:
                    ppo_all.append(train_ppo_and_get_rets(env, seed))
                bl_rets = ppo_all[seeds.index(bootstrap_seed_for_test)]

            min_len = min(len(sbca_test_rets), len(bl_rets))
            res = block_bootstrap_one_sided_test(sbca_test_rets[:min_len], bl_rets[:min_len])
            for metric, r in res.items():
                if r["p_value"] < 0.01:
                    sig = "***"
                elif r["p_value"] < 0.05:
                    sig = "**"
                elif r["p_value"] < 0.10:
                    sig = "*"
                else:
                    sig = "-"
                all_rows.append({
                    "Group": suffix,
                    "Comparison": f"SBCA vs {bname}",
                    "Metric": metric,
                    "diff_true": r["diff_true"],
                    "p_value": r["p_value"],
                    "sig": sig
                })
                print(f"      {metric:8s} | diff={r['diff_true']:8.4f} | p={r['p_value']:.4f} | {sig}")

    # Export
    out_dir = os.path.join(".", "result")
    os.makedirs(out_dir, exist_ok=True)
    df_boot = pd.DataFrame(all_rows)
    df_seed_summary = pd.DataFrame(seed_summary_rows)
    with pd.ExcelWriter(os.path.join(out_dir, "bootstrap_all.xlsx")) as writer:
        df_boot.to_excel(writer, sheet_name="BootstrapTest", index=False)
        df_seed_summary.to_excel(writer, sheet_name="SeedRobustness", index=False)
    print(f"\n  Bootstrap results saved to result/bootstrap_all.xlsx")
    return df_boot, df_seed_summary


# ===========================================================================
# SECTION 9: Sensitivity Tests
# ===========================================================================
def run_sbca_single(close_arr, delta_arr, window_size, ema_alpha, seed):
    """Run one SBCA experiment with given hyperparameters and seed."""
    set_seed(seed)
    env = StockEnv(close_arr, delta_arr, window_size=window_size)
    TR = range(window_size, train_end - 1)
    VR = range(train_end, val_end - 1)
    TE = range(val_end, len(close_arr) - 1)
    n_assets = close_arr.shape[1]

    model = SBCA(window_size, n_assets).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    desc = f"W{window_size}_a{ema_alpha}_s{seed}"
    model = _train_loop(env, model, opt, TR, VR, True, True,
                        desc, "SBCA", f"_sens_{desc}", COMMISSION, None, ema_alpha, lean=True)

    pv, _, _ = _backtest(env, model, TE, True, True, COMMISSION, ema_alpha)
    return evaluate(pv)


def test_window_sensitivity():
    """SBCA with W ∈ {10, 20, 30, 50}."""
    print("\n" + "=" * 60)
    print("  SENSITIVITY: Observation Window (W ∈ {10, 20, 30, 50})")
    print("=" * 60)
    windows = [10, 20, 30, 50]
    stock_groups = [
        (all_stocks[:2], "2assets"),
        (all_stocks[:4], "4assets"),
        (all_stocks[:6], "6assets"),
    ]
    results = []
    for stocks, suffix in stock_groups:
        close_arr = df_close[stocks].values
        delta_arr = df_delta[stocks].values
        print(f"\n  Group: {suffix} ({len(stocks)} assets)")
        for w in windows:
            print(f"    W={w}...")
            ev = run_sbca_single(close_arr, delta_arr, w, EMA_ALPHA, seed=42)
            results.append({"Group": suffix, "WindowSize": w, **ev})
            print(f"      PV={ev['PV']:.4f}  SR={ev['SR']:.4f}  MDD={ev['MDD']:.4f}")

    df_out = pd.DataFrame(results)
    df_out.to_excel("./result/sensitivity_window.xlsx", index=False)
    print(f"  Saved to result/sensitivity_window.xlsx")
    return df_out


def test_ema_sensitivity():
    """SBCA with α ∈ {0.2, 0.4, 0.6, 0.8}."""
    print("\n" + "=" * 60)
    print("  SENSITIVITY: EMA Smoothing (α ∈ {0.2, 0.4, 0.6, 0.8})")
    print("=" * 60)
    alphas = [0.2, 0.4, 0.6, 0.8]
    stock_groups = [
        (all_stocks[:2], "2assets"),
        (all_stocks[:4], "4assets"),
        (all_stocks[:6], "6assets"),
    ]
    results = []
    for stocks, suffix in stock_groups:
        close_arr = df_close[stocks].values
        delta_arr = df_delta[stocks].values
        print(f"\n  Group: {suffix} ({len(stocks)} assets)")
        for alpha in alphas:
            print(f"    α={alpha}...")
            ev = run_sbca_single(close_arr, delta_arr, WINDOW_SIZE, alpha, seed=42)
            results.append({"Group": suffix, "EMA": alpha, **ev})
            print(f"      PV={ev['PV']:.4f}  SR={ev['SR']:.4f}  MDD={ev['MDD']:.4f}")

    df_out = pd.DataFrame(results)
    df_out.to_excel("./result/sensitivity_ema.xlsx", index=False)
    print(f"  Saved to result/sensitivity_ema.xlsx")
    return df_out


def test_multiseed():
    """SBCA with 5 seeds, report mean ± std."""
    print("\n" + "=" * 60)
    print("  SENSITIVITY: Multi-Seed Stability (5 seeds)")
    print("=" * 60)
    seeds = [42, 123, 456, 789, 1024]
    stock_groups = [
        (all_stocks[:2], "2assets"),
        (all_stocks[:4], "4assets"),
        (all_stocks[:6], "6assets"),
    ]
    results = []
    for stocks, suffix in stock_groups:
        close_arr = df_close[stocks].values
        delta_arr = df_delta[stocks].values
        print(f"\n  Group: {suffix} ({len(stocks)} assets)")
        for seed in seeds:
            print(f"    seed={seed}...")
            ev = run_sbca_single(close_arr, delta_arr, WINDOW_SIZE, EMA_ALPHA, seed)
            results.append({"Group": suffix, "Seed": seed, **ev})
            print(f"      PV={ev['PV']:.4f}  SR={ev['SR']:.4f}")

    df_out = pd.DataFrame(results)
    summary = df_out.groupby("Group").agg(
        PV_mean=("PV", "mean"), PV_std=("PV", "std"),
        SR_mean=("SR", "mean"), SR_std=("SR", "std"),
        MDD_mean=("MDD", "mean"), MDD_std=("MDD", "std"),
    ).reset_index()
    for col in ["PV", "SR", "MDD"]:
        summary[col] = summary.apply(
            lambda r, c=col: f"{r[f'{c}_mean']:.4f} ± {r[f'{c}_std']:.4f}", axis=1)

    print("\n  Summary (mean ± std):")
    print(summary[["Group", "PV", "SR", "MDD"]].to_string(index=False))

    df_out.to_excel("./result/sensitivity_multiseed.xlsx", index=False)
    summary.to_excel("./result/sensitivity_multiseed_summary.xlsx", index=False)
    print(f"  Saved to result/sensitivity_multiseed*.xlsx")
    return df_out, summary


# ===========================================================================
# SECTION 10: Pure Cost Sensitivity
# ===========================================================================
def run_pure_cost_sensitivity():
    """
    Train SBCA once at 0.25%, then evaluate at c ∈ {0.1%, 0.25%, 0.5%, 1.0%}
    WITHOUT retraining. Also compute EW and BH at each rate for comparison.
    """
    print("\n" + "=" * 60)
    print("  PURE COST SENSITIVITY (Fixed Model, No Retraining)")
    print("=" * 60)

    stock_groups = [
        (all_stocks[:2], "2assets"),
        (all_stocks[:4], "4assets"),
        (all_stocks[:6], "6assets"),
    ]
    costs = [0.001, 0.0025, 0.005, 0.01]
    all_results = []

    for stocks, suffix in stock_groups:
        env = StockEnv(df_close[stocks].values, df_delta[stocks].values)
        TR = range(WINDOW_SIZE, train_end - 1)
        VR = range(train_end, val_end - 1)
        TE = range(val_end, len(df_close) - 1)
        n_assets = len(stocks)
        print(f"\n  Group: {suffix} ({n_assets} assets)")

        # Train SBCA once at 0.25%
        set_seed(42)
        model = SBCA(WINDOW_SIZE, n_assets).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        model = _train_loop(env, model, opt, TR, VR, True, True,
                            "SBCA_cs", "SBCA", f"_cost_sens{suffix}",
                            COMMISSION, None, EMA_ALPHA, lean=True)

        # Evaluate at each cost rate
        for c in costs:
            pv_sbca = _backtest(env, model, TE, True, True, c)[0]
            sr_sbca = calc_sharpe(pv_sbca)
            sr_ew   = calc_sharpe(equal_weight(env, TE, c))
            sr_bh   = calc_sharpe(buy_hold(env, TE, c))

            all_results.append([suffix, c, sr_sbca, sr_ew, sr_bh])
            print(f"    c={c:.4f}  SBCA_SR={sr_sbca:.4f}  EW_SR={sr_ew:.4f}  BH_SR={sr_bh:.4f}")

    df = pd.DataFrame(all_results, columns=["Group", "Commission", "SBCA_SR", "EW_SR", "BH_SR"])
    df.to_excel("./result/cost_sensitivity_fixed.xlsx", index=False)
    print(f"\n  Saved to result/cost_sensitivity_fixed.xlsx")
    return df


# ===========================================================================
# SECTION 11: Plotting Helpers
# ===========================================================================
def plot_signals(name, wdf, sf="", key_trade_days=None):
    px = df_close.loc[wdf.index]
    asset_cols = wdf.columns.tolist()
    n_asset = len(asset_cols)
    for s in asset_cols:
        plt.figure(figsize=(16, 6))
        plt.plot(px[s], color="black", linewidth=1.2)
        if n_asset == 1 and key_trade_days is not None:
            init_t = key_trade_days["init"]; end_t = key_trade_days["end"]
            rebalance_ts = key_trade_days["rebalance"]
            pos = px.index.get_indexer([date_index[init_t]], method="nearest")[0]
            if 0 <= pos < len(px):
                plt.scatter(px.index[pos], px[s].iloc[pos], color="red", marker="^", s=100, label="Initial Buy")
            for t in rebalance_ts:
                pos = px.index.get_indexer([date_index[t]], method="nearest")[0]
                if 0 <= pos < len(px):
                    plt.scatter(px.index[pos], px[s].iloc[pos], color="red", marker="^", s=70)
            pos = px.index.get_indexer([date_index[end_t]], method="nearest")[0]
            if 0 <= pos < len(px):
                plt.scatter(px.index[pos], px[s].iloc[pos], color="green", marker="v", s=100, label="Final Sell")
        else:
            w_diff = wdf[s].diff()
            add_mask = w_diff > 0; sub_mask = w_diff < 0
            plt.scatter(px[s][add_mask].index, px[s][add_mask], color="red", marker="^", s=70, label="Add")
            plt.scatter(px[s][sub_mask].index, px[s][sub_mask], color="green", marker="v", s=70, label="Reduce")
        plt.legend(fontsize=22); plt.grid(alpha=0.2)
        plt.locator_params(axis='x', nbins=6); plt.xticks(rotation=30, fontsize=10)
        plt.tight_layout()
        plt.savefig(f"./Plots/individual_stocks/{name}{sf}_{s}.png", dpi=120)
        plt.close()


def _plot_step_curve(name, suffix, step_losses, step_val_pvs):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Training Steps (x100)"); ax1.set_ylabel("Loss", color='tab:blue')
    ax1.plot(step_losses, color='tab:blue', linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx(); ax2.set_ylabel("Val PV", color="tab:orange")
    ax2.plot(step_val_pvs, color="tab:orange", linewidth=2)
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    fig.tight_layout()
    plt.savefig(f"./Plots/learning_curves/{name}{suffix}_step_curve.png", dpi=150)
    plt.close()


def _plot_learning_curve(name, suffix, train_losses, val_pvs, best_epoch):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.plot(train_losses, color="tab:blue", marker="o")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx(); ax2.set_ylabel("Val PV", color="tab:orange")
    ax2.plot(val_pvs, color="tab:orange", marker="s")
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Epoch={best_epoch+1}")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"./Plots/learning_curves/{name}{suffix}_curve.png", dpi=150)
    plt.close()


def plot_group_pv(pv_dict, sf=""):
    plt.figure(figsize=(16, 7))
    for n, pv in pv_dict.items():
        plt.plot(pv, linewidth=2, label=n)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"./Plots/Group_Comparison{sf}_pv.png", dpi=150)
    plt.close()




# ===========================================================================
# SECTION 12: Main Orchestrator
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SBCA Complete Experiment Suite")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "ablation", "bootstrap", "sensitivity", "cost_sensitivity_fixed"],
                        help="Experiment mode to run")
    args = parser.parse_args()

    stock_groups = [
        (all_stocks[:2], "_2assets"),
        (all_stocks[:4], "_4assets"),
        (all_stocks[:6], "_6assets"),
    ]

    # ============================================================
    # Mode 1: Ablation Study (4 SBCA variants + EW/BH/DJ + MVO γ=2 + PPO)
    # ============================================================
    if args.mode in ("all", "ablation"):
        print("\n" + "=" * 60)
        print("  ABLATION STUDY: 2×2 (CM × AC) + Baselines + MVO(γ=2) + PPO")
        print("=" * 60)

        ablation_results = []

        for stocks, suffix in stock_groups:
            print(f"\n{'='*60}\n  Asset group: {suffix[1:]}\n{'='*60}")
            env = StockEnv(df_close[stocks].values, df_delta[stocks].values)
            TR = range(WINDOW_SIZE, train_end - 1)
            VR = range(train_end, val_end - 1)
            TE = range(val_end, DAYS - 1)
            date_ranges = (TR, VR, TE)

            # 4 SBCA variants
            pv1, _, wdf1 = run_SARL_BERT(env, date_ranges, suffix)
            pv2, _, wdf2 = run_SARL_BERT_AC(env, date_ranges, suffix)
            pv3, _, wdf3 = run_SARL_BERT_CM(env, date_ranges, suffix)
            pv4, _, wdf4 = run_SARL_BERT_CM_AC(env, date_ranges, suffix)

            # Baselines
            pv_ew = equal_weight(env, TE)
            pv_bh = buy_hold(env, TE)
            pv_dj = dow_jones_market(env, TE)
            pv_mvo = mvo_backtest(env, TE, gamma=2.0)
            pv_ppo = run_ppo_pv(env)

            plot_signals("SARL_BERT", wdf1, suffix)
            plot_signals("SARL_BERT_AC", wdf2, suffix)
            plot_signals("SARL_BERT_CM", wdf3, suffix)
            plot_signals("SARL_BERT_CM_AC", wdf4, suffix)

            pv_dict = {
                "SARL_BERT": pv1, "SARL_BERT_AC": pv2,
                "SARL_BERT_CM": pv3, "SARL_BERT_CM_AC": pv4,
                "EqualWeight": pv_ew, "BuyHold": pv_bh,
                "DowJones": pv_dj, f"MVO(γ=2)": pv_mvo, "PPO": pv_ppo,
            }
            plot_group_pv(pv_dict, suffix)

            print(f"\n  {'Model':25s}  {'PV':>7}  {'SR':>7}  {'MDD':>7}")
            for tag, pv in pv_dict.items():
                ev = evaluate(pv)
                ablation_results.append([f"Ablation{suffix}", tag, *ev.values()])
                print(f"  {tag:25s}  {ev['PV']:>7.4f}  {ev['SR']:>7.3f}  {ev['MDD']:>7.3f}")

        pd.DataFrame(ablation_results,
                     columns=["Group", "Model", "PV", "AR", "SR", "Sortino", "MDD", "Calmar"]
                     ).to_excel("./result/Ablation_Complete.xlsx", index=False)

        print("\n  Ablation complete!")

    # ============================================================
    # Mode 2: Bootstrap Significance Tests
    # ============================================================
    if args.mode in ("all", "bootstrap"):
        df_boot, _ = run_bootstrap_all()

    # ============================================================
    # Mode 3: Sensitivity Tests (Window, EMA, Multi-Seed)
    # ============================================================
    if args.mode in ("all", "sensitivity"):
        test_window_sensitivity()
        test_ema_sensitivity()
        test_multiseed()
        print("\n  All sensitivity tests complete!")

    # ============================================================
    # Mode 4: Pure Cost Sensitivity (Fixed Model)
    # ============================================================
    if args.mode in ("all", "cost_sensitivity_fixed"):
        run_pure_cost_sensitivity()

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
