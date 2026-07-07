# SBCA: Cross-Modal BERT-driven Actor-Critic for Multi-Asset Portfolio Optimization

This repository contains the official implementation of the paper:

> **SBCA: Cross-Modal BERT-driven Actor-Critic for Multi-Asset Portfolio Optimization**
>
> Jinfeng Pan, Mi Liu, Xiao Ping, Jiahao Chen
>
> Submitted to *Financial Innovation*

## Overview

SBCA is a deep reinforcement learning framework for dynamic portfolio optimization that integrates price time-series with BERT-based financial news sentiment through a gated cross-modal fusion mechanism, trained with a risk-sensitive Actor-Critic architecture.


## Repository Structure

```
.
├── code/                          # Experiment scripts
│   ├── exp1_main_G.py             # Main results (SBCA vs. 6 baselines × 11 groups)
│   ├── exp2_ablation.py           # Ablation study (5 SBCA variants)
│   ├── exp3_ema.py                # EMA smoothing coefficient sensitivity
│   ├── exp4_turnover.py           # Turnover penalty coefficient sensitivity
│   ├── exp5_gamma.py              # Discount factor γ sensitivity
│   ├── exp6_lr.py                 # Learning rate sensitivity
│   ├── exp7_cost.py               # Transaction cost sensitivity (0.1%–1.0%)
│   ├── exp8_regime.py             # Market regime decomposition
│   ├── exp9_seed.py               # Multi-seed stability (5 seeds × 11 groups)
│   ├── ato_baselines.py           # Annual turnover computation for all baselines
│   ├── bootstrap_ci.py            # Bootstrap confidence intervals
│   ├── train_pv_data.py           # Portfolio value trajectory collection
│   └── plot_pv_curves.py          # Visualization utilities
├── data/                          # Input data
│   ├── bert_pred_24stocks.csv     # BERT sentiment scores for 24 stocks
│   └── bert_pred_24stocks_delta_G.csv  # Delta BERT scores
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Environment Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA 12.x (tested on RTX 3060 12GB)
- Windows 11 / Linux

### Installation

```bash
# Create conda environment
conda create -n SBCA python=3.10
conda activate SBCA

# Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

## Data

The dataset covers 24 U.S. stocks (8 large-cap, 8 mid-cap, 8 small-cap) from January 2012 to December 2022. Daily price data are sourced from public financial APIs, and news headlines are processed through a fine-tuned BERT model (bert-base-uncased) to produce sentiment scores (delta_bert ∈ [0, 1]).

The dataset and code are publicly available at:
[https://anonymous.4open.science/r/SBCA-17EC](https://anonymous.4open.science/r/SBCA-17EC)

## Running Experiments

All experiments follow a unified workflow: train models → evaluate on test set → save results to `result/`.

### Main Results (Table 1)
```bash
python code/exp1_main_G.py
```
Trains SBCA and all baselines (PPO, MinVar, Risk-Parity, EW, BH, DJ) across 11 portfolio groups.

### Ablation Study (Table 4)
```bash
python code/exp2_ablation.py
```
Evaluates five SBCA variants (SB-P, S-BERT, SBC, SBA, SBCA) on 8Large, 8Small, and 12Mix.

### Robustness Checks (Tables 5–6, Appendix)
```bash
python code/exp3_ema.py            # EMA smoothing coefficient grid search
python code/exp4_turnover.py       # Turnover penalty coefficient grid search
python code/exp5_gamma.py          # Discount factor grid search
python code/exp6_lr.py             # Learning rate grid search
python code/exp7_cost.py           # Transaction cost sensitivity (0.1%–1.0%)
python code/exp8_regime.py         # Market regime decomposition (COVID/Bull/Bear)
python code/exp9_seed.py           # Multi-seed stability (5 seeds × 11 groups)
```

### Annual Turnover
```bash
python code/ato_baselines.py        # Compute ATO for all baselines + PPO
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Price window | 30 trading days |
| Commission rate | 0.25% |
| Risk-free rate | 2% annual |
| Discount factor γ | 0.99 |
| Risk penalty λ_risk | 0.1 |
| Turnover penalty λ_turnover | 0.005 |
| EMA smoothing α | 0.4 |
| Entropy regularization β | 0.1 |
| Optimizer | AdamW (lr=3×10⁻⁴, weight decay=1×10⁻⁵) |
| Max epochs | 30 (early stopping, patience=5) |

## Results

Output files are saved to the `result/` directory in CSV format. Key metrics include: Final Portfolio Value (PV), Annual Return (AR), Sharpe Ratio (SR), Sortino Ratio, Maximum Drawdown (MDD), Calmar Ratio, and Annual Turnover (ATO).

## Citation

```bibtex
@article{pan2026sbca,
  title={SBCA: Cross-Modal BERT-driven Actor-Critic for Multi-Asset Portfolio Optimization},
  author={Jinfeng Pan, Mi Liu, Xiao Ping, Jiahao Chen},
  year={2026},
}
```

## License

This project is available for academic and research purposes. See the paper for details on data and code availability.
