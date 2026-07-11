# SBCA Redo — Final Experiments

## Changes from original

1. **FinBERT sentiment**: Precomputed scores from `bert_pred_24stocks_delta_FinBERT.csv`
2. **Hidden dimension 128**: CM module uses h=128; PPO backbone 256→128
3. **Updated stock universe**: 24 US stocks per S&P 2012 criteria, 9 GICS sectors, with news coverage documentation
4. **7 ablation variants**: SB-P, S-BERT, SB-A, SB-rand, SBA, SBC, SBCA
5. **Reward decomposition**: 2×2 factorial (Full/NoRisk/NoTurn/LogOnly) + SBC actor-only baseline

## Repository Structure

```
.
├── code/                          # Experiment scripts
│   ├── exp1_main_G.py             # Main results (SBCA vs. 6 baselines × 11 groups)
│   ├── exp1_ato.py                # ATO computation + merge into exp1_main_G.csv
│   ├── exp2_ablation_v2.py        # Ablation study (7 SBCA variants × 11 groups)
│   ├── exp3_ema.py                # EMA smoothing coefficient sweep
│   ├── exp4_turnover.py           # Turnover penalty coefficient sweep
│   ├── exp5_gamma.py              # Discount factor sweep
│   ├── exp6_lr.py                 # Learning rate sweep
│   ├── exp7_cost.py               # Cost sensitivity (0.1%–1.0% commission)
│   ├── exp8_regime.py             # Market regime decomposition
│   ├── exp9_seed.py               # SBCA multi-seed stability (5 seeds)
│   ├── reward_decomp.py           # Reward function decomposition (2×2 factorial)
│   ├── sbp_seed.py                # SB-P multi-seed stability (5 seeds × 11 groups)
│   └── finbert_score.py           # FinBERT sentiment extraction
├── data/
│   ├── bert_pred_24stocks_NEW.csv             # Raw data with headlines
│   └── bert_pred_24stocks_delta_FinBERT.csv   # FinBERT sentiment scores
├── result/                        # Experiment outputs (CSV)
├── Plots/                         # Generated figures
├── requirements.txt               # Python dependencies
├── elsarticle-template-harv.tex   # LaTeX source
├── sn-bibliography.bib            # Bibliography
└── README.md
```

## Environment Setup

```bash
conda create -n SBCA python=3.10
conda activate SBCA
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Requirements: Python 3.10+, CUDA 12.1, RTX 3060 12GB (tested on Windows 11).

## Experiments

All scripts run from project root. Output saved to `result/`.

| Script | Description | ~Time |
|:--|:--|:--|
| `exp1_main_G.py` | Main results + ATO | 30min |
| `exp2_ablation_v2.py` | 7-variant ablation | 60min |
| `exp3_ema.py` | EMA α ∈ [0.2, 0.6] | 20min |
| `exp4_turnover.py` | λ_turnover ∈ [0.001, 0.05] | 20min |
| `exp5_gamma.py` | γ ∈ [0.90, 0.99] | 20min |
| `exp6_lr.py` | η ∈ [10⁻⁴, 10⁻³] | 20min |
| `exp7_cost.py` | Commission 0.1%–1.0% | 90min |
| `exp8_regime.py` | COVID/Bull/Bear regimes | 30min |
| `exp9_seed.py` | SBCA 5-seed stability | 30min |
| `reward_decomp.py` | 2×2 factorial reward test | 90min |
| `sbp_seed.py` | SB-P 5-seed stability | 60min |

## Hyperparameters

| Parameter | Value |
|:--|:--|
| Price window W | 30 trading days |
| Commission rate | 0.25% |
| Risk-free rate | 2% annual |
| Discount factor γ | 0.99 |
| Risk penalty λ_risk | 0.1 |
| Turnover penalty λ_turnover | 0.005 |
| EMA smoothing α | 0.4 |
| Entropy β (SBCA) | 0.1 |
| Entropy β (PPO) | 0.01 |
| Hidden dimension | 128 |
| Optimizer | AdamW (lr=3×10⁻⁴, wd=1×10⁻⁵) |
| Epochs | 30 (early stopping, patience=5) |
| Seed | 42 |

## Key Results

SBCA achieves SR comparable to PPO (ΔSR +0.003, p=0.21) with 80× lower turnover (0.010 vs 0.80). The AC architecture enforces 92–99.9% turnover reduction; BERT sentiment provides conditional gains; gated CM fusion serves as a safeguard. Performance is insensitive to hyperparameters (SR range < 0.006), transaction costs (SR Δ < 0.0003), and random seeds (CV 0.02% vs 5.33% for SB-P).
