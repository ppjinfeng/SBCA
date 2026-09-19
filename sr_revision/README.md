# `sr_revision/` — analysis accompanying the Scientific Reports revision

This folder holds the code and data for the significance analyses added to the
Scientific Reports manuscript

> SBCA: Cross-Modal Gated Actor-Critic with Frozen BERT-Based Sentiment Features
> for Multi-Asset Portfolio Optimization

**It is not part of the `redo/` project in the parent directory.** The two are
different studies and should not be confused:

| | `sr_revision/` (this folder) | `redo/` |
| --- | --- | --- |
| Manuscript | Scientific Reports (Springer Nature), 6 assets | Elsevier template, 24 assets / 9 GICS sectors / 11 groups |
| Sentiment model | BERT-base-uncased, chronological fine-tuning split | FinBERT |
| Latent dimension | 64 | 128 |
| Ablation design | 2 × 2 (SB / SBA / SBC / SBCA) | 7 variants |
| Extra analyses | block-bootstrap significance tests, factorial decomposition | regime decomposition, ATO, reward 2 × 2 |
| Environment | PyTorch 2.6.0, CUDA 11.8 | PyTorch ≥ 2.0, CUDA 12.1 |

---

## 1. What this analysis does

The revision adds four things to the empirical protocol:

1. **A paired circular block-bootstrap test for the internal 2 × 2 ablation.**
   30,000 resamples, block length 60 trading days, one-sided, `H0`-centred, with a
   single block index applied to both series so that the pairing is preserved.
   This is the analysis behind **Table 5** of the manuscript.
2. **Benjamini–Hochberg FDR control** over the external family (60 tests) and the
   internal family (36 tests), reported in Section 6.1 and Section 6.2.
3. **A factorial decomposition** of the 2 × 2 design into the two module main
   effects and their interaction, with 95 % percentile bootstrap intervals
   (**Supplementary Table S4**).
4. **A correction of the sentiment feature construction.** The originally released
   pipeline fine-tuned BERT on the first 80 % of rows after sorting by stock rather
   than on a time window, so four of the six assets were entirely inside the
   fine-tuning set and received in-sample sentiment scores during the test period.
   The feature was re-derived with a strictly chronological split (fine-tuning only
   on headlines dated on or before 31 December 2018) and **every experiment was
   re-run**. The audit is in `results/bert/` and in **Supplementary Table S5**.

## 2. Layout

```
code/
  run_all_old.py                    full experiment suite: data loading, environment,
                                    the four SBCA variants, the EW/BH/DJ/MVO/PPO
                                    baselines, the training loop, the metrics and the
                                    original (non-vectorised) block-bootstrap test
  ablation_bootstrap_timesplit.py   re-trains the four variants on the corrected
                                    features, saves their daily net log returns, and
                                    runs the pairwise bootstrap tests (Tables 4 and 5)
  ablation_bootstrap.py             the same driver against the *pre-correction*
                                    features, kept for the audit trail
  ts_derived.py                     block-length sweep and factorial CIs (corrected)
  factorial_ci.py                   factorial contrasts with confidence intervals
  blocklength_sensitivity.py        block lengths 20 / 40 / 60 / 120 (vectorised)
  blocklength_check.py              which block length reproduces the published p-values
  blocklength_robustness.py         block length 8 vs 60 for the external baselines
  bert_timesplit.py                 re-derives the sentiment feature with a
                                    chronological fine-tuning split
  bert_audit.py                     per-stock, per-period AUC audit of the correction
  compose_figures.py                assembles the merged main figures
  smoke_test.py                     environment / import / bootstrap self-check
  tools/                            scripts that regenerate the manuscript's LaTeX
                                    tables from the CSVs (authors' convenience)

data/                               inputs
  README.md                         what is shipped and what must be downloaded
  bert_pred_for_SARL_timesplit.csv  the corrected sentiment feature (12,300 rows)

results/                            outputs used in the revised manuscript
  ablation_metrics_<group>.csv      Table 4 point estimates
  ablation_rets_<group>.csv         daily net log returns per variant — the direct
                                    input to the tests in Table 5
  ablation_pv_<group>.csv           portfolio value paths
  ablation_bootstrap_internal.xlsx  Table 5 p-values + recomputed ablation metrics
  bootstrap_all.xlsx                external-baseline bootstrap (Table 3)
  cost_sensitivity_fixed.xlsx       cost sensitivity (Table 6)
  sensitivity_{window,ema,multiseed,multiseed_summary}.xlsx
                                    hyperparameter and seed sensitivity (S1, S2)
  blocklength_sensitivity.csv       Supplementary Table S3
  factorial_ci.csv                  Supplementary Table S4
  external_bh_survivors.csv         which external tests survive BH control
  param_counts.csv                  trainable parameter counts of the four variants
  bert/                             correction audit (Supplementary Table S5)

results_pre_correction/             the pre-correction run, retained so that the
                                    before/after comparison in the response letter
                                    can be reproduced

runs/                               written by re-running the scripts (gitignored)
```

## 3. Reproducing the results

All commands are run **from this directory** (`sr_revision/`). Committed inputs
are read from `data/` and `results/`; anything a script produces goes to `runs/`
(and to `result/`, `Plots/`, `logs/`, `models/` for the training suite), all of
which are gitignored.

```bash
pip install -r requirements.txt

# 0. check the environment first -- takes about a minute and trains nothing
python code/smoke_test.py
#    -> verifies packages, the committed inputs, that every module imports, and
#       replays a 400-resample bootstrap against the committed p-values

# 1. block-length sweep and factorial confidence intervals
#    (reads results/ablation_rets_*.csv; writes runs/)
python code/ts_derived.py

# 2. linear-scale bootstrap against the four closed-form baselines
python code/blocklength_check.py
python code/blocklength_robustness.py
python code/blocklength_sensitivity.py

# 3. factorial contrasts with confidence intervals
python code/factorial_ci.py
```

Steps 1–3 reproduce **Supplementary Tables S3 and S4** and the factorial
intervals reported in Section 6.2 of the manuscript, using only the files shipped
here.

The remaining steps need inputs that are not shipped (see `data/README.md`) and
are considerably more expensive:

```bash
# 4. re-train the four ablation variants on the corrected features and run the
#    internal bootstrap tests -- ~45 min on an RTX 3060. Reproduces Tables 4 and 5.
#    Needs data/bert_pred_for_SARL_timesplit.csv (shipped) and writes runs/.
python code/ablation_bootstrap_timesplit.py

# 5. external-baseline bootstrap, cost sensitivity and hyperparameter sweeps.
#    These read the same feature table; run_all_old.py takes the path from the
#    SBCA_FEATURES environment variable, so no copy of the suite is needed:
#      Windows :  set SBCA_FEATURES=data\bert_pred_for_SARL_timesplit.csv
#      bash    :  export SBCA_FEATURES=data/bert_pred_for_SARL_timesplit.csv
python code/run_all_old.py --mode bootstrap
python code/run_all_old.py --mode cost_sensitivity_fixed
python code/run_all_old.py --mode sensitivity

# 6. re-derive the sentiment feature with a chronological fine-tuning split.
#    Needs data/stock_news_trading_data.csv; downloads bert-base-uncased.
python code/bert_timesplit.py

# 7. audit the correction (per-stock, per-period AUC). Needs the original
#    data/bert_pred_for_SARL.csv for the "before" column. Reproduces Table S5.
python code/bert_audit.py
```

`code/run_all_old.py` is the single experiment suite. It reads its feature table
from `SBCA_FEATURES` (default `data/bert_pred_for_SARL.csv`), which is how the
same code serves both the corrected and the pre-correction runs; pointing it at
`data/bert_pred_for_SARL_timesplit.csv` gives the runs reported in the revised
manuscript.

`code/tools/` holds the scripts that turn the CSVs into the LaTeX tables of the
manuscript. They are authors' convenience tooling and assume a source tree
containing the `.tex` files; they are not needed to reproduce any number.

### What the smoke test does not cover

Step 4 trains models and step 6 downloads a pre-trained encoder; neither is
exercised by `smoke_test.py`. Everything that runs purely on the committed series
is.

## 4. Statistical procedure, exactly as implemented

1. The daily net log-return series of SBCA and of the comparison strategy form a
   paired series of length `n` (503 on the test period).
2. Per replicate, block start positions are drawn uniformly from `{0, …, n−1}`,
   each block has the stated length, blocks are concatenated circularly and the
   result is truncated to `n`.
3. **The same index is applied to both series**, preserving the pairing.
4. Both metrics are recomputed from scratch on each resampled pair of paths.
   Maximum drawdown is therefore evaluated on the resampled cumulative wealth
   path, not resampled from observed drawdown values.
5. The test statistic is `Δ = M(SBCA) − M(comparison)`; maximum drawdown is
   sign-flipped so that a larger value always means better performance.
6. The null distribution is formed by centring, `Δ⁰_b = Δ*_b − mean(Δ*)`, and the
   one-sided p-value is the proportion of replicates with `Δ⁰_b ≥ Δ`.

Configuration: 30,000 resamples, fixed block length 60 trading days.

## 5. Headline numbers

| Quantity | Value |
| --- | --- |
| SBCA Sharpe ratio, corrected features (2 / 4 / 6 assets) | 0.8184 / 0.8108 / 0.8945 |
| External tests surviving BH-FDR at 5 % | 17 of 60 (largest surviving p = 0.0129) |
| Internal tests surviving BH-FDR at 5 % | 0 of 36 (smallest p = 0.0017) |
| Factorial intervals excluding zero | 2 of 54, both the Actor-Critic main effect in the 2-asset group |
| Corrected sentiment AUC, out of sample | 0.46–0.55 (intervals contain 0.50) |
| Pre-correction sentiment AUC, test period | 0.58–0.76 for the four affected assets |

## 6. Environment

Python 3.10.16, PyTorch 2.6.0 (CUDA 11.8), Transformers 4.51.1, NumPy 1.26.4,
pandas 2.2.3, SciPy 1.15.3, single NVIDIA GeForce RTX 3060 (12 GB), Windows.
See `requirements.txt`.

> Note: the `requirements.txt` in the parent directory pins `torch>=2.0.0` with
> CUDA 12.1 for the `redo/` project. This folder's `requirements.txt` reflects the
> environment that actually produced the numbers reported in the manuscript.
