# Input data

## What is shipped here

| File | Rows | Purpose |
| --- | --- | --- |
| `bert_pred_for_SARL_timesplit.csv` | 12,300 | The **corrected** sentiment feature. Daily OHLCV with a `delta_bert` column re-derived using a chronological fine-tuning split, plus `Titles_combined` (the raw headlines) and `label` (next-day direction). This is the table the revised manuscript's experiments consume. |

Everything needed to reproduce the **statistical analysis** (the block-bootstrap
tests, the block-length sweep and the factorial decomposition) is therefore
present: those steps operate on `results/ablation_rets_*.csv`, which is also
committed.

## What must be downloaded separately

Two files are needed only for the steps that *re-derive* the sentiment feature or
compare it against the pre-correction version. They are not shipped here because
they belong to the associated data release:

| File | Needed by | Where to get it |
| --- | --- | --- |
| `stock_news_trading_data.csv` | `code/bert_timesplit.py` (re-fine-tunes BERT) | the project dataset, <https://huggingface.co/datasets/Changahou/SBCA> |
| `bert_pred_for_SARL.csv` | `code/bert_audit.py`, `code/ablation_bootstrap.py` (the pre-correction audit) | same dataset |

Put them in this directory. If they are absent, `code/smoke_test.py` reports them
as warnings rather than failures, and the analysis steps above still run.

## Columns

Both feature tables share the same schema:

| Column | Meaning |
| --- | --- |
| `Date` | trading day |
| `volume`, `open`, `high`, `low`, `close`, `adj close` | daily price data |
| `Stock_symbol` | ticker (CAT, GILD, GS, KO, MRK, NVDA) |
| `Titles_combined` | financial news headlines for that stock and day, joined with `\|\|\|` |
| `label` | 1 if the next day's close is higher, else 0 — this is the target the sentiment encoder is fine-tuned against |
| `delta_bert` | sentiment score in `[0, 1]` produced by the encoder; higher means a stronger predicted probability of a price increase |

## Why there are two versions of the feature

The originally released pipeline fine-tuned BERT on the first 80 % of rows after
sorting by `[Stock_symbol, Date]`. Because the table is grouped by stock, that
split is **by asset rather than by time**: four of the six assets (CAT, GILD, GS,
KO) sat entirely inside the fine-tuning set, so those assets received in-sample
sentiment scores — including throughout the reinforcement-learning test period of
2021–2022.

`code/bert_timesplit.py` re-derives the feature using a strictly chronological
split (fine-tuning only on headlines dated on or before 31 December 2018) and
`code/bert_audit.py` quantifies the difference per stock and per period. The
result is in `results/bert/` and, in the manuscript, Supplementary Table S5.
