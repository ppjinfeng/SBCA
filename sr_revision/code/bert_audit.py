"""Supplementary Table S5: full audit of the sentiment-feature correction.

Per stock and per period: sample size, positive-class rate, and the AUC of the
sentiment score under (a) the original stock-wise split and (b) the corrected
chronological split, with percentile bootstrap 95% confidence intervals.

Output: ./out_ts/bert_audit_table.csv  (+ printed LaTeX rows)
"""
import io
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "runs"
OUT_TS = "runs"
N_BOOT = 2000
SEED = 20260920
PERIODS = [("2012--2018 (RL training)", "2012-01-01", "2018-12-31"),
           ("2019--2020 (RL validation)", "2019-01-01", "2020-12-31"),
           ("2021--2022 (RL test)", "2021-01-01", "2022-12-31")]

# role of each stock under the ORIGINAL stock-wise 80/20 split (Bert.py)
ROLE = {"CAT": "in BERT fine-tuning set", "GILD": "in BERT fine-tuning set",
        "GS": "in BERT fine-tuning set", "KO": "in BERT fine-tuning set",
        "MRK": "partly in BERT fine-tuning set", "NVDA": "held out of BERT fine-tuning set"}


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(s).rank().values
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_ci(y, s, rng, n_boot=N_BOOT):
    y = np.asarray(y); s = np.asarray(s)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(auc(yy, ss))
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main():
    old = pd.read_csv(os.path.join(OUT, "bert_pred_timesplit.csv"))
    old["Date"] = pd.to_datetime(old["Date"])
    orig = pd.read_csv(os.environ.get("SBCA_ORIGINAL_FEATURES", "data/bert_pred_for_SARL.csv"))[["Date", "Stock_symbol", "label", "delta_bert"]]
    orig["Date"] = pd.to_datetime(orig["Date"])
    old = old.merge(orig, on=["Date", "Stock_symbol"], how="left", suffixes=("", "_o"))
    miss = old["delta_bert"].isna().sum()
    print("merged rows: %d, missing original delta_bert: %d" % (len(old), miss), flush=True)
    rng = np.random.default_rng(SEED)

    rows = []
    for st, g in old.groupby("Stock_symbol"):
        g = g.sort_values("Date")
        for plabel, a, b in PERIODS:
            sub = g[(g["Date"] >= a) & (g["Date"] <= b)]
            if len(sub) < 30:
                continue
            y = sub["label"].values
            a_old = auc(y, sub["delta_bert"].values)
            a_new = auc(y, sub["delta_bert_new"].values)
            lo, hi = auc_ci(y, sub["delta_bert_new"].values, rng)
            rows.append(dict(Stock=st, Role=ROLE[st], Period=plabel, n=len(sub),
                             pos_rate=round(float(np.mean(y)), 3),
                             AUC_original=round(a_old, 3) if not np.isnan(a_old) else None,
                             AUC_corrected=round(a_new, 3) if not np.isnan(a_new) else None,
                             CI_lo=round(lo, 3), CI_hi=round(hi, 3)))

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT_TS, "bert_audit_table.csv"), index=False)
    pd.set_option("display.width", 200)
    print(d.to_string(index=False))

    # LaTeX rows
    lines = []
    for st in ["CAT", "GILD", "GS", "KO", "MRK", "NVDA"]:
        s = d[d.Stock == st].reset_index(drop=True)
        lines.append("\\multirow{%d}{*}{%s}" % (len(s), st))
        for i, r in s.iterrows():
            role = r["Role"] if i == 0 else ""
            lines.append("& %s & %s & %d & %.3f & %.3f & %.3f & $[%+.3f, %+.3f]$ \\\\"
                         % (role, r["Period"], r["n"], r["pos_rate"],
                            r["AUC_original"], r["AUC_corrected"], r["CI_lo"], r["CI_hi"]))
            if i < len(s) - 1:
                lines.append("\\cline{2-8}")
        lines.append("\\hline")
    with io.open(os.path.join(OUT_TS, "bert_audit_rows.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nLaTeX rows -> out_ts/bert_audit_rows.tex")


if __name__ == "__main__":
    main()
