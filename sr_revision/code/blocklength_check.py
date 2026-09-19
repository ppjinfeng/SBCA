"""
Verify which circular block-bootstrap block length produced the published
external-baseline p-values in Table 2 (2-asset column), and report both.

Uses the SBCA seed-42 daily log returns saved by ablation_bootstrap.py, so no
re-training of SBCA is required. Deterministic baselines (EW / BH / DJ / MVO)
need no training either; PPO is excluded because it requires extra training and
is reported separately.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_old as R

PUBLISHED_2ASSETS = {
    "Buy & Hold":   {"AR": 0.0011, "SR": 0.0001, "Sortino": 0.0006, "MDD": 0.0129},
    "Equal Weight": {"AR": 0.0168, "SR": 0.0006, "Sortino": 0.0054, "MDD": 0.0383},
    "Dow Jones":    {"AR": 0.2977, "SR": 0.0897, "Sortino": 0.0968, "MDD": 0.0405},
    "MVO (g=2)":    {"AR": 0.3393, "SR": 0.0965, "Sortino": 0.0949, "MDD": 0.0379},
}

BASE_DEFS = [("bh", "Buy & Hold", "simple"),
             ("ew", "Equal Weight", "simple"),
             ("dj", "Dow Jones", "simple"),
             ("mvo", "MVO (g=2)", "classical")]


def main():
    sbca = pd.read_csv("./out/ablation_rets_2assets.csv")["SBCA"].values
    stocks = R.all_stocks[:2]
    close_arr = R.df_close[stocks].values
    TE = range(R.val_end, len(close_arr) - 1)
    n = len(sbca)
    print(f"n = {n} test days;  n^(1/3) = {n ** (1/3):.2f};  paper block = "
          f"{max(5, n ** (1/3)):.2f};  code BLOCK_SIZE = {R.BLOCK_SIZE}\n")

    rows = []
    for block in (int(round(max(5, n ** (1 / 3)))), R.BLOCK_SIZE):
        R.BLOCK_SIZE = block
        for bkey, bname, btype in BASE_DEFS:
            if btype == "simple":
                bl = R.baseline_daily_log_rets(close_arr, TE, bkey)
            else:
                bl = R.classical_daily_log_rets(close_arr, TE, gamma=2.0)
            m = min(len(sbca), len(bl))
            res = R.block_bootstrap_one_sided_test(sbca[:m], bl[:m])
            for metric in ["AR", "SR", "Sortino", "MDD"]:
                p = res[metric]["p_value"]
                pub = PUBLISHED_2ASSETS[bname][metric]
                rows.append({"block": block, "baseline": bname, "metric": metric,
                             "p_reproduced": round(p, 4), "p_published": pub,
                             "match": "YES" if abs(p - pub) < 0.0006 else ""})
                print(f"  block={block:3d} {bname:13s} {metric:8s} "
                      f"p={p:.4f}  published={pub:.4f}  "
                      f"{'<-- match' if abs(p - pub) < 0.0006 else ''}", flush=True)

    pd.DataFrame(rows).to_csv("./out/blocklength_check.csv", index=False)
    print("\nSaved ./out/blocklength_check.csv")


if __name__ == "__main__":
    main()
