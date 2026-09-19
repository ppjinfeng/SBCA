"""
Block-length robustness for the external-baseline significance tests.

Recomputes the one-sided circular block-bootstrap p-values for SBCA versus the
four deterministic baselines (EW / BH / DJ / MVO) in all three asset groups,
under (a) the fixed block length of 60 trading days actually used by the
released code, and (b) the data-dependent rule max(5, n^(1/3)) ~ 8 days stated
in the previous manuscript text. PPO is excluded (it requires extra training).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_old as R

BASE_DEFS = [("bh", "Buy & Hold", "simple"),
             ("ew", "Equal Weight", "simple"),
             ("dj", "Dow Jones", "simple"),
             ("mvo", "MVO", "classical")]

GROUPS = [("2assets", 2), ("4assets", 4), ("6assets", 6)]


def main():
    rows = []
    for suffix, k in GROUPS:
        sbca = pd.read_csv(f"./out/ablation_rets_{suffix}.csv")["SBCA"].values
        stocks = R.all_stocks[:k]
        close_arr = R.df_close[stocks].values
        env = R.StockEnv(close_arr, R.df_delta[stocks].values)
        TE = range(R.val_end, env.days - 1)
        n = len(sbca)
        blocks = sorted({int(round(max(5, n ** (1 / 3)))), R.BLOCK_SIZE})
        print(f"\n=== {suffix}: n={n}, candidate block lengths {blocks} ===", flush=True)

        for block in blocks:
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
                    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "-"
                    rows.append({"Group": suffix, "block": block, "baseline": bname,
                                 "metric": metric, "p": round(p, 4), "sig": sig})
                    print(f"  block={block:3d} {bname:13s} {metric:8s} p={p:.4f} {sig}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("./out/blocklength_robustness.csv", index=False)
    pivot = df.pivot_table(index=["Group", "baseline", "metric"], columns="block",
                           values="p", aggfunc="first")
    print("\n=== p-value comparison: block=8 (n^1/3) vs block=60 ===\n")
    print(pivot.to_string())
    print("\nSaved ./out/blocklength_robustness.csv")


if __name__ == "__main__":
    main()
