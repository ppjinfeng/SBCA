"""
Recompute the analyses that are derivable from the four time-split variant series
plus the (feature-independent) deterministic baselines.

Produces, in runs/:
    blocklength_sensitivity.csv   external + internal p-values at blocks 20/40/60/120
    factorial_ci.csv              2x2 main effects / interaction with 95% bootstrap CIs
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SBCA_FEATURES", "data/bert_pred_for_SARL_timesplit.csv")
import run_all_old as RT
from blocklength_sensitivity import metrics_2d, bootstrap_pvalues

IN = "results"
OUT = "runs"
GROUPS = [("2assets", 2), ("4assets", 4), ("6assets", 6)]
BLOCKS = [20, 40, 60, 120]
METRICS = ["AR", "SR", "Sortino", "MDD"]
INT_VAR = ["SB", "SBA", "SBC"]
EXT_BASE = [("bh", "Buy & Hold", "simple"), ("ew", "Equal Weight", "simple"),
            ("dj", "Dow Jones", "simple"), ("mvo", "MVO", "classical")]
N_BOOT = 30000
CHUNK = 5000
RNG_SEED = 20260919
FULL = ["PV", "AR", "SR", "Sortino", "MDD", "Calmar"]


def metrics_full(R_):
    m = metrics_2d(R_)
    m["PV"] = np.exp(R_.sum(axis=1))
    m["Calmar"] = m["AR"] / (np.abs(m["MDD"]) + 1e-8)
    return m


def contrasts(m):
    return {"dCM": 0.5 * ((m["SBC"] - m["SB"]) + (m["SBCA"] - m["SBA"])),
            "dAC": 0.5 * ((m["SBA"] - m["SB"]) + (m["SBCA"] - m["SBC"])),
            "dInt": m["SBCA"] - m["SBA"] - m["SBC"] + m["SB"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    bl_rows, fac_rows = [], []

    for gname, k in GROUPS:
        df = pd.read_csv(f"{IN}/ablation_rets_{gname}.csv")
        sbca = df["SBCA"].values
        close = RT.df_close[RT.all_stocks[:k]].values
        env = RT.StockEnv(close, RT.df_delta[RT.all_stocks[:k]].values)
        TE = range(RT.val_end, env.days - 1)

        # -------- block-length sweep (internal + external) --------
        for var in INT_VAR:
            other = df[var].values
            n = min(len(sbca), len(other))
            for blk in BLOCKS:
                res = bootstrap_pvalues(sbca[:n], other[:n], blk)
                for me in METRICS:
                    bl_rows.append(dict(Group=gname, Family="internal",
                                        Comparison=f"SBCA vs {var}", Metric=me,
                                        block=blk, p_value=round(res[me]["p_value"], 4)))
        for key, name, typ in EXT_BASE:
            bl = (RT.baseline_daily_log_rets(close, TE, key) if typ == "simple"
                  else RT.classical_daily_log_rets(close, TE, gamma=2.0))
            n = min(len(sbca), len(bl))
            for blk in BLOCKS:
                res = bootstrap_pvalues(sbca[:n], bl[:n], blk)
                for me in METRICS:
                    bl_rows.append(dict(Group=gname, Family="external",
                                        Comparison=f"SBCA vs {name}", Metric=me,
                                        block=blk, p_value=round(res[me]["p_value"], 4)))
        print(f"[{gname}] block-length sweep done", flush=True)

        # -------- factorial contrasts with CIs --------
        series = {v: df[v].values[:min(len(df[v]), len(sbca))] for v in ["SB", "SBA", "SBC", "SBCA"]}
        n = min(len(s) for s in series.values())
        series = {v: s[:n] for v, s in series.items()}
        obs = {me: {v: metrics_full(series[v][None, :])[me][0] for v in series} for me in FULL}
        boot = {me: {k: [] for k in ["dCM", "dAC", "dInt"]} for me in FULL}
        done = 0
        while done < N_BOOT:
            m = min(CHUNK, N_BOOT - done)
            nblk = int(np.ceil(n / 60))
            idx = (rng.integers(0, n, size=(m, nblk))[:, :, None] + np.arange(60)[None, None, :]) % n
            idx = idx.reshape(m, -1)[:, :n]
            bm = {v: metrics_full(series[v][idx]) for v in series}
            for me in FULL:
                c = contrasts({v: bm[v][me] for v in series})
                for kk in c:
                    boot[me][kk].append(c[kk])
            done += m
        for me in FULL:
            cp = contrasts(obs[me])
            for kk in ["dCM", "dAC", "dInt"]:
                arr = np.concatenate(boot[me][kk])
                lo, hi = np.percentile(arr, [2.5, 97.5])
                fac_rows.append(dict(Group=gname, Metric=me, Contrast=kk,
                                     point=round(cp[kk], 4), ci_lo=round(lo, 4),
                                     ci_hi=round(hi, 4),
                                     excludes_zero="yes" if (lo > 0 or hi < 0) else "no"))
        print(f"[{gname}] factorial CIs done", flush=True)

    pd.DataFrame(bl_rows).to_csv(f"{OUT}/blocklength_sensitivity.csv", index=False)
    pd.DataFrame(fac_rows).to_csv(f"{OUT}/factorial_ci.csv", index=False)
    n_ex = sum(1 for r in fac_rows if r["excludes_zero"] == "yes")
    print(f"\nfactor CIs excluding zero: {n_ex}/{len(fac_rows)}")
    print("saved runs/blocklength_sensitivity.csv and runs/factorial_ci.csv")


if __name__ == "__main__":
    main()
