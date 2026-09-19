"""
Block-length sensitivity for both the external-baseline and the internal-ablation
block-bootstrap tests.

Block lengths: 20, 40, 60, 120 trading days (60 is the value used for the
published tests).  30,000 resamples, one-sided, H0-centred, paired, exactly as
run_all_old.block_bootstrap_one_sided_test.

The bootstrap is vectorised for speed; a sanity check against the original
(non-vectorised) implementation is printed before the sweep.

Input : results/ablation_rets_{2,4,6}assets.csv  (daily net log returns per variant)
Output: runs/blocklength_sensitivity.csv
        runs/blocklength_sensitivity_summary.txt
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_old as R

BLOCKS = [20, 40, 60, 120]
N_BOOT = 30000
METRICS = ["AR", "SR", "Sortino", "MDD"]
CHUNK = 5000
RNG_SEED = 20260919

EXT_BASE = [("bh", "Buy & Hold", "simple"),
            ("ew", "Equal Weight", "simple"),
            ("dj", "Dow Jones", "simple"),
            ("mvo", "MVO", "classical")]
INT_VAR = ["SB", "SBA", "SBC"]


# --------------------------------------------------------------------------
# vectorised metrics on a 2-D array (rows = bootstrap replicates)
# --------------------------------------------------------------------------
def metrics_2d(R_):
    daily_rf = R.RISK_FREE / 252.0
    out = {}
    out["AR"] = np.exp(R_.mean(axis=1) * 252.0) - 1.0
    exc = R_ - daily_rf
    out["SR"] = exc.mean(axis=1) / (exc.std(axis=1) + 1e-8) * np.sqrt(252.0)
    mask = R_ < 0
    cnt = mask.sum(axis=1)
    s1 = np.where(mask, R_, 0.0).sum(axis=1)
    s2 = np.where(mask, R_ * R_, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_d = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
        var_d = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mean_d ** 2, 0.0)
        std_d = np.sqrt(np.maximum(var_d, 0.0))
        sor = R_.mean(axis=1) / (std_d + 1e-8) * np.sqrt(252.0)
    out["Sortino"] = np.where(cnt == 0, 10.0, sor)
    cum = np.exp(np.cumsum(R_, axis=1))
    run_max = np.maximum.accumulate(cum, axis=1)
    out["MDD"] = ((cum - run_max) / run_max).min(axis=1)
    return out


def bootstrap_pvalues(a, b, block, n_boot=N_BOOT, rng=None, version="vector"):
    """Paired one-sided circular block bootstrap with H0 centring."""
    n = len(a)
    obs = {k: metrics_2d(a[None, :])[k][0] - metrics_2d(b[None, :])[k][0] for k in METRICS}
    raw = {k: [] for k in METRICS}
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    done = 0
    while done < n_boot:
        m = min(CHUNK, n_boot - done)
        nblk = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(m, nblk))
        base = np.arange(block)[None, None, :]
        idx = (starts[:, :, None] + base) % n
        idx = idx.reshape(m, -1)[:, :n]
        ma = metrics_2d(a[idx])
        mb = metrics_2d(b[idx])
        for k in METRICS:
            raw[k].append(ma[k] - mb[k])
        done += m
    res = {}
    for k in METRICS:
        arr = np.concatenate(raw[k])
        centred = arr - arr.mean()
        res[k] = {"diff_true": float(obs[k]),
                  "p_value": float(np.mean(centred >= obs[k]))}
    return res


def main():
    # ---------------- sanity check against the original implementation ----
    a = pd.read_csv("results/ablation_rets_2assets.csv")["SBCA"].values
    b = pd.read_csv("results/ablation_rets_2assets.csv")["SB"].values
    print("sanity check: 2assets SBCA vs SB, block = 60", flush=True)
    np.random.seed(RNG_SEED)
    R.BLOCK_SIZE = 60
    orig = R.block_bootstrap_one_sided_test(a, b)
    vec = bootstrap_pvalues(a, b, 60)
    for k in METRICS:
        print(f"   {k:8s} original p={orig[k]['p_value']:.4f}   vectorised p={vec[k]['p_value']:.4f}",
              flush=True)

    # ---------------- sweep -----------------------------------------------
    rows = []
    groups = [("2assets", 2), ("4assets", 4), ("6assets", 6)]
    for gname, k in groups:
        df = pd.read_csv(f"results/ablation_rets_{gname}.csv")
        sbca = df["SBCA"].values
        close_arr = R.df_close[R.all_stocks[:k]].values
        delta_arr = R.df_delta[R.all_stocks[:k]].values
        env = R.StockEnv(close_arr, delta_arr)
        TE = range(R.val_end, env.days - 1)

        for label, series in [("internal", None)]:
            pass
        # internal comparisons
        for var in INT_VAR:
            other = df[var].values
            n = min(len(sbca), len(other))
            for blk in BLOCKS:
                res = bootstrap_pvalues(sbca[:n], other[:n], blk)
                for kk in METRICS:
                    rows.append({"Group": gname, "Family": "internal",
                                 "Comparison": f"SBCA vs {var}", "Metric": kk,
                                 "block": blk,
                                 "p_value": round(res[kk]["p_value"], 4)})
            print(f"  [{gname}] internal SBCA vs {var}: done", flush=True)
        # external comparisons
        for bkey, bname, btype in EXT_BASE:
            if btype == "simple":
                bl = R.baseline_daily_log_rets(close_arr, TE, bkey)
            else:
                bl = R.classical_daily_log_rets(close_arr, TE, gamma=2.0)
            n = min(len(sbca), len(bl))
            for blk in BLOCKS:
                res = bootstrap_pvalues(sbca[:n], bl[:n], blk)
                for kk in METRICS:
                    rows.append({"Group": gname, "Family": "external",
                                 "Comparison": f"SBCA vs {bname}", "Metric": kk,
                                 "block": blk,
                                 "p_value": round(res[kk]["p_value"], 4)})
            print(f"  [{gname}] external SBCA vs {bname}: done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("runs", exist_ok=True)
    df.to_csv("runs/blocklength_sensitivity.csv", index=False)

    lines = ["Block-length sensitivity (30,000 resamples, one-sided, H0-centred)",
             "block lengths: %s" % BLOCKS, ""]
    for fam in ["external", "internal"]:
        sub = df[df.Family == fam]
        lines.append("=" * 78)
        lines.append(f"FAMILY: {fam}")
        lines.append("=" * 78)
        piv = sub.pivot_table(index=["Group", "Comparison", "Metric"],
                              columns="block", values="p_value", aggfunc="first")
        lines.append(piv.to_string())
        lines.append("")
    txt = "\n".join(lines)
    with open("runs/blocklength_sensitivity_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt, flush=True)
    print("\nSaved runs/blocklength_sensitivity.csv")


if __name__ == "__main__":
    main()
