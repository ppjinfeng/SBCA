"""
Bootstrap confidence intervals for the 2x2 factorial contrasts (main effects and
interaction) of the ablation design.

Contrasts:
  dCM   = 1/2[(SBC - SB) + (SBCA - SBA)]
  dAC   = 1/2[(SBA - SB) + (SBCA - SBC)]
  dInt  = SBCA - SBA - SBC + SB

Point estimates are reported both from the published Table 4 values and from the
replication series, so the two can be compared.  Percentile 95% intervals come
from the same paired circular block bootstrap used elsewhere (30,000 resamples,
block length 60), with the SAME block index applied to all four variants.

Output: runs/factorial_ci.csv  and  ./out/factorial_ci_summary.txt
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_old as R
from blocklength_sensitivity import metrics_2d

METRICS = ["PV", "AR", "SR", "Sortino", "MDD", "Calmar"]
N_BOOT = 30000
BLOCK = 60
CHUNK = 5000
RNG_SEED = 20260919
GROUPS = ["2assets", "4assets", "6assets"]
VARIANTS = ["SB", "SBA", "SBC", "SBCA"]


def metrics_full(R_):
    """metrics_2d (AR/SR/Sortino/MDD) plus PV and Calmar."""
    m = metrics_2d(R_)
    m["PV"] = np.exp(R_.sum(axis=1))
    m["Calmar"] = m["AR"] / (np.abs(m["MDD"]) + 1e-8)
    return m

PUB = {
    ("2assets", "SB"):   (1.4270, 0.1946, 0.7966, 1.3891, -0.2140, 0.9094),
    ("2assets", "SBA"):  (1.4389, 0.1996, 0.8175, 1.4222, -0.2085, 0.9572),
    ("2assets", "SBC"):  (1.4388, 0.1995, 0.8163, 1.4215, -0.2084, 0.9574),
    ("2assets", "SBCA"): (1.4389, 0.1996, 0.8179, 1.4241, -0.2084, 0.9578),
    ("4assets", "SB"):   (1.3535, 0.1634, 0.8055, 1.4125, -0.1774, 0.9211),
    ("4assets", "SBA"):  (1.3563, 0.1646, 0.8108, 1.4211, -0.1775, 0.9275),
    ("4assets", "SBC"):  (1.3549, 0.1640, 0.8111, 1.4213, -0.1764, 0.9298),
    ("4assets", "SBCA"): (1.3563, 0.1646, 0.8107, 1.4208, -0.1775, 0.9273),
    ("6assets", "SB"):   (1.5147, 0.2307, 0.8585, 1.7248, -0.2202, 1.0480),
    ("6assets", "SBA"):  (1.5153, 0.2310, 0.8938, 1.7445, -0.2111, 1.0942),
    ("6assets", "SBC"):  (1.5143, 0.2306, 0.8948, 1.7460, -0.2102, 1.0972),
    ("6assets", "SBCA"): (1.5154, 0.2310, 0.8943, 1.7456, -0.2110, 1.0949),
}


def contrasts_from_metrics(m):
    """m: dict metric -> array over bootstrap rows (or scalars)."""
    return {
        "dCM":  0.5 * ((m["SBC"] - m["SB"]) + (m["SBCA"] - m["SBA"])),
        "dAC":  0.5 * ((m["SBA"] - m["SB"]) + (m["SBCA"] - m["SBC"])),
        "dInt": m["SBCA"] - m["SBA"] - m["SBC"] + m["SB"],
    }


def main():
    os.makedirs("runs", exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    lines = []
    for g in GROUPS:
        df = pd.read_csv(f"results/ablation_rets_{g}.csv")
        series = {v: df[v].values for v in VARIANTS}
        n = min(len(s) for s in series.values())
        series = {v: s[:n] for v, s in series.items()}

        # observed (from the replication series)
        obs_rep = {me: {v: metrics_full(series[v][None, :])[me][0] for v in VARIANTS}
                   for me in METRICS}
        # observed (from published Table 4)
        obs_pub = {me: {v: PUB[(g, v)][i] for v in VARIANTS}
                   for i, me in enumerate(METRICS)}

        # bootstrap
        boot = {me: {k: [] for k in ["dCM", "dAC", "dInt"]} for me in METRICS}
        done = 0
        while done < N_BOOT:
            m = min(CHUNK, N_BOOT - done)
            nblk = int(np.ceil(n / BLOCK))
            starts = rng.integers(0, n, size=(m, nblk))
            idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n
            idx = idx.reshape(m, -1)[:, :n]
            bm = {v: metrics_full(series[v][idx]) for v in VARIANTS}
            for me in METRICS:
                c = contrasts_from_metrics({v: bm[v][me] for v in VARIANTS})
                for k in c:
                    boot[me][k].append(c[k])
            done += m

        lines.append("=" * 96)
        lines.append(f"GROUP {g}   (n={n}, {N_BOOT} resamples, block={BLOCK})")
        lines.append("=" * 96)
        lines.append(f"{'Metric':8s} {'contrast':8s} {'from T4':>10s} {'from rep':>10s} "
                     f"{'2.5%':>10s} {'97.5%':>10s} {'excl 0':>7s}")
        for me in METRICS:
            cp = contrasts_from_metrics(obs_pub[me])
            cr = contrasts_from_metrics(obs_rep[me])
            for k in ["dCM", "dAC", "dInt"]:
                arr = np.concatenate(boot[me][k])
                lo, hi = np.percentile(arr, [2.5, 97.5])
                excl = "yes" if (lo > 0 or hi < 0) else "no"
                lines.append(f"{me:8s} {k:8s} {cp[k]:+10.4f} {cr[k]:+10.4f} "
                             f"{lo:+10.4f} {hi:+10.4f} {excl:>7s}")
                rows.append({"Group": g, "Metric": me, "Contrast": k,
                             "point_Table4": round(cp[k], 4),
                             "point_replication": round(cr[k], 4),
                             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                             "excludes_zero": excl})
        lines.append("")
        print("\n".join(lines[-22:]), flush=True)

    pd.DataFrame(rows).to_csv("runs/factorial_ci.csv", index=False)
    txt = "\n".join(lines)
    with open("runs/factorial_ci_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    n_excl = sum(1 for r in rows if r["excludes_zero"] == "yes")
    print(f"\nContrasts whose 95% CI excludes zero: {n_excl} of {len(rows)}")
    print("Saved ./out/factorial_ci.csv")


if __name__ == "__main__":
    main()
