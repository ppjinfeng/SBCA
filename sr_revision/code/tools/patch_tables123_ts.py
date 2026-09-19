"""Rebuild Tables 1, 2, 3 of the manuscript from the time-split results, and report
the Benjamini-Hochberg outcome for the external test family."""
import io
import numpy as np
import pandas as pd

MAIN = r"H:\老婆\v3\SBCA-SR.tex"
OUT_TS = r"H:\老婆\v3\analysis\out_ts"
TSRUN = r"H:\老婆\v3\analysis\ts_run\result"

# price-only baselines: unaffected by the sentiment feature
BASE = {
    "2assets": {"Equal Weight": (1.4208, 0.1920, 0.7870, 1.3709, -0.2136, 0.8987),
                "Buy \\& Hold": (1.3953, 0.1812, 0.7423, 1.2937, -0.2247, 0.8065),
                "Dow Jones":    (1.3614, 0.1668, 0.5594, 0.8881, -0.2832, 0.5888),
                "MVO ($\\gamma=2$)": (1.3204, 0.1491, 0.4038, 0.6437, -0.3285, 0.4539)},
    "4assets": {"Equal Weight": (1.3428, 0.1588, 0.7804, 1.3717, -0.1797, 0.8836),
                "Buy \\& Hold": (1.3124, 0.1456, 0.7010, 1.2549, -0.1931, 0.7539),
                "Dow Jones":    (1.3126, 0.1457, 0.5584, 1.0278, -0.2490, 0.5851),
                "MVO ($\\gamma=2$)": (1.1878, 0.0899, 0.2772, 0.4966, -0.3027, 0.2969)},
    "6assets": {"Equal Weight": (1.5423, 0.2419, 0.9366, 1.8265, -0.2097, 1.1537),
                "Buy \\& Hold": (1.3712, 0.1710, 0.5993, 1.1275, -0.3062, 0.5585),
                "Dow Jones":    (1.3556, 0.1643, 0.5708, 1.0520, -0.3137, 0.5238),
                "MVO ($\\gamma=2$)": (1.7705, 0.3306, 0.4997, 1.2558, -0.4761, 0.6945)},
}
GROUPS = ["2assets", "4assets", "6assets"]
METRICS = ["PV", "AR", "SR", "Sortino", "MDD", "Calmar"]


def star(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def fmt_p(p):
    return "$<$0.0001" if p <= 0 else "%.4f" % p


# ---------------------------------------------------------------- Table 1
def table1():
    ppo = pd.read_csv(f"{TSRUN}/ppo_table1.csv").set_index("Group")
    lines = []
    for g in GROUPS:
        ab = pd.read_csv(f"{OUT_TS}/ablation_metrics_{g}.csv").set_index("Model")
        rows = []
        for name, v in BASE[g].items():
            rows.append((name, v))
        p = ppo.loc[g]
        rows.append(("PPO", tuple(float(p[k]) for k in METRICS)))
        rows.append(("SBCA", tuple(float(ab.loc["SBCA", k]) for k in METRICS)))
        lines.append("\\multirow{6}{*}{%s}" % g)
        for i, (name, v) in enumerate(rows):
            cells = " & ".join("%.4f" % x for x in v)
            lines.append("& %s & %s \\\\" % (name, cells))
            if i < len(rows) - 1:
                lines.append("\\cline{2-8}")
        lines.append("\\hline")
    return "\n".join(lines)


# ---------------------------------------------------------------- Table 3
def table3():
    b = pd.read_excel(f"{TSRUN}/bootstrap_all.xlsx", "BootstrapTest")
    order = ["Equal Weight", "Buy & Hold", "Dow Jones", "MVO (γ=2)", "PPO"]
    disp = {"Equal Weight": "Equal Weight", "Buy & Hold": "Buy \\& Hold",
            "Dow Jones": "Dow Jones", "MVO (γ=2)": "MVO ($\\gamma$=2)", "PPO": "PPO"}
    lines = []
    for base in order:
        s = b[b.Comparison == "SBCA vs " + base]
        lines.append("\\multirow{4}{*}{%s}" % disp[base])
        for i, me in enumerate(["AR", "SR", "Sortino", "MDD"]):
            cells = []
            for g in GROUPS:
                p = float(s[(s.Group == g) & (s.Metric == me)].p_value.iloc[0])
                cells.append(fmt_p(p) + star(p))
            lines.append("& %s & %s \\\\" % (me, " & ".join(cells)))
            if i < 3:
                lines.append("\\cline{2-5}")
        lines.append("\\hline")
    return "\n".join(lines)


# ---------------------------------------------------------------- Table 2
def table2():
    c = pd.read_excel(f"{TSRUN}/cost_sensitivity_fixed.xlsx", "Sheet1")
    lines = []
    for comm in [0.0010, 0.0025, 0.0050, 0.0100]:
        cells = []
        for g in GROUPS:
            s = c[c.Group == g].set_index("Commission")
            base = s.loc[0.0025]
            for col in ["SBCA_SR", "EW_SR", "BH_SR"]:
                rel = (s.loc[comm, col] - base[col]) / base[col] * 100.0
                cells.append("%.4f" % rel)
        lines.append("%.4f\n& %s \\\\" % (comm, " & ".join(cells)))
        lines.append("\\hline")
    return "\n".join(lines)


def replace_table(text, label, new_block):
    i = text.index("\\label{%s}" % label)
    end = text.index("\\end{table}", i) + len("\\end{table}")
    start = text.rindex("\\begin{table}", 0, i)
    return text[:start] + new_block + text[end:]


def main():
    t = io.open(MAIN, encoding="utf-8").read()

    t1 = (r"""\begin{table}[ht]
\centering
\begin{tabular}{|l|l|c|c|c|c|c|c|}
\hline
Group & Model & PV & AR & SR & Sortino & MDD & Calmar \\
\hline
""" + table1() + r"""
\end{tabular}
\caption{Overall performance comparison across 2-, 4-, and 6-asset groups.}
\label{tab:overall_perf}
\end{table}""")

    t3 = (r"""\begin{table}[ht]
\centering
\footnotesize
\begin{tabular}{|l|l|c|c|c|}
\hline
\multirow{2}{*}{Baseline} & \multirow{2}{*}{Metric} & \multicolumn{3}{c|}{$p$-value} \\
\cline{3-5}
& & 2assets & 4assets & 6assets \\
\hline
""" + table3() + r"""
\end{tabular}
\caption{Block-bootstrap significance test results (30,000 resamples, one-sided, fixed block length of 60 trading days). $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. The null hypothesis is that SBCA does not outperform the baseline on the specified metric.}
\label{tab:bootstrap}
\end{table}""")

    t2 = (r"""\begin{table}[ht]
\centering
\footnotesize
\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}
\hline
\multirow{2}{*}{Commission}
& \multicolumn{3}{c|}{2assets}
& \multicolumn{3}{c|}{4assets}
& \multicolumn{3}{c|}{6assets} \\
\cline{2-10}
& SBCA & Equal Weight & Buy \& Hold & SBCA & Equal Weight & Buy \& Hold & SBCA & Equal Weight & Buy \& Hold \\
\hline
""" + table2() + r"""
\end{tabular}
\caption{Cost sensitivity analysis. We compare SBCA with Equal Weight and Buy \& Hold strategies. Entries denote relative percentage changes (in \%) of Sharpe ratios relative to the baseline commission rate of 0.25\% (0.0025). A single SBCA model trained at the baseline commission rate of 0.25\% is evaluated at each rate without retraining. Equal Weight rebalances to \(1/N\) monthly, incurring costs on each rebalancing date. Buy \& Hold pays a one-time initial cost to establish the position and incurs no further costs. Negative values indicate a decline in risk-adjusted performance compared with the baseline setting.}
\label{tab:cost_sensitivity}
\end{table}""")

    t = replace_table(t, "tab:overall_perf", t1)
    t = replace_table(t, "tab:bootstrap", t3)
    t = replace_table(t, "tab:cost_sensitivity", t2)
    io.open(MAIN, "w", encoding="utf-8", newline="\n").write(t)
    print("Tables 1, 2, 3 replaced")

    # BH-FDR for the external family
    b = pd.read_excel(f"{TSRUN}/bootstrap_all.xlsx", "BootstrapTest")
    ps = sorted(b.p_value.tolist())
    m = len(ps)
    kmax = 0
    for k, v in enumerate(ps, 1):
        if v <= k * 0.05 / m:
            kmax = k
    print("external BH: m=%d survivors=%d largest surviving p=%.4f" % (m, kmax, ps[kmax - 1] if kmax else float('nan')))
    surv = [r for r in b.sort_values("p_value").itertuples() if r.p_value <= kmax * 0.05 / m]
    print("\nsurviving external tests:")
    for r in surv:
        print("   %-8s %-20s %-8s p=%.4f" % (r.Group, r.Comparison, r.Metric, r.p_value))
    pd.DataFrame([{"Group": r.Group, "Comparison": r.Comparison, "Metric": r.Metric, "p": r.p_value}
                  for r in surv]).to_csv(f"{OUT_TS}/external_bh_survivors.csv", index=False)


if __name__ == "__main__":
    main()
