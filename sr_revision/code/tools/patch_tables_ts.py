"""Patch SBCA-SR.tex: replace Table 4 and Table 5 with the time-split results and
remove the now-obsolete replication wording (Table 4 and Table 5 come from the
same run under the time-split features)."""
import io
import re
import sys
import pandas as pd

MAIN = os.environ.get("SBCA_TEX", "../SBCA-SR.tex")
OUT = "../results"

METRICS = ["PV", "AR", "SR", "Sortino", "MDD", "Calmar"]
MODELS = ["SB", "SBA", "SBC", "SBCA"]
GROUPS = ["2assets", "4assets", "6assets"]


def star(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def ablation_tabular():
    lines = []
    for g in GROUPS:
        d = pd.read_csv(f"{OUT}/ablation_metrics_{g}.csv").set_index("Model")
        for i, m in enumerate(MODELS):
            vals = " & ".join("%.4f" % float(d.loc[m, k]) for k in METRICS)
            lead = ("\\multirow{4}{*}{%s}\n" % g) if i == 0 else ""
            lines.append("%s& %s & %s \\\\" % (lead, m, vals))
            if i < 3:
                lines.append("\\cline{2-8}")
        lines.append("\\hline")
    return "\n".join(lines)


def bootstrap_tabular():
    b = pd.read_excel(f"{OUT}/ablation_bootstrap_internal.xlsx", "InternalBootstrap")
    lines = []
    for comp in ["SBCA vs SB", "SBCA vs SBA", "SBCA vs SBC"]:
        s = b[b.Comparison == comp]
        for i, me in enumerate(["AR", "SR", "Sortino", "MDD"]):
            cells = []
            for g in GROUPS:
                p = float(s[(s.Group == g) & (s.Metric == me)].p_value.iloc[0])
                cells.append("%.4f%s" % (p, star(p)))
            short = comp.replace("SBCA vs ", "")
            lead = ("\\multirow{4}{*}{SBCA vs %s}\n" % short) if i == 0 else ""
            lines.append("%s& %s & %s \\\\" % (lead, me, " & ".join(cells)))
            if i < 3:
                lines.append("\\cline{2-5}")
        lines.append("\\hline")
    return "\n".join(lines)


def replace_table(text, label, new_block):
    i = text.index("\\label{%s}" % label)
    end = text.index("\\end{table}", i) + len("\\end{table}")
    start = text.rindex("\\begin{table}", 0, i)
    return text[:start] + new_block + text[end:]


def main():
    t = io.open(MAIN, encoding="utf-8").read()

    t4 = (r"""\begin{table}[ht]
\centering
\begin{tabular}{|l|l|c|c|c|c|c|c|}
\hline
Group & Model & PV & AR & SR & Sortino & MDD & Calmar \\
\hline
""" + ablation_tabular() + r"""
\end{tabular}
\caption{Ablation study of model components. SB uses neither the cross-modal gated fusion module nor the Actor-Critic mechanism; SBA adds only Actor-Critic; SBC adds only cross-modal gated fusion; SBCA is the full model. All variants are trained under the identical protocol, hyperparameters and seed, and the significance tests of Table~\ref{tab:ablation_bootstrap} are computed on the daily net log-return series of these same runs.}
\label{tab:ablation}
\end{table}""")

    t5 = (r"""\begin{table}[ht]
\centering
\footnotesize
\begin{tabular}{|l|l|c|c|c|}
\hline
\multirow{2}{*}{Comparison} & \multirow{2}{*}{Metric} & \multicolumn{3}{c|}{$p$-value} \\
\cline{3-5}
& & 2assets & 4assets & 6assets \\
\hline
""" + bootstrap_tabular() + r"""
\end{tabular}
\caption{Internal block-bootstrap significance tests for the $2\times2$ ablation design (30,000 resamples, one-sided, fixed block length of 60 trading days), computed on the daily net log-return series of the runs reported in Table~\ref{tab:ablation}. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. The null hypothesis is that SBCA does not outperform the specified ablation variant on the given metric; maximum drawdown is sign-flipped so that ``outperform'' again means a larger metric value.}
\label{tab:ablation_bootstrap}
\end{table}""")

    t = replace_table(t, "tab:ablation", t4)
    t = replace_table(t, "tab:ablation_bootstrap", t5)

    # drop the obsolete replication paragraph in 6.2
    para = t[t.index("Finally, the $p$-values of Table~\\ref{tab:ablation_bootstrap}"):
             t.index("Finally, the $p$-values of Table~\\ref{tab:ablation_bootstrap}") + 1200]
    end = para.index("\n\n")
    t = t.replace(para[:end], "")

    io.open(MAIN, "w", encoding="utf-8", newline="\n").write(t)
    print("patched:", MAIN)
    print("replication paragraph removed:", "were not retained from the runs behind" not in t)


if __name__ == "__main__":
    main()
