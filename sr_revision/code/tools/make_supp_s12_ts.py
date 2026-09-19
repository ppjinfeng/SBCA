"""Regenerate Supplementary Tables S1 (EMA) and S2 (window) from the time-split runs."""
import io
import pandas as pd

SI = os.environ.get("SBCA_SUPP", "../supplementary/supplementary.tex")
RES = "../runs/result"
GROUPS = ["2assets", "4assets", "6assets"]
METRICS = ["PV", "AR", "SR", "Sortino", "MDD", "Calmar"]


def cells(row, base):
    out = []
    for k in METRICS:
        d = float(row[k]) - float(base[k])
        if abs(d) < 0.00005:
            out.append("0.0000")
        else:
            out.append(("$-$" if d < 0 else "$+$") + "%.4f" % abs(d))
    return " & ".join(out)


def build(path, sheet, col, default, header_tex):
    d = pd.read_excel(path, sheet)
    lines = []
    for g in GROUPS:
        s = d[d.Group == g].set_index(col)
        base = s.loc[default]
        lines.append("\\multirow{4}{*}{%s}" % g)
        vals = [v for v in sorted(s.index) if v != default]
        order = [v for v in sorted(s.index)]
        for i, v in enumerate(order):
            if v == default:
                body = "0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000 & 0.0000"
                lab = "\\textbf{%s}" % header_tex(v)
            else:
                body = cells(s.loc[v], base)
                lab = header_tex(v)
            lines.append("& %s & %s \\\\" % (lab, body))
            if i < len(order) - 1:
                lines.append("\\cline{2-8}")
        lines.append("\\hline")
    return "\n".join(lines)


def h_ema(v):
    return ("%g" % v)


def h_win(v):
    return ("%g" % v)


s1 = build(f"{RES}/sensitivity_ema.xlsx", "Sheet1", "EMA", 0.4, h_ema)
s2 = build(f"{RES}/sensitivity_window.xlsx", "Sheet1", "WindowSize", 30, h_win)

NEW_S1 = r"""\begin{table}[H]
\centering
\footnotesize

\begin{tabular}{|l|c|c|c|c|c|c|c|}
\hline
\textbf{Group} & $\boldsymbol{\alpha}$ & $\Delta$\textbf{PV} & $\Delta$\textbf{AR} & $\Delta$\textbf{SR} & $\Delta$\textbf{Sortino} & $\Delta$\textbf{MDD} & $\Delta$\textbf{Calmar} \\
\hline
""" + s1 + r"""
\end{tabular}
\caption{Deviation of each evaluation metric from the default $\alpha=0.4$ (highlighted row). Values denote $\Delta = \text{new} - \text{default}$; positive $\Delta$MDD indicates reduced drawdown.}
\label{tab:s1}
\end{table}"""

NEW_S2 = r"""\begin{table}[H]
\centering
\footnotesize

\begin{tabular}{|l|c|c|c|c|c|c|c|}
\hline
\textbf{Group} & $\boldsymbol{W}$ & $\Delta$\textbf{PV} & $\Delta$\textbf{AR} & $\Delta$\textbf{SR} & $\Delta$\textbf{Sortino} & $\Delta$\textbf{MDD} & $\Delta$\textbf{Calmar} \\
\hline
""" + s2 + r"""
\end{tabular}
\caption{Deviation of each evaluation metric from the default $W=30$ (highlighted row). Values denote $\Delta = \text{new} - \text{default}$; positive $\Delta$MDD indicates reduced drawdown.}
\label{tab:s2}
\end{table}"""


def main():
    t = io.open(SI, encoding="utf-8").read()
    a = t.index("\\begin{table}[H]", t.index("Supplementary Table S1"))
    b = t.index("\\end{table}", a) + len("\\end{table}")
    t = t[:a] + NEW_S1 + t[b:]
    a = t.index("\\begin{table}[H]", t.index("Supplementary Table S2"))
    b = t.index("\\end{table}", a) + len("\\end{table}")
    t = t[:a] + NEW_S2 + t[b:]
    io.open(SI, "w", encoding="utf-8", newline="\n").write(t)
    print("S1 and S2 regenerated")


if __name__ == "__main__":
    main()
