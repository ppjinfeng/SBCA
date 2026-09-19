"""Generate Supplementary Tables S3 (block-length) and S4 (factorial + CIs) from the
time-split results. S5 (reported-vs-replicated difference) is obsolete: Table 4 and
Table 5 of the manuscript now come from the same runs."""
import io
import pandas as pd

OUT_TS = "../results"
DEST = "../supplementary/_tables_new.tex"

GROUPS = ["2assets", "4assets", "6assets"]
METRICS = ["PV", "AR", "SR", "Sortino", "MDD", "Calmar"]


def star(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def table_s3():
    df = pd.read_csv(f"{OUT_TS}/blocklength_sensitivity.csv")
    out = []
    for fam in ["external", "internal"]:
        sub = df[df.Family == fam]
        comps = list(dict.fromkeys(sub.Comparison))
        for g in GROUPS:
            for c in comps:
                s = sub[(sub.Group == g) & (sub.Comparison == c)]
                if s.empty:
                    continue
                lines = []
                for me in ["AR", "SR", "Sortino", "MDD"]:
                    cells = []
                    for b in [20, 40, 60, 120]:
                        p = float(s[(s.Metric == me) & (s.block == b)].p_value.iloc[0])
                        cells.append("%.4f%s" % (p, star(p)))
                    lines.append("& %s & %s \\\\" % (me, " & ".join(cells)))
                lbl = c.replace("&", r"\&")
                block = "\\multirow{4}{*}{%s} & \\multirow{4}{*}{%s}\n" % (g, lbl)
                block += "\n\\cline{3-7}\n".join(lines)
                out.append(block)
                out.append("\\hline")
    return "\n".join(out)


def table_s4():
    ci = pd.read_csv(f"{OUT_TS}/factorial_ci.csv")
    key = {(r.Group, r.Metric, r.Contrast): r for r in ci.itertuples()}
    lines = []
    for g in GROUPS:
        for i, me in enumerate(METRICS):
            cells = []
            for k in ["dCM", "dAC", "dInt"]:
                r = key[(g, me, k)]
                mark = "$^{\\dagger}$" if r.excludes_zero == "yes" else ""
                cells.append("%+.4f & [%+.4f, %+.4f]%s" % (r.point, r.ci_lo, r.ci_hi, mark))
            lead = "\\multirow{6}{*}{%s}\n" % g if i == 0 else ""
            lines.append("%s& %s & %s \\\\" % (lead, me, " & ".join(cells)))
            if i < 5:
                lines.append("\\cline{2-8}")
        lines.append("\\hline")
    return "\n".join(lines)


s3, s4 = table_s3(), table_s4()

doc = r"""% ===== auto-generated supplementary tables S3-S4 (time-split results) =====

\newpage
\subsection*{Supplementary Table S3: Block-Length Sensitivity of the Block-Bootstrap $p$-Values}

\begin{table}[H]
\centering
\footnotesize
\begin{tabular}{|l|l|l|c|c|c|c|}
\hline
\multirow{2}{*}{Group} & \multirow{2}{*}{Comparison} & \multirow{2}{*}{Metric} & \multicolumn{4}{c|}{$p$-value at block length} \\
\cline{4-7}
& & & 20 & 40 & 60 & 120 \\
\hline
""" + s3 + r"""
\end{tabular}
\caption{Block-length sensitivity of the one-sided, $H_0$-centred, paired circular block-bootstrap tests for the external-baseline comparison (top block) and the internal ablation comparison (bottom block), across block lengths of 20, 40, 60 and 120 trading days. Each cell uses 30{,}000 resamples. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. The block length used for the tests reported in the main text is 60 trading days.}
\label{tab:s3}
\end{table}

\newpage
\subsection*{Supplementary Table S4: $2\times2$ Factorial Main Effects and Interaction}

\begin{table}[H]
\centering
\footnotesize
\begin{tabular}{|l|l|c|c|c|c|c|c|}
\hline
\multirow{2}{*}{Group} & \multirow{2}{*}{Metric} & \multicolumn{2}{c|}{$\Delta_{\mathrm{CM}}$} & \multicolumn{2}{c|}{$\Delta_{\mathrm{AC}}$} & \multicolumn{2}{c|}{$\Delta_{\mathrm{int}}$} \\
\cline{3-8}
& & point est. & 95\% CI & point est. & 95\% CI & point est. & 95\% CI \\
\hline
""" + s4 + r"""
\end{tabular}
\caption{Factorial decomposition of the $2\times2$ ablation into the two module main effects and their interaction, computed on the daily net log-return series of the runs reported in Table~4 of the main text. $\Delta_{\mathrm{CM}}=\frac{1}{2}\left[(M_{\mathrm{SBC}}-M_{\mathrm{SB}})+(M_{\mathrm{SBCA}}-M_{\mathrm{SBA}})\right]$ is the average effect of adding gated fusion; $\Delta_{\mathrm{AC}}=\frac{1}{2}\left[(M_{\mathrm{SBA}}-M_{\mathrm{SB}})+(M_{\mathrm{SBCA}}-M_{\mathrm{SBC}})\right]$ is the average effect of adding Actor-Critic; $\Delta_{\mathrm{int}}=M_{\mathrm{SBCA}}-M_{\mathrm{SBA}}-M_{\mathrm{SBC}}+M_{\mathrm{SB}}$ is the interaction. MDD is negative by construction, so for MDD a positive value still means less drawdown. The 95\% confidence intervals are percentile intervals from the same paired circular block bootstrap used elsewhere (30{,}000 resamples, block length 60 trading days, one block index applied to all four variants simultaneously). A dagger ($^{\dagger}$) marks intervals that exclude zero; only two of the 54 do, both the Actor-Critic main effect in the 2-asset group. These intervals are not corrected for multiple comparisons and should be read together with Table~5 of the main text.}
\label{tab:s4}
\end{table}
"""

io.open(DEST, "w", encoding="utf-8").write(doc)
print("wrote", DEST)
