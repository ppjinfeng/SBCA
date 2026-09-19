"""Append Supplementary Table S5 (sentiment-feature correction audit) to the
supplementary tables file, and rename it to tables_s3_s5.tex."""
import io
import os
import shutil

SRC = os.environ.get("SBCA_SUPP_TMPL", "../supplementary/_tables_new.tex")
DST = os.environ.get("SBCA_SUPP_OUT", "../supplementary/tables_s3_s5.tex")
ROWS = "../runs/bert_audit_rows.tex"

rows = io.open(ROWS, encoding="utf-8").read()

S5 = r"""
\newpage
\subsection*{Supplementary Table S5: Audit of the Sentiment-Feature Correction}

\begin{table}[H]
\centering
\footnotesize
\begin{tabular}{|l|l|l|c|c|c|c|c|}
\hline
\multirow{2}{*}{Stock} & \multirow{2}{*}{Role under the original split} & \multirow{2}{*}{Period} & \multirow{2}{*}{$n$} & \multirow{2}{*}{Pos.\ rate} & \multirow{2}{*}{AUC original} & \multirow{2}{*}{AUC corrected} & \multicolumn{1}{c|}{95\% CI} \\
& & & & & & & \multicolumn{1}{c|}{(corrected)} \\
\hline
""" + rows + r"""
\end{tabular}
\caption{Per-stock, per-period audit of the sentiment feature before and after the correction. ``AUC original'' is the area under the ROC curve of the score produced by the originally released pipeline, in which the BERT fine-tuning set was formed by taking the first 80\% of rows after sorting by stock, so that four of the six assets were entirely inside the fine-tuning set. ``AUC corrected'' is the same quantity for the score re-derived here, where the encoder is fine-tuned only on headlines dated on or before 31 December 2018. The confidence interval is a percentile bootstrap interval (2{,}000 resamples) for the corrected AUC. In the training period the corrected encoder attains AUC $0.69$--$0.91$, which is expected for an in-sample fit; in the validation and test periods every corrected AUC lies in $[0.46, 0.55]$ with an interval that contains $0.50$, whereas the original score reached $0.76$ in the test period for KO.}
\label{tab:s5}
\end{table}
"""

t = io.open(SRC, encoding="utf-8").read().rstrip() + "\n" + S5
io.open(DST, "w", encoding="utf-8", newline="\n").write(t)
if os.path.exists(SRC):
    os.remove(SRC)
print("wrote", DST, len(t), "chars")
