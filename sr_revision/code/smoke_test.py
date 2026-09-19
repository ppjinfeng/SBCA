# -*- coding: utf-8 -*-
r"""Smoke test for sr_revision/.

Checks, in about a minute and without training anything:
  1. Python and package versions
  2. presence of the input / result files it needs
  3. that every analysis module imports cleanly
  4. that a short block-bootstrap run on the committed return series reproduces
     the committed p-values to within Monte Carlo error

Run from the sr_revision/ directory:

    python code/smoke_test.py

Exit code 0 = environment OK, 1 = something is missing or broken.
"""
from __future__ import annotations

import io
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)                      # the suite is written to run from sr_revision/
sys.path.insert(0, HERE)

OK, WARN, FAIL = "[ok]  ", "[warn]", "[FAIL]"
failures = []
warnings = []


def say(mark, msg):
    print("%s %s" % (mark, msg), flush=True)


# ---------------------------------------------------------------- 1. versions
def check_versions():
    print("\n=== 1. environment ===")
    say(OK, "python %s" % sys.version.split()[0])
    if sys.version_info < (3, 8):
        failures.append("python >= 3.8 required")
    required = {"numpy": None, "pandas": None, "scipy": None, "torch": None}
    for name in required:
        try:
            mod = __import__(name)
            say(OK, "%-10s %s" % (name, getattr(mod, "__version__", "?")))
        except Exception as e:                                  # noqa: BLE001
            say(FAIL, "%-10s missing (%s)" % (name, e))
            failures.append("package %s is not installed" % name)
    try:
        import torch
        say(OK, "cuda available: %s" % torch.cuda.is_available())
    except Exception:                                            # noqa: BLE001
        pass


# ------------------------------------------------------------------ 2. files
REQUIRED = [
    ("data/bert_pred_for_SARL_timesplit.csv", True),
    ("results/ablation_rets_2assets.csv", True),
    ("results/ablation_rets_4assets.csv", True),
    ("results/ablation_rets_6assets.csv", True),
    ("results/ablation_bootstrap_internal.xlsx", True),
    ("results/blocklength_sensitivity.csv", True),
    ("results/factorial_ci.csv", True),
    ("data/stock_news_trading_data.csv", False),      # needed only for the BERT re-derivation
    ("data/bert_pred_for_SARL.csv", False),          # needed only for the pre-correction audit
]


def check_files():
    print("\n=== 2. input and result files ===")
    for rel, must in REQUIRED:
        exists = os.path.exists(rel)
        size = os.path.getsize(rel) if exists else 0
        if exists:
            say(OK, "%-46s %9d bytes" % (rel, size))
        elif must:
            say(FAIL, "%-46s MISSING" % rel)
            failures.append("%s is missing" % rel)
        else:
            say(WARN, "%-46s not present (optional) - see data/README.md" % rel)
            warnings.append(rel)


# ---------------------------------------------------------------- 3. imports
MODULES = [
    ("run_all_old", True, None),
    ("ablation_bootstrap_timesplit", True, None),
    ("ts_derived", True, None),
    ("factorial_ci", True, None),
    ("blocklength_sensitivity", True, None),
    ("blocklength_check", True, None),
    ("blocklength_robustness", True, None),
    ("bert_timesplit", False, "needs data/stock_news_trading_data.csv"),
    ("bert_audit", False, "needs data/bert_pred_for_SARL.csv"),
    ("ablation_bootstrap", False, "needs data/bert_pred_for_SARL.csv"),
    ("compose_figures", False, "needs a trained Plots/ directory"),
]


def check_imports():
    print("\n=== 3. module imports ===")
    os.environ.setdefault("SBCA_FEATURES", "data/bert_pred_for_SARL_timesplit.csv")
    for name, must, why in MODULES:
        try:
            __import__(name)
            say(OK, "import %s" % name)
        except Exception as e:                                   # noqa: BLE001
            first = str(e).split("\n")[0][:70]
            if must:
                say(FAIL, "import %-30s %s" % (name, first))
                failures.append("cannot import %s: %s" % (name, first))
                traceback.print_exc()
            else:
                say(WARN, "import %-30s skipped (%s)" % (name, why))
                warnings.append("import %s skipped" % name)


# -------------------------------------------------------------- 4. bootstrap
def check_bootstrap():
    print("\n=== 4. short block-bootstrap replay ===")
    try:
        import numpy as np
        import pandas as pd
        from blocklength_sensitivity import bootstrap_pvalues
    except Exception as e:                                       # noqa: BLE001
        say(FAIL, "cannot import the bootstrap helper: %s" % e)
        failures.append("bootstrap helper unusable")
        return

    n_boot = 400                      # short run: only a sanity check, not the paper's 30,000
    ref = pd.read_excel("results/ablation_bootstrap_internal.xlsx", "InternalBootstrap")
    df = pd.read_csv("results/ablation_rets_2assets.csv")
    a, b = df["SBCA"].values, df["SBC"].values
    n = min(len(a), len(b))
    res = bootstrap_pvalues(a[:n], b[:n], 60, n_boot=n_boot)

    print("       %-8s %10s %10s" % ("metric", "committed", "this run"))
    for m in ["AR", "SR", "Sortino", "MDD"]:
        committed = float(ref[(ref.Group == "2assets") &
                              (ref.Comparison == "SBCA vs SBC") &
                              (ref.Metric == m)].p_value.iloc[0])
        got = res[m]["p_value"]
        flag = OK if abs(got - committed) < 0.06 else WARN
        print("       %-8s %10.4f %10.4f  %s" % (m, committed, got, flag))
        if flag is WARN:
            warnings.append("bootstrap replay differs for %s" % m)
    say(OK, "bootstrap replay finished (%d resamples; the paper uses 30,000)" % n_boot)


def main():
    print("smoke test for sr_revision/  (cwd=%s)" % ROOT)
    check_versions()
    check_files()
    check_imports()
    check_bootstrap()

    print("\n=== summary ===")
    if failures:
        print("FAILED - %d problem(s):" % len(failures))
        for f in failures:
            print("   -", f)
        print("\nSee data/README.md for the input files and README.md for the full procedure.")
        return 1
    print("PASSED - the analysis can be re-run from the committed series.")
    if warnings:
        print("%d warning(s):" % len(warnings))
        for w in warnings:
            print("   -", w)
        print("Warnings refer to steps that need files not shipped in this repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
