"""
Ablation internal significance test.

Reproduces the 2x2 ablation (SB / SBA / SBC / SBCA) at seed 42 for the three
asset groups, then runs the same circular block-bootstrap one-sided test used
for the external-baseline comparison, but between SBCA and each internal
ablation variant.

Outputs (./out):
  ablation_rets_<group>.csv      daily net log returns per variant
  ablation_metrics_<group>.csv   PV/AR/SR/Sortino/MDD/Calmar per variant
  ablation_bootstrap_internal.xlsx   pairwise one-sided p-values
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_all_old as R

SEED = 42
METRICS = ["AR", "SR", "Sortino", "MDD"]

VARIANTS = [
    ("SB",   R.Policy_BERT,      False, False),
    ("SBA",  R.Policy_BERT_AC,   False, True),
    ("SBC",  R.Policy_BERT_CM,   True,  False),
    ("SBCA", R.Policy_BERT_CM_AC, True, True),
]


def model_kwargs(model_cls, env):
    if model_cls in (R.Policy_BERT, R.Policy_BERT_AC):
        return {"state_dim": R.WINDOW_SIZE * env.n_assets + env.n_assets,
                "n_assets": env.n_assets}
    return {"win_size": R.WINDOW_SIZE, "n_assets": env.n_assets}


def run_group(stocks, suffix):
    close_arr = R.df_close[stocks].values
    delta_arr = R.df_delta[stocks].values
    env = R.StockEnv(close_arr, delta_arr)
    TR = range(R.WINDOW_SIZE, R.train_end - 1)
    VR = range(R.train_end, R.val_end - 1)
    TE = range(R.val_end, env.days - 1)

    rets, metrics_rows, pv_series = {}, [], {}
    for tag, cls, is_cm, is_ac in VARIANTS:
        R.set_seed(SEED)
        model = cls(**model_kwargs(cls, env)).to(R.DEVICE)
        opt = R.optim.AdamW(model.parameters(), lr=R.LR, weight_decay=1e-5)
        sched = R.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", 0.5, 1, verbose=False)
        model = R._train_loop(env, model, opt, TR, VR, is_cm, is_ac,
                              f"{tag} {suffix}", tag, f"_{suffix}",
                              R.COMMISSION, sched, R.EMA_ALPHA, lean=True)
        pv, simple_rets, _ = R._backtest(env, model, TE, is_cm, is_ac, R.COMMISSION)
        log_rets = np.log(1.0 + np.array(simple_rets))
        rets[tag] = log_rets
        pv_series[tag] = pv
        row = {"Group": suffix, "Model": tag, **R.evaluate(pv)}
        metrics_rows.append(row)
        print(f"  [{suffix}] {tag:5s} " +
              " ".join(f"{k}={row[k]:.4f}" for k in ["PV", "AR", "SR", "Sortino", "MDD", "Calmar"]),
              flush=True)

    n = min(len(v) for v in rets.values())
    df_rets = pd.DataFrame({k: v[:n] for k, v in rets.items()})
    df_rets.to_csv(f"./out/ablation_rets_{suffix}.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(f"./out/ablation_metrics_{suffix}.csv", index=False)
    pd.DataFrame(pv_series).to_csv(f"./out/ablation_pv_{suffix}.csv", index=False)
    return rets, metrics_rows


def main():
    os.makedirs("./out", exist_ok=True)
    stock_groups = [
        (R.all_stocks[:2], "2assets"),
        (R.all_stocks[:4], "4assets"),
        (R.all_stocks[:6], "6assets"),
    ]

    boot_rows, metric_rows = [], []
    for stocks, suffix in stock_groups:
        print(f"\n=== Group {suffix} ===", flush=True)
        rets, mrows = run_group(stocks, suffix)
        metric_rows.extend(mrows)

        ref = rets["SBCA"]
        for tag in ["SB", "SBA", "SBC"]:
            cmp_rets = rets[tag]
            n = min(len(ref), len(cmp_rets))
            res = R.block_bootstrap_one_sided_test(ref[:n], cmp_rets[:n])
            for metric, r in res.items():
                p = r["p_value"]
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "-"
                boot_rows.append({"Group": suffix, "Comparison": f"SBCA vs {tag}",
                                  "Metric": metric, "diff_true": r["diff_true"],
                                  "p_value": p, "sig": sig})
                print(f"    SBCA vs {tag:4s} {metric:8s} diff={r['diff_true']:+.4f} p={p:.4f} {sig}",
                      flush=True)

    df_metrics = pd.DataFrame(metric_rows)
    df_boot = pd.DataFrame(boot_rows)
    with pd.ExcelWriter("./out/ablation_bootstrap_internal.xlsx") as w:
        df_boot.to_excel(w, sheet_name="InternalBootstrap", index=False)
        df_metrics.to_excel(w, sheet_name="AblationMetrics", index=False)
    print("\nSaved ./out/ablation_bootstrap_internal.xlsx", flush=True)
    print(df_boot.to_string(index=False))


if __name__ == "__main__":
    main()
