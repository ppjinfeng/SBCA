# -*- coding: utf-8 -*-
"""
Re-derive the BERT sentiment feature with a CHRONOLOGICAL split.

The original Bert.py fine-tunes on `df.iloc[:int(0.8*len(df))]` after sorting by
[Stock_symbol, Date], i.e. on whole stocks rather than on a time window.  That
puts every row of CAT, GILD, GS and KO -- including the RL test period 2021-2022
-- inside the fine-tuning set, while the RL agent consumes delta_bert for all
rows.  This script instead fine-tunes only on news dated on or before
2018-12-31 (the RL training period), so that the validation (2019-2020) and test
(2021-2022) sentiment features are genuinely out of sample.

Hyperparameters are kept identical to Bert.py: bert-base-uncased, 2 labels,
4 epochs, lr 2e-5, batch 16, max_len 64.

Outputs
  ./out/bert_pred_timesplit.csv        full table with the new delta_bert
  ./out/bert_split_compare.csv         per-stock AUC, old vs new, by period
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(encoding="utf-8")

SRC = os.environ.get("SBCA_RAW_NEWS", "data/stock_news_trading_data.csv")
OLD = os.environ.get("SBCA_ORIGINAL_FEATURES", "data/bert_pred_for_SARL.csv")
OUTDIR = "runs"
BERT_NAME = "bert-base-uncased"
TRAIN_END = pd.Timestamp("2018-12-31")   # RL training period ends here
MAX_LEN = 64
BATCH = 16
EPOCHS = 4
LR = 2e-5
SEED = 42

os.makedirs(OUTDIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_frame():
    df = pd.read_csv(SRC)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Stock_symbol", "Date"]).reset_index(drop=True)
    df["label"] = df.groupby("Stock_symbol")["close"].transform(
        lambda s: (s.shift(-1) > s).astype("float"))
    df = df.dropna(subset=["label", "Titles_combined"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tok):
        self.texts, self.labels, self.tok = texts, labels, tok

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(str(self.texts[i]), truncation=True, padding="max_length",
                       max_length=MAX_LEN, return_tensors="pt")
        return {"input_ids": enc["input_ids"].flatten(),
                "attention_mask": enc["attention_mask"].flatten(),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(s).rank().values
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    from transformers import BertTokenizer, BertForSequenceClassification

    df = build_frame()
    print(f"rows after cleaning: {len(df)}", flush=True)
    tr_mask = (df["Date"] <= TRAIN_END).values
    print(f"BERT fine-tuning rows (Date <= {TRAIN_END.date()}): {tr_mask.sum()}   "
          f"held out: {(~tr_mask).sum()}", flush=True)

    tok = BertTokenizer.from_pretrained(BERT_NAME)
    model = BertForSequenceClassification.from_pretrained(BERT_NAME, num_labels=2).to(DEVICE)

    tr = df[tr_mask]
    ds = NewsDataset(tr["Titles_combined"].tolist(), tr["label"].tolist(), tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for b in dl:
            ids = b["input_ids"].to(DEVICE)
            am = b["attention_mask"].to(DEVICE)
            lb = b["label"].to(DEVICE)
            opt.zero_grad()
            out = model(ids, attention_mask=am, labels=lb)
            out.loss.backward()
            opt.step()
            tot += out.loss.item()
        print(f"  epoch {ep+1}/{EPOCHS}  mean loss {tot/len(dl):.4f}", flush=True)

    # predict for ALL rows (same as Bert.py line 73)
    model.eval()
    probs = []
    with torch.no_grad():
        allds = NewsDataset(df["Titles_combined"].tolist(), df["label"].tolist(), tok)
        alldl = DataLoader(allds, batch_size=BATCH, shuffle=False)
        for b in alldl:
            out = model(b["input_ids"].to(DEVICE), attention_mask=b["attention_mask"].to(DEVICE))
            probs.extend(torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy().tolist())
    df["delta_bert_new"] = probs
    df.to_csv(os.path.join(OUTDIR, "bert_pred_timesplit.csv"), index=False)
    print(f"\nsaved {os.path.join(OUTDIR, 'bert_pred_timesplit.csv')}", flush=True)

    # ---- compare with the original feature ----
    old = pd.read_csv(OLD)[["Date", "Stock_symbol", "label", "delta_bert"]]
    old["Date"] = pd.to_datetime(old["Date"])
    m = df.merge(old, on=["Date", "Stock_symbol"], how="inner", suffixes=("", "_old"))
    print(f"merged rows: {len(m)}", flush=True)

    rows = []
    for st, g in m.groupby("Stock_symbol"):
        ins = (g["Date"] <= TRAIN_END).all()
        tag = "in-sample (orig)" if ins else ("partial (orig)" if (g["Date"] <= TRAIN_END).any() else "out (orig)")
        te = g[g["Date"] >= "2021-01-01"]
        va = g[(g["Date"] >= "2019-01-01") & (g["Date"] <= "2020-12-31")]
        rows.append(dict(
            stock=st,
            orig_split_role=tag,
            auc_test_old=auc(te["label"], te["delta_bert"]),
            auc_test_new=auc(te["label"], te["delta_bert_new"]),
            acc_test_old=((te["delta_bert"] > .5).astype(int) == te["label"]).mean(),
            acc_test_new=((te["delta_bert_new"] > .5).astype(int) == te["label"]).mean(),
            auc_val_new=auc(va["label"], va["delta_bert_new"]),
            auc_train_new=auc(g[g["Date"] <= TRAIN_END]["label"],
                              g[g["Date"] <= TRAIN_END]["delta_bert_new"]),
        ))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUTDIR, "bert_split_compare.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n" + r.to_string(index=False, float_format=lambda x: "%.4f" % x), flush=True)


if __name__ == "__main__":
    main()
