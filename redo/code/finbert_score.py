"""
finbert_score.py — Generate delta_bert scores using pre-trained FinBERT
No fine-tuning needed — FinBERT is pre-trained for 3-class financial sentiment
Changes: ProsusAI/finbert, max_len=128
Output: data/bert_pred_24stocks_delta_FinBERT.csv
"""
import pandas as pd, numpy as np, torch, os, sys
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Device: {DEVICE}")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
MODEL_NAME = 'ProsusAI/finbert'
BATCH_SIZE, MAX_LEN = 16, 128

data_csv = f"{BASE}/data/bert_pred_24stocks_NEW.csv"
out_csv = f"{BASE}/data/bert_pred_24stocks_delta_FinBERT.csv"

df = pd.read_csv(data_csv); df['Date'] = pd.to_datetime(df['Date'])

# ========== Generate delta_bert scores (pre-trained, no fine-tuning) ==========
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

print(f"Loading FinBERT: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEVICE)
model.eval()

# Only process rows with actual news (exclude NaN and empty)
has_news = df[df['Titles_combined'].notna() & (df['Titles_combined'] != '')]
print(f"Rows with news: {len(has_news)} / {len(df)}")

class ND(Dataset):
    def __init__(s, tx): s.tx = tx
    def __len__(s): return len(s.tx)
    def __getitem__(s, idx):
        inp = tokenizer(str(s.tx[idx]), truncation=True, padding='max_length',
                      max_length=MAX_LEN, return_tensors='pt')
        return {"input_ids": inp['input_ids'].flatten(),
                "attention_mask": inp['attention_mask'].flatten()}

ds = ND(has_news['Titles_combined'].values)
ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

probs = []
with torch.no_grad():
    for batch in tqdm(ld, desc="FinBERT"):
        ids = batch['input_ids'].to(DEVICE)
        am = batch['attention_mask'].to(DEVICE)
        logits = model(ids, attention_mask=am).logits
        s = torch.softmax(logits, dim=1)
        # 3-class: 0=neg, 1=neutral, 2=pos → score = 0.5*(1 + P(pos) - P(neg))
        score = 0.5 * (1 + s[:, 2] - s[:, 0])
        probs.extend(score.cpu().numpy())

df['delta_bert'] = 0.5  # default neutral
df.loc[df['Titles_combined'].notna() & (df['Titles_combined'] != ''), 'delta_bert'] = probs

df.to_csv(out_csv, index=False)
scores = df['delta_bert'].values
print(f"\nSaved: {out_csv}")
print(f"FinBERT scores: mean={scores.mean():.4f}, std={scores.std():.4f}, "
      f"range=[{scores.min():.4f}, {scores.max():.4f}]")
print(f"pct in [0.45,0.55]: {np.mean((scores>=0.45)&(scores<=0.55))*100:.1f}%")
print(f"pct in [0.48,0.52]: {np.mean((scores>=0.48)&(scores<=0.52))*100:.1f}%")
print("Done.")
