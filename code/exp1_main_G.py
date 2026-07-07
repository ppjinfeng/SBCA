"""
exp1_main_G.py — Main experiment: SBCA vs. 8 baselines x 11 groups
=============================================================================
Step 1: Fine-tune 3 BERT models (cutoff=2017-12-31)
Step 2: Generate delta_bert sentiment scores
Step 3: Train and evaluate SBCA vs. baselines
Output: result/exp1_main_G.csv, result/exp1_main_G_H.xlsx
"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
import random, copy, os, sys
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{BASE}/result", exist_ok=True); os.makedirs(f"{BASE}/models_bert", exist_ok=True)
print(f"Device: {DEVICE}")

W, COMM, GAM = 30, 0.0025, 0.99; LRISK, EA, TP = 0.1, 0.4, 0.005
EP, PAT, LR, ENT = 30, 5, 3e-4, 0.1
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ========== Temporal Split ==========
BERT_CUTOFF = '2017-12-31'
TR_START, TR_END = '2012-01-01', '2017-12-31'
VA_END = '2019-12-31'
TE_END = '2021-12-31'

LARGE=['NVDA','ORCL','CRM','QCOM','WFC','MRK','KO','CAT']
MID=['BBY','CLX','BIIB','AA','KEY','WDC','AAL','HAL']
SMALL=['SPWR','URBN','ANF','FL','PLUG','KSS','NVAX','ALK']
# ALL_24 already defined via LARGE+MID+SMALL, update MID below
ALL_24 = LARGE + MID + SMALL

# ========== Step 1: Fine-tune BERT Models ==========
data_csv = f"{BASE}/data/bert_pred_24stocks.csv"
delta_csv_G = f"{BASE}/data/bert_pred_24stocks_delta_G.csv"
bert_paths = {t: f"{BASE}/models_bert/best_{t}_G.pth" for t in ['large','mid','small']}
TIERS = {'large': LARGE, 'mid': MID, 'small': SMALL}

if not all(os.path.exists(p) for p in bert_paths.values()):
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertForSequenceClassification, AdamW
    B, MLEN = 16, 64

    df = pd.read_csv(data_csv); df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(["Stock_symbol", "Date"]).reset_index(drop=True)

    for tier_name, tier_stocks in TIERS.items():
        mp = bert_paths[tier_name]
        if os.path.exists(mp): print(f"BERT-{tier_name}_G exists, skip"); continue
        df_t = df[df['Stock_symbol'].isin(tier_stocks)]
        tr = df_t[df_t['Date'] <= BERT_CUTOFF]
        va = df_t[(df_t['Date'] > BERT_CUTOFF) & (df_t['Date'] <= '2019-12-31')]
        print(f"BERT-{tier_name}_G: Train={len(tr)}, Val={len(va)} (cutoff={BERT_CUTOFF})")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(DEVICE)

        class ND(Dataset):
            def __init__(s,tx,lb): s.tx=tx; s.lb=lb
            def __len__(s): return len(s.tx)
            def __getitem__(s,idx):
                inp = tokenizer(str(s.tx[idx]),truncation=True,padding='max_length',max_length=MLEN,return_tensors='pt')
                return {"input_ids":inp['input_ids'].flatten(),"attention_mask":inp['attention_mask'].flatten(),"label":torch.tensor(s.lb[idx],dtype=torch.long)}

        tr_ds=ND(tr['Titles_combined'].values,tr['label'].values)
        va_ds=ND(va['Titles_combined'].values,va['label'].values)
        tr_ld=DataLoader(tr_ds,batch_size=B,shuffle=True)
        va_ld=DataLoader(va_ds,batch_size=B,shuffle=False)
        opt=AdamW(model.parameters(),lr=2e-5); model.train(); best_va=0
        for ep in range(4):
            tl=0
            for batch in tqdm(tr_ld,desc=f"  E{ep+1}",leave=False):
                ids=batch['input_ids'].to(DEVICE); am=batch['attention_mask'].to(DEVICE); lb=batch['label'].to(DEVICE)
                opt.zero_grad(); loss=model(ids,attention_mask=am,labels=lb).loss; loss.backward(); opt.step(); tl+=loss.item()
            model.eval(); vc,vt=0,0
            with torch.no_grad():
                for batch in va_ld:
                    ids=batch['input_ids'].to(DEVICE); am=batch['attention_mask'].to(DEVICE); lb=batch['label'].to(DEVICE)
                    out=model(ids,attention_mask=am); vc+=(torch.argmax(out.logits,dim=1)==lb).sum().item(); vt+=lb.size(0)
            va_acc=vc/vt; print(f"  E{ep+1}: loss={tl/len(tr_ld):.4f} val_acc={va_acc:.4f}")
            model.train()
            if va_acc>best_va: best_va=va_acc; torch.save(model.state_dict(),mp)
        print(f"  BERT-{tier_name}_G: best_val_acc={best_va:.4f}")
else:
    print("All 3 BERT_G models exist, skip training")

# ========== Step 2: Generate delta_bert Scores ==========
if not os.path.exists(delta_csv_G):
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertForSequenceClassification
    B, MLEN = 16, 64

    df = pd.read_csv(data_csv); df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(["Stock_symbol", "Date"]).reset_index(drop=True)

    for tier_name, tier_stocks in TIERS.items():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(DEVICE)
        model.load_state_dict(torch.load(bert_paths[tier_name])); model.eval()

        class ND(Dataset):
            def __init__(s,tx,lb): s.tx=tx; s.lb=lb
            def __len__(s): return len(s.tx)
            def __getitem__(s,idx):
                inp = tokenizer(str(s.tx[idx]),truncation=True,padding='max_length',max_length=MLEN,return_tensors='pt')
                return {"input_ids":inp['input_ids'].flatten(),"attention_mask":inp['attention_mask'].flatten(),"label":torch.tensor(s.lb[idx],dtype=torch.long)}

        t_df = df[df['Stock_symbol'].isin(tier_stocks)]
        ds = ND(t_df['Titles_combined'].values, t_df['label'].values)
        ld = DataLoader(ds, batch_size=B, shuffle=False)
        probs = []
        with torch.no_grad():
            for batch in tqdm(ld, desc=f"delta_{tier_name}", leave=False):
                ids=batch['input_ids'].to(DEVICE); am=batch['attention_mask'].to(DEVICE)
                probs.extend(torch.softmax(model(ids,attention_mask=am).logits,dim=1)[:,1].cpu().numpy())
        df.loc[df['Stock_symbol'].isin(tier_stocks), 'delta_bert'] = probs

    df.to_csv(delta_csv_G, index=False)
    print(f"delta_bert_G saved: mean={df['delta_bert'].mean():.3f}")
else:
    df = pd.read_csv(delta_csv_G); df['Date'] = pd.to_datetime(df['Date'])
    print(f"delta_bert_G exists: mean={df['delta_bert'].mean():.3f}")

# ========== Step 3: SBCA Experiments ==========
print("\n" + "="*60)
print("  EXP1_G: Train=2012-2017 Val=2018-2019 Test=2020-2021")
print("="*60)

df_c = df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
df_d = df.pivot_table(index='Date',columns='Stock_symbol',values='delta_bert').fillna(0.5)

dates = df_c.index
TE = dates.get_indexer([pd.to_datetime(TR_END)], method='pad')[0]
VE = dates.get_indexer([pd.to_datetime(VA_END)], method='pad')[0]
TEST_END = dates.get_indexer([pd.to_datetime(TE_END)], method='pad')[0]
DAYS_ = TEST_END
print(f"Train: {dates[0].date()}~{dates[TE].date()}, Val: ~{dates[VE].date()}, Test: ~{dates[TEST_END].date()}")

all_s = df_c.columns.tolist()
SIZE_MAP={}
for s in LARGE: SIZE_MAP[s]='large'
for s in MID: SIZE_MAP[s]='mid'
for s in SMALL: SIZE_MAP[s]='small'
L=[s for s in all_s if SIZE_MAP.get(s)=='large']
M=[s for s in all_s if SIZE_MAP.get(s)=='mid']
S=[s for s in all_s if SIZE_MAP.get(s)=='small']

GROUPS=[(L[:4],'4Large'),(L[:8],'8Large'),(M[:4],'4Mid'),(M[:8],'8Mid'),
        (S[:4],'4Small'),(S[:8],'8Small'),
        (L[:1]+M[:1]+S[:1],'3Mix'),(L[:2]+M[:2]+S[:2],'6Mix'),
        (L[:4]+M[:4]+S[:4],'12Mix'),(L[:6]+M[:6]+S[:6],'18Mix'),(L[:8]+M[:8]+S[:8],'24All')]

# Models & Env & Training (same as exp1_main.py, abbreviated)
def _out_dim(n): return n
def _trim_w(w,n): return w[...,:n]

class CM(nn.Module):
    def __init__(s,pd,td,h=64):
        super().__init__(); s.pe=nn.Sequential(nn.Linear(pd,h),nn.LayerNorm(h),nn.ReLU())
        s.te=nn.Sequential(nn.Linear(td,h),nn.LayerNorm(h),nn.ReLU()); s.sc=nn.Sequential(nn.Linear(h,h),nn.Tanh()); s.fu=nn.Sequential(nn.Linear(h,h),nn.ReLU())
    def forward(s,p,t): pf=s.pe(p); tf=s.te(t); return s.fu(pf*(1+s.sc(tf))+tf)

class SBCA(CM):
    def __init__(s,ws,n): super().__init__(ws*n,n); s.na=n; s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1)); s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,p,t): f=super().forward(p,t); return s.ac(f),s.cr(f)

class SBP(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.net=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,n),nn.Softmax(-1))
    def forward(s,x): return s.net(x)

class SBP_AC(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.bb=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU()); s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1)); s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,x): f=s.bb(x); return s.ac(f),s.cr(f)

class PPO_AC(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.bb=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU()); s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1)); s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,x): f=s.bb(x); return s.ac(f),s.cr(f)

class Env:
    def __init__(s,ca,da=None): s.c=ca; s.d=da; s.na=ca.shape[1]; s.dy=ca.shape[0]; s.pm,s.ps=s._cp()
    def _cp(s):
        rs=[]
        for t in range(W,TE):
            pw=s.c[t-W:t]
            if len(pw)>1: ret=np.diff(pw,axis=0)/(pw[:-1]+1e-8); rs.append(ret.flatten())
        rs=np.concatenate(rs) if rs else np.zeros(1)
        return np.mean(rs),np.std(rs)+1e-8
    def gs(s,t):
        if t<W: pf=np.zeros(W*s.na,dtype=np.float32)
        else:
            pw=s.c[t-W:t]; ret=np.diff(pw,axis=0)/(pw[:-1]+1e-8)
            ret=np.concatenate([np.zeros((1,s.na)),ret],axis=0); pf=((ret.flatten()-s.pm)/s.ps).astype(np.float32)
        tf=None
        if s.d is not None: tf=((s.d[t].copy()-0.5)*2).astype(np.float32)
        return pf,tf

class BMEnv:
    """Lightweight Env for benchmarks (EW/BH/DJ/MinVar/RP), no price normalization needed."""
    def __init__(s,ca): s.c=ca; s.na=ca.shape[1]; s.dy=ca.shape[0]

def ew(env,te,c=COMM):
    pv=[1.0]; w=np.ones(env.na)/env.na
    for t in te:
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]; gr=np.sum(w*y); pv.append(pv[-1]*gr); w=w*y/np.sum(w*y)
        if (t-te.start)%22==0: wn=np.ones(env.na)/env.na; pv[-1]=pv[-1]*(1-c*np.sum(np.abs(wn-w))/2.0); w=wn.copy()
    return np.array(pv)
def bh(env,te,c=COMM):
    pv=[1.0]
    w=np.ones(env.na)/env.na
    pv[-1]=pv[-1]*(1-c*np.sum(w)/2.0)
    for t in te:
        if t+1>=env.dy:
            break
        y=env.c[t+1]/env.c[t]
        gr=np.sum(w*y)
        pv.append(pv[-1]*gr)
        w=w*y/np.sum(w*y)
    return np.array(pv)

def dj(env,te,c=COMM):
    pv=[1.0]
    ip=env.c[te[0]]
    w=ip/np.sum(ip)
    pv[-1]=pv[-1]*(1-c*np.sum(w)/2.0)
    for t in te:
        if t+1>=env.dy:
            break
        y=env.c[t+1]/env.c[t]
        gr=np.sum(w*y)
        pv.append(pv[-1]*gr)
        w=w*y/np.sum(w*y)
    return np.array(pv)
def mv(env,te,wd=60,c=COMM):
    pv=[1.0]; na=env.na; w=np.ones(na)/na
    for ti,t in enumerate(te):
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]; gr=np.sum(w*y); pv.append(pv[-1]*gr); w=w*y/np.sum(w*y)
        if ti>0 and ti%22==0 and t>=wd:
            rets=np.array([env.c[i+1]/env.c[i]-1 for i in range(t-wd,t)])
            try:
                cov=np.cov(rets.T); inv=np.linalg.inv(cov); wn=inv@np.ones(na)/(np.ones(na)@inv@np.ones(na)); wn=np.clip(wn,0,1); wn/=wn.sum()+1e-8
                pv[-1]=pv[-1]*(1-c*np.sum(np.abs(wn-w))/2.0); w=wn
            except: pass
    return np.array(pv)
def rp(env,te,wd=60,c=COMM):
    pv=[1.0]; na=env.na; w=np.ones(na)/na
    for ti,t in enumerate(te):
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]; gr=np.sum(w*y); pv.append(pv[-1]*gr); w=w*y/np.sum(w*y)
        if ti>0 and ti%22==0 and t>=wd:
            rets=np.array([env.c[i+1]/env.c[i]-1 for i in range(t-wd,t)]); vols=np.std(rets,axis=0); inv=1.0/(vols+1e-8); wn=inv/inv.sum()
            pv[-1]=pv[-1]*(1-c*np.sum(np.abs(wn-w))/2.0); w=wn
    return np.array(pv)

def csh(pv):
    ret=np.diff(np.log(pv)); drf=0.02/252
    return (np.mean(ret-drf)/(np.std(ret)+1e-8))*np.sqrt(252)
def evaluate(pv):
    pv=np.array(pv); sr=csh(pv); ret=np.diff(np.log(pv)); d=ret[ret<0]
    so=np.mean(ret)/(np.std(d)+1e-8)*np.sqrt(252) if len(d)>0 else 10
    pk=np.maximum.accumulate(pv); md=np.min((pv-pk)/(pk+1e-8)); ar=pv[-1]**(252/len(pv))-1; ca=ar/(abs(md)+1e-8)
    return dict(PV=round(float(pv[-1]),4),SR=round(sr,4),Sortino=round(so,4),MDD=round(md,4),AR=round(ar,4),Calmar=round(ca,4))

def NA(a): return (a-a.mean())/(a.std()+1e-8) if a.numel()>1 else a
def RA(w,y,wo,c):
    tv=torch.sum(torch.abs(w-wo))/2.0; nr=torch.sum(w*y)-c*tv; lr=torch.log(nr.clamp(min=1e-4))
    return lr,torch.clamp(-lr,min=0.0)**2,tv

def train_sarl(model,env,tr,vr,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5); sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train(); wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy: break
            s=env.gs(t)[0]; pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE); wr=model(pt).squeeze(0)
            w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na)
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE)
            lr_,risk,tv=RA(w,y,wo,COMM); ent=-(w*torch.log(w+1e-8)).sum()
            loss=-lr_+TP*tv+LRISK*risk-ENT*ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            wo=EA*w.detach()+(1-EA)*wo
        vp=1.0; wv=np.ones(env.na)/env.na; model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy: break
                s=env.gs(t)[0]; pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE); wr=model(pt).squeeze(0)
                w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0; y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
                vp=max(vp*(gr-COMM*tv),1e-4); wv=ws
        sch.step(vp)
        if vp>bv: bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else: pc+=1
        if pc>=PAT: break
    model.load_state_dict(bs); return model

def train_ac(model,env,tr,vr,is_cm,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5); sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train(); wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy: break
            pf,tf=env.gs(t)
            if is_cm: pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,v=model(pt,tt)
            else: s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); wr,v=model(s)
            w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na)
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE)
            lr_,risk,tv=RA(w,y,wo,COMM); ent=-(w*torch.log(w+1e-8)).sum()
            pfn,tfn=env.gs(min(t+1,env.dy-1))
            if is_cm: ptn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE); ttn=torch.from_numpy(tfn).unsqueeze(0).to(DEVICE); _,vn=model(ptn,ttn)
            else: sn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE); _,vn=model(sn)
            vn=vn.squeeze(); target=lr_+GAM*vn; adv=NA((target-v.squeeze()).detach())
            loss=-adv*lr_+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-ENT*ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            wo=EA*w.detach()+(1-EA)*wo
        vp=1.0; wv=np.ones(env.na)/env.na; model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy: break
                pf,tf=env.gs(t)
                if is_cm: pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,_=model(pt,tt)
                else: s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); wr,_=model(s)
                w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0; y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
                vp=max(vp*(gr-COMM*tv),1e-4); wv=ws
        sch.step(vp)
        if vp>bv: bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else: pc+=1
        if pc>=PAT: break
    model.load_state_dict(bs); return model

def train_ppo(model,env,tr,vr,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5); sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train(); wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy: break
            s=env.gs(t)[0]; pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE); wr,v=model(pt)
            w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na)
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE)
            lr_,risk,tv=RA(w,y,wo,COMM); ent=-(w*torch.log(w+1e-8)).sum()
            sn=env.gs(min(t+1,env.dy-1))[0]; ptn=torch.from_numpy(sn).unsqueeze(0).to(DEVICE); _,vn=model(ptn); vn=vn.squeeze()
            target=lr_+GAM*vn; adv=(target-v.squeeze()).detach(); adv=adv/(adv.std()+1e-8)
            olp=torch.log(wr.softmax(-1).clamp(min=1e-8)).sum(); nlp=torch.log(w.clamp(min=1e-8)).sum()
            ratio=(nlp-olp.detach()).exp(); clipped=torch.clamp(ratio,0.8,1.2)
            loss=-torch.min(ratio*adv,clipped*adv)+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-0.01*ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            wo=EA*w.detach()+(1-EA)*wo
        vp=1.0; wv=np.ones(env.na)/env.na; model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy: break
                s=env.gs(t)[0]; pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE); wr,_=model(pt)
                w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0; y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
                vp=max(vp*(gr-COMM*tv),1e-4); wv=ws
        sch.step(vp)
        if vp>bv: bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else: pc+=1
        if pc>=PAT: break
    model.load_state_dict(bs); return model

def bt(model,env,te,is_cm):
    pv,turns=[1.0],[]; wo=np.ones(env.na)/env.na; model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy: break
            pf,tf=env.gs(t)
            if is_cm: pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); out=model(pt,tt)
            else: s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); out=model(s)
            wr=out[0].squeeze(0) if isinstance(out,tuple) else out.squeeze(0)
            w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0; turns.append(tv); y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
            pv.append(max(pv[-1]*(gr-COMM*tv),1e-4)); wo=ws
    return np.array(pv), np.mean(turns)*252

# ========== Run ==========
TR=range(W,TE); VR=range(TE,VE); TEST_=range(VE,DAYS_)
out_csv=f"{BASE}/result/exp1_main_G.csv"

models_order = ['EW','BH','DJ','MinVar','RP','PPO','SBCA']
labels = {'EW':'Equal Weight','BH':'Buy & Hold','DJ':'Dow Jones','MinVar':'Min Variance','RP':'Risk Parity','PPO':'PPO','SBCA':'SBCA'}

if not os.path.exists(out_csv):
    rows=[]
    for stks,gn in tqdm(GROUPS,desc="EXP1_G"):
        if len(stks)<2: continue
        na=len(stks); ca=df_c[stks].values; da=df_d[stks].values; env_bm=BMEnv(ca)
        print(f"\n{gn} ({na} assets)")

        set_seed(42); m_sbca=SBCA(W,na).to(DEVICE); env_s=Env(ca,da)
        m_sbca=train_ac(m_sbca,env_s,TR,VR,True,f"SBCA_{gn}")
        pv_s,_=bt(m_sbca,env_s,TEST_,True); ev_s=evaluate(pv_s)
        rows.append([gn,na,'SBCA',ev_s['PV'],ev_s['SR'],ev_s['Sortino'],ev_s['MDD'],ev_s['AR'],ev_s['Calmar']])

        set_seed(42); m_pp=PPO_AC(W*na,na).to(DEVICE); env_pp=Env(ca)
        m_pp=train_ppo(m_pp,env_pp,TR,VR,f"PPO_{gn}")
        pv_pp,_=bt(m_pp,env_pp,TEST_,False); ev_pp=evaluate(pv_pp)
        rows.append([gn,na,'PPO',ev_pp['PV'],ev_pp['SR'],ev_pp['Sortino'],ev_pp['MDD'],ev_pp['AR'],ev_pp['Calmar']])

        for name,fn in [('EW',ew),('BH',bh),('DJ',dj),('MinVar',mv),('RP',rp)]:
            pv_b=fn(env_bm,TEST_); ev_b=evaluate(pv_b)
            rows.append([gn,na,name,ev_b['PV'],ev_b['SR'],ev_b['Sortino'],ev_b['MDD'],ev_b['AR'],ev_b['Calmar']])

        ev_bh=evaluate(bh(env_bm,TEST_)); ev_dj=evaluate(dj(env_bm,TEST_))
        print(f"  SBCA:{ev_s['SR']:.4f} PPO:{ev_pp['SR']:.4f} EW:{evaluate(ew(env_bm,TEST_))['SR']:.4f} BH:{ev_bh['SR']:.4f} DJ:{ev_dj['SR']:.4f}")

    cols=["Group","N","Model","PV","SR","Sortino","MDD","AR","Calmar"]
    pd.DataFrame(rows,columns=cols).to_csv(out_csv,index=False)
    print(f"\nSaved: {out_csv}")

# ========== Generate H table ==========
df_r = pd.read_csv(out_csv)
out_h = f"{BASE}/result/exp1_main_G_H.xlsx"
rows_h = []
for g in ['4Large','8Large','4Mid','8Mid','4Small','8Small','3Mix','6Mix','12Mix','18Mix','24All']:
    sub = df_r[df_r.Group==g]
    n = sub.iloc[0]['N']
    best = {}
    for c in ['PV','AR','SR','Sortino','MDD','Calmar']:
        vals = {m: sub[sub.Model==m][c].values[0] for m in models_order if len(sub[sub.Model==m])>0 and pd.notna(sub[sub.Model==m][c].values[0])}
        if vals:
            if c in ['MDD']: best[c] = max(vals, key=vals.get)
            else: best[c] = max(vals, key=vals.get)
    rows_h.append({'Group': f'{g} (N={n})', 'Model': '', 'PV': '', 'AR': '', 'SR': '', 'Sortino': '', 'MDD': '', 'Calmar': ''})
    for m in models_order:
        r = sub[sub.Model==m]
        if len(r)>0:
            r = r.iloc[0]; row = {'Group': '', 'Model': labels[m]}
            for c in ['PV','AR','SR','Sortino','MDD','Calmar']:
                val = r.get(c, 0) if c in r.index else 0
                row[c] = f'**{val}**' if (c in best and m==best[c]) else val
            rows_h.append(row)
    rows_h.append({'Group': '', 'Model': '---', 'PV': '', 'AR': '', 'SR': '', 'Sortino': '', 'MDD': '', 'Calmar': ''})

pd.DataFrame(rows_h).to_excel(out_h, index=False)
print(f"\nH table saved: {out_h}")

# Summary
print("\n" + "="*60)
print("  EXP1_G Summary")
print("="*60)
for g in ['4Large','8Large','4Mid','8Mid','4Small','8Small','3Mix','6Mix','12Mix','18Mix','24All']:
    sub = df_r[df_r.Group==g]
    srs = {r['Model']: r['SR'] for _,r in sub.iterrows()}
    best_m = max(srs, key=srs.get)
    s_sbca = srs.get('SBCA',0); s_ew = srs.get('EW',0); s_ppo = srs.get('PPO',0)
    print(f'{g:8s} SBCA={s_sbca:.4f} PPO={s_ppo:.4f} EW={s_ew:.4f} best={best_m}')

print("\nDone.")
