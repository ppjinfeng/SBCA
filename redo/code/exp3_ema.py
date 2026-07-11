"""
exp3_ema.py — EMA Price Smoothing Coefficient Grid Search
==================================================================
Tests price_alpha in {0.2, 0.3, 0.4, 0.5, 0.6} across 6 groups.
Fixed: SBCA model, position smoothing ea=0.4, W=30, COMM=0.0025, TP=0.005
Groups: 4Large/8Large/4Mid/8Mid/4Small/8Small
Train=2012-2017, Val=2018-2019, Test=2020-2021
Output: result/exp3_ema.csv
"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
import random, copy, os, sys
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.makedirs(f"{BASE}/result", exist_ok=True)
print(f"Device: {DEVICE}")

W, COMM, GAM = 30, 0.0025, 0.99; LRISK, EA, TP = 0.1, 0.4, 0.005
EP, PAT, LR, ENT = 30, 5, 3e-4, 0.1
PRICE_ALPHAS = [0.2, 0.3, 0.4, 0.5, 0.6]  # EMA price smoothing coefficient (search target)
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

df = pd.read_csv(f"{BASE}/data/bert_pred_24stocks_delta_FinBERT.csv"); df["Date"] = pd.to_datetime(df["Date"])
df_c = df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
df_d = df.pivot_table(index='Date',columns='Stock_symbol',values='delta_bert').shift(1).ffill().fillna(0.5)
dates = df_c.index
TE = dates.get_indexer([pd.to_datetime('2017-12-31')], method='pad')[0]
VE = dates.get_indexer([pd.to_datetime('2019-12-31')], method='pad')[0]
DAYS_ = dates.get_indexer([pd.to_datetime('2021-12-31')], method='pad')[0]  # Consistent with TR/VR split (no +1)

all_s = df_c.columns.tolist()
LARGE=['GILD','COP','EOG','MRK','WFC','ORCL','CMCSA','CAT']
L_sorted='CMCSA GILD MRK ORCL COP WFC CAT EOG'.split()
MID=['WDC','BIIB','KEY','CLX','RRC','DECK','CI','FTI']
M_sorted='BIIB WDC CI RRC CLX KEY DECK FTI'.split()
SMALL=['BGS','WBS','EXLS','CLH','DDS','ALNY','CF','KBH']
S_sorted='CF DDS BGS KBH ALNY WBS EXLS CLH'.split()
SZ={}
for s in LARGE: SZ[s]='large'
for s in MID: SZ[s]='mid'
for s in SMALL: SZ[s]='small'
L=[s for s in all_s if SZ.get(s)=='large']
M=[s for s in all_s if SZ.get(s)=='mid']
S=[s for s in all_s if SZ.get(s)=='small']
KEY=[(L_sorted[:4],"4Large"),(L_sorted[:8],"8Large"),(M_sorted[:4],"4Mid"),(M_sorted[:8],"8Mid"),(S_sorted[:4],"4Small"),(S_sorted[:8],"8Small")]

def _out_dim(n): return n
def _trim_w(w,n): return w[...,:n]
class CM(nn.Module):
    def __init__(s,pd,td,h=128):
        super().__init__(); s.pe=nn.Sequential(nn.Linear(pd,h),nn.LayerNorm(h),nn.ReLU())
        s.te=nn.Sequential(nn.Linear(td,h),nn.LayerNorm(h),nn.ReLU())
        s.gamma=nn.Linear(h,h); s.beta=nn.Linear(h,h); s.fu=nn.Sequential(nn.Linear(h,h),nn.ReLU())
    def forward(s,p,t): pf=s.pe(p); tf=s.te(t); return s.fu(s.gamma(tf)*pf+s.beta(tf))
class SBCA(CM):
    def __init__(s,ws,n): super().__init__(ws*n,n); s.na=n; s.ac=nn.Sequential(nn.Linear(128,n),nn.Softmax(-1)); s.cr=nn.Sequential(nn.Linear(128,1))
    def forward(s,p,t): f=super().forward(p,t); return s.ac(f),s.cr(f)

class Env:
    def __init__(s, ca, da, price_alpha):
        s.c = ca; s.d = da; s.na = ca.shape[1]; s.dy = ca.shape[0]
        s.c_smooth = s._ema(ca, price_alpha)  # EMA smoothed price
        s.pm, s.ps = s._cp()  # Compute normalized statistics from smoothed price
    def _ema(s, arr, alpha):
        out = np.zeros_like(arr)
        for i in range(arr.shape[1]):
            col = arr[:, i]; ema = col[0]; out[0, i] = ema
            for t in range(1, len(col)):
                ema = alpha * col[t] + (1 - alpha) * ema; out[t, i] = ema
        return out
    def _cp(s):
        rs=[]
        for t in range(W, TE):
            pw = s.c_smooth[t-W:t]
            if len(pw)>1: ret=np.diff(pw,axis=0)/(pw[:-1]+1e-8); rs.append(ret.flatten())
        rs=np.concatenate(rs) if rs else np.zeros(1)
        return np.mean(rs), np.std(rs)+1e-8
    def gs(s, t):
        if t<W: pf=np.zeros(W*s.na,dtype=np.float32)
        else:
            pw = s.c_smooth[t-W:t]; ret=np.diff(pw,axis=0)/(pw[:-1]+1e-8)
            ret=np.concatenate([np.zeros((1,s.na)),ret],axis=0)
            pf=((ret.flatten()-s.pm)/s.ps).astype(np.float32)
        tf=((s.d[t].copy()-0.5)*2).astype(np.float32)
        return pf, tf

def calc_sharpe(pv):
    ret=np.diff(np.log(pv)); drf=0.02/252
    return (np.mean(ret-drf)/(np.std(ret)+1e-8))*np.sqrt(252)
def evaluate(pv):
    pv=np.array(pv); sr=calc_sharpe(pv); ret=np.diff(np.log(pv)); d=ret[ret<0]
    so=np.mean(ret)/(np.std(d)+1e-8)*np.sqrt(252) if len(d)>0 else 10
    pk=np.maximum.accumulate(pv); md=np.min((pv-pk)/(pk+1e-8))
    ar=pv[-1]**(252/(len(pv)-1))-1; ca=ar/(abs(md)+1e-8)
    return dict(PV=round(float(pv[-1]),4),SR=round(sr,4),Sortino=round(so,4),MDD=round(md,4),AR=round(ar,4),Calmar=round(ca,4))

def NA(a): return (a-a.mean())/(a.std()+1e-8) if a.numel()>1 else a
def RA(w,y,wo,c):
    tv=torch.sum(torch.abs(w-wo))/2.0; nr=torch.sum(w*y)-c*tv; lr=torch.log(nr.clamp(min=1e-4))
    return lr,torch.clamp(-lr,min=0.0)**2,tv

def train_ac(model,env,tr,vr,ea,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train(); wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy: break
            pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,v=model(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na)
            ws=ea*w+(1-ea)*wo
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE)
            lr_,risk,tv=RA(ws,y,wo,COMM); ent=-(w*torch.log(w+1e-8)).sum()
            pfn,tfn=env.gs(min(t+1,env.dy-1)); ptn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE); ttn=torch.from_numpy(tfn).unsqueeze(0).to(DEVICE); _,vn=model(ptn,ttn); vn=vn.squeeze()
            target=lr_+GAM*vn; adv=NA((target-v.squeeze()).detach())
            loss=-adv*lr_+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-ENT*ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            wo=ws.detach()
        vp=1.0; wv=np.ones(env.na)/env.na; model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy: break
                pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,_=model(pt,tt)
                w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=ea*wn+(1-ea)*wv
                tv=np.sum(np.abs(ws-wv))/2.0; y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y); vp=max(vp*(gr-COMM*tv),1e-4); wv=ws
        sch.step(vp)
        if vp>bv: bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else: pc+=1
        if pc>=PAT: break
    model.load_state_dict(bs); return model

def bt(model,env,te,ea):
    pv,turns=[1.0],[]; wo=np.ones(env.na)/env.na; model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy: break
            pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,_=model(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=ea*wn+(1-ea)*wo
            tv=np.sum(np.abs(ws-wo))/2.0; turns.append(tv); y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
            pv.append(max(pv[-1]*(gr-COMM*tv),1e-4)); wo=ws
    return np.array(pv), np.mean(turns)*252

TR=range(W,TE); VR=range(TE,VE); TEST_=range(VE,DAYS_)
out=f"{BASE}/result/exp3_ema.csv"
if not os.path.exists(out):
    rows=[]
    for stks,gn in KEY:
        if len(stks)<2: continue
        na=len(stks); ca=df_c[stks].values; da=df_d[stks].values
        print(f"\n{gn} ({na} assets)")
        for a in PRICE_ALPHAS:
            env=Env(ca, da, a); set_seed(42); m=SBCA(W,na).to(DEVICE)
            m=train_ac(m, env, TR, VR, EA, f"EMA{a}_{gn}")
            pv,ato=bt(m, env, TEST_, EA); ev=evaluate(pv)
            rows.append([gn,na,a,ev['PV'],ev['SR'],ev['Sortino'],ev['MDD'],ev['AR'],ev['Calmar'],round(ato,4)])
            print(f"  price_alpha={a}: SR={ev['SR']:.4f} ATO={round(ato,4)}")
    pd.DataFrame(rows,columns=["Group","N","PriceAlpha","PV","SR","Sortino","MDD","AR","Calmar","ATO"]).to_csv(out,index=False)
    print(f"\nSaved: {out}")
print("Done.")
