"""
exp1_ato.py — Compute ATO for exp1 main results (SBCA/PPO retrain + baselines)
Merges ATO into existing exp1_main_G.csv
"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
import random, copy, os, sys
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Device: {DEVICE}")

W, COMM, GAM = 30, 0.0025, 0.99; LRISK, EA, TP = 0.1, 0.4, 0.005
EP, PAT, LR, ENT, HID = 30, 5, 3e-4, 0.1, 128

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ── Data loading (same as exp1) ──
df = pd.read_csv(f"{BASE}/data/bert_pred_24stocks_delta_FinBERT.csv"); df['Date'] = pd.to_datetime(df['Date'])
df_c = df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
df_d = df.pivot_table(index='Date',columns='Stock_symbol',values='delta_bert').shift(1).ffill().fillna(0.5)
dates = df_c.index
TE = dates.get_indexer([pd.to_datetime('2017-12-31')], method='pad')[0]
VE = dates.get_indexer([pd.to_datetime('2019-12-31')], method='pad')[0]
DAYS_ = dates.get_indexer([pd.to_datetime('2021-12-31')], method='pad')[0]

L_sorted = "CMCSA GILD MRK ORCL COP WFC CAT EOG".split()
M_sorted = "BIIB WDC CI RRC CLX KEY DECK FTI".split()
S_sorted = "CF DDS BGS KBH ALNY WBS EXLS CLH".split()
GROUPS = [(L_sorted[:4],"4Large"),(L_sorted[:8],"8Large"),(M_sorted[:4],"4Mid"),(M_sorted[:8],"8Mid"),
          (S_sorted[:4],"4Small"),(S_sorted[:8],"8Small"),
          (L_sorted[:1]+M_sorted[:1]+S_sorted[:1],"3Mix"),(L_sorted[:2]+M_sorted[:2]+S_sorted[:2],"6Mix"),
          (L_sorted[:4]+M_sorted[:4]+S_sorted[:4],"12Mix"),(L_sorted[:6]+M_sorted[:6]+S_sorted[:6],"18Mix"),
          (L_sorted[:8]+M_sorted[:8]+S_sorted[:8],"24All")]

# ── Model definitions (same as exp1) ──
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

class PPO_AC(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.bb=nn.Sequential(nn.Linear(sd,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()); s.ac=nn.Sequential(nn.Linear(128,n),nn.Softmax(-1)); s.cr=nn.Sequential(nn.Linear(128,1))
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
    def __init__(s,ca): s.c=ca; s.na=ca.shape[1]; s.dy=ca.shape[0]

def NA(a): return (a-a.mean())/(a.std()+1e-8) if a.numel()>1 else a
def RA(w,y,wo,c):
    tv=torch.sum(torch.abs(w-wo))/2.0; nr=torch.sum(w*y)-c*tv; lr=torch.log(nr.clamp(min=1e-4))
    return lr,torch.clamp(-lr,min=0.0)**2,tv

# ── Training functions (same as exp1) ──
def train_ac(model,env,tr,vr,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5); sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train(); wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy: break
            pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,v=model(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); ws=EA*w+(1-EA)*wo
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE); lr_,risk,tv=RA(ws,y,wo,COMM); ent=-(w*torch.log(w+1e-8)).sum()
            pfn,tfn=env.gs(min(t+1,env.dy-1)); ptn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE); ttn=torch.from_numpy(tfn).unsqueeze(0).to(DEVICE); _,vn=model(ptn,ttn)
            vn=vn.squeeze(); target=lr_+GAM*vn; adv=NA((target-v.squeeze()).detach())
            loss=-adv*lr_+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-ENT*ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step(); wo=ws.detach()
        vp=1.0; wv=np.ones(env.na)/env.na; model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy: break
                pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,_=model(pt,tt)
                w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0; y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y); vp=max(vp*(gr-COMM*tv),1e-4); wv=ws
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
            w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); ws=EA*w+(1-EA)*wo
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE); lr_,risk,tv=RA(ws,y,wo,COMM); ent=-(w*torch.log(w+1e-8)).sum()
            sn=env.gs(min(t+1,env.dy-1))[0]; ptn=torch.from_numpy(sn).unsqueeze(0).to(DEVICE); _,vn=model(ptn); vn=vn.squeeze()
            target=lr_+GAM*vn; adv=(target-v.squeeze()).detach(); adv=adv/(adv.std()+1e-8)
            olp=torch.log(wr.softmax(-1).clamp(min=1e-8)).sum(); nlp=torch.log(w.clamp(min=1e-8)).sum()
            ratio=(nlp-olp.detach()).exp(); clipped=torch.clamp(ratio,0.8,1.2)
            loss=-torch.min(ratio*adv,clipped*adv)+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-0.01*ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step(); wo=ws.detach()
        vp=1.0; wv=np.ones(env.na)/env.na; model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy: break
                s=env.gs(t)[0]; pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE); wr,_=model(pt)
                w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0; y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y); vp=max(vp*(gr-COMM*tv),1e-4); wv=ws
        sch.step(vp)
        if vp>bv: bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else: pc+=1
        if pc>=PAT: break
    model.load_state_dict(bs); return model

# ── ATO-only backtest ──
def bt_ato(model,env,te,is_cm):
    """Backtest, return annualized ATO only"""
    wo = np.ones(env.na)/env.na; turns = []; model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy: break
            pf,tf = env.gs(t)
            if is_cm:
                pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE)
                wr = model(pt,tt)[0].squeeze(0)
            else:
                s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE)
                wr = model(s)[0].squeeze(0)
            w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy()
            ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0; turns.append(tv); wo=ws
    return np.mean(turns)*252

# ── ATO for traditional baselines ──
def ew_ato(env,te,c=COMM):
    """EW: monthly rebalance, return annualized ATO"""
    turns = []; w=np.ones(env.na)/env.na
    for ti,t in enumerate(te):
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]; w=w*y/np.sum(w*y)
        if (t-te.start)%22==0:
            wn=np.ones(env.na)/env.na
            turns.append(np.sum(np.abs(wn-w))/2.0); w=wn.copy()
    return np.mean(turns)*252 if turns else 0

def bh_ato(env,te,c=COMM):
    """BH: initial purchase only, no rebalancing → ATO≈0"""
    return 0.0

def dj_ato(env,te,c=COMM):
    """DJ: price-weighted initial, then BH → ATO≈0"""
    return 0.0

def mv_ato(env,te,wd=60,c=COMM):
    """MinVar: monthly rebalance, return annualized ATO"""
    turns = []; na=env.na; w=np.ones(na)/na
    for ti,t in enumerate(te):
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]; w=w*y/np.sum(w*y)
        if ti>0 and ti%22==0 and t>=wd:
            rets=np.array([env.c[i+1]/env.c[i]-1 for i in range(t-wd,t)])
            try:
                cov=np.cov(rets.T); inv=np.linalg.inv(cov); wn=inv@np.ones(na)/(np.ones(na)@inv@np.ones(na))
                wn=np.clip(wn,0,1); wn/=wn.sum()+1e-8
                turns.append(np.sum(np.abs(wn-w))/2.0); w=wn
            except: pass
    return np.mean(turns)*252 if turns else 0

def rp_ato(env,te,wd=60,c=COMM):
    """RP: monthly rebalance, return annualized ATO"""
    turns = []; na=env.na; w=np.ones(na)/na
    for ti,t in enumerate(te):
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]; w=w*y/np.sum(w*y)
        if ti>0 and ti%22==0 and t>=wd:
            rets=np.array([env.c[i+1]/env.c[i]-1 for i in range(t-wd,t)])
            vols=np.std(rets,axis=0); inv=1.0/(vols+1e-8); wn=inv/inv.sum()
            turns.append(np.sum(np.abs(wn-w))/2.0); w=wn
    return np.mean(turns)*252 if turns else 0

# ── Main ──
TR=range(W,TE); VR=range(TE,VE); TEST_=range(VE,DAYS_)
out_csv = f"{BASE}/result/exp1_ato.csv"

print("\nComputing ATO for all models...\n")

rows = []
for stks,gn in tqdm(GROUPS, desc="Groups"):
    if len(stks)<2: continue
    na=len(stks); ca=df_c[stks].values; da=df_d[stks].values

    # SBCA (retrain + backtest for ATO)
    set_seed(42); env_s = Env(ca,da); m_sbca = SBCA(W,na).to(DEVICE)
    m_sbca = train_ac(m_sbca, env_s, TR, VR, f"SBCA_{gn}")
    ato_s = bt_ato(m_sbca, env_s, TEST_, True)
    rows.append([gn, 'SBCA', round(ato_s,4)])
    print(f"  {gn:10s} SBCA ATO={ato_s:.4f}")

    # PPO (retrain + backtest for ATO)
    set_seed(42); env_pp = Env(ca); m_pp = PPO_AC(W*na,na).to(DEVICE)
    m_pp = train_ppo(m_pp, env_pp, TR, VR, f"PPO_{gn}")
    ato_p = bt_ato(m_pp, env_pp, TEST_, False)
    rows.append([gn, 'PPO', round(ato_p,4)])
    print(f"  {gn:10s} PPO  ATO={ato_p:.4f}")

    # Analytical baselines
    env_bm = BMEnv(ca)
    ato_ew = ew_ato(env_bm, TEST_); rows.append([gn, 'EW', round(ato_ew,4)])
    ato_bh = bh_ato(env_bm, TEST_); rows.append([gn, 'BH', round(ato_bh,4)])
    ato_dj = dj_ato(env_bm, TEST_); rows.append([gn, 'DJ', round(ato_dj,4)])
    ato_mv = mv_ato(env_bm, TEST_); rows.append([gn, 'MinVar', round(ato_mv,4)])
    ato_rp = rp_ato(env_bm, TEST_); rows.append([gn, 'RP', round(ato_rp,4)])

# Save ATO table
df_ato = pd.DataFrame(rows, columns=["Group","Model","ATO"])
df_ato.to_csv(out_csv, index=False)
print(f"\nATO saved: {out_csv}")

# ── Merge with exp1 CSV ──
df_exp = pd.read_csv(f"{BASE}/result/exp1_main_G.csv")
df_merged = df_exp.merge(df_ato, on=["Group","Model"], how="left")

# Reorder columns to put ATO at the end
cols = ["Group","N","Model","PV","SR","Sortino","MDD","AR","Calmar","ATO"]
df_merged = df_merged[cols]

merged_csv = f"{BASE}/result/exp1_main_G.csv"
df_merged.to_csv(merged_csv, index=False)
print(f"Merged CSV saved: {merged_csv}")

# ── Summary ──
print("\n=== ATO SUMMARY ===")
print(f"{'Model':>8s} {'Avg ATO':>10s} {'Min':>10s} {'Max':>10s}")
for m in ['SBCA','PPO','EW','BH','DJ','MinVar','RP']:
    sub = df_ato[df_ato.Model==m]['ATO']
    print(f"{m:>8s} {sub.mean():10.4f} {sub.min():10.4f} {sub.max():10.4f}")

# SBCA vs PPO ATO comparison
print("\n=== SBCA vs PPO ATO ===")
for _,r in df_ato[df_ato.Model.isin(['SBCA','PPO'])].pivot_table('ATO','Group','Model').iterrows():
    gap = r['SBCA'] - r['PPO']
    flag = "← lower" if gap < 0 else "← higher"
    print(f"  {r.name:10s} SBCA={r['SBCA']:.4f} PPO={r['PPO']:.4f} Δ={gap:+.4f} {flag if abs(gap)>0.001 else ''}")

print("\nDone.")
