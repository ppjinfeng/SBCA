"""
ato_baselines.py — Compute ATO for baselines + PPO, reusing exact main-script logic.
"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
import random, copy, os, sys
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{BASE}/result", exist_ok=True)
print(f"Device: {DEVICE}")

W, COMM, GAM = 30, 0.0025, 0.99
LRISK, EA, TP = 0.1, 0.4, 0.005
EP, PAT, LR, ENT = 30, 5, 3e-4, 0.1

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

TR_START, TR_END = '2012-01-01', '2017-12-31'
VA_END = '2019-12-31'; TE_END = '2021-12-31'

LARGE=['NVDA','ORCL','CRM','QCOM','WFC','MRK','KO','CAT']
MID=['BBY','CLX','BIIB','AA','KEY','WDC','AAL','HAL']
SMALL=['SPWR','URBN','ANF','FL','PLUG','KSS','NVAX','ALK']
ALL_24 = LARGE + MID + SMALL

GROUPS=[('4Large',LARGE[:4]),('8Large',LARGE),('4Mid',MID[:4]),('8Mid',MID),
        ('4Small',SMALL[:4]),('8Small',SMALL),('3Mix',[LARGE[0],MID[0],SMALL[0]]),
        ('6Mix',LARGE[:2]+MID[:2]+SMALL[:2]),('12Mix',LARGE[:4]+MID[:4]+SMALL[:4]),
        ('18Mix',LARGE[:6]+MID[:6]+SMALL[:6]),('24All',ALL_24)]

# ========== Load Data ==========
delta_csv = f"{BASE}/data/bert_pred_24stocks_delta_G.csv"
df = pd.read_csv(delta_csv, parse_dates=['Date'])
df_c = df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
df_d = df.pivot_table(index='Date',columns='Stock_symbol',values='delta_bert').fillna(0.5)

dates = df_c.index
TE = dates.get_indexer([pd.to_datetime(TR_END)], method='pad')[0]
VE = dates.get_indexer([pd.to_datetime(VA_END)], method='pad')[0]
TEST_END = dates.get_indexer([pd.to_datetime(TE_END)], method='pad')[0]
DAYS_ = TEST_END
TR=range(W,TE); VR=range(TE,VE); TEST_=range(VE,DAYS_)
print(f"Train: {dates[0].date()}~{dates[TE].date()}, Val: ~{dates[VE].date()}, Test: ~{dates[DAYS_].date()}")

# ========== EXACT copies from exp1_main_G.py ==========
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
        tf=None;
        if s.d is not None: tf=((s.d[t].copy()-0.5)*2).astype(np.float32)
        return pf,tf

class BMEnv:
    def __init__(s,ca): s.c=ca; s.na=ca.shape[1]; s.dy=ca.shape[0]

def RA(w,y,wo,c):
    tv=torch.sum(torch.abs(w-wo))/2.0; nr=torch.sum(w*y)-c*tv; lr=torch.log(nr.clamp(min=1e-4))
    return lr,torch.clamp(-lr,min=0.0)**2,tv

def _trim_w(w,n): return w[...,:n]
def _out_dim(n): return n

class PPO_AC(nn.Module):
    def __init__(s,in_dim,na):
        super().__init__(); s.na=na
        s.fc=nn.Sequential(nn.Linear(in_dim,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        s.actor=nn.Linear(64,na); s.critic=nn.Linear(64,1)
    def forward(s,x): h=s.fc(x); return s.actor(h),s.critic(h).squeeze(-1)

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

# ========== Baseline ATO (with turnover tracking) ==========
def ew_bt(env,te):
    pv,turns=[1.0],[]; w=np.ones(env.na)/env.na
    for t in te:
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]
        if (t-te.start)%22==0:
            wn=np.ones(env.na)/env.na; tv=np.sum(np.abs(wn-w))/2.0; turns.append(tv)
            pv.append(pv[-1]*(np.sum(w*y)-COMM*tv)); w=wn
        else: gr=np.sum(w*y); pv.append(pv[-1]*gr); w=w*y/np.sum(w*y)
    return np.array(pv), np.mean(turns)*252

def bh_bt(env,te):
    pv=[1.0]; w=np.ones(env.na)/env.na
    for t in te:
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]; pv.append(pv[-1]*np.sum(w*y)); w=w*y/np.sum(w*y)
    return np.array(pv), 0.0

def dj_bt(env,te):
    pv,turns=[1.0],[]; w=np.ones(env.na)/env.na
    for t in te:
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]; wn=env.c[t]/np.sum(env.c[t])
        tv=np.sum(np.abs(wn-w))/2.0; turns.append(tv)
        pv.append(max(pv[-1]*(np.sum(wn*y)-COMM*tv),1e-4)); w=wn
    return np.array(pv), np.mean(turns)*252

def mv_bt(env,te,wd=60):
    pv,turns=[1.0],[]; w=np.ones(env.na)/env.na
    for t in te:
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]
        if (t-te.start)%22==0 and t>=wd:
            rets=np.log(env.c[t-wd:t]/env.c[t-wd-1:t-1]); cov=np.cov(rets.T)
            inv_cov=np.linalg.pinv(cov); ones=np.ones(env.na)
            wn=inv_cov@ones/(ones@inv_cov@ones); wn=np.clip(wn,0,1); wn=wn/wn.sum()
        else: wn=w*y/np.sum(w*y)
        if (t-te.start)%22==0: tv=np.sum(np.abs(wn-w))/2.0; turns.append(tv); pv.append(max(pv[-1]*(np.sum(w*y)-COMM*tv),1e-4))
        else: pv.append(pv[-1]*np.sum(wn*y))
        w=wn
    return np.array(pv), np.mean(turns)*252

def rp_bt(env,te,wd=60):
    pv,turns=[1.0],[]; w=np.ones(env.na)/env.na
    for t in te:
        if t+1>=env.dy: break
        y=env.c[t+1]/env.c[t]
        if (t-te.start)%22==0 and t>=wd:
            rets=np.log(env.c[t-wd:t]/env.c[t-wd-1:t-1]); vols=np.std(rets,axis=0)
            wn=(1.0/(vols+1e-8)); wn=wn/wn.sum()
        else: wn=w*y/np.sum(w*y)
        if (t-te.start)%22==0: tv=np.sum(np.abs(wn-w))/2.0; turns.append(tv); pv.append(max(pv[-1]*(np.sum(w*y)-COMM*tv),1e-4))
        else: pv.append(pv[-1]*np.sum(wn*y))
        w=wn
    return np.array(pv), np.mean(turns)*252

# ========== Run ==========
out_csv = f"{BASE}/result/ato_all.csv"
rows = []

for gn,stks in tqdm(GROUPS, desc="ATO"):
    if len(stks)<2: continue
    na=len(stks); ca=df_c[stks].values; env_bm=BMEnv(ca)
    # Baselines
    for name,fn in [('EW',ew_bt),('BH',bh_bt),('DJ',dj_bt),('MinVar',mv_bt),('RP',rp_bt)]:
        _,ato=fn(env_bm,TEST_); rows.append([gn,na,name,round(ato,4)])
    # PPO (train + backtest)
    set_seed(42); env_pp=Env(ca); m_pp=PPO_AC(W*na,na).to(DEVICE)
    m_pp=train_ppo(m_pp,env_pp,TR,VR,f"PPO_{gn}")
    _,ato_pp=bt(m_pp,env_pp,TEST_,False)
    rows.append([gn,na,'PPO',round(ato_pp,4)])
    print(f"  {gn}: PPO ATO={ato_pp:.4f}")

# Merge SBCA ATO from ablation
abl = pd.read_csv(f"{BASE}/result/exp2_ablation.csv")
sba = abl[abl.Model=='SBCA'][['Group','ATO']].copy()
sba['N']=sba['Group'].map({gn:len(stks) for gn,stks in GROUPS})
sba['Model']='SBCA'
for _,r in sba.iterrows(): rows.append([r['Group'],r['N'],'SBCA',round(r['ATO'],4)])

df_out = pd.DataFrame(rows, columns=["Group","N","Model","ATO"])
df_out.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print(df_out.pivot(index='Group',columns='Model',values='ATO').round(4).to_string())
