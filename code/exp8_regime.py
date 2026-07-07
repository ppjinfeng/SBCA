"""
exp8_regime.py — Market Regime Decomposition Backtest
==================================================================
Three regimes: COVID Crash / Bull / Bear x SBCA/PPO/EW/BH
Pre-trained models, all test periods after VE=2019-12-31
Groups: 8Large/8Small/12Mix
Output: result/exp8_regime.csv (PV/AR/SR/Sortino/MDD/Calmar)
"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
import random, copy, os, sys
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.makedirs(f"{BASE}/result", exist_ok=True)
print(f"Device: {DEVICE}")

W,COMM,GAM=30,0.0025,0.99;LRISK,EA,TP=0.1,0.4,0.005;EP,PAT,LR,ENT=30,5,3e-4,0.1
def set_seed(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

df=pd.read_csv(f"{BASE}/data/bert_pred_24stocks_delta_G.csv");df["Date"]=pd.to_datetime(df["Date"])
df_c=df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
df_d=df.pivot_table(index='Date',columns='Stock_symbol',values='delta_bert').fillna(0.5)
dates=df_c.index
TE=dates.get_indexer([pd.to_datetime('2017-12-31')],method='pad')[0]
VE=dates.get_indexer([pd.to_datetime('2019-12-31')],method='pad')[0]

REGIMES={
    'COVID_Crash(2020Q1)':(
        dates.get_indexer([pd.to_datetime('2020-01-01')],method='pad')[0],
        dates.get_indexer([pd.to_datetime('2020-03-31')],method='pad')[0]),
    'Bull(2020H2-2021)':(
        dates.get_indexer([pd.to_datetime('2020-07-01')],method='pad')[0],
        dates.get_indexer([pd.to_datetime('2021-12-31')],method='pad')[0]),
    'Bear(2022H1)':(
        dates.get_indexer([pd.to_datetime('2022-01-01')],method='pad')[0],
        dates.get_indexer([pd.to_datetime('2022-06-30')],method='pad')[0]),
}
# Verify: all test periods are after the validation set
for rn,(rs,re) in REGIMES.items():
    assert rs>=VE, f"{rn} starts at {dates[rs].date()} before VE={dates[VE].date()}!"
print("Regime boundary check passed: all regimes after validation period")

all_s=df_c.columns.tolist()
LARGE=['NVDA','ORCL','CRM','QCOM','WFC','MRK','KO','CAT']
SMALL=['SPWR','URBN','ANF','FL','PLUG','KSS','NVAX','ALK']
MID=['BBY','CLX','BIIB','AA','KEY','WDC','AAL','HAL']
SZ={}
for s in LARGE:SZ[s]='large'
for s in MID:SZ[s]='mid'
for s in SMALL:SZ[s]='small'
L=[s for s in all_s if SZ.get(s)=='large']
M=[s for s in all_s if SZ.get(s)=='mid']
S=[s for s in all_s if SZ.get(s)=='small']
KEY=[(L[:8],'8Large'),(S[:8],'8Small'),(L[:4]+M[:4]+S[:4],'12Mix')]

def _out_dim(n):return n
def _trim_w(w,n):return w[...,:n]
class CM(nn.Module):
    def __init__(s,pd,td,h=64):
        super().__init__();s.pe=nn.Sequential(nn.Linear(pd,h),nn.LayerNorm(h),nn.ReLU())
        s.te=nn.Sequential(nn.Linear(td,h),nn.LayerNorm(h),nn.ReLU())
        s.sc=nn.Sequential(nn.Linear(h,h),nn.Tanh());s.fu=nn.Sequential(nn.Linear(h,h),nn.ReLU())
    def forward(s,p,t):pf=s.pe(p);tf=s.te(t);return s.fu(pf*(1+s.sc(tf))+tf)
class SBCA(CM):
    def __init__(s,ws,n):super().__init__(ws*n,n);s.na=n;s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1));s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,p,t):f=super().forward(p,t);return s.ac(f),s.cr(f)
class PPO_AC(nn.Module):
    def __init__(s,sd,n):super().__init__();s.na=n;s.bb=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU());s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1));s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,x):f=s.bb(x);return s.ac(f),s.cr(f)

class Env:
    def __init__(s,ca,da):s.c=ca;s.d=da;s.na=ca.shape[1];s.dy=ca.shape[0];s.pm,s.ps=s._cp()
    def _cp(s):
        rs=[]
        for t in range(W,TE):
            pw=s.c[t-W:t]
            if len(pw)>1:ret=np.diff(pw,axis=0)/(pw[:-1]+1e-8);rs.append(ret.flatten())
        rs=np.concatenate(rs) if rs else np.zeros(1);return np.mean(rs),np.std(rs)+1e-8
    def gs(s,t):
        if t<W:pf=np.zeros(W*s.na,dtype=np.float32)
        else:
            pw=s.c[t-W:t];ret=np.diff(pw,axis=0)/(pw[:-1]+1e-8)
            ret=np.concatenate([np.zeros((1,s.na)),ret],axis=0);pf=((ret.flatten()-s.pm)/s.ps).astype(np.float32)
        tf=None
        if s.d is not None:tf=((s.d[t].copy()-0.5)*2).astype(np.float32)
        return pf,tf

class BMEnv:
    def __init__(s,ca):s.c=ca;s.na=ca.shape[1];s.dy=ca.shape[0]

def calc_sharpe(pv):
    ret=np.diff(np.log(pv));drf=0.02/252;return (np.mean(ret-drf)/(np.std(ret)+1e-8))*np.sqrt(252)
def evaluate(pv):
    pv=np.array(pv);sr=calc_sharpe(pv);ret=np.diff(np.log(pv));d=ret[ret<0]
    so=np.mean(ret)/(np.std(d)+1e-8)*np.sqrt(252) if len(d)>0 else 10
    pk=np.maximum.accumulate(pv);md=np.min((pv-pk)/(pk+1e-8));ar=pv[-1]**(252/len(pv))-1;ca=ar/(abs(md)+1e-8)
    return dict(PV=round(float(pv[-1]),4),AR=round(ar,4),SR=round(sr,4),Sortino=round(so,4),MDD=round(md,4),Calmar=round(ca,4))

def NA(a):return (a-a.mean())/(a.std()+1e-8) if a.numel()>1 else a
def RA(w,y,wo,c):
    tv=torch.sum(torch.abs(w-wo))/2.0;nr=torch.sum(w*y)-c*tv;lr=torch.log(nr.clamp(min=1e-4));return lr,torch.clamp(-lr,min=0.0)**2,tv

def train_ac(model,env,tr,vr,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5);sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train();wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy:break
            pf,tf=env.gs(t);pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE);tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE);wr,v=model(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na)
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE);lr_,risk,tv=RA(w,y,wo,COMM);ent=-(w*torch.log(w+1e-8)).sum()
            pfn,tfn=env.gs(min(t+1,env.dy-1));ptn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE);ttn=torch.from_numpy(tfn).unsqueeze(0).to(DEVICE);_,vn=model(ptn,ttn);vn=vn.squeeze()
            target=lr_+GAM*vn;adv=NA((target-v.squeeze()).detach())
            loss=-adv*lr_+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-ENT*ent
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),0.5);opt.step();wo=EA*w.detach()+(1-EA)*wo
        vp=1.0;wv=np.ones(env.na)/env.na;model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy:break
                pf,tf=env.gs(t);pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE);tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE);wr,_=model(pt,tt)
                w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0;y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);vp=max(vp*(gr-COMM*tv),1e-4);wv=ws
        sch.step(vp)
        if vp>bv:bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else:pc+=1
        if pc>=PAT:break
    model.load_state_dict(bs);return model

def train_ppo(model,env,tr,vr,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5);sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train();wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy:break
            s=env.gs(t)[0];pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE);wr,v=model(pt)
            w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na)
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE);lr_,risk,tv=RA(w,y,wo,COMM);ent=-(w*torch.log(w+1e-8)).sum()
            sn=env.gs(min(t+1,env.dy-1))[0];ptn=torch.from_numpy(sn).unsqueeze(0).to(DEVICE);_,vn=model(ptn);vn=vn.squeeze()
            target=lr_+GAM*vn;adv=(target-v.squeeze()).detach();adv=adv/(adv.std()+1e-8)
            olp=torch.log(wr.softmax(-1).clamp(min=1e-8)).sum();nlp=torch.log(w.clamp(min=1e-8)).sum()
            ratio=(nlp-olp.detach()).exp();clipped=torch.clamp(ratio,0.8,1.2)
            loss=-torch.min(ratio*adv,clipped*adv)+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-0.01*ent
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),0.5);opt.step();wo=EA*w.detach()+(1-EA)*wo
        vp=1.0;wv=np.ones(env.na)/env.na;model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy:break
                s=env.gs(t)[0];pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE);wr,_=model(pt)
                w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0;y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);vp=max(vp*(gr-COMM*tv),1e-4);wv=ws
        sch.step(vp)
        if vp>bv:bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else:pc+=1
        if pc>=PAT:break
    model.load_state_dict(bs);return model

def bt_cm(model,env,te):
    pv=[1.0];wo=np.ones(env.na)/env.na;model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy:break
            pf,tf=env.gs(t);pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE);tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE);wr,_=model(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0;y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);pv.append(max(pv[-1]*(gr-COMM*tv),1e-4));wo=ws
    return np.array(pv)

def bt_pp(model,env,te):
    pv=[1.0];wo=np.ones(env.na)/env.na;model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy:break
            s=env.gs(t)[0];pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE);wr,_=model(pt)
            w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0;y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);pv.append(max(pv[-1]*(gr-COMM*tv),1e-4));wo=ws
    return np.array(pv)

def ew(env,te):
    pv=[1.0];w=np.ones(env.na)/env.na
    for t in te:
        if t+1>=env.dy:break
        y=env.c[t+1]/env.c[t];gr=np.sum(w*y);pv.append(pv[-1]*gr);w=w*y/np.sum(w*y)
        if (t-te.start)%22==0:
            wn=np.ones(env.na)/env.na;pv[-1]=pv[-1]*(1-COMM*np.sum(np.abs(wn-w))/2.0);w=wn.copy()
    return np.array(pv)

def bh(env,te):
    pv=[1.0]
    w=np.ones(env.na)/env.na
    pv[-1]=pv[-1]*(1-COMM*np.sum(w)/2.0)
    for t in te:
        if t+1>=env.dy:
            break
        y=env.c[t+1]/env.c[t]
        gr=np.sum(w*y)
        pv.append(pv[-1]*gr)
        w=w*y/np.sum(w*y)
    return np.array(pv)

TR=range(W,TE);VR=range(TE,VE)
out=f"{BASE}/result/exp8_regime.csv"
if not os.path.exists(out):
    rows=[]
    for stks,gn in KEY:
        na=len(stks);ca=df_c[stks].values;da=df_d[stks].values
        env_rl=Env(ca,da);env_bm=BMEnv(ca)
        set_seed(42);ms=SBCA(W,na).to(DEVICE);ms=train_ac(ms,env_rl,TR,VR,f"SBCA_{gn}")
        set_seed(123);mp=PPO_AC(W*na,na).to(DEVICE);mp=train_ppo(mp,env_rl,TR,VR,f"PPO_{gn}")
        for rn,(rs,re) in REGIMES.items():
            te_r=range(rs,min(re,len(dates)-1))
            pv_s=bt_cm(ms,env_rl,te_r);ev_s=evaluate(pv_s)
            pv_p=bt_pp(mp,env_rl,te_r);ev_p=evaluate(pv_p)
            pv_e=ew(env_bm,te_r);ev_e=evaluate(pv_e)
            pv_b=bh(env_bm,te_r);ev_b=evaluate(pv_b)
            for name,ev in [('SBCA',ev_s),('PPO',ev_p),('EW',ev_e),('BH',ev_b)]:
                rows.append([gn,rn,name,ev['PV'],ev['AR'],ev['SR'],ev['Sortino'],ev['MDD'],ev['Calmar']])
            print(f"{gn} {rn}: SBCA={ev_s['SR']:.4f} PPO={ev_p['SR']:.4f} EW={ev_e['SR']:.4f} BH={ev_b['SR']:.4f}")
    pd.DataFrame(rows,columns=["Group","Regime","Model","PV","AR","SR","Sortino","MDD","Calmar"]).to_csv(out,index=False)
print("Done.")
