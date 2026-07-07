"""Step 1: Train all models and save PV arrays for 8Small and 24All"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
import random, copy, os; from tqdm import tqdm; import warnings; warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"); SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pv_data")
os.makedirs(SAVE_DIR, exist_ok=True)
W, COMM, GAM = 30, 0.0025, 0.99; LRISK, EA, TP = 0.1, 0.4, 0.005; EP, PAT, LR, ENT = 30, 5, 3e-4, 0.1
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

df = pd.read_csv(f"{DATA_DIR}/bert_pred_24stocks_delta_G.csv"); df["Date"] = pd.to_datetime(df["Date"])
df_c = df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
df_d = df.pivot_table(index='Date',columns='Stock_symbol',values='delta_bert').fillna(0.5)
dates = df_c.index
TE = dates.get_indexer([pd.to_datetime('2017-12-31')],method='pad')[0]
VE = dates.get_indexer([pd.to_datetime('2019-12-31')],method='pad')[0]
DAYS_ = dates.get_indexer([pd.to_datetime('2021-12-31')],method='pad')[0]

SMALL8 = ['SPWR','URBN','ANF','FL','PLUG','KSS','NVAX','ALK']
ALL24 = ['NVDA','ORCL','CRM','QCOM','WFC','MRK','KO','CAT','BBY','CLX','BIIB','AA','KEY','WDC','AAL','HAL']+SMALL8

def _out_dim(n): return n
def _trim_w(w,n): return w[...,:n]

class CM(nn.Module):
    def __init__(s,pd,td,h=64):
        super().__init__(); s.pe=nn.Sequential(nn.Linear(pd,h),nn.LayerNorm(h),nn.ReLU())
        s.te=nn.Sequential(nn.Linear(td,h),nn.LayerNorm(h),nn.ReLU())
        s.sc=nn.Sequential(nn.Linear(h,h),nn.Tanh()); s.fu=nn.Sequential(nn.Linear(h,h),nn.ReLU())
    def forward(s,p,t): pf=s.pe(p); tf=s.te(t); return s.fu(pf*(1+s.sc(tf))+tf)

class SB_P(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.net=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,n),nn.Softmax(-1))
    def forward(s,x): return s.net(x), None

class S_BERT(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.net=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,n),nn.Softmax(-1))
    def forward(s,x): return s.net(x), None

class SBA_AC(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.bb=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU()); s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1)); s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,x): f=s.bb(x); return s.ac(f),s.cr(f)

class SBC_CM(CM):
    def __init__(s,ws,n): super().__init__(ws*n,n); s.na=n; s.hd=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1))
    def forward(s,p,t): return s.hd(super().forward(p,t)), None

class SBCA(CM):
    def __init__(s,ws,n): super().__init__(ws*n,n); s.na=n; s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1)); s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,p,t): f=super().forward(p,t); return s.ac(f),s.cr(f)

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
            ret=np.concatenate([np.zeros((1,s.na)),ret],axis=0)
            pf=((ret.flatten()-s.pm)/s.ps).astype(np.float32)
        tf=None
        if s.d is not None: tf=((s.d[t].copy()-0.5)*2).astype(np.float32)
        return pf,tf

def NA(a): return (a-a.mean())/(a.std()+1e-8) if a.numel()>1 else a
def RA(w,y,wo,c):
    tv=torch.sum(torch.abs(w-wo))/2.0; nr=torch.sum(w*y)-c*tv
    lr_=torch.log(nr.clamp(min=1e-4)); return lr_,torch.clamp(-lr_,min=0.0)**2,tv

TR=range(W,TE); VR=range(TE,VE); TEST_R=range(VE,DAYS_)

MODELS = [
    ('SB-P',   lambda n: SB_P(W*n, n), False, False),
    ('S-BERT', lambda n: S_BERT(W*n+n, n), False, True),
    ('SBA',    lambda n: SBA_AC(W*n+n, n), False, True),
    ('SBC',    lambda n: SBC_CM(W, n), True, True),
    ('SBCA',   lambda n: SBCA(W, n), True, True),
    ('PPO',    lambda n: PPO_AC(W*n, n), False, False),
]

def train_and_bt(stocks, name):
    ca=df_c[stocks].values; da=df_d[stocks].values; n=len(stocks); results={}
    for mname, factory, is_cm, has_text in MODELS:
        fname = f"{SAVE_DIR}/pv_{name}_{mname}.npy"
        if os.path.exists(fname):
            results[mname] = np.load(fname)
            print(f"  {name} {mname}: loaded")
            continue
        print(f"  {name} {mname}: training...")
        env=Env(ca, da if has_text else None); set_seed(42); m=factory(n).to(DEVICE)
        opt=optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5)
        sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
        bv,pc,bs=-np.inf,0,copy.deepcopy(m.state_dict())
        for ep in range(EP):
            m.train(); wo=torch.ones(n).to(DEVICE)/n
            for t in tqdm(TR,desc=f"{mname} E{ep+1}",leave=False):
                if t+1>=env.dy: break
                pf,tf=env.gs(t)
                if is_cm:
                    pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,v=m(pt,tt)
                elif has_text:
                    s=torch.from_numpy(np.concatenate([pf,tf])).unsqueeze(0).to(DEVICE); wr,v=m(s)
                else:
                    s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); wr,v=m(s)
                w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,n)
                y=torch.from_numpy(ca[t+1]/ca[t]).float().to(DEVICE)
                lr_,risk,tv=RA(w,y,wo,COMM); ent=-(w*torch.log(w+1e-8)).sum()
                if v is not None:
                    pfn,tfn=env.gs(min(t+1,env.dy-1))
                    if is_cm:
                        ptn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE); ttn=torch.from_numpy(tfn).unsqueeze(0).to(DEVICE); _,vn=m(ptn,ttn)
                    elif has_text:
                        sn=torch.from_numpy(np.concatenate([pfn,tfn])).unsqueeze(0).to(DEVICE); _,vn=m(sn)
                    else:
                        sn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE); _,vn=m(sn)
                    vn=vn.squeeze(); target=lr_+GAM*vn; adv=NA((target-v.squeeze()).detach())
                    loss=-adv*lr_+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-ENT*ent
                else:
                    loss=-lr_+TP*tv+LRISK*risk-ENT*ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),0.5); opt.step()
                wo=EA*w.detach()+(1-EA)*wo
            m.eval(); vp=1.0; wv=np.ones(n)/n
            with torch.no_grad():
                for t in VR:
                    if t+1>=env.dy: break
                    pf,tf=env.gs(t)
                    if is_cm:
                        pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,_=m(pt,tt)
                    elif has_text:
                        s=torch.from_numpy(np.concatenate([pf,tf])).unsqueeze(0).to(DEVICE); wr,_=m(s)
                    else:
                        s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); wr,_=m(s)
                    w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,n); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wv
                    tv=np.sum(np.abs(ws-wv))/2.0; y=ca[t+1]/ca[t]; gr=np.sum(ws*y); vp=max(vp*(gr-COMM*tv),1e-4); wv=ws
            sch.step(vp)
            if vp>bv: bv,bs,pc=vp,copy.deepcopy(m.state_dict()),0
            else: pc+=1
            if pc>=PAT: break
        m.load_state_dict(bs)
        pv=[1.0]; wo2=np.ones(n)/n; m.eval()
        with torch.no_grad():
            for t in TEST_R:
                if t+1>=env.dy: break
                pf,tf=env.gs(t)
                if is_cm:
                    pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,_=m(pt,tt)
                elif has_text:
                    s=torch.from_numpy(np.concatenate([pf,tf])).unsqueeze(0).to(DEVICE); wr,_=m(s)
                else:
                    s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); wr,_=m(s)
                w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,n); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wo2
                tv=np.sum(np.abs(ws-wo2))/2.0; y=ca[t+1]/ca[t]; gr=np.sum(ws*y)
                pv.append(max(pv[-1]*(gr-COMM*tv),1e-4)); wo2=ws
        results[mname]=np.array(pv)
        np.save(fname, results[mname])
        print(f"  Saved {fname}")
    # EW
    efname = f"{SAVE_DIR}/pv_{name}_EW.npy"
    if os.path.exists(efname):
        results['EW'] = np.load(efname)
    else:
        pv=[1.0]; w=np.ones(n)/n
        for ti,t in enumerate(TEST_R):
            if t+1>=len(ca): break
            y=ca[t+1]/ca[t]; gr=np.sum(w*y); pv.append(pv[-1]*gr); w=w*y/np.sum(w*y)
            if ti%22==0:
                wn=np.ones(n)/n; pv[-1]=pv[-1]*(1-COMM*np.sum(np.abs(wn-w))/2.0); w=wn.copy()
        results['EW']=np.array(pv)
        np.save(efname, results['EW'])
    return results

for stocks, name in [(SMALL8,'8Small'), (ALL24,'24All')]:
    print(f"\n{'='*40}\n  {name}\n{'='*40}")
    train_and_bt(stocks, name)

print("\nAll PV data saved to", SAVE_DIR)
