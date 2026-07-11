"""
reward_decomp.py — Reward Function Decomposition
2x2 factorial: Full / NoRisk / NoTurn / LogOnly
Also compares SBC (actor-only) for ATO decomposition
Output: result/reward_decomp.csv (for tab:reward_decomp and tab:ato_decomp)
"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
import random, copy, os, sys
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{BASE}/result", exist_ok=True)
print(f"Device: {DEVICE}")

W, COMM, GAM = 30, 0.0025, 0.99; EA = 0.4
EP, PAT, LR, ENT, HID = 30, 5, 3e-4, 0.1, 128

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

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
    def __init__(s,ca,da): s.c=ca; s.d=da; s.na=ca.shape[1]; s.dy=ca.shape[0]; s.pm,s.ps=s._cp()
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
        tf=((s.d[t].copy()-0.5)*2).astype(np.float32)
        return pf,tf

def NA(a): return (a-a.mean())/(a.std()+1e-8) if a.numel()>1 else a

def RA(w,y,wo,c,LRISK,TP):
    tv=torch.sum(torch.abs(w-wo))/2.0; nr=torch.sum(w*y)-c*tv; lr=torch.log(nr.clamp(min=1e-4))
    return lr, LRISK*torch.clamp(-lr,min=0.0)**2, TP*tv, tv

def evaluate(pv):
    pv=np.array(pv); ret=np.diff(np.log(pv)); drf=0.02/252
    sr=(np.mean(ret-drf)/(np.std(ret)+1e-8))*np.sqrt(252)
    d=ret[ret<0]; so=np.mean(ret)/(np.std(d)+1e-8)*np.sqrt(252) if len(d)>0 else 10
    pk=np.maximum.accumulate(pv); md=np.min((pv-pk)/(pk+1e-8))
    return float(pv[-1]),sr,so,md

def train_sbca(env,tr,vr,lrisk,tp,desc=""):
    """Train SBCA with given LRISK and TP"""
    model=SBCA(W,env.na).to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train(); wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy: break
            pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,v=model(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); ws=EA*w+(1-EA)*wo
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE)
            lr_,risk,penalty,tv=RA(ws,y,wo,COMM,lrisk,tp); ent=-(w*torch.log(w+1e-8)).sum()
            pfn,tfn=env.gs(min(t+1,env.dy-1)); ptn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE); ttn=torch.from_numpy(tfn).unsqueeze(0).to(DEVICE); _,vn=model(ptn,ttn)
            vn=vn.squeeze(); target=lr_+GAM*vn; adv=NA((target-v.squeeze()).detach())
            loss=-adv*lr_+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+penalty+risk-ENT*ent
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

def bt(model,env,te):
    """Backtest with given model, return PV array, ATO, and metrics"""
    pv,turns=[1.0],[]; wo=np.ones(env.na)/env.na; model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy: break
            pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr,_=model(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0; turns.append(tv); y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
            pv.append(max(pv[-1]*(gr-COMM*tv),1e-4)); wo=ws
    pv,sr,so,md = evaluate(np.array(pv))
    return pv, np.mean(turns)*252, sr, so, md

# SBC (actor-only, CM fusion, no critic)
class SBC_Model(CM):
    def __init__(s,ws,n): super().__init__(ws*n,n); s.na=n; s.ac=nn.Sequential(nn.Linear(128,n),nn.Softmax(-1))
    def forward(s,p,t): f=super().forward(p,t); return s.ac(f)

def train_sbc(env,tr,vr,tp,desc=""):
    """Train SBC (actor-only) with given TP"""
    model=SBC_Model(W,env.na).to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train(); wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy: break
            pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr=model(pt,tt).squeeze(0)
            w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na); ws=EA*w+(1-EA)*wo
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE)
            tv=torch.sum(torch.abs(ws-wo))/2.0; nr=torch.sum(ws*y)-COMM*tv; lr=torch.log(nr.clamp(min=1e-4))
            loss=-lr+tp*tv; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step(); wo=ws.detach()
        vp=1.0; wv=np.ones(env.na)/env.na; model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy: break
                pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr=model(pt,tt).squeeze(0)
                w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0; y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y); vp=max(vp*(gr-COMM*tv),1e-4); wv=ws
        sch.step(vp)
        if vp>bv: bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else: pc+=1
        if pc>=PAT: break
    model.load_state_dict(bs); return model

def bt_sbc(model,env,te):
    """Backtest SBC"""
    pv,turns=[1.0],[]; wo=np.ones(env.na)/env.na; model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy: break
            pf,tf=env.gs(t); pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); wr=model(pt,tt).squeeze(0)
            w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na); wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0; turns.append(tv); y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
            pv.append(max(pv[-1]*(gr-COMM*tv),1e-4)); wo=ws
    return np.mean(turns)*252

# Configs for SBCA
configs = [
    ("Full",    0.1, 0.005),
    ("NoRisk",  0.0, 0.005),
    ("NoTurn",  0.1, 0.0),
    ("LogOnly", 0.0, 0.0),
]

TR=range(W,TE); VR=range(TE,VE); TEST_=range(VE,DAYS_)

out1 = f"{BASE}/result/reward_decomp.csv"
out2 = f"{BASE}/result/ato_decomp.csv"

rows_decomp = []
rows_ato = []

for stks,gn in tqdm(GROUPS, desc="Groups"):
    if len(stks)<2: continue
    na=len(stks); ca=df_c[stks].values; da=df_d[stks].values
    print(f"\n{gn} ({na} assets)")

    # SBCA: 4 configs
    for cfg_name, lrisk, tp in configs:
        set_seed(42); env=Env(ca,da); m=train_sbca(env,TR,VR,lrisk,tp,f"SBCA_{gn}_{cfg_name}")
        pv,ato,sr,so,md = bt(m,env,TEST_)
        rows_decomp.append([gn,cfg_name,round(pv,4),round(sr,4),round(so,4),round(md,4),round(ato,4)])
        print(f"  SBCA {cfg_name:8s}: SR={sr:.4f} ATO={ato:.4f}")

    # SBC (actor-only): with and without TP
    for tp_val, tp_name in [(0.005,"SBC_TP"),(0.0,"SBC_noTP")]:
        set_seed(42); env_sbc=Env(ca,da); m_sbc=train_sbc(env_sbc,TR,VR,tp_val,f"SBC_{gn}_{tp_name}")
        ato_sbc = bt_sbc(m_sbc,env_sbc,TEST_)
        rows_ato.append([gn,tp_name.replace('_',' '),round(ato_sbc,4)])
        print(f"  SBC {tp_name:8s}: ATO={ato_sbc:.4f}")

# Save
pd.DataFrame(rows_decomp,columns=["Group","Config","PV","SR","Sortino","MDD","ATO"]).to_csv(out1,index=False)
pd.DataFrame(rows_ato,columns=["Group","Config","ATO"]).to_csv(out2,index=False)
print(f"\nSaved: {out1}")
print(f"Saved: {out2}")
print("Done.")
