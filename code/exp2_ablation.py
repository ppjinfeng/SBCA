"""
exp2_ablation.py — Ablation study: SB-P/S-BERT/SBA/SBC/SBCA x 11 groups
======================================================================
SB-P   = Price only, no CM, no AC (Policy_PriceOnly)
S-BERT = Price + text concat, no CM, no AC (Policy_BERT)
SBA    = Price + text concat, no CM, with AC (Policy_BERT_AC)
SBC    = Price + text CM fusion, with CM, no AC (Policy_BERT_CM)
SBCA   = Price + text CM fusion, with CM, with AC (Policy_BERT_CM_AC)
Output: result/exp2_ablation.csv
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
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

df = pd.read_csv(f"{BASE}/data/bert_pred_24stocks_delta_G.csv"); df["Date"] = pd.to_datetime(df["Date"])
df_c = df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
df_d = df.pivot_table(index='Date',columns='Stock_symbol',values='delta_bert').fillna(0.5)
dates = df_c.index; TE = dates.get_indexer([pd.to_datetime('2017-12-31')],method='pad')[0]
VE = dates.get_indexer([pd.to_datetime('2019-12-31')],method='pad')[0]
DAYS_ = dates.get_indexer([pd.to_datetime('2021-12-31')],method='pad')[0]

LARGE=['NVDA','ORCL','CRM','QCOM','WFC','MRK','KO','CAT']
MID=['BBY','CLX','BIIB','AA','KEY','WDC','AAL','HAL']
SMALL=['SPWR','URBN','ANF','FL','PLUG','KSS','NVAX','ALK']
SIZE_MAP={}
for s in LARGE: SIZE_MAP[s]='large'
for s in MID: SIZE_MAP[s]='mid'
for s in SMALL: SIZE_MAP[s]='small'
all_s=df_c.columns.tolist()
L=[s for s in all_s if SIZE_MAP.get(s)=='large']
M=[s for s in all_s if SIZE_MAP.get(s)=='mid']
S=[s for s in all_s if SIZE_MAP.get(s)=='small']
GROUPS=[(L[:4],'4Large'),(L[:8],'8Large'),(M[:4],'4Mid'),(M[:8],'8Mid'),(S[:4],'4Small'),(S[:8],'8Small'),(L[:1]+M[:1]+S[:1],'3Mix'),(L[:2]+M[:2]+S[:2],'6Mix'),(L[:4]+M[:4]+S[:4],'12Mix'),(L[:6]+M[:6]+S[:6],'18Mix'),(L[:8]+M[:8]+S[:8],'24All')]

# Models: matching original SBCA.py
def _out_dim(n): return n
def _trim_w(w,n): return w[...,:n]

class CM(nn.Module):
    def __init__(s,pd,td,h=64):
        super().__init__()
        s.pe=nn.Sequential(nn.Linear(pd,h),nn.LayerNorm(h),nn.ReLU())
        s.te=nn.Sequential(nn.Linear(td,h),nn.LayerNorm(h),nn.ReLU())
        s.sc=nn.Sequential(nn.Linear(h,h),nn.Tanh()); s.fu=nn.Sequential(nn.Linear(h,h),nn.ReLU())
    def forward(s,p,t): pf=s.pe(p); tf=s.te(t); return s.fu(pf*(1+s.sc(tf))+tf)

# SB-P: Policy_PriceOnly (SBCA_extended.py)
class SBP(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.net=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,n),nn.Softmax(-1))
    def forward(s,x): return s.net(x)

# S-BERT: Policy_BERT (SBCA.py line149)
class SBERT(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.net=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,n),nn.Softmax(-1))
    def forward(s,x): return s.net(x)

# SBA: Policy_BERT_AC (SBCA.py line159)
class SBA(nn.Module):
    def __init__(s,sd,n): super().__init__(); s.na=n; s.bb=nn.Sequential(nn.Linear(sd,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,64),nn.ReLU()); s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1)); s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,x): f=s.bb(x); return s.ac(f),s.cr(f)

# SBC: Policy_BERT_CM (SBCA.py line172)
class SBC(CM):
    def __init__(s,ws,n): super().__init__(ws*n,n); s.na=n; s.hd=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1))
    def forward(s,p,t): return s.hd(super().forward(p,t))

# SBCA: Policy_BERT_CM_AC (SBCA.py line179)
class SBCA(CM):
    def __init__(s,ws,n): super().__init__(ws*n,n); s.na=n; s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1)); s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,p,t): f=super().forward(p,t); return s.ac(f),s.cr(f)

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

def calc_sharpe(pv):
    ret=np.diff(np.log(pv)); drf=0.02/252
    return (np.mean(ret-drf)/(np.std(ret)+1e-8))*np.sqrt(252)
def evaluate(pv):
    pv=np.array(pv); sr=calc_sharpe(pv); ret=np.diff(np.log(pv)); d=ret[ret<0]
    so=np.mean(ret)/(np.std(d)+1e-8)*np.sqrt(252) if len(d)>0 else 10
    pk=np.maximum.accumulate(pv); md=np.min((pv-pk)/(pk+1e-8))
    ar=pv[-1]**(252/len(pv))-1; ca=ar/(abs(md)+1e-8)
    return dict(PV=round(float(pv[-1]),4),SR=round(sr,4),Sortino=round(so,4),MDD=round(md,4),AR=round(ar,4),Calmar=round(ca,4))

def NA(a): return (a-a.mean())/(a.std()+1e-8) if a.numel()>1 else a
def RA(w,y,wo,c):
    tv=torch.sum(torch.abs(w-wo))/2.0; nr=torch.sum(w*y)-c*tv; lr=torch.log(nr.clamp(min=1e-4))
    return lr,torch.clamp(-lr,min=0.0)**2,tv

def vpv(env,model,vr,is_cm,is_ac):
    model.eval(); pv,wo=1.0,np.ones(env.na)/env.na
    with torch.no_grad():
        for t in vr:
            if t+1>=env.dy: break
            pf,tf=env.gs(t)
            if is_cm: pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); out=model(pt,tt)
            elif tf is not None: s=torch.from_numpy(np.concatenate([pf,tf])).unsqueeze(0).to(DEVICE); out=model(s)
            else: s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); out=model(s)
            wr=out[0].squeeze(0) if is_ac else out.squeeze(0)
            w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na)
            wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0; y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
            pv=max(pv*(gr-COMM*tv),1e-4); wo=ws
    return pv

def train_loop(model,env,tr,vr,is_cm,is_ac,has_text,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train(); wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy: break
            pf,tf=env.gs(t)
            if is_cm: pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); out=model(pt,tt)
            elif has_text: s=torch.from_numpy(np.concatenate([pf,tf])).unsqueeze(0).to(DEVICE); out=model(s)
            else: s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); out=model(s)
            if is_ac: wr,v=out[0].squeeze(0),out[1]
            else: wr,v=out.squeeze(0),None
            w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na)
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE)
            lr_,risk,tv=RA(w,y,wo,COMM); ent=-(w*torch.log(w+1e-8)).sum()
            if is_ac:
                pfn,tfn=env.gs(min(t+1,env.dy-1))
                if is_cm: ptn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE); ttn=torch.from_numpy(tfn).unsqueeze(0).to(DEVICE); _,vn=model(ptn,ttn)
                elif has_text: sn=torch.from_numpy(np.concatenate([pfn,tfn])).unsqueeze(0).to(DEVICE); _,vn=model(sn)
                else: sn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE); _,vn=model(sn)
                vn=vn.squeeze(); target=lr_+GAM*vn; adv=NA((target-v.squeeze()).detach())
                loss=-adv*lr_+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-ENT*ent
            else: loss=-lr_+TP*tv+LRISK*risk-ENT*ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            wo=EA*w.detach()+(1-EA)*wo
        vp=vpv(env,model,vr,is_cm,is_ac); sch.step(vp)
        if vp>bv: bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else: pc+=1
        if pc>=PAT: break
    model.load_state_dict(bs); return model

def bt(model,env,te,is_cm,is_ac,has_text):
    pv,turns=[1.0],[]; wo=np.ones(env.na)/env.na; model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy: break
            pf,tf=env.gs(t)
            if is_cm: pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE); out=model(pt,tt)
            elif has_text: s=torch.from_numpy(np.concatenate([pf,tf])).unsqueeze(0).to(DEVICE); out=model(s)
            else: s=torch.from_numpy(pf).unsqueeze(0).to(DEVICE); out=model(s)
            wr=out[0].squeeze(0) if is_ac else out.squeeze(0)
            w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na)
            wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0; turns.append(tv); y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
            pv.append(max(pv[-1]*(gr-COMM*tv),1e-4)); wo=ws
    return np.array(pv), np.mean(turns)*252

# Run
TR=range(W,TE); VR=range(TE,VE); TEST_=range(VE,DAYS_)
out_csv=f"{BASE}/result/exp2_ablation.csv"

if not os.path.exists(out_csv):
    rows=[]
    configs=[
        ('SB-P',SBP,False,False,False),('S-BERT',SBERT,False,False,True),
        ('SBA',SBA,False,True,True),('SBC',SBC,True,False,True),('SBCA',SBCA,True,True,True)]
    for stks,gn in tqdm(GROUPS,desc="EXP2"):
        if len(stks)<2: continue
        na=len(stks); ca=df_c[stks].values; da=df_d[stks].values
        print(f"\n{gn} ({na} assets)")
        for name,cls,is_cm,is_ac,has_txt in configs:
            if is_cm: sd=W*na
            elif has_txt: sd=W*na+na
            else: sd=W*na
            env=Env(ca,da if has_txt else None); set_seed(42)
            m=cls(sd,na).to(DEVICE) if not is_cm else cls(W,na).to(DEVICE)
            m=train_loop(m,env,TR,VR,is_cm,is_ac,has_txt,f"{name}_{gn}")
            pv,ato=bt(m,env,TEST_,is_cm,is_ac,has_txt); ev=evaluate(pv)
            rows.append([gn,na,name,ev['PV'],ev['SR'],ev['Sortino'],ev['MDD'],ev['AR'],ev['Calmar'],round(ato,4)])
            print(f"  {name}: SR={ev['SR']:.4f}")
    pd.DataFrame(rows,columns=["Group","N","Model","PV","SR","Sortino","MDD","AR","Calmar","ATO"]).to_csv(out_csv,index=False)
    print(f"\nSaved: {out_csv}")
else:
    print(f"Already exists: {out_csv}")
print("Done.")
