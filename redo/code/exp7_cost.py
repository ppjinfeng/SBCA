"""
exp7_cost.py — Transaction Cost Sensitivity Analysis
==================================================================
COMM in {0.001, 0.0025, 0.005, 0.01} x SBCA/EW/BH x 4 groups.
Fixed: SBCA, GAM=0.99, EA=0.4, TP=0.005, W=30
Groups: 4Large/8Large/4Small/8Small
Output: result/exp7_cost.csv
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

W,GAM=30,0.99;LRISK,EA,TP=0.1,0.4,0.005;EP,PAT,LR,ENT=30,5,3e-4,0.1
COST_GRID=[0.001,0.0025,0.005,0.01]
def set_seed(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

df=pd.read_csv(f"{BASE}/data/bert_pred_24stocks_delta_FinBERT.csv");df["Date"]=pd.to_datetime(df["Date"])
df_c=df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
df_d=df.pivot_table(index='Date',columns='Stock_symbol',values='delta_bert').shift(1).ffill().fillna(0.5)
dates=df_c.index
TE=dates.get_indexer([pd.to_datetime('2017-12-31')],method='pad')[0]
VE=dates.get_indexer([pd.to_datetime('2019-12-31')],method='pad')[0]
DAYS_=dates.get_indexer([pd.to_datetime('2021-12-31')],method='pad')[0]
all_s=df_c.columns.tolist()
LARGE=['GILD','COP','EOG','MRK','WFC','ORCL','CMCSA','CAT']
L_sorted='CMCSA GILD MRK ORCL COP WFC CAT EOG'.split()
SMALL=['BGS','WBS','EXLS','CLH','DDS','ALNY','CF','KBH']
S_sorted='CF DDS BGS KBH ALNY WBS EXLS CLH'.split()
MID=['WDC','BIIB','KEY','CLX','RRC','DECK','CI','FTI']
M_sorted='BIIB WDC CI RRC CLX KEY DECK FTI'.split()
SZ={}
for s in LARGE:SZ[s]='large'
for s in MID:SZ[s]='mid'
for s in SMALL:SZ[s]='small'
L=[s for s in all_s if SZ.get(s)=='large'];M=[s for s in all_s if SZ.get(s)=='mid'];S=[s for s in all_s if SZ.get(s)=='small']
KEY=[(L_sorted[:4],"4Large"),(L_sorted[:8],"8Large"),(M_sorted[:4],"4Mid"),(M_sorted[:8],"8Mid"),(S_sorted[:4],"4Small"),(S_sorted[:8],"8Small"),(L_sorted[:1]+M_sorted[:1]+S_sorted[:1],"3Mix"),(L_sorted[:2]+M_sorted[:2]+S_sorted[:2],"6Mix"),(L_sorted[:4]+M_sorted[:4]+S_sorted[:4],"12Mix"),(L_sorted[:6]+M_sorted[:6]+S_sorted[:6],"18Mix"),(L_sorted[:8]+M_sorted[:8]+S_sorted[:8],"24All")]

def _out_dim(n):return n
def _trim_w(w,n):return w[...,:n]
class CM(nn.Module):
    def __init__(s,pd,td,h=128):
        super().__init__();s.pe=nn.Sequential(nn.Linear(pd,h),nn.LayerNorm(h),nn.ReLU());s.te=nn.Sequential(nn.Linear(td,h),nn.LayerNorm(h),nn.ReLU());s.gamma=nn.Linear(h,h);s.beta=nn.Linear(h,h);s.fu=nn.Sequential(nn.Linear(h,h),nn.ReLU())
    def forward(s,p,t):pf=s.pe(p);tf=s.te(t);return s.fu(s.gamma(tf)*pf+s.beta(tf))
class SBCA(CM):
    def __init__(s,ws,n):super().__init__(ws*n,n);s.na=n;s.ac=nn.Sequential(nn.Linear(128,n),nn.Softmax(-1));s.cr=nn.Sequential(nn.Linear(128,1))
    def forward(s,p,t):f=super().forward(p,t);return s.ac(f),s.cr(f)
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
        else:pw=s.c[t-W:t];ret=np.diff(pw,axis=0)/(pw[:-1]+1e-8);ret=np.concatenate([np.zeros((1,s.na)),ret],axis=0);pf=((ret.flatten()-s.pm)/s.ps).astype(np.float32)
        tf=((s.d[t].copy()-0.5)*2).astype(np.float32);return pf,tf

def calc_sharpe(pv):
    ret=np.diff(np.log(pv));drf=0.02/252;return (np.mean(ret-drf)/(np.std(ret)+1e-8))*np.sqrt(252)
def max_dd(pv):
    pk=np.maximum.accumulate(pv);return np.min((pv-pk)/(pk+1e-8))
def evaluate(pv):
    pv=np.array(pv);sr=calc_sharpe(pv);ret=np.diff(np.log(pv));d=ret[ret<0]
    so=np.mean(ret)/(np.std(d)+1e-8)*np.sqrt(252) if len(d)>0 else 10
    pk=np.maximum.accumulate(pv);md=np.min((pv-pk)/(pk+1e-8));ar=pv[-1]**(252/(len(pv)-1))-1;ca=ar/(abs(md)+1e-8)
    return dict(PV=round(float(pv[-1]),4),SR=round(sr,4),Sortino=round(so,4),MDD=round(md,4),AR=round(ar,4),Calmar=round(ca,4))

def NA(a):return (a-a.mean())/(a.std()+1e-8) if a.numel()>1 else a
def RA(w,y,wo,c):
    tv=torch.sum(torch.abs(w-wo))/2.0;nr=torch.sum(w*y)-c*tv;lr=torch.log(nr.clamp(min=1e-4));return lr,torch.clamp(-lr,min=0.0)**2,tv

def train_ac(model,env,tr,vr,comm,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5);sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train();wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy:break
            pf,tf=env.gs(t);pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE);tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE);wr,v=model(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na)
            ws=EA*w+(1-EA)*wo
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE);lr_,risk,tv=RA(ws,y,wo,comm);ent=-(w*torch.log(w+1e-8)).sum()
            pfn,tfn=env.gs(min(t+1,env.dy-1));ptn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE);ttn=torch.from_numpy(tfn).unsqueeze(0).to(DEVICE);_,vn=model(ptn,ttn);vn=vn.squeeze()
            target=lr_+GAM*vn;adv=NA((target-v.squeeze()).detach())
            loss=-adv*lr_+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-ENT*ent
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),0.5);opt.step();wo=ws.detach()
        vp=1.0;wv=np.ones(env.na)/env.na;model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy:break
                pf,tf=env.gs(t);pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE);tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE);wr,_=model(pt,tt)
                w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0;y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);vp=max(vp*(gr-comm*tv),1e-4);wv=ws
        sch.step(vp)
        if vp>bv:bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else:pc+=1
        if pc>=PAT:break
    model.load_state_dict(bs);return model

class PPO_AC(nn.Module):
    def __init__(s,sd,n):super().__init__();s.na=n;s.bb=nn.Sequential(nn.Linear(sd,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,128),nn.ReLU());s.ac=nn.Sequential(nn.Linear(128,n),nn.Softmax(-1));s.cr=nn.Sequential(nn.Linear(128,1))
    def forward(s,x):f=s.bb(x);return s.ac(f),s.cr(f)

def bt_pp(model,env,te,comm):
    pv,turns=[1.0],[];wo=np.ones(env.na)/env.na;model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy:break
            s=env.gs(t)[0];pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE);wr,_=model(pt)
            w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0;turns.append(tv);y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);pv.append(max(pv[-1]*(gr-comm*tv),1e-4));wo=ws
    return np.array(pv),np.mean(turns)*252

def train_ppo(model,env,tr,vr,comm,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5);sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train();wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tr:
            if t+1>=env.dy:break
            s=env.gs(t)[0];pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE);wr,v=model(pt)
            w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na);ws=EA*w+(1-EA)*wo
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE);lr_,risk,tv=RA(ws,y,wo,comm);ent=-(w*torch.log(w+1e-8)).sum()
            sn=env.gs(min(t+1,env.dy-1))[0];ptn=torch.from_numpy(sn).unsqueeze(0).to(DEVICE);_,vn=model(ptn);vn=vn.squeeze()
            target=lr_+GAM*vn;adv=(target-v.squeeze()).detach();adv=adv/(adv.std()+1e-8)
            olp=torch.log(wr.softmax(-1).clamp(min=1e-8)).sum();nlp=torch.log(w.clamp(min=1e-8)).sum()
            ratio=(nlp-olp.detach()).exp();clipped=torch.clamp(ratio,0.8,1.2)
            loss=-torch.min(ratio*adv,clipped*adv)+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+LRISK*risk-0.01*ent
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),0.5);opt.step();wo=ws.detach()
        vp=1.0;wv=np.ones(env.na)/env.na;model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy:break
                s=env.gs(t)[0];pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE);wr,_=model(pt)
                w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0;y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);vp=max(vp*(gr-comm*tv),1e-4);wv=ws
        sch.step(vp)
        if vp>bv:bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else:pc+=1
        if pc>=PAT:break
    model.load_state_dict(bs);return model

def dj(env,te,c):
    pv=[1.0];ip=env.c[te[0]];w=ip/np.sum(ip);pv[-1]=pv[-1]*(1-c*np.sum(w)/2.0)
    for t in te:
        if t+1>=env.dy:break
        y=env.c[t+1]/env.c[t];gr=np.sum(w*y);pv.append(pv[-1]*gr);w=w*y/np.sum(w*y)
    return np.array(pv)

def mv(env,te,c,wd=60):
    pv=[1.0];na=env.na;w=np.ones(na)/na
    for ti,t in enumerate(te):
        if t+1>=env.dy:break
        y=env.c[t+1]/env.c[t];gr=np.sum(w*y);pv.append(pv[-1]*gr);w=w*y/np.sum(w*y)
        if ti>0 and ti%22==0 and t>=wd:
            rets=np.array([env.c[i+1]/env.c[i]-1 for i in range(t-wd,t)])
            try:
                cov=np.cov(rets.T);inv=np.linalg.inv(cov);wn=inv@np.ones(na)/(np.ones(na)@inv@np.ones(na));wn=np.clip(wn,0,1);wn/=wn.sum()+1e-8
                pv[-1]=pv[-1]*(1-c*np.sum(np.abs(wn-w))/2.0);w=wn
            except:pass
    return np.array(pv)

def rp(env,te,c,wd=60):
    pv=[1.0];na=env.na;w=np.ones(na)/na
    for ti,t in enumerate(te):
        if t+1>=env.dy:break
        y=env.c[t+1]/env.c[t];gr=np.sum(w*y);pv.append(pv[-1]*gr);w=w*y/np.sum(w*y)
        if ti>0 and ti%22==0 and t>=wd:
            rets=np.array([env.c[i+1]/env.c[i]-1 for i in range(t-wd,t)]);vols=np.std(rets,axis=0);inv=1.0/(vols+1e-8);wn=inv/inv.sum()
            pv[-1]=pv[-1]*(1-c*np.sum(np.abs(wn-w))/2.0);w=wn
    return np.array(pv)

def bt(model,env,te,comm):
    pv,turns=[1.0],[];wo=np.ones(env.na)/env.na;model.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy:break
            pf,tf=env.gs(t);pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE);tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE);wr,_=model(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1);w=_trim_w(w,model.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0;turns.append(tv);y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);pv.append(max(pv[-1]*(gr-comm*tv),1e-4));wo=ws
    return np.array(pv),np.mean(turns)*252

def ew(env,te,c):
    pv=[1.0];w=np.ones(env.na)/env.na
    for t in te:
        if t+1>=env.dy:break
        y=env.c[t+1]/env.c[t];gr=np.sum(w*y);pv.append(pv[-1]*gr);w=w*y/np.sum(w*y)
        if (t-te.start)%22==0:wn=np.ones(env.na)/env.na;pv[-1]=pv[-1]*(1-c*np.sum(np.abs(wn-w))/2.0);w=wn.copy()
    return np.array(pv)

def bh(env,te,c):
    pv=[1.0];w=np.ones(env.na)/env.na;pv[-1]=pv[-1]*(1-c*np.sum(w)/2.0)
    for t in te:
        if t+1>=env.dy:break
        y=env.c[t+1]/env.c[t];gr=np.sum(w*y);pv.append(pv[-1]*gr);w=w*y/np.sum(w*y)
    return np.array(pv)

TR=range(W,TE);VR=range(TE,VE);TEST_=range(VE,DAYS_)
out=f"{BASE}/result/exp7_cost.csv"
if not os.path.exists(out):
    rows=[]
    for stks,gn in KEY:
        na=len(stks);ca=df_c[stks].values;da=df_d[stks].values
        for cost in COST_GRID:
            env=Env(ca,da);set_seed(42);m=SBCA(W,na).to(DEVICE)
            m=train_ac(m,env,TR,VR,cost,f"COST{cost}")
            pv,ato=bt(m,env,TEST_,cost);ev=evaluate(pv)
            rows.append([gn,na,cost,'SBCA',ev['PV'],ev['SR'],ev['Sortino'],ev['MDD'],ev['AR'],ev['Calmar'],round(ato,4)])
            pv_ew=ew(env,TEST_,cost);ev_ew=evaluate(pv_ew)
            rows.append([gn,na,cost,'EW',ev_ew['PV'],ev_ew['SR'],ev_ew['Sortino'],ev_ew['MDD'],ev_ew['AR'],ev_ew['Calmar'],0])
            pv_bh=bh(env,TEST_,cost);ev_bh=evaluate(pv_bh)
            rows.append([gn,na,cost,'BH',ev_bh['PV'],ev_bh['SR'],ev_bh['Sortino'],ev_bh['MDD'],ev_bh['AR'],ev_bh['Calmar'],0])
            # DJ/MinVar/RP (no ATO tracking)
            pv_dj=dj(env,TEST_,cost);ev_dj=evaluate(pv_dj)
            rows.append([gn,na,cost,'DJ',ev_dj['PV'],ev_dj['SR'],ev_dj['Sortino'],ev_dj['MDD'],ev_dj['AR'],ev_dj['Calmar'],0])
            pv_mv=mv(env,TEST_,cost,60);ev_mv=evaluate(pv_mv)
            rows.append([gn,na,cost,'MinVar',ev_mv['PV'],ev_mv['SR'],ev_mv['Sortino'],ev_mv['MDD'],ev_mv['AR'],ev_mv['Calmar'],0])
            pv_rp=rp(env,TEST_,cost,60);ev_rp=evaluate(pv_rp)
            rows.append([gn,na,cost,'RP',ev_rp['PV'],ev_rp['SR'],ev_rp['Sortino'],ev_rp['MDD'],ev_rp['AR'],ev_rp['Calmar'],0])
            # PPO per cost level
            set_seed(42);m_pp=PPO_AC(W*na,na).to(DEVICE)
            m_pp=train_ppo(m_pp,env,TR,VR,cost,f"PPO_c{cost}")
            pv_pp,ato_pp=bt_pp(m_pp,env,TEST_,cost);ev_pp=evaluate(pv_pp)
            rows.append([gn,na,cost,'PPO',ev_pp['PV'],ev_pp['SR'],ev_pp['Sortino'],ev_pp['MDD'],ev_pp['AR'],ev_pp['Calmar'],round(ato_pp,4)])
            print(f"{gn} cost={cost}: SBCA_SR={ev['SR']:.4f} EW_SR={ev_ew['SR']:.4f}")
    pd.DataFrame(rows,columns=["Group","N","Cost","Model","PV","SR","Sortino","MDD","AR","Calmar","ATO"]).to_csv(out,index=False)
print("Done.")
