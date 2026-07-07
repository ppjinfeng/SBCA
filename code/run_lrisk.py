"""λ_risk sensitivity: SBCA with varying risk penalty on 8Large/8Small"""
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
import random, copy; from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

DEVICE=torch.device('cuda'); FI=r'C:\pony\投稿\FI - 副本'
W,COMM,GAM=30,0.0025,0.99; EA,TP=0.4,0.005; EP,PAT,LR,ENT=30,5,3e-4,0.1
def set_seed(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

LARGE=['NVDA','ORCL','CRM','QCOM','WFC','MRK','KO','CAT']
SMALL=['SPWR','URBN','ANF','FL','PLUG','KSS','NVAX','ALK']

df=pd.read_csv(FI+r'\data\bert_pred_24stocks_delta_G.csv');df['Date']=pd.to_datetime(df['Date'])
df_c=df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
df_d=df.pivot_table(index='Date',columns='Stock_symbol',values='delta_bert').fillna(0.5)
dates=df_c.index;TE=dates.get_indexer([pd.to_datetime('2017-12-31')],method='pad')[0]
VE=dates.get_indexer([pd.to_datetime('2019-12-31')],method='pad')[0]
DY=dates.get_indexer([pd.to_datetime('2021-12-31')],method='pad')[0]

def _tw(w,n):return w[...,:n]
class CM(nn.Module):
    def __init__(s,pd,td,h=64):
        super().__init__();s.pe=nn.Sequential(nn.Linear(pd,h),nn.LayerNorm(h),nn.ReLU())
        s.te=nn.Sequential(nn.Linear(td,h),nn.LayerNorm(h),nn.ReLU());s.sc=nn.Sequential(nn.Linear(h,h),nn.Tanh());s.fu=nn.Sequential(nn.Linear(h,h),nn.ReLU())
    def forward(s,p,t):pf=s.pe(p);tf=s.te(t);return s.fu(pf*(1+s.sc(tf))+tf)
class SBCA(CM):
    def __init__(s,ws,n):super().__init__(ws*n,n);s.na=n;s.ac=nn.Sequential(nn.Linear(64,n),nn.Softmax(-1));s.cr=nn.Sequential(nn.Linear(64,1))
    def forward(s,p,t):f=super().forward(p,t);return s.ac(f),s.cr(f)
class Env:
    def __init__(s,ca,da):s.c=ca;s.d=da;s.na=ca.shape[1];s.dy=ca.shape[0];s.pm,s.ps=s._cp()
    def _cp(s):
        rs=[]
        for t in range(W,TE):pw=s.c[t-W:t]
        if len(pw)>1:ret=np.diff(pw,axis=0)/(pw[:-1]+1e-8);rs.append(ret.flatten())
        rs=np.concatenate(rs) if rs else np.zeros(1);return np.mean(rs),np.std(rs)+1e-8
    def gs(s,t):
        if t<W:pf=np.zeros(W*s.na,dtype=np.float32)
        else:
            pw=s.c[t-W:t];ret=np.diff(pw,axis=0)/(pw[:-1]+1e-8)
            ret=np.concatenate([np.zeros((1,s.na)),ret],axis=0);pf=((ret.flatten()-s.pm)/s.ps).astype(np.float32)
        tf=((s.d[t].copy()-0.5)*2).astype(np.float32);return pf,tf
def NA(a):return (a-a.mean())/(a.std()+1e-8) if a.numel()>1 else a
def RA(w,y,wo,c,lrisk):
    tv=torch.sum(torch.abs(w-wo))/2.0;nr=torch.sum(w*y)-c*tv;lr=torch.log(nr.clamp(min=1e-4));return lr,torch.clamp(-lr,min=0.0)**2,tv
def csh(pv):ret=np.diff(np.log(pv));drf=0.02/252;return (np.mean(ret-drf)/(np.std(ret)+1e-8))*np.sqrt(252)
def eva(pv):
    pv=np.array(pv);sr=csh(pv);ret=np.diff(np.log(pv));d=ret[ret<0]
    so=np.mean(ret)/(np.std(d)+1e-8)*np.sqrt(252) if len(d)>0 else 10
    pk=np.maximum.accumulate(pv);md=np.min((pv-pk)/(pk+1e-8));ar=pv[-1]**(252/len(pv))-1;ca=ar/(abs(md)+1e-8)
    return dict(SR=round(sr,4),Sortino=round(so,4),MDD=round(md,4),AR=round(ar,4),Calmar=round(ca,4))

def train_ac(m,env,tr,vr,lrisk,desc=''):
    opt=optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5);sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(m.state_dict())
    for ep in range(EP):
        m.train();wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f'{desc} E{ep+1}',leave=False):
            if t+1>=env.dy:break
            pf,tf=env.gs(t);pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE);tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE);wr,v=m(pt,tt)
            w=torch.softmax(wr.squeeze(0),dim=-1);w=_tw(w,m.na);y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE)
            lr_,risk,tv=RA(w,y,wo,COMM,lrisk);ent=-(w*torch.log(w+1e-8)).sum()
            pfn,tfn=env.gs(min(t+1,env.dy-1));ptn=torch.from_numpy(pfn).unsqueeze(0).to(DEVICE);ttn=torch.from_numpy(tfn).unsqueeze(0).to(DEVICE);_,vn=m(ptn,ttn)
            vn=vn.squeeze();target=lr_+GAM*vn;adv=NA((target-v.squeeze()).detach())
            loss=-adv*lr_+0.5*nn.functional.mse_loss(v.squeeze(),target.detach())+TP*tv+lrisk*risk-ENT*ent
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),0.5);opt.step();wo=EA*w.detach()+(1-EA)*wo
        vp=1.0;wv=np.ones(env.na)/env.na;m.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy:break
                pf,tf=env.gs(t);pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE);tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE);wr,_=m(pt,tt)
                w=torch.softmax(wr.squeeze(0),dim=-1);w=_tw(w,m.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0;y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);vp=max(vp*(gr-COMM*tv),1e-4);wv=ws
        sch.step(vp)
        if vp>bv:bv,bs,pc=vp,copy.deepcopy(m.state_dict()),0
        else:pc+=1
        if pc>=PAT:break
    m.load_state_dict(bs);return m

def bt(m,env,te):
    pv=[1.0];wo=np.ones(env.na)/env.na;m.eval()
    with torch.no_grad():
        for t in te:
            if t+1>=env.dy:break
            pf,tf=env.gs(t);pt=torch.from_numpy(pf).unsqueeze(0).to(DEVICE);tt=torch.from_numpy(tf).unsqueeze(0).to(DEVICE);out=m(pt,tt)
            wr=out[0].squeeze(0);w=torch.softmax(wr,dim=-1);w=_tw(w,m.na);wn=w.cpu().numpy();ws=EA*wn+(1-EA)*wo
            tv=np.sum(np.abs(ws-wo))/2.0;y=env.c[t+1]/env.c[t];gr=np.sum(ws*y);pv.append(max(pv[-1]*(gr-COMM*tv),1e-4));wo=ws
    return np.array(pv)

TR=range(W,TE);VR=range(TE,VE);TE_=range(VE,DY)
LRISK_VALS=[0,0.05,0.1,0.2,0.5]
GROUPS=[(LARGE,'8Large'),(SMALL,'8Small')]

print(f"{'Group':10s} {'λ_risk':>8s} {'SR':>8s} {'MDD':>8s} {'Sortino':>8s}")
for stks,gn in GROUPS:
    na=len(stks);ca=df_c[stks].values;da=df_d[stks].values
    for lr in LRISK_VALS:
        set_seed(42);env=Env(ca,da);m=SBCA(W,na).to(DEVICE);m=train_ac(m,env,TR,VR,lr,f'SBCA_lr{lr}_{gn}')
        pv=bt(m,env,TE_);ev=eva(pv)
        print(f'{gn:10s} {lr:8.3f} {ev["SR"]:8.4f} {ev["MDD"]:8.4f} {ev["Sortino"]:8.4f}')
print('Done!')
