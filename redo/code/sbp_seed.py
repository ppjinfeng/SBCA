"""
SB-P seed stability: compare seed sensitivity of SB-P (price-only) vs SBCA
SBCA seed results already in result/exp9_seed.csv
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

W, COMM, GAM = 30, 0.0025, 0.99; LRISK, EA, TP = 0.1, 0.4, 0.005
EP, PAT, LR, ENT, HID = 30, 5, 3e-4, 0.1, 128

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

df = pd.read_csv(f"{BASE}/data/bert_pred_24stocks_delta_FinBERT.csv"); df['Date'] = pd.to_datetime(df['Date'])
df_c = df.pivot_table(index='Date',columns='Stock_symbol',values='close').ffill().bfill()
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

def _trim_w(w,n): return w[...,:n]

class SB_P(nn.Module):
    """Pure price policy gradient, no BERT, no AC"""
    def __init__(s,sd,n):
        super().__init__(); s.na=n
        s.net=nn.Sequential(nn.Linear(sd,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,n),nn.Softmax(-1))
    def forward(s,x): return s.net(x)

class Env:
    def __init__(s,ca,da=None):
        s.c=ca; s.d=da; s.na=ca.shape[1]; s.dy=ca.shape[0]
        s.pm,s.ps=s._cp()
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
        return pf

def evaluate(pv):
    pv=np.array(pv); ret=np.diff(np.log(pv)); drf=0.02/252
    sr=(np.mean(ret-drf)/(np.std(ret)+1e-8))*np.sqrt(252); d=ret[ret<0]
    so=np.mean(ret)/(np.std(d)+1e-8)*np.sqrt(252) if len(d)>0 else 10
    pk=np.maximum.accumulate(pv); md=np.min((pv-pk)/(pk+1e-8))
    ar=pv[-1]**(252/(len(pv)-1))-1; ca=ar/(abs(md)+1e-8)
    return dict(PV=round(float(pv[-1]),4),SR=round(sr,4),Sortino=round(so,4),MDD=round(md,4),AR=round(ar,4),Calmar=round(ca,4))

def train_sbp(model,env,tr,vr,desc=""):
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
    sch=optim.lr_scheduler.ReduceLROnPlateau(opt,'max',0.5,1)
    bv,pc,bs=-np.inf,0,copy.deepcopy(model.state_dict())
    for ep in range(EP):
        model.train(); wo=torch.ones(env.na).to(DEVICE)/env.na
        for t in tqdm(tr,desc=f"{desc} E{ep+1}",leave=False):
            if t+1>=env.dy: break
            s=env.gs(t); pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE)
            wr=model(pt).squeeze(0); w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na)
            ws=EA*w+(1-EA)*wo
            y=torch.from_numpy(env.c[t+1]/env.c[t]).float().to(DEVICE)
            tv=torch.sum(torch.abs(ws-wo))/2.0
            lr_=torch.log((torch.sum(ws*y)-COMM*tv).clamp(min=1e-4))
            loss=-lr_+TP*tv; opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step(); wo=ws.detach()
        vp=1.0; wv=np.ones(env.na)/env.na; model.eval()
        with torch.no_grad():
            for t in vr:
                if t+1>=env.dy: break
                s=env.gs(t); pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE)
                wr=model(pt).squeeze(0); w=torch.softmax(wr,dim=-1); w=_trim_w(w,model.na)
                wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wv
                tv=np.sum(np.abs(ws-wv))/2.0; y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
                vp=max(vp*(gr-COMM*tv),1e-4); wv=ws
        sch.step(vp)
        if vp>bv: bv,bs,pc=vp,copy.deepcopy(model.state_dict()),0
        else: pc+=1
        if pc>=PAT: break
    model.load_state_dict(bs); return model

TR=range(W,TE); VR=range(TE,VE); TEST_=range(VE,DAYS_)
SEEDS = [42, 123, 456, 789, 1024]

rows = []
for stks,gn in tqdm(GROUPS, desc="Groups"):
    if len(stks)<2: continue
    na=len(stks); ca=df_c[stks].values
    srs = []; atos = []
    for seed in SEEDS:
        set_seed(seed); env=Env(ca); m=SB_P(W*na,na).to(DEVICE)
        m=train_sbp(m,env,TR,VR,f"SB-P_{gn}_seed{seed}")
        # Backtest
        pv=[1.0]; turns=[]; wo=np.ones(na)/na; m.eval()
        with torch.no_grad():
            for t in TEST_:
                if t+1>=env.dy: break
                s=env.gs(t); pt=torch.from_numpy(s).unsqueeze(0).to(DEVICE)
                wr=m(pt).squeeze(0); w=torch.softmax(wr,dim=-1); w=_trim_w(w,na)
                wn=w.cpu().numpy(); ws=EA*wn+(1-EA)*wo
                tv=np.sum(np.abs(ws-wo))/2.0; turns.append(tv)
                y=env.c[t+1]/env.c[t]; gr=np.sum(ws*y)
                pv.append(max(pv[-1]*(gr-COMM*tv),1e-4)); wo=ws
        ev=evaluate(np.array(pv)); ato=np.mean(turns)*252
        srs.append(ev['SR']); atos.append(ato)
    srs=np.array(srs); atos=np.array(atos)
    rows.append([gn,srs.mean(),srs.std(),srs.std()/abs(srs.mean())*100 if abs(srs.mean())>1e-6 else 0,atos.mean(),atos.std()])
    print(f'{gn}: SR={srs.mean():.4f}±{srs.std():.4f} (CV={rows[-1][3]:.2f}%), ATO={atos.mean():.4f}±{atos.std():.4f}')

out=f"{BASE}/result/sbp_seed.csv"
pd.DataFrame(rows,columns=["Group","SR_mean","SR_std","SR_CV%","ATO_mean","ATO_std"]).to_csv(out,index=False)
print(f"\nSaved: {out}")

# Compare with SBCA
d9=pd.read_csv(f"{BASE}/result/exp9_seed.csv")
print("\n=== SB-P vs SBCA Seed Stability ===")
print(f"{'Group':>10s} {'SB-P CV%':>10s} {'SBCA CV%':>10s} {'Ratio':>8s}")
for _,r in pd.DataFrame(rows,columns=["Group","SR_mean","SR_std","SR_CV%","ATO_mean","ATO_std"]).iterrows():
    d9r = d9[d9['Group']==r['Group']]
    if len(d9r)>0:
        sbca_cv = d9r.iloc[0]['CV%']
        ratio = r['SR_CV%']/sbca_cv if sbca_cv > 0 else float('inf')
        g = r['Group']; cv = r['SR_CV%']
        print(f'{g:>10s} {cv:>10.2f} {sbca_cv:>10.2f} {ratio:>8.1f}x')
print("Done.")
