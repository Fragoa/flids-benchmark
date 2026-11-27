"""FLIDS 10–13: source-locked experimental suite. Run using the companion notebook."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json, hashlib, itertools, time, random, zipfile, argparse, shutil, math
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import binomtest, beta
from lightgbm import LGBMClassifier
import engine02 as original

SEEDS=[0,42,3407,123,2026]
BASE=['LogisticRegression','RandomForest','LightGBM','PooledMLP','LocalMLP','FedAvg','FedProx','FedAdam','FedYogi']
EXTRA=['GroupDRO','CORAL','DANN_Source','SCAFFOLD','FedNova','FedBN_SourceEnsemble']
FL=['FedAvg','FedProx','FedAdam','FedYogi','SCAFFOLD','FedNova','FedBN_SourceEnsemble']
RATES=[.001,.01,.05]
DEADLINE=float('inf')
class PauseRun(Exception):pass
def timegate():
 if time.monotonic()>=DEADLINE:raise PauseRun('Configured session budget reached; resume from candidate/bundle records.')


def seed(*a):return int.from_bytes(hashlib.sha256(repr(a).encode()).digest()[:4],'little')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2,default=str));q.replace(p)
def frame(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');pd.DataFrame(rows).to_csv(q,index=False);q.replace(p)
def npwrite(p,**kw):
 p=Path(p);q=p.with_suffix('.tmp');
 with q.open('wb') as f:np.savez_compressed(f,**kw)
 q.replace(p)
def rng(*a):return np.random.default_rng(seed(*a))
def cut(s,a):
 s=np.sort(np.asarray(s,dtype=np.float64));k=math.floor(a*len(s));return float(np.nextafter(s[len(s)-k-1],np.inf)) if len(s) else np.inf
def conformal(s,a):
 s=np.sort(s);k=math.ceil((len(s)+1)*(1-a));return float(np.nextafter(s[k-1],np.inf)) if k<=len(s) else np.inf
def tolcut(s,a,delta):
 # Smallest order-statistic threshold whose one-sided binomial tolerance coverage is >= 1-delta.
 from scipy.stats import binom
 s=np.sort(s);n=len(s);k=int(binom.ppf(1-delta,n,1-a))+1
 return float(np.nextafter(s[k-1],np.inf)) if k<=n else np.inf
def auc(y,s,a=None):return float(roc_auc_score(y,s,max_fpr=a))
def evalop(y,s,t):
 n=s[y==0];p=s[y==1];fp=int((n>=t).sum());tp=int((p>=t).sum());return fp/len(n),tp/len(p),fp,tp

def dedup(x,y,d,f):
 # Exclude all cross-domain or conflicting-label/family fingerprints, keep one same-domain duplicate.
 keys=np.array([hashlib.blake2b(np.ascontiguousarray(v).tobytes(),digest_size=16).hexdigest() for v in x]);_,inv,cnt=np.unique(keys,return_inverse=True,return_counts=True)
 bad=set();keep=[];seen={}
 for i,k in enumerate(inv):
  val=(str(d[i]),int(y[i]),str(f[i]))
  if k in seen:
   if seen[k][0]!=val:bad.add(k)
  else:seen[k]=(val,i)
 keep=np.array([i for k,(_,i) in seen.items() if k not in bad],dtype=int);return keep,dict(original_rows=len(x),kept_rows=len(keep),conflicting_or_cross_domain_groups=len(bad),duplicate_groups=int((cnt>1).sum()))

def load_nf(work,cfg):
 p=work/'nf_cache.npz';audit=work/'nf_audit.json'
 if p.exists():
  a=np.load(p,allow_pickle=False);return tuple(a[k] for k in ('x','y','d','family'))
 if cfg['smoke']:
  original.CFG['max_rows_per_domain_class']=100;original.CFG['seeds']=[0];os.environ['FLIDS_EXP02_SMOKE']='1';original.prepare()
  a=np.load(original.CACHE)
 elif cfg.get('exp09_zip'):
  with zipfile.ZipFile(cfg['exp09_zip']) as z:
   a=np.load(__import__('io').BytesIO(z.read('reproducibility/flids_exp02_cache.npz'))); provenance=json.loads(z.read('reproducibility/data_preparation/dataset_audit.json'))
   save(work/'original_nf_provenance.json',provenance)
   if provenance.get('source_rows_scanned',0)<1_000_000:raise ValueError('Refusing synthetic or incomplete Exp09 cache in full run')
 else:
  original.prepare();a=np.load(original.CACHE)
 x,y,d,f=[a[k] for k in ('x','y','d','attack')];k,info=dedup(x,y,d,f);save(audit,info);npwrite(p,x=x[k],y=y[k],d=d[k],family=f[k]);return x[k],y[k],d[k],f[k]

def load_edge(work,cfg):
 path=cfg.get('edge_csv');features=cfg.get('edge_features',[])
 if not path or not features:raise ValueError('EDGE_CSV and exact audited EDGE_FEATURES (Safe-30 list) required; no guessed columns')
 label=cfg['edge_label'];family=cfg['edge_family'];assert len(features)==(30 if cfg.get('edge_schema')=='Safe-30' else 25) and len(set(features))==len(features) and not {label,family}&set(features)
 if any(any(t in c.lower() for t in ['ip.src','ip.dst','srcport','dstport','timestamp','frame.time','payload','attack','label']) for c in features):raise ValueError('Unsafe Edge feature names; verify audited list')
 df=pd.read_csv(path,usecols=features+[label,family],low_memory=False);y=pd.to_numeric(df[label],errors='raise').to_numpy(dtype=int);assert set(y)=={0,1};f=df[family].astype(str).to_numpy(dtype=str);assert len(set(f[y==1]))==14, 'Expected 14 attack families; verify source file'
 # Seeded capped per-family/class sample, not natural prevalence.
 ix=[]
 for _,g in df.groupby([label,family]):ix.extend(g.sample(min(len(g),cfg['edge_cap_per_family']),random_state=2077).index)
 ix=np.array(ix);x=df.loc[ix,features].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64);y=y[ix];f=f[ix];d=np.full(len(y),'Edge-IIoTset');k,info=dedup(x,y,d,f);assert len(set(f[k][y[k]==1]))==14, 'Attack family lost after deduplication';save(work/'edge_audit.json',info|{'schema':cfg.get('edge_schema'),'features':features});return x[k],y[k],d[k],f[k]

def load_external(path,features,domain,label,family):
 df=pd.read_csv(path,usecols=features+[label,family]);y=pd.to_numeric(df[label],errors='raise').to_numpy(dtype=int);assert set(y)=={0,1};f=df[family].astype(str).to_numpy(dtype=str);ix=[]
 for v in (0,1):
  ids=np.flatnonzero(y==v);ix.extend(rng(domain,v).choice(ids,min(12000,len(ids)),replace=False))
 ix=np.array(ix);return df.iloc[ix][features].apply(pd.to_numeric,errors='raise').to_numpy(),y[ix],np.full(len(ix),domain),f[ix]

def groupsplit(ids,y,s,frac):
 a,b=train_test_split(np.asarray(ids),test_size=frac,random_state=s,stratify=y[ids]);return np.sort(a),np.sort(b)
def split_outer(y,d,f,held,s,track):
 if track=='edge':
  pos=np.flatnonzero((y==1)&(f==held[0]));ben=np.flatnonzero(y==0);remain,ben_test=groupsplit_ben(ben,s,.20);src=np.r_[remain,np.flatnonzero((y==1)&(f!=held[0]))];test=np.r_[ben_test,pos];domains=[str(d[0])]
 else:
  src=np.flatnonzero(~np.isin(d,held));test=np.flatnonzero(np.isin(d,held));domains=sorted(set(d[src]))
 fit=[];stop=[];cal=[];st=[]
 for dom in domains:
  ids=src[d[src]==dom];a,b=groupsplit(ids,y,seed(s,dom,'outer'),.30);c,e=groupsplit(b,y,seed(s,dom,'cal'),2/3);g,h=groupsplit(e,y,seed(s,dom,'st'),.5);fit.extend(a);stop.extend(c);cal.extend(g);st.extend(h)
 parts=tuple(np.array(v,dtype=int) for v in (fit,stop,cal,st,test));assert len(np.unique(np.concatenate(parts)))==sum(map(len,parts));assert all(set(y[v])=={0,1} for v in parts);return parts
def groupsplit_ben(ids,s,frac):
 v=rng(s,'ben').permutation(ids);k=int(len(v)*frac);return v[k:],v[:k]

def grid(m):
 if m=='LogisticRegression':return [dict(C=v) for v in [.01,.1,1,10]]
 if m=='RandomForest':return [dict(max_depth=d,min_samples_leaf=l) for d,l in [(12,1),(None,1),(16,5),(None,5)]]
 if m=='LightGBM':return [dict(learning_rate=l,num_leaves=n) for l,n in [(.03,31),(.1,31),(.03,63),(.1,63)]]
 lrs=[.001,.003,.01,.03] if m=='SCAFFOLD' else [.0003,.001,.003,.01]
 if m=='FedProx':return [dict(lr=l,mu=mu) for l,mu in [( .001,.001),(.003,.001),(.01,.001),(.01,.01)]]
 if m in ('FedAdam','FedYogi'):return [dict(lr=l,server_lr=v,tau=.01) for l,v in [(.001,.01),(.003,.01),(.003,.05),(.01,.05)]]
 return [dict(lr=l) for l in lrs]

def partition(ids,y,d,s,scenario='native'):
 r=rng(s,'allocation');ids=np.asarray(ids)
 if scenario=='native' and len(set(d[ids]))>=2:return [ids[d[ids]==k] for k in sorted(set(d[ids]))]
 K=5 if len(set(d[ids]))==1 else 3
 if scenario=='native':
  buckets=[[] for _ in range(K)]
  for lab in (0,1):
   a=r.permutation(ids[y[ids]==lab]);p=r.dirichlet(np.full(K,.5));chunks=np.split(a,(np.cumsum(p)[:-1]*len(a)).astype(int));
   for j,c in enumerate(chunks):buckets[j].extend(c)
  return [np.asarray(a,dtype=int) for a in buckets if len(a)]
 if scenario=='domain':return [ids[d[ids]==k] for k in sorted(set(d[ids]))]
 buckets=[[] for _ in range(K)]
 # Stratify jointly on domain and binary class, retaining every source row.
 for dom in sorted(set(d[ids])):
  for lab in (0,1):
   a=r.permutation(ids[(d[ids]==dom)&(y[ids]==lab)])
   if scenario.startswith('quantity'):p=np.array([.7,.2,.1]) if K==3 else np.arange(K,0,-1)/sum(range(1,K+1))
   elif scenario.startswith('label'):p=np.array([.85,.1,.05]) if lab else np.array([.05,.1,.85]);p=p if K==3 else np.roll(np.array([.7,.1,.1,.05,.05]),lab*3)
   else:p=np.full(K,1/K)
   for j,c in enumerate(np.split(a,(np.cumsum(p)[:-1]*len(a)).astype(int))):buckets[j].extend(c)
 return [np.asarray(a,dtype=int) for a in buckets if len(a)]

def score(net,x,ids,device):
 net.eval();out=[]
 with torch.no_grad():
  for a in np.array_split(ids,max(1,math.ceil(len(ids)/4096))):out.append(torch.sigmoid(net(x[a])).cpu().numpy())
 return np.concatenate(out).astype(np.float64)

class MLPBN(nn.Module):
 def __init__(self,n):
  super().__init__();self.net=nn.Sequential(nn.Linear(n,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def fit_bn(X,y,tr,va,clients,cfg,s,opts,participation,t0):
 device=opts['device'];xx=torch.as_tensor(X,device=device);yy=torch.as_tensor(y,dtype=torch.float32,device=device);torch.manual_seed(s);r=rng(s,'fedbn');nets=[MLPBN(X.shape[1]).to(device) for _ in clients];init=nets[0].state_dict()
 for n in nets:n.load_state_dict(init)
 shared=[k for k in init if k.startswith(('net.0.','net.3.','net.6.'))];state={k:init[k].detach().clone() for k in shared};history=[];steps=comm=0;best=-np.inf;best_state=None;best_round=wait=0
 def pred(ids):return np.mean([score(n,xx,ids,device) for n in nets],axis=0)
 def sync():
  for n in nets:
   st=n.state_dict();st.update(state);n.load_state_dict(st)
 for rd in range(1,opts['rounds']+1):
  sync();picked=r.choice(len(clients),max(1,math.ceil(participation*len(clients))),replace=False);w=np.array([len(clients[i]) for i in picked],float);w/=w.sum();updates=[];losses=[]
  for i in picked:
   n=nets[i];n.train();opt=torch.optim.Adam(n.parameters(),lr=cfg['lr']);ids=clients[i]
   for _ in range(opts['local_steps']):
    b=r.choice(ids,max(2,min(opts['batch'],len(ids))),replace=len(ids)<2);opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(n(xx[b]),yy[b]);loss.backward();opt.step();losses.append(float(loss.detach()));steps+=1
   updates.append({k:n.state_dict()[k].detach().clone() for k in shared});comm+=2*sum(v.numel()*v.element_size() for v in state.values())
  state={k:sum(float(a)*u[k] for a,u in zip(w,updates)) for k in shared};sync();row=dict(round=rd,train_loss=float(np.mean(losses)),steps=steps,bytes=comm)
  if rd%opts['check_every']==0 or rd==opts['rounds']:
   v=auc(y[va],pred(va),.05);row['source_stop_pauc05']=v
   if v>best+1e-5:best=v;best_round=rd;best_state=[{k:v.detach().cpu().clone() for k,v in n.state_dict().items()} for n in nets];wait=0
   else:wait+=1
  history.append(row)
  if rd>=opts['min_round'] and wait>=opts['patience']:break
 for n,st in zip(nets,best_state):n.load_state_dict(st)
 return pred,history,dict(seconds=time.perf_counter()-t0,steps=steps,bytes=comm,best_round=best_round,rounds_run=rd,stop_reason='patience' if rd<opts['rounds'] else 'round_cap',source_stop_pauc05=best,client_sizes=[len(i) for i in clients],inference='source BN ensemble; no target BN fitting')

def fit_model(raw,y,d,tr,va,method,cfg,s,opts,scenario='native',participation=1.):
 """Returns fitted predictor, source-only early-stopping history, metadata. Source val only."""
 t0=time.perf_counter();trans=original.transform(raw);sc=StandardScaler().fit(trans[tr]);X=sc.transform(trans).astype(np.float32);X=np.clip(X,-20,20)
 if method=='FedBN_SourceEnsemble':return fit_bn(X,y,tr,va,partition(tr,y,d,s,scenario),cfg,s,opts,participation,t0)
 # Pooled source normalization is an explicit shared-statistics protocol, not private secure aggregation.
 if method in ('LogisticRegression','RandomForest','LightGBM'):
  if method=='LogisticRegression':m=LogisticRegression(C=cfg['C'],max_iter=2000,random_state=s)
  elif method=='RandomForest':m=RandomForestClassifier(n_estimators=300 if not opts['smoke'] else 8,n_jobs=2,random_state=s,**cfg)
  else:m=LGBMClassifier(n_estimators=1500 if not opts['smoke'] else 10,verbosity=-1,n_jobs=2,random_state=s,**cfg)
  if method=='LightGBM':
   import lightgbm as lgb
   m.fit(X[tr],y[tr],eval_set=[(X[va],y[va])],eval_metric='binary_logloss',callbacks=[lgb.early_stopping(50,verbose=False)])
  else:m.fit(X[tr],y[tr])
  return lambda ids:m.predict_proba(X[ids])[:,1].astype(np.float64),[],dict(seconds=time.perf_counter()-t0,steps=None,bytes=0,stop_reason='classical_fit',best_iteration=getattr(m,'best_iteration_',None),iterations=getattr(m,'n_iter_',None))
 device=opts['device'];xx=torch.as_tensor(X,device=device);yy=torch.as_tensor(y,dtype=torch.float32,device=device);torch.manual_seed(s);r=rng(s,'train');model=original.MLP(X.shape[1]).to(device);init={k:v.detach().clone() for k,v in model.state_dict().items()};clients=partition(tr,y,d,s,scenario);steps=0;comm=0;history=[];best=-np.inf;best_state=None;best_round=0;wait=0
 rounds=opts['rounds'];local_steps=opts['local_steps'];check=opts['check_every'];patience=opts['patience'];min_round=opts['min_round'];B=opts['batch'];lr=cfg['lr'];criterion=nn.BCEWithLogitsLoss();
 def draw(ids):return r.choice(ids,min(B,len(ids)),replace=False)
 def local_update(net,opt,ids,ref=None,control=None):
  b=draw(ids);opt.zero_grad();loss=criterion(net(xx[b]),yy[b]);
  if method=='FedProx' and ref is not None:loss=loss+.5*cfg['mu']*sum((p-q).square().sum() for p,q in zip(net.parameters(),ref))
  loss.backward();
  if control is not None:
   for p,c in zip(net.parameters(),control):p.grad.add_(c)
  opt.step();return float(loss.detach())
 if method=='LocalMLP':
  nets=[original.MLP(X.shape[1]).to(device) for _ in clients];[n.load_state_dict(init) for n in nets];optim=[torch.optim.Adam(n.parameters(),lr=lr) for n in nets]
 else:nets=[];optim=[]
 if method in ('PooledMLP','GroupDRO','CORAL','DANN_Source'):
  adversary=nn.Linear(32,len(clients)).to(device) if method=='DANN_Source' else None
  opt=torch.optim.Adam(list(model.parameters())+(list(adversary.parameters()) if adversary is not None else []),lr=lr)
 momentum={k:torch.zeros_like(v) for k,v in init.items()};second={k:torch.zeros_like(v) for k,v in init.items()};cglobal=[torch.zeros_like(p) for p in model.parameters()];clocal=[[torch.zeros_like(p) for p in model.parameters()] for _ in clients];groupw=np.ones(len(clients))/len(clients)
 for rd in range(1,rounds+1):
  losses=[]
  if method=='LocalMLP':
   for n,o,ids in zip(nets,optim,clients):
    n.train()
    for _ in range(local_steps):losses.append(local_update(n,o,ids));steps+=1
  elif method in ('PooledMLP','GroupDRO','CORAL','DANN_Source'):
   model.train()
   for _ in range(local_steps*len(clients)):
    if method=='PooledMLP':losses.append(local_update(model,opt,tr));steps+=1;continue
    opt.zero_grad();b=[draw(i) for i in clients];ls=torch.stack([criterion(model(xx[i]),yy[i]) for i in b]);loss=ls.mean()
    if method=='GroupDRO':
     groupw*=np.exp(np.clip(.05*ls.detach().cpu().numpy(),-10,10));groupw/=groupw.sum();loss=(ls*torch.tensor(groupw,device=device,dtype=ls.dtype)).sum()
    if method=='DANN_Source':
     # Gradient reversal: forward identity; reversed encoder gradient, ordinary domain-head gradient.
     h=torch.cat([model.net[:-1](xx[i]) for i in b]);rev=2*h.detach()-h;labs=torch.cat([torch.full((len(ids),),j,device=device,dtype=torch.long) for j,ids in enumerate(b)]);loss=loss+.1*nn.functional.cross_entropy(adversary(rev),labs)
    if method=='CORAL':
     # Align hidden representations in pooled source training; no target data.
     h=[model.net[:-1](xx[i]) for i in b];cov=[(v-v.mean(0)).T@(v-v.mean(0))/max(1,len(v)-1) for v in h];pairs=list(itertools.combinations(cov,2));loss=loss+(.1*torch.stack([(a-b).square().mean() for a,b in pairs]).mean() if pairs else 0.)
    loss.backward();opt.step();losses.append(float(loss.detach()));steps+=1
  else:
   state={k:v.detach().clone() for k,v in model.state_dict().items()};picked=r.choice(len(clients),max(1,math.ceil(participation*len(clients))),replace=False);weights=np.array([len(clients[i]) for i in picked],float);weights/=weights.sum();deltas=[];counts=[];dc=[]
   for i in picked:
    ids=clients[i];local=original.MLP(X.shape[1]).to(device);local.load_state_dict(state);local.train();optimizer=torch.optim.SGD(local.parameters(),lr=lr) if method in ('SCAFFOLD','FedNova') else torch.optim.Adam(local.parameters(),lr=lr);ref=[p.detach().clone() for p in local.parameters()];nsteps=local_steps if method!='FedNova' else max(1,min(4*local_steps,round(local_steps*len(ids)/np.mean([len(j) for j in clients]))));control=[a-b for a,b in zip(cglobal,clocal[i])] if method=='SCAFFOLD' else None
    for _ in range(nsteps):losses.append(local_update(local,optimizer,ids,ref,control));steps+=1
    dd={k:v.detach()-state[k] for k,v in local.state_dict().items()};deltas.append(dd);counts.append(nsteps);comm+=2*sum(v.numel()*v.element_size() for v in state.values())
    if method=='SCAFFOLD':
     new=[old-g+(q-p.detach())/(nsteps*lr) for old,g,q,p in zip(clocal[i],cglobal,ref,local.parameters())];dc.append([a-b for a,b in zip(new,clocal[i])]);clocal[i]=new;comm+=2*sum(v.numel()*v.element_size() for v in cglobal)
   if method=='SCAFFOLD':
    for j,g in enumerate(cglobal):g.add_(sum(v[j] for v in dc)/len(clients))
   avg={k:sum(float(w)*a[k] for w,a in zip(weights,deltas)) for k in state}
   if method=='FedNova':avg={k:float(np.dot(weights,counts))*sum(float(w/n)*a[k] for w,n,a in zip(weights,counts,deltas)) for k in state}
   if method in ('FedAdam','FedYogi'):
    for k,g in avg.items():
     momentum[k]=.9*momentum[k]+.1*g;second[k]=.99*second[k]+.01*g.square() if method=='FedAdam' else second[k]-.01*g.square()*torch.sign(second[k]-g.square());avg[k]=cfg['server_lr']*(momentum[k]/(1-.9**rd))/((second[k]/(1-.99**rd)).clamp_min(0).sqrt()+cfg['tau'])
   model.load_state_dict({k:state[k]+avg[k] for k in state})
  if not np.isfinite(losses).all():raise FloatingPointError('nonfinite loss')
  row=dict(round=rd,train_loss=float(np.mean(losses)),steps=steps,bytes=comm)
  if rd%check==0 or rd==rounds:
   pred=np.mean([score(n,xx,va,device) for n in nets],axis=0) if nets else score(model,xx,va,device);v=auc(y[va],pred,.05);row['source_stop_pauc05']=v
   if v>best+1e-5:
    best=v;best_round=rd;best_state=[{k:v.detach().cpu().clone() for k,v in n.state_dict().items()} for n in (nets or [model])];wait=0
   else:wait+=1
  history.append(row)
  if rd>=min_round and wait>=patience:break
 for n,st in zip(nets or [model],best_state):n.load_state_dict(st)
 return (lambda ids:np.mean([score(n,xx,ids,device) for n in nets],axis=0)) if nets else (lambda ids:score(model,xx,ids,device)),history,dict(seconds=time.perf_counter()-t0,steps=steps,bytes=comm,best_round=best_round,rounds_run=rd,stop_reason='patience' if rd<rounds else 'round_cap',source_stop_pauc05=best,client_sizes=[len(i) for i in clients])

def bundle(work,raw,y,d,f,held,s,track,method,opts,scenario='native',participation=1.,chosen=None,split_override=None):
 timegate();fit,stop,cal,src_test,test=split_override if split_override is not None else split_outer(y,d,f,held,s,track);key=hashlib.sha256(repr((track,held,s,method,scenario,participation)).encode()).hexdigest()[:20];folder=work/'bundles'/key;folder.mkdir(parents=True,exist_ok=True)
 identity=dict(track=track,held=list(held),seed=s,method=method,scenario=scenario,participation=participation,split_digests={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in zip(['fit','stop','cal','source_test','target'],(fit,stop,cal,src_test,test))});meta=folder/'done.json'
 if meta.exists() and (folder/'scores.npz').exists():
  obj=json.loads(meta.read_text());assert obj['identity']==identity;assert obj['score_sha256']==sha(folder/'scores.npz');return obj['config'],folder
 if chosen is None:
  candidates=grid(method)[:opts['trials']];rows=[]
  domains=sorted(set(d[fit]));inner=[]
  if track!='edge' and len(domains)>=2:
   for h in domains:
    a=fit[d[fit]!=h];v=stop[d[stop]!=h];b=np.r_[fit[d[fit]==h],stop[d[stop]==h]];inner.append((h,a,v,b))
  else:
   if track=='temporal':
    k=int(len(fit)*.75);a,b=fit[:k],fit[k:]
   else:a,b=groupsplit(fit,y,seed(s,'inner'),.25)
   inner=[('source_validation',a,b,stop)]
  for j,cfg in enumerate(candidates):
   for h,a,v,b in inner:
    p=folder/f'trial_{j}_{hashlib.sha1(str(h).encode()).hexdigest()[:8]}.json'
    if p.exists():r=json.loads(p.read_text())
    else:
     timegate()
     pred,curve,info=fit_model(raw,y,d,a,v,method,cfg,seed(s,h,j*0),opts,scenario,participation);val=pred(b);r=dict(candidate=j,inner_holdout=h,pauc05=auc(y[b],val,.05),pauc01=auc(y[b],val,.01),auroc=auc(y[b],val),config=cfg,**info);frame(p.with_suffix('.csv'),curve);save(p,r)
    rows.append(r)
  chosen_id=int(pd.DataFrame(rows).groupby('candidate').pauc05.mean().sort_values(ascending=False,kind='stable').index[0]);chosen=candidates[chosen_id];frame(folder/'tuning.csv',rows)
  if track!='edge' and len(domains)>=2:
   inner_rows=[]
   for h,a,v,b in inner:
    timegate();cc=cal[d[cal]!=h];ee=src_test[d[src_test]==h]
    pred0,_,_=fit_model(raw,y,d,a,v,method,chosen,seed(s,h,'inner_control'),opts,scenario,participation);ss0=pred0(cc);ss1=pred0(ee)
    for alpha in RATES:
     n0=ss0[y[cc]==0];n1=ss1[y[ee]==0];min_n=math.ceil(math.log(.05)/math.log(1-alpha));
     if min(len(n0),len(n1))<min_n and not opts['smoke']:continue
     t0=cut(n0,alpha);ff,rr,_,_=evalop(y[ee],ss1,t0);inner_rows.append(dict(track=track,held=' + '.join(held),seed=s,method=method,inner_holdout=h,alpha=alpha,fpr=ff,recall=rr,calibration_benign=len(n0),source_test_benign=len(n1)))
   frame(folder/'inner_source_control.csv',inner_rows)
 timegate();pred,curve,info=fit_model(raw,y,d,fit,stop,method,chosen,seed(s,'final'),opts,scenario,participation)
 # Target predictions are produced only after source-only selection is frozen.
 scores={};parts={'fit':fit,'stop':stop,'cal':cal,'source_test':src_test,'target':test}
 for role,ids in parts.items():
  scores.update({role+'_ids':ids,role+'_y':y[ids],role+'_d':d[ids].astype(str),role+'_family':f[ids].astype(str),role+'_scores':pred(ids)})
  assert np.isfinite(scores[role+'_scores']).all()
 npwrite(folder/'scores.npz',**scores);frame(folder/'learning.csv',curve);save(meta,dict(identity=identity,config=chosen,score_sha256=sha(folder/'scores.npz'),**info));print('DONE',track,held,s,method,scenario,participation,flush=True);return chosen,folder

def analyze(folder,opts,work):
 if (folder/'analysis_complete.json').exists():return
 obj=json.loads((folder/'done.json').read_text());a=np.load(folder/'scores.npz');i=obj['identity'];base={**i,'held':' + '.join(i['held'])};base.pop('split_digests');rows=[];policies=[];few=[];prev=[];alerts=[];families=[];audit=[]
 cy,cs,cd=a['cal_y'],a['cal_scores'],a['cal_d'];sy,ss,sd=a['source_test_y'],a['source_test_scores'],a['source_test_d'];ty,ts,td,tf=a['target_y'],a['target_scores'],a['target_d'],a['target_family'];neg=cs[cy==0];domneg=[neg[cd[cy==0]==j] for j in sorted(set(cd[cy==0]))];
 # Source-fitted Platt calibration; calibration and test roles remain separate.
 pl=LogisticRegression(max_iter=1000).fit(cs.reshape(-1,1),cy);pp=pl.predict_proba(ts.reshape(-1,1))[:,1];bins=np.minimum((pp*15).astype(int),14);ece=sum(np.mean(bins==k)*abs(pp[bins==k].mean()-ty[bins==k].mean()) for k in set(bins));
 for dom in sorted(set(td)):
  sel=td==dom;y=ty[sel];s=ts[sel];n=s[y==0];p=s[y==1];assert len(n) and len(p)
  common=base|dict(target_domain=dom,auroc=auc(y,s),pauc01=auc(y,s,.01),pauc05=auc(y,s,.05),ap=float(average_precision_score(y,s)),score_inversion=auc(y,s)<.5,unique_scores=len(np.unique(s)),largest_tie_fraction=np.unique(s,return_counts=True)[1].max()/len(s),brier=float(brier_score_loss(ty,pp)),ece=ece)
  for alpha in RATES:
   min_n=math.ceil(math.log(.05)/math.log(1-alpha));adequate=len(neg)>=min_n and len(n)>=min_n
   if not adequate and not opts['smoke']:
    rows.append(common|dict(alpha=alpha,status='insufficient_tail_sample',cal_n=len(neg),target_benign_n=len(n),min_n=min_n));continue
   t=cut(neg,alpha);oracle=cut(n,alpha);fpr,rec,fp,tp=evalop(y,s,t);of,orr,_,_=evalop(y,s,oracle);sf,sr,_,_=evalop(sy,ss,t);N=len(n);lo=0 if fp==0 else beta.ppf(.025,fp,N-fp+1);hi=1 if fp==N else beta.ppf(.975,fp+1,N-fp)
   rows.append(common|dict(alpha=alpha,status='evaluated',threshold=t,target_fpr=fpr,recall=rec,fp=fp,tp=tp,cal_n=len(neg),target_benign_n=N,target_attack_n=len(p),source_test_fpr=sf,source_test_recall=sr,oracle_fpr=of,oracle_recall=orr,oracle_minus_locked_recall=orr-rec,fpr_ratio=fpr/alpha,flow_iid_binom_p=binomtest(fp,N,alpha,alternative='greater').pvalue,fpr_ci_low=lo,fpr_ci_high=hi,nominal_excess=fpr>alpha))
   # Three correction policies plus a candidate federated risk-aware tolerance rule.
   # FedRAT prototype uses per-client tolerance bounds, Bonferroni over source clients.
   # It is a source-side heuristic, not a distribution-free guarantee for unseen domains.
   rules={'empirical_pooled':t,'pooled_conformal':conformal(neg,alpha),'worst_source_conformal':max(conformal(v,alpha) for v in domneg),'minimax_empirical':max(cut(v,alpha) for v in domneg),'FedRAT_tolerance_prototype':max(tolcut(v,alpha,.05/len(domneg)) for v in domneg)}
   for name,thr in rules.items():
    f0,r0,_,_=evalop(sy,ss,thr);f1,r1,_,_=evalop(y,s,thr);policies.append(common|dict(alpha=alpha,policy=name,threshold=thr,abstain_all=not np.isfinite(thr),source_test_fpr=f0,source_test_recall=r0,target_fpr=f1,recall=r1,quantile_payload_bytes=(8*len(domneg) if name.startswith(('worst','FedRAT','minimax')) else None)))
   for source in sorted(set(sd)):
    q=sd==source;val=evalop(sy[q],ss[q],t);audit.append(common|dict(alpha=alpha,source_domain=source,source_test_fpr=val[0],source_test_recall=val[1]))
   if alpha==.01:
    for family in sorted(set(tf[sel][y==1])):
     v=s[(y==1)&(tf[sel]==family)];families.append(common|dict(attack_family=family,n=len(v),locked_recall=float((v>=t).mean()),oracle_recall=float((v>=oracle).mean())))
    for rep in range(5 if not opts['smoke'] else 1):
     v=rng(i['held'],i['seed'],dom,rep,'few').permutation(n);cap=min(2000,len(n)//2);pool=v[:cap];test=v[cap:]
     for count in [0,100,250,500,1000,2000]:
      if count>cap:continue
      thr=t if count==0 else conformal(pool[:count],alpha);few.append(common|dict(repeat=rep,n_target_cal=count,target_fpr=float((test>=thr).mean()),recall=float((p>=thr).mean()),target_test_benign=len(test),target_informed=count>0))
    for prevalence in [.001,.005,.01,.05,.1]:
     den=prevalence*rec+(1-prevalence)*fpr;want=round(len(n)*prevalence/(1-prevalence));want=min(want,len(p));sample=rng(i['held'],i['seed'],dom,prevalence,'prevalence').choice(p,want,replace=False);ppv=float((sample>=t).sum()/max(1,(n>=t).sum()+(sample>=t).sum()));prev.append(common|dict(prevalence_assumed=prevalence,ppv_formula=prevalence*rec/den if den else 0,alerts_per_100k=100000*den,subsample_attack_n=want,realized_sample_prevalence=want/(want+len(n)),subsample_ppv=ppv))
    for budget in [50,100,500,1000]:
     # source-only assumed prevalence weighted tail; deterministic ties, no target selection.
     pi=.01;unique=np.unique(cs);sortneg=np.sort(cs[cy==0]);sortpos=np.sort(cs[cy==1]);rates=(1-pi)*(1-np.searchsorted(sortneg,unique,side='right')/len(sortneg))+pi*(1-np.searchsorted(sortpos,unique,side='right')/len(sortpos));idx=np.flatnonzero(rates<=budget/100000);j=idx[0] if len(idx) else len(unique)-1;thr=float(np.nextafter(unique[j],np.inf));f0,r0,_,_=evalop(y,s,thr);alerts.append(common|dict(budget=budget,assumed_prevalence=pi,threshold=thr,source_expected_alerts=100000*rates[j],target_expected_alerts=100000*((1-pi)*f0+pi*r0),target_fpr=f0,recall=r0,source_tie_count=int((cs==unique[j]).sum())))
 for name,v in [('operating',rows),('policies',policies),('few_shot',few),('prevalence',prev),('alert_budget',alerts),('attack_family',families),('source_control',audit)]:
  if v:frame(folder/(name+'.csv'),v)
 save(folder/'analysis_complete.json',{'scores_sha256':obj['score_sha256']})

def feature_shift(raw,y,d,held,s,work,fit,test):
 if len(set(d[test]))<1:return
 rows=[];X=original.transform(raw)
 for h in sorted(set(d[test])):
  a=fit[y[fit]==0];b=test[(y[test]==0)&(d[test]==h)];n=min(2000,len(a),len(b));r=rng(held,s,h,'shift');ix=np.r_[r.choice(a,n,replace=False),r.choice(b,n,replace=False)];lab=np.r_[np.zeros(n),np.ones(n)];tr,te=train_test_split(np.arange(len(ix)),test_size=.5,stratify=lab,random_state=s);sc=StandardScaler().fit(X[ix[tr]]);model=LogisticRegression(max_iter=1000).fit(sc.transform(X[ix[tr]]),lab[tr]);rows.append(dict(held=' + '.join(held),seed=s,target_domain=h,domain_auc=auc(lab[te],model.predict_proba(sc.transform(X[ix[te]]))[:,1])))
 frame(work/'shift'/f'{seed(held,s)}.csv',rows)

def export(work,opts,status):
 summary=work/'summary';summary.mkdir(exist_ok=True)
 for name in ['operating','policies','few_shot','prevalence','alert_budget','attack_family','source_control','inner_source_control','tuning']:
  paths=list((work/'bundles').glob('*/'+name+'.csv'));dfs=[pd.read_csv(p).assign(bundle=p.parent.name) for p in paths if p.stat().st_size>1];
  if not dfs:continue
  df=pd.concat(dfs,ignore_index=True);df.to_csv(summary/(name+'.csv'),index=False)
  if name=='operating':
   good=df[df.status=='evaluated'];cols=['recall','target_fpr','auroc','pauc01','pauc05','oracle_recall','fpr_ratio','source_test_fpr'];keys=['track','scenario','participation','method','held','target_domain','alpha'];g=good.groupby(keys)[cols].agg(['mean','std','count']);g.columns=['_'.join(c) for c in g.columns];g.to_csv(summary/'domain_mean_sd.csv');
   # Domain-level summaries with domain bootstrap, descriptive only (four NF sources).
   groups=['track','scenario','participation','method','alpha'];dm=good.groupby(groups+['target_domain'])[cols].mean().reset_index();out=[]
   for k,g in dm.groupby(groups):
    for c in cols:
     v=g[c].dropna().to_numpy();n=len(v)
     if not n:continue
     b=rng(k,c,'bootstrap').choice(v,(2000,n),replace=True).mean(axis=1);out.append(dict(zip(groups,k))|dict(metric=c,n_domains=n,mean=v.mean(),sd=v.std(ddof=1) if n>1 else np.nan,domain_bootstrap_low=np.quantile(b,.025) if n>1 else np.nan,domain_bootstrap_high=np.quantile(b,.975) if n>1 else np.nan))
   frame(summary/'descriptive_domain_bootstrap.csv',out)
   # Two-stage domain/flow bootstrap approximation: conditional binomial resampling within randomly chosen seed runs.
   # It assumes flow-level independence and is descriptive, not a substitute for capture-cluster inference.
   two=[]
   for k,g in good.groupby(groups):
    ds=[v for _,v in g.groupby('target_domain')];nd=len(ds)
    if nd<2:continue
    R=1000;rr=rng(k,'two_stage');domain_draw=rr.integers(nd,size=(R,nd));
    for metric,ncol in [('target_fpr','target_benign_n'),('recall','target_attack_n')]:
     values=np.empty((R,nd))
     for j,v in enumerate(ds):
      ii,jj=np.where(domain_draw==j);chosen=v.iloc[rr.integers(len(v),size=len(ii))];prob=chosen[metric].to_numpy();nn=chosen[ncol].to_numpy(dtype=int);values[ii,jj]=rr.binomial(nn,prob)/nn
     means=values.mean(1);two.append(dict(zip(groups,k))|dict(metric=metric,n_domains=nd,lower=np.quantile(means,.025),upper=np.quantile(means,.975),assumption='conditional iid flow bootstrap; descriptive'))
   if two:frame(summary/'two_stage_descriptive_bootstrap.csv',two)
   # Holm correction of flow-level p values within each matched protocol/alpha, not network-level inference.
   res=good.copy();res['flow_iid_p_holm']=np.nan
   for _,g in res.groupby(groups):
    p=g.flow_iid_binom_p.to_numpy();order=np.argsort(p);adjusted=np.empty(len(p));adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)));res.loc[g.index,'flow_iid_p_holm']=adjusted
   res.to_csv(summary/'flow_iid_tests.csv',index=False)
   # Score equality audit across methods on identical target IDs; exact tie audit is also in operating.csv.
   equality=[];bundles={}
   for p in (work/'bundles').glob('*/done.json'):
    m=json.loads(p.read_text());i=m['identity'];key=(i['track'],tuple(i['held']),i['seed'],i['scenario'],i['participation']);bundles.setdefault(key,[]).append((i['method'],p.parent/'scores.npz'))
   for k,v in bundles.items():
    loaded=[]
    for ma,pa in v:
     with np.load(pa) as a:loaded.append((ma,a['target_ids'],a['target_scores']))
    for (ma,ia,sa),(mb,ib,sb) in itertools.combinations(loaded,2):
     if np.array_equal(ia,ib):equality.append(dict(track=k[0],held=' + '.join(k[1]),seed=k[2],scenario=k[3],participation=k[4],method_a=ma,method_b=mb,identical_scores=bool(np.array_equal(sa,sb)),max_abs_difference=float(np.max(np.abs(sa-sb)))))
   frame(summary/'score_equality_audit.csv',equality)
 shifts=list((work/'shift').glob('*.csv')); 
 if shifts:pd.concat([pd.read_csv(p) for p in shifts]).to_csv(summary/'feature_shift.csv',index=False)
 status['completed_model_bundles']=len(list((work/'bundles').glob('*/done.json')));status['completed_tuning_trials']=len(list((work/'bundles').glob('*/trial_*.json')));status['completed_analysis_bundles']=len(list((work/'bundles').glob('*/analysis_complete.json')));
 metadata=[]
 for p in (work/'bundles').glob('*/done.json'):
  m=json.loads(p.read_text());i=m['identity'];metadata.append({k:v for k,v in i.items() if k!='split_digests'}|{k:v for k,v in m.items() if k not in ['identity','score_sha256','config']}|{'config':json.dumps(m['config']),'best_at_round_cap':m.get('best_round')==opts['rounds'] if 'best_round' in m else None})
 if metadata:frame(summary/'convergence_and_cost.csv',metadata)
 save(work/'status.json',status);save(work/'environment.json',dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.cuda.is_available(),device=opts['device']))
 # Reproducibility archive excludes temporary files and raw input CSVs, includes score arrays, split IDs, configs and code.
 zpath=work.parent/('FLIDS_Packages10_13_SOFTWARE_TEST_ONLY.zip' if opts['smoke'] else 'FLIDS_Packages10_13_Results.zip');tmp=zpath.with_suffix('.tmp')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
  for p in work.rglob('*'):
   if p.is_file() and p.suffix not in ('.tmp','.pyc') and not p.name.startswith('synthetic_nf'):z.write(p,p.relative_to(work))
 tmp.replace(zpath);return zpath

def lock_optional(work,name,identity):
 p=work/(name+'_input_identity.json')
 if p.exists() and json.loads(p.read_text())!=identity:raise RuntimeError(name+' input changed; use a fresh work folder for that track.')
 save(p,identity)

def load_temporal(work,e):
 features=original.FEATURES;label=e.get('label','Label');family=e.get('family','Attack');tc=e['time_col'];gc=e.get('group_col');cols=features+[label,family,tc]+([gc] if gc else []);assert len(set(cols))==len(cols)
 # Full chronological scan of timestamps and labels only; second pass loads bounded feature sample.
 meta=pd.read_csv(e['path'],usecols=[tc,label]+([gc] if gc else []),low_memory=False)
 times=pd.to_datetime(pd.to_numeric(meta[tc],errors='raise'),unit=e['time_unit'],utc=True,errors='raise') if e.get('time_unit') else pd.to_datetime(meta[tc],utc=True,errors='raise',format='mixed')
 assert times.notna().all() and times.nunique()>=10
 t=times.astype('datetime64[ns, UTC]').astype('int64').to_numpy();y=pd.to_numeric(meta[label],errors='raise').to_numpy();assert set(y)=={0,1}
 order=np.argsort(t,kind='stable');ts=t[order];boundaries=[ts[min(len(ts)-1,int(len(ts)*q))] for q in [.50,.60,.70,.80]];assert len(set(boundaries))==4,'Insufficient distinct chronological boundaries'
 block=np.searchsorted(boundaries,t,side='right');ids=[np.flatnonzero(block==j) for j in range(5)];info={'source_rows':len(t),'boundaries_ns':[int(v) for v in boundaries],'capture_group':gc,'pre_cap_class_counts':[{str(v):int((y[ix]==v).sum()) for v in (0,1)} for ix in ids]}
 if gc:
  g=meta[gc].astype(str);gframe=pd.DataFrame({'group':g,'block':block});bad=gframe.groupby('group').block.nunique();bad=set(bad[bad>1].index);good=~g.isin(bad).to_numpy();ids=[ix[good[ix]] for ix in ids];info['groups_spanning_time_blocks_excluded']=len(bad)
 assert all(set(y[ix])=={0,1} for ix in ids),'A predeclared chronological block lacks a class; supply valid timestamped dataset, do not tune boundaries on target labels.'
 cap=int(e.get('cap_per_class_block',12000));chosen=np.sort(np.concatenate([rng('temporal',j,v).choice(ix[y[ix]==v],min(cap,int((y[ix]==v).sum())),replace=False) for j,ix in enumerate(ids) for v in (0,1)]));selected=[];offset=0
 for chunk in pd.read_csv(e['path'],usecols=features+[label,family],chunksize=100000,low_memory=False):
  lo=np.searchsorted(chosen,offset);hi=np.searchsorted(chosen,offset+len(chunk));
  if hi>lo:selected.append(chunk.iloc[chosen[lo:hi]-offset].copy())
  offset+=len(chunk)
 assert offset==len(t),'Input row count changed between scans';df=pd.concat(selected,ignore_index=True);assert len(df)==len(chosen)
 x=df[features].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64);assert np.isfinite(x).all(),'Nonfinite temporal feature; verify extraction before evaluation';yy=pd.to_numeric(df[label],errors='raise').to_numpy(dtype=int);assert np.array_equal(yy,y[chosen]);f=df[family].astype(str).to_numpy(dtype=str);d=np.full(len(yy),str(e.get('domain','TemporalNetFlow')));tt=t[chosen];bb=block[chosen];sort=np.argsort(tt,kind='stable');x,yy,d,f,tt,bb=x[sort],yy[sort],d[sort],f[sort],tt[sort],bb[sort];k,du=dedup(x,yy,d,f);x,yy,d,f,tt,bb=x[k],yy[k],d[k],f[k],tt[k],bb[k];sp=tuple(np.flatnonzero(bb==j) for j in range(5))
 assert all(set(yy[ix])=={0,1} for ix in sp),'Deduplication removed a class from chronological block';assert all(tt[a].max()<tt[b].min() for a,b in zip(sp,sp[1:]));n=int(len(sp[0])*.75);assert set(yy[sp[0][:n]])=={0,1} and set(yy[sp[0][n:]])=={0,1},'Chronological inner split lacks both classes'
 npwrite(work/'temporal_cache.npz',x=x,y=yy,d=d,family=f,timestamps_ns=tt,block=bb);save(work/'temporal_audit.json',info|du|{'split_sizes':[len(ix) for ix in sp],'capture_independence_established':False,'natural_prevalence_preserved':False});return x,yy,d,f,sp

def prepare_optional(work,opts,raw,y,d,f):
 # Pre-train full data preparation of supplied optional tracks, not just header checks.
 # Successful cached preparations are content-locked; missing or invalid tracks are reported explicitly.
 out={};status={}
 for name in ('edge','external','temporal'):
  configured={'edge':bool(opts.get('edge_csv') and opts.get('edge_features')),'external':bool(opts.get('external_inputs')),'temporal':bool(opts.get('temporal_input'))}[name]
  if not configured:status[name]={'state':'blocked_missing_input'};continue
  try:
   if name=='edge':
    identity=dict(sha=sha(opts['edge_csv']),features=opts['edge_features'],schema=opts.get('edge_schema'),label=opts['edge_label'],family=opts['edge_family'],cap=opts['edge_cap_per_family'])
   elif name=='external':identity=[{k:v for k,v in e.items() if k!='path'}|{'sha':sha(e['path'])} for e in opts['external_inputs']]
   else:
    e=opts['temporal_input'];identity={k:v for k,v in e.items() if k!='path'}|{'sha':sha(e['path'])}
   cached=work/(name+'_cache.npz');lock=work/(name+'_input_identity.json');audit=work/(name+'_audit.json')
   if lock.exists() and json.loads(lock.read_text())!=identity:raise RuntimeError(name+' input changed; start new work directory for this track.')
   if cached.exists() and lock.exists() and audit.exists():
    with np.load(cached,allow_pickle=False) as a:
     vals=tuple(a[k] for k in ('x','y','d','family'))
     if name=='temporal':vals=vals+(tuple(np.flatnonzero(a['block']==j) for j in range(5)),)
   else:
    if name=='edge':vals=load_edge(work,opts)
    elif name=='external':
     arrays=[(raw,y,d,f)]
     for e in opts['external_inputs']:arrays.append(load_external(e['path'],original.FEATURES,e['domain'],e.get('label','Label'),e.get('family','Attack')))
     xr,yr,dr,fr=[np.concatenate([a[j] for a in arrays]) for j in range(4)];assert np.isfinite(xr).all(),'Nonfinite external data';k,info=dedup(xr,yr,dr,fr);vals=(xr[k],yr[k],dr[k],fr[k]);save(audit,info|{'provenance':opts['external_inputs']})
     for dom in set(vals[2]):assert set(vals[1][vals[2]==dom])=={0,1},f'Class missing in {dom}'
    else:vals=load_temporal(work,opts['temporal_input'])
    if name!='temporal':npwrite(cached,x=vals[0],y=vals[1],d=vals[2].astype(str),family=vals[3].astype(str))
    save(lock,identity)
   if name=='edge':assert np.isfinite(vals[0]).all() and len(set(vals[3][vals[1]==1]))==14
   # Validate model-relevant split feasibility for every actual requested family/domain and seed before training.
   seeds=[0] if opts['smoke'] else SEEDS
   if name=='edge':
    for h in sorted(set(vals[3][vals[1]==1])):
     for s in seeds:split_outer(vals[1],vals[2],vals[3],(h,),s,'edge')
   if name=='external':
    for e in opts['external_inputs']:
     for s in seeds:split_outer(vals[1],vals[2],vals[3],(e['domain'],),s,'external')
   out[name]=vals;status[name]={'state':'prepared','rows':len(vals[1]),'features':vals[0].shape[1]}
  except Exception as e:status[name]={'state':'blocked_invalid_input','reason':repr(e)}
  print('PREPARE',name,status[name],flush=True)
 save(work/'optional_preparation_status.json',status)
 if opts.get('require_all_inputs') and any(v['state']!='prepared' for v in status.values()):raise RuntimeError('Full run blocked before model training: see optional_preparation_status.json')
 return out,status

def main(config):
 global DEADLINE
 opts=json.loads(Path(config).read_text());DEADLINE=time.monotonic()+3600*float(opts.get('session_hours',9));work=Path(opts['work']);work.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);opts['device']='cuda' if torch.cuda.is_available() and not opts['smoke'] else 'cpu';seeds=[0] if opts['smoke'] else SEEDS;methods=BASE+EXTRA;status=dict(mode='software_test_only' if opts['smoke'] else 'research',packages={},notes=['All results are exploratory benchmark results. No unseen-domain FPR guarantee. No DP or secure aggregation.','Equal candidate counts, not equal wall-clock time; FedNova and SCAFFOLD use SGD and report different step budgets.','Record-level CIs assume independent flows; only four NF domains; no confirmatory mixed-effects inference.','FedRAT is a tolerance-bound prototype, not established novel or distribution-free under shift. Source-domain groups are not physical clients, and in synthetic allocations they do not correspond to client IDs.','Historical Exp03 and Edge score equality cannot be established without their raw original artifacts; new-run audits only.','FedBN is an explicit source-BN-ensemble adaptation for an unseen site; DANN_Source is a source-only adversarial baseline, not unsupervised target adaptation.']);
 identity={k:v for k,v in opts.items() if k not in ['device','work','exp09_zip','edge_csv','external_inputs','stages','session_hours','edge_features','edge_label','edge_family','edge_cap_per_family','temporal_input','edge_schema','nf_content_sha','require_all_inputs']};identity['source_sha']=sha(Path(__file__));identity['input_exp09_sha']=opts.get('nf_content_sha') if opts.get('exp09_zip') else None
 # Fingerprint optional data contracts separately; paths may change on Kaggle resume, content must match.
 # Optional tracks have independent input locks, so later sessions can attach them without invalidating NF results.
 ident=work/'identity.json'
 if ident.exists() and json.loads(ident.read_text())!=identity:raise RuntimeError('Run identity changed; use a new work directory or restore original config.')
 save(ident,identity);save(work/'config.json',opts);shutil.copy2(__file__,work/'master.py') if Path(__file__).resolve()!=(work/'master.py').resolve() else None
 if (Path(__file__).parent/'engine02.py').resolve()!=(work/'engine02.py').resolve():shutil.copy2(Path(__file__).parent/'engine02.py',work/'engine02.py')
 raw,y,d,f=load_nf(work,opts);domains=sorted(set(d));assert len(domains)==4
 scenarios=[('native',1.),('label',1.)] if opts['smoke'] else [('native',1.),('native',2/3),('mixed',1.),('quantity',1.),('label',1.),('domain',1.),('quantity',2/3),('label',2/3)]
 tuned={};stages=opts['stages'];optional={};optional_status={};holds=[(h,) for h in domains][:1 if opts['smoke'] else len(domains)]
 try:
  if 13 in stages and not opts['smoke']:
   optional,optional_status=prepare_optional(work,opts,raw,y,d,f)
   status['optional_inputs']=optional_status
  for package in (10,11,12,13):
   if package not in stages:status['packages'][str(package)]='not_requested';continue
   if package in (10,11,12):
    for held in holds:
     for s in seeds:
      parts=split_outer(y,d,f,held,s,'nf');
      if package==11:feature_shift(raw,y,d,held,s,work,parts[0],parts[-1])
      for method in methods:
       chosen,folder=bundle(work,raw,y,d,f,held,s,'nf',method,opts);tuned[held,s,method]=chosen
       if package in (11,12):analyze(folder,opts,work)
       if package==11 and method in FL+['LocalMLP']:
        for scenario,part in scenarios[1:]:
         _,p=bundle(work,raw,y,d,f,held,s,'nf',method,opts,scenario,part,chosen);analyze(p,opts,work)
    status['packages'][str(package)]='completed_on_nf'
   else:
    # Every held pair is independent of tuning; re-tune using only remaining sources.
    pairs=list(itertools.combinations(domains,2))[:1 if opts['smoke'] else 6]
    for held in pairs:
     for s in seeds:
      for method in methods:
       _,p=bundle(work,raw,y,d,f,held,s,'nf_leave_two',method,opts);analyze(p,opts,work)
    edge=optional_status.get('edge',{}).get('state','not_supplied');external=optional_status.get('external',{}).get('state','not_supplied');temporal=optional_status.get('temporal',{}).get('state','not_supplied')
    if 'edge' in optional:
     er,ey,ed,ef=optional['edge'];families=sorted(set(ef[ey==1]));
     for h in families:
      for s in seeds:
       for method in methods:
        _,p=bundle(work,er,ey,ed,ef,(h,),s,'edge',method,opts);analyze(p,opts,work)
     edge='completed_all_14_families_'+opts.get('edge_schema','unknown_schema')
    if 'external' in optional:
     xr,yr,dr,fr=optional['external']
     for e in opts['external_inputs']:
      held=(e['domain'],)
      for s in seeds:
       for method in methods:
        _,p=bundle(work,xr,yr,dr,fr,held,s,'external',method,opts);analyze(p,opts,work)
     external='completed_supplied_harmonized_domains'
    if 'temporal' in optional:
     xr,yr,dr,fr,sp=optional['temporal']
     for s in seeds:
      for method in methods:
       _,p=bundle(work,xr,yr,dr,fr,('future',),s,'temporal',method,opts,split_override=sp);analyze(p,opts,work)
     temporal='completed_supplied_chronological_split'
    status['packages']['13']=dict(nf_leave_two='completed',edge=edge,external=external,temporal=temporal)
   save(work/'status.json',status);print('PACKAGE',package,status['packages'][str(package)],flush=True)
  status['state']='completed_requested_supported_stages';status['all_requested_data_tracks_completed']=bool(13 in stages and all(str(status['packages']['13'].get(k,'')).startswith('completed') for k in ('nf_leave_two','edge','external','temporal')));status['all_reviewer_requirements_completed']=False;status['unresolved']=['Historical score artifacts not supplied','Temporal/external/Edge completion depends on supplied data contracts','Only descriptive domain inference; no reliable mixed-effects inference with four NF domains','Algorithm novelty and bibliographic verification require independent review']
 except PauseRun as e:
  status['state']='paused_for_resume';status['reason']=str(e);save(work/'status.json',status)
 except BaseException as e:
  status['state']='interrupted_or_failed';status['error']=repr(e);save(work/'status.json',status);raise
 finally:
  # On normal completion or caught error, export partial state; Kaggle hard kill may prevent export.
  out=export(work,opts,status);print('ARCHIVE',out,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
