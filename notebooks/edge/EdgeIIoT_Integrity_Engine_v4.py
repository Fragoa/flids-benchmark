"""Edge-IIoTset dataset integrity and feature sufficiency benchmark.
Research diagnostic: simulated FL, grouped splits, capped sampling, no claims about deployment prevalence.
English-only, standalone, Kaggle or local. No hidden test-set tuning.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS'):
 os.environ[_key]='1'
import faulthandler
faulthandler.enable(all_threads=True)
import sys,time,json,zipfile,traceback,random,argparse,re,hashlib,gc,csv,itertools
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve
from sklearn.model_selection import GroupShuffleSplit

ROOT=Path(os.getenv('EDGE_AUDIT_ROOT','/kaggle/working' if Path('/kaggle/working').is_dir() else '/mnt/data/edge_integrity_local'))
OUT=ROOT/'EdgeIIoT_Integrity_Results'; OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'edge_integrity_cache_v4.pkl'
CFG=dict(seeds=[0,42,3407],families=['Port_Scanning','DDoS_UDP','SQL_injection','Ransomware'],
 n_clients=5,rounds=5,steps=8,batch_size=128,lr=0.0015,hidden=24,embedding=12,bases=5,
 max_scan_rows=None,audit_sample_probability=0.16,chunk_size=8000,pool_cap_per_family=5000,benign_cap=14000,
 threshold_targets=[0.01,0.05],dirichlet_alpha=0.5,
 feature_sets=['safe30','expanded_numeric'],dedup_policies=['none','full_signature','safe30_signature'],
 models=['FedAvgMLP','FedAvgKAN'],archive='EdgeIIoT_Integrity_Results.zip')
if os.getenv('EDGE_AUDIT_SMOKE')=='1':
 CFG.update(seeds=[0],families=['Port_Scanning','SQL_injection'],n_clients=3,rounds=1,steps=2,
            pool_cap_per_family=500,benign_cap=700,max_scan_rows=None,audit_sample_probability=1.0)
if os.getenv('EDGE_AUDIT_SEEDS'):
 CFG['seeds']=[int(x) for x in os.environ['EDGE_AUDIT_SEEDS'].split(',')]

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda v: v.item() if isinstance(v,np.generic) else str(v)),encoding='utf8')
def seedall(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(min(2,os.cpu_count() or 2))
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def norm(x):return ''.join(c for c in str(x).lower() if c.isalnum())
SAFE=['arp.opcode','arp.hw.size','icmp.seq_le','icmp.unused','http.content_length','http.tls_port','tcp.ack','tcp.connection.fin','tcp.connection.rst','tcp.connection.syn','tcp.connection.synack','tcp.flags','tcp.flags.ack','tcp.len','udp.time_delta','dns.qry.qu','dns.qry.type','dns.retransmission','dns.retransmit_request','dns.retransmit_request_in','mqtt.conflag.cleansess','mqtt.conflags','mqtt.hdrflags','mqtt.len','mqtt.msgtype','mqtt.proto_len','mqtt.topic_len','mqtt.ver','mbtcp.len','mbtcp.unit_id']
# Prespecified excluded ID/label or direct attack marker fields: avoid using target proxies.
BAD=['attack','label','class','category','srcip','dstip','ip.src','ip.dst','ip.src_host','ip.dst_host','sourceip','destinationip','timestamp','frame.time','frame.number','flow.id','flowid','uid','mac.src','mac.dst','eth.src','eth.dst','http.file_data','http.request.full_uri','http.request.uri','http.host','http.user_agent','http.cookie','dns.qry.name','mqtt.topic','mqtt.msg','http.request.method_raw','http.response.phrase']


def standard_csv_header(path):
 """Read the column header without invoking pandas' native CSV parser."""
 with open(path,'r',encoding='utf-8-sig',errors='replace',newline='') as stream:
  reader=csv.reader(stream,strict=False)
  return next(reader)

def safe_csv_chunks(path,columns,chunk_size,row_limit=None):
 """Stream selected fields using Python's standard-library CSV reader.
 Malformed-width rows are skipped and counted; no pandas C/pyarrow CSV parser is used.
 """
 csv.field_size_limit(10_000_000)
 selected=set(columns)
 good=0;bad=0;count=0
 with open(path,'r',encoding='utf-8-sig',errors='replace',newline='',buffering=1024*1024) as stream:
  reader=csv.reader(stream,strict=False)
  header=next(reader)
  locations=[i for i,c in enumerate(header) if c in selected]
  names=[header[i] for i in locations]
  if len(names)!=len(selected):raise ValueError(f'Missing CSV fields: {selected-set(names)}')
  batch=[]
  for row in reader:
   count+=1
   if row_limit is not None and count>row_limit:break
   if len(row)!=len(header):
    bad+=1
    if bad<=5:print('CSV_MALFORMED_WIDTH',count,'columns',len(row),'expected',len(header),flush=True)
    continue
   batch.append([row[i] for i in locations])
   good+=1
   if len(batch)>=chunk_size:
    yield pd.DataFrame(batch,columns=names)
    batch=[]
  if batch:yield pd.DataFrame(batch,columns=names)
 print('CSV_READER_COMPLETE valid_rows=',good,'malformed_rows=',bad,'raw_rows_seen=',count,flush=True)

def locate():
 if os.getenv('EDGE_AUDIT_SMOKE')=='1':
  p=ROOT/'synthetic_integrity_smoke_only.csv'
  if not p.exists():
   rng=np.random.default_rng(71);blocks=[]
   for cls,n,shift in [('Normal',1300,0),('Port_Scanning',520,2),('DDoS_UDP',480,1.1),('SQL_injection',430,.35),('Ransomware',420,.2),('Backdoor',490,1)]:
    x=rng.normal(size=(n,15));x[:,0]+=shift;x[:,1]+=shift/2
    a=pd.DataFrame(x,columns=SAFE[:15]);a['http.response.code']=(rng.integers(200,505,n)).astype(float);a['Attack_type']=cls;a['Attack_label']=int(cls!='Normal')
    blocks.append(a)
   d=pd.concat(blocks,ignore_index=True);d=pd.concat([d,d.sample(200,random_state=3)],ignore_index=True)
   # Exact non-label feature conflict to exercise label conflict detector.
   d.loc[len(d)]=d.iloc[1].copy();d.loc[len(d)-1,'Attack_type']='SQL_injection';d.loc[len(d)-1,'Attack_label']=1
   d.to_csv(p,index=False)
  return p,'synthetic smoke data'
 roots=[Path(os.environ['EDGE_IIOT_CSV'])] if os.getenv('EDGE_IIOT_CSV') else []
 roots.extend([Path('/kaggle/input'),ROOT/'dataset'])
 for r in roots:
  candidates=[r] if r.is_file() else list(r.rglob('*.csv')) if r.is_dir() else []
  for p in sorted(candidates,key=lambda a:(0 if 'dnn-edgeiiot' in a.name.lower() else 1,len(str(a)))):
   try:
    h={norm(x) for x in standard_csv_header(p)}
    if 'attacktype' in h and 'attacklabel' in h:return p,'attached/local'
   except Exception:pass
 try:
  import kagglehub
  r=Path(kagglehub.dataset_download('mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot'))
  for p in sorted(r.rglob('*.csv'),key=lambda a:(0 if 'dnn-edgeiiot' in a.name.lower() else 1,len(str(a)))):
   try:
    h={norm(x) for x in standard_csv_header(p)}
    if 'attacktype' in h and 'attacklabel' in h:return p,'kagglehub author dataset'
   except Exception:pass
 except Exception as ex: print('KaggleHub fallback unavailable:',str(ex)[:200],flush=True)
 raise FileNotFoundError('Attach author Edge-IIoTset ML/DNN CSV under Kaggle Add Input or enable Internet/KaggleHub.')

def find_cols(head):
 lab=next(x for x in head if norm(x)=='attacktype');binary=next(x for x in head if norm(x)=='attacklabel')
 nonlabel=[x for x in head if x not in (lab,binary) and not any(b in x.lower() for b in ['attack_type','attack_label'])]
 safe=[x for x in nonlabel if norm(x) in {norm(k) for k in SAFE}]
 if len(safe)<8:raise ValueError('Insufficient Safe-30 fields in Edge-IIoTset schema')
 return lab,binary,nonlabel,safe

def numeric_schema(sample,nonlabel,safe):
 expanded=[];excluded=[]
 for c in nonlabel:
  n=norm(c)
  if any(norm(b) in n for b in BAD): excluded.append(c);continue
  s=pd.to_numeric(sample[c],errors='coerce')
  if s.notna().mean()>=.80 and s.nunique(dropna=True)>=2:expanded.append(c)
  else:excluded.append(c)
 expanded=list(dict.fromkeys(safe+expanded))
 return expanded,excluded

def prepare():
 print('PREPARE_BEGIN: full-file class-stratified streaming mode',flush=True)
 ROOT.mkdir(parents=True,exist_ok=True)
 source,origin=locate();head=standard_csv_header(source)
 lab,binary,nonlabel,safe=find_cols(head)
 sample=next(safe_csv_chunks(source,head,1500,row_limit=1500))
 expanded,excluded=numeric_schema(sample,nonlabel,safe)
 print('PREPARE source',str(source),'safe=',len(safe),'expanded=',len(expanded),'all=',len(nonlabel),'scan_limit=',CFG['max_scan_rows'],'audit_sample_probability=',CFG['audit_sample_probability'],flush=True)
 # Audit signatures on a random subset across the full stream; retain per-class pool across the entire source.
 # "full" hashes every non-label source column; "safe" hashes ONLY numeric Safe-30 values.
 # Hash collisions are astronomically unlikely but not formally impossible.
 full_h=[];safe_h=[];family_ids=[];binary_ids=[];raw_scanned=0
 pool={}; rng=np.random.default_rng(1417)
 use=list(dict.fromkeys(nonlabel+[lab,binary]))
 for j,chunk in enumerate(safe_csv_chunks(source,use,CFG['chunk_size'],row_limit=CFG['max_scan_rows'])):
  chunk=chunk.copy()
  chunk[lab]=chunk[lab].astype(str).str.strip().str.replace(' ','_',regex=False)
  chunk[binary]=pd.to_numeric(chunk[binary],errors='coerce')
  chunk=chunk.loc[chunk[binary].isin([0,1])].copy()
  if chunk.empty:continue
  raw_scanned+=len(chunk)
  # Avoid materializing a second, huge, stringified copy of all columns.
  # Hash the parsed columns directly; audit signatures are approximate 64-bit hashes.
  hs=pd.util.hash_pandas_object(chunk[nonlabel],index=False).to_numpy(dtype='uint64',copy=True)
  sn=chunk[safe].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
  hsafe=pd.util.hash_pandas_object(sn.fillna(-999999.0),index=False).to_numpy('uint64')
  # Audit is a random subset covering the entire file, not a potentially single-class prefix.
  audit_take=rng.random(len(chunk))<CFG['audit_sample_probability']
  if audit_take.any():
   full_h.append(hs[audit_take]);safe_h.append(hsafe[audit_take])
   family_ids.extend(chunk.loc[audit_take,lab].tolist())
   binary_ids.extend(chunk.loc[audit_take,binary].astype('uint8').tolist())
  # Per-family priority reservoir with a fixed capacity. Does not decide dedup policy.
  priorities=rng.random(len(chunk))
  piece=chunk[expanded+[lab,binary]].copy();piece['_full_sig']=hs;piece['_safe_sig']=hsafe;piece['_priority']=priorities
  for family,g in piece.groupby(lab,sort=False):
   cap=CFG['benign_cap'] if int(g[binary].iloc[0])==0 else CFG['pool_cap_per_family']
   cat=pd.concat([pool[family],g],ignore_index=True) if family in pool else g
   pool[family]=cat.nsmallest(cap,'_priority').reset_index(drop=True)
  if (j+1)%5==0:
   print('SCAN rows',raw_scanned,'chunks',j+1,'pool_rows',sum(map(len,pool.values())),'families',len(pool),flush=True)
  if (j+1)%10==0:gc.collect()
  del chunk,piece
 if not pool: raise RuntimeError('No valid samples read from CSV. Check format/encoding.\n')
 data=pd.concat(pool.values(),ignore_index=True).drop(columns=['_priority'])
 data=data.rename(columns={lab:'family',binary:'binary_label'})
 counts=data.family.value_counts().to_dict()
 print('PREPARE_FAMILY_COUNTS',counts,flush=True)
 missing=[f for f in CFG['families'] if counts.get(f,0)<25]
 if missing:raise RuntimeError('DATASET_COVERAGE_ERROR: missing or insufficient heldout attack families: '+str(missing)+'; sampled='+str(counts))
 if int((data.binary_label==0).sum())<100:raise RuntimeError('DATASET_COVERAGE_ERROR: insufficient benign examples')
 print('HASH_SCAN_COMPLETE: building compact signature audit',raw_scanned,flush=True)
 # Store integer-coded family labels instead of millions of Python strings.
 uniqfam,encodedfam=np.unique(np.asarray(family_ids,dtype=str),return_inverse=True)
 raw_fam=pd.DataFrame(dict(sig=np.concatenate(full_h),safe=np.concatenate(safe_h),family=encodedfam.astype(np.int16),binary=np.asarray(binary_ids,dtype=np.uint8)))
 del family_ids,binary_ids,full_h,safe_h,encodedfam
 gc.collect()
 audits=[]
 for sigcol,label in [('sig','full_signature'),('safe','safe30_signature')]:
  print('AUDITING_SIGNATURES',label,flush=True)
  g=raw_fam.groupby(sigcol,sort=False).agg(n=('binary','size'),bin_unique=('binary','nunique'),family_unique=('family','nunique'))
  coll=g[g.n>1];conf=g[g.bin_unique>1];famconf=g[g.family_unique>1]
  audits.append(dict(signature=label,audited_rows=len(raw_fam),total_source_rows=raw_scanned,audit_sample_probability=CFG['audit_sample_probability'],unique_signatures=len(g),duplicate_extra_rows=int((g.n-1).sum()),duplicate_group_count=len(coll),binary_conflict_groups=len(conf),binary_conflict_rows=int(conf.n.sum()),family_conflict_groups=len(famconf),family_conflict_rows=int(famconf.n.sum()),sampled_pool_rows=len(data)))
 audit_count=len(raw_fam)
 del raw_fam
 gc.collect()
 # Keep a fixed sampled pool; all dedup comparisons take place on identical source samples.
 payload=dict(df=data,safe=safe,expanded=expanded,audit=audits,source=str(source),origin=origin,
              excluded=excluded,raw_scanned=raw_scanned,n_rows=len(data))
 pd.to_pickle(payload,CACHE)
 pd.DataFrame(audits).to_csv(OUT/'global_duplicate_audit.csv',index=False)
 dump(OUT/'dataset_provenance.json',dict(origin=origin,source=str(source),raw_scanned=raw_scanned,audit_sample_rows=audit_count,pool_rows=len(data),safe_features=safe,expanded_numeric_features=expanded,excluded_columns=excluded,
    raw_family_counts=data.family.value_counts().to_dict(),caveats=['64-bit content hashes, not cryptographic proofs; full signatures may include record-specific metadata','Source scanned sequentially across full CSV; audit is a Bernoulli sample across the file and counts are not full-dataset duplicate totals','sampled pool changes natural attack prevalence','signature grouping prevents exact-row overlap; independent device/session metadata may not be available','same observed features do not guarantee same traffic context','categorical/payload text excluded from numeric modeling']))
 print('AUDIT',audits,'sampled pool=',len(data),flush=True)
 return payload

def policy_subset(df,policy):
 sigcol='_full_sig' if policy=='full_signature' else '_safe_sig'
 # Always exclude contradictory exact-full-feature signatures across ALL policies:
 # even the no-dedup baseline must not place identical but differently labeled
 # inputs in train/test groups or silently choose one arbitrary ground truth.
 full=df.groupby('_full_sig').agg(families=('family','nunique'),binary=('binary_label','nunique'))
 fullbad=full.index[(full.families>1)|(full.binary>1)]
 clean=df[~df['_full_sig'].isin(fullbad)].copy()
 if policy=='none':
  return clean,dict(policy=policy,removed=len(df)-len(clean),conflict_groups_removed=len(fullbad))
 grouped=clean.groupby(sigcol).agg(families=('family','nunique'),binary=('binary_label','nunique'))
 bad=grouped.index[(grouped.families>1)|(grouped.binary>1)]
 clean=clean[~clean[sigcol].isin(bad)]
 result=clean.drop_duplicates(subset=[sigcol],keep='first').copy()
 return result,dict(policy=policy,removed=len(df)-len(result),conflict_groups_removed=len(fullbad)+len(bad),n_after=len(result))

def split_fold(df,held,seed):
 fam=df.family.to_numpy();y=df.binary_label.to_numpy(dtype=np.int64)
 unknown=np.flatnonzero((fam==held)&(y==1));benign=np.flatnonzero(y==0);known=np.flatnonzero((y==1)&(fam!=held))
 if len(unknown)<25 or len(benign)<100 or len(known)<100:raise ValueError(f'Insufficient samples fold={held} unknown={len(unknown)} benign={len(benign)} known={len(known)}')
 # Purge any exact full-feature signature shared by heldout-family and source families.
 full=df._full_sig.to_numpy()
 reserved=np.unique(full[unknown]);remaining=np.r_[benign,known]
 remaining=remaining[~np.isin(full[remaining],reserved)]
 if len(remaining)<300:raise ValueError('Insufficient source examples after unknown-family signature purge')
 # GroupShuffleSplit prevents duplicate full-feature records crossing train, validation, benign test.
 def gss(indices,test_frac,s):
  spl=GroupShuffleSplit(n_splits=1,test_size=test_frac,random_state=s)
  a,b=next(spl.split(indices,groups=full[indices]));return indices[a],indices[b]
 src,te_b=gss(remaining[y[remaining]==0],0.23,seed+11)
 benign_tr,benign_val=gss(src,0.18,seed+12)
 known_tr,known_val=gss(remaining[y[remaining]==1],0.18,seed+13)
 tr=np.r_[benign_tr,known_tr];va=np.r_[benign_val,known_val];te=np.r_[te_b,unknown]
 assert not set(full[tr]).intersection(set(full[va]))
 assert not set(full[tr]).intersection(set(full[te]))
 assert not set(full[va]).intersection(set(full[te]))
 assert not np.any(fam[tr]==held) and not np.any(fam[va]==held)
 return tr,va,te,dict(heldout_family=held,seed=seed,n_train=len(tr),n_validation=len(va),n_test=len(te),n_unknown=len(unknown),
  benign_test=len(te_b),train_attack_frac=float(np.mean(y[tr])),test_attack_frac=float(np.mean(y[te])),
  source_shared_unknown_signatures_purged=int(len(np.r_[benign,known])-len(remaining)),
  train_full_signature_intersection_test=0,validation_full_signature_intersection_test=0)

class RBF(nn.Module):
 def __init__(self,ni,no,bases):
  super().__init__();self.register_buffer('grid',torch.linspace(-2.5,2.5,bases));self.scale=nn.Parameter(torch.ones(ni));self.proj=nn.Linear(ni*bases,no)
 def forward(self,x):return self.proj(torch.exp(-((x.unsqueeze(-1)-self.grid)/(.8+self.scale.abs().unsqueeze(-1)))**2).flatten(1))
class Network(nn.Module):
 def __init__(self,d,kan):
  super().__init__();self.a=RBF(d,CFG['hidden'],CFG['bases']) if kan else nn.Linear(d,CFG['hidden'])
  self.b=RBF(CFG['hidden'],CFG['embedding'],CFG['bases']) if kan else nn.Linear(CFG['hidden'],CFG['embedding']);self.out=nn.Linear(CFG['embedding'],1)
 def forward(self,x):return self.out(torch.tanh(self.b(F.relu(self.a(x))))).flatten()

def train(X,y,tr,seed,modelname,device):
 seedall(seed);kind=modelname.endswith('KAN');d=X.shape[1]
 initial=Network(d,kind);state={k:v.cpu().clone() for k,v in initial.state_dict().items()}
 rng=np.random.default_rng(seed+123);train_local=np.asarray(tr)
 buckets=[[] for _ in range(CFG['n_clients'])]
 for cls in [0,1]:
  ix=train_local[y[train_local]==cls].copy();rng.shuffle(ix)
  probs=rng.dirichlet([CFG['dirichlet_alpha']]*CFG['n_clients'])
  pieces=np.split(ix,np.cumsum(rng.multinomial(len(ix),probs))[:-1])
  for k,v in enumerate(pieces):buckets[k].extend(v.tolist())
 # Protect against zero-class clients in tiny smoke tests.
 buckets=[np.asarray(b,dtype=int) for b in buckets]
 steps=0;start=time.perf_counter()
 for roundid in range(CFG['rounds']):
  weights=[];models=[]
  for cid,indices in enumerate(buckets):
   if len(indices)==0:continue
   m=Network(d,kind);m.load_state_dict(state);m.to(device).train()
   opt=torch.optim.AdamW(m.parameters(),lr=CFG['lr'])
   pos=indices[y[indices]==1];neg=indices[y[indices]==0];r=np.random.default_rng(seed+roundid*500+cid*13)
   for t in range(CFG['steps']):
    if len(pos) and len(neg):
     a=r.choice(pos,CFG['batch_size']//2,replace=True);b=r.choice(neg,CFG['batch_size']-len(a),replace=True);ids=np.r_[a,b]
    else:ids=r.choice(indices,CFG['batch_size'],replace=True)
    xx=torch.from_numpy(X[ids]).to(device);yy=torch.from_numpy(y[ids].astype(np.float32)).to(device)
    opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(m(xx),yy);loss.backward();opt.step();steps+=1
   models.append({k:v.detach().cpu().float().clone() for k,v in m.state_dict().items()});weights.append(len(indices))
   m.cpu()
  weight=np.asarray(weights,dtype=float);weight/=weight.sum()
  state={k:sum(v[k]*float(w) for v,w in zip(models,weight)) for k in state}
 final=Network(d,kind);final.load_state_dict(state)
 sec=time.perf_counter()-start
 par=sum(p.numel() for p in final.parameters())
 return final,steps,par,par*4*2*CFG['rounds']*len(buckets),sec

@torch.inference_mode()
def predict(m,X,idx,device):
 m.to(device).eval();scores=[]
 for b in np.array_split(idx,max(1,int(np.ceil(len(idx)/2048)))):
  if len(b):scores.append(torch.sigmoid(m(torch.from_numpy(X[b]).to(device))).cpu().numpy())
 m.cpu();return np.concatenate(scores)

def one_eval(payload,held,seed,policy,features,method,device):
 original=payload['df'];df,pinfo=policy_subset(original,policy)
 if held not in set(df.family):raise ValueError('INFEASIBLE_POLICY: heldout family absent after dedup '+held+' / '+policy)
 tr,va,te,manifest=split_fold(df,held,seed)
 x=df[features].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
 x[~np.isfinite(x)]=np.nan
 imp=SimpleImputer(strategy='median',keep_empty_features=True);imp.fit(x[tr]);x=imp.transform(x)
 sc=RobustScaler(quantile_range=(10,90));sc.fit(x[tr]);x=np.clip(sc.transform(x),-12,12).astype(np.float32)
 y=df.binary_label.to_numpy(dtype=np.int64)
 m,steps,pars,comm,secs=train(x,y,tr,seed,method,device)
 scores=predict(m,x,np.r_[va,te],device);sv=scores[:len(va)];st=scores[len(va):]
 valneg=sv[y[va]==0];testy=y[te]
 try:auroc=float(roc_auc_score(testy,st));auprc=float(average_precision_score(testy,st))
 except ValueError:auroc=auprc=float('nan')
 results=[]
 for target in CFG['threshold_targets']:
  threshold=float(np.quantile(valneg,1-target,method='higher'))
  predicted=st>=threshold
  tp=int(np.sum(predicted&(testy==1)));fp=int(np.sum(predicted&(testy==0)))
  fn=int(np.sum((~predicted)&(testy==1)));tn=int(np.sum((~predicted)&(testy==0)))
  results.append(dict(heldout_family=held,seed=seed,feature_set='safe30' if features==payload['safe'] else 'expanded_numeric',dedup_policy=policy,
    model=method,val_target_fpr=target,n_test=len(te),n_attack=int((testy==1).sum()),n_benign=int((testy==0).sum()),threshold=threshold,
    recall=tp/max(1,tp+fn),fpr=fp/max(1,fp+tn),attack_f1=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,tp+fp),auroc=auroc,auprc=auprc,tp=tp,fp=fp,tn=tn,fn=fn))
 # Diagnostic test-only feature-group masking: replace selected scaled features with zero,
 # the training-median reference under RobustScaler. This is an input intervention,
 # NOT retraining or evidence of causal feature importance.
 group_ablation=[]
 if features==payload['expanded']:
  groups={'HTTP':['http.'],'TCP':['tcp.'],'MQTT':['mqtt.'],'DNS':['dns.'],
          'UDP':['udp.'],'ICMP':['icmp.'],'ARP':['arp.'],'Modbus':['mbtcp.']}
  for group,starts in groups.items():
   cols=[j for j,col in enumerate(features) if any(col.lower().startswith(prefix) for prefix in starts)]
   if not cols:continue
   changed=x[te].copy();changed[:,cols]=0.0
   # Predict on modified test features; validation threshold stays unchanged.
   mask_model=m.to(device).eval();pieces=[]
   with torch.inference_mode():
    for piece in np.array_split(changed,max(1,int(np.ceil(len(changed)/2048)))):
     if len(piece):pieces.append(torch.sigmoid(mask_model(torch.from_numpy(piece).to(device))).cpu().numpy())
   m.cpu();ss=np.concatenate(pieces)
   try:auc_mask=float(roc_auc_score(testy,ss))
   except ValueError:auc_mask=float('nan')
   for target in CFG['threshold_targets']:
    threshold=next(rec['threshold'] for rec in results if rec['val_target_fpr']==target)
    pred=ss>=threshold;tp=int(np.sum(pred&(testy==1)));fn=int(np.sum((~pred)&(testy==1)))
    fp=int(np.sum(pred&(testy==0)));tn=int(np.sum((~pred)&(testy==0)))
    group_ablation.append(dict(heldout_family=held,seed=seed,dedup_policy=policy,feature_set='expanded_numeric',model=method,
        group=group,n_masked_features=len(cols),val_target_fpr=target,
        masked_recall=tp/max(1,tp+fn),masked_fpr=fp/max(1,fp+tn),masked_auroc=auc_mask,
        original_recall=next(rec['recall'] for rec in results if rec['val_target_fpr']==target),
        original_fpr=next(rec['fpr'] for rec in results if rec['val_target_fpr']==target),
        original_auroc=auroc))
 resource=dict(heldout_family=held,seed=seed,feature_set=results[0]['feature_set'],dedup_policy=policy,model=method,
  train_rows=len(tr),validation_rows=len(va),test_rows=len(te),features=len(features),params=pars,training_steps=steps,
  communication_bytes=comm,training_seconds=secs,feature_ablation_json=json.dumps(group_ablation),policy_rows=len(df),policy_removed=pinfo['removed'],
  conflicting_signature_groups_removed=pinfo['conflict_groups_removed'])
 # Return small score samples for audit, without raw network data.
 rng=np.random.default_rng(seed+91);selected=rng.choice(len(te),min(400,len(te)),replace=False)
 ps=[dict(heldout_family=held,seed=seed,feature_set=resource['feature_set'],dedup_policy=policy,model=method,
          label=int(testy[i]),family=str(df.family.iloc[te[i]]),score=float(st[i])) for i in selected]
 return results,resource,ps,manifest

def worker(worker_id,workers):
 if not CACHE.exists():raise FileNotFoundError('Missing prepared cache; run --mode prepare first.')
 payload=pd.read_pickle(CACHE);device=f'cuda:{worker_id%torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
 tasks=[(held,seed,policy,fn,model) for held in CFG['families'] for seed in CFG['seeds'] for policy in CFG['dedup_policies'] for fn in CFG['feature_sets'] for model in CFG['models']]
 tasks=[t for i,t in enumerate(tasks) if i%workers==worker_id]
 mm=[];rr=[];pp=[];man=[];fail=[];skipped=[]
 for count,(held,seed,policy,fn,model) in enumerate(tasks):
  try:
   out,res,scores,manifest=one_eval(payload,held,seed,policy,payload['safe'] if fn=='safe30' else payload['expanded'],model,device)
   if out is not None:mm.extend(out);rr.append(res);pp.extend(scores);man.append(dict(**manifest,feature_set=fn,dedup_policy=policy,model=model))
  except ValueError as ex:
   if 'Insufficient samples' in str(ex) or 'INFEASIBLE_POLICY' in str(ex) or 'Insufficient source examples' in str(ex):
    skipped.append(dict(heldout_family=held,seed=seed,dedup_policy=policy,feature_set=fn,model=model,reason=str(ex)))
    print('SKIPPED',held,seed,policy,fn,model,str(ex),flush=True)
   else:
    fail.append(dict(heldout_family=held,seed=seed,dedup_policy=policy,feature_set=fn,model=model,error=str(ex)))
    print('FAILED',held,seed,policy,fn,model,repr(ex),flush=True)
  except Exception as ex:
   fail.append(dict(heldout_family=held,seed=seed,dedup_policy=policy,feature_set=fn,model=model,error=str(ex)))
   print('FAILED',t if False else (held,seed,policy,fn,model),repr(ex),flush=True)
  if (count+1)%12==0:print('WORKER',worker_id,count+1,'/',len(tasks),flush=True)
 pd.DataFrame(mm).to_csv(OUT/f'worker_{worker_id}_metrics.csv',index=False)
 pd.DataFrame(rr).to_csv(OUT/f'worker_{worker_id}_resources.csv',index=False)
 pd.DataFrame(pp).to_csv(OUT/f'worker_{worker_id}_scores.csv',index=False)
 dump(OUT/f'worker_{worker_id}_manifest.json',man);dump(OUT/f'worker_{worker_id}_failures.json',fail);dump(OUT/f'worker_{worker_id}_skipped.json',skipped)
 print('FINISHED WORKER',worker_id,'successful',len(rr),'skipped',len(skipped),'failed',len(fail),flush=True)
 if fail:raise RuntimeError(f'{len(fail)} experiment cells failed in worker {worker_id}; see failures JSON')

def aggregate(workers):
 files=[OUT/f'worker_{i}_metrics.csv' for i in range(workers)]
 for f in files:
  if not f.exists():raise RuntimeError('Missing worker output '+str(f))
 def load_nonempty(name):
  frames=[]
  for i in range(workers):
   f=OUT/f'worker_{i}_{name}.csv'
   if f.exists() and f.stat().st_size>0:
    try:frames.append(pd.read_csv(f))
    except pd.errors.EmptyDataError:pass
  return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
 metrics=load_nonempty('metrics')
 resources=load_nonempty('resources')
 scores=load_nonempty('scores')
 skipped=[]
 for i in range(workers):
  f=OUT/f'worker_{i}_skipped.json'
  if f.exists():skipped.extend(json.loads(f.read_text()))
 dump(OUT/'skipped_experiments.json',skipped)
 if metrics.empty or resources.empty:
  raise RuntimeError('NO_VALID_EXPERIMENTS: no valid method results; check PREPARE_FAMILY_COUNTS and skipped_experiments.json')
 expected=len(CFG['families'])*len(CFG['seeds'])*len(CFG['dedup_policies'])*len(CFG['feature_sets'])*len(CFG['models'])
 if len(resources)+len(skipped)!=expected or len(metrics)!=2*len(resources):
  raise RuntimeError(f'Incomplete experiment: completed={len(resources)} skipped={len(skipped)} expected={expected}')
 print('AGGREGATE_COMPLETED',len(resources),'SKIPPED_INFEASIBLE',len(skipped),'EXPECTED',expected,flush=True)
 metrics.to_csv(OUT/'all_metrics.csv',index=False);resources.to_csv(OUT/'all_resources.csv',index=False);scores.to_csv(OUT/'prediction_scores_sample.csv',index=False)
 keys=['heldout_family','feature_set','dedup_policy','model','val_target_fpr']
 s=metrics.groupby(keys)[['recall','fpr','attack_f1','precision','auroc','auprc']].agg(['mean','std']).reset_index()
 s.columns=['_'.join(str(a) for a in c if a!='') for c in s.columns];s.to_csv(OUT/'summary_mean_std.csv',index=False)
 r=resources.groupby(['feature_set','dedup_policy','model'])[['params','training_steps','communication_bytes','train_rows','test_rows','training_seconds']].agg(['mean','std']).reset_index()
 r.columns=['_'.join(str(a) for a in c if a!='') for c in r.columns];r.to_csv(OUT/'resource_summary.csv',index=False)
 # Feature group diagnostics are unpacked from each matched run's resource record.
 ablations=[]
 for value in resources.feature_ablation_json.dropna():
  ablations.extend(json.loads(value))
 pd.DataFrame(ablations).to_csv(OUT/'feature_group_ablation.csv',index=False)
 # Paired comparisons at fixed fold and seed; the sample inclusion differs across dedup settings.
 piv=metrics.pivot_table(index=['heldout_family','seed','dedup_policy','model','val_target_fpr'],columns='feature_set',values=['recall','auroc','auprc','fpr'])
 delta=piv.copy()
 cols=[]
 for metric in ['recall','auroc','auprc','fpr']:
  if (metric,'safe30') in piv.columns and (metric,'expanded_numeric') in piv.columns:
   delta[(metric,'delta_expanded_minus_safe')]=piv[(metric,'expanded_numeric')]-piv[(metric,'safe30')]
   cols.append((metric,'delta_expanded_minus_safe'))
 delta[cols].to_csv(OUT/'paired_feature_deltas.csv')
 try:
  import matplotlib
  matplotlib.use('Agg');import matplotlib.pyplot as plt
  pick=s[(s.dedup_policy=='full_signature')&(s.val_target_fpr==0.01)]
  fig,ax=plt.subplots(figsize=(10,5))
  for (model,fs),g in pick.groupby(['model','feature_set']):ax.plot(g.heldout_family,g.recall_mean,marker='o',label=model+' / '+fs)
  ax.set_ylabel('Unknown attack recall');ax.set_ylim(0,1);ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'feature_comparison.png',dpi=160);plt.close(fig)
 except Exception as e:print('Plot skipped:',str(e))
 dump(OUT/'run_status.json',dict(status='complete_with_infeasible_policies' if skipped else 'complete',expected_tasks=expected,completed_tasks=len(resources),skipped_tasks=len(skipped),operating_point_rows=len(metrics),
   seeds=CFG['seeds'],families=CFG['families'],device_count=torch.cuda.device_count(),
   caveat='FPR targets are validation operating points; actual test FPR is separate. Capped test prevalence is not real-world prevalence.'))
 dump(OUT/'config.json',CFG)
 archive=ROOT/CFG['archive']
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in OUT.rglob('*'):
   if p.is_file():z.write(p,arcname=p.relative_to(ROOT))
 print('ARCHIVE',archive,archive.stat().st_size,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','worker','aggregate'],required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);args=p.parse_args()
 try:
  if args.mode=='prepare':prepare()
  elif args.mode=='worker':worker(args.worker,args.workers)
  else:aggregate(args.workers)
 except Exception:traceback.print_exc();sys.exit(1)
