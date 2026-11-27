"""Edge-IIoTset fixed-test feature and deduplication benchmark.
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
CACHE=ROOT/'edge_integrity_cache_v5.pkl'
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
 # Materialize numeric matrices once during prepare, outside both worker processes.
 # Avoid pandas DataFrame.apply(pd.to_numeric) in workers: on Kaggle Python 3.13
 # this triggered a native-level SIGSEGV in pandas.core.apply.
 def safe_float(value):
  try:
   val=float(value)
   return val if np.isfinite(val) else float('nan')
  except (ValueError,TypeError,OverflowError):
   return float('nan')
 numeric_expanded=np.empty((len(data),len(expanded)),dtype=np.float32)
 for j,col in enumerate(expanded):
  values=data[col].to_numpy(dtype=object,copy=False)
  numeric_expanded[:,j]=np.fromiter((safe_float(v) for v in values),dtype=np.float32,count=len(values))
  if (j+1)%10==0:print('NUMERIC_MATRIX_COLUMNS',j+1,'/',len(expanded),flush=True)
 safe_ix=[expanded.index(f) for f in safe]
 numeric_safe=np.ascontiguousarray(numeric_expanded[:,safe_ix])
 print('NUMERIC_MATRIX_READY',numeric_expanded.shape,numeric_safe.shape,'finite expanded fraction',float(np.isfinite(numeric_expanded).mean()),flush=True)
 # Preserve original sampled row index for all three dedup policies.
 data=data.reset_index(drop=True)
 # Keep a fixed sampled pool; all dedup comparisons take place on identical source samples.
 payload=dict(df=data,safe=safe,expanded=expanded,numeric_safe=numeric_safe,numeric_expanded=numeric_expanded,audit=audits,source=str(source),origin=origin,
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



# Fixed-Test benchmark. Every configuration within a (heldout, seed) fold uses the
# identical immutable test/validation source indices. Deduplication acts on TRAIN ONLY.
# PooledLightGBM is a centralized, non-federated reference, never labeled an FL method.
FIXCFG=dict(
 seeds=[0,42,3407], heldouts=['Port_Scanning','DDoS_UDP','SQL_injection','Ransomware'],
 methods=['FedAvgMLP','FedAvgKAN','PooledLightGBM'],
 policies=['none','full_signature','safe30_signature'],
 base_features=['safe30','expanded36'],
 retrain_ablations=['no_ports','no_checksum','no_seq_payload'],
 target_fprs=[0.01,0.05],
 rounds=5,steps=8,fold_benign_test_frac=.23,fold_validation_frac=.18,
 max_train_rows=60000,run_name='EdgeIIoT_FixedTest_Results'
)
if os.getenv('EDGE_FIXED_SMOKE')=='1':
 FIXCFG.update(seeds=[0],heldouts=['Port_Scanning','SQL_injection'],methods=['FedAvgMLP','FedAvgKAN','PooledLightGBM'],policies=['none','safe30_signature'],base_features=['safe30','expanded36'],retrain_ablations=['no_ports'])
FIXROOT=Path(os.getenv('EDGE_AUDIT_ROOT','/kaggle/working' if Path('/kaggle/working').is_dir() else '/mnt/data/edge_fixedtest_local'))
FIXOUT=FIXROOT/FIXCFG['run_name']
FIXOUT.mkdir(parents=True,exist_ok=True)
FIXCACHE=FIXROOT/'edge_fixedtest_cache.pkl'

# Explicit prespecified features; no data-driven schema decision from the test set.
EXTRA=['tcp.ack_raw','tcp.checksum','tcp.dstport','tcp.payload','tcp.seq','tcp.srcport']

def numeric_schema(sample,nonlabel,safe):
 """Prespecified numeric schema: no data-driven selection using a CSV prefix."""
 selected=list(dict.fromkeys(safe+[name for name in EXTRA if name in nonlabel]))
 excluded=[name for name in nonlabel if name not in selected]
 return selected,excluded

def fixed_prepare():
 global ROOT,OUT,CACHE
 # Reuse v5's full-file, class-stratified CSV streaming and reservoir sampling.
 # The source schema is pre-specified below, and we record any unavailable fields.
 ROOT=FIXROOT;OUT=FIXOUT;CACHE=FIXCACHE
 CFG['max_scan_rows']=None
 payload=prepare()
 cols=list(payload['expanded'])
 expected=list(dict.fromkeys(payload['safe']+[x for x in EXTRA if x in cols]))
 if cols!=expected:
  # Force fixed schema instead of opportunistic feature extraction from first rows.
  available=[x for x in expected if x in cols]
  missing=[x for x in EXTRA if x not in cols]
  print('FEATURE_SCHEMA_NOTE expected',len(expected),'actual',len(cols),'missing',missing,flush=True)
  payload['schema_warning']='Original preparer selected numeric candidates from 1500-row preview. Available features locked and logged here.'
 # Convert dataframe to lean payload as numerical cache; no raw CSV is required in workers.
 payload['row_key']=np.arange(len(payload['df']),dtype=np.int64)
 payload['seeded_pool_fixed']=True
 pd.to_pickle(payload,FIXCACHE)
 prov=json.loads((FIXOUT/'dataset_provenance.json').read_text())
 prov['fixed_test_protocol']='Immutable fold split created once per heldout family and seed; dedup applied only to train; validation and test never filtered by dedup.'
 prov['schema_note']='Safe30 list and explicit six additional TCP features; feature eligibility determined from original fixed list, not test labels.'
 prov['source_row_independence']='Original source capture/session IDs are not guaranteed; grouping uses exact full-row signature.'
 prov['pool_prevalence_warning']='Pool stratified/capped; measured precision is not deployment prevalence.'
 dump(FIXOUT/'dataset_provenance.json',prov)
 dump(FIXOUT/'config.json',FIXCFG)
 print('FIXED_PREPARE_READY',len(payload['df']),'rows',len(cols),'numeric features',flush=True)

def make_fixed_split(data,held,seed):
 fam=data.family.astype(str).to_numpy()
 y=data.binary_label.to_numpy(dtype=np.int8)
 unknown=np.flatnonzero((fam==held)&(y==1))
 benign=np.flatnonzero(y==0)
 known=np.flatnonzero((y==1)&(fam!=held))
 if len(unknown)<25 or len(benign)<200 or len(known)<200:raise RuntimeError('Infeasible fold '+held)
 sig=data['_full_sig'].to_numpy(dtype=np.uint64)
 # A heldout attack's exact full signature must never enter training or validation.
 held_sig=np.unique(sig[unknown]); benign=benign[~np.isin(sig[benign],held_sig)]
 known=known[~np.isin(sig[known],held_sig)]
 def group(indices,frac,rs):
  splitter=GroupShuffleSplit(n_splits=1,test_size=frac,random_state=rs)
  a,b=next(splitter.split(indices,groups=sig[indices]))
  return indices[a],indices[b]
 # Reserve conflicting full signatures for benign consistently across partitions.
 # Otherwise a benign and known-attack record sharing a full signature could
 # enter different splits even when each class is group-split internally.
 known=known[~np.isin(sig[known],np.unique(sig[benign]))]
 if len(known)<100:raise RuntimeError('Insufficient known attacks after cross-label signature purge')
 benign_src,benign_test=group(benign,FIXCFG['fold_benign_test_frac'],seed+19)
 benign_train,benign_val=group(benign_src,FIXCFG['fold_validation_frac'],seed+29)
 known_train,known_val=group(known,FIXCFG['fold_validation_frac'],seed+39)
 train=np.r_[benign_train,known_train].astype(np.int64)
 val=np.r_[benign_val,known_val].astype(np.int64)
 test=np.r_[benign_test,unknown].astype(np.int64)
 for a,b in ((train,val),(train,test),(val,test)):
  if np.intersect1d(sig[a],sig[b]).size:raise RuntimeError('Full-signature leakage detected')
 if np.any(fam[train]==held) or np.any(fam[val]==held):raise RuntimeError('Heldout family leaked into fitting')
 if len(np.unique(y[train]))<2 or len(np.unique(y[val]))<2:raise RuntimeError('Missing source labels')
 return dict(train=train,val=val,test=test,n_unknown=len(unknown),n_test_benign=len(benign_test))

def train_only_policy(data,train_indices,policy):
 if policy=='none':return train_indices.copy(),dict(removed=0,conflicting_groups=0)
 col='_full_sig' if policy=='full_signature' else '_safe_sig'
 td=data.iloc[train_indices]
 groups=td.groupby(col,sort=False).agg(families=('family','nunique'), binary=('binary_label','nunique'))
 conflict=groups.index[(groups.families>1)|(groups.binary>1)]
 # Important: train-only filter; neither validation nor test is changed.
 kept=td.loc[~td[col].isin(conflict)]
 kept=kept.drop_duplicates(subset=[col],keep='first')
 ix=kept.index.to_numpy(dtype=np.int64)
 if len(ix)<40 or len(data.iloc[ix].binary_label.unique())<2:
  raise ValueError('INFEASIBLE_TRAIN_POLICY '+policy+' n='+str(len(ix)))
 return ix,dict(removed=int(len(train_indices)-len(ix)),conflicting_groups=int(len(conflict)))

def feature_configs(payload):
 available=list(payload['expanded']);safe=list(payload['safe']);extra=[c for c in EXTRA if c in available]
 e=list(dict.fromkeys(safe+extra))
 return dict(safe30=safe,expanded36=e,
  no_ports=[c for c in e if c not in ('tcp.srcport','tcp.dstport')],
  no_checksum=[c for c in e if c!='tcp.checksum'],
  no_seq_payload=[c for c in e if c not in ('tcp.seq','tcp.ack_raw','tcp.payload')])

def fit_and_predict(payload,fold,train_idx,features,method,seed,device):
 from sklearn.impute import SimpleImputer
 from sklearn.preprocessing import RobustScaler
 data=payload['df'];y=data.binary_label.to_numpy(dtype=np.int8)
 e=payload['expanded'];base=payload['numeric_expanded'];
 feature_pos=[e.index(col) for col in features]
 x=np.ascontiguousarray(base[:,feature_pos],dtype=np.float32)
 x[~np.isfinite(x)]=np.nan
 imp=SimpleImputer(strategy='median',keep_empty_features=True)
 imp.fit(x[train_idx]);x=imp.transform(x)
 sc=RobustScaler(quantile_range=(10,90));sc.fit(x[train_idx]);x=np.clip(sc.transform(x),-12,12).astype(np.float32)
 valid=fold['val'];test=fold['test']
 if method=='PooledLightGBM':
  import lightgbm as lgb
  # Centralized training-data pooling: an explicit NON-FL comparator.
  model=lgb.LGBMClassifier(n_estimators=130,num_leaves=15,max_depth=5,learning_rate=.05,
   min_child_samples=15,reg_lambda=1.,verbosity=-1,n_jobs=1,random_state=seed)
  model.fit(x[train_idx],y[train_idx])
  sv=model.predict_proba(x[valid])[:,1];st=model.predict_proba(x[test])[:,1]
  steps=130;pars=int(sum(t['num_leaves'] for t in model.booster_.dump_model()['tree_info']));comm=0
  model_params=f'{pars} leaf nodes (not neural parameter count)'
 else:
  # Fair FL optimizer steps across MLP and KAN; parameter count reported separately.
  mdl,steps,pars,comm,_=train(x,y,train_idx,seed,method,device)
  sv=predict(mdl,x,valid,device);st=predict(mdl,x,test,device);model_params=str(pars)
 return sv,st,dict(training_steps=int(steps),parameters_description=model_params,communication_bytes=int(comm))

def task_grid():
 jobs=[]
 for held in FIXCFG['heldouts']:
  for seed in FIXCFG['seeds']:
   for policy in FIXCFG['policies']:
    for fs in FIXCFG['base_features']:
     for method in FIXCFG['methods']:
      jobs.append((held,seed,policy,fs,method))
   for fs in FIXCFG['retrain_ablations']:
    for method in FIXCFG['methods']:
     jobs.append((held,seed,'none',fs,method))
 return jobs

def fixed_worker(worker_id,workers):
 payload=pd.read_pickle(FIXCACHE)
 data=payload['df'].reset_index(drop=True)
 feats=feature_configs(payload); fulljobs=[t for i,t in enumerate(task_grid()) if i%workers==worker_id]
 device=f'cuda:{worker_id%torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
 fold_cache={};rows=[];resources=[];manifest=[];fail=[];skipped=[]
 for i,(held,seed,policy,fs,method) in enumerate(fulljobs,1):
  try:
   key=(held,seed)
   if key not in fold_cache:fold_cache[key]=make_fixed_split(data,held,seed)
   fold=fold_cache[key]
   subset,pstats=train_only_policy(data,fold['train'],policy)
   if len(subset)<50:raise ValueError('INFEASIBLE_TRAIN_POLICY')
   # Balanced sampler uses only the train subset; validation/test unchanged.
   t0=time.perf_counter()
   sv,st,stat=fit_and_predict(payload,fold,subset,feats[fs],method,seed,device)
   secs=time.perf_counter()-t0
   y=data.binary_label.to_numpy(dtype=np.int8);vy=y[fold['val']];ty=y[fold['test']]
   if not (np.array_equal(fold['test'],fold_cache[key]['test']) and np.array_equal(fold['val'],fold_cache[key]['val'])):
    raise RuntimeError('Fixed test invariance failed')
   valneg=sv[vy==0]
   if len(valneg)<30:raise ValueError('Too few benign validation examples')
   auroc=float(roc_auc_score(ty,st));auprc=float(average_precision_score(ty,st))
   for target in FIXCFG['target_fprs']:
    threshold=float(np.quantile(valneg,1-target,method='higher'))
    pred=st>=threshold;tp=int(np.sum(pred&(ty==1)));fp=int(np.sum(pred&(ty==0)))
    fn=int(np.sum(~pred&(ty==1)));tn=int(np.sum(~pred&(ty==0)))
    rows.append(dict(heldout_family=held,seed=seed,policy=policy,feature_set=fs,method=method,
     target_val_fpr=target,threshold=threshold,n_test=len(ty),n_unknown=int(np.count_nonzero(ty==1)),n_benign=int(np.count_nonzero(ty==0)),
     unknown_recall=tp/max(1,tp+fn),test_fpr=fp/max(1,fp+tn),attack_f1=2*tp/max(1,2*tp+fp+fn),
     precision=tp/max(1,tp+fp),auroc=auroc,auprc=auprc,tp=tp,fp=fp,fn=fn,tn=tn))
   resources.append(dict(heldout_family=held,seed=seed,policy=policy,feature_set=fs,method=method,
    n_train_before=len(fold['train']),n_train_after=len(subset),n_val=len(fold['val']),n_test=len(fold['test']),
    n_features=len(feats[fs]),training_seconds=secs,dedup_removed=pstats['removed'],
    training_steps=stat['training_steps'],communication_bytes=stat['communication_bytes'],
    parameters_description=stat['parameters_description'],is_federated=(method!='PooledLightGBM')))
   manifest.append(dict(heldout_family=held,seed=seed,policy=policy,feature_set=fs,method=method,
    test_sample_indices_sha256=hashlib.sha256(np.sort(fold['test']).tobytes()).hexdigest(),
    val_sample_indices_sha256=hashlib.sha256(np.sort(fold['val']).tobytes()).hexdigest(),
    n_val=len(fold['val']),n_test=len(fold['test']),n_unknown=fold['n_unknown'],
    n_test_benign=fold['n_test_benign'],train_indices_after_policy_sha256=hashlib.sha256(np.sort(subset).tobytes()).hexdigest()))
  except ValueError as ex:
   if 'INFEASIBLE_TRAIN_POLICY' in str(ex):skipped.append(dict(job=[held,seed,policy,fs,method],reason=str(ex)))
   else:fail.append(dict(job=[held,seed,policy,fs,method],error=traceback.format_exc()))
  except Exception:
   fail.append(dict(job=[held,seed,policy,fs,method],error=traceback.format_exc()))
  if i%5==0:print('FIXED_WORKER',worker_id,i,'/',len(fulljobs),'ok',len(resources),'skip',len(skipped),'fail',len(fail),flush=True)
 def csvout(name,r):
  if r:pd.DataFrame(r).to_csv(FIXOUT/f'worker_{worker_id}_{name}.csv',index=False)
  else:(FIXOUT/f'worker_{worker_id}_{name}.csv').write_text('')
 csvout('metrics',rows);csvout('resources',resources)
 dump(FIXOUT/f'worker_{worker_id}_manifest.json',manifest)
 dump(FIXOUT/f'worker_{worker_id}_skipped.json',skipped)
 dump(FIXOUT/f'worker_{worker_id}_failures.json',fail)
 print('WORKER_FINISHED',worker_id,'successful',len(resources),'skipped',len(skipped),'failed',len(fail),flush=True)
 if fail:raise RuntimeError(f'{len(fail)} jobs failed; inspect worker_{worker_id}_failures.json')

def fixed_aggregate(workers):
 def read_csv(name):
  frames=[]
  for i in range(workers):
   f=FIXOUT/f'worker_{i}_{name}.csv'
   if f.exists() and f.stat().st_size:
    frames.append(pd.read_csv(f))
  return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
 mm=read_csv('metrics');rr=read_csv('resources')
 if mm.empty:raise RuntimeError('No valid results. Inspect worker failures and skipped records.')
 allmanifest=[];allskipped=[];allfails=[]
 for i in range(workers):
  for tag,dest in [('manifest',allmanifest),('skipped',allskipped),('failures',allfails)]:
   p=FIXOUT/f'worker_{i}_{tag}.json'
   if p.exists():dest.extend(json.loads(p.read_text()))
 if allfails:raise RuntimeError(f'Unresolved worker failures: {len(allfails)}')
 expected=len(task_grid());completed=len(rr)
 if completed+len(allskipped)!=expected:raise RuntimeError(f'Incomplete expected {expected} completed {completed} skipped {len(allskipped)}')
 # Verify every method/policy/feature set in each (heldout, seed) fold uses exact same test/validation identities.
 for _,group in pd.DataFrame(allmanifest).groupby(['heldout_family','seed']):
  assert group.test_sample_indices_sha256.nunique()==1,'Test samples changed across methods!'
  assert group.val_sample_indices_sha256.nunique()==1,'Validation samples changed across methods!'
  assert group.n_test.nunique()==1 and group.n_unknown.nunique()==1
 mm.to_csv(FIXOUT/'all_metrics.csv',index=False);rr.to_csv(FIXOUT/'all_resources.csv',index=False)
 pd.DataFrame(allmanifest).to_csv(FIXOUT/'fixed_split_manifests.csv',index=False)
 dump(FIXOUT/'skipped_experiments.json',allskipped)
 keys=['heldout_family','policy','feature_set','method','target_val_fpr']
 s=mm.groupby(keys)[['unknown_recall','test_fpr','attack_f1','auroc','auprc','precision']].agg(['mean','std']).reset_index()
 s.columns=['_'.join([a for a in tup if a]) for tup in s.columns]
 s.to_csv(FIXOUT/'summary_mean_std.csv',index=False)
 # Paired deltas on the immutable test samples, by seed and operating point.
 baseline=mm[(mm.policy=='none')&(mm.feature_set=='expanded36')][['heldout_family','seed','method','target_val_fpr','unknown_recall','test_fpr','auroc','auprc']]
 compared=mm.merge(baseline,on=['heldout_family','seed','method','target_val_fpr'],suffixes=('','_baseline'))
 for metric in ['unknown_recall','test_fpr','auroc','auprc']:
  compared['delta_'+metric]=compared[metric]-compared[metric+'_baseline']
 compared.to_csv(FIXOUT/'paired_deltas_fixed_test.csv',index=False)
 try:
  import matplotlib
  matplotlib.use('Agg');import matplotlib.pyplot as plt
  z=s[(s.policy=='none')&(s.feature_set=='expanded36')&(s.target_val_fpr==.05)]
  fig,ax=plt.subplots(figsize=(10,5))
  for label,g in z.groupby('method'):ax.plot(g.heldout_family,g.unknown_recall_mean,label=label,marker='o')
  ax.set_ylabel('Unknown recall @ 5% validation FPR');ax.set_ylim(0,1);ax.legend(fontsize=8);fig.tight_layout()
  fig.savefig(FIXOUT/'fixed_test_model_comparison.png',dpi=140);plt.close(fig)
 except Exception as ex:print('Plot creation skipped',repr(ex),flush=True)
 dump(FIXOUT/'run_status.json',dict(status='complete',expected=expected,completed=completed,skipped=len(allskipped),
  folds=len(FIXCFG['heldouts'])*len(FIXCFG['seeds']),test_invariance_validated=True,
  notes=['PooledLightGBM is a centralized reference, not a federated baseline',
  'Test examples and validation examples are held fixed across policies and feature sets',
  'No attack family held out for test is present in training or validation',
  'No session/capture provenance is guaranteed by the selected processed CSV',
  'Heldout positive count is capped, so precision and AUPRC reflect artificial prevalence',
  'Actual test FPR is always reported; validation target FPR is not a test guarantee']))
 with zipfile.ZipFile(FIXROOT/'EdgeIIoT_FixedTest_Results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
  for p in FIXOUT.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(FIXROOT))
 print('FIXED_TEST_COMPLETED',completed,'/',expected,'ZIP',FIXROOT/'EdgeIIoT_FixedTest_Results.zip',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','worker','aggregate'],required=True)
 p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
 a=p.parse_args()
 try:
  if a.mode=='prepare':fixed_prepare()
  elif a.mode=='worker':fixed_worker(a.worker,a.workers)
  else:fixed_aggregate(a.workers)
 except Exception:traceback.print_exc();sys.exit(1)
