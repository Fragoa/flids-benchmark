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
OUT=ROOT/'FLIDS_Experiment01_Results'; OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'flids_exp01_cache.pkl'
CFG=dict(seeds=[0,42,3407],families=['Port_Scanning','DDoS_UDP','SQL_injection','Ransomware'],
 n_clients=5,rounds=5,steps=8,batch_size=128,lr=0.0015,hidden=24,embedding=12,bases=5,
 max_scan_rows=None,audit_sample_probability=0.16,chunk_size=8000,pool_cap_per_family=5000,benign_cap=14000,
 threshold_targets=[0.001,0.01,0.05],dirichlet_alpha=0.5,
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
 expanded,excluded=(safe,[c for c in nonlabel if c not in safe])
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


# Experiment 01: immutable audit, grouped open-set folds, validation-only calibration.
# The legacy helper above is included to keep this file fully standalone.
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import shutil

EXPERIMENT = dict(
    title='Leakage-Audited Open-Set FL-IDS Benchmark',
    seeds=[0,42,3407,123,2026],
    heldouts=['Port_Scanning','DDoS_UDP','SQL_injection','Ransomware'],
    methods=['PooledMLP','PooledLightGBM','LocalMLP','FedAvgMLP','FedProxMLP'],
    n_clients=5, rounds=30, local_epochs=1, batch_size=256,
    client_round_cap=1024, lr=0.001, prox_mu=0.01,
    dirichlet_alpha=0.5, target_fprs=[0.001,0.01,0.05],
    val_frac=0.20, benign_test_frac=0.22,
    min_benign_val_for_001=1000,
    archive='FLIDS_Experiment01_Results.zip',
    design='simulation: no independent capture/session IDs were verified; grouped by 64-bit complete feature signature',
)
if os.getenv('FLIDS_SMOKE')=='1':
    EXPERIMENT.update(seeds=[0],heldouts=['Port_Scanning','SQL_injection'],n_clients=3,rounds=2,client_round_cap=256)
OUT=ROOT/'FLIDS_Experiment01_Results'
OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'flids_exp01_cache.pkl'

def sha_ids(ix):
    return hashlib.sha256(np.sort(np.asarray(ix,dtype=np.int64)).tobytes()).hexdigest()

def parse_folds(p):
    df=p['df'].reset_index(drop=True)
    lab=df.family.astype(str).to_numpy()
    y=df.binary_label.to_numpy(dtype=np.int8)
    sig=df._full_sig.to_numpy(dtype=np.uint64)
    # Conflicting exact signatures are excluded from ALL splits, never arbitrarily relabeled.
    stats=df.groupby('_full_sig',sort=False).agg(labels=('binary_label','nunique'),families=('family','nunique'))
    bad=stats.index[(stats.labels>1)|(stats.families>1)].to_numpy(dtype=np.uint64)
    valid=np.flatnonzero(~np.isin(sig,bad))
    folds={}; man=[]; audits=[]
    for held in EXPERIMENT['heldouts']:
        for seed in EXPERIMENT['seeds']:
            unknown=valid[(lab[valid]==held)&(y[valid]==1)]
            benign=valid[y[valid]==0]
            known=valid[(lab[valid]!=held)&(y[valid]==1)]
            # Disjoint full signatures across classes and held-out attack families.
            reserved=np.unique(sig[unknown])
            benign=benign[~np.isin(sig[benign],reserved)]
            known=known[~np.isin(sig[known],reserved)]
            known=known[~np.isin(sig[known],np.unique(sig[benign]))]
            if len(unknown)<25 or len(benign)<500 or len(known)<100:
                raise RuntimeError(f'INFEASIBLE_FOLD {held}, seed={seed}, counts={(len(unknown),len(benign),len(known))}')
            def split(indices,fraction,s):
                sp=GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=s)
                ia,ib=next(sp.split(indices,groups=sig[indices]));return indices[ia],indices[ib]
            ben_source,ben_test=split(benign,EXPERIMENT['benign_test_frac'],seed+11)
            ben_tr,ben_val=split(ben_source,EXPERIMENT['val_frac'],seed+12)
            kn_tr,kn_val=split(known,EXPERIMENT['val_frac'],seed+13)
            tr=np.sort(np.r_[ben_tr,kn_tr]);va=np.sort(np.r_[ben_val,kn_val]);te=np.sort(np.r_[ben_test,unknown]);
            if len(np.unique(y[tr]))<2 or len(np.unique(y[va]))<2:raise RuntimeError('SOURCE_BINARY_LABEL_MISSING')
            for a,b in ((tr,va),(tr,te),(va,te)):
                if np.intersect1d(sig[a],sig[b]).size:raise AssertionError('SIGNATURE_LEAK')
            if np.any(lab[tr]==held) or np.any(lab[va]==held):raise AssertionError('HELDOUT_LEAK')
            key=(held,seed)
            folds[key]=dict(train=tr,val=va,test=te,held=held,seed=seed)
            for part,ix in [('train',tr),('validation',va),('test',te)]:
                for row in ix:
                    man.append(dict(heldout_family=held,seed=seed,split=part,row_id=int(row),family=str(lab[row]),binary_label=int(y[row]),full_signature=f'{int(sig[row]):016x}'))
            audits.append(dict(heldout_family=held,seed=seed,train=len(tr),validation=len(va),test=len(te),
                benign_val=int((y[va]==0).sum()),benign_test=len(ben_test),unknown_test=len(unknown),
                sha_train=sha_ids(tr),sha_validation=sha_ids(va),sha_test=sha_ids(te),
                full_signature_leakage=0,heldout_train_count=0,heldout_val_count=0,
                proxy_split_warning='Full row signature only; no capture/device independence established'))
    pd.DataFrame(man).to_csv(OUT/'split_manifest.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'split_audit.csv',index=False)
    dump(OUT/'split_index.json',{f'{k[0]}__{k[1]}':{s:v[s].tolist() for s in ('train','val','test')} for k,v in folds.items()})
    return folds

def prepare_exp():
    # Inherited full-file streaming reads the author CSV and caches float32 Safe30 once.
    print('EXP01_PREPARE_START',flush=True)
    p=prepare()
    df=p['df']
    if len(p['safe'])!=30 and os.getenv('FLIDS_SMOKE')!='1':raise RuntimeError(f'Expected Safe30; observed {len(p["safe"])} columns. Audit requires 30.')
    folds=parse_folds(p)
    pd.DataFrame([dict(a, scan_scope='random sample spanning full CSV; counts are NOT complete source duplicate totals') for a in p['audit']]).to_csv(OUT/'dataset_audit.csv',index=False)
    # Read-only config and data provenance: no test-label-informed hyperparameter selection.
    dump(OUT/'experiment_config.json',EXPERIMENT)
    audit=json.loads((OUT/'dataset_provenance.json').read_text())
    audit['explicit_protocol_caveats']=[
        'Full-source scan, but capped stratified reservoir used to train/test: nonrepresentative prevalence',
        'Dataset audit signatures computed on a randomized subset; no complete all-row dedup census',
        'Full-signature content grouping is only a proxy for independent capture/session splits',
        'The benchmark is binary known-vs-unknown attack detection; attack categories are not modeled as open-set classes',
        'Validation threshold is based only on benign validation scores, not source attacks or heldout attacks',
        'Prespecified training configuration; pilot is software smoke check, not test-data-based tuning',
    ]
    dump(OUT/'dataset_provenance.json',audit)
    print('EXP01_PREPARE_READY',len(df),'sampled rows',len(folds),'folds',len(p['safe']),'features',flush=True)

def fit_features(p,fold):
    raw=p['numeric_safe'].astype(np.float32,copy=True)
    raw[~np.isfinite(raw)]=np.nan
    tr=fold['train']
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    tx=imp.fit_transform(raw[tr]); xx=imp.transform(raw)
    sc=RobustScaler(quantile_range=(10,90));sc.fit(tx)
    xx=np.clip(sc.transform(xx),-12,12).astype(np.float32)
    if not np.isfinite(xx).all():raise RuntimeError('NONFINITE_PREPROCESSED_INPUT')
    return xx

def make_clients(tr,y,seed,clients):
    rng=np.random.default_rng(seed+911)
    buckets=[[] for _ in range(clients)]
    for label in (0,1):
        a=np.asarray(tr)[y[tr]==label].copy();rng.shuffle(a)
        probs=rng.dirichlet(np.ones(clients)*EXPERIMENT['dirichlet_alpha'])
        sizes=rng.multinomial(len(a),probs)
        for cid,sub in enumerate(np.split(a,np.cumsum(sizes)[:-1])):buckets[cid].extend(sub.tolist())
    return [np.asarray(b,dtype=np.int64) for b in buckets]

class MLP(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def fit_steps(model,base_state,X,y,idx,round_id,cid,seed,device,prox=False):
    model.load_state_dict(base_state)
    model.to(device).train()
    opt=torch.optim.Adam(model.parameters(),lr=EXPERIMENT['lr'])
    local_rng=np.random.default_rng(seed*31+round_id*17011+cid*167+9)
    cap=min(len(idx),EXPERIMENT['client_round_cap'])
    steps=max(1,int(np.ceil(cap/EXPERIMENT['batch_size'])))*EXPERIMENT['local_epochs']
    full_global=[w.detach().to(device).clone() for w in base_state.values()]
    for step in range(steps):
        picked=local_rng.choice(idx,size=min(EXPERIMENT['batch_size'],len(idx)),replace=len(idx)<EXPERIMENT['batch_size'])
        xb=torch.as_tensor(X[picked],device=device);yb=torch.as_tensor(y[picked].astype(np.float32),device=device)
        opt.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(model(xb),yb)
        if prox:
            penalty=torch.zeros((),device=device)
            for par,g in zip(model.parameters(),full_global):penalty=penalty+(par-g).square().sum()
            loss=loss+.5*EXPERIMENT['prox_mu']*penalty
        loss.backward();opt.step()
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.cpu()
    return state,steps

@torch.no_grad()
def infer_mlp(model,X,idx,device):
    model.to(device).eval();pieces=[]
    for block in np.array_split(idx,max(1,int(np.ceil(len(idx)/2048)))):
        if len(block):pieces.append(torch.sigmoid(model(torch.as_tensor(X[block],device=device))).cpu().numpy())
    model.cpu();return np.concatenate(pieces)

def fit_method(method,X,y,fold,clients,seed,device,rounds):
    tr=fold['train'];va=fold['val'];te=fold['test'];start=time.perf_counter()
    total_steps=0;bytes_comm=0;local_details=[]
    if method=='PooledLightGBM':
        import lightgbm as lgb
        clf=lgb.LGBMClassifier(n_estimators=130,num_leaves=15,max_depth=5,learning_rate=.05,
            min_child_samples=15,reg_lambda=1,verbosity=-1,n_jobs=1,random_state=seed)
        clf.fit(X[tr],y[tr])
        sv=clf.predict_proba(X[va])[:,1];st=clf.predict_proba(X[te])[:,1];total_steps=130
        params='130 trees, max depth 5; central raw-data pooling'
    else:
        torch.manual_seed(seed)
        initial=MLP(X.shape[1]);state={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}
        paramcount=sum(t.numel() for t in initial.parameters()); params=f'{paramcount} neural parameters'
        if method=='PooledMLP':
            # Same approximate sum of local gradient update steps as FL (but different data access).
            model=MLP(X.shape[1]);model.load_state_dict(state)
            stepbudget=rounds*EXPERIMENT['n_clients']*max(1,int(np.ceil(EXPERIMENT['client_round_cap']/EXPERIMENT['batch_size'])))
            # Fit once on the whole pooled train without FedAvg resets.
            saved_epochs=EXPERIMENT['local_epochs']
            for block in range(rounds*EXPERIMENT['n_clients']):
                state,steps=fit_steps(model,state,X,y,tr,block,0,seed,device,False);total_steps+=steps
            model.load_state_dict(state)
            sv=infer_mlp(model,X,va,device);st=infer_mlp(model,X,te,device)
        elif method=='LocalMLP':
            vs=[];ts=[]
            for cid,idx in enumerate(clients):
                if len(idx)<10:raise RuntimeError('LOCAL_CLIENT_TOO_SMALL')
                model=MLP(X.shape[1]);cur={k:v.detach().cpu().clone() for k,v in state.items()}
                # Same per-client training budget as a FL client, without communication.
                for r in range(rounds):cur,steps=fit_steps(model,cur,X,y,idx,r,cid,seed,device);total_steps+=steps
                model.load_state_dict(cur)
                v=infer_mlp(model,X,va,device);t=infer_mlp(model,X,te,device)
                vs.append(v);ts.append(t)
                benign_val=v[y[va]==0]
                if len(benign_val):
                    local_thr,local_val_fpr=threshold_for_target(benign_val,0.01)
                    local_pred=np.asarray(t,dtype=np.float64)>=local_thr
                    local_recall=float(np.mean(local_pred[y[te]==1])) if np.any(y[te]==1) else np.nan
                    local_fpr=float(np.mean(local_pred[y[te]==0])) if np.any(y[te]==0) else np.nan
                else:
                    local_val_fpr=local_recall=local_fpr=float('nan')
                local_details.append(dict(client=cid,train_samples=len(idx),test_auc=(float(roc_auc_score(y[te],t)) if len(np.unique(y[te]))==2 else np.nan),test_unknown_recall_at_val_1pct_fpr=local_recall,test_fpr_at_val_1pct_fpr=local_fpr,validation_fpr=local_val_fpr))
            sv=np.mean(vs,axis=0);st=np.mean(ts,axis=0)
            params+='; local ensemble of 5 models; inference score averaging; deployment communication not counted'
        else:
            for r in range(rounds):
                states=[];weights=[]
                for cid,idx in enumerate(clients):
                    if len(idx)<10:continue
                    model=MLP(X.shape[1]);local_state,steps=fit_steps(model,state,X,y,idx,r,cid,seed,device,method=='FedProxMLP')
                    states.append(local_state);weights.append(len(idx));total_steps+=steps
                if not states:raise RuntimeError('NO_ACTIVE_CLIENTS')
                weight=np.asarray(weights,dtype=float);weight/=weight.sum()
                state={k:sum(s[k]*float(w) for s,w in zip(states,weight)) for k in state}
                # Count client uploads and global broadcasts (approx float32 weights only).
                bytes_comm+=2*paramcount*4*len(states)
            model=MLP(X.shape[1]);model.load_state_dict(state)
            sv=infer_mlp(model,X,va,device);st=infer_mlp(model,X,te,device)
    return sv,st,dict(seconds=time.perf_counter()-start,steps=int(total_steps),communication_bytes=int(bytes_comm),parameters=params,local_clients=local_details)

def threshold_for_target(val_neg,target):
    """Choose the least restrictive finite cutoff meeting an empirical FPR budget.

    All comparisons use float64. NumPy can silently cast a Python float64
    threshold down to float32 when comparing with float32 predictions, which
    re-includes the score tied at the threshold. This caused Kaggle pilot
    AssertionError for neural network outputs. The caller must use float64
    for its test-set comparison as well.

    If ties mean no nonzero FPR is feasible, return a cutoff above the maximum
    validation score (observed FPR 0), without random tie-breaking.
    """
    values=np.asarray(val_neg,dtype=np.float64)
    if values.ndim!=1 or values.size==0 or not np.isfinite(values).all():
        raise ValueError('VALIDATION_NEGATIVE_SCORES_INVALID')
    if not (0<=target<1):
        raise ValueError(f'INVALID_TARGET_FPR {target}')
    order=np.sort(values)
    k=int(np.floor(target*len(order)))
    pivot=order[-k] if k else order[-1]
    threshold=float(np.nextafter(pivot,np.inf))
    realized=float(np.mean(values>=threshold))
    if realized>target+1e-12:
        raise RuntimeError(f'FPR_THRESHOLD_INVARIANT_FAILED target={target} realized={realized} n={len(values)}')
    return threshold,realized

def jobs(pilot=False):
    methods=EXPERIMENT['methods'];holds=EXPERIMENT['heldouts'];seeds=EXPERIMENT['seeds']
    if pilot: holds=holds[:1];seeds=seeds[:1]
    return [(h,s,m) for h in holds for s in seeds for m in methods]

def worker(mode,worker_id,workers):
    p=pd.read_pickle(CACHE)
    folds=json.loads((OUT/'split_index.json').read_text())
    df=p['df'].reset_index(drop=True);y=df.binary_label.to_numpy(dtype=np.int8)
    pilot=(mode=='pilot')
    tasks=[t for i,t in enumerate(jobs(pilot)) if i%workers==worker_id]
    rounds=2 if pilot else EXPERIMENT['rounds']
    dev=f'cuda:{worker_id%torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
    metric_path=OUT/f'{mode}_worker_{worker_id}_metrics.csv'
    res_path=OUT/f'{mode}_worker_{worker_id}_resources.csv'
    pred_path=OUT/f'{mode}_worker_{worker_id}_predictions.csv'
    client_path=OUT/f'{mode}_worker_{worker_id}_clients.csv'
    fail_path=OUT/f'{mode}_worker_{worker_id}_failures.json'
    # Previous failed attempts are not sticky: retry unfinished tasks on rerun.
    # The current failure report is authoritative for the current invocation.
    errors=[]
    if fail_path.exists():
        old_errors=json.loads(fail_path.read_text())
        if old_errors:
            dump(OUT/f'{mode}_worker_{worker_id}_previous_failures.json',old_errors)
        dump(fail_path,[])
    completed=set()
    if metric_path.exists() and metric_path.stat().st_size:
        for a in pd.read_csv(metric_path,usecols=['heldout_family','seed','method']).itertuples(index=False):completed.add((str(a[0]),int(a[1]),str(a[2])))
    def append_csv(path,rows):
        if not rows:return
        x=pd.DataFrame(rows);x.to_csv(path,mode='a',header=not path.exists() or path.stat().st_size==0,index=False)
    for count,(held,seed,method) in enumerate(tasks,1):
        if (held,seed,method) in completed:
            print('RESUME_SKIP',mode,held,seed,method,flush=True);continue
        try:
            ix=folds[f'{held}__{seed}'];fold={k:np.asarray(v,dtype=np.int64) for k,v in ix.items()}
            X=fit_features(p,fold)
            clients=make_clients(fold['train'],y,seed,EXPERIMENT['n_clients'])
            if min(map(len,clients))<10:raise RuntimeError(f'INFEASIBLE_CLIENT_ALLOCATION {[len(a) for a in clients]}')
            sv,st,info=fit_method(method,X,y,fold,clients,seed,dev,rounds)
            vy=y[fold['val']];ty=y[fold['test']]
            neg=sv[vy==0]
            auroc=float(roc_auc_score(ty,st));auprc=float(average_precision_score(ty,st))
            metric=[]
            for target in EXPERIMENT['target_fprs']:
                if target==.001 and len(neg)<EXPERIMENT['min_benign_val_for_001']:
                    metric.append(dict(heldout_family=held,seed=seed,method=method,target_fpr=target,
                        status='insufficient_benign_validation',n_benign_val=len(neg),test_sha=sha_ids(fold['test'])))
                    continue
                thr,valfpr=threshold_for_target(neg,target)
                pred=np.asarray(st,dtype=np.float64)>=thr
                tp=int(np.sum(pred&(ty==1)));fp=int(np.sum(pred&(ty==0)))
                fn=int(np.sum(~pred&(ty==1)));tn=int(np.sum(~pred&(ty==0)))
                metric.append(dict(heldout_family=held,seed=seed,method=method,target_fpr=target,
                    status='evaluated',n_benign_val=len(neg),validation_fpr=valfpr,threshold=thr,
                    test_sha=sha_ids(fold['test']),validation_sha=sha_ids(fold['val']),
                    unknown_recall=tp/max(1,tp+fn),test_fpr=fp/max(1,fp+tn),
                    precision=tp/max(1,tp+fp),attack_f1=2*tp/max(1,2*tp+fp+fn),
                    auroc=auroc,auprc=auprc,tp=tp,fp=fp,fn=fn,tn=tn,n_test=len(ty)))
            # Write metrics LAST. They are the durable per-task completion marker.
            append_csv(res_path,[dict(heldout_family=held,seed=seed,method=method,
                train_rows=len(fold['train']),validation_rows=len(fold['val']),test_rows=len(fold['test']),
                client_sizes=json.dumps(list(map(len,clients))),train_seconds=info['seconds'],
                optimizer_steps=info['steps'],communication_bytes=info['communication_bytes'],
                parameters=info['parameters'],pilot=pilot)])
            append_csv(pred_path,[dict(heldout_family=held,seed=seed,method=method,row_id=int(i),
                true_label=int(y[i]),family=str(df.family.iloc[i]),attack_score=float(v)) for i,v in zip(fold['test'],st)])
            append_csv(client_path,[dict(heldout_family=held,seed=seed,method=method,**d) for d in info['local_clients']])
            append_csv(metric_path,metric)
            print('TASK_OK',mode,worker_id,count,'/',len(tasks),held,seed,method,'AUROC',round(auroc,4),flush=True)
        except Exception:
            errors.append(dict(heldout_family=held,seed=seed,method=method,error=traceback.format_exc()))
            dump(fail_path,errors)
            print('TASK_FAILED',mode,worker_id,held,seed,method,errors[-1]['error'][-1200:],flush=True)
    dump(fail_path,errors)
    if errors:raise RuntimeError(f'{len(errors)} errors in {mode} worker {worker_id}; inspect failures JSON')

def validate_gate():
    expected=len(jobs(True))
    have=[]
    for w in range(2):
        x=OUT/f'pilot_worker_{w}_metrics.csv'
        if x.exists() and x.stat().st_size:have.append(pd.read_csv(x))
    if not have:raise RuntimeError('PILOT_EMPTY')
    x=pd.concat(have,ignore_index=True)
    count=x[['heldout_family','seed','method']].drop_duplicates().shape[0]
    if count!=expected:raise RuntimeError(f'PILOT_INCOMPLETE: {count}/{expected}')
    if x.loc[x.status=='evaluated','auroc'].isna().any():raise RuntimeError('PILOT_INVALID_AUROC')
    dump(OUT/'pilot_gate.json',dict(status='passed',expected=expected,observed=count,description='Software sanity gate only; not a statistical performance target.'))
    print('PILOT_GATE_PASSED',count,flush=True)

def aggregate():
    mats=[];ress=[];preds=[];clients=[];fails=[]
    for w in range(2):
        for name,dest in [('metrics',mats),('resources',ress),('predictions',preds),('clients',clients)]:
            path=OUT/f'main_worker_{w}_{name}.csv'
            if path.exists() and path.stat().st_size:dest.append(pd.read_csv(path))
        p=OUT/f'main_worker_{w}_failures.json'
        if p.exists():fails.extend(json.loads(p.read_text()))
    if fails:raise RuntimeError(f'AGGREGATION_BLOCKED_BY_ERRORS {len(fails)}')
    if not mats:raise RuntimeError('NO_MAIN_RESULTS')
    mm=pd.concat(mats,ignore_index=True);rr=pd.concat(ress,ignore_index=True)
    unique=mm[['heldout_family','seed','method']].drop_duplicates()
    expected=len(jobs(False))
    if len(unique)!=expected:raise RuntimeError(f'INCOMPLETE {len(unique)}/{expected}')
    if len(mm)!=expected*len(EXPERIMENT['target_fprs']):raise RuntimeError('INCOMPLETE_THRESHOLD_METRICS')
    assert (mm.groupby(['heldout_family','seed'])['test_sha'].nunique()==1).all(), 'TEST_SET_CHANGED'
    assert (mm.loc[mm.status=='evaluated','validation_fpr']<=mm.loc[mm.status=='evaluated','target_fpr']+1e-12).all(), 'VALIDATION_FPR_EXCEEDS_TARGET'
    mm.to_csv(OUT/'experiment_results.csv',index=False)
    mm.to_csv(OUT/'low_fpr_results.csv',index=False)
    rr.to_csv(OUT/'communication_cost.csv',index=False)
    if preds:pd.concat(preds,ignore_index=True).to_csv(OUT/'predictions.csv',index=False)
    if clients:pd.concat(clients,ignore_index=True).to_csv(OUT/'client_results.csv',index=False)
    keys=['heldout_family','method','target_fpr']
    summary=mm[mm.status=='evaluated'].groupby(keys)[['unknown_recall','test_fpr','precision','attack_f1','auroc','auprc']].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(map(str,c)).strip('_') for c in summary.columns]
    summary.to_csv(OUT/'summary_mean_std.csv',index=False)
    piv=mm[mm.status=='evaluated'].pivot_table(index=['heldout_family','seed','target_fpr'],columns='method',values=['unknown_recall','test_fpr','auroc','auprc'])
    deltas=[]
    for (held,seed,target),row in piv.iterrows():
        for metric in ['unknown_recall','test_fpr','auroc','auprc']:
            for baseline in ('LocalMLP','PooledMLP'):
                for method in ('FedAvgMLP','FedProxMLP'):
                    try:
                        a=float(row[(metric,method)]);b=float(row[(metric,baseline)])
                        if np.isfinite(a) and np.isfinite(b):
                            deltas.append(dict(heldout_family=held,seed=seed,target_fpr=target,metric=metric,method=method,baseline=baseline,delta=a-b))
                    except KeyError:pass
    pd.DataFrame(deltas).to_csv(OUT/'paired_comparisons.csv',index=False)
    dump(OUT/'run_status.json',dict(status='complete',expected_model_runs=expected,completed_model_runs=len(unique),
        evaluated_operating_points=int((mm.status=='evaluated').sum()),unavailable_operating_points=int((mm.status!='evaluated').sum()),
        leakage_checks_passed=True,limitations=EXPERIMENT['design']))
    # Bundle dataset audit, manifest, raw predictions, failures, and pilot evidence.
    archive=ROOT/EXPERIMENT['archive']
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in OUT.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(ROOT))
    print('ARCHIVE_READY',archive,archive.stat().st_size,'model_runs',len(unique),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['prepare','pilot','gate','main','aggregate'])
    ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    if args.mode=='prepare':prepare_exp()
    elif args.mode=='gate':validate_gate()
    elif args.mode=='aggregate':aggregate()
    else:worker(args.mode,args.worker,args.workers)
if __name__=='__main__':main()
