"""Experiment 03: source-only threshold transfer, calibration, and feature shortcut audit.
Strict leave-one-source-dataset-out NetFlow V2 protocol. Target labels are used only
for final evaluation and explicitly marked non-deployable oracle diagnostics.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,hashlib,json,time,traceback,zipfile,faulthandler,random
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score,average_precision_score
faulthandler.enable(all_threads=True)
ROOT=Path(os.getenv('FLIDS_EXP03_ROOT','/kaggle/working' if Path('/kaggle/working').exists() else '/mnt/data/exp02_local'))
OUT=ROOT/'FLIDS_Experiment03_Results';OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'flids_exp03_cache.npz'
DOMAINS=['NF-UNSW-NB15','NF-ToN-IoT','NF-BoT-IoT','NF-CSE-CIC-IDS2018']
FEATURES=['L4_SRC_PORT','L4_DST_PORT','PROTOCOL','L7_PROTO','IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS','FLOW_DURATION_MILLISECONDS','TCP_FLAGS','CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS','DURATION_IN','DURATION_OUT','MIN_TTL','MAX_TTL','LONGEST_FLOW_PKT','SHORTEST_FLOW_PKT','MIN_IP_PKT_LEN','MAX_IP_PKT_LEN','SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_SECOND_BYTES','RETRANSMITTED_IN_BYTES','RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_BYTES','RETRANSMITTED_OUT_PKTS','SRC_TO_DST_AVG_THROUGHPUT','DST_TO_SRC_AVG_THROUGHPUT','NUM_PKTS_UP_TO_128_BYTES','NUM_PKTS_128_TO_256_BYTES','NUM_PKTS_256_TO_512_BYTES','NUM_PKTS_512_TO_1024_BYTES','NUM_PKTS_1024_TO_1514_BYTES','TCP_WIN_MAX_IN','TCP_WIN_MAX_OUT','ICMP_TYPE','ICMP_IPV4_TYPE','DNS_QUERY_ID','DNS_QUERY_TYPE','DNS_TTL_ANSWER','FTP_COMMAND_RET_CODE']
METHODS=['PooledMLP','PooledLightGBM','LocalMLP','FedAvgMLP','FedProxMLP']
FEATURE_VARIANTS={'all':[], 'no_ports':['L4_SRC_PORT','L4_DST_PORT'], 'no_ports_ttl':['L4_SRC_PORT','L4_DST_PORT','MIN_TTL','MAX_TTL']}
FEATURE_VARIANTS={k:[f for f in FEATURES if f not in drop] for k,drop in FEATURE_VARIANTS.items()}
CFG=dict(seeds=[0,42,3407,123,2026],domains=DOMAINS,methods=METHODS,features=FEATURES,source_clients=3,rounds=30,local_epochs=1,batch_size=256,client_round_cap=512,lr=.001,prox_mu=.01,target_fprs=[.001,.01,.05],minimum_benign_for_001=1000,max_rows_per_domain_class=12000,source_validation_fraction=.20,lightgbm_trees=130,archive='FLIDS_Experiment03_Results.zip',schema='UQ NetFlow V2; do not substitute Edge-IIoTset',feature_variants=FEATURE_VARIANTS,risk_notes=['Three different source datasets are FL clients, not actual organizational deployments','Stratified reservoir changes prevalence and reported precision/AUPRC','No target-domain examples or labels used in training, scaling, model selection, or thresholds','Source-only validation thresholds do not guarantee the same target FPR','Oracle target-label thresholds are retrospective upper-bound diagnostics, never deployable methods','Feature distribution shift statistics use target inputs post hoc only and never inform fitting or thresholding','LocalMLP means mean score ensemble of three separately trained domain models','Fixed score log transform is used, no target-fitted normalization','Monotonic score calibration cannot repair AUROC ranking and is not applied to threshold selection','Full numeric feature fingerprints are a row-overlap proxy, not session/capture independence','Centralized baselines require data pooling; recorded FL bytes omit network overhead'])
if os.getenv('FLIDS_EXP03_SMOKE')=='1':CFG.update(seeds=[0],domains=DOMAINS[:2],rounds=2,client_round_cap=128,max_rows_per_domain_class=220,lightgbm_trees=18,minimum_benign_for_001=1000)
if os.getenv('FLIDS_EXP03_SEEDS'):CFG['seeds']=[int(z) for z in os.environ['FLIDS_EXP03_SEEDS'].split(',')]

def save_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def safe_header(path):
 with open(path,'r',encoding='utf-8-sig',errors='replace',newline='') as f:return next(csv.reader(f))
def normalized(t):return ''.join(c for c in str(t).lower() if c.isalnum())
def canonical_domain(t):
 s=normalized(t)
 if 'unsw' in s:return DOMAINS[0]
 if 'ton' in s and 'iot' in s:return DOMAINS[1]
 if 'bot' in s and 'iot' in s:return DOMAINS[2]
 if 'cse' in s or 'cicids2018' in s:return DOMAINS[3]
 return None

def discover():
 roots=[Path('/kaggle/input'),ROOT/'input']
 if os.getenv('NF_UQ_DATA'):roots.insert(0,Path(os.environ['NF_UQ_DATA']))
 found=[];diagnostics=[]
 for root in roots:
  if not root.exists():continue
  paths=[root] if root.is_file() else list(root.rglob('*.csv'))
  for p in paths:
   try:
    head=safe_header(p);cols=set(head);has_features=set(FEATURES).issubset(cols)
    has_label=bool({'Label','label'}&cols)
    has_domain='Dataset' in cols or canonical_domain(p.name) is not None
    diagnostics.append(dict(path=str(p),columns=len(cols),valid_features=has_features,valid_label=has_label,domain_source=has_domain,missing_features=sorted(set(FEATURES)-cols)[:12]))
    if has_features and has_label and has_domain:found.append((p,head))
   except Exception as e:diagnostics.append(dict(path=str(p),error=str(e)[:180]))
 save_json(OUT/'input_discovery.json',dict(files=diagnostics,valid_candidates=[str(p) for p,_ in found]))
 combined=[v for v in found if normalized(v[0].name).startswith('nfuqnidsv2')]
 if combined:return combined[:1]
 selected=[];used=set()
 for p,h in sorted(found,key=lambda v:str(v[0])):
  domain=canonical_domain(p.name)
  if domain and domain not in used:selected.append((p,h));used.add(domain)
 if len(selected)==4:return selected
 if found and any('Dataset' in h for _,h in found):return [next(v for v in found if 'Dataset' in v[1])]
 return selected

def synthetic_fixture():
 p=ROOT/'synthetic_nf_exp03_SOFTWARE_TEST_ONLY.csv';ROOT.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(51)
 with p.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(FEATURES+['Label','Attack','Dataset'])
  for d,domain in enumerate(DOMAINS):
   for y in (0,1):
    for i in range(260):
     a=rng.gamma(2,1,len(FEATURES));a[0]=100+d*35+y*4;a[1]=200+d*11;a[4]+=y*(1.4+d*.2);a[5]+=y*.6;a[6]+=y*.8
     w.writerow([f'{v:.6f}' for v in a]+[y,'Attack' if y else 'Benign',domain+'-v2'])
 return p

def prepare():
 save_json(OUT/'experiment_config.json',CFG)
 if os.getenv('FLIDS_EXP03_SMOKE')=='1':p=synthetic_fixture();found=[(p,safe_header(p))]
 else:found=discover()
 if not found:
  raise FileNotFoundError('NF_UQ_NETFLOW_V2_NOT_FOUND: Attach the combined NF-UQ-NIDS-v2.csv or all four NF-v2 files in Kaggle Add Input; see input_discovery.json. Edge-IIoTset cannot be substituted. Set NF_UQ_DATA to a local CSV path if needed.')
 rng=np.random.default_rng(2077);reservoir=defaultdict(list);seen=Counter();invalid=Counter();files_report=[]
 csv.field_size_limit(10_000_000)
 for path,head in found:
  col={name:i for i,name in enumerate(head)}
  lcol=col.get('Label',col.get('label'))
  dcol=col.get('Dataset');acol=col.get('Attack')
  fixed=canonical_domain(path.name)
  good=bad=0
  with path.open('r',encoding='utf-8-sig',errors='replace',newline='',buffering=1024*1024) as f:
   reader=csv.reader(f);next(reader)
   for row in reader:
    if len(row)!=len(head):bad+=1;continue
    domain=canonical_domain(row[dcol]) if dcol is not None else fixed
    if domain not in DOMAINS:invalid['unknown_domain']+=1;continue
    label=normalized(row[lcol])
    if label in ('0','normal','benign'):y=0
    elif label in ('1','attack','malicious'):y=1
    else:invalid['invalid_binary_label']+=1;continue
    try:
     vals=np.array([float(row[col[c]]) if row[col[c]].strip() else 0. for c in FEATURES],dtype=np.float32)
     if not np.isfinite(vals).all():invalid['nonfinite_values_replaced']+=1;vals=np.nan_to_num(vals,nan=0.,posinf=0.,neginf=0.)
    except Exception:bad+=1;continue
    key=(domain,y);seen[key]+=1;n=seen[key]
    item=(vals,str(row[acol]) if acol is not None else '')
    buf=reservoir[key]
    if len(buf)<CFG['max_rows_per_domain_class']:buf.append(item)
    else:
     j=int(rng.integers(0,n))
     if j<len(buf):buf[j]=item
    good+=1
   
  files_report.append(dict(path=str(path),valid_rows=good,malformed_or_invalid_numeric=bad))
  print('SCANNED',path.name,good,'invalid',bad,flush=True)
 x=[];y=[];d=[];attack=[]
 for (domain,label),rows in sorted(reservoir.items()):
  for a,t in rows:x.append(a);y.append(label);d.append(domain);attack.append(t)
 if not x:raise RuntimeError('NO_VALID_SOURCE_ROWS')
 x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.int8);d=np.asarray(d);attack=np.asarray(attack)
 cov={f'{domain}|{label}':int(( (d==domain)&(y==label)).sum()) for domain in DOMAINS for label in (0,1)}
 min_count=100 if os.getenv('FLIDS_EXP03_SMOKE')!='1' else 50
 lacking=[k for k,v in cov.items() if v<min_count]
 save_json(OUT/'data_coverage.json',dict(pooled_rows=len(x),domain_class_counts=cov,minimum_samples_per_class=min_count,valid=not lacking,missing_or_small=lacking))
 if lacking:raise RuntimeError('DOMAIN_CLASS_COVERAGE_FAILED '+str(lacking))
 np.savez_compressed(CACHE,x=x,y=y,d=d,attack=attack)
 # Exact numeric feature rows are used only as a *proxy* for duplicate overlap.
 fingerprints=np.asarray([hashlib.blake2b(np.ascontiguousarray(row).tobytes(),digest_size=12).hexdigest() for row in x])
 np.save(ROOT/'flids_exp03_fingerprints.npy',fingerprints)
 collision_groups=0;cross_domain_groups=0
 uniq,inv=np.unique(fingerprints,return_inverse=True)
 groups=defaultdict(set)
 for idx,k in enumerate(inv):groups[int(k)].add(str(d[idx]))
 cross_domain_groups=sum(len(v)>1 for v in groups.values())
 save_json(OUT/'dataset_audit.json',dict(original_counts={f'{a}|{b}':int(v) for (a,b),v in seen.items()},sample_counts=cov,source_files=files_report,invalid=dict(invalid),source_rows_scanned=sum(r['valid_rows'] for r in files_report),sampled_rows=len(x),unique_numeric_fingerprints=len(uniq),duplicate_extra_rows=len(x)-len(uniq),cross_domain_fingerprint_groups=cross_domain_groups,duplicate_scope='capped sample, not full-source dedup',split_independence='source dataset + exact feature fingerprint grouping for source train/validation; not true session-level independence',features=FEATURES,sampling='uniform per domain and binary class; prevalence intentionally altered',source_domain_definition='four separately collected sources within common NF-UQ NetFlow-v2 schema'))
 manifests=[]
 for held in DOMAINS:
  for seed in CFG['seeds']:
   tr,va,te=split_indices(y,d,fingerprints,held,seed)
   for role,ids in [('train',tr),('validation',va),('test',te)]:
    manifests.append(dict(holdout=held,seed=seed,role=role,n=len(ids),benign=int(np.sum(y[ids]==0)),attack=int(np.sum(y[ids]==1)),index_digest=digest(ids),source_domains=';'.join(sorted(set(d[ids])))))
 pd.DataFrame(manifests).to_csv(OUT/'split_manifest.csv',index=False)
 print('EXP03_PREPARED',len(x),'rows, domain counts',cov,'folds',len(CFG['seeds'])*len(DOMAINS),flush=True)

def digest(ids):return hashlib.sha256(np.sort(np.asarray(ids,dtype=np.int64)).tobytes()).hexdigest()
def split_indices(y,d,fp,held,seed):
 tr=[];va=[];te=np.flatnonzero(d==held)
 for domain in DOMAINS:
  if domain==held:continue
  ids=np.flatnonzero(d==domain)
  groups=fp[ids];unique=np.unique(groups)
  if len(unique)<10:raise RuntimeError(f'INSUFFICIENT_UNIQUE_GROUPS {domain}')
  a,b=next(GroupShuffleSplit(n_splits=1,test_size=CFG['source_validation_fraction'],random_state=seed).split(ids,y[ids],groups))
  tr.extend(ids[a]);va.extend(ids[b])
 tr=np.asarray(tr,dtype=np.int64);va=np.asarray(va,dtype=np.int64)
 assert not np.intersect1d(tr,va).size and not np.intersect1d(tr,te).size
 assert set(d[tr])==set(DOMAINS)-{held} and set(d[te])=={held}
 # Numeric fingerprint group overlap across *source* domains may occur; report, do not hide.
 for role,ids in [('train',tr),('validation',va),('test',te)]:
  if len(set(y[ids]))!=2:raise RuntimeError(f'MISSING_BINARY_CLASS {held} {seed} {role}')
 return tr,va,te

def transform(raw):
 # Fixed, source-agnostic signed log transform; no statistics or target data fitted.
 x=np.asarray(raw,dtype=np.float64)
 return np.clip(np.sign(x)*np.log1p(np.abs(np.clip(x,-1e12,1e12))),-18,18).astype(np.float32)

def threshold(scores,target):
 scores=np.asarray(scores,dtype=np.float64)
 if not scores.size or not np.isfinite(scores).all():raise RuntimeError('BAD_BENIGN_VALIDATION_SCORES')
 order=np.sort(scores);k=int(np.floor(target*len(scores)))
 pivot=order[-k] if k else order[-1]
 t=float(np.nextafter(pivot,np.inf))
 actual=float(np.mean(scores>=t))
 if actual>target+1e-12:raise RuntimeError('THRESHOLD_INVARIANT_BROKEN')
 return t,actual

def seedall(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)
class MLP(nn.Module):
 def __init__(self,n):super().__init__();self.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def local_train(initial,x,y,idx,seed,steps,device,prox=False,global_state=None):
 model=MLP(x.shape[1]);model.load_state_dict(initial);model.to(device);model.train()
 opt=torch.optim.Adam(model.parameters(),lr=CFG['lr'])
 rng=np.random.default_rng(seed);global_weights=[w.detach().to(device) for w in initial.values()]
 for _ in range(steps):
  batch=rng.choice(idx,size=min(CFG['batch_size'],len(idx)),replace=len(idx)<CFG['batch_size'])
  xb=torch.as_tensor(x[batch],device=device);yb=torch.as_tensor(y[batch],device=device,dtype=torch.float32)
  opt.zero_grad(set_to_none=True)
  loss=F.binary_cross_entropy_with_logits(model(xb),yb)
  if prox:
   penalty=sum((p-g).square().sum() for p,g in zip(model.parameters(),global_weights))
   loss=loss+CFG['prox_mu']*.5*penalty
  loss.backward();opt.step()
 return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

@torch.no_grad()
def predict_model(state,x,idx,device):
 m=MLP(x.shape[1]);m.load_state_dict(state);m.to(device).eval();scores=[]
 for chunk in np.array_split(idx,max(1,int(np.ceil(len(idx)/4096)))):
  if len(chunk):scores.append(torch.sigmoid(m(torch.as_tensor(x[chunk],device=device))).cpu().numpy())
 return np.concatenate(scores).astype(np.float64)

def fit_method(method,x,y,d,train,val,test,seed,rounds,device):
 tic=time.perf_counter();communication=0;steps=0;local_stats=[]
 if method=='PooledLightGBM':
  from lightgbm import LGBMClassifier
  model=LGBMClassifier(n_estimators=CFG['lightgbm_trees'],num_leaves=15,max_depth=5,learning_rate=.05,min_child_samples=15,reg_lambda=1,verbosity=-1,n_jobs=1,random_state=seed)
  model.fit(x[train],y[train]);sv=model.predict_proba(x[val])[:,1];st=model.predict_proba(x[test])[:,1];steps=CFG['lightgbm_trees']
 else:
  seedall(seed);m=MLP(x.shape[1]);base={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
  params=sum(w.numel() for w in m.parameters())
  clients=[train[d[train]==domain] for domain in DOMAINS if (d[train]==domain).any()]
  if len(clients)!=3 or min(map(len,clients))<10:raise RuntimeError('CLIENT_DOMAIN_ALLOCATION_INVALID')
  step_per_client=[max(1,int(np.ceil(min(len(ids),CFG['client_round_cap'])/CFG['batch_size'])))*CFG['local_epochs'] for ids in clients]
  if method=='PooledMLP':
   s=base
   for r in range(rounds):
    for c,stepcount in enumerate(step_per_client):
     s=local_train(s,x,y,train,seed+r*300+c*79,stepcount,device)
     steps+=stepcount
   sv=predict_model(s,x,val,device);st=predict_model(s,x,test,device)
  elif method=='LocalMLP':
   vals=[];tests=[]
   for c,ids in enumerate(clients):
    s=base
    for r in range(rounds):
     s=local_train(s,x,y,ids,seed+r*300+c*79,step_per_client[c],device)
     steps+=step_per_client[c]
    pv=predict_model(s,x,val,device);pt=predict_model(s,x,test,device)
    vals.append(pv);tests.append(pt)
    local_stats.append(dict(client=c,source_domain=str(d[ids[0]]),train_samples=len(ids),auroc=float(roc_auc_score(y[test],pt)),auprc=float(average_precision_score(y[test],pt))))
   sv=np.mean(vals,axis=0);st=np.mean(tests,axis=0)
  else:
   s=base
   for r in range(rounds):
    states=[];sizes=[]
    for c,ids in enumerate(clients):
     states.append(local_train(s,x,y,ids,seed+r*300+c*79,step_per_client[c],device,prox=(method=='FedProxMLP')))
     sizes.append(len(ids));steps+=step_per_client[c]
    weights=np.asarray(sizes,dtype=float);weights/=weights.sum()
    s={name:sum(item[name]*float(w) for item,w in zip(states,weights)) for name in s}
    communication+=2*len(clients)*params*4
   sv=predict_model(s,x,val,device);st=predict_model(s,x,test,device)
 return np.asarray(sv,dtype=np.float64),np.asarray(st,dtype=np.float64),dict(train_seconds=time.perf_counter()-tic,optimizer_steps=steps,communication_bytes=communication,local_clients=local_stats)

def oracle_threshold(scores, labels, target):
    """NON-DEPLOYABLE: uses held-out target labels for retrospective comparison only."""
    neg=np.asarray(scores,dtype=np.float64)[np.asarray(labels)==0]
    if len(neg)<2:return float('inf'),float('nan')
    return threshold(neg,target)

def calc_scores(labels,scores,t):
    labels=np.asarray(labels); pred=np.asarray(scores,dtype=np.float64)>=float(t)
    tp=int(np.sum(pred&(labels==1))); fp=int(np.sum(pred&(labels==0)))
    tn=int(np.sum(~pred&(labels==0))); fn=int(np.sum(~pred&(labels==1)))
    return dict(attack_recall=tp/max(1,tp+fn),test_fpr=fp/max(1,fp+tn),
                precision=tp/max(1,tp+fp),attack_f1=2*tp/max(1,2*tp+fp+fn),tp=tp,fp=fp,tn=tn,fn=fn)

def source_domain_threshold(source_scores, source_labels, source_domains, target):
    """Conservative single deployment threshold from per-source benign quantiles."""
    ts=[]
    for domain in sorted(set(source_domains)):
        mask=(source_domains==domain)&(source_labels==0)
        if np.count_nonzero(mask)<30:raise RuntimeError('INSUFFICIENT_SOURCE_BENIGN_FOR_DOMAIN_THRESHOLD '+str(domain))
        ts.append(threshold(source_scores[mask],target)[0])
    return float(max(ts)),{str(d):float(t) for d,t in zip(sorted(set(source_domains)),ts)}

def score_distribution(scores, labels):
    scores=np.asarray(scores,dtype=np.float64);labels=np.asarray(labels)
    out={}
    for v,name in ((0,'benign'),(1,'attack')):
        z=scores[labels==v]
        if z.size:
            out.update({f'{name}_score_p{q}':float(np.percentile(z,q)) for q in (5,50,95)})
            out[f'{name}_score_mean']=float(np.mean(z))
    return out

def feature_shift_diagnostics(train_x,test_x,names,seed,holdout):
    """Post hoc unlabelled target feature shift only; no model tuning."""
    from scipy.stats import ks_2samp
    rng=np.random.default_rng(seed+187)
    a=train_x[rng.choice(len(train_x),min(4000,len(train_x)),replace=False)]
    b=test_x[rng.choice(len(test_x),min(4000,len(test_x)),replace=False)]
    out=[]
    for j,name in enumerate(names):
        ks=float(ks_2samp(a[:,j],b[:,j],method='asymp').statistic)
        out.append(dict(holdout=holdout,seed=seed,feature=name,ks_statistic=ks,
                        source_median=float(np.median(a[:,j])),target_median=float(np.median(b[:,j])),
                        interpretation='posthoc_unlabelled_target_shift_not_used_to_train'))
    return out

def calibration_metrics(labels, probs, bins=15):
    labels=np.asarray(labels,dtype=np.float64)
    probs=np.clip(np.asarray(probs,dtype=np.float64),0.0,1.0)
    brier=float(np.mean((probs-labels)**2))
    cuts=np.minimum(np.floor(probs*bins).astype(np.int32),bins-1)
    ece=0.0
    for b in range(bins):
        group=cuts==b
        if np.any(group):
            ece+=float(np.mean(group))*abs(float(np.mean(labels[group]))-float(np.mean(probs[group])))
    return dict(brier=brier,ece=float(ece))

def source_only_platt(val_scores,val_labels,test_scores):
    from sklearn.linear_model import LogisticRegression
    vals=np.clip(np.asarray(val_scores,dtype=np.float64),1e-6,1-1e-6)
    tests=np.clip(np.asarray(test_scores,dtype=np.float64),1e-6,1-1e-6)
    val_logit=np.log(vals/(1-vals)).reshape(-1,1)
    test_logit=np.log(tests/(1-tests)).reshape(-1,1)
    # Fit only with source-validation labels, never target labels.
    model=LogisticRegression(C=1.0,max_iter=300,random_state=0)
    model.fit(val_logit,val_labels)
    return model.predict_proba(val_logit)[:,1],model.predict_proba(test_logit)[:,1]

def tasks(pilot=False):
 ds=CFG['domains'][:1] if pilot else DOMAINS
 ss=CFG['seeds'][:1] if pilot else CFG['seeds']
 ms=['PooledLightGBM','FedAvgMLP'] if pilot else METHODS
 variants=['all'] if pilot else list(FEATURE_VARIANTS)
 return [(a,b,c,v) for a in ds for b in ss for c in ms for v in variants]

def worker(mode,worker_id,nworkers):
 data=np.load(CACHE);raw=data['x'];full_x=transform(raw);y=data['y'];d=data['d'];fp=np.load(ROOT/'flids_exp03_fingerprints.npy')
 pending=[task for i,task in enumerate(tasks(mode=='pilot')) if i%nworkers==worker_id]
 errors=[];device=f'cuda:{worker_id%torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
 for ix,(hold,seed,method,variant) in enumerate(pending,1):
  key=f'{mode}__{normalized(hold)}__{seed}__{method}__{variant}'
  result_path=OUT/'records'/f'{key}.json';pred_path=OUT/'predictions_raw'/f'{key}.npz'
  if result_path.exists() and pred_path.exists():print('RESUME_SKIP',key,flush=True);continue
  try:
   tr,va,te=split_indices(y,d,fp,hold,seed)
   names=FEATURE_VARIANTS[variant];col=[FEATURES.index(z) for z in names];x=full_x[:,col]
   sv,st,meta=fit_method(method,x,y,d,tr,va,te,seed,(2 if mode=='pilot' else CFG['rounds']),device)
   label=y[te];source_labels=y[va];source_domains=d[va];neg=sv[source_labels==0]
   points=[];score_audit=[]
   for target in CFG['target_fprs']:
    if target==.001 and len(neg)<CFG['minimum_benign_for_001']:
     points.append(dict(target_fpr=target,status='insufficient_source_benign_validation',n_source_benign=len(neg)));continue
    thr,vfpr=threshold(neg,target)
    stable_thr,per_source=source_domain_threshold(sv,source_labels,source_domains,target)
    oracle_thr,oracle_vfpr=oracle_threshold(st,label,target)
    for strategy,t,source_only in [('pooled_source_quantile',thr,True),('worst_source_quantile',stable_thr,True),('target_oracle_POSTHOC',oracle_thr,False)]:
     metric=calc_scores(label,st,t)
     points.append(dict(target_fpr=target,status='evaluated',threshold_strategy=strategy,
           deployable_source_only=bool(source_only),threshold=t,source_validation_fpr=float(np.mean(neg>=t)),
           **metric))
    score_audit.append(dict(target_fpr=target,source_pooled_threshold=thr,source_fpr=vfpr,
       worst_source_threshold=stable_thr,per_source_thresholds=per_source,
       target_oracle_threshold_POSTHOC=oracle_thr,oracle_target_fpr=oracle_vfpr,
       source_only_target_fpr=calc_scores(label,st,thr)['test_fpr'],
       oracle_recall_upper_bound_POSTHOC=calc_scores(label,st,oracle_thr)['attack_recall']))
   source_platt,target_platt=source_only_platt(sv,source_labels,st)
   calibration={
      'source_raw':calibration_metrics(source_labels,sv),
      'target_raw_POSTHOC':calibration_metrics(label,st),
      'source_platt_fit_on_source_validation':calibration_metrics(source_labels,source_platt),
      'target_platt_POSTHOC':calibration_metrics(label,target_platt),
      'note':'Platt mapping is fitted only on source validation. Target labels are used solely for evaluation. The source in-sample calibrated ECE may be optimistic; no target adaptation is performed.'}
   record=dict(holdout=hold,seed=seed,method=method,feature_variant=variant,feature_names=names,mode=mode,
      source_domains=[z for z in DOMAINS if z!=hold],n_train=len(tr),n_validation=len(va),n_test=len(te),
      n_target_benign=int(np.sum(label==0)),n_target_attacks=int(np.sum(label==1)),
      test_sha=digest(te),validation_sha=digest(va),
      auroc=float(roc_auc_score(label,st)),auprc=float(average_precision_score(label,st)),
      source_auroc=float(roc_auc_score(source_labels,sv)),source_auprc=float(average_precision_score(source_labels,sv)),
      source_distribution=score_distribution(sv,source_labels),target_distribution_POSTHOC=score_distribution(st,label),
      threshold_audit=score_audit,calibration=calibration,**meta,operating_points=points)
   pred_path.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(pred_path,test_ids=te,test_labels=label,test_scores=st.astype(np.float32),
      validation_ids=va,validation_labels=source_labels,validation_scores=sv.astype(np.float32),
      source_validation_domains=source_domains,source_platt_scores=source_platt.astype(np.float32),target_platt_scores=target_platt.astype(np.float32))
   result_path.parent.mkdir(parents=True,exist_ok=True);save_json(result_path,record)
   if method=='PooledLightGBM':
    shifted=feature_shift_diagnostics(full_x[tr],full_x[te],FEATURES,seed,hold)
    pp=OUT/'feature_shift'/f'{normalized(hold)}__{seed}.json';pp.parent.mkdir(parents=True,exist_ok=True)
    if not pp.exists():save_json(pp,shifted)
   print('TASK_OK',mode,worker_id,ix,'/',len(pending),hold,seed,method,variant,'AUROC',round(record['auroc'],4),flush=True)
  except Exception:
   msg=traceback.format_exc();errors.append(dict(holdout=hold,seed=seed,method=method,variant=variant,error=msg))
   print('TASK_FAILED',hold,seed,method,variant,msg[-1400:],flush=True)
 save_json(OUT/f'{mode}_worker_{worker_id}_failures.json',errors)
 if errors:raise RuntimeError(f'{len(errors)} {mode} tasks failed; inspect failures JSON')

def gate():
 paths=list((OUT/'records').glob('pilot__*.json')) if (OUT/'records').exists() else []
 if len(paths)!=len(tasks(True)):raise RuntimeError(f'PILOT_GATE_INCOMPLETE {len(paths)}/{len(tasks(True))}')
 data=[json.loads(p.read_text()) for p in paths]
 if any(not np.isfinite(r['auroc']) for r in data):raise RuntimeError('PILOT_BAD_AUROC')
 save_json(OUT/'pilot_gate.json',dict(status='passed',model_runs=len(data),note='Software feasibility gate only. Pilot scores are not used for model selection or target tuning.'))
 print('PILOT_GATE_PASSED',len(data),flush=True)

def aggregate():
 records=[json.loads(p.read_text()) for p in sorted((OUT/'records').glob('main__*.json'))]
 expected=len(tasks(False));unique={(r['holdout'],r['seed'],r['method'],r['feature_variant']) for r in records}
 if len(unique)!=expected or len(records)!=expected:raise RuntimeError(f'INCOMPLETE_OR_DUPLICATE_MAIN_RUNS {len(unique)}/{expected}')
 for hold in DOMAINS:
  for seed in CFG['seeds']:
   rs=[r for r in records if r['holdout']==hold and r['seed']==seed]
   if len(rs)!=len(METHODS)*len(FEATURE_VARIANTS) or len({r['test_sha'] for r in rs})!=1 or len({r['validation_sha'] for r in rs})!=1:
    raise RuntimeError('FIXED_TEST_VALIDATION_MISMATCH '+hold+' '+str(seed))
 rows=[];resources=[];threshold_rows=[];dist_rows=[];calibration_rows=[]
 for r in records:
  h={k:r[k] for k in ('holdout','seed','method','feature_variant','n_train','n_validation','n_test','test_sha','validation_sha','auroc','auprc','source_auroc','source_auprc')}
  for point in r['operating_points']:
   rows.append({**h,**point})
   if point['status']=='evaluated' and point['deployable_source_only'] and point['source_validation_fpr']>point['target_fpr']+1e-12:raise RuntimeError('SOURCE_THRESHOLD_FPR_INVARIANT')
  for t in r['threshold_audit']:
   threshold_rows.append({**{k:r[k] for k in ['holdout','seed','method','feature_variant']},**{k:v for k,v in t.items() if k!='per_source_thresholds'},'per_source_thresholds_json':json.dumps(t['per_source_thresholds'])})
  for location,metrics in r['calibration'].items():
   if isinstance(metrics,dict):calibration_rows.append({**{k:r[k] for k in ('holdout','seed','method','feature_variant')},'location':location,**metrics})
  for prefix,dd in [('source',r['source_distribution']),('target_POSTHOC',r['target_distribution_POSTHOC'])]:
   dist_rows.append({**{k:r[k] for k in ['holdout','seed','method','feature_variant']},'distribution':prefix,**dd})
  resources.append({**{k:r[k] for k in ['holdout','seed','method','feature_variant','n_train','n_validation','n_test']},**{k:r[k] for k in ('train_seconds','optimizer_steps','communication_bytes')}})
 df=pd.DataFrame(rows);df.to_csv(OUT/'threshold_transfer_metrics.csv',index=False)
 pd.DataFrame(threshold_rows).to_csv(OUT/'oracle_gap_POSTHOC.csv',index=False)
 pd.DataFrame(dist_rows).to_csv(OUT/'score_distribution_audit.csv',index=False)
 pd.DataFrame(calibration_rows).to_csv(OUT/'calibration_audit.csv',index=False)
 pd.DataFrame(resources).to_csv(OUT/'resource_metrics.csv',index=False)
 audit=[]
 for p in sorted((OUT/'feature_shift').glob('*.json')):audit.extend(json.loads(p.read_text()))
 pd.DataFrame(audit).to_csv(OUT/'feature_shift_ks.csv',index=False)
 ev=df[df.status=='evaluated'];numeric=['attack_recall','test_fpr','auroc','auprc','attack_f1','precision','source_auroc']
 summ=ev.groupby(['holdout','feature_variant','method','threshold_strategy','target_fpr'])[numeric].agg(['mean','std','count']).reset_index()
 summ.columns=['_'.join(map(str,c)).strip('_') for c in summ.columns];summ.to_csv(OUT/'summary_mean_std.csv',index=False)
 diffs=[]
 for (hold,seed,method,target,strategy),group in ev.groupby(['holdout','seed','method','target_fpr','threshold_strategy']):
  base=group[group.feature_variant=='all']
  if len(base)!=1:continue
  for _,r in group.iterrows():
   if r.feature_variant=='all':continue
   for metric in ('attack_recall','test_fpr','auroc','auprc'):
    diffs.append(dict(holdout=hold,seed=seed,method=method,threshold_strategy=strategy,target_fpr=target,variant=r.feature_variant,metric=metric,delta=float(r[metric]-base.iloc[0][metric])))
 pd.DataFrame(diffs).to_csv(OUT/'paired_feature_ablation_deltas.csv',index=False)
 failures=[]
 for f in OUT.glob('*_worker_*_failures.json'):failures.extend(json.loads(f.read_text()))
 if failures:raise RuntimeError('FAILURES_PRESENT '+str(len(failures)))
 run_status=dict(status='complete',completed_main_runs=len(records),expected_main_runs=expected,evaluated_operating_points=int(len(ev)),fixed_test_passed=True,
    source_only_fit_and_deployable_thresholds=True,oracle_thresholds_are_posthoc_non_deployable=True,
    feature_shift_diagnostics_posthoc_only=True,features_retrained_per_variant=True,
    limitation='No capture/session independence, target labels used only for posthoc oracle and metrics, sample prevalence is balanced by design')
 save_json(OUT/'run_status.json',run_status)
 archive=ROOT/CFG['archive']
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
  for f in OUT.rglob('*'):
   if f.is_file():z.write(f,f.relative_to(ROOT))
 print('FINAL_ZIP',archive,'bytes',archive.stat().st_size,'runs',expected,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','pilot','gate','main','aggregate'],required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2)
 a=p.parse_args()
 if a.mode=='prepare':prepare()
 elif a.mode=='gate':gate()
 elif a.mode=='aggregate':aggregate()
 else:worker(a.mode,a.worker,a.workers)
if __name__=='__main__':main()
