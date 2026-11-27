"""FL-IDS Experiment 05: client heterogeneity and partial participation.

Research benchmark, not a new algorithm. Uses NF-UQ-NIDS-v2 source-heldout
splits. Target-domain rows and labels never enter fitting/thresholds.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, csv, hashlib, json, math, random, sys, time, traceback, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(os.getenv('FLIDS_EXP05_ROOT', '/kaggle/working' if Path('/kaggle/working').exists() else '/mnt/data/exp05_local'))
OUT = ROOT/'FLIDS_Experiment05_Results'
OUT.mkdir(parents=True,exist_ok=True)
CACHE = ROOT/'flids_exp05_cache.npz'
DOMAINS = ['NF-UNSW-NB15','NF-ToN-IoT','NF-BoT-IoT','NF-CSE-CIC-IDS2018']
FEATURES = ['L4_SRC_PORT','L4_DST_PORT','PROTOCOL','L7_PROTO','IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS','FLOW_DURATION_MILLISECONDS','TCP_FLAGS','CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS','DURATION_IN','DURATION_OUT','MIN_TTL','MAX_TTL','LONGEST_FLOW_PKT','SHORTEST_FLOW_PKT','MIN_IP_PKT_LEN','MAX_IP_PKT_LEN','SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_SECOND_BYTES','RETRANSMITTED_IN_BYTES','RETRANSMITTED_IN_PKTS','RETRANSMITTED_OUT_BYTES','RETRANSMITTED_OUT_PKTS','SRC_TO_DST_AVG_THROUGHPUT','DST_TO_SRC_AVG_THROUGHPUT','NUM_PKTS_UP_TO_128_BYTES','NUM_PKTS_128_TO_256_BYTES','NUM_PKTS_256_TO_512_BYTES','NUM_PKTS_512_TO_1024_BYTES','NUM_PKTS_1024_TO_1514_BYTES','TCP_WIN_MAX_IN','TCP_WIN_MAX_OUT','ICMP_TYPE','ICMP_IPV4_TYPE','DNS_QUERY_ID','DNS_QUERY_TYPE','DNS_TTL_ANSWER','FTP_COMMAND_RET_CODE']
CFG=dict(
    seeds=[0,42,3407,123,2026], domains=DOMAINS, features=FEATURES,
    methods=['FedAvgMLP','FedProxMLP'],
    allocation_modes=['source_domain','quantity_skew','label_skew'],
    participation=[3,2], clients=3, rounds=30, batch_size=256,
    client_round_cap=512, local_epochs=1, learning_rate=.001, prox_mu=.01,
    label_dirichlet_alpha=.15, quantity_weights=[.15,.25,.60],
    validation_fraction=.20, max_rows_per_domain_label=12000,
    min_train_samples_per_client=32, fpr_targets=[.001,.01,.05],
    min_benign_validation_001=1000, archive='FLIDS_Experiment05_Results.zip',
    notes=['Source_domain clients follow the four-source-dataset grouping (three training clients per heldout fold).',
           'Quantity_skew and label_skew repartition the same training rows among synthetic clients; no samples are dropped.',
           'All modes use identical fixed train/validation/target-test rows for each domain and seed.',
           'Test domain is excluded from model training and source threshold selection.',
           'Full numeric row fingerprints are not a substitute for capture/session IDs.',
           'Audit checks of 40,000 rows do not establish absence of duplicates in 76 million rows.',
           'Client byte counts are simulated raw float32 parameter payloads, not measured network overhead.',
           'No privacy protection is implied by basic FedAvg/FedProx.'])
SMOKE=os.getenv('FLIDS_EXP05_SMOKE')=='1'
if SMOKE:
    CFG.update(seeds=[0],domains=DOMAINS[:2],rounds=2,max_rows_per_domain_label=180,client_round_cap=128,min_benign_validation_001=1000)

def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def clean(s):return ''.join(c for c in str(s).lower() if c.isalnum())
def domain_name(s):
    a=clean(s)
    if 'unsw' in a:return DOMAINS[0]
    if 'ton' in a and 'iot' in a:return DOMAINS[1]
    if 'bot' in a and 'iot' in a:return DOMAINS[2]
    if 'cse' in a or 'cicids2018' in a:return DOMAINS[3]
    return None

def file_header(path):
    with path.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
        return next(csv.reader(f))

def find_input():
    roots=[]
    if os.getenv('FLIDS_EXP05_INPUT'): roots.append(Path(os.environ['FLIDS_EXP05_INPUT']))
    roots += [Path('/kaggle/input'),ROOT/'input']
    found=[]; inspected=[]
    for root in roots:
        if not root.exists():continue
        candidates=[root] if root.is_file() else root.rglob('*.csv')
        for p in candidates:
            try:
                h=file_header(p); valid=set(FEATURES).issubset(h) and ('Label' in h or 'label' in h) and ('Dataset' in h or domain_name(p.name))
                inspected.append(dict(path=str(p),valid=bool(valid),missing_features=sorted(set(FEATURES)-set(h))[:8]))
                if valid:found.append((p,h))
            except Exception as e:inspected.append(dict(path=str(p),error=str(e)[:200]))
    dump(OUT/'input_discovery.json',dict(inspected=inspected,valid=[str(p) for p,_ in found]))
    combined=[z for z in found if clean(z[0].name).startswith('nfuqnidsv2')]
    if combined:return combined[:1]
    selected=[];used=set()
    for p,h in sorted(found,key=lambda x:str(x[0])):
        d=domain_name(p.name)
        if d and d not in used:selected.append((p,h));used.add(d)
    if len(selected)==4:return selected
    for z in found:
        if 'Dataset' in z[1]:return [z]
    return selected

def synth_file():
    p=ROOT/'nf_exp05_SOFTWARE_TEST_ONLY.csv';ROOT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(1811)
    with p.open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerow(FEATURES+['Label','Attack','Dataset'])
        for di,d in enumerate(DOMAINS):
            for yy in [0,1]:
                for j in range(200):
                    a=rng.gamma(2,2,size=len(FEATURES));a[0]=500+di*50+yy*8;a[1]=80+di*11;a[4]+=yy*2;a[5]+=yy
                    wr.writerow([f'{v:.5f}' for v in a]+[yy,'Attack' if yy else 'Benign',d+'-v2'])
    return p

def row_fingerprint(row):
    return hashlib.blake2b(np.ascontiguousarray(row,dtype=np.float32).tobytes(),digest_size=12).hexdigest()

def digest(ids):
    return hashlib.sha256(np.sort(np.asarray(ids,dtype=np.int64)).tobytes()).hexdigest()

def preprocess(x):
    arr=np.asarray(x,dtype=np.float64)
    arr=np.clip(arr,-1e12,1e12)
    return np.clip(np.sign(arr)*np.log1p(np.abs(arr)),-18,18).astype(np.float32)

def prepare():
    dump(OUT/'experiment_config.json',CFG)
    if SMOKE:
        p=synth_file();sources=[(p,file_header(p))];input_mode='synthetic software smoke only'
    else:
        sources=find_input();input_mode='nf-uq-netflow-v2 csv'
    if not sources:
        raise FileNotFoundError('NO_NF_UQ_V2_INPUT. Connect the NF-UQ-NIDS-v2 dataset via Kaggle Add Input (or set FLIDS_EXP05_INPUT). Inspect input_discovery.json.')
    cap=CFG['max_rows_per_domain_label'];rng=np.random.default_rng(2077)
    samples=defaultdict(list);counts=Counter();invalid=Counter()
    csv.field_size_limit(10_000_000)
    for path,head in sources:
        ix={k:i for i,k in enumerate(head)};li=ix.get('Label',ix.get('label'));di=ix.get('Dataset')
        fd=domain_name(path.name);processed=0
        with path.open('r',encoding='utf-8-sig',errors='replace',newline='',buffering=1024*1024) as f:
            reader=csv.reader(f);next(reader)
            for line in reader:
                if len(line)!=len(head):invalid['column_mismatch']+=1;continue
                d=domain_name(line[di]) if di is not None else fd
                if d not in DOMAINS:invalid['unidentified_domain']+=1;continue
                ys=clean(line[li])
                if ys in ('0','benign','normal'):y=0
                elif ys in ('1','attack','malicious'):y=1
                else:invalid['invalid_binary_label']+=1;continue
                try:
                    values=np.asarray([float(line[ix[c]]) if line[ix[c]].strip() else 0. for c in FEATURES],dtype=np.float32)
                except (TypeError,ValueError,OverflowError):invalid['bad_numeric']+=1;continue
                if not np.isfinite(values).all():invalid['nonfinite_replaced']+=1;values=np.nan_to_num(values,nan=0.,posinf=0.,neginf=0.)
                key=(d,y);counts[key]+=1;n=counts[key];buf=samples[key]
                if len(buf)<cap:buf.append(values)
                else:
                    j=int(rng.integers(0,n))
                    if j<cap:buf[j]=values
                processed+=1
                if processed%10000000==0:print('SCAN_PROGRESS',path.name,processed,flush=True)
        print('SOURCE_SCAN_COMPLETE',path.name,processed,flush=True)
    xx=[];yy=[];dd=[]
    for (d,y),rs in sorted(samples.items()):
        for row in rs:xx.append(row);yy.append(y);dd.append(d)
    if not xx:raise RuntimeError('NO_VALID_ROWS')
    x=np.asarray(xx,dtype=np.float32);y=np.asarray(yy,dtype=np.int8);d=np.asarray(dd)
    cov={f'{dom}|{cl}':int(np.sum((d==dom)&(y==cl))) for dom in DOMAINS for cl in (0,1)}
    minimum=70 if SMOKE else 500
    missing=[a for a,v in cov.items() if v<minimum]
    dump(OUT/'data_coverage.json',dict(sampling_method='uniform reservoir for each source-domain and binary label',original_counts={f'{a}|{b}':int(v) for (a,b),v in counts.items()},pool_counts=cov,minimum_count=minimum,coverage_passed=not missing,missing=missing,invalid=dict(invalid),pool_rows=len(y),source_rows=sum(counts.values())))
    if missing:raise RuntimeError('DOMAIN_LABEL_COVERAGE_FAILED '+str(missing))
    fp=np.asarray([row_fingerprint(a) for a in x]);np.savez_compressed(CACHE,x=x,y=y,d=d,fp=fp)
    index=[]
    for held in CFG['domains']:
        for seed in CFG['seeds']:
            tr,va,te=split_indices(y,d,fp,held,seed)
            for role,ids in [('train',tr),('validation',va),('test',te)]:
                index.append(dict(heldout=held,seed=seed,role=role,count=len(ids),benign=int(np.sum(y[ids]==0)),attack=int(np.sum(y[ids]==1)),row_id_digest=digest(ids),domain_names=';'.join(sorted(set(d[ids])))))
    pd.DataFrame(index).to_csv(OUT/'fixed_split_manifest.csv',index=False)
    overlaps=[]
    for held in CFG['domains']:
        for seed in CFG['seeds']:
            tr,va,te=split_indices(y,d,fp,held,seed)
            ff=[set(fp[ids]) for ids in [tr,va,te]]
            overlaps.append(dict(heldout=held,seed=seed,train_validation_overlap=len(ff[0]&ff[1]),train_test_overlap=len(ff[0]&ff[2]),validation_test_overlap=len(ff[1]&ff[2])))
    pd.DataFrame(overlaps).to_csv(OUT/'sampled_fingerprint_overlap.csv',index=False)
    # Signal warnings; cross-domain duplicate fingerprints are NOT silently removed.
    leak=any(row['train_test_overlap']>0 or row['validation_test_overlap']>0 for row in overlaps)
    gate=dict(status='conditional_pass' if not leak else 'blocked',data_source=input_mode,source_isolation_passed=True,fingerprint_overlap_with_test_detected=leak,session_capture_independence_verified=False,notes='Dataset-heldout isolation is verified by domain labels, but duplicate fingerprints across source domains cannot be ruled out from this sampled pool; full-dataset audit is not exhaustive.')
    dump(OUT/'data_gate.json',gate)
    if leak:raise RuntimeError('SAMPLED_TARGET_FINGERPRINT_OVERLAP_DETECTED: Holdout fingerprints found in source data; do not proceed until resolved.')
    print('EXP05_PREPARED',len(y),'rows; heldouts',len(CFG['domains']),'seeds',len(CFG['seeds']),'gate',gate['status'],flush=True)

def split_indices(y,d,fp,held,seed):
    tr=[];va=[];te=np.flatnonzero(d==held)
    for source in DOMAINS:
        if source==held:continue
        idx=np.flatnonzero(d==source)
        aa,bb=next(GroupShuffleSplit(n_splits=1,test_size=CFG['validation_fraction'],random_state=seed).split(idx,y[idx],fp[idx]))
        tr.extend(idx[aa]);va.extend(idx[bb])
    tr=np.asarray(tr,dtype=np.int64);va=np.asarray(va,dtype=np.int64)
    if len(set(y[tr]))<2 or len(set(y[va]))<2 or len(set(y[te]))<2:
        raise RuntimeError(f'MISSING_LABEL_IN_SPLIT heldout={held} seed={seed}')
    if set(d[tr])!={a for a in DOMAINS if a!=held} or set(d[te])!={held}:
        raise RuntimeError('SOURCE_TARGET_SPLIT_VIOLATION')
    if len(set(fp[tr])&set(fp[va])):
        raise RuntimeError('SOURCE_TRAIN_VAL_FINGERPRINT_OVERLAP')
    return tr,va,te

def allocate(train_idx,y,d,held,seed,mode):
    rng=np.random.default_rng(seed+17091)
    if mode=='source_domain':
        out=[train_idx[d[train_idx]==dom] for dom in DOMAINS if dom!=held]
    elif mode=='quantity_skew':
        ids=rng.permutation(train_idx)
        weights=np.asarray(CFG['quantity_weights'],dtype=float);weights/=weights.sum()
        sizes=np.floor(weights*len(ids)).astype(int)
        sizes[-1]=len(ids)-sum(sizes[:-1]);split=np.cumsum(sizes)[:-1]
        out=list(np.split(ids,split))
    elif mode=='label_skew':
        bins=[[] for _ in range(3)]
        for label in (0,1):
            ids=rng.permutation(train_idx[y[train_idx]==label])
            # Dirichlet allocations are fixed per task; all training rows are retained.
            proportions=rng.dirichlet(np.full(3,CFG['label_dirichlet_alpha']))
            reserved=CFG['min_train_samples_per_client']//2
            if len(ids)<3*reserved:raise RuntimeError('INSUFFICIENT_LABEL_TO_RESERVE')
            for j in range(3):
                bins[j].extend(ids[j*reserved:(j+1)*reserved].tolist())
            tail=ids[3*reserved:]
            parts=rng.multinomial(len(tail),proportions)
            for j,chunk in enumerate(np.split(tail,np.cumsum(parts)[:-1])):
                bins[j].extend(chunk.tolist())
        out=[np.asarray(a,dtype=np.int64) for a in bins]
    else:raise ValueError(mode)
    if len(out)!=3 or any(len(z)<CFG['min_train_samples_per_client'] for z in out):
        raise RuntimeError('INSUFFICIENT_CLIENT_ROWS '+str([(len(a)) for a in out]))
    if any(len(set(y[z]))<2 for z in out):
        raise RuntimeError('CLIENT_LABEL_MISSING '+str([Counter(y[z].tolist()) for z in out]))
    all_ids=np.concatenate(out)
    if len(all_ids)!=len(train_idx) or len(np.unique(all_ids))!=len(train_idx) or not np.array_equal(np.sort(all_ids),np.sort(train_idx)):
        raise RuntimeError('ALLOCATION_IS_NOT_A_PARTITION')
    return [np.sort(z) for z in out]

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)

class MLP(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def train_local(state,x,y,idx,seed,steps,device,prox=False):
    model=MLP(x.shape[1]);model.load_state_dict(state);model=model.to(device).train()
    opt=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate']);rng=np.random.default_rng(seed)
    init=[a.to(device) for a in state.values()]
    for _ in range(steps):
        choices=rng.choice(idx,size=min(CFG['batch_size'],len(idx)),replace=len(idx)<CFG['batch_size'])
        inp=torch.as_tensor(x[choices],device=device);truth=torch.as_tensor(y[choices],dtype=torch.float32,device=device)
        opt.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(model(inp),truth)
        if prox:loss=loss+CFG['prox_mu']*.5*sum((param-w).square().sum() for param,w in zip(model.parameters(),init))
        loss.backward();opt.step()
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

@torch.no_grad()
def inference(state,x,ids,device):
    m=MLP(x.shape[1]);m.load_state_dict(state);m=m.to(device).eval()
    scores=[]
    for inds in np.array_split(ids,max(1,int(np.ceil(len(ids)/4096)))):
        if len(inds):scores.append(torch.sigmoid(m(torch.as_tensor(x[inds],device=device))).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)

def choose_threshold(scores,target):
    scores=np.asarray(scores,dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all():raise RuntimeError('INVALID_VALIDATION_BENIGN_SCORES')
    order=np.sort(scores);permitted=int(np.floor(target*len(scores)))
    pivot=order[-permitted] if permitted else order[-1]
    threshold=float(np.nextafter(pivot,np.inf))
    realized=float(np.mean(scores>=threshold))
    if realized>target+1e-12:raise RuntimeError('THRESHOLD_CONSTRAINT_VIOLATED')
    return threshold,realized

def task_list(pilot=False):
    if pilot:
        held=CFG['domains'][0];seed=CFG['seeds'][0]
        return [(held,seed,'source_domain',3,'FedAvgMLP'),(held,seed,'label_skew',2,'FedProxMLP')]
    return [(held,seed,mode,p,method)
            for held in CFG['domains'] for seed in CFG['seeds']
            for mode in CFG['allocation_modes'] for p in CFG['participation']
            for method in CFG['methods']]

def run_task(raw,y,d,fp,held,seed,mode,participation,method,stage,device):
    x=preprocess(raw)
    tr,va,te=split_indices(y,d,fp,held,seed)
    clients=allocate(tr,y,d,held,seed,mode)
    seed_all(seed);m=MLP(x.shape[1]);global_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    npar=sum(p.numel() for p in m.parameters());steps=0;communication=0;begin=time.perf_counter()
    rng=np.random.default_rng(seed+4101)
    nrounds=2 if stage=='pilot' else CFG['rounds']
    selected_counts=[0,0,0];all_rounds=[]
    # Same model initialization and client-participation draw for FedAvg/FedProx at matched task.
    for rr in range(nrounds):
        chosen=np.sort(rng.choice(3,size=participation,replace=False))
        states=[];sizes=[]
        for ci in chosen:
            idx=clients[int(ci)];substeps=max(1,int(math.ceil(min(len(idx),CFG['client_round_cap'])/CFG['batch_size'])))*CFG['local_epochs']
            seed_local=seed+rr*300+int(ci)*79
            current=train_local(global_state,x,y,idx,seed_local,substeps,device,prox=(method=='FedProxMLP'))
            states.append(current);sizes.append(len(idx));steps+=substeps;selected_counts[int(ci)]+=1
        weights=np.asarray(sizes,dtype=float);weights/=weights.sum()
        global_state={k:sum(z[k]*float(w) for z,w in zip(states,weights)) for k in global_state}
        communication+=2*len(chosen)*npar*4
        all_rounds.append(','.join(str(i) for i in chosen))
    vscore=inference(global_state,x,va,device);tscore=inference(global_state,x,te,device)
    ytest=y[te];benign=vscore[y[va]==0]
    points=[]
    for target in CFG['fpr_targets']:
        if target==.001 and len(benign)<CFG['min_benign_validation_001']:
            points.append(dict(requested_val_fpr=target,status='insufficient_benign_source_validation',n_benign_validation=len(benign)));continue
        thr,vfpr=choose_threshold(benign,target)
        pred=tscore>=thr;tp=int(np.sum(pred&(ytest==1)));fpn=int(np.sum(pred&(ytest==0)));fn=int(np.sum((~pred)&(ytest==1)));tn=int(np.sum((~pred)&(ytest==0)))
        points.append(dict(requested_val_fpr=target,status='evaluated',threshold=thr,source_val_fpr=vfpr,actual_test_fpr=fpn/max(1,fpn+tn),target_recall=tp/max(1,tp+fn),target_precision=tp/max(1,tp+fpn),target_f1=2*tp/max(1,2*tp+fn+fpn),tp=tp,fp=fpn,tn=tn,fn=fn,n_benign_validation=len(benign)))
    stats=[]
    for i,idx in enumerate(clients):
        stats.append(dict(client_id=i,n_train=len(idx),n_benign=int(np.sum(y[idx]==0)),n_attack=int(np.sum(y[idx]==1)),attack_fraction=float(np.mean(y[idx])),n_original_domains=len(set(d[idx])),selected_rounds=selected_counts[i],source_domain=';'.join(sorted(set(d[idx])))))
    record=dict(stage=stage,heldout=held,seed=seed,allocation_mode=mode,participating_clients=participation,method=method,n_clients=3,train_sha=digest(tr),validation_sha=digest(va),test_sha=digest(te),training_partition_digest=digest(np.concatenate(clients)),n_train=len(tr),n_source_validation=len(va),n_target_test=len(te),n_target_attack=int(np.sum(ytest==1)),n_target_benign=int(np.sum(ytest==0)),model_auroc=float(roc_auc_score(ytest,tscore)),model_auprc=float(average_precision_score(ytest,tscore)),train_seconds=time.perf_counter()-begin,communication_bytes=communication,optimizer_steps=steps,rounds=nrounds,selection_trace=all_rounds,client_stats=stats,operating_points=points)
    return record,dict(test_ids=te,test_labels=ytest,test_scores=tscore.astype(np.float32),validation_ids=va,validation_labels=y[va],validation_scores=vscore.astype(np.float32))

def worker(stage,wid,total):
    with np.load(CACHE) as data:
        raw=data['x'];y=data['y'];d=data['d'];fp=data['fp']
    selected=[t for j,t in enumerate(task_list(stage=='pilot')) if j%total==wid]
    errors=[]
    device=f'cuda:{wid%torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
    for count,(held,seed,mode,p,method) in enumerate(selected,1):
        task_key=f'{stage}__{clean(held)}__{seed}__{mode}__p{p}__{method}'
        jpath=OUT/'records'/f'{task_key}.json';ppath=OUT/'predictions_raw'/f'{task_key}.npz'
        if jpath.exists() and ppath.exists():print('RESUME_SKIP',task_key,flush=True);continue
        try:
            rec,pred=run_task(raw,y,d,fp,held,seed,mode,p,method,stage,device)
            ppath.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(ppath,**pred)
            dump(jpath,rec)
            print('TASK_OK',count,'/',len(selected),task_key,'AUROC',round(rec['model_auroc'],4),flush=True)
        except Exception:
            err=traceback.format_exc();errors.append(dict(task=task_key,traceback=err));print('TASK_FAILED',task_key,err[-1600:],flush=True)
    dump(OUT/f'{stage}_worker_{wid}_failures.json',errors)
    if errors:raise RuntimeError(f'{len(errors)} failed tasks in {stage}; inspect failure JSON')

def gate():
    expected=len(task_list(True))
    files=list((OUT/'records').glob('pilot__*.json'))
    if len(files)!=expected:raise RuntimeError(f'PILOT_INCOMPLETE {len(files)}/{expected}')
    records=[json.loads(p.read_text()) for p in files]
    if any(not np.isfinite(r['model_auroc']) for r in records):raise RuntimeError('PILOT_METRIC_NOT_FINITE')
    for r in records:
        if r['train_sha']!=r['training_partition_digest']:raise RuntimeError('PILOT_PARTITION_MISMATCH')
    dump(OUT/'pilot_gate.json',dict(status='passed',expected=expected,completed=len(files),scope='software feasibility only; no target-tuned hyperparameters'))
    print('PILOT_GATE_PASSED',expected,flush=True)

def aggregate():
    expected=task_list(False);files=list((OUT/'records').glob('main__*.json'))
    actual=[json.loads(f.read_text()) for f in files]
    ids={(r['heldout'],r['seed'],r['allocation_mode'],r['participating_clients'],r['method']) for r in actual}
    if len(actual)!=len(expected) or len(ids)!=len(expected):raise RuntimeError(f'INCOMPLETE_MAIN_RECORDS {len(ids)}/{len(expected)}')
    for held in CFG['domains']:
        for seed in CFG['seeds']:
            group=[r for r in actual if r['heldout']==held and r['seed']==seed]
            if len(group)!=len(CFG['allocation_modes'])*len(CFG['participation'])*len(CFG['methods']):raise RuntimeError('MISSING_MATCHED_TASK')
            for field in ['train_sha','validation_sha','test_sha','training_partition_digest']:
                if len({r[field] for r in group})!=1:raise RuntimeError('NONMATCHED_SPLIT '+field)
    mm=[];res=[];cl=[];participation=[]
    for r in actual:
        head={k:r[k] for k in ['heldout','seed','allocation_mode','participating_clients','method','n_train','n_source_validation','n_target_test','train_sha','validation_sha','test_sha','model_auroc','model_auprc']}
        for p in r['operating_points']:
            mm.append(dict(**head,**p))
            if p['status']=='evaluated' and p['source_val_fpr']>p['requested_val_fpr']+1e-12:
                raise RuntimeError('SOURCE_FPR_INVARIANT_BROKEN')
        res.append({**head,**{q:r[q] for q in ['train_seconds','communication_bytes','optimizer_steps','rounds']}})
        for z in r['client_stats']:cl.append(dict(**{q:r[q] for q in ['heldout','seed','allocation_mode','participating_clients','method']},**z))
        for rr,trace in enumerate(r['selection_trace']):participation.append(dict(**{q:r[q] for q in ['heldout','seed','allocation_mode','participating_clients','method']},round=rr,selected_client_ids=trace))
    metric=pd.DataFrame(mm)
    metric.to_csv(OUT/'experiment_results.csv',index=False)
    pd.DataFrame(res).to_csv(OUT/'communication_cost.csv',index=False)
    pd.DataFrame(cl).to_csv(OUT/'client_distributions.csv',index=False)
    pd.DataFrame(participation).to_csv(OUT/'client_participation_trace.csv',index=False)
    good=metric[metric.status=='evaluated'].copy()
    if (good.source_val_fpr>good.requested_val_fpr+1e-12).any():raise RuntimeError('SOURCE_FPR_INVARIANT_BROKEN')
    summary_cols=['model_auroc','model_auprc','target_recall','actual_test_fpr','target_precision','target_f1']
    summary=good.groupby(['heldout','allocation_mode','participating_clients','method','requested_val_fpr'])[summary_cols].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(map(str,c)).strip('_') for c in summary.columns]
    summary.to_csv(OUT/'summary_mean_std.csv',index=False)
    paired=[]
    for keys,table in good.groupby(['heldout','seed','allocation_mode','participating_clients','requested_val_fpr']):
        indexed=table.set_index('method')
        if all(q in indexed.index for q in CFG['methods']):
            for metric_name in ['target_recall','actual_test_fpr','model_auroc']:
                paired.append(dict(heldout=keys[0],seed=keys[1],allocation_mode=keys[2],participating_clients=keys[3],requested_val_fpr=keys[4],metric=metric_name,delta_fedprox_minus_fedavg=float(indexed.at['FedProxMLP',metric_name]-indexed.at['FedAvgMLP',metric_name])))
    pd.DataFrame(paired).to_csv(OUT/'paired_prox_avg.csv',index=False)
    failures=[]
    for f in OUT.glob('*_worker_*_failures.json'):failures.extend(json.loads(f.read_text()))
    if failures:raise RuntimeError('FAILURES_PRESENT '+str(len(failures)))
    status=dict(status='complete',expected=len(expected),completed=len(actual),operating_points=len(metric),valid_points=len(good),fixed_split_verified=True,source_target_isolation=True,session_independence_verified=False)
    dump(OUT/'run_status.json',status)
    with zipfile.ZipFile(ROOT/CFG['archive'],'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in OUT.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(ROOT))
    print('AGGREGATION_COMPLETE',status,'ZIP',ROOT/CFG['archive'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','pilot','gate','main','aggregate'],required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='gate':gate()
    elif args.mode=='aggregate':aggregate()
    else:worker(args.mode,args.worker,args.workers)

if __name__=='__main__':main()
