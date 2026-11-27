"""FLIDS Experiment 08: source-only fair FedOpt tuning and convergence audit.
Four source-domain holdouts; candidate-selection and threshold-calibration are disjoint.
No heldout-domain scores are computed until a source-only winner is selected.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import csv, sys, json, hashlib, random, zipfile, argparse, traceback, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

METHODS=['FedProxMLP']
SEEDS=[0,42,3407,123,2026]
# Four trials per method, using comparable budget (same client steps, rounds, splits).
# FedOpt lr_s and tau combinations intentionally extend beyond the earlier un-tuned eta_s=0.001.
CANDIDATES={
 'FedAvgMLP':[dict(client_lr=x,server_lr=1.0,mu=0.,tau=0.01) for x in [0.0003,0.001,0.003,0.0001]],
 'FedProxMLP':[dict(client_lr=lr,server_lr=1.0,mu=mu,tau=0.01) for lr in [0.001,0.003] for mu in [0.001,0.01]],
 'FedAdamMLP':[dict(client_lr=0.001,server_lr=a,mu=0.,tau=t) for a,t in [(0.001,0.01),(0.01,0.01),(0.05,0.01),(0.01,0.1)]],
 'FedYogiMLP':[dict(client_lr=0.001,server_lr=a,mu=0.,tau=t) for a,t in [(0.001,0.01),(0.01,0.01),(0.05,0.01),(0.01,0.1)]],
}

def digest(a):return hashlib.sha256(np.sort(np.asarray(a,dtype=np.int64)).tobytes()).hexdigest()
def put_json(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,allow_nan=False,default=str))
def load_module(path):
 import importlib.util
 sp=importlib.util.spec_from_file_location('flids_exp02_for_exp08',str(path));m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m

def seed_all(s):
 random.seed(int(s));np.random.seed(int(s));torch.manual_seed(int(s));torch.set_num_threads(1)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(int(s))

def audit_val_split(va,y,fp,d,seed):
 # Deterministic grouping by feature signatures, used only within source validation.
 groups={}
 for idx in va:
  key=str(fp[idx]);groups.setdefault(key,[]).append(int(idx))
 by_domain={z:[] for z in sorted(set(d[va]))}
 for key,ids in groups.items():
  domain=str(d[ids[0]])
  # Signature could exist across domains: group must not cross selection/calibration.
  if len(set(d[ids]))>1: domain='__cross__'
  by_domain.setdefault(domain,[]).append((key,ids))
 pick=[];cal=[]
 for domain,items in sorted(by_domain.items()):
  for key,ids in items:
   u=int.from_bytes(hashlib.sha256((key+'|'+str(seed)+'|exp08').encode()).digest()[:8],'big')
   (pick if u%2==0 else cal).extend(ids)
 pick=np.asarray(sorted(pick),dtype=np.int64);cal=np.asarray(sorted(cal),dtype=np.int64)
 if not len(pick) or not len(cal):raise RuntimeError('EMPTY_TUNE_OR_CALIBRATION')
 if set(fp[pick])&set(fp[cal]):raise RuntimeError('TUNE_CAL_FINGERPRINT_OVERLAP')
 if set(y[pick])!={0,1} or set(y[cal])!={0,1}:raise RuntimeError('TUNE_CAL_CLASS_MISSING')
 if np.sum(y[cal]==0)<(30 if os.getenv('FLIDS_EXP02_SMOKE')=='1' else 1000):raise RuntimeError('CALIBRATION_BENIGN_TOO_SMALL_FOR_0_1_PERCENT')
 return pick,cal

def train_one(m,x,y,client_ids,seed,method,cfg,rounds,device,round_log):
 seed_all(seed)
 model=m.MLP(x.shape[1]);state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 momentum={k:torch.zeros_like(v) for k,v in state.items()}
 second={k:torch.zeros_like(v) for k,v in state.items()}
 sizes=np.asarray([len(i) for i in client_ids],dtype=float);weights=sizes/sizes.sum()
 cap=int(m.CFG['client_round_cap']);batch_size=int(m.CFG['batch_size'])
 client_steps=[max(1,int(np.ceil(min(len(ix),cap)/batch_size)))*int(m.CFG['local_epochs']) for ix in client_ids]
 params=sum(p.numel() for p in model.parameters());total_bytes=0
 beta1,beta2=0.9,0.99
 rng=np.random.default_rng(seed+40017)
 audit_ids=np.unique(np.concatenate([rng.choice(ids,size=min(384,len(ids)),replace=False) for ids in client_ids]))
 audit_x=torch.from_numpy(x[audit_ids]).to(device);audit_y=torch.as_tensor(y[audit_ids],dtype=torch.float32,device=device)
 for rnd in range(rounds):
  states=[];losses=[]
  for ci,ids in enumerate(client_ids):
   local=m.MLP(x.shape[1]);local.load_state_dict(state);local.to(device).train()
   optimizer=torch.optim.Adam(local.parameters(),lr=cfg['client_lr'])
   rg=np.random.default_rng(int(seed)+rnd*300+ci*79)
   ref=[p.detach().clone() for p in local.parameters()]
   for step in range(client_steps[ci]):
    b=rg.choice(ids,size=min(batch_size,len(ids)),replace=len(ids)<batch_size)
    xx=torch.from_numpy(x[b]).to(device);yy=torch.as_tensor(y[b],dtype=torch.float32,device=device)
    optimizer.zero_grad(set_to_none=True)
    loss=F.binary_cross_entropy_with_logits(local(xx),yy)
    if method=='FedProxMLP':
     loss=loss+0.5*float(cfg['mu'])*sum((p-q).square().sum() for p,q in zip(local.parameters(),ref))
    if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_CLIENT_LOSS')
    loss.backward();optimizer.step()
   states.append({k:v.detach().cpu().clone() for k,v in local.state_dict().items()});losses.append(float(loss.item()))
  avg={k:sum(float(w)*a[k] for w,a in zip(weights,states)) for k in state}
  delta={k:avg[k]-state[k] for k in state}
  delta_norm=float(sum(z.double().square().sum().item() for z in delta.values())**.5)
  nextstate={}
  if method in ('FedAvgMLP','FedProxMLP'):
   nextstate=avg
  else:
   for k in state:
    g=delta[k];momentum[k]=beta1*momentum[k]+(1-beta1)*g
    if method=='FedAdamMLP':second[k]=beta2*second[k]+(1-beta2)*g.square()
    else:second[k]=second[k]-(1-beta2)*g.square()*torch.sign(second[k]-g.square())
    # Same explicit debiasing convention for both FedOpt methods.
    mh=momentum[k]/(1-beta1**(rnd+1));vh=(second[k]/(1-beta2**(rnd+1))).clamp_min(0)
    nextstate[k]=state[k]+float(cfg['server_lr'])*mh/(vh.sqrt()+float(cfg['tau']))
  step_norm=float(sum((nextstate[k]-state[k]).double().square().sum().item() for k in state)**.5)
  if not np.isfinite(step_norm) or step_norm>1e6:raise FloatingPointError('DIVERGENT_SERVER_STEP')
  state=nextstate
  total_bytes+=2*len(client_ids)*params*4
  monitor=m.MLP(x.shape[1]);monitor.load_state_dict(state);monitor.to(device).eval()
  with torch.no_grad():audit_loss=float(F.binary_cross_entropy_with_logits(monitor(audit_x),audit_y).item())
  if not np.isfinite(audit_loss):raise FloatingPointError('NONFINITE_AUDIT_LOSS')
  round_log.append(dict(round=rnd+1,client_last_loss_mean=float(np.mean(losses)),source_train_audit_loss=audit_loss,client_delta_norm=delta_norm,server_update_norm=step_norm,communication_bytes=total_bytes))
 return state,total_bytes,int(sum(client_steps)*rounds)

def threshold(negative,rate):
 neg=np.sort(np.asarray(negative,dtype=np.float64));k=int(np.floor(rate*len(neg)))
 t=float(np.nextafter(neg[-k] if k else neg[-1],np.inf))
 return t,float(np.mean(neg>=t))

def metric_scores(yt,sc):
 if set(np.unique(yt))!={0,1}:raise RuntimeError('AUC_REQUIRES_BOTH_CLASSES')
 return dict(auroc=float(roc_auc_score(yt,sc)),auprc=float(average_precision_score(yt,sc)),pauc_fpr05=float(roc_auc_score(yt,sc,max_fpr=0.05)))

def select_best(rows):
 # Strictly source-only: standardized partial ROC AUC on tuning partition; AUROC breaks ties.
 good=[r for r in rows if r['status']=='complete' and np.isfinite(r['tuning_pauc_fpr05'])]
 if not good:raise RuntimeError('NO_VALID_CANDIDATE')
 return sorted(good,key=lambda r:(-r['tuning_pauc_fpr05'],-r['tuning_auroc'],r['candidate']))[0]

def run_fold(m,work,hold,seed,methods,rounds,smoke):
 xdata=np.load(m.CACHE);x=m.transform(xdata['x']);y=xdata['y'];d=xdata['d'];fp=np.load(m.ROOT/'flids_exp02_fingerprints.npy')
 tr,va,te=m.split_indices(y,d,fp,hold,seed)
 tune,cal=audit_val_split(va,y,fp,d,seed)
 clients=[tr[d[tr]==z] for z in m.DOMAINS if z!=hold]
 hashes=dict(train=digest(tr),validation=digest(va),tuning=digest(tune),calibration=digest(cal),test=digest(te))
 out=work/'FLIDS_Experiment09_FedProx_Retune';records=out/'records';records.mkdir(parents=True,exist_ok=True)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 for method in methods:
  cfgs=CANDIDATES[method]
  trials=[]
  for candidate,cfg in enumerate(cfgs):
   key=f'{m.normalized(hold)}__{seed}__{method}__c{candidate:02d}'
   path=records/f'trial__{key}.json';state_path=records/f'trial__{key}.pt';curve_path=records/f'curve__{key}.csv'
   if path.exists() and state_path.exists() and curve_path.exists():
    o=json.loads(path.read_text())
    if o.get('status')=='complete' and o.get('config')==cfg and o.get('split_hashes')==hashes and o.get('rounds')==rounds and len(pd.read_csv(curve_path))==rounds:
     trials.append(o);print('RESUME_TRIAL',key,flush=True);continue
    path.unlink(missing_ok=True);state_path.unlink(missing_ok=True);curve_path.unlink(missing_ok=True)
   tic=time.perf_counter();curve=[]
   try:
    state,comm,steps=train_one(m,x,y,clients,seed,method,cfg,rounds,device,curve)
    scores=m.predict_model(state,x,tune,device) # selection uses source tune only
    met=metric_scores(y[tune],scores)
    if not np.isfinite(met['pauc_fpr05']):raise RuntimeError('NONFINITE_TUNING_METRIC')
    o=dict(holdout=hold,seed=seed,method=method,candidate=candidate,config=cfg,status='complete',
      **{'tuning_'+k:v for k,v in met.items()},rounds=rounds,optimizer_steps=steps,communication_bytes=comm,
      source_training_audit_loss_last=curve[-1]['source_train_audit_loss'],max_server_update_norm=max(z['server_update_norm'] for z in curve),
      train_seconds=time.perf_counter()-tic,split_hashes=hashes)
    torch.save(state,state_path);pd.DataFrame(curve).to_csv(curve_path,index=False);put_json(path,o)
    print('TRIAL_OK',key,'pAUC05',round(met['pauc_fpr05'],5),'last_loss',round(curve[-1]['source_train_audit_loss'],5),flush=True)
   except Exception as e:
    o=dict(holdout=hold,seed=seed,method=method,candidate=candidate,config=cfg,status='failed',error=repr(e),split_hashes=hashes)
    put_json(path,o);print('TRIAL_FAILED',key,repr(e),flush=True)
   trials.append(o)
  if any(t['status']!='complete' for t in trials):raise RuntimeError('INCOMPLETE_CANDIDATES: rerun to resume failed trials')
  winner=select_best(trials)
  outcome=records/f'chosen__{m.normalized(hold)}__{seed}__{method}.json'
  preds=records/f'chosen__{m.normalized(hold)}__{seed}__{method}.npz'
  # Re-evaluate the source-selected winner on every resume.
  state=torch.load(records/f'trial__{m.normalized(hold)}__{seed}__{method}__c{winner["candidate"]:02d}.pt',map_location='cpu',weights_only=True)
  sv=np.asarray(m.predict_model(state,x,cal,device),dtype=np.float64)
  st=np.asarray(m.predict_model(state,x,te,device),dtype=np.float64)
  results=[];negative=sv[y[cal]==0]
  for target in [0.001,0.01,0.05]:
   if len(negative)<1000 and target==0.001:
    results.append(dict(target_fpr=target,status='insufficient_benign_calibration'));continue
   cut,valfpr=threshold(negative,target);p=st>=cut;yy=y[te]
   tp=int((p&(yy==1)).sum());fpn=int((p&(yy==0)).sum());tn=int(((~p)&(yy==0)).sum());fn=int(((~p)&(yy==1)).sum())
   results.append(dict(target_fpr=target,status='evaluated',threshold=cut,source_calibration_fpr=valfpr,
      attack_recall=tp/max(1,tp+fn),test_fpr=fpn/max(1,fpn+tn),precision=tp/max(1,tp+fpn),tp=tp,fp=fpn,tn=tn,fn=fn))
  obj=dict(holdout=hold,seed=seed,method=method,chosen_candidate=winner['candidate'],chosen_config=winner['config'],
    tuning_auroc=winner['tuning_auroc'],tuning_pauc_fpr05=winner['tuning_pauc_fpr05'],test_metrics=metric_scores(y[te],st),
    n_train=len(tr),n_tune=len(tune),n_calibration=len(cal),n_test=len(te),n_benign_cal=int((y[cal]==0).sum()),
    split_hashes=hashes,communication_bytes=winner['communication_bytes'],optimizer_steps=winner['optimizer_steps'],results=results,
    model_selected_without_target=True,threshold_calibration_disjoint_from_tuning=True,score_dtype=str(st.dtype))
  np.savez_compressed(preds,test_ids=te,test_labels=y[te],test_scores=st.astype(np.float64),calibration_ids=cal,
    calibration_labels=y[cal],calibration_scores=sv.astype(np.float64),tuning_ids=tune,tuning_labels=y[tune])
  put_json(outcome,obj)
  print('WINNER_OK',hold,seed,method,'candidate',winner['candidate'],'target_auroc_diagnostic',round(obj['test_metrics']['auroc'],4),flush=True)

def aggregate(work,holds,seeds,methods,smoke):
 out=work/'FLIDS_Experiment09_FedProx_Retune';rec=out/'records'
 chosen=[json.loads(p.read_text()) for p in rec.glob('chosen__*.json')]
 expected=len(holds)*len(seeds)*len(methods)
 if len(chosen)!=expected:raise RuntimeError(f'EXPECTED {expected} WINNERS, GOT {len(chosen)}')
 assert len({(r['holdout'],r['seed'],r['method']) for r in chosen})==expected
 lines=[];trials=[];curves=[];splitrows=[]
 for p in rec.glob('trial__*.json'):
  obj=json.loads(p.read_text());trials.append({k:v for k,v in obj.items() if k not in ('config','split_hashes')}
    | {'config':json.dumps(obj['config'],sort_keys=True)})
 for p in rec.glob('curve__*.csv'):
  # include per-round logs for every candidate, including rejected trials.
  df=pd.read_csv(p);df.insert(0,'trial_file',p.stem.replace('curve__',''));curves.append(df)
 for r in chosen:
  head=dict(holdout=r['holdout'],seed=r['seed'],method=r['method'],chosen_candidate=r['chosen_candidate'],
   chosen_config=json.dumps(r['chosen_config'],sort_keys=True),n_train=r['n_train'],n_tune=r['n_tune'],n_calibration=r['n_calibration'],
   n_test=r['n_test'],communication_bytes=r['communication_bytes'],optimizer_steps=r['optimizer_steps'],
   tuning_pauc_fpr05=r['tuning_pauc_fpr05'],**r['test_metrics'])
  for op in r['results']:lines.append(head|op)
  splitrows.append(dict(holdout=r['holdout'],seed=r['seed'],method=r['method'],**r['split_hashes']))
 pd.DataFrame(lines).to_csv(out/'selected_models_operating_points.csv',index=False)
 pd.DataFrame(trials).to_csv(out/'source_only_hyperparameter_trials.csv',index=False)
 pd.concat(curves,ignore_index=True).to_csv(out/'roundwise_learning_curves.csv',index=False)
 pd.DataFrame(splitrows).to_csv(out/'fixed_split_digests.csv',index=False)
 df=pd.DataFrame(lines);evaled=df[df.status=='evaluated']
 evaled.groupby(['method','target_fpr'],as_index=False)[['attack_recall','test_fpr','auroc','pauc_fpr05','auprc','communication_bytes']].mean().to_csv(out/'method_summary_descriptive.csv',index=False)
 # Fixed split within each holdout/seed, regardless of model; tuning/calibration fixed too.
 for (h,s),group in pd.DataFrame(splitrows).groupby(['holdout','seed']):
  for k in ['train','validation','tuning','calibration','test']:
   if group[k].nunique()!=1:raise RuntimeError('SPLIT_MISMATCH '+h+' '+str(s)+' '+k)
 status=dict(status='complete',expected_winners=expected,completed_winners=len(chosen),
  fixed_source_splits=True,source_only_tuning=True,disjoint_tuning_and_calibration=True,
  target_only_evaluated_after_selection=True,float64_scores_saved=True,failed_trials=sum(t['status']!='complete' for t in trials),
  note='With four held-out source datasets, inference is descriptive and domain-limited. Tuning candidates use equal training rounds/steps, not equal wall time or algorithmic complexity.')
 put_json(out/'run_status.json',status)
 zip_path=work/'FLIDS_Experiment09_FedProx_Retune.zip'
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(work))
 print('EXP08_COMPLETE',status,'ZIP',zip_path,flush=True)
 return zip_path

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',default='/kaggle/working');p.add_argument('--engine02',default='')
 p.add_argument('--mode',choices=['prepare','run','aggregate','all'],default='all');p.add_argument('--smoke',action='store_true')
 p.add_argument('--holds',default='');p.add_argument('--seeds',default='');p.add_argument('--methods',default='')
 a=p.parse_args();work=Path(a.work);work.mkdir(parents=True,exist_ok=True)
 os.environ['FLIDS_EXP02_ROOT']=str(work)
 if a.smoke:os.environ['FLIDS_EXP02_SMOKE']='1'
 engine=Path(a.engine02) if a.engine02 else work/'FLIDS_Experiment02_Engine.py'
 if not engine.exists():raise FileNotFoundError(f'MISSING_EXP02_ENGINE {engine}')
 m=load_module(engine)
 holds=a.holds.split(',') if a.holds else (m.DOMAINS[:1] if a.smoke else m.DOMAINS)
 seeds=[int(s) for s in a.seeds.split(',')] if a.seeds else ([0] if a.smoke else SEEDS)
 methods=a.methods.split(',') if a.methods else METHODS
 if set(holds)-set(m.DOMAINS) or set(methods)-set(METHODS):raise ValueError('INVALID_HOLDOUT_OR_METHOD')
 if a.mode in ('prepare','all'):
  if not (m.CACHE.exists() and (work/'flids_exp02_fingerprints.npy').exists()):
   print('CACHE_MISSING_PREPARE_NF_UQ_STREAM',flush=True);m.prepare()
  else:print('CACHE_FOUND',m.CACHE,flush=True)
 if a.mode in ('run','all'):
  if not (m.CACHE.exists() and (work/'flids_exp02_fingerprints.npy').exists()):raise RuntimeError('CACHE_MISSING_RUN_PREPARE_FIRST')
  data=np.load(m.CACHE)
  # Explicit schema coverage guard.
  if len(data['y'])<100 or set(np.unique(data['y']))!={0,1}:raise RuntimeError('INVALID_INPUT_CACHE')
  rounds=2 if a.smoke else 30
  for h in holds:
   for s in seeds:
    print('FOLD_START',h,s,flush=True)
    run_fold(m,work,h,s,methods,rounds,a.smoke)
 if a.mode in ('aggregate','all'):aggregate(work,holds,seeds,methods,a.smoke)
if __name__=='__main__':main()
