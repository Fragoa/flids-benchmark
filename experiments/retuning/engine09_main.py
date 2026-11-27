import os, sys, json, zipfile, io, hashlib, shutil, argparse, importlib.util, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score, average_precision_score

METHODS=['FedAvgMLP','FedProxMLP','FedAdamMLP','FedYogiMLP']
SEEDS=[0,42,3407,123,2026]
DOMAINS=['NF-UNSW-NB15','NF-ToN-IoT','NF-BoT-IoT','NF-CSE-CIC-IDS2018']
RATES=[.001,.01,.05]

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def digest(ids):return hashlib.sha256(np.sort(np.asarray(ids,dtype=np.int64)).tobytes()).hexdigest()

def load_archive(path, expected, expected_rounds, expected_trials):
 rows={}
 with zipfile.ZipFile(path) as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise RuntimeError('Duplicate archive paths')
  stat=[n for n in names if n.endswith('/run_status.json')]
  if len(stat)!=1:raise RuntimeError('Missing or ambiguous run_status.json')
  status=json.loads(z.read(stat[0]))
  if status.get('status')!='complete' or status.get('failed_trials')!=0:raise RuntimeError('Input experiment not complete')
  trial_names=[n for n in names if '/records/trial__' in n and n.endswith('.json')]
  trials={}
  for name in trial_names:
   t=json.loads(z.read(name));key=(t['holdout'],int(t['seed']),t['method'],int(t['candidate']))
   if key in trials or t.get('status')!='complete' or t.get('rounds')!=expected_rounds:raise RuntimeError('Invalid trial: '+name)
   trials[key]=t
  wanted={(h,s,m,c) for h,s,m in expected for c in range(expected_trials)}
  if set(trials)!=wanted:raise RuntimeError('Unexpected/missing candidate trials: '+str(len(trials)))
  for name in names:
   if '/records/chosen__' not in name or not name.endswith('.npz'):continue
   meta=json.loads(z.read(name[:-4]+'.json'));key=(meta['holdout'],int(meta['seed']),meta['method'])
   if key in rows:raise RuntimeError('Duplicate selected model')
   if not meta.get('model_selected_without_target') or not meta.get('threshold_calibration_disjoint_from_tuning'):raise RuntimeError('Invalid source-only protocol')
   candidates=[trials[(*key,c)] for c in range(expected_trials)]
   best=sorted(candidates,key=lambda t:(-t['tuning_pauc_fpr05'],-t['tuning_auroc'],t['candidate']))[0]
   if meta['chosen_candidate']!=best['candidate'] or meta['chosen_config']!=best['config'] or meta['split_hashes']!=best['split_hashes']:raise RuntimeError('Winner selection mismatch: '+str(key))
   with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:
    required=['test_ids','test_labels','test_scores','calibration_ids','calibration_labels','calibration_scores','tuning_ids','tuning_labels']
    arr={k:a[k] for k in required}
   for role in ['test','calibration','tuning']:
    ids=arr[role+'_ids'];y=arr[role+'_labels']
    if ids.ndim!=1 or len(ids)!=len(y) or len(np.unique(ids))!=len(ids) or set(np.unique(y))!={0,1} or digest(ids)!=meta['split_hashes'][role]:raise RuntimeError('Invalid ids/labels: '+str(key)+' '+role)
    if role!='tuning':
     score=arr[role+'_scores']
     if score.dtype!=np.float64 or score.shape!=y.shape or not np.isfinite(score).all():raise RuntimeError('Expected finite float64 scores: '+str(key))
   for a,b in [('test','calibration'),('test','tuning'),('calibration','tuning')]:
    if np.intersect1d(arr[a+'_ids'],arr[b+'_ids']).size:raise RuntimeError('Split overlap')
   rows[key]=(meta,arr)
 if set(rows)!=expected:raise RuntimeError('Unexpected/missing winners: '+str(len(rows)))
 return rows,trials

def match(a,b):
 ma,aa=a;mb,ab=b
 if ma['split_hashes']!=mb['split_hashes']:raise RuntimeError('Split hash mismatch')
 for k in ['test_ids','test_labels','calibration_ids','calibration_labels','tuning_ids','tuning_labels']:
  if not np.array_equal(aa[k],ab[k]):raise RuntimeError('Split row/order mismatch: '+k)

def interval(x,n):
 return (0. if x==0 else float(beta.ppf(.025,x,n-x+1)),1. if x==n else float(beta.ppf(.975,x+1,n-x)))

def threshold(scores,rate):
 order=np.sort(np.asarray(scores,dtype=np.float64));k=int(np.floor(rate*len(order)));pivot=order[-k] if k else order[-1]
 return float(np.nextafter(pivot,np.inf)),float(pivot),k

def evaluate(y,s,t):
 p=s>=t;tp=int(np.sum(p&(y==1)));fp=int(np.sum(p&(y==0)));tn=int(np.sum(~p&(y==0)));fn=int(np.sum(~p&(y==1)))
 return dict(attack_recall=tp/(tp+fn),test_fpr=fp/(fp+tn),precision=tp/(tp+fp) if tp+fp else 0.,tp=tp,fp=fp,tn=tn,fn=fn)

def audit(key,bundle,version):
 h,s,method=key;meta,a=bundle;cy=a['calibration_labels'];cs=a['calibration_scores'];ty=a['test_labels'];ts=a['test_scores'];cn=cs[cy==0];tn=ts[ty==0]
 c=dict(holdout=h,seed=s,method=method,version=version,chosen_candidate=meta['chosen_candidate'],chosen_config=json.dumps(meta['chosen_config'],sort_keys=True),n_cal_benign=len(cn),n_test_benign=len(tn),auroc=roc_auc_score(ty,ts),auprc=average_precision_score(ty,ts),pauc_fpr01=roc_auc_score(ty,ts,max_fpr=.01),pauc_fpr05=roc_auc_score(ty,ts,max_fpr=.05))
 op=[];precision=[];shift=[]
 recorded={float(r['target_fpr']):r for r in meta['results']}
 for rate in RATES:
  base=c|dict(target_fpr=rate,calibration_resolution=1/len(cn),zero_fp_one_sided_95_upper=1-.05**(1/len(cn)))
  old=recorded[rate]
  if rate==.001 and len(cn)<1000:
   if old['status']!='insufficient_benign_calibration':raise RuntimeError('Unexpected recorded status')
   op.append(base|dict(status='insufficient_benign_calibration'));continue
  t,pivot,k=threshold(cn,rate);fpr=float(np.mean(cn>=t));ev=evaluate(ty,ts,t);lo,hi=interval(ev['fp'],ev['fp']+ev['tn'])
  if fpr>rate+1e-12:raise RuntimeError('Source threshold violation')
  # Original archive stores the same conservative nextafter policy. Require exact reconstruction.
  if old['status']!='evaluated' or t!=old['threshold'] or any(ev[k]!=old[k] for k in ['tp','fp','tn','fn']) or abs(fpr-old['source_calibration_fpr'])>1e-12:raise RuntimeError('Recorded threshold/count mismatch: '+str(key)+' '+str(rate))
  op.append(base|dict(status='evaluated',threshold=t,source_calibration_fpr=fpr,threshold_matches=True,counts_match=True,target_fpr_excess=ev['test_fpr']-rate,target_fpr_ratio=ev['test_fpr']/rate,target_fpr_within_goal=ev['test_fpr']<=rate,flow_iid_ci95_low=lo,flow_iid_ci95_high=hi,**ev))
  # Diagnostic only: simulate float32 score/threshold storage, promote to float64 before comparison.
  t32=float(np.float32(t));s32=ts.astype(np.float32).astype(np.float64);c32=cn.astype(np.float32).astype(np.float64)
  re_t,_,_=threshold(c32,rate);e32=evaluate(ty,s32,t32);e32re=evaluate(ty,s32,re_t)
  precision.append(base|dict(threshold_float64=t,recorded_threshold=old['threshold'],threshold_float32_roundtrip=t32,threshold_abs_error=abs(t32-t),pivot=pivot,tail_budget=k,source_scores_equal_pivot=int(np.sum(cn==pivot)),target_scores_equal_pivot=int(np.sum(ts==pivot)),unique_source_benign_scores=len(np.unique(cn)),unique_target_scores=len(np.unique(ts)),source_fpr_float32_stored_threshold=float(np.mean(c32>=t32)),target_fpr_float32_stored_threshold=e32['test_fpr'],recall_float32_stored_threshold=e32['attack_recall'],target_fpr_float32_recalibrated=e32re['test_fpr'],recall_float32_recalibrated=e32re['attack_recall'],changed_predictions_stored_threshold=int(np.sum((ts>=t)!=(s32>=t32))),changed_predictions_recalibrated=int(np.sum((ts>=t)!=(s32>=re_t))),source_tail_exceedance_target=float(np.mean(tn>=t))))
 ks=ks_2samp(cn,tn)
 shift.append(c|dict(benign_ks_statistic=float(ks.statistic),benign_wasserstein=float(wasserstein_distance(cn,tn)),attack_ks_statistic=float(ks_2samp(cs[cy==1],ts[ty==1]).statistic),attack_wasserstein=float(wasserstein_distance(cs[cy==1],ts[ty==1])),**{f'{role}_q{q}':float(np.quantile(v,q/1000)) for role,v in [('source_benign',cn),('target_benign',tn)] for q in [500,900,950,990,995,999]}))
 return op,precision,shift

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();w=Path(a.work);w.mkdir(parents=True,exist_ok=True)
 os.environ['FLIDS_EXP02_ROOT']=str(w)
 if a.smoke:os.environ['FLIDS_EXP02_SMOKE']='1'
 else:os.environ.pop('FLIDS_EXP02_SMOKE',None)
 m=load_module('exp02',w/'engine02.py');oldengine=load_module('exp08_original',w/'engine08_original.py');newengine=load_module('exp09_retune',w/'engine09_retune.py')
 holds=DOMAINS[:1] if a.smoke else DOMAINS;seeds=SEEDS[:1] if a.smoke else SEEDS;rounds=2 if a.smoke else 30
 oldkeys={(h,s,b) for h in holds for s in seeds for b in METHODS};newkeys={(h,s,'FedProxMLP') for h in holds for s in seeds}
 # Stage 0: prepare a single common dataset cache and validate its identity before training.
 fp_path=w/'flids_exp02_fingerprints.npy'
 stage08_identity=w/'stage08_identity.json'
 stage08_records=w/'FLIDS_Experiment08_Results'/'records'
 original=w/'FLIDS_Experiment08_Results.zip'
 if m.CACHE.exists()!=fp_path.exists():raise RuntimeError('Incomplete cache/fingerprint pair; restore both or remove both before preparing')
 if not m.CACHE.exists():
  if stage08_identity.exists() or stage08_records.exists():raise RuntimeError('Resume requires the original cache and fingerprints')
  print('STAGE_0_PREPARING_NF_UQ_CACHE',flush=True);m.prepare()
 else:print('STAGE_0_REUSING_CACHE',flush=True)
 with np.load(m.CACHE,allow_pickle=False) as data:
  x=data['x'];y=data['y'];d=data['d']
 fp=np.load(fp_path,allow_pickle=False)
 if x.ndim!=2 or x.shape[1]!=len(m.FEATURES) or len(x)!=len(y) or len(fp)!=len(y) or len(d)!=len(y) or not np.isfinite(x).all():raise RuntimeError('Invalid cache shape or values')
 expected_per_class=220 if a.smoke else 12000
 if len(y)!=8*expected_per_class or set(np.unique(y))!={0,1} or set(np.unique(d))!=set(DOMAINS) or any(int(np.sum((d==h)&(y==label)))!=expected_per_class for h in DOMAINS for label in [0,1]):raise RuntimeError('Unexpected reservoir size/domain coverage; expected original NF-UQ balanced reservoir')
 computed=np.asarray([hashlib.blake2b(np.ascontiguousarray(row).tobytes(),digest_size=12).hexdigest() for row in x])
 if not np.array_equal(computed,fp):raise RuntimeError('Cache/fingerprints mismatch')
 identity08={'cache_sha256':sha(m.CACHE),'fingerprints_sha256':sha(fp_path),'engine02_sha256':sha(w/'engine02.py'),'engine08_sha256':sha(w/'engine08_original.py'),'holds':holds,'seeds':seeds,'methods':METHODS,'rounds':rounds,'smoke':a.smoke}
 if stage08_identity.exists():
  if json.loads(stage08_identity.read_text())!=identity08:raise RuntimeError('Experiment 08 resume identity mismatch; restore matching artifacts or use a clean work directory')
 else:
  if stage08_records.exists() or original.exists():raise RuntimeError('Existing Experiment 08 artifacts lack provenance; use a clean work directory')
  stage08_identity.write_text(json.dumps(identity08,indent=2))
 print('STAGE_1_EXPERIMENT_08_START',len(oldkeys)*(2 if a.smoke else 4),'candidate trainings',flush=True)
 if original.exists():
  # An already completed archive is kept byte-for-byte for the Experiment 09 resume identity.
  try: old,oldtrials=load_archive(original,oldkeys,rounds,2 if a.smoke else 4)
  except (zipfile.BadZipFile, RuntimeError, KeyError, ValueError, EOFError) as exc:
   if (w/'FLIDS_Experiment09_FedProx_Retune'/'resume_identity.json').exists():raise RuntimeError('Original archive is invalid after retune started; restore the exact archive') from exc
   print('REBUILD_INCOMPLETE_STAGE08_ARCHIVE',repr(exc),flush=True);original.unlink()
 if not original.exists():
  for h in holds:
   for s in seeds:
    print('EXPERIMENT08_FOLD',h,s,flush=True)
    oldengine.run_fold(m,w,h,s,METHODS,rounds,a.smoke)
  original=oldengine.aggregate(w,holds,seeds,METHODS,a.smoke)
 else:print('REUSE_COMPLETED_EXPERIMENT08_ARCHIVE',flush=True)
 old,oldtrials=load_archive(original,oldkeys,rounds,2 if a.smoke else 4)
 print('STAGE_1_EXPERIMENT_08_COMPLETE',len(oldtrials),len(old),flush=True)
 # Validate archived thresholds and counts before any expensive retuning.
 for key,bundle in old.items():audit(key,bundle,'preflight')
 for (h,s,method,c),t in oldtrials.items():
  if t['config']!=oldengine.CANDIDATES[method][c]:raise RuntimeError('Unexpected original candidate configuration')
 print('ORIGINAL_ARCHIVE_VALIDATED',len(old),flush=True)
 if not (m.CACHE.exists() and (w/'flids_exp02_fingerprints.npy').exists()):m.prepare()
 data=np.load(m.CACHE,allow_pickle=False);y=data['y'];d=data['d'];x=data['x'];fp=np.load(w/'flids_exp02_fingerprints.npy',allow_pickle=False)
 if len(x)!=len(y) or len(fp)!=len(y) or len(d)!=len(y) or not np.isfinite(x).all():raise RuntimeError('Invalid cache')
 # Verify that cached fingerprints are actually derived from the cached rows.
 computed=np.asarray([hashlib.blake2b(np.ascontiguousarray(row).tobytes(),digest_size=12).hexdigest() for row in x])
 if not np.array_equal(computed,fp):raise RuntimeError('Cache/fingerprints mismatch')
 if not a.smoke and len(y)!=96000:raise RuntimeError('Expected original 96,000-row NF-UQ reservoir; check input/cache')
 for h in holds:
  for s in seeds:
   tr,va,te=m.split_indices(y,d,fp,h,s);tune,cal=oldengine.audit_val_split(va,y,fp,d,s)
   hashes={k:digest(ids) for k,ids in [('train',tr),('validation',va),('test',te),('tuning',tune),('calibration',cal)]}
   first=old[(h,s,METHODS[0])]
   for b in METHODS:
    bundle=old[(h,s,b)];match(first,bundle)
    if hashes!=bundle[0]['split_hashes']:raise RuntimeError('Prepared split differs from Experiment 08; attach original cache or exact CSV')
    for role,ids in [('test',te),('tuning',tune),('calibration',cal)]:
     if not np.array_equal(bundle[1][role+'_ids'],ids) or not np.array_equal(bundle[1][role+'_labels'],y[ids]):raise RuntimeError('Prepared split ids/labels mismatch')
 print('ALL_ORIGINAL_SPLITS_VERIFIED',len(holds)*len(seeds),flush=True)
 # Resume only if previous cache and original archive identities agree.
 identity={'original_sha256':sha(original),'cache_sha256':sha(m.CACHE),'fingerprints_sha256':sha(w/'flids_exp02_fingerprints.npy'),'grid':newengine.CANDIDATES['FedProxMLP'],'smoke':a.smoke,'retune_engine_sha256':sha(w/'engine09_retune.py')}
 identity_path=w/'FLIDS_Experiment09_FedProx_Retune'/'resume_identity.json';identity_path.parent.mkdir(exist_ok=True)
 if (identity_path.parent/'records').exists() and not identity_path.exists():raise RuntimeError('Existing records without resume identity')
 if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise RuntimeError('Resume provenance mismatch: use clean work folder or original inputs')
 identity_path.write_text(json.dumps(identity,indent=2))
 print('STAGE_2_EXPERIMENT_09_FEDPROX_RETUNE_START',len(newkeys)*4,'candidate trainings',flush=True)
 for h in holds:
  for s in seeds:
   print('FEDPROX_RETUNE_FOLD',h,s,flush=True)
   newengine.run_fold(m,w,h,s,['FedProxMLP'],rounds,a.smoke)
 newzip=newengine.aggregate(w,holds,seeds,['FedProxMLP'],a.smoke)
 new,newtrials=load_archive(newzip,newkeys,rounds,4)
 for (h,s,method,c),t in newtrials.items():
  if t['config']!=newengine.CANDIDATES[method][c]:raise RuntimeError('Unexpected retune configuration')
 for key in new:match(old[key],new[key])
 print('STAGE_3_THRESHOLD_AUDIT_START',flush=True)
 out=w/'FLIDS_Experiment09_Results';out.mkdir(exist_ok=True)
 ops=[];prec=[];shifts=[]
 for key,b in old.items():
  v='old_fedprox' if key[2]=='FedProxMLP' else 'final'
  a1,b1,c1=audit(key,b,v);ops+=a1;prec+=b1;shifts+=c1
 for key,b in new.items():
  a1,b1,c1=audit(key,b,'final');ops+=a1;prec+=b1;shifts+=c1
 df=pd.DataFrame(ops);df.to_csv(out/'operating_points_all.csv',index=False);pd.DataFrame(prec).to_csv(out/'threshold_precision_audit.csv',index=False);pd.DataFrame(shifts).to_csv(out/'score_shift.csv',index=False)
 final=df[df.version=='final'];final.to_csv(out/'final_operating_points.csv',index=False)
 metrics=['attack_recall','test_fpr','precision','auroc','auprc','pauc_fpr01','pauc_fpr05','target_fpr_excess','target_fpr_ratio']
 domain=final.groupby(['method','holdout','target_fpr'],as_index=False)[metrics].mean();domain.to_csv(out/'domain_summary.csv',index=False)
 summary=domain.groupby(['method','target_fpr'])[metrics].agg(['mean','std']);summary.columns=['_'.join(k) for k in summary.columns];summary.reset_index().to_csv(out/'equal_domain_summary.csv',index=False)
 evaluated=final[final.status=='evaluated'];evaluated.groupby(['method','target_fpr']).agg(evaluated_models=('seed','size'),domains=('holdout','nunique'),models_within_fpr_goal=('target_fpr_within_goal','sum')).reset_index().to_csv(out/'fpr_goal_coverage.csv',index=False)
 pair=df[df.method=='FedProxMLP'];pair=pair[pair.status=='evaluated'].pivot(index=['holdout','seed','target_fpr'],columns='version',values=['attack_recall','test_fpr','auroc','pauc_fpr01','pauc_fpr05']);pair.columns=['_'.join(k) for k in pair.columns]
 for metric in ['attack_recall','test_fpr','auroc','pauc_fpr01','pauc_fpr05']:pair[metric+'_delta']=pair[metric+'_final']-pair[metric+'_old_fedprox']
 pair.reset_index().to_csv(out/'fedprox_paired_before_after.csv',index=False)
 pd.DataFrame([t|{'experiment':'08_original'} for t in oldtrials.values()]+[t|{'experiment':'09_retune'} for t in newtrials.values()]).to_csv(out/'hyperparameter_trials_combined.csv',index=False)
 status={'status':'complete','software_test_only':a.smoke,'original_models':len(old),'original_training_trials':len(oldtrials),'total_candidate_trainings':len(oldtrials)+len(newtrials),'retuned_fedprox_models':len(new),'new_training_trials':len(newtrials),'final_models':len(final.groupby(['holdout','seed','method'])),'final_operating_points':len(final),'evaluated_final_operating_points':len(evaluated),'verified_split_folds':len(holds)*len(seeds),'all_evaluated_thresholds_and_counts_match':True,'provenance':identity}
 (out/'run_status.json').write_text(json.dumps(status,indent=2))
 (out/'INTERPRETATION.txt').write_text('The original threshold policy is preserved (conservative nextafter; may undershoot nominal FPR). Float64 storage is not float64 neural inference. Float32 roundtrip is a sensitivity simulation, not recovery of Experiment 03. Old FedProx is retained only for paired diagnostics; final table uses fresh four-candidate grid. Equal trial count per comparison is not equal total historical tuning effort or proof of optimality. Four target domains, seeds nested. Flow-level binomial intervals assume independent records, which may not hold. Balanced reservoir changes prevalence; precision/AP not deployment estimates. Index/fingerprint checks are not capture independence and original archive lacks full feature hash. Target labels used only for evaluation. No target threshold fitting.')
 # Include all new candidate records, learning curves, checkpoints, selected scores, and audit CSVs.
 target=w/('FLIDS_Experiment09_SOFTWARE_TEST_ONLY.zip' if a.smoke else 'FLIDS_Experiment09_Results.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
  for root in [out,w/'FLIDS_Experiment09_FedProx_Retune']:
   for f in sorted(root.rglob('*')):
    if f.is_file():z.write(f,f.relative_to(w))
  z.write(original,'experiment08/'+original.name)
  for name in ['stage08_identity.json','flids_exp02_cache.npz','flids_exp02_fingerprints.npy']:
   z.write(w/name,'reproducibility/'+name)
  for f in sorted((w/'FLIDS_Experiment02_Results').glob('*.json')):
   z.write(f,'reproducibility/data_preparation/'+f.name)
  for name in ['engine02.py','engine08_original.py','engine09_retune.py','engine09_main.py']:
   z.write(w/name,'code/'+name)
 print('EXPERIMENT09_COMPLETE',json.dumps(status),str(target),flush=True)
if __name__=='__main__':main()
