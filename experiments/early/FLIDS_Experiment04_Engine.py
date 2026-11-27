"""FL-IDS Experiment 04: deployment-prevalence and alert-budget sensitivity audit.

Consumes saved Experiment 03 predictions; does NOT train new models or claim to
reconstruct operational populations from a class-balanced benchmark sample.
All deployable thresholds are selected from SOURCE validation scores only.
Target labels are used solely to compute retrospectively evaluated rates.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import argparse, csv, io, json, math, re, sys, zipfile, hashlib, platform, traceback
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('FLIDS_EXP04_ROOT','/kaggle/working' if Path('/kaggle/working').exists() else '/mnt/data'))
OUT=ROOT/'FLIDS_Experiment04_Results'; OUT.mkdir(parents=True,exist_ok=True)
CONFIG={
 'title':'Experiment 04: Operational Reliability Under Hypothetical Prevalence and Alert Budgets',
 'upstream':'Experiment 03 saved predictions',
 'prevalence_scenarios':[0.001,0.01,0.05],
 'alert_budgets_per_100k':[50,100,500,1000],
 'population_size':100000,
 'source_fpr_targets':[0.001,0.01,0.05],
 'source_only_strategies':['pooled_source_quantile','worst_source_quantile'],
 'primary_feature_variant':'all',
 'assumption':'Class-conditional TPR/FPR stay unchanged under hypothetical prior probability shift.',
 'limitation':'Reweighting does not simulate covariate shift, temporal drift, alert correlation, or true deployment prevalence.',
 'privacy':'Inputs contain saved per-sample predictions; do not publish prediction IDs or metadata without review.',
 'oracle':'Target oracle thresholds are NOT used to define operational decisions.',
}

def jdump(p,o):Path(p).write_text(json.dumps(o,indent=2,default=str),encoding='utf-8')

def discover():
 explicit=os.environ.get('FLIDS_EXP03_ZIP','')
 if explicit:
  p=Path(explicit)
  if p.is_file() and zipfile.is_zipfile(p):return p
  raise FileNotFoundError('FLIDS_EXP03_ZIP points to a missing or invalid ZIP: '+str(p))
 choices=[]
 for r in [ROOT,Path('/kaggle/input'),Path('/mnt/data')]:
  if r.exists():
   for p in r.rglob('FLIDS_Experiment03_Results.zip'):
    if p.is_file() and zipfile.is_zipfile(p):choices.append(p)
 if choices:return sorted(set(choices),key=lambda p:(len(p.parts),str(p)))[0]
 raise FileNotFoundError('Experiment 03 ZIP not found. Attach FLIDS_Experiment03_Results.zip via Kaggle Add Input, or set FLIDS_EXP03_ZIP to its actual location. NF-UQ CSV is not needed for this analytic reuse.')

def input_members(z):
 names=z.namelist()
 def one(suffix):
  hits=[n for n in names if n.endswith('/'+suffix) or n==suffix]
  if len(hits)!=1:raise ValueError(f'Expected one {suffix}, found {len(hits)}')
  return hits[0]
 return one('run_status.json'),one('threshold_transfer_metrics.csv'),one('dataset_audit.json')

def audit(path):
 with zipfile.ZipFile(path) as z:
  a,b,c=input_members(z)
  status=json.loads(z.read(a));data=json.loads(z.read(c));df=pd.read_csv(z.open(b))
  preds=[n for n in z.namelist() if '/predictions_raw/main__' in n and n.endswith('.npz')]
  if status.get('completed_main_runs')!=300 or not status.get('fixed_test_passed'):
   raise ValueError('Experiment03 source run is incomplete or fixed-test check failed: '+str(status))
  req={'holdout','seed','method','feature_variant','target_fpr','threshold_strategy','threshold','attack_recall','test_fpr','test_sha','status','deployable_source_only'}
  if req-set(df.columns):raise ValueError('Missing metric columns: '+str(req-set(df.columns)))
  source=df[df['threshold_strategy'].isin(CONFIG['source_only_strategies'])]
  if not source['deployable_source_only'].astype(bool).all():raise ValueError('Found non-deployable threshold among source strategies')
  runs=source[['holdout','seed','method','feature_variant']].drop_duplicates()
  keys=['holdout','seed']
  if df.groupby(keys)['test_sha'].nunique().max()!=1:raise ValueError('Test manifest hashes inconsistent between methods')
  if len(preds)!=300 or len(runs)!=300:raise ValueError(f'Expected 300 prediction NPZ / run identities; got {len(preds)} / {len(runs)}')
  jdump(OUT/'input_audit.json',{'source_file':str(path),'sha256':hashfile(path),'run_status':status,'model_runs':len(runs),'prediction_files':len(preds),'metric_rows':len(df),'source_only_metric_rows':len(source),'fixed_test_verified':True,'original_class_counts':data.get('original_counts',{}),'original_prevalence_is_NOT_deployment':True})
  return df,data,preds

def hashfile(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def wilson_bounds(k,n,z=1.959963984540054):
 """Marginal Wilson score interval, for class-conditional test rates."""
 if n<=0:return float('nan'),float('nan')
 p=k/n;den=1+z*z/n
 ctr=(p+z*z/(2*n))/den
 half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return max(0.,ctr-half),min(1.,ctr+half)

def precision_with_rates(r,f,p):
 numerator=p*r;denominator=numerator+(1-p)*f
 return numerator/denominator if denominator>0 else float('nan')

def scenario(rate_attack,rate_benign,pop,prevalence):
 p=float(prevalence);r=float(rate_attack);f=float(rate_benign)
 tp=pop*p*r; fn=pop*p*(1-r); fp=pop*(1-p)*f; tn=pop*(1-p)*(1-f)
 alerts=tp+fp; ppv=tp/alerts if alerts else np.nan
 f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn>0 else np.nan
 return dict(expected_tp_per_100k=tp,expected_fp_per_100k=fp,expected_fn_per_100k=fn,expected_tn_per_100k=tn,expected_alerts_per_100k=alerts,expected_precision=ppv,expected_f1=f1,expected_false_alerts_per_true_alert=fp/tp if tp else np.nan,expected_alerts_per_detected_attack=alerts/tp if tp else np.nan)

def conf_rates(y,s,threshold):
 y=np.asarray(y,dtype=np.int8);s=np.asarray(s,dtype=np.float64)
 if not len(y):raise ValueError('empty predictions')
 y0=y==0;y1=y==1
 if not y0.any() or not y1.any():raise ValueError('Both target classes required')
 pos=s>float(threshold)
 return float(np.mean(pos[y1])),float(np.mean(pos[y0]))

def validate_scores(y,s,label):
 if y.ndim!=1 or s.ndim!=1 or len(y)!=len(s):raise ValueError('bad shapes for '+label)
 if not np.isfinite(s).all():raise ValueError('nonfinite scores for '+label)
 if set(np.unique(y).tolist())!={0,1}:raise ValueError('must have both classes for '+label)

def get_prevalence(data,domain):
 counts=data.get('original_counts',{})
 n0=int(counts.get(f'{domain}|0',0));n1=int(counts.get(f'{domain}|1',0))
 return n1/(n0+n1) if n0+n1>0 else None

def source_budget_selection(y,s,p,limit,pop=100000):
 """Choose least restrictive SOURCE-derived threshold satisfying expected budget.

 All unique source score values are valid thresholds for strictly greater-than
 decisions; identical scores always share one decision. Target scores/labels
 never enter this function.
 """
 y=np.asarray(y,dtype=np.int8);s=np.asarray(s,dtype=np.float64)
 v0=np.sort(s[y==0]);v1=np.sort(s[y==1])
 cand=np.unique(s)
 f=(len(v0)-np.searchsorted(v0,cand,side='right'))/len(v0)
 r=(len(v1)-np.searchsorted(v1,cand,side='right'))/len(v1)
 burden=pop*((1-p)*f+p*r)
 viable=np.flatnonzero(burden<=limit+1e-9)
 if not len(viable):raise RuntimeError('No feasible threshold, even no-alert threshold')
 j=int(viable[0])
 return float(cand[j]),float(r[j]),float(f[j]),float(burden[j])

def run():
 path=discover();print('INPUT',path,flush=True)
 df,data,preds=audit(path)
 df=df[(df['status']=='evaluated') & df['threshold_strategy'].isin(CONFIG['source_only_strategies'])].copy()
 df['target_fpr']=pd.to_numeric(df['target_fpr'],errors='raise')
 if not (df['deployable_source_only'].astype(bool)).all():raise ValueError('Non-source threshold slipped into evaluation')
 config=dict(CONFIG,experiment03_sha256=hashfile(path),n_model_runs=300)
 jdump(OUT/'experiment_config.json',config)
 domain_counts=[]
 for d in sorted(df['holdout'].unique()):
  p=get_prevalence(data,d)
  if p is not None:
   n0=data['original_counts'].get(f'{d}|0',0);n1=data['original_counts'].get(f'{d}|1',0)
   domain_counts.append(dict(holdout=d,original_count_benign=n0,original_count_attack=n1,original_dataset_attack_fraction=p,interpretation='original dataset prevalence; not verified operational prevalence'))
 pd.DataFrame(domain_counts).to_csv(OUT/'observed_dataset_prevalence_CONTEXT_ONLY.csv',index=False)
 perf=[]
 for row in df.itertuples(index=False):
  for p in CONFIG['prevalence_scenarios']:
   rl,rh=wilson_bounds(int(row.tp),int(row.tp)+int(row.fn))
   fl,fh=wilson_bounds(int(row.fp),int(row.fp)+int(row.tn))
   perf.append(dict(holdout=row.holdout,seed=int(row.seed),method=row.method,feature_variant=row.feature_variant,threshold_strategy=row.threshold_strategy,target_validation_fpr=float(row.target_fpr),threshold=float(row.threshold),target_attack_recall=float(row.attack_recall),target_test_fpr=float(row.test_fpr),prevalence_scenario=p,test_sha=row.test_sha,source_validation_fpr=float(row.source_validation_fpr),decision_uses_target_labels=False,
     marginal_wilson_recall_lower=rl,marginal_wilson_recall_upper=rh,marginal_wilson_fpr_lower=fl,marginal_wilson_fpr_upper=fh,
     conditional_precision_lower_sensitivity=precision_with_rates(rl,fh,p),conditional_precision_upper_sensitivity=precision_with_rates(rh,fl,p),
     **scenario(row.attack_recall,row.test_fpr,CONFIG['population_size'],p)))
 perf=pd.DataFrame(perf)
 perf.to_csv(OUT/'prevalence_reweighted_operating_points.csv',index=False)
 print('REWEIGHTED_ROWS',len(perf),flush=True)
 # Budget thresholds selected only on source validation predictions.
 budget=[];pr_curves=[];counts=Counter();check=[]
 with zipfile.ZipFile(path) as z:
  for j,member in enumerate(sorted(preds)):
   with np.load(io.BytesIO(z.read(member)),allow_pickle=False) as q:
    required={'validation_labels','validation_scores','test_labels','test_scores'}
    if not required.issubset(q.files):raise ValueError(f'Missing fields in {member}: {required-set(q.files)}')
    sy=np.asarray(q['validation_labels'],dtype=np.int8).copy()
    ss=np.asarray(q['validation_scores'],dtype=np.float64).copy()
    ty=np.asarray(q['test_labels'],dtype=np.int8).copy()
    ts=np.asarray(q['test_scores'],dtype=np.float64).copy()
   validate_scores(sy,ss,'source '+member);validate_scores(ty,ts,'target '+member)
   key=Path(member).stem.split('__')
   if len(key)!=5 or key[0]!='main':raise ValueError('Unexpected prediction filename '+member)
   _,shortdomain,seed_txt,method,variant=key
   candidates=df[(df.seed==int(seed_txt)) & (df.method==method) & (df.feature_variant==variant)]
   candidates=candidates[candidates.holdout.str.lower().str.replace(r'[^a-z0-9]','',regex=True)==shortdomain]
   if candidates.empty:raise ValueError('Prediction file has no metrics identity '+member)
   holdout=str(candidates.iloc[0].holdout);ref=candidates.iloc[0]
   if len(ty)!=int(ref.n_test):raise ValueError('Target prediction count mismatch '+member)
   # Anti-tamper agreement: replay a threshold from saved metrics.
   replay=candidates[(candidates.threshold_strategy=='pooled_source_quantile') & np.isclose(candidates.target_fpr,.01)]
   if len(replay)!=1:raise ValueError('Expected one source 1% record for '+member)
   r=replay.iloc[0]
   # Experiment 03 evaluated float32 target scores with a scalar threshold.
   # Preserve this comparison behavior in the provenance replay check; a
   # strict float64 comparison can differ for values tied at that threshold.
   raw_target=np.asarray(ts,dtype=np.float32)
   pred=raw_target > float(r.threshold)
   tr=float(pred[ty==1].mean());tf=float(pred[ty==0].mean())
   if abs(tr-float(r.attack_recall))>1e-6 or abs(tf-float(r.test_fpr))>1e-6:
    raise ValueError('Score replay mismatch for '+member)
   check.append(dict(holdout=holdout,seed=int(seed_txt),method=method,feature_variant=variant,target_rates_replayed=True,test_sha=ref.test_sha))
   for p in CONFIG['prevalence_scenarios']:
    for limit in CONFIG['alert_budgets_per_100k']:
     threshold,sr,sf,salerts=source_budget_selection(sy,ss,p,limit)
     tr,tf=conf_rates(ty,ts,threshold)
     row=dict(holdout=holdout,seed=int(seed_txt),method=method,feature_variant=variant,prevalence_scenario=p,source_alert_budget_per_100k=limit,chosen_source_only_threshold=threshold,expected_source_alerts_per_100k=salerts,source_validation_tpr=sr,source_validation_fpr=sf,target_attack_recall=tr,target_test_fpr=tf,source_budget_met=salerts<=limit+1e-6,decision_uses_target_labels=False,**scenario(tr,tf,CONFIG['population_size'],p))
     row['target_budget_exceeded']=bool(row['expected_alerts_per_100k']>limit+1e-6)
     row['target_budget_excess_per_100k']=max(0.,row['expected_alerts_per_100k']-limit)
     budget.append(row)
   counts['completed']+=1
   if (j+1)%50==0:print('PREDICTIONS_ANALYZED',j+1,'/',len(preds),flush=True)
 budgets=pd.DataFrame(budget);budgets.to_csv(OUT/'source_budget_transfer.csv',index=False)
 pd.DataFrame(check).to_csv(OUT/'prediction_replay_checks.csv',index=False)
 # Mean per target domain / method / variant: seeds are not independent domains.
 keys=['method','feature_variant','threshold_strategy','target_validation_fpr','prevalence_scenario']
 summary=perf.groupby(keys,as_index=False).agg(mean_precision=('expected_precision','mean'),sd_precision=('expected_precision','std'),mean_alerts_per_100k=('expected_alerts_per_100k','mean'),mean_false_alerts_per_100k=('expected_fp_per_100k','mean'),mean_recall=('target_attack_recall','mean'),mean_test_fpr=('target_test_fpr','mean'),n=('seed','size'))
 summary.to_csv(OUT/'prevalence_summary_mean_sd.csv',index=False)
 bkeys=['method','feature_variant','prevalence_scenario','source_alert_budget_per_100k']
 bsum=budgets.groupby(bkeys,as_index=False).agg(mean_target_alerts=('expected_alerts_per_100k','mean'),mean_precision=('expected_precision','mean'),mean_recall=('target_attack_recall','mean'),budget_violation_rate=('target_budget_exceeded','mean'),n=('seed','size'))
 bsum.to_csv(OUT/'budget_transfer_summary.csv',index=False)
 # Macro-averaging four domains and five seeds gives equal domain weights.
 dsum=perf.groupby(['holdout','method','feature_variant','threshold_strategy','target_validation_fpr','prevalence_scenario'],as_index=False).agg(mean_precision=('expected_precision','mean'),mean_alerts_per_100k=('expected_alerts_per_100k','mean'),mean_recall=('target_attack_recall','mean'),mean_test_fpr=('target_test_fpr','mean'),n_seeds=('seed','nunique'))
 dsum.to_csv(OUT/'domain_specific_summary.csv',index=False)
 # Source budget comparisons matched on fold and seed.
 primary=budgets[budgets.feature_variant==CONFIG['primary_feature_variant']]
 pivot=primary.pivot_table(index=['holdout','seed','prevalence_scenario','source_alert_budget_per_100k'],columns='method',values=['target_attack_recall','expected_precision','expected_alerts_per_100k'],aggfunc='first')
 pairs=[]
 for comparator in ['LocalMLP','PooledMLP','PooledLightGBM','FedProxMLP']:
  for idx,row in pivot.iterrows():
   for metric in ['target_attack_recall','expected_precision','expected_alerts_per_100k']:
    x=row.get((metric,'FedAvgMLP'),np.nan);y=row.get((metric,comparator),np.nan)
    if pd.notna(x) and pd.notna(y):
     pairs.append(dict(holdout=idx[0],seed=idx[1],prevalence_scenario=idx[2],source_alert_budget_per_100k=idx[3],metric=metric,reference='FedAvgMLP',comparator=comparator,difference_fedavg_minus_comparator=float(x-y)))
 pd.DataFrame(pairs).to_csv(OUT/'paired_collaboration_deltas.csv',index=False)
 # Generate concise, research-ready figures (not a claim of measured natural traffic).
 try:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
  plot=perf[(perf.feature_variant=='all')&(perf.threshold_strategy=='pooled_source_quantile')&np.isclose(perf.target_validation_fpr,.01)]
  fig,ax=plt.subplots(figsize=(8,4.8))
  for m,g in plot.groupby('method'):
   temp=g.groupby('prevalence_scenario').expected_precision.mean()
   ax.plot(100*temp.index.to_numpy(),temp.to_numpy(),marker='o',label=m)
  ax.set(xlabel='Hypothetical attack prevalence (%)',ylabel='Expected precision',title='Prevalence-shift sensitivity (source-only 1% FPR target)',ylim=(0,1.03))
  ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(figdir/'precision_vs_assumed_prevalence.png',dpi=180);plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,4.8))
  q=budgets[(budgets.feature_variant=='all')&np.isclose(budgets.prevalence_scenario,.01)]
  for m,g in q.groupby('method'):
   t=g.groupby('source_alert_budget_per_100k').target_budget_exceeded.mean()
   ax.plot(t.index,t.values,marker='o',label=m)
  ax.set(xlabel='Source alert budget / 100,000 flows',ylabel='Fraction of shifted-target runs over budget',title='Budget-transfer failure at assumed 1% prevalence',ylim=(0,1.03))
  ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(figdir/'alert_budget_violation_rate.png',dpi=180);plt.close(fig)
 except Exception as e:
  (OUT/'plot_warning.txt').write_text(str(e))
 status={'status':'complete','input_experiment':'03','model_runs_checked':counts['completed'],'expected_model_runs':300,'prevalence_rows':len(perf),'source_budget_rows':len(budgets),'paired_deltas':len(pairs),'all_target_predictions_replayed':len(check)==300,'source_only_thresholds':True,'no_new_model_training':True,'hypothetical_prior_shift_only':True,'original_dataset_prevalence_is_not_real_world_prevalence':True}
 if counts['completed']!=300 or len(perf)!=5400 or len(budgets)!=3600:raise AssertionError('Unexpected row count '+str(status))
 jdump(OUT/'run_status.json',status)
 (OUT/'README_METHODS.txt').write_text('''Experiment 04: Operational reliability sensitivity from Experiment 03 saved scores.\n\nNo model training; uses 300 existing model-run predictions.\nSYNTHETIC PRIOR SHIFT: rates conditioned on class are assumed invariant.\nFor prevalence p, TPR r, FPR f, N=100000:\nTP=N*p*r; FP=N*(1-p)*f; alerts=TP+FP; PPV=TP/(TP+FP).\nF1=2TP/(2TP+FP+FN). These are EXPECTED counts, not observed flows.\nAll thresholds are learned from held-out SOURCE validation ONLY.\nSource-budget thresholds use both benign and attack source validation scores\nand satisfy expected alerts under the ASSUMED prevalence on source.\nTheir true transferred target alert volumes can exceed the budget.\nTarget labels used only for retrospective TPR/FPR and checks.\nTarget oracle from Experiment 03 is excluded from deployable analyses.\nSource-prescribed target FPR is not a target guarantee.\nOriginal source CSV prevalence is DATASET COMPOSITION, not operational prevalence.\nThe input test was class-balanced and thus cannot establish PPV in the wild.\nMarginal Wilson intervals for test rates give only conditional sensitivity ranges, not joint confidence intervals.\nCorrelations, capture-specific base rates, costs, and temporal changes are absent.\nFive random seeds are not five independent target domains.\n''')
 archive=ROOT/'FLIDS_Experiment04_Results.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in OUT.rglob('*'):
   if f.is_file():z.write(f,arcname=f.relative_to(ROOT))
 print('RESULT',archive,archive.stat().st_size,'STATUS',status,flush=True)
 return archive

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['run','audit'],default='run');args=p.parse_args()
 if args.mode=='audit':print('AUDIT',audit(discover())[0].shape)
 else:run()
