"""Prespecified supplemental baselines on exactly the main-track cached splits.
Separate results: do not silently change original six-phase synthesis.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import argparse, importlib.util, json, hashlib, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

METHODS=['LogisticRegression','RandomForest','FedAdamMLP','FedYogiMLP']
SEEDS=[0,42,3407,123,2026]

def load_engine(path, tag):
    sp=importlib.util.spec_from_file_location(tag,str(path));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def fedopt(m, x, y, tr, clients, seed, device, rounds, method, track):
    m.seedall(seed)
    net=m.MLP(x.shape[1]);state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    moment={k:torch.zeros_like(v) for k,v in state.items()}
    second={k:torch.zeros_like(v) for k,v in state.items()}
    # Fixed, prespecified FedOpt hyperparameters; not selected on held-out targets.
    eta=0.001; beta1=.9;beta2=.99;tau=.01
    steps_total=0;bytes_total=0;paramcount=sum(t.numel() for t in net.parameters())
    for rnd in range(rounds):
        states=[];weights=[]
        for cid,ix in enumerate(clients):
            if len(ix)<10: continue
            if track==2:
                step=max(1,int(np.ceil(min(len(ix),m.CFG['client_round_cap'])/m.CFG['batch_size'])))*m.CFG['local_epochs']
                local=m.local_train(state,x,y,ix,seed+rnd*300+cid*79,step,device)
            else:
                local_model=m.MLP(x.shape[1]);local,step=m.fit_steps(local_model,state,x,y,ix,rnd,cid,seed,device)
            steps_total+=step;states.append(local);weights.append(len(ix))
        if not states:raise RuntimeError('NO_CLIENT_UPDATES')
        w=np.array(weights,dtype=np.float64);w/=w.sum()
        avg={k:sum(s[k]*float(z) for s,z in zip(states,w)) for k in state}
        updated={}
        for k in state:
            delta=avg[k]-state[k]
            moment[k]=beta1*moment[k]+(1-beta1)*delta
            if method=='FedAdamMLP':second[k]=beta2*second[k]+(1-beta2)*delta.square()
            else:second[k]=second[k]-(1-beta2)*delta.square()*torch.sign(second[k]-delta.square())
            mh=moment[k]/(1-beta1**(rnd+1))
            # nonnegative, stabilized Yogi second moment
            vh=(second[k]/(1-beta2**(rnd+1))).clamp_min(0)
            updated[k]=state[k]+eta*mh/(vh.sqrt()+tau)
        state=updated
        bytes_total+=2*paramcount*4*len(states)
    return state,steps_total,bytes_total

def run(track, work, engine, worker_id=0, workers=1, smoke=False):
    m=load_engine(engine,'flids_primary_engine_'+str(track));output=work/f'FLIDS_Extra_Baselines_Exp{track:02d}'
    (output/'records').mkdir(parents=True,exist_ok=True)
    if track==2:
        with np.load(m.CACHE) as z:
            x=m.transform(z['x']);y=z['y'];d=z['d']
        fp=np.load(m.ROOT/'flids_exp02_fingerprints.npy')
        seeds=m.CFG['seeds'];holds=m.CFG['domains'] if smoke else m.DOMAINS
        tasks=[(h,s,met) for h in holds for s in seeds for met in METHODS]
    else:
        p=pd.read_pickle(m.CACHE);y=p['df'].binary_label.to_numpy(dtype=np.int8)
        folds=json.loads((m.OUT/'split_index.json').read_text())
        seeds=m.EXPERIMENT['seeds'];holds=m.EXPERIMENT['heldouts']
        tasks=[(h,s,met) for h in holds for s in seeds for met in METHODS]
    errors=[];device=f'cuda:{worker_id%torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu'
    for ti,(held,seed,method) in enumerate(tasks):
        if ti%workers!=worker_id: continue
        path=output/'records'/f'{track}__{m.normalized(held) if track==2 else held}__{seed}__{method}.json'
        if path.exists():
            print('RESUME',ti+1,len(tasks),path.name,flush=True);continue
        try:
            if track==2:
                tr,va,te=m.split_indices(y,d,fp,held,seed);clients=[tr[d[tr]==domain] for domain in m.DOMAINS if np.any(d[tr]==domain)]
                xx=x;threshold=m.threshold;hashval=m.digest(te);valhash=m.digest(va);rounds=m.CFG['rounds']
            else:
                fold={k:np.asarray(v,dtype=np.int64) for k,v in folds[f'{held}__{seed}'].items()}
                tr=fold['train'];va=fold['val'];te=fold['test'];xx=m.fit_features(p,fold)
                clients=m.make_clients(tr,y,seed,m.EXPERIMENT['n_clients']);threshold=m.threshold_for_target
                hashval=m.sha_ids(te);valhash=m.sha_ids(va);rounds=m.EXPERIMENT['rounds']
            tic=time.perf_counter()
            if method=='LogisticRegression':
                model=LogisticRegression(C=1.0,max_iter=250,solver='liblinear',random_state=seed)
                model.fit(xx[tr],y[tr]);sv=model.predict_proba(xx[va])[:,1];st=model.predict_proba(xx[te])[:,1];steps=250;comm=0
            elif method=='RandomForest':
                model=RandomForestClassifier(n_estimators=150,max_depth=12,min_samples_leaf=5,n_jobs=1,random_state=seed)
                model.fit(xx[tr],y[tr]);sv=model.predict_proba(xx[va])[:,1];st=model.predict_proba(xx[te])[:,1];steps=150;comm=0
            else:
                state,steps,comm=fedopt(m,xx,y,tr,clients,seed,device,rounds,method,track)
                if track==2:sv=m.predict_model(state,xx,va,device);st=m.predict_model(state,xx,te,device)
                else:
                    model=m.MLP(xx.shape[1]);model.load_state_dict(state)
                    sv=m.infer_mlp(model,xx,va,device);st=m.infer_mlp(model,xx,te,device)
            results=[];neg=np.asarray(sv[y[va]==0],dtype=np.float64);st=np.asarray(st,dtype=np.float64)
            for target in [.001,.01,.05]:
                if target==.001 and len(neg)<1000:
                    results.append(dict(target_fpr=target,status='insufficient_benign_validation'));continue
                cut,valfpr=threshold(neg,target);pred=st>=cut;gt=y[te]
                tp=int(np.sum(pred & (gt==1)));fpnum=int(np.sum(pred & (gt==0)));fn=int(np.sum(~pred & (gt==1)));tn=int(np.sum(~pred & (gt==0)))
                if valfpr>target+1e-12:raise RuntimeError('VAL_FPR_VIOLATION')
                results.append(dict(target_fpr=target,status='evaluated',threshold=cut,validation_fpr=valfpr,test_fpr=fpnum/max(1,fpnum+tn),attack_recall=tp/max(1,tp+fn),precision=tp/max(1,tp+fpnum),tp=tp,fp=fpnum,tn=tn,fn=fn))
            obj=dict(holdout=held,seed=seed,method=method,track=track,train_rows=len(tr),val_rows=len(va),test_rows=len(te),test_sha=hashval,validation_sha=valhash,auroc=float(roc_auc_score(y[te],st)),auprc=float(average_precision_score(y[te],st)),train_seconds=time.perf_counter()-tic,optimizer_steps=steps,communication_bytes=comm,results=results,notes='Supplementary baseline. Pooled models are not federated. FedAdam/Yogi use simple explicit server FedOpt, fixed un-tuned hyperparameters.')
            path.write_text(json.dumps(obj,indent=2));print('BASELINE_OK',track,ti+1,len(tasks),held,seed,method,flush=True)
        except Exception:
            errors.append(dict(heldout=held,seed=seed,method=method,trace=traceback.format_exc()))
            print('BASELINE_FAILED',held,seed,method,errors[-1]['trace'][-1000:],flush=True)
    (output/f'failures_worker_{worker_id}.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError(f'{len(errors)} baseline tasks failed')

def aggregate(track,work,engine,smoke):
    m=load_engine(engine,'flids_for_aggregate_'+str(track));out=work/f'FLIDS_Extra_Baselines_Exp{track:02d}'
    rec=[json.loads(p.read_text()) for p in (out/'records').glob('*.json')]
    holds=m.CFG['domains'] if track==2 else m.EXPERIMENT['heldouts'];seeds=m.CFG['seeds'] if track==2 else m.EXPERIMENT['seeds']
    expect=len(holds)*len(seeds)*len(METHODS)
    if len(rec)!=expect:raise RuntimeError(f'INCOMPLETE_EXTRA_BASELINES {len(rec)}/{expect}')
    rows=[]
    for r in rec:
        for op in r['results']:
            rows.append({k:r[k] for k in ('holdout','seed','method','track','test_sha','validation_sha','auroc','auprc','train_seconds','communication_bytes')}|op)
    df=pd.DataFrame(rows);df.to_csv(out/'extra_baseline_metrics.csv',index=False)
    summ=df[df.status=='evaluated'].groupby(['holdout','method','target_fpr'])[['attack_recall','test_fpr','auroc']].agg(['mean','std','count'])
    summ.to_csv(out/'extra_baseline_summary.csv')
    # Compare to original split digests, avoiding mixed-split headline comparisons.
    audit=[]
    if track==2:
        original=pd.read_csv(m.OUT/'experiment_results.csv')
        original=original[['holdout','seed','test_sha','validation_sha']].dropna(subset=['validation_sha']).drop_duplicates()
    else:
        original=pd.read_csv(m.OUT/'experiment_results.csv')
        original=original[['heldout_family','seed','test_sha','validation_sha']].rename(columns={'heldout_family':'holdout'}).dropna(subset=['validation_sha']).drop_duplicates()
    for (h,s),g in df.groupby(['holdout','seed']):
        src=original[(original.holdout==h)&(original.seed==s)]
        ok=len(src)==1 and len(set(g.test_sha))==1 and g.test_sha.iloc[0]==src.test_sha.iloc[0] and g.validation_sha.iloc[0]==src.validation_sha.iloc[0]
        audit.append(dict(holdout=h,seed=int(s),matched=bool(ok)))
    if not all(z['matched'] for z in audit):raise RuntimeError('SUPPLEMENTARY_SPLIT_MISMATCH')
    (out/'split_match_audit.json').write_text(json.dumps(audit,indent=2))
    (out/'run_status.json').write_text(json.dumps(dict(completed=expect,methods=METHODS,matched_main_splits=True,not_in_original_exp07_synthesis=True),indent=2))
    import zipfile
    archive=work/f'FLIDS_Extra_Baselines_Exp{track:02d}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(work))
    print('SUPPLEMENTAL_READY',archive,expect,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--track',type=int,choices=[1,2],required=True);ap.add_argument('--mode',choices=['main','aggregate'],required=True);ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--work',default='/kaggle/working');ap.add_argument('--engine',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.mode=='main':run(args.track,Path(args.work),args.engine,args.worker,args.workers,args.smoke)
    else:aggregate(args.track,Path(args.work),args.engine,args.smoke)
