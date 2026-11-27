"""Run from the source directory: python analyze.py /path/to/fresh-results.zip"""
import sys,zipfile,io,json,math
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent; T=R/'tables'; z=zipfile.ZipFile(sys.argv[1])
def load(name):return pd.read_csv(z.open('summary/'+name+'.csv')).query('track=="nf"')
o=load('operating').query('alpha==.01 and status=="evaluated"');b=o.query('scenario=="native" and participation==1').copy();assert len(b)==300 and len(o)==1420
pol=load('policies').query('alpha==.01 and scenario=="native" and participation==1').copy()
def data(name):return np.load(io.BytesIO(z.read(name)),allow_pickle=False)
def cut(s,alpha):return np.sort(s.astype(float))[len(s)-math.floor(alpha*len(s))-1]
f=load('few_shot').query('scenario=="native" and participation==1 and method=="FedAvg"')
fig,axes=plt.subplots(2,4,figsize=(9,4.2),layout='constrained',sharey='row')
cache=data('nf_cache.npz');X=cache['x'];dom=cache['d'];yall=cache['y'];bt=dom=='NF-BoT-IoT'
# Describe distances in the same signed-log coordinates, scaling on the three source datasets.
x=np.clip(np.sign(X.astype(float))*np.log1p(np.abs(np.clip(X.astype(float),-1e12,1e12))),-18,18)
sd=x[~bt].std(0);x=x/np.where(sd>0,sd,1);tree=cKDTree(x[bt & (yall==1)])
tail=[];bot=[];all_scores={}
for idx,row in b.iterrows():
 a=data('bundles/'+row.bundle+'/scores.npz');y=a['target_y'];v=a['target_scores'];neg=v[y==0];pos=v[y==1]
 q=cut(a['cal_scores'][a['cal_y']==0],.01)
 assert np.isclose((neg>q).mean(),row.target_fpr) and np.isclose((pos>q).mean(),row.recall)
 rates={str(alpha):float((pos>cut(neg,alpha)).mean()) for alpha in [.001,.01,.05,.1]}
 assert np.isclose(rates['0.01'],row.oracle_recall)
 tail.append(dict(method=row.method,held=row.held,seed=row.seed,**rates))
 if row.method=='FedAvg':
  j=sorted(b.held.unique()).index(row.held);ax=axes[0,j];fp,tp,_=roc_curve(y,v,drop_intermediate=False);ax.plot(np.maximum(fp*100,.001),tp*100,lw=.65);ax.scatter([row.target_fpr*100],[row.recall*100],s=10,color='black')
  if j==0:
   ids=a['target_ids'][y==0];assert (dom[ids]==row.held).all() and (yall[ids]==0).all()
   if not all_scores:all_scores['dist']=tree.query(x[ids],k=1)[0];all_scores['ids']=ids
   assert np.array_equal(all_scores['ids'],ids)
   ii=np.argsort(-neg,kind='stable')[:math.ceil(.01*len(neg))];dst=all_scores['dist'];mask=np.zeros(len(ids),bool);mask[ii]=True
   bot.append(dict(seed=row.seed,n=len(ii),exact=int((dst[ii]==0).sum()),median_top=np.median(dst[ii]),median_rest=np.median(dst[~mask]),cutoff_ties=int((neg==neg[ii[-1]]).sum())))
for j,h in enumerate(sorted(b.held.unique())):
 ax=axes[0,j];ax.set(xscale='log',xlim=(.01,100),ylim=(-3,103),title=h[3:],xlabel='Target FPR (%)');ax.axvline(1,color='grey',ls=':')
 v=f[f.held==h].groupby('n_target_cal').recall.mean();axes[1,j].plot(v.index,v*100,'o-',ms=3);axes[1,j].set_xlabel('Benign calibration flows')
for a in axes[:,0]:a.set_ylabel('Attack recall (%)')
fig.savefig(R/'figs/tail_calibration.pdf');pd.DataFrame(bot).to_csv(T/'bot_tail.csv',index=False)
t=pd.DataFrame(tail);t.to_csv(T/'tail_per_run.csv',index=False)
base=b.set_index(['held','seed']).query('method=="FedAvg"');cols=['auroc','pauc01','target_fpr','recall','oracle_recall']
pair=pd.concat({k:g.set_index(['held','seed'])[cols]-base[cols] for k,g in b.groupby('method')},names=['method']);pair.to_csv(T/'paired_differences.csv')
# Finite log error at zero false positives via a half-count correction.
pol=pol.merge(b[['bundle','target_benign_n']],on='bundle',validate='many_to_one');pol['abs_log_ratio']=abs(np.log(((pol.target_fpr*pol.target_benign_n+.5)/(pol.target_benign_n+1))/.01));pol['source_coverage']=pol.source_test_fpr<=.01;pol['target_coverage']=pol.target_fpr<=.01
pol.groupby(['method','policy'])[['source_coverage','target_coverage','abs_log_ratio','source_test_fpr','target_fpr','recall']].mean().to_csv(T/'policy_coverage.csv')
b['alerts']=1e5*(.99*b.target_fpr+.01*b.recall);b['precision']=1e3*b.recall/b.alerts;b.groupby('method')[['alerts','precision']].mean().to_csv(T/'alerts.csv')
print('key',b.oracle_recall.mean(),b.groupby('method')[['target_fpr','oracle_recall']].mean().corr().iloc[0,1],sum(b.source_test_fpr>.01))
print(pol.groupby('policy')[['source_coverage','target_coverage','abs_log_ratio','target_fpr','recall']].mean().round(4))
print(b.query('method=="FedAvg"')[['alerts','precision']].mean(),b.query('method=="FedAvg"').recall.mean()*.01/(b.query('method=="FedAvg"').alerts.mean()/1e5));print(bot)
print(o.query('method=="FedAvg" and participation==1').groupby('scenario')[cols].mean());print(z.read('nf_audit.json'))
