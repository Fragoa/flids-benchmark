"""Usage: python operational.py FRESH_RESULTS.zip"""
from pathlib import Path
import sys,zipfile
import numpy as np,pandas as pd
from scipy.stats import binomtest
r=Path(__file__).resolve().parent/'tables'
with zipfile.ZipFile(sys.argv[1]) as z:
 for name,groups,metrics in [('prevalence',['method','prevalence_assumed'],['alerts_per_100k','ppv_formula']),('alert_budget',['method','budget','assumed_prevalence'],['source_expected_alerts','target_expected_alerts','target_fpr','recall','exceeded']),('operating',['method'],['ece','brier','auroc','pauc01'])]:
  d=pd.read_csv(z.open('summary/'+name+'.csv')).query('track=="nf" and scenario=="native" and participation==1')
  if name=='alert_budget':d['exceeded']=d.target_expected_alerts>d.budget
  if name=='operating':d=d.query('alpha==.01')
  d.groupby(groups)[metrics].mean().to_csv(r/('tuned_'+name+'.csv'))
d=pd.read_csv(r/'paired_differences.csv').groupby(['method','held']).oracle_recall.mean()
rows=[]
for k,g in d.groupby(level=0):
 n=int((g!=0).sum());wins=int((g>0).sum())
 rows.append([k,wins,n,np.nan if not n else binomtest(wins,n,.5).pvalue,g.mean(),g.std()])
pd.DataFrame(rows,columns=['method','positive_domains','nonzero_domains','exact_two_sided_p','mean_delta','domain_sd']).to_csv(r/'paired_sign.csv',index=False)
