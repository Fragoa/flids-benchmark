"""Usage: python source_check.py FRESH_RESULTS.zip EXPERIMENT01_RESULTS.zip"""
import zipfile,io,sys
from pathlib import Path
from scipy.stats import binom
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
with zipfile.ZipFile(sys.argv[1]) as z:
 p=pd.read_csv(z.open('summary/policies.csv'),float_precision='round_trip').query('track=="nf" and scenario=="native" and participation==1 and alpha==.01 and policy=="FedRAT_tolerance_prototype"')
 rows=[]
 for row in p.itertuples():
  with np.load(io.BytesIO(z.read('bundles/'+row.bundle+'/scores.npz'))) as a:
   y=a['source_test_y'];s=a['source_test_scores'].astype(float);d=a['source_test_d']
   cy=a['cal_y'];cs=a['cal_scores'].astype(float);cd=a['cal_d'];groups=np.unique(cd[cy==0])
   thresholds=[]
   for g in groups:
    vals=np.sort(cs[(cy==0)&(cd==g)]);i=int(binom.ppf(1-.05/len(groups),len(vals),.99))
    thresholds.append(np.nextafter(vals[i],np.inf) if i<len(vals) else np.inf)
   threshold=max(thresholds)
   rates=[]
   for g in np.unique(d):
    mask=(y==0)&(d==g);rates.append((s[mask]>=threshold).mean());rows.append([row.bundle,g,rates[-1]])
   if not np.isclose((s[y==0]>=threshold).mean(),row.source_test_fpr):print(row.method,threshold,row.source_test_fpr,(s[y==0]>=threshold).mean(),flush=True)
   assert np.isclose((s[y==0]>=threshold).mean(),row.source_test_fpr)
 df=pd.DataFrame(rows,columns=['bundle','source','test_fpr']);df.to_csv(R/'tables/source_groups.csv',index=False);print(sum(df.test_fpr<=.01),sum(df.groupby('bundle').test_fpr.max()<=.01))
with zipfile.ZipFile(sys.argv[2]) as z:
 p=pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith('/predictions.csv'))),float_precision='round_trip')
 print(p.method.unique());v=p.pivot(index=['heldout_family','seed','row_id'],columns='method',values='attack_score')
 k=[c for c in v.columns if 'Fed' in c];print(k)
 if len(k)==2:
  v=v[k].dropna();g=(v[k[0]]==v[k[1]]).groupby(['heldout_family','seed']).agg(['mean','size']);g.to_csv(R/'tables/edge_score_equality.csv');print(g['mean'].describe(),len(g))
