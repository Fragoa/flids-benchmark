import argparse
import json
import zipfile
from pathlib import Path
import pandas as pd
parser=argparse.ArgumentParser()
parser.add_argument('archive',type=Path)
parser.add_argument('--output',type=Path,default=Path('outputs'))
args=parser.parse_args()
r=args.output
r.mkdir(parents=True,exist_ok=True)
z=zipfile.ZipFile(args.archive)
def read(n):return pd.read_csv(z.open('summary/'+n+'.csv')).query('track=="nf"')
o=read('operating').query('alpha==0.01 and status=="evaluated"'); b=o.query('scenario=="native" and participation==1').copy(); assert len(b)==300 and len(o)==1420
cols=['auroc','pauc01','source_test_fpr','target_fpr','recall','oracle_fpr','oracle_recall']
def tab(d,keys,name):
 t=d.groupby(keys)[cols].agg(['size','mean','std','median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)]);t.columns=[k+'_'+v.replace('<lambda_0>','q25').replace('<lambda_1>','q75') for k,v in t.columns];t.to_csv(r/name)
tab(b,['method','held'],'native_domains.csv');tab(o,['method','scenario','participation'],'robustness.csv')
p=read('policies').query('alpha==0.01 and scenario=="native" and participation==1');p.groupby(['method','held','policy'])[['target_fpr','recall','abstain_all']].mean().to_csv(r/'policy_by_domain.csv')
f=read('few_shot').query('scenario=="native" and participation==1');f.groupby(['method','held','n_target_cal'])[['target_fpr','recall']].mean().to_csv(r/'target_calibration.csv')
t=b.query('method=="FedAvg" and held=="NF-ToN-IoT"').copy();t['selected_config']=[json.dumps(json.loads(z.read('bundles/'+k+'/done.json'))['config']) for k in t.bundle];t[['seed','selected_config','target_fpr','recall']].to_csv(r/'ton_seeds.csv',index=False)
z.close()
