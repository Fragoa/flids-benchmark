import tempfile,json,sys,os,zipfile,shutil,hashlib,importlib.util
from pathlib import Path
n=json.load(open(Path(__file__).resolve().parents[1]/'notebooks/current/FLIDS_Master_V3.ipynb'));D=Path.home()/'Downloads'
with tempfile.TemporaryDirectory() as t:
 WORK=Path(t)/'work';WORK.mkdir();INPUT=Path(t)/'input';INPUT.mkdir()
 EXP09_ZIP=str(D/'FLIDS_Experiment09_Results.zip');RESUME_ZIP=str(D/'FLIDS_Packages10_13_Results.zip')
 EDGE_CSV=EDGE_FEATURES_JSON='';EXTERNAL_INPUTS=EDGE_FEATURES=[];TEMPORAL_INPUT=None;SMOKE_TEST=REQUIRE_ALL_INPUTS=False;ALLOW_NEW_EDGE_PROTOCOL25=True
 display=FileLink=lambda x:x
 for i in [2,3,4,5,6,6]:exec(''.join(n['cells'][i]['source']))
 import ast,numpy as np,pandas as pd
 from types import SimpleNamespace
 original=SimpleNamespace(FEATURES=NF_FEATURES)
 mod=ast.parse((WORK/'master.py').read_text())
 for f in mod.body:
  if getattr(f,'name','') in ['load_temporal','load_edge','dedup','save','npwrite','rng','seed']:
   exec(compile(ast.Module([f],[]),'fixture','exec'))
 features=[c for c in pre.EDGE_PROTOCOL25 if not c.startswith('dns.')];df=pd.DataFrame({c:np.arange(300) for c in features});df['Attack_label']=[0]*20+[1]*280;df['Attack_type']=['benign']*20+list(map(str,range(14)))*20;df.to_csv(WORK/'edge.csv',index=False)
 load_edge(WORK,dict(edge_csv=WORK/'edge.csv',edge_features=features,edge_schema='Edge-Numeric19-v3',edge_label='Attack_label',edge_family='Attack_type',edge_cap_per_family=6000))
 df=pd.DataFrame({c:np.arange(300,dtype=float) for c in NF_FEATURES});df['time']=range(300);df['Label']=np.arange(300)%2;df['Attack']='test';df.loc[250,NF_FEATURES[0]]=np.inf;df.to_csv(WORK/'time.csv',index=False)
 vals=load_temporal(WORK,dict(path=WORK/'time.csv',time_col='time',time_unit='ms'));assert len(vals[0])==299
 print('PASS optional fixtures')
