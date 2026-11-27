import zipfile,csv,io,collections,statistics as s, json
from pathlib import Path
p=Path.home()/'Downloads'; z=zipfile.ZipFile(p/'FLIDS_Packages10_13_Results.zip'); o=zipfile.ZipFile(p/'FLIDS_Exp09_Canonical_Input.zip')
def rows(n):return list(csv.DictReader(io.TextIOWrapper(z.open('summary/'+n+'.csv'))))

import numpy as np,math
ref={r['bundle']:r for r in rows('operating') if r['alpha']=='0.01'}
dd=collections.defaultdict(list);err=[];ex=[]
for n in z.namelist():
 if not n.endswith('/done.json'):continue
 i=json.loads(z.read(n))['identity']
 if i['scenario']!='native' or i['participation']!=1:continue
 a=np.load(io.BytesIO(z.read(n[:-9]+'scores.npz')))
 cs=a['cal_scores'][a['cal_y']==0];t=float(np.nextafter(np.sort(cs.astype(float))[len(cs)-math.floor(.01*len(cs))-1],np.inf))
 def op(k):
  y=a[k+'_y'];v=a[k+'_scores'];return [(v[y==c]>=t).mean() for c in (0,1)]
 f,r=op('target');sf,_=op('source_test');dd[i['held'][0],i['method']].append([sf,f,r]);ex.append(f>.01)
 b=n.split('/')[1]
 if b in ref:err.append(abs(f-float(ref[b]['target_fpr'])))
print('MATCH',len(err),max(err),'excess',sum(ex),len(ex))
for k,v in sorted(dd.items()):
 if k[1] in ('FedAvg','LogisticRegression','PooledMLP'):print(k,np.round(np.mean(v,axis=0)*100,3),len(v), 'range',np.round(np.array(v)[:,1]*100,1))
