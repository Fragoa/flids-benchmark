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
import matplotlib.pyplot as plt, numpy as np, io
from sklearn.metrics import roc_curve
o=pd.read_csv(z.open('summary/operating.csv')).query('track=="nf" and method=="FedAvg" and scenario=="native" and participation==1 and alpha==0.01');f=pd.read_csv(r/'target_calibration.csv').query('method=="FedAvg"');fig,ax=plt.subplots(2,4,figsize=(9,4),sharey='row',layout='constrained')
for j,(dom,g) in enumerate(o.groupby('held')):
 for key in g.bundle:
  with np.load(io.BytesIO(z.read('bundles/'+key+'/scores.npz'))) as a:
   x,y,_=roc_curve(a['target_y'],a['target_scores']);ax[0,j].plot(x.clip(1e-5),y,lw=.7,alpha=.7)
 a=ax[0,j];a.set(xscale='log',xlim=(.0001,1),title=dom[3:],xlabel='Target FPR');a.axvline(.01,ls=':',color='black')
 h=f[f.held==dom];ax[1,j].plot(h.n_target_cal,h.recall*100,'o-',ms=3);ax[1,j].set(xlabel='Benign calibration flows')
ax[0,0].set_ylabel('Attack recall');ax[1,0].set_ylabel('Attack recall (%)');fig.savefig(r/'tail_calibration.pdf')
z.close()
