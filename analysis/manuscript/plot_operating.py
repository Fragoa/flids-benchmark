"""Render the primary operating-point figure from the archived five-seed holdout means."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
root=Path(__file__).resolve().parent
d=pd.read_csv(root/'tables/native_domains.csv').groupby('method').mean(numeric_only=True)
styles={'Federated':('o','#176987'),'Centralized':('s','#be5829'),'Local':('D','#527540')}
fig,(ax,key)=plt.subplots(1,2,figsize=(7.1,2.65),gridspec_kw={'width_ratios':[2.3,1]},layout='constrained')
key.axis('off');ax.axvline(1,color='0.35',ls='--',lw=.9)
for i,(method,row) in enumerate(d.iterrows(),1):
    group='Federated' if method.startswith('Fed') or method=='SCAFFOLD' else 'Local' if method=='LocalMLP' else 'Centralized'
    marker,color=styles[group];x=100*row.target_fpr_mean;y=100*row.recall_mean
    ax.scatter(x,y,marker=marker,color=color,s=24)
    ofs={1:(4,3),2:(4,3),3:(-7,1),4:(-9,3),5:(2,3),6:(-8,-8),7:(4,-7),8:(2,-8),9:(3,-9),10:(4,-7),11:(4,0),12:(4,2),13:(4,1),14:(4,1),15:(3,-8)}[i]
    ax.annotate(str(i),(x,y),xytext=ofs,textcoords='offset points',fontsize=7)
    key.text(0,1-(i-.5)/15,f'{i:2d}   '+method.replace('_SourceEnsemble',' ensemble').replace('_',' '),color=color,fontsize=8,va='center')
for name,(marker,color) in styles.items():ax.scatter([],[],marker=marker,color=color,label=name,s=22)
ax.legend(loc='upper left',fontsize=7,frameon=False);ax.set(xlabel='Target false-positive rate (%)',ylabel='Attack recall (%)',xlim=(0,32),ylim=(19,64));ax.grid(alpha=.16)
fig.savefig(root/'figs/tuned_operating.pdf',bbox_inches='tight')
