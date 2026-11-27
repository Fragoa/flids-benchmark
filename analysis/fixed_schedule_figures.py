import matplotlib
matplotlib.use('Agg')
import pandas as pd,numpy as np,matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import PercentFormatter
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('data',type=Path,help='Directory containing the expanded Experiment 07 CSV tables')
parser.add_argument('output',type=Path)
args=parser.parse_args()
D=args.data
O=args.output
O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.right':False,'axes.spines.top':False,'savefig.bbox':'tight','pdf.fonttype':42})
N={'PooledLightGBM':'Pooled LGBM','RandomForest':'Random Forest','PooledMLP':'Pooled MLP','LogisticRegression':'Logistic Reg.','LocalMLP':'Local MLP','FedAvgMLP':'FedAvg','FedProxMLP':'FedProx','FedAdamMLP':'FedAdam','FedYogiMLP':'FedYogi'}
colors={'Federated':'#4265aa','Centralized':'#d48137','Local':'#359b85'}
# Fig 1 pathway
fig,ax=plt.subplots(figsize=(7.1,1.5));ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
items=[('Source scan\n& audit',0.02),('Frozen\nholdouts',2.03),('9-model\ntraining',4.05),('Source-only\nthresholds',6.07),('Reliability\n& uncertainty',8.09)]
for t,x in items:
 r=plt.Rectangle((x,.6),1.82,.85,facecolor='#eaf0f7',edgecolor='#6b7895',linewidth=1.0);ax.add_patch(r)
 ax.text(x+.91,1.015,t,ha='center',va='center',size=8.1,fontweight='bold')
for i in range(4):
 x=1.86+2.02*i;ax.annotate('',xy=(x+.17,1.02),xytext=(x+.01,1.02),arrowprops={'arrowstyle':'->','lw':1,'color':'#444'})
ax.text(5,0.19,'Independent tasks: Edge-IIoTset unseen attack families  |  NF-UQ-NIDS-v2 unseen source datasets',ha='center',size=7.2,color='#555')
fig.savefig(O/'pipeline.pdf');plt.close(fig)
# Fig 2 9-method tradeoff (cross domain)
df=pd.read_csv(D/'table_exp02_all9_domain_CI.csv');df=df[df.target_fpr==.01]
fig,ax=plt.subplots(figsize=(6.95,3.0))
for _,r in df.iterrows():
 c=colors[r.method_family]
 ax.scatter(r.test_fpr_mean*100,r.attack_recall_mean*100,s=75,color=c,edgecolor='white',linewidth=1,zorder=3)
 shifts={'FedAvgMLP':(7,6),'FedProxMLP':(8,-9),'FedAdamMLP':(8,8),'FedYogiMLP':(7,-9),'PooledLightGBM':(-3,-13),'RandomForest':(-3,-12),'PooledMLP':(8,5),'LocalMLP':(7,4),'LogisticRegression':(7,-5)}
 ax.annotate(N[r.method],(r.test_fpr_mean*100,r.attack_recall_mean*100),xytext=shifts[r.method],textcoords='offset points',ha='left' if shifts[r.method][0]>0 else 'right',size=7.5)
ax.axvline(1,color='#c63636',ls='--',lw=1,label='Source validation FPR target: 1%');ax.set(xlabel='Realized target-domain false-positive rate (%)',ylabel='Target attack recall (%)',xlim=(-1.5,39),ylim=(-4,76));ax.grid(alpha=.14)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker='o',linestyle='',color='w',markerfacecolor=col,markersize=7,label=grp) for grp,col in colors.items()]+[Line2D([0],[0],color='#c63636',ls='--',label='1% source target')],loc='upper left',fontsize=7,ncol=2)
fig.savefig(O/'cross_domain_scatter.pdf');plt.close(fig)
# Fig 3 heterogeneity bar with AUROC line secondary? differences 3 panels 2 dims
H=pd.read_csv(D/'table_exp06_controlled_heterogeneity_CI.csv');h=H[H.method=='FedAvgMLP'].set_index('allocation_mode')
order=['baseline','quantity_mild','quantity_severe','label_mild','label_severe','domain_mild','domain_severe'];labs=['Control','Qty mild','Qty severe','Label mild','Label severe','Domain mild','Domain severe']
fig,axs=plt.subplots(1,2,figsize=(7.0,2.7),gridspec_kw={'width_ratios':[1.7,1]})
vs=[h.loc[k,'target_recall_mean']*100 for k in order];fs=[h.loc[k,'actual_test_fpr_mean']*100 for k in order]
barcolors=['#5d7ba0']+['#4f9a89']*2+['#d58256']*2+['#b1657b']*2
axs[0].bar(range(7),vs,color=barcolors);axs[0].set_ylabel('Recall (%)');axs[0].set_xticks(range(7),labs,rotation=42,ha='right',fontsize=7);axs[0].grid(axis='y',alpha=.17)
axs[1].bar(range(7),fs,color=barcolors);axs[1].set_ylabel('Target FPR (%)');axs[1].set_xticks(range(7),labs,rotation=42,ha='right',fontsize=7);axs[1].grid(axis='y',alpha=.17)
fig.subplots_adjust(bottom=.27,wspace=.4)
fig.savefig(O/'heterogeneity.pdf');plt.close(fig)
# Fig 4 prevalence panels
E=pd.read_csv(D/'table_exp04_hypothetical_prevalence.csv');E=E[E.prevalence_scenario==.01].set_index('method')
methods=['PooledLightGBM','PooledMLP','LocalMLP','FedAvgMLP','FedProxMLP'];labs=['Pooled LGBM','Pooled MLP','Local MLP','FedAvg','FedProx']
tp=np.array([E.loc[m,'expected_tp_per_100k'] for m in methods]);fp=np.array([E.loc[m,'expected_fp_per_100k'] for m in methods]);
fig,axs=plt.subplots(1,2,figsize=(7,2.45),gridspec_kw={'width_ratios':[1.4,1]})
y=np.arange(len(methods));axs[0].barh(y,fp,color='#a9bbc9',label='Expected false alerts');axs[0].barh(y,tp,left=fp,color='#4475ab',label='Expected true alerts');axs[0].set_yticks(y,labs,fontsize=7);axs[0].invert_yaxis();axs[0].set_xlabel('Expected alerts per 100k flows');axs[0].grid(axis='x',alpha=.15);axs[0].legend(fontsize=6.8,loc='lower right')
ppv=np.array([E.loc[m,'expected_precision']*100 for m in methods]);axs[1].barh(y,ppv,color='#568b82');axs[1].set_yticks(y,[]);axs[1].invert_yaxis();axs[1].set_xlabel('Mean precision (%)');axs[1].grid(axis='x',alpha=.15)
for i,x in enumerate(ppv):axs[1].text(x+.15,i,f'{x:.1f}%',va='center',fontsize=7)
fig.subplots_adjust(wspace=.33)
fig.savefig(O/'prevalence_operational.pdf');plt.close(fig)
print([str(p) for p in O.glob('*')])
