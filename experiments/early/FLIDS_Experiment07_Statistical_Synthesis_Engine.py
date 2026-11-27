#!/usr/bin/env python3
"""Reproducible statistical synthesis for FL-IDS Experiments 01--06.

Inputs: six original ZIP archives. No model training or target-label threshold selection.
A domain, not a random seed, is the primary independent cluster for cross-domain inference.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, csv, io, json, zipfile, hashlib, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

NAMES = {i:f'FLIDS_Experiment{i:02d}_Results.zip' for i in range(1,7)}
SEEDS_EXPECTED=5

def locate(name, roots):
    hits=[]
    for root in roots:
        root=Path(root)
        if not root.exists():continue
        if root.is_file() and root.name==name:hits.append(root)
        elif root.is_dir():hits.extend(root.rglob(name))
    hits=list(dict.fromkeys(str(p.resolve()) for p in hits if p.is_file()))
    if not hits:
        # Kaggle Notebook Outputs can expose extracted directories rather than the original ZIP.
        folder_name=name[:-4]
        folders=[]
        for root in roots:
            root=Path(root)
            if not root.is_dir():continue
            folders += [p for p in root.rglob(folder_name) if p.is_dir() and (p/'run_status.json').exists()]
        if folders:
            src=sorted(folders,key=str)[0]
            cache=Path(os.getenv('FLIDS_EXP07_CACHE','/kaggle/working/FLIDS_Experiment07_InputCache'))
            cache.mkdir(parents=True,exist_ok=True)
            target=cache/name
            if not target.exists():
                required=['run_status.json','experiment_results.csv','threshold_transfer_metrics.csv','calibration_audit.csv','prevalence_reweighted_operating_points.csv','source_budget_transfer.csv','communication_cost.csv']
                with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
                    for file_name in required:
                        match=src/file_name
                        if match.is_file():z.write(match,arcname=folder_name+'/'+file_name)
            print('REPACKED_EXTRACTED_OUTPUT',src,'->',target,flush=True)
            return target
        raise FileNotFoundError(f'MISSING_SOURCE_ARCHIVE: {name}. Attach all six original experiment outputs using Kaggle Add Input or provide --input-root.')
    if len(hits)>1:
        print(f'WARNING_MULTIPLE_COPIES {name}: selected {hits[0]} among {len(hits)} copies',flush=True)
    return Path(hits[0])

def load_table(z, suffix):
    matches=[x for x in z.namelist() if x.endswith('/'+suffix) or x==suffix]
    if len(matches)!=1:raise RuntimeError(f'Expected one {suffix}; found {len(matches)}')
    with z.open(matches[0]) as f:return pd.read_csv(f)

def load_json(z,suffix):
    matches=[x for x in z.namelist() if x.endswith('/'+suffix) or x==suffix]
    if len(matches)!=1:raise RuntimeError(f'Expected one {suffix}; found {len(matches)}')
    return json.loads(z.read(matches[0]))

def cluster_ci(vals):
    a=np.array(list(vals),dtype=float);a=a[np.isfinite(a)]
    if len(a)<2:return float(np.mean(a)) if len(a) else np.nan,np.nan,np.nan,len(a)
    mu=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
    w=float(t.ppf(.975,len(a)-1)*se)
    return mu,mu-w,mu+w,len(a)

def domain_summary(df, domain='holdout', group=('method',), metrics=('attack_recall','test_fpr','auroc')):
    seed_means=df.groupby([domain,*group,'seed'],as_index=False)[list(metrics)].mean()
    domain_means=seed_means.groupby([domain,*group],as_index=False)[list(metrics)].mean()
    records=[]
    for k,g in domain_means.groupby(list(group),dropna=False):
        if not isinstance(k,tuple):k=(k,)
        rec=dict(zip(group,k))
        for col in metrics:
            mu,lo,hi,n=cluster_ci(g[col]);rec.update({col+'_mean':mu,col+'_CI95_low':lo,col+'_CI95_high':hi})
        rec['n_domains']=len(g);rec['n_seed_runs']=len(seed_means[seed_means[list(group)[0]]==k[0]]) if len(group)==1 else np.nan
        records.append(rec)
    return pd.DataFrame(records),domain_means

def paired_cluster(df, baseline, candidate, domain='holdout', group=(), metric='attack_recall'):
    d=df[df.method.isin([baseline,candidate])].copy()
    keys=[domain,'seed',*group]
    pv=d.pivot_table(index=keys,columns='method',values=metric,aggfunc='mean').reset_index()
    pv=pv.dropna(subset=[baseline,candidate])
    pv['delta']=pv[candidate]-pv[baseline]
    dm=pv.groupby(domain,as_index=False)['delta'].mean()
    mu,lo,hi,n=cluster_ci(dm.delta)
    return {'comparison':candidate+' minus '+baseline,'metric':metric,'difference_mean':mu,'domain_CI95_low':lo,'domain_CI95_high':hi,'n_domains':n,'n_paired_seed_units':len(pv)},pv,dm

def save_csv(df,path):df.to_csv(path,index=False,float_format='%.9g')

def style():
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'figure.dpi':120,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})

def saveplot(fig,path):fig.savefig(path,bbox_inches='tight');plt.close(fig)

def plot_bar(df,x,y,hue,title,ylabel,path,scale=1):
    pivot=df.pivot(index=x,columns=hue,values=y).sort_index()
    ax=pivot.mul(scale).plot(kind='bar',figsize=(11,5),rot=18,width=.78)
    ax.set(title=title,ylabel=ylabel,xlabel='');ax.grid(axis='y',alpha=.22);ax.legend(loc='best',fontsize=8)
    saveplot(ax.figure,path)

def run(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    tabs={};status={};provenance=[]
    for i in range(1,7):
        p=locate(NAMES[i],[root,'/kaggle/input','/kaggle/working'])
        sha=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):sha.update(b)
        with zipfile.ZipFile(p) as z:
            tabs[i]={'result':load_table(z,'experiment_results.csv') if i!=4 and i!=3 else None,
                     'status':load_json(z,'run_status.json')}
            if i==3:
                tabs[i]['metrics']=load_table(z,'threshold_transfer_metrics.csv')
                tabs[i]['calibration']=load_table(z,'calibration_audit.csv')
            elif i==4:
                tabs[i]['prevalence']=load_table(z,'prevalence_reweighted_operating_points.csv')
                tabs[i]['budget']=load_table(z,'source_budget_transfer.csv')
            if i==2:tabs[i]['comm']=load_table(z,'communication_cost.csv')
        status[str(i)]=tabs[i]['status'];provenance.append({'experiment':i,'path':str(p),'sha256':sha.hexdigest(),'bytes':p.stat().st_size})
        print('LOADED',i,p,'rows',len(tabs[i]['result']) if tabs[i]['result'] is not None else '-',flush=True)
    save_csv(pd.DataFrame(provenance),out/'source_archive_provenance.csv')
    (out/'source_run_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    # Strict completeness checks: operating points differ from model runs.
    expected={1:300,2:300,3:2700,4:5400,5:720,6:840}
    actual={1:len(tabs[1]['result']),2:len(tabs[2]['result']),3:len(tabs[3]['metrics']),4:len(tabs[4]['prevalence']),5:len(tabs[5]['result']),6:len(tabs[6]['result'])}
    for i in expected:
        if actual[i]!=expected[i]:raise RuntimeError(f'INCOMPLETE_EXPERIMENT {i}: found {actual[i]} expected {expected[i]}')
    # Exp01: four held-out attacks, source-FPR target 1%.
    e1=tabs[1]['result'];e1=e1[np.isclose(e1.target_fpr,.01)].copy().rename(columns={'heldout_family':'holdout','unknown_recall':'attack_recall'})
    s1,dm1=domain_summary(e1,metrics=('attack_recall','test_fpr','auroc','auprc'))
    save_csv(s1,out/'table_exp01_open_set_domain_CI.csv')
    save_csv(dm1,out/'exp01_domain_seed_means.csv')
    plot_bar(dm1,'holdout','attack_recall','method','Experiment 01: unseen attack recall (source FPR target 1%)','Recall (%)',figdir/'fig01_open_set_recall.png',100)
    # Exp02: cross-domain methods.
    e2=tabs[2]['result'];e2=e2[np.isclose(e2.target_fpr,.01)].copy()
    s2,dm2=domain_summary(e2,metrics=('attack_recall','test_fpr','auroc','auprc'))
    save_csv(s2,out/'table_exp02_cross_domain_CI.csv');save_csv(dm2,out/'exp02_domain_seed_means.csv')
    plot_bar(dm2,'holdout','test_fpr','method','Experiment 02: target FPR after transfer','Target FPR (%)',figdir/'fig02_cross_domain_FPR.png',100)
    # Detailed paired effects on shared domain x seed in exp02.
    pairings=[('LocalMLP','FedAvgMLP'),('FedAvgMLP','FedProxMLP'),('PooledMLP','FedAvgMLP'),('PooledLightGBM','FedAvgMLP')]
    paired=[]
    for ref,cand in pairings:
        for metric in ['attack_recall','test_fpr','auroc']:
            stat,pv,domain=paired_cluster(e2,ref,cand,metric=metric)
            paired.append(stat)
    save_csv(pd.DataFrame(paired),out/'table_exp02_paired_domain_effects.csv')
    # Exp03 source thresholds only; posthoc oracle EXCLUDED from deployable summary.
    e3=tabs[3]['metrics'];e3=e3[(e3.feature_variant=='all') & np.isclose(e3.target_fpr,.01)].copy()
    s3,dm3=domain_summary(e3,group=('threshold_strategy','method'),metrics=('attack_recall','test_fpr','auroc'))
    save_csv(s3,out/'table_exp03_threshold_strategies_DESCRIPTIVE.csv')
    save_csv(dm3,out/'exp03_domain_threshold_means.csv')
    cal=tabs[3]['calibration'];cal=cal[cal.feature_variant=='all'].copy()
    sc=cal.groupby(['method','location'],as_index=False)[['brier','ece']].mean();save_csv(sc,out/'table_exp03_calibration.csv')
    plot_cal=sc[sc.location.isin(['source_raw','target_raw_POSTHOC'])].copy()
    plot_cal['stage']=plot_cal.location.map({'source_raw':'Source validation','target_raw_POSTHOC':'Held-out target'})
    fig,ax=plt.subplots(figsize=(9,4.5))
    for method,g in plot_cal.groupby('method'):
        g=g.set_index('stage').reindex(['Source validation','Held-out target']).reset_index()
        ax.plot(g.stage,g.ece,marker='o',label=method)
    ax.set(title='Experiment 03: calibration shift',ylabel='Expected calibration error (ECE)',xlabel='');ax.grid(alpha=.25);ax.legend(fontsize=8)
    saveplot(fig,figdir/'fig03_calibration_shift.png')
    # Exp04: only source-only deployable points; hypothetical prevalence.
    e4=tabs[4]['prevalence'];e4=e4[(e4.feature_variant=='all') & np.isclose(e4.target_validation_fpr,.01) & (e4.threshold_strategy=='pooled_source_quantile')].copy()
    if 'decision_uses_target_labels' in e4:
        e4=e4[e4.decision_uses_target_labels.astype(str).str.lower().isin(['false','0'])]
    cols=['expected_precision','expected_alerts_per_100k','expected_tp_per_100k','expected_fp_per_100k','target_attack_recall','target_test_fpr']
    s4=e4.groupby(['method','prevalence_scenario'],as_index=False)[cols].mean()
    save_csv(s4,out/'table_exp04_hypothetical_prevalence.csv')
    fig,ax=plt.subplots(figsize=(9,4.5))
    for method,g in s4.groupby('method'):
        g=g.sort_values('prevalence_scenario');ax.plot(g.prevalence_scenario*100,g.expected_precision*100,marker='o',label=method)
    ax.set(title='Experiment 04: expected precision under prior-shift assumptions',xlabel='Hypothetical attack prevalence (%)',ylabel='Expected precision (%)');ax.grid(alpha=.25);ax.legend(fontsize=8)
    saveplot(fig,figdir/'fig04_hypothetical_prevalence.png')
    bu=tabs[4]['budget'];bu=bu[(bu.feature_variant=='all') & np.isclose(bu.prevalence_scenario,.01)].copy()
    bud=bu.groupby(['method','source_alert_budget_per_100k'],as_index=False).agg(target_alerts_mean=('expected_alerts_per_100k','mean'),target_recall_mean=('target_attack_recall','mean'),budget_violation_rate=('expected_alerts_per_100k',lambda x:np.nan)).reset_index(drop=True)
    violation=bu.assign(violated=bu.expected_alerts_per_100k>bu.source_alert_budget_per_100k).groupby(['method','source_alert_budget_per_100k']).violated.mean().reset_index()
    bud=bud.drop(columns=['budget_violation_rate']).merge(violation,on=['method','source_alert_budget_per_100k']).rename(columns={'violated':'budget_violation_rate'})
    save_csv(bud,out/'table_exp04_budget_transfer.csv')
    # Exp05: native, quantity, label, partial participation.
    e5=tabs[5]['result'];e5=e5[np.isclose(e5.requested_val_fpr,.01)].copy().rename(columns={'heldout':'holdout'})
    s5,dm5=domain_summary(e5,group=('allocation_mode','participating_clients','method'),metrics=('target_recall','actual_test_fpr','model_auroc'))
    save_csv(s5,out/'table_exp05_participation_heterogeneity_CI.csv')
    # Exp06: controlled interventions with within-domain+seed paired deltas.
    e6=tabs[6]['result'];e6=e6[np.isclose(e6.requested_val_fpr,.01)].copy().rename(columns={'heldout':'holdout'})
    s6,dm6=domain_summary(e6,group=('allocation_mode','method'),metrics=('target_recall','actual_test_fpr','model_auroc'))
    save_csv(s6,out/'table_exp06_controlled_heterogeneity_CI.csv')
    diffs=[];perdom=[]
    for method in sorted(e6.method.unique()):
        for cond in sorted(e6.allocation_mode.unique()):
            if cond=='baseline':continue
            for metric in ['target_recall','actual_test_fpr','model_auroc']:
                subset=e6[e6.method==method].copy()
                pv=subset[subset.allocation_mode.isin(['baseline',cond])].pivot_table(index=['holdout','seed'],columns='allocation_mode',values=metric,aggfunc='mean').reset_index().dropna(subset=['baseline',cond])
                pv['delta']=pv[cond]-pv['baseline'];pdm=pv.groupby('holdout',as_index=False).delta.mean()
                mu,lo,hi,n=cluster_ci(pdm.delta)
                diffs.append({'method':method,'intervention':cond,'metric':metric,'paired_delta_mean':mu,'domain_CI95_low':lo,'domain_CI95_high':hi,'n_domains':n,'n_seed_pairs':len(pv)})
                for _,row in pdm.iterrows():perdom.append({'method':method,'intervention':cond,'metric':metric,'holdout':row.holdout,'domain_mean_paired_delta':row.delta})
    save_csv(pd.DataFrame(diffs),out/'table_exp06_domain_cluster_paired_deltas.csv')
    save_csv(pd.DataFrame(perdom),out/'exp06_per_domain_paired_deltas.csv')
    df6=s6[(s6.method=='FedAvgMLP')].copy();order=['baseline','quantity_mild','quantity_severe','label_mild','label_severe','domain_mild','domain_severe']
    df6['allocation_mode']=pd.Categorical(df6.allocation_mode,categories=order,ordered=True);df6=df6.sort_values('allocation_mode')
    fig,ax=plt.subplots(figsize=(10,5));y=np.arange(len(df6))
    ax.barh(y,df6.target_recall_mean*100)
    ax.set_yticks(y,df6.allocation_mode.astype(str));ax.invert_yaxis()
    ax.set(xlabel='Mean target recall (%)',title='Experiment 06: controlled heterogeneity — FedAvg')
    ax.grid(axis='x',alpha=.22);saveplot(fig,figdir/'fig05_controlled_heterogeneity.png')
    # Exp02: Pareto in (higher recall, lower FPR, lower comm). Communication apples-to-oranges:
    # pooled/local are NOT assigned zero-cost, because centralization and inference traffic excluded.
    comm=tabs[2]['comm'].groupby('method',as_index=False).communication_bytes.mean()
    pc=s2.merge(comm,on='method',how='left')
    pc['communication_cost_comparable'] = pc.method.isin(['FedAvgMLP','FedProxMLP'])
    pc['communication_measured_mib']=pc.communication_bytes/(1024**2)
    save_csv(pc,out/'table_exp02_pareto_with_cost_CAVEATS.csv')
    pareto=[]
    only=pc[pc.communication_cost_comparable].copy()
    for _,a in only.iterrows():
        dominated=False
        for _,b in only.iterrows():
            if a.method==b.method:continue
            if (b.attack_recall_mean>=a.attack_recall_mean and b.test_fpr_mean<=a.test_fpr_mean and b.communication_measured_mib<=a.communication_measured_mib and
              (b.attack_recall_mean>a.attack_recall_mean or b.test_fpr_mean<a.test_fpr_mean or b.communication_measured_mib<a.communication_measured_mib)):
                dominated=True
        pareto.append({'method':a.method,'nondominated_under_reported_metrics':not dominated})
    save_csv(pd.DataFrame(pareto),out/'table_exp02_federated_only_pareto.csv')
    fig,ax=plt.subplots(figsize=(8,5))
    for _,r in pc.iterrows():
        ax.scatter(r.test_fpr_mean*100,r.attack_recall_mean*100,s=90)
        ax.annotate(r.method,(r.test_fpr_mean*100,r.attack_recall_mean*100),xytext=(5,5),textcoords='offset points',fontsize=9)
    ax.set(title='Experiment 02: recall vs realized target FPR',xlabel='Target FPR (%) — lower is better',ylabel='Target recall (%) — higher is better');ax.grid(alpha=.2)
    saveplot(fig,figdir/'fig06_operational_tradeoff.png')
    # Explicit audit summary and reproducibility note.
    data={
        'status':'complete','model_runs':{'01':100,'02':100,'03':300,'04':0,'05':240,'06':280},
        'observed_operating_points':actual,
        'statistical_unit':'held-out domain (n=4) for cross-domain inference; seeds treated as repeated runs within domain',
        'confidence_intervals':'95% Student-t intervals across four domain-level means; descriptive, very imprecise with n=4; no p-values claimed',
        'oracle_policy':'Experiment 03 target-oracle thresholds appear in descriptive strategy table only, never deployed/pooled with source-only decisions',
        'phase04_policy':'hypothetical class-prior reweighting (fixed conditional rates), not observed traffic or additional independent experiments',
        'communication_policy':'Central and local communication costs omitted from comparable Pareto because raw centralization and model distribution unmeasured',
        'leakage_limitations':'No verified session/capture independence; duplicate audits were sampled',
        'experiment_comparability':'Phase 01 and Phase 02 differ in data and evaluation setting; avoid interpreting cross-phase mean differences as controlled domain degradation',
        'complete_from_archives':True
    }
    (out/'analysis_manifest.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    lines=['# FL-IDS Benchmark — Experiment 07: Statistical Reliability and Synthesis','',
           '## Provenance','All outputs are derived from original Experiment 01–06 archives. No model was retrained. SHA256 checksums and full source run statuses are provided.',
           '', '## Key methodological rules',
           '- A held-out dataset/domain, not a training seed, is the independent cross-domain cluster (only four domains).',
           '- Seed results are aggregated within domain before computing 95% descriptive t intervals across four domain averages; these intervals are unstable and not population-level guarantees.',
           '- Experiment 03 target oracle is post hoc and never treated as deployable.',
           '- Experiment 04 describes hypothetical prevalence scenarios rather than new observed network conditions.',
           '- Centralization/data movement and local ensemble model distribution costs were not measured, so no fair 3D Pareto across central/local/federated is asserted.',
           '- Data leakage cannot be ruled out at session/capture level; overlap audits are sample-limited.',
           '', '## Key tables',
           '1. `table_exp01_open_set_domain_CI.csv` and `table_exp02_cross_domain_CI.csv`: protocol-specific means and domain intervals.',
           '2. `table_exp02_paired_domain_effects.csv`: paired between-model differences on common domain+seed.',
           '3. `table_exp03_threshold_strategies_DESCRIPTIVE.csv`: explicitly includes nondeployable posthoc oracle for diagnosis.',
           '4. `table_exp03_calibration.csv`: source/target calibration.',
           '5. `table_exp04_hypothetical_prevalence.csv` and `table_exp04_budget_transfer.csv`: prevalence assumptions and budget violations.',
           '6. `table_exp05_participation_heterogeneity_CI.csv`: partial participation.',
           '7. `table_exp06_controlled_heterogeneity_CI.csv` and `table_exp06_domain_cluster_paired_deltas.csv`: isolate heterogeneity effects.',
           '', '## Recommendations for the manuscript',
           '- Separate claims about in-dataset unseen attack detection from cross-domain transfer.',
           '- Present realized target FPR alongside target recall and source validation FPR.',
           '- Do not treat 5 seeds x 4 domains as 20 independent environments.',
           '- Include exact fixed split manifest hashes and make known limitations prominent.',
           '- Report uncertainty without overstating inferential confidence for n=4 independent domains.',
           '']
    (out/'README_RESEARCH_SYNTHESIS.md').write_text('\n'.join(lines),encoding='utf-8')
    # Archive only derived data (no raw predictions or duplicated source archives).
    archive=out.parent/'FLIDS_Experiment07_Statistical_Synthesis_Results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,arcname='FLIDS_Experiment07_Results/'+str(p.relative_to(out)))
    print('FINISHED',archive,'files',len(list(out.rglob('*'))),flush=True)
    return archive

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-root',default='/kaggle/input');p.add_argument('--output-root',default='/kaggle/working');a=p.parse_args()
    run(a.input_root,Path(a.output_root)/'FLIDS_Experiment07_Results')
