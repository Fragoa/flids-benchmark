"""Matched tuned ablations. Run with the three archived engine modules alongside this file."""
import argparse,hashlib,io,json,os,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def run(archive,work,hours,limit=None):
    import master as m
    work.mkdir(exist_ok=True,parents=True)
    m.DEADLINE=time.monotonic()+hours*3600
    with zipfile.ZipFile(archive) as z:
        raw=z.read('nf_cache.npz')
        assert hashlib.sha256(raw).hexdigest()=='4952eb09930fbe35b5a2f9338b95023b197276d55cf0f3015c60e306aa51bff4'
        for file in ['master.py','engine02.py']:
            assert (Path(__file__).parent/file).read_bytes()==z.read(file)
        with np.load(io.BytesIO(raw)) as a:X,y,d,f=[a[k] for k in ['x','y','d','family']]
        opts=json.loads(z.read('config.json'));opts['device']='cuda' if m.torch.cuda.is_available() else 'cpu'
        m.torch.set_num_threads(1)
        baseline=pd.read_csv(z.open('summary/operating.csv')).query('track=="nf" and alpha==0.01 and scenario=="native" and participation==1 and status=="evaluated"')
        methods=['LogisticRegression','RandomForest','LightGBM',*m.FL,'PooledMLP','LocalMLP','GroupDRO','CORAL','DANN_Source']
        assert len(set(methods))==15 and len(baseline)==300
        native={(r.held,r.seed,r.method):r for r in baseline.itertuples()}
        complete=True;results=[]
        try:
            for variant,drop in [('no_ports',['L4_SRC_PORT','L4_DST_PORT']),('no_ports_ttl',['L4_SRC_PORT','L4_DST_PORT','MIN_TTL','MAX_TTL'])]:
                sub=work/variant;sub.mkdir(exist_ok=True);keep=[i for i,n in enumerate(m.original.FEATURES) if n not in drop];x=X[:,keep].copy()
                protocol={'driver':m.sha(__file__),'baseline':json.loads(z.read('identity.json')),'variant':variant,'features':[m.original.FEATURES[i] for i in keep]}
                lock=sub/'protocol.json'
                if lock.exists():assert json.loads(lock.read_text())==protocol
                elif (sub/'bundles').exists():raise ValueError('Unidentified partial checkpoint')
                m.save(lock,protocol)
                for h in sorted(set(d)):
                    for seed in m.SEEDS:
                        parts=m.split_outer(y,d,f,(h,),seed,'nf')
                        digests={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in zip(['fit','stop','cal','source_test','target'],parts)}
                        for method in methods:
                            base=native[h,seed,method];original=json.loads(z.read('bundles/'+base.bundle+'/done.json'))
                            assert original['identity']['split_digests']==digests
                            _,folder=m.bundle(sub,x,y,d,f,(h,),seed,'nf',method,opts,split_override=parts)
                            m.analyze(folder,opts,sub)
                            row=pd.read_csv(folder/'operating.csv').query('alpha==0.01 and status=="evaluated"').iloc[0]
                            result=dict(variant=variant,holdout=h,seed=seed,method=method)
                            for metric in ('target_fpr','recall','auroc','pauc01','oracle_recall'):
                                result[metric]=float(row[metric]);result[metric+'_minus_full']=result[metric]-getattr(base,metric)
                            results.append(result)
                            if limit and len(results)>=limit:raise m.PauseRun('software test limit')
        except m.PauseRun:complete=False
        finally:
            # Reconstructible summaries; score bundles and trial JSONs are authoritative checkpoints.
            if results:pd.DataFrame(results).to_csv(work/'paired_results.csv',index=False)
            m.save(work/'progress.json',dict(analyses=sum(1 for _ in work.glob('*/bundles/*/analysis_complete.json')),expected=600,all_pairs_reached=complete and len(results)==600))
            target=work.parent/'FLIDS_Review_Ablation.zip';tmp=target.with_suffix('.tmp')
            with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as output:
                for path in work.rglob('*'):
                    if path.is_file() and path.suffix!='.tmp':output.write(path,path.relative_to(work))
            tmp.replace(target)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('work',type=Path);parser.add_argument('--hours',type=float,default=11);parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(args.archive,args.work,args.hours,args.limit)
