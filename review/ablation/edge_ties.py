"""Replay held-out DDoS-UDP predictions against archived counts and full-feature signatures."""
from pathlib import Path
import sys, zipfile
import numpy as np
import pandas as pd

def audit(archive, output):
    with zipfile.ZipFile(archive) as z:
        def read(name):
            return pd.read_csv(z.open('FLIDS_Experiment01_Results/'+name+'.csv'),float_precision='round_trip')
        predictions=read('predictions')
        manifest=read('split_manifest')
        results=read('experiment_results')
    family=next(h for h in predictions.heldout_family.unique() if h.lower().replace('_','-')=='ddos-udp')
    subset=predictions.query('heldout_family==@family')
    test=manifest.query('heldout_family==@family and split=="test"')
    matched=subset.merge(test,on=['row_id','seed','heldout_family'],validate='many_to_one',how='left',suffixes=('','_manifest'))
    assert matched.full_signature.notna().all() and (matched.true_label==matched.binary_label).all()
    rows=[]; detected_ids={}
    for (seed,method),g in matched.groupby(['seed','method']):
        ref=results.query('heldout_family==@family and seed==@seed and method==@method and target_fpr==0.01').iloc[0]
        scores=g.attack_score.to_numpy(); pos=g.true_label.to_numpy()==1
        # Round-trip precision and the archived TP/FP determine whether boundary ties matter.
        masks=[m for m in (scores>=ref.threshold,scores>ref.threshold) if m[pos].sum()==ref.tp and m[~pos].sum()==ref.fp]
        if not masks: raise ValueError(f'Cannot reproduce counts: {seed} {method}')
        det=g.loc[pos & masks[0]]; attacks=g.loc[pos]
        detected_ids[seed,method]=set(det.row_id)
        rows.append(dict(seed=seed,method=method,attack_rows=len(attacks),detected=len(det),detected_unique_signatures=det.full_signature.nunique(),all_unique_signatures=attacks.full_signature.nunique(),largest_detected_signature=int(det.full_signature.value_counts().max()) if len(det) else 0,recall=ref.unknown_recall))
    for row in rows:
        key=row['seed'],row['method']; ref=detected_ids[row['seed'],'FedAvgMLP']; current=detected_ids[key]
        row['same_detected_rows_as_FedAvg']=current==ref
        row['overlap_with_FedAvg']=len(current & ref)
    frame=pd.DataFrame(rows);Path(output).parent.mkdir(parents=True,exist_ok=True);frame.to_csv(output,index=False)
if __name__=='__main__':audit(sys.argv[1],sys.argv[2])
