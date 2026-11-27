"""FL-IDS Benchmark | NF-UQ-NIDS-v2 dataset and split validity audit.

No model training. All source records are streamed using Python's csv module.
Full-stream counts are exact for the readable labeled rows; fingerprint checks
are performed on a stratified reservoir and MUST NOT be described as exhaustive.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

DOMAINS = ('NF-UNSW-NB15','NF-ToN-IoT','NF-BoT-IoT','NF-CSE-CIC-IDS2018')
FEATURES = ('L4_SRC_PORT','L4_DST_PORT','PROTOCOL','L7_PROTO','IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS',
'FLOW_DURATION_MILLISECONDS','TCP_FLAGS','CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS','DURATION_IN','DURATION_OUT',
'MIN_TTL','MAX_TTL','LONGEST_FLOW_PKT','SHORTEST_FLOW_PKT','MIN_IP_PKT_LEN','MAX_IP_PKT_LEN',
'SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_SECOND_BYTES','RETRANSMITTED_IN_BYTES','RETRANSMITTED_IN_PKTS',
'RETRANSMITTED_OUT_BYTES','RETRANSMITTED_OUT_PKTS','SRC_TO_DST_AVG_THROUGHPUT','DST_TO_SRC_AVG_THROUGHPUT',
'NUM_PKTS_UP_TO_128_BYTES','NUM_PKTS_128_TO_256_BYTES','NUM_PKTS_256_TO_512_BYTES',
'NUM_PKTS_512_TO_1024_BYTES','NUM_PKTS_1024_TO_1514_BYTES','TCP_WIN_MAX_IN','TCP_WIN_MAX_OUT',
'ICMP_TYPE','ICMP_IPV4_TYPE','DNS_QUERY_ID','DNS_QUERY_TYPE','DNS_TTL_ANSWER','FTP_COMMAND_RET_CODE')
ROOT = Path(os.environ.get('FLIDS_AUDIT_ROOT','/kaggle/working' if Path('/kaggle/working').exists() else '/mnt/data'))
OUT = ROOT / 'FLIDS_Dataset_Split_Validity_Audit'
SAMPLE_CAP = int(os.environ.get('FLIDS_AUDIT_PER_STRATUM','5000'))
SCAN_LIMIT = int(os.environ.get('FLIDS_AUDIT_SCAN_LIMIT','0'))
RANDOM_SEED = 20261124


def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def write_csv(name, rows, columns):
    with (OUT/name).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def norm(s):
    return ''.join(c for c in str(s).lower() if c.isalnum())


def domain_name(s):
    a=norm(s)
    if 'unsw' in a: return DOMAINS[0]
    if 'ton' in a and 'iot' in a: return DOMAINS[1]
    if 'bot' in a and 'iot' in a: return DOMAINS[2]
    if 'cse' in a or 'cicids2018' in a: return DOMAINS[3]
    return None


def header_of(p):
    with p.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
        return next(csv.reader(f))


def discover():
    roots=[Path('/kaggle/input'),ROOT/'input']
    if os.environ.get('NF_UQ_DATA'): roots.insert(0,Path(os.environ['NF_UQ_DATA']))
    reports=[]; candidates=[]
    for root in roots:
        if not root.exists():continue
        paths=[root] if root.is_file() else root.rglob('*.csv')
        for path in paths:
            try:
                head=header_of(path)
                present=set(head)
                ok=set(FEATURES).issubset(present) and ('Label' in present or 'label' in present) and ('Dataset' in present or domain_name(path.name))
                reports.append({'path':str(path),'eligible':bool(ok),'columns':len(head),'missing_features':sorted(set(FEATURES)-present)[:12]})
                if ok:candidates.append((path,head))
            except Exception as e:reports.append({'path':str(path),'eligible':False,'error':str(e)[:200]})
    dump('input_discovery.json',{'inspected_files':reports,'eligible_files':[str(x[0]) for x in candidates]})
    combined=[x for x in candidates if norm(x[0].name).startswith('nfuqnidsv2')]
    if combined:return [combined[0]]
    selected=[];added=set()
    for p,h in sorted(candidates,key=lambda x:str(x[0])):
        d=domain_name(p.name)
        if d and d not in added:selected.append((p,h));added.add(d)
    if len(selected)==4:return selected
    for x in candidates:
        if 'Dataset' in x[1]:return [x]
    return selected


def simulated_input():
    import numpy as np
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'nf_uq_Software_Smoke_Only.csv'
    rng=np.random.default_rng(101)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow([*FEATURES,'Label','Attack','Dataset'])
        for di,d in enumerate(DOMAINS):
            for y in [0,1]:
                for j in range(110):
                    vals=rng.normal(12+di,2,len(FEATURES))
                    vals[0]=1000+di*500+y*20
                    vals[1]=80+di*17+y*10
                    vals[4]=10+di+y*3+rng.normal()
                    w.writerow([f'{v:.6f}' for v in vals]+[y,'attack' if y else 'benign',d+'-v2'])
        # A cross-domain numerical collision with contradictory label.
        duplicated=['0']*len(FEATURES)
        w.writerow(duplicated+[0,'benign',DOMAINS[0]])
        w.writerow(duplicated+[1,'attack',DOMAINS[1]])
    return path


def hash_record(raw_features):
    # Preserve the feature representation used by Experiment 02: float32.
    # This is a row-feature proxy, not a session identifier.
    import numpy as np
    v=[];bad=0;missing=0
    for s in raw_features:
        if not s.strip(): missing+=1;v.append(0.0);continue
        try:
            x=float(s)
            if not math.isfinite(x):bad+=1;x=0.0
            if x > 3.4028234663852886e38 or x < -3.4028234663852886e38:
                bad+=1;x=0.0
            v.append(x)
        except (ValueError,OverflowError):bad+=1;v.append(0.0)
    arr=np.array(v,dtype=np.float32)
    # Collapse +0 / -0 for numerical equivalence.
    arr[arr==0]=0
    h=hashlib.blake2b(arr.tobytes(),digest_size=16).hexdigest()
    return h, arr, missing, bad


def scan(found):
    rng=random.Random(RANDOM_SEED)
    seen=Counter(); malformed=Counter(); file_report=[]; sample=defaultdict(list)
    raw_rows=0;progress=2_000_000
    for path,head in found:
        cols={c:i for i,c in enumerate(head)}
        labelidx=cols.get('Label',cols.get('label'))
        domidx=cols.get('Dataset')
        attackidx=cols.get('Attack')
        fixed=domain_name(path.name)
        file_ok=0;file_bad=0
        with path.open('r',encoding='utf-8-sig',errors='replace',newline='',buffering=1024*1024) as f:
            r=csv.reader(f)
            next(r)
            for row in r:
                raw_rows+=1
                if SCAN_LIMIT and raw_rows>SCAN_LIMIT:break
                if len(row)!=len(head):
                    malformed['column_count_mismatch']+=1;file_bad+=1;continue
                d=domain_name(row[domidx]) if domidx is not None else fixed
                if d not in DOMAINS:
                    malformed['unrecognized_domain']+=1;file_bad+=1;continue
                rawlabel=norm(row[labelidx]);y=0 if rawlabel in ('0','normal','benign') else 1 if rawlabel in ('1','attack','malicious') else None
                if y is None:
                    malformed['invalid_binary_label']+=1;file_bad+=1;continue
                key=(d,y);seen[key]+=1
                # Uniform reservoir within every domain/label group; sampling is not first-N.
                bucket=sample[key]
                entry=None
                if len(bucket)<SAMPLE_CAP:
                    entry=([row[cols[c]] for c in FEATURES],row[attackidx] if attackidx is not None else '',str(path),raw_rows)
                    bucket.append(entry)
                else:
                    j=rng.randrange(seen[key])
                    if j<SAMPLE_CAP:
                        entry=([row[cols[c]] for c in FEATURES],row[attackidx] if attackidx is not None else '',str(path),raw_rows)
                        bucket[j]=entry
                file_ok+=1
                if raw_rows>=progress:
                    print('STREAM_PROGRESS',raw_rows,'sampled',sum(map(len,sample.values())),flush=True)
                    progress+=2_000_000
        file_report.append({'source':str(path),'valid_rows':file_ok,'invalid_rows':file_bad})
        print('SOURCE_SCANNED',path.name,'valid',file_ok,'invalid',file_bad,flush=True)
        if SCAN_LIMIT and raw_rows>SCAN_LIMIT:break
    return seen,malformed,file_report,sample,raw_rows


def sample_audit(sample):
    import numpy as np
    profile_rows=[];records=[];dup_groups=defaultdict(list)
    feature_count={f:Counter() for f in FEATURES}
    domain_feat_profile=defaultdict(lambda: defaultdict(list))
    for (domain,label),bucket in sorted(sample.items()):
        for raw,attack,file,rowno in bucket:
            h,features,missing,bad=hash_record(raw)
            records.append({'domain':domain,'label':label,'attack':attack,'source_file':file,'source_row':rowno,
                            'fingerprint':h,'missing_numeric_fields':missing,'bad_numeric_fields':bad})
            dup_groups[h].append(len(records)-1)
            for j,feat in enumerate(FEATURES):
                s=raw[j]
                if not s.strip():feature_count[feat]['missing']+=1
                try:
                    v=float(s)
                    if not math.isfinite(v):feature_count[feat]['nonfinite']+=1
                except (ValueError,OverflowError):feature_count[feat]['unparseable']+=1
                # Keep small per-domain numerical samples for domain shift and label purity indicators.
                if j in (0,1,2,4,5,6,7,14,15,16,19,25,26):
                    domain_feat_profile[(domain,label)][feat].append(float(features[j]))
    duplicate_rows=0;cross_groups=0;conflict_groups=0;within_groups=0;cross_conflicts=0
    groups=[]
    for h,ids in dup_groups.items():
        if len(ids)==1:continue
        ds=sorted({records[i]['domain'] for i in ids})
        ys=sorted({records[i]['label'] for i in ids})
        duplicate_rows+=len(ids)-1
        cross=len(ds)>1; conflict=len(ys)>1
        cross_groups+=int(cross);conflict_groups+=int(conflict)
        within_groups+=int(not cross)
        cross_conflicts+=int(cross and conflict)
        groups.append({'fingerprint':h,'sampled_rows':len(ids),'domains':';'.join(ds),'binary_labels':';'.join(map(str,ys)),
                       'cross_domain':int(cross),'label_conflict':int(conflict)})
    write_csv('sampled_duplicate_groups.csv',groups,['fingerprint','sampled_rows','domains','binary_labels','cross_domain','label_conflict'])
    write_csv('sampled_rows_manifest.csv',records,['domain','label','attack','source_file','source_row','fingerprint','missing_numeric_fields','bad_numeric_fields'])
    total=len(records)
    features=[]
    for f in FEATURES:
        c=feature_count[f]
        features.append({'feature':f,'missing_count':c['missing'],'nonfinite_count':c['nonfinite'],
                         'unparseable_count':c['unparseable'],'sampled_n':total,'missing_rate':c['missing']/total if total else 0})
    write_csv('feature_quality_sample.csv',features,['feature','sampled_n','missing_count','nonfinite_count','unparseable_count','missing_rate'])
    feature_shift=[]
    # KS calculated only on the reservoir. It is descriptive, not causal.
    try:
        from scipy.stats import ks_2samp
        for feat in (0,1,2,4,5,6,7,14,15,16,19,25,26):
            name=FEATURES[feat]
            for target in DOMAINS:
                tar=[];src=[]
                for (d,y),by_feature in domain_feat_profile.items():
                    if d==target:tar.extend(by_feature.get(name,[]))
                    else:src.extend(by_feature.get(name,[]))
                if len(tar)>=10 and len(src)>=10:
                    ks=float(ks_2samp(src,tar,method='asymp').statistic)
                    feature_shift.append({'feature':name,'target_domain':target,'sampled_ks':ks,
                                          'source_sample_n':len(src),'target_sample_n':len(tar)})
    except ImportError:
        pass
    write_csv('sampled_feature_shift_ks.csv',feature_shift,
              ['feature','target_domain','sampled_ks','source_sample_n','target_sample_n'])
    # Row overlap at the sample level, independent of exact source/validation split choices.
    overlap=[]
    for target in DOMAINS:
        target_fp={x['fingerprint'] for x in records if x['domain']==target}
        source_fp={x['fingerprint'] for x in records if x['domain']!=target}
        hit=target_fp&source_fp
        target_rows=sum(x['domain']==target and x['fingerprint'] in hit for x in records)
        overlap.append({'heldout_domain':target,'sampled_target_rows':sum(x['domain']==target for x in records),
                        'overlap_fingerprint_groups':len(hit),'sampled_target_rows_matching_sources':target_rows,
                        'scope':'stratified reservoir only; zero DOES NOT prove no full-data overlap'})
    write_csv('leave_one_domain_overlap_sample.csv',overlap,
              ['heldout_domain','sampled_target_rows','overlap_fingerprint_groups','sampled_target_rows_matching_sources','scope'])
    return {'sampled_rows':total,'unique_feature_fingerprints':len(dup_groups),
            'sample_duplicate_extra_rows':duplicate_rows,'sample_cross_domain_duplicate_groups':cross_groups,
            'sample_within_domain_duplicate_groups':within_groups,'sample_binary_label_conflict_groups':conflict_groups,
            'sample_cross_domain_conflicting_label_groups':cross_conflicts,
            'scope':'STRATIFIED RESERVOIR ONLY. Counts are not full-data duplicate rates. Fingerprints are 41-feature float32 row proxies.',
            'per_stratum_sample_cap':SAMPLE_CAP},overlap


def compare_exp02_zip():
    candidates=[]
    env=os.environ.get('FLIDS_EXP02_ZIP')
    if env:candidates.append(Path(env))
    for r in [Path('/kaggle/input'),ROOT]:
        if r.exists():candidates.extend(r.rglob('FLIDS_Experiment02_Results.zip'))
    for p in candidates:
        if not p.is_file():continue
        try:
            with zipfile.ZipFile(p) as z:
                names=z.namelist()
                def member(x):return next((n for n in names if n.endswith('/'+x) or n==x),None)
                da=member('dataset_audit.json'); st=member('run_status.json');man=member('split_manifest.csv')
                if da and st:
                    d=json.loads(z.read(da));s=json.loads(z.read(st))
                    summary={'status':'EXPERIMENT02_ARCHIVE_FOUND','path':str(p),
                             'previous_dataset_audit':d,'previous_run_status':s,
                             'has_split_manifest':bool(man),'warning':'Historical split_manifest.csv stores only aggregate digests and counts; without per-row IDs it cannot independently verify all cross-split fingerprint overlaps.'}
                    dump('previous_experiment02_context.json',summary)
                    return summary
        except Exception as e:
            continue
    dump('previous_experiment02_context.json',{'status':'NO_EXPERIMENT02_ARCHIVE','note':'Optional prior results were not attached. Current audit is independent.'})
    return None


def verdicts(seen,errors,quality,overlap):
    sample=quality['sampled_rows']
    checks=[]
    def add(id,status,detail,action):checks.append({'check_id':id,'status':status,'finding':detail,'required_action':action})
    coverage={f'{d}|{y}':seen[(d,y)] for d in DOMAINS for y in (0,1)}
    add('SOURCE_COVERAGE','PASS' if all(coverage.values()) else 'FAIL',
        f'Four domains x two binary classes, full-stream observed counts: {coverage}',
        'Attach missing domain sources or correct labels before a new benchmark run.')
    add('SCHEMA','PASS','All selected files have the 41 fixed NetFlow-v2 input feature names. This does not prove semantic equivalence.',
        'Check NF-v2 extractor documentation, units, feature definitions, and source-specific preprocessing.')
    add('ROW_VALIDITY','WARNING' if errors else 'PASS',f'Invalid/malformed rows in full stream: {dict(errors)}',
        'Document exclusion rules and treat missing/nonfinite values in train-only preprocessing.')
    dup=quality['sample_duplicate_extra_rows'];cross=quality['sample_cross_domain_duplicate_groups'];conf=quality['sample_binary_label_conflict_groups']
    add('SAMPLED_DUPLICATES','WARNING' if dup else 'PASS',
        f'{dup} extra duplicate feature rows in a stratified sample of {sample}; this is NOT a full dataset duplicate estimate.',
        'Use stable row manifest and full-source duplicate audit if exact leakage quantification is required.')
    add('CROSS_DOMAIN_OVERLAP','WARNING' if cross else 'PASS',
        f'{cross} cross-domain identical feature-signature groups in the sample; no observed overlap does NOT prove independence.',
        'Exclude exact overlaps across source/target in a sensitivity analysis while preserving the primary fixed test.')
    add('LABEL_CONFLICT','WARNING' if conf else 'PASS',
        f'{conf} feature-identical groups with conflicting binary labels in the sample.',
        'Review label/feature provenance. Never silently remove ambiguous examples from only one test condition.')
    add('SPLIT_INDEPENDENCE','WARNING',
        'Experiment 02 split uses source-dataset holdout plus feature fingerprint proxies; session/capture/device independence is unverified.',
        'Acquire trusted capture/session identifiers or explicitly limit generalization claims to source-dataset holdouts.')
    add('SOURCE_TARGET_LEAKAGE','WARNING',
        'A sample overlap scan cannot certify zero duplicates across roughly 76 million original records.',
        'Use disk-backed exhaustive hashing if a strict zero-overlap claim is required; optionally pair with independent capture identifiers.')
    add('PREVALENCE','WARNING',
        'Experiment 02 capped each domain/class at 12,000 rows, creating 50% attack prevalence in its evaluation pool.',
        'Report original domain prevalence; do not use pooled Precision/AUPRC as deployment prevalence metrics.')
    add('FEATURE_SHORTCUTS','WARNING',
        'Ports, TTL and other flow metadata may correlate with source domain. Distribution shift alone is not proof of shortcut learning.',
        'Perform grouped ablations and check data collection protocol; never select features using target test labels.')
    add('LOW_FPR','WARNING',
        'Target FPR of 0.1% is sensitive to benign validation sample size and score ties.',
        'Report realized validation/test FPR, confidence intervals, and alert budgets, not target FPR alone.')
    write_csv('audit_gate.csv',checks,['check_id','status','finding','required_action'])
    fail=sum(c['status']=='FAIL' for c in checks)
    warn=sum(c['status']=='WARNING' for c in checks)
    status='FAIL' if fail else 'CONDITIONAL_PASS_WITH_WARNINGS' if warn else 'PASS'
    gate={'status':status,'pass_count':len(checks)-warn-fail,'warning_count':warn,'fail_count':fail,
          'recommendation':'Proceed to Experiment 05 ONLY as a source-dataset-held-out benchmark with explicit limitations; do not claim capture-level independence. Prioritize full-source overlap checks if using leakage-free in title.',
          'strict_evidence_note':'No model training in this notebook. No zero-leakage claim without verified full-data/group checks.'}
    dump('audit_gate.json',gate)
    return gate


def make_readme(seen,malformed,quality,gate,files):
    lines=['# FL-IDS Dataset & Split Validity Audit', '', '## Scope',
           'This is a **no-model-training** preflight audit of NF-UQ-NIDS-v2 for Experiment 05.',
           'Input records are streamed using Python csv.reader; counts are exact for readable rows.',
           'Fingerprint duplicate and feature-quality claims are **sample-level** only.',
           'The sample is uniform within each of four source-domain x two binary-label strata.',
           '', '## Gate',f"**{gate['status']}**; warnings={gate['warning_count']}, failures={gate['fail_count']}",
           '', '## Exact source counts', '| Source domain | Benign | Attack | Original attack prevalence |',
           '|---|---:|---:|---:|']
    for d in DOMAINS:
        b=seen[(d,0)];a=seen[(d,1)]
        lines.append(f'| {d} | {b:,} | {a:,} | {a/(a+b):.4%} |' if a+b else f'| {d} | 0 | 0 | N/A |')
    lines+=['','## Sample-only feature fingerprint checks',
            f"Sample rows: {quality['sampled_rows']:,}; extra duplicates: {quality['sample_duplicate_extra_rows']:,}; cross-domain groups: {quality['sample_cross_domain_duplicate_groups']:,}; conflicting labels: {quality['sample_binary_label_conflict_groups']:,}.",
            '**These figures must NOT be extrapolated as exact full-source duplicate counts.**',
            '', '## Caveats',
            '- Cross-source row overlap alone does not establish temporal, capture, device, or session independence.',
            '- Schema name equality does not establish identical network traffic semantics across sources.',
            '- This audit does not validate that the previous 96,000-row pooled data includes every original record.',
            '- Existing `split_manifest.csv` contains aggregate index digests, not reusable per-record split membership.',
            '- Dataset source held out from training is the correct claim; new organization or unseen time is not established.',
            '', '## Files',
            '- `full_source_domain_counts.csv`: exact full-stream domain and label counts.',
            '- `sampled_rows_manifest.csv`: representative per-stratum sampled row lineage and feature hashes.',
            '- `sampled_duplicate_groups.csv`: within/cross-source feature hash groups for sample.',
            '- `leave_one_domain_overlap_sample.csv`: overlap proxies for four source holdouts.',
            '- `feature_quality_sample.csv`, `sampled_feature_shift_ks.csv`: descriptive probes.',
            '- `audit_gate.csv`, `audit_gate.json`: methodological decisions.',
            '- `previous_experiment02_context.json`: optional historical archive reference.',
            '', '## Experiment 05 go/no-go',
            'Do not proceed if any domain or binary class is absent, or if feature schema differs.',
            'For `CONDITIONAL_PASS_WITH_WARNINGS`, allow a clearly caveated pilot only; resolve source/schema questions before confirmatory paper claims.',
            '']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def make_zip():
    zpath=ROOT/'FLIDS_Dataset_Split_Validity_Audit.zip'
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,arcname=str(Path('FLIDS_Dataset_Split_Validity_Audit')/p.relative_to(OUT)))
    return zpath


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    t=time.perf_counter()
    if os.environ.get('FLIDS_AUDIT_SMOKE')=='1':
        p=simulated_input();found=[(p,header_of(p))]
        dump('input_discovery.json',{'mode':'SYNTHETIC_SOFTWARE_SMOKE_ONLY','eligible_files':[str(p)]})
    else:found=discover()
    if not found:raise FileNotFoundError('NF_UQ_V2_NOT_FOUND: Add NF-UQ-NIDS-v2 (combined CSV) in Kaggle Input; inspect input_discovery.json. Optionally set NF_UQ_DATA to the exact CSV path.')
    dump('audit_configuration.json',{'input_sources':[str(p) for p,h in found],'scan_limit':SCAN_LIMIT,'sample_cap_per_domain_label':SAMPLE_CAP,
         'feature_names':FEATURES,'domains':DOMAINS,'random_seed':RANDOM_SEED,'full_scan':SCAN_LIMIT==0,
         'status_note':'All numerical fingerprint checks are sample-level; all parsed label/domain counters are full-stream when scan_limit=0.'})
    print('AUDIT_BEGIN',list(map(str,[p for p,h in found])),'stratum_cap',SAMPLE_CAP,'scan_limit',SCAN_LIMIT,flush=True)
    seen,malformed,files,sample,raw_rows=scan(found)
    counts=[]
    for d in DOMAINS:
        n0=seen[(d,0)];n1=seen[(d,1)]
        counts.append({'domain':d,'benign_rows':n0,'attack_rows':n1,'total_rows':n0+n1,
                       'attack_prevalence':n1/(n0+n1) if n0+n1 else None})
    write_csv('full_source_domain_counts.csv',counts,['domain','benign_rows','attack_rows','total_rows','attack_prevalence'])
    dump('stream_summary.json',{'raw_rows_seen':raw_rows,'valid_label_domain_rows':sum(seen.values()),'unusable_rows':dict(malformed),'source_files':files,
                               'full_source_scan':SCAN_LIMIT==0})
    print('SAMPLE_ANALYSIS_BEGIN',sum(map(len,sample.values())),flush=True)
    quality,overlap=sample_audit(sample)
    dump('sample_fingerprint_summary.json',quality)
    compare_exp02_zip()
    gate=verdicts(seen,malformed,quality,overlap)
    make_readme(seen,malformed,quality,gate,files)
    dump('run_status.json',{'completed':True,'status':gate['status'],'elapsed_seconds':round(time.perf_counter()-t,2),
                            'source_rows_parsed':sum(seen.values()),'sampled_rows':quality['sampled_rows'],
                            'software_smoke_only':os.environ.get('FLIDS_AUDIT_SMOKE')=='1'})
    archive=make_zip()
    print('AUDIT_COMPLETED',gate,'ARCHIVE',archive,'BYTES',archive.stat().st_size,flush=True)

if __name__=='__main__':main()
