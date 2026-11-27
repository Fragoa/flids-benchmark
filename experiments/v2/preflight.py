"""Validate Kaggle datasets before any model fit. No data downloading or inferred harmonization."""
from pathlib import Path
import hashlib, io, json, zipfile, csv, re
import numpy as np
import pandas as pd

MEMBERS=('reproducibility/flids_exp02_cache.npz','reproducibility/flids_exp02_fingerprints.npy','reproducibility/data_preparation/dataset_audit.json')
DOMAINS={'NF-UNSW-NB15','NF-ToN-IoT','NF-BoT-IoT','NF-CSE-CIC-IDS2018'}
# New, predeclared protocol-field subset, not the historical Safe-30 list.
EDGE_PROTOCOL25=['arp.opcode','arp.hw.size','icmp.unused','http.content_length','http.response','tcp.connection.fin','tcp.connection.rst','tcp.connection.syn','tcp.connection.synack','tcp.flags.ack','tcp.len','udp.time_delta','dns.qry.name.len','dns.qry.qu','dns.qry.type','dns.retransmission','dns.retransmit_request','dns.retransmit_request_in','mqtt.conflag.cleansess','mqtt.len','mqtt.msgtype','mqtt.proto_len','mqtt.topic_len','mqtt.ver','mbtcp.len']

def pick(paths, label):
    paths=sorted(set(Path(p).resolve() for p in paths))
    if len(paths)>1: raise ValueError(f'Multiple {label} matches; set explicit path: {paths}')
    return paths[0] if paths else None

def read_exp09(root, explicit=''):
    root=Path(root)
    if explicit:
        p=Path(explicit)
        if not p.exists(): raise FileNotFoundError(p)
        candidates=[p]
    else:
        candidates=list(root.rglob('FLIDS_Experiment09_Results.zip'))
        for p in root.rglob('flids_exp02_cache.npz'):
            if p.parent.name=='reproducibility' and all((p.parent.parent/n).is_file() for n in MEMBERS):candidates.append(p.parent.parent)
    found=[]
    for p in candidates:
        if p.is_dir():
            data={n:(p/n).read_bytes() for n in MEMBERS}
        else:
            with zipfile.ZipFile(p) as z:data={n:z.read(n) for n in MEMBERS}
        h=hashlib.sha256()
        for n in MEMBERS:h.update(n.encode());h.update(hashlib.sha256(data[n]).digest())
        found.append((p,data,h.hexdigest()))
    if not found:return None
    if len({v[2] for v in found})!=1:raise ValueError('Different Exp09 inputs found; set EXP09_ZIP explicitly.')
    return found[0]

def validate_nf(root,work,features,explicit=''):
    found=read_exp09(root,explicit)
    if found is None:raise FileNotFoundError('Attach FLIDS_Experiment09_Results as ZIP or extracted folders (including reproducibility/).')
    p,data,digest=found
    audit=json.loads(data[MEMBERS[2]])
    if audit.get('features')!=features:raise ValueError('NF feature order differs from Experiment 02 engine.')
    if audit.get('source_rows_scanned')!=75987976:raise ValueError('Unexpected original source row count; not the historical full Exp09 input.')
    with np.load(io.BytesIO(data[MEMBERS[0]]),allow_pickle=False) as a:
        assert {'x','y','d','attack'}<=set(a.files)
        assert a['x'].shape==(96000,41) and all(a[k].shape==(96000,) for k in ('y','d','attack'))
        assert np.isfinite(a['x']).all() and set(a['y'])=={0,1} and set(a['d'])==DOMAINS
        for d in DOMAINS:
            for y in (0,1):assert int(((a['d']==d)&(a['y']==y)).sum())==12000
    fp=np.load(io.BytesIO(data[MEMBERS[1]]),allow_pickle=False);assert fp.shape[0]==96000
    # Build a canonical minimal ZIP outside the work folder, so the existing engine can consume either input layout.
    target=Path(work).parent/'FLIDS_Exp09_Canonical_Input.zip';tmp=target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:
        for n in MEMBERS:z.writestr(zipfile.ZipInfo(n,date_time=(2026,1,1,0,0,0)),data[n])
    tmp.replace(target)
    return str(target),digest,{'state':'ready','source':str(p),'samples':96000,'features':41,'domains':sorted(DOMAINS),'source_rows':75987976,'content_sha256':digest}

def header(path):
    return list(pd.read_csv(path,nrows=0).columns)

def check_csv(path,columns,numeric):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(p)
    missing=set(columns)-set(header(p))
    if missing:raise ValueError(f'{p.name} missing columns: {sorted(missing)}')
    df=pd.read_csv(p,usecols=columns,nrows=2048,low_memory=False)
    if df.empty:raise ValueError(f'{p.name} empty')
    for col in numeric:pd.to_numeric(df[col],errors='raise')
    return str(p)

def resolve_manifest(root,manifest_name,entries):
    p=pick(Path(root).rglob(manifest_name),manifest_name)
    if p is None:return entries
    if entries:raise ValueError(f'Both manual and {manifest_name} supplied; use just one.')
    result=json.loads(p.read_text())
    if not isinstance(result,list):raise ValueError('External manifest must contain a JSON list.')
    for e in result:
        q=Path(e['path']);e['path']=str(q if q.is_absolute() else p.parent/q)
    return result

def run_preflight(root,work,features,*,exp09='',edge_csv='',edge_features=None,edge_features_json='',allow_new_edge=True,external_inputs=None,temporal_input=None,require_all=False):
    root=Path(root);work=Path(work);report={};resolved={};external_inputs=external_inputs or []
    try:
        path,dig,info=validate_nf(root,work,features,exp09);report['NF_UQ']=info;resolved.update(exp09_zip=path,nf_content_sha=dig)
    except Exception as e:report['NF_UQ']={'state':'blocked','reason':str(e)}
    try:
        p=Path(edge_csv) if edge_csv else pick(root.rglob('DNN-EdgeIIoT-dataset.csv'),'DNN Edge CSV')
        if p is None:raise FileNotFoundError('Add official Edge-IIoTset; DNN-EdgeIIoT-dataset.csv required.')
        js=Path(edge_features_json) if edge_features_json else pick(root.rglob('edge_safe30_features.json'),'Safe-30 JSON')
        if edge_features and js:raise ValueError('Set either EDGE_FEATURES or Safe-30 JSON, not both.')
        fs=list(edge_features or (json.loads(js.read_text()) if js else []))
        if fs:
            if len(fs)!=30 or len(set(fs))!=30:raise ValueError('Original Safe-30 requires exactly 30 unique names.')
            schema='Safe-30'
        elif allow_new_edge:fs=EDGE_PROTOCOL25;schema='Edge-Protocol25-new-not-Safe30'
        else:raise ValueError('Missing audited original edge_safe30_features.json.')
        check_csv(p,fs+['Attack_label','Attack_type'],fs+['Attack_label'])
        report['Edge']={'state':'ready_schema_check','source':str(p),'schema':schema,'features':fs,'note':'Full numeric and 14-family validation during preparation; new subset is not historical Safe-30.'};resolved.update(edge_csv=str(p),edge_features=fs,edge_schema=schema)
    except Exception as e:report['Edge']={'state':'blocked','reason':str(e)}
    try:
        entries=resolve_manifest(root,'FLIDS_external_inputs.json',external_inputs)
        if not entries:raise ValueError('Add verified harmonized independent external domain CSV(s) and FLIDS_external_inputs.json; raw CICIoT2023/X-IIoTID cannot be used directly.')
        names=[]
        for e in entries:
            if not e.get('schema_verified') or not e.get('provenance'):raise ValueError('Each external domain requires schema_verified=true and provenance of equivalent NetFlow feature extraction.')
            name=e['domain'];names.append(name)
            if name in DOMAINS or 'v3' in name.lower() and any(t in name.lower() for t in ['unsw','ton','bot','cicids2018']):raise ValueError('NF-v3 versions of original domains are not independent external domains.')
            check_csv(e['path'],features+[e.get('label','Label'),e.get('family','Attack')],features+[e.get('label','Label')])
        assert len(set(names))==len(names)
        report['External']={'state':'ready_schema_check','domains':names};resolved['external_inputs']=entries
    except Exception as e:report['External']={'state':'blocked','reason':str(e)}
    try:
        e=dict(temporal_input) if temporal_input else None
        if e is None:
            matches=list(root.rglob('NF-UNSW-NB15-v3.csv'))
            if not matches:
                for candidate in root.rglob('*.csv'):
                    if 'nfunswnb15v3' in re.sub(r'[^a-z0-9]','',str(candidate.relative_to(root)).lower()):
                        try:
                            if set(features+['FLOW_START_MILLISECONDS','Label','Attack'])<=set(header(candidate)):matches.append(candidate)
                        except Exception:pass
            p=pick(matches,'official NF-UNSW-NB15-v3 CSV')
            if p is None:raise FileNotFoundError('Add official NF-UNSW-NB15-v3.csv (timestamped) or configure TEMPORAL_INPUT.')
            e={'path':str(p),'domain':'NF-UNSW-NB15-v3-temporal','time_col':'FLOW_START_MILLISECONDS','time_unit':'ms','label':'Label','family':'Attack','group_col':None,'cap_per_class_block':12000}
        cols=features+[e.get('label','Label'),e.get('family','Attack'),e['time_col']]+([e['group_col']] if e.get('group_col') else [])
        check_csv(e['path'],cols,features+[e.get('label','Label')]+([e['time_col']] if e.get('time_unit') else []))
        report['Temporal']={'state':'ready_schema_check','source':e['path'],'time_col':e['time_col'],'time_unit':e.get('time_unit'),'capture_group':e.get('group_col'),'note':'Chronological block validity checked on full data later; not an independent new domain.'};resolved['temporal_input']=e
    except Exception as e:report['Temporal']={'state':'blocked','reason':str(e)}
    report['Stages']={'10':'NF_UQ','11':'NF_UQ','12':'NF_UQ','13':['NF_UQ (leave-two)','Edge','External','Temporal']}
    report['all_inputs_ready']=all(report[k]['state'].startswith('ready') for k in ('NF_UQ','Edge','External','Temporal'))
    (work/'input_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if report['NF_UQ']['state']!='ready':raise RuntimeError('NF-UQ input invalid. See input_preflight.json.')
    if require_all and not report['all_inputs_ready']:raise RuntimeError('Full run blocked by missing/invalid inputs; see input_preflight.json. No model fits started. Set REQUIRE_ALL_INPUTS=False only for partial execution.')
    return resolved,report
