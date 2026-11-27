from pathlib import Path
import json,hashlib,zipfile,base64,zlib,ast
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'experiments/v2';D=Path.home()/'Downloads'
nb=json.loads((ROOT/'notebooks/legacy/FLIDS_Packages10_13_Master_Kaggle_AutoRun_v2.ipynb').read_text());cells=nb['cells']
def get(i):return ''.join(cells[i]['source'])
def put(i,s):cells[i]['source']=s.splitlines(True)
H=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(D/'FLIDS_Packages10_13_Results.zip') as z:
 ident=json.loads(z.read('identity.json'));assert ident==json.loads((R/'identity.json').read_text())
 m=z.read('master.py').decode();p=z.read('preflight.py').decode();e=z.read('engine02.py').decode()
 assert H(m.encode())==ident['source_sha']
 files={n:H(z.read(n)) for n in sorted(z.namelist()) if n.startswith('bundles/') or n=='nf_cache.npz'}
 for n in files:
  if n.endswith('/done.json'):
   s=json.loads(z.read(n))['score_sha256'];assert files[n[:-9]+'scores.npz']==s
   a=n[:-9]+'analysis_complete.json'
   if a in files:assert json.loads(z.read(a))['scores_sha256']==s
 counts=[sum(('/trial_' in n and n.endswith('.json')) if k=='trial' else n.endswith('/'+k+'.json') for n in files) for k in ('trial','done','analysis_complete')];assert counts==[3600,817,658]
 cfg=json.loads(z.read('config.json'));assert all(cfg[k]==v for k,v in ident.items() if k in cfg)
print(counts,flush=True)
import sys
sys.path.insert(0,str(R.resolve()));import preflight
assert preflight.read_exp09(D,D/'FLIDS_Experiment09_Results.zip')[2]==ident['input_exp09_sha']
put(0,'# FLIDS Master V3 — Resume & Recovery\nAttach the Experiment 09 and Packages 10–13 results as ZIPs or extracted datasets. Attach optional CSVs when available. Run all cells. The recovery gate checks the historical files before training. To continue a later session, attach its V3 results archive. If several checkpoints are attached, select RESUME_ZIP.\n')
put(1,get(1).replace('flids_master_v2','flids_master_v3').replace('AUTO_RESUME = True','AUTO_RESUME = True').replace('STAGES = [10,11,12,13]','STAGES = [10,11,12,13]  # Completed NF bundles are reused.'))
p=p.replace("fs=EDGE_PROTOCOL25;schema='Edge-Protocol25-new-not-Safe30'","fs=[c for c in EDGE_PROTOCOL25 if not c.startswith('dns.')];schema='Edge-Numeric19-v3'")
p=p.replace("fs+['Attack_label'])","['Attack_label'])").replace("cols,features+[e.get('label','Label')]+","cols,[e.get('label','Label')]+")
m=m.replace('else 25)','else 19)').replace("ix=np.array(ix);x=df.loc[ix,features].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64);","""ix=np.array(ix)
 fields=df.loc[ix,features].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
 invalid=~np.isfinite(fields)
 save(work/'edge_numeric_audit.json',{'features':features,'invalid_by_feature':dict(zip(features,invalid.sum(axis=0).tolist())),'excluded_rows':int(invalid.any(axis=1).sum())})
 valid=~invalid.any(axis=1);x=fields[valid];ix=ix[valid];""")
m=m.replace("x=df[features].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64);assert np.isfinite(x).all(),'Nonfinite temporal feature; verify extraction before evaluation';","""x=df[features].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
 invalid=~np.isfinite(x);valid=~invalid.any(axis=1)
 audit=[{'block':j,'selected':int(mask.sum()),'excluded':int((mask & ~valid).sum()),'invalid_by_feature':dict(zip(features,invalid[mask].sum(axis=0).tolist())),'retained_by_class':{str(v):int((mask & valid & (y[chosen]==v)).sum()) for v in (0,1)}} for j in range(5) for mask in [block[chosen]==j]]
 save(work/'temporal_nonfinite_audit.json',audit)
 df=df.iloc[valid];chosen=chosen[valid];x=x[valid];""").replace("k,du=dedup(x,yy,d,f);","""seen=set();keep=[]
 for i,row in enumerate(x):
  fingerprint=row.tobytes()
  if fingerprint not in seen:seen.add(fingerprint);keep.append(i)
 k=np.asarray(keep,dtype=int);du={'later_duplicates_removed':len(x)-len(k)};""").replace("gframe.groupby('group').block.nunique();bad=set(bad[bad>1].index);good=~g.isin(bad).to_numpy()","gframe.groupby('group').block.transform('min');good=(block==bad.to_numpy())").replace("excluded']=len(bad)","excluded']=int(g[~good].nunique())")
m=m.replace("cached=work/(name+'_cache.npz')","identity={'protocol':'v3','contract':identity};cached=work/(name+'_cache.npz')").replace("meta=folder/'done.json'","meta=folder/'done.json'\n if track in ('edge','temporal','external'):identity['input_identity']=json.loads((work/(track+'_input_identity.json')).read_text())")
for i,s,name in [(2,p,'preflight.py'),(3,e,'engine02.py'),(4,m,'master.py')]:
 compile(s,name,'exec');put(i,f'SOURCE = {s!r}\n(WORK/{name!r}).write_text(SOURCE,encoding="utf-8")\n')
old=ident.copy();new=dict(old,source_sha=H(m.encode()))
restore=get(5).split('if AUTO_RESUME')[0].replace("paths=list(INPUT.rglob('FLIDS_Packages10_13_Results.zip'))","paths=list(INPUT.rglob('FLIDS_Packages10_13_Results.zip'))")
manifest=base64.b64encode(zlib.compress(json.dumps(files).encode(),9)).decode()
restore+='\nimport base64,zlib,tempfile\nBASE=json.loads(zlib.decompress(base64.b64decode('+repr(manifest)+')))\nOLD='+repr(old)+'\nNEW='+repr(new)+'''
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def verify(root):
    identity=json.loads((root/'identity.json').read_text())
    assert identity in (OLD,NEW), 'Unknown checkpoint identity'
    assert digest(root/'master.py')==identity['source_sha'], 'Checkpoint source mismatch'
    config=json.loads((root/'config.json').read_text())
    assert all(config.get(k)==v for k,v in OLD.items() if k not in ('source_sha','input_exp09_sha'))
    assert config['nf_content_sha']==OLD['input_exp09_sha']
    for name,expected in BASE.items():
        assert digest(root/name)==expected, 'Historical checkpoint mismatch: '+name
    for marker in (root/'bundles').glob('*/done.json'):
        score=json.loads(marker.read_text())['score_sha256']
        assert digest(marker.parent/'scores.npz')==score
        analysis=marker.parent/'analysis_complete.json'
        if analysis.exists(): assert json.loads(analysis.read_text())['scores_sha256']==score
    return identity

if not SMOKE_TEST:
    assert RESOLVED['nf_content_sha']==OLD['input_exp09_sha'], 'Experiment 09 content mismatch'
    if (WORK/'recovery_ready').exists():
        assert verify(WORK)==NEW
    else:
        candidates=resume_candidates()
        if len(candidates)!=1: raise ValueError(f'Select one checkpoint with RESUME_ZIP: {candidates}')
        with tempfile.TemporaryDirectory(dir=WORK.parent) as tmp:
            root=candidates[0]
            if root.is_file():
                with zipfile.ZipFile(root) as archive:
                    for item in archive.infolist():
                        assert (Path(tmp)/item.filename).resolve().is_relative_to(Path(tmp).resolve())
                    archive.extractall(tmp)
                root=Path(tmp)
            previous=verify(root)
            for item in root.iterdir():
                if item.name not in ('bundles','summary','shift','nf_cache.npz','nf_audit.json','original_nf_provenance.json','config.json','status.json','run.log') and not (previous==NEW and item.name.startswith(('edge_','external_','temporal_'))): continue
                dest=WORK/item.name
                if item.is_dir(): shutil.copytree(item,dest,dirs_exist_ok=True)
                else: shutil.copy2(item,dest)
            (WORK/'identity.json').write_text(json.dumps(NEW))
            verify(WORK)
            (WORK/'recovery_ready').touch()
    print('Verified historical checkpoint and Experiment 09 input.')
'''
put(5,get(6));put(6,restore)
for c in cells:
 if c['cell_type']=='code':c.update(outputs=[],execution_count=None);compile(''.join(c['source']),'cell','exec')
nb['metadata'].pop('widgets',None)
Path('FLIDS_Master_V3.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
