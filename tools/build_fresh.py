from pathlib import Path
import json,ast
p=Path('FLIDS_Master_V3.ipynb');n=json.loads(p.read_text());cs=n['cells']
def read(i):return ''.join(cs[i]['source'])
def write(i,s):cs[i]['source']=s.splitlines(True)
write(0,"# FLIDS Master V3 · Fresh run\nAttach Experiment 09. Attach Edge and Temporal if required. Run all cells. A new session starts with empty checkpoints. To continue, set RESUME_ZIP to the fresh-run results ZIP. An existing working directory resumes automatically. The default budget reserves 45 minutes of a 12-hour session for the final fit and export.\n")
write(1,read(1).replace('SESSION_HOURS = 8.0','import time\nSESSION_STARTED = time.monotonic()\nSESSION_HOURS = 11.25').replace('AUTO_RESUME = True','').replace('flids_master_v3_', 'flids_fresh_v3_').replace('optional ZIP OR extracted v2 results root','optional fresh-run output ZIP').replace('Completed NF bundles are reused.','All four packages.'))
m=ast.literal_eval(ast.parse(read(4)).body[0].value)
m=m.replace("FLIDS_Packages10_13_Results.zip","FLIDS_Master_V3_Fresh_Results.zip")
m=m.replace('tmp.replace(zpath);return zpath', 'tmp.replace(zpath);return zpath')
write(4,'SOURCE = '+repr(m)+"\n(WORK/'master.py').write_text(SOURCE,encoding='utf-8')\n")
write(6,'''# Validate the research dataset before a fresh start or an explicit restore.
import tempfile
def validate_checkpoint(root):
    ident=json.loads((root/'identity.json').read_text())
    assert ident['source_sha']==hashlib.sha256((WORK/'master.py').read_bytes()).hexdigest()
    assert ident['smoke']==SMOKE_TEST and ident['input_exp09_sha']==RESOLVED['nf_content_sha']
    for marker in (root/'bundles').glob('*/done.json'):
        score=json.loads(marker.read_text())['score_sha256']
        assert hashlib.sha256((marker.parent/'scores.npz').read_bytes()).hexdigest()==score
        analysis=marker.parent/'analysis_complete.json'
        if analysis.exists(): assert json.loads(analysis.read_text())['scores_sha256']==score
if not SMOKE_TEST:
    assert RESOLVED['nf_content_sha']=='3cb34f8a174bb9a9ef84647e2c5e49599b0da3abaad589e73f62c968682f97eb', 'Experiment 09 content mismatch'
if (WORK/'identity.json').exists():
    validate_checkpoint(WORK)
    print('Continuing the current work directory.')
elif RESUME_ZIP:
    with tempfile.TemporaryDirectory(dir=WORK.parent) as tmp, zipfile.ZipFile(RESUME_ZIP) as archive:
        root=Path(tmp).resolve()
        for item in archive.infolist():
            assert (root/item.filename).resolve().is_relative_to(root)
        archive.extractall(root)
        validate_checkpoint(root)
        for item in root.iterdir():
            if item.name not in ('bundles','shift','nf_cache.npz','nf_audit.json','original_nf_provenance.json','run.log') and not item.name.startswith(('edge_','temporal_','external_')): continue
            if item.is_dir(): shutil.copytree(item,WORK/item.name,dirs_exist_ok=True)
            else: shutil.copy2(item,WORK/item.name)
        shutil.copy2(root/'identity.json',WORK/'identity.json')
else:
    assert not (WORK/'bundles').exists(), 'Incomplete restore: select its checkpoint archive.'
    print('Starting all requested stages from zero.')
''')
write(7,read(7).replace('session_hours=SESSION_HOURS','session_hours=max(0,SESSION_HOURS-(time.monotonic()-SESSION_STARTED)/3600)'))
write(8,read(8).replace('FLIDS_Packages10_13_Results.zip','FLIDS_Master_V3_Fresh_Results.zip'))
for c in cs:
 if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec')
n['metadata'].pop('papermill',None)
Path('FLIDS_Master_V3_Fresh.ipynb').write_text(json.dumps(n,indent=1,ensure_ascii=False))
