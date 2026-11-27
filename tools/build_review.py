import json,hashlib
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'review/ablation';cells=[]
def cell(s,code=True):
 d={'cell_type':'code' if code else 'markdown','metadata':{},'source':s.splitlines(True),'id':f'flids-{len(cells):02}'}
 if code:d.update(outputs=[],execution_count=None)
 cells.append(d)
cell('# FLIDS reviewer experiments\nAttach FLIDS_Master_V3_Fresh_Results.zip. Optionally attach FLIDS_Experiment01_Results.zip for the DDoS-UDP replay. Each file should be available as a ZIP under Kaggle Inputs. Set an explicit path if multiple copies exist. Run all cells. The projected-feature audit runs before any fit. Expected research scope: 2 variants × 15 methods × 4 holdouts × 5 seeds.\n',False)
cell("""from pathlib import Path
import sys, os, subprocess, hashlib, json, zipfile, shutil, tempfile
INPUT=Path('/kaggle/input')
OUTPUT=Path('/kaggle/working')
FRESH=''   # Optional explicit path to the fresh results ZIP
EDGE=''    # Optional explicit path to the Experiment 01 results ZIP
RESUME=''  # Optional previous FLIDS_Review_Ablation.zip (autodetected if present)
TRAIN=True
HOURS=11.0
CODE=OUTPUT/'review_code'
WORK=OUTPUT/'review_ablation'
CODE.mkdir(parents=True,exist_ok=True)
def find(name,override,required):
    paths=[Path(override)] if override else sorted(INPUT.rglob(name))
    if not paths and not required:return None
    if len(paths)!=1 or not paths[0].is_file():raise ValueError(f'Select exactly one {name} ZIP: {paths}')
    return paths[0]
fresh=find('FLIDS_Master_V3_Fresh_Results.zip',FRESH,True)
edge=find('FLIDS_Experiment01_Results.zip',EDGE,False)
resume=find('FLIDS_Review_Ablation.zip',RESUME,False)
""")
for f in ['engine02.py','master.py','ablation.py','edge_ties.py','projection.py']:
 cell(f"(CODE/{f!r}).write_text({(r/f).read_text()!r},encoding='utf-8')\n")
cell("""def run_script(name,*args):
    env=dict(os.environ,FLIDS_EXP02_ROOT=str(OUTPUT/'review_engine'),PYTHONUNBUFFERED='1')
    subprocess.run([sys.executable,str(CODE/name),*map(str,args)],env=env,check=True)
run_script('projection.py',fresh,OUTPUT/'projection_audit.csv')
if edge:run_script('edge_ties.py',edge,OUTPUT/'edge_replay.csv')
else:print('Edge replay skipped: attach Experiment 01 results ZIP.')
""")
cell("""if resume and not WORK.exists():
    with tempfile.TemporaryDirectory(dir=OUTPUT) as temp,zipfile.ZipFile(resume) as z:
        target=Path(temp)/'snapshot'
        for member in z.infolist():
            if not (target/member.filename).resolve().is_relative_to(target.resolve()):raise ValueError('Invalid checkpoint path')
        z.extractall(target)
        info=json.loads((target/'progress.json').read_text())
        markers=list(target.glob('*/bundles/*/analysis_complete.json'))
        assert info['expected']==600 and len(markers)==info['analyses']
        for marker in target.glob('*/bundles/*/done.json'):
            h=hashlib.sha256((marker.parent/'scores.npz').read_bytes()).hexdigest()
            assert h==json.loads(marker.read_text())['score_sha256']
        for marker in markers:
            done=json.loads((marker.parent/'done.json').read_text())
            assert json.loads(marker.read_text())['scores_sha256']==done['score_sha256']
            assert (marker.parent/'operating.csv').is_file()
        target.rename(WORK)
elif resume:print('Continuing the existing working directory.')
if TRAIN:
    run_script('ablation.py',fresh,WORK,'--hours',HOURS)
    print((WORK/'progress.json').read_text())
    print('Download:',OUTPUT/'FLIDS_Review_Ablation.zip')
""")
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
for c in cells:
 if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec')
(r/'FLIDS_Review_Fixes.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
