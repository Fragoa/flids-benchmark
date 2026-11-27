import ast,json,zipfile,hashlib,tempfile,io,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
n=json.load(open(Path(__file__).resolve().parents[1]/'notebooks/current/FLIDS_Master_V3.ipynb'));m=ast.literal_eval(ast.parse(''.join(n['cells'][4]['source'])).body[0].value)
for fn in ast.parse(m).body:
 if getattr(fn,'name',None) in ['bundle','split_outer','groupsplit','timegate','seed','sha','analyze']:exec(compile(ast.Module([fn],[]),'master','exec'))
DEADLINE=float('inf')
with tempfile.TemporaryDirectory() as tmp,zipfile.ZipFile(Path.home()/'Downloads/FLIDS_Packages10_13_Results.zip') as z:
 w=Path(tmp);a=np.load(io.BytesIO(z.read('nf_cache.npz')));y,d,f=[a[k] for k in ('y','d','family')]
 name=next(n for n in z.namelist() if n.endswith('analysis_complete.json'));folder=Path(name).parent
 for n in ('analysis_complete.json','done.json','scores.npz'):z.extract(str(folder/n),w)
 i=json.loads((w/folder/'done.json').read_text())['identity'];held=tuple(v for v in sorted(set(d)) if v in i['held'])
 cfg,p=bundle(w,None,y,d,f,held,i['seed'],i['track'],i['method'],{},i['scenario'],i['participation']);assert p==w/folder;analyze(p,{},w);print('Existing NF bundle and analysis skipped successfully')
