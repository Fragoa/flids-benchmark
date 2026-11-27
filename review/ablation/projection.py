"""Exact cross-domain signature overlap created by removing features; no fitting."""
import ast,hashlib,io,sys,zipfile
from pathlib import Path
import numpy as np,pandas as pd
def audit(path,out):
    with zipfile.ZipFile(path) as z:
        raw=z.read('nf_cache.npz')
        assert hashlib.sha256(raw).hexdigest()=='4952eb09930fbe35b5a2f9338b95023b197276d55cf0f3015c60e306aa51bff4'
        features=next(ast.literal_eval(n.value) for n in ast.parse(z.read('engine02.py')).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='FEATURES' for t in n.targets))
    with np.load(io.BytesIO(raw)) as a:x,y,d=a['x'],a['y'],a['d']
    rows=[]
    for name,drop in [('full',[]),('no_ports',['L4_SRC_PORT','L4_DST_PORT']),('no_ports_ttl',['L4_SRC_PORT','L4_DST_PORT','MIN_TTL','MAX_TTL'])]:
        v=np.ascontiguousarray(x[:,[i for i,k in enumerate(features) if k not in drop]])
        ids=np.unique(v.view(np.dtype((np.void,v.shape[1]*v.itemsize))).ravel(),return_inverse=True)[1].ravel()
        for h in sorted(set(d)):
            target=d==h;collide=np.isin(ids[target],ids[~target])
            rows.append(dict(variant=name,holdout=h,rows=int(target.sum()),matching_other_domain=int(collide.sum()),matching_benign=int(collide[y[target]==0].sum()),matching_attack=int(collide[y[target]==1].sum())))
    pd.DataFrame(rows).to_csv(out,index=False)
if __name__=='__main__':audit(sys.argv[1],sys.argv[2])
