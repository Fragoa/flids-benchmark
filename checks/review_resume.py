from pathlib import Path
import tempfile,json,sys
r=Path(__file__).resolve().parents[1]/'review/ablation';n=json.loads((r/'FLIDS_Review_Fixes.ipynb').read_text())
with tempfile.TemporaryDirectory() as tmp:
 env={};out=Path(tmp)/'output';out.mkdir()
 for c in n['cells']:
  if c['cell_type']!='code':continue
  s=''.join(c['source']).replace("INPUT=Path('/kaggle/input')","INPUT=Path.home()/'Downloads'").replace("OUTPUT=Path('/kaggle/working')",f"OUTPUT=Path({str(out)!r})").replace("RESUME=''","RESUME="+repr(str(Path(sys.argv[1]).resolve()))+"").replace("HOURS=11.0","HOURS=1").replace("WORK,'--hours',HOURS)","WORK,'--hours',HOURS,'--limit',1)")
  exec(s,env)
 assert json.loads((env['WORK']/'progress.json').read_text())['analyses']==1
 assert (env['WORK']/'paired_results.csv').read_bytes()==Path(sys.argv[2]).read_bytes()
 print('PASS restored and replayed real tuned fit')
