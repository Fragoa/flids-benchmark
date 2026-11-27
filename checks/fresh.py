import json,tempfile,os,sys
from pathlib import Path
n=json.load(open(Path(__file__).resolve().parents[1]/'notebooks/current/FLIDS_Master_V3_Fresh.ipynb'))
def run(root,smoke,hours,archive=''):
 scope={}
 for i,c in enumerate(n['cells']):
  if c['cell_type']!='code':continue
  s=''.join(c['source'])
  if i==1:s=s.replace("Path('/mnt/data')","Path("+repr(str(root))+")").replace('SMOKE_TEST = False',f'SMOKE_TEST = {smoke}').replace('SESSION_HOURS = 11.25',f'SESSION_HOURS = {hours}').replace("EXP09_ZIP = ''","EXP09_ZIP = "+repr(str(Path.home()/'Downloads/FLIDS_Experiment09_Results.zip'))).replace("RESUME_ZIP = ''","RESUME_ZIP = "+repr(archive))
  s=s.replace('from IPython.display import display, FileLink','display=print; FileLink=str')
  exec(s,scope)
 return scope['result'],json.load(open(scope['WORK']/'status.json'))
with tempfile.TemporaryDirectory() as t:
 root=Path(t);a,s=run(root,False,0);assert s['state']=='paused_for_resume' and s['completed_tuning_trials']==0
 dest=root/'restored';dest.mkdir();a,s=run(dest,False,0,str(a));assert s['state']=='paused_for_resume'
 a,s=run(root,True,1);assert s['state']=='completed_requested_supported_stages';print('PASS',s['completed_model_bundles'],s['completed_analysis_bundles'])
