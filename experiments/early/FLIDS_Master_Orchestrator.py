"""FL-IDS Master v1: auditable, resumable staged orchestration of seven experiments.
The independent original engines are bundled unchanged; supplemental baselines are
reported separately so previously published protocols remain comparable.
"""
import os,sys,subprocess,json,zipfile,hashlib,time,traceback,platform
from pathlib import Path

STAGES=['AUDIT','01','EXTRA01','02','EXTRA02','03','04','05','06','07','BUNDLE']
FILES={'AUDIT':'FLIDS_DatasetSplit_Audit_Engine.py',
       '01':'FLIDS_Experiment01_Engine_v1_1.py',
       '02':'FLIDS_Experiment02_Engine.py',
       '03':'FLIDS_Experiment03_Engine.py',
       '04':'FLIDS_Experiment04_Engine.py',
       '05':'FLIDS_Experiment05_Engine.py',
       '06':'FLIDS_Experiment06_Controlled_Heterogeneity_Engine.py',
       '07':'FLIDS_Experiment07_Statistical_Synthesis_Engine.py',
       'BUNDLE':None,
       'EXTRA01':'FLIDS_Master_Extra_Baselines.py',
       'EXTRA02':'FLIDS_Master_Extra_Baselines.py'}
ARCHIVES={'AUDIT':'FLIDS_Dataset_Split_Validity_Audit.zip',
          **{f'{i:02d}':f'FLIDS_Experiment{i:02d}_Results.zip' for i in range(1,7)},
          '07':'FLIDS_Experiment07_Statistical_Synthesis_Results.zip',
          'BUNDLE':'FLIDS_Master_All_Results.zip',
          'EXTRA01':'FLIDS_Extra_Baselines_Exp01.zip','EXTRA02':'FLIDS_Extra_Baselines_Exp02.zip'}

class Runner:
    def __init__(self, work='/kaggle/working', scripts=None, smoke=False, workers=2):
        self.work=Path(work);self.work.mkdir(parents=True,exist_ok=True)
        self.scripts=Path(scripts or self.work/'flids_master_scripts')
        self.smoke=bool(smoke);self.workers=int(workers)
        self.logs=self.work/'FLIDS_Master_Logs';self.logs.mkdir(exist_ok=True)
        self.manifest=self.work/'FLIDS_Master_Manifest.json'
        self.report=json.loads(self.manifest.read_text()) if self.manifest.exists() else {'stages':{},'created':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source':'FLIDS Master v1'}
    def env(self):
        e=os.environ.copy()
        for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):e[k]='1'
        if self.smoke:
            e.update(EDGE_AUDIT_SMOKE='1',FLIDS_SMOKE='1',FLIDS_EXP02_SMOKE='1',FLIDS_EXP03_SMOKE='1',FLIDS_EXP05_SMOKE='1',FLIDS_EXP06_SMOKE='1',FLIDS_AUDIT_SMOKE='1')
        e.update(EDGE_AUDIT_ROOT=str(self.work),FLIDS_EXP02_ROOT=str(self.work),FLIDS_EXP03_ROOT=str(self.work),FLIDS_EXP04_ROOT=str(self.work),FLIDS_EXP05_ROOT=str(self.work),FLIDS_EXP06_ROOT=str(self.work),FLIDS_AUDIT_ROOT=str(self.work))
        return e
    def cmd(self, stage, action, worker=None):
        name=FILES[stage];p=self.scripts/name
        if not p.exists():raise FileNotFoundError('Missing embedded source '+str(p))
        cmd=[sys.executable,'-X','faulthandler',str(p)]
        if stage=='AUDIT':return cmd
        if stage=='07':return cmd+['--input-root',str(self.work),'--output-root',str(self.work)]
        if stage=='04':return cmd+['--mode','run']
        if stage.startswith('EXTRA'):
            cmd+=['--track',str(int(stage[-2:])), '--mode',action,'--work',str(self.work), '--engine',str(self.scripts/FILES[stage[-2:]])]
            if worker is not None:cmd+=['--worker',str(worker),'--workers',str(self.workers)]
            if self.smoke:cmd+=['--smoke']
            return cmd
        cmd+=['--mode',action]
        if worker is not None:cmd+=['--worker',str(worker),'--workers',str(self.workers)]
        return cmd
    def subprocess(self,stage,action,worker=None):
        suffix=f'_{worker}' if worker is not None else ''
        log=self.logs/f'{stage}_{action}{suffix}.log'
        cmd=self.cmd(stage,action,worker)
        with log.open('w',encoding='utf8') as f:
            f.write('COMMAND: '+' '.join(cmd)+'\n');f.flush()
            process=subprocess.run(cmd,cwd=self.work,stdout=f,stderr=subprocess.STDOUT,env=self.env())
        data=log.read_text(errors='replace')
        print(f'{stage} {action} {worker if worker is not None else ""} exit={process.returncode}',flush=True)
        print(data[-1300:],flush=True)
        if process.returncode:raise RuntimeError(f'Failed stage {stage}/{action}/{worker}, inspect {log}. Exit={process.returncode}')
    def parallel(self,stage,action):
        running=[]
        for worker in range(self.workers):
            log=self.logs/f'{stage}_{action}_{worker}.log'
            with log.open('w',encoding='utf8') as f:f.write('WORKER '+str(worker)+'\n')
            handle=log.open('a',encoding='utf8')
            p=subprocess.Popen(self.cmd(stage,action,worker),cwd=self.work,stdout=handle,stderr=subprocess.STDOUT,env=self.env())
            running.append((worker,p,handle,log))
        failed=[]
        for worker,p,h,log in running:
            code=p.wait();h.close()
            print(f'{stage} {action} worker{worker}: exit={code}',flush=True)
            tail=log.read_text(errors='replace')[-1300:]
            if code:failed.append((worker,code));print(tail,flush=True)
        if failed:raise RuntimeError(f'{stage} {action} failed workers {failed}; logs in {self.logs}')
    def valid_zip(self,stage):
        p=self.work/ARCHIVES[stage]
        if not p.exists() or p.stat().st_size<100:return False
        try:
            with zipfile.ZipFile(p) as z:
                if z.testzip() is not None:return False
                names=z.namelist()
                if stage=='BUNDLE':
                    if not any(x.endswith('FLIDS_Master_Manifest.json') for x in names):return False
                elif stage=='07':
                    if not any(x.endswith('/analysis_manifest.json') for x in names):return False
                elif stage not in ('AUDIT','EXTRA01','EXTRA02') and not any(x.endswith('/run_status.json') for x in names):return False
        except Exception:return False
        return True
    def bundle(self):
        missing=[s for s in STAGES if s!='BUNDLE' and not self.valid_zip(s)]
        if missing:raise RuntimeError('Cannot bundle incomplete stages: '+str(missing))
        destination=self.work/ARCHIVES['BUNDLE']
        self.report['bundle_notes']=[
            'Original experiment protocols preserved; supplemental baselines not silently added to original Experiment07 synthesis.',
            'Experiment04 uses hypothetical class-prior reweighting, not observed deployment.',
            'Four source domains are the independent inference clusters, not 20 independent data sources.',
            'Independent upstream sampling means samples and split indices are not always identical across distinct experiments.',
            'Cross-domain fingerprint checks do not establish capture/session independence.',
        ]
        self.save()
        with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for stage in STAGES:
                if stage=='BUNDLE':continue
                path=self.work/ARCHIVES[stage]
                z.write(path,arcname='stage_archives/'+path.name)
            z.write(self.manifest,arcname='FLIDS_Master_Manifest.json')
        print('MASTER_BUNDLE_READY',destination,round(destination.stat().st_size/2**20,2),'MiB',flush=True)

    def run_stage(self,stage,force=False):
        if stage not in STAGES:raise ValueError(f'Unknown stage {stage}; valid {STAGES}')
        if not force and self.valid_zip(stage):
            print(f'SKIP_VERIFIED_ARCHIVE {stage}: {self.work/ARCHIVES[stage]}',flush=True)
            self.report['stages'][stage]=dict(status='verified_archive',archive=str(self.work/ARCHIVES[stage]));self.save();return
        tic=time.time();self.report['stages'][stage]=dict(status='running');self.save()
        try:
            if stage=='BUNDLE':self.bundle()
            elif stage in ('AUDIT','04','07'):self.subprocess(stage,'run')
            elif stage.startswith('EXTRA'):
                self.parallel(stage,'main');self.subprocess(stage,'aggregate')
            else:
                self.subprocess(stage,'prepare')
                self.parallel(stage,'pilot')
                self.subprocess(stage,'gate')
                self.parallel(stage,'main')
                self.subprocess(stage,'aggregate')
            if not self.valid_zip(stage):raise RuntimeError(f'Stage {stage} returned success but ZIP is missing or invalid')
            self.report['stages'][stage]=dict(status='complete',archive=str(self.work/ARCHIVES[stage]),elapsed_seconds=round(time.time()-tic,2),zip_bytes=(self.work/ARCHIVES[stage]).stat().st_size)
            self.save()
            print('STAGE_COMPLETE',stage,flush=True)
        except Exception as e:
            self.report['stages'][stage]=dict(status='failed',error=str(e),elapsed_seconds=round(time.time()-tic,2))
            self.save();raise
    def save(self):self.manifest.write_text(json.dumps(self.report,indent=2),encoding='utf8')
    def run(self,stages=None):
        stages=STAGES if stages is None else stages
        for s in stages:self.run_stage(s)
        print('MASTER_STAGES_FINISHED',stages,flush=True)
        return self.manifest

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--work',default='/kaggle/working');ap.add_argument('--scripts');ap.add_argument('--stages',default=','.join(STAGES));ap.add_argument('--smoke',action='store_true');ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args();Runner(a.work,a.scripts,a.smoke,a.workers).run([x.strip() for x in a.stages.split(',') if x.strip()])
