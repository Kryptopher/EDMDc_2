"""Authorized late continuation; immutable old artifacts, resume absent tasks only."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,uuid,hashlib,datetime,importlib.util,traceback,subprocess,tempfile
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
PID_ROOT=ROOT/'wrench_pid_baseline_v1/attempt_20260930';INT_ROOT=ROOT/'wrench_interception_v1'
MODULES=None
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable(task):return hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()
def key(r):return (r.get('arm','PID'),r['candidate_id'],r['case_id'])

def robust_dump(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    for attempt in range(20):
        try:os.replace(tmp,p);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.01*2**attempt,.5))

def event(stage,**extra):
    value=dict(time=now(),stage=stage,**extra)
    with (HERE/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
    robust_dump(HERE/'progress.json',value);print(json.dumps(value),flush=True)

def modules():
    global MODULES
    if MODULES is not None:return MODULES
    sys.path.insert(0,str(ROOT));loaded=[]
    for name,path in [('recovered_pid_campaign',PID_ROOT),('recovered_int_campaign',INT_ROOT)]:
        sys.path.insert(0,str(path));spec=importlib.util.spec_from_file_location(name,path/'campaign.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);loaded.append(m)
    import pid,core
    # Only JSON persistence is changed; controller/model/plant arithmetic is untouched.
    for m in (*loaded,pid,core):m.dump=robust_dump
    MODULES=tuple(loaded);return MODULES

def collect(folder,stage):
    tasks=read(folder/stage/'tasks.json');lookup={key(t):t for t in tasks};rows={};records={}
    assert len(lookup)==len(tasks)
    for p in (folder/stage/'runs').rglob('metrics.json'):
        r=read(p);k=key(r);assert k in lookup and k not in rows,(p,k)
        assert r['task_sha256']==stable(lookup[k]),p
        trace=Path(r['trace']);assert trace.is_file(),trace
        rows[k]=r;records[str(p)]=sha(p);records[str(trace)]=sha(trace)
    missing=[t for t in tasks if key(t) not in rows]
    return tasks,list(rows.values()),missing,records

def copy_file(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(dst)==sha(src),(src,dst)
    else:dst.write_bytes(src.read_bytes())

def prepare():
    assert not (HERE/'amendment.json').exists()
    pm,im=modules();pm.verify();im.verify()
    assert read(PID_ROOT/'gate.json')['passed'] and read(PID_ROOT/'blind_test/pairing_gate.json')['passed'] and read(INT_ROOT/'gate.json')['passed']
    assert sha(PID_ROOT/'selection.json')==read(PID_ROOT/'selection.sha256.json')['sha256']
    inputs={};inventory={}
    for name,folder,stage,expected,missing_count in [('pid',PID_ROOT,'blind_test',180,20),('interception',INT_ROOT,'validation',765,3)]:
        tasks,rows,missing,files=collect(folder,stage);assert len(rows)==expected and len(missing)==missing_count
        inputs.update(files);inventory[name]=dict(stage=stage,planned=len(tasks),recovered=len(rows),missing=len(missing),missing_tasks=missing)
        robust_dump(HERE/name/stage/'original_tasks.json',tasks);robust_dump(HERE/name/stage/'recovered_episodes.json',rows)
        for file in ('protocol.json','protocol.sha256.json','candidates.json','gate.json'):
            copy_file(folder/file,HERE/name/file);inputs[str(folder/file)]=sha(folder/file)
        for stage_name in ('screening','validation','blind_test' if name=='pid' else 'evaluation'):
            src=folder/stage_name/'scenarios.json'
            if src.exists():copy_file(src,HERE/name/stage_name/'scenarios.json');inputs[str(src)]=sha(src)
        for rel in ('screening/accounting.json','screening/ranking.json','validation/finalists.json'):
            src=folder/rel
            if src.exists():copy_file(src,HERE/name/rel);inputs[str(src)]=sha(src)
    for rel in ('selection.json','selection.sha256.json','validation/accounting.json','validation/ranking.json','blind_test/pairing_gate.json'):
        copy_file(PID_ROOT/rel,HERE/'pid'/rel);inputs[str(PID_ROOT/rel)]=sha(PID_ROOT/rel)
    assert not (INT_ROOT/'selection.json').exists() and not (INT_ROOT/'evaluation/tasks.json').exists()
    robust_dump(HERE/'inventory.json',inventory)
    amendment=dict(created_utc=now(),authorization='User: run the missing ones then; after status identifying 20 missing PID blind tasks, 3 missing interception validation tasks and the not-yet-run interception evaluation.',
        deadline_extension=True,classification='Disclosed late continuation of frozen protocols; PID remains post-hoc exploratory; interception evaluation prospective under original outcome rules plus this timing amendment.',
        unchanged='Controller/model/plant code, all gain/weight candidate lists, seeds, case counts, gates, metrics, selection rankings and outcome sentences. Existing failures/aborts retained, never rerun. PID c02 selection unchanged.',
        change='Retry atomic JSON replace up to 20 attempts on PermissionError using unique temporary files. Recover all per-episode records including worker results omitted from stale indexes. New outputs in this folder; original artifacts preserved. Original time cutoffs waived by user continuation instruction.',
        sequence=['20 missing PID blind episodes','3 missing interception validation episodes','rank all 768 validation records by original rule and freeze selections','40 frozen seed-paired interception scenarios x 4 arms (160 episodes)','reports and explicit-path commit; no push'],
        prior_pid_protocol_sha256=sha(PID_ROOT/'protocol.json'),prior_int_protocol_sha256=sha(INT_ROOT/'protocol.json'),pid_selection_sha256=sha(PID_ROOT/'selection.json'),
        interception_outcome_sentences=read(INT_ROOT/'protocol.json')['outcomes'],workers=4,source_sha256=sha(Path(__file__)),preserved_input_hashes=inputs,no_retuning_on_evaluation=True,hardware_actions=False)
    robust_dump(HERE/'amendment.json',amendment);(HERE/'amendment.sha256').write_text(sha(HERE/'amendment.json')+'\n')
    # Exercise the precise failure mode without a controller/plant run.
    original_replace=os.replace;calls=[]
    def transient(src,dst):
        calls.append(1)
        if len(calls)<=3:raise PermissionError('simulated Windows sharing violation')
        return original_replace(src,dst)
    try:
        os.replace=transient;robust_dump(HERE/'writer_test/value.json',dict(value='complete'))
    finally:os.replace=original_replace
    assert read(HERE/'writer_test/value.json')==dict(value='complete') and len(calls)==4
    robust_dump(HERE/'writer_test/result.json',dict(passed=True,simulated_permission_failures=3,successful_replace_attempt=4,no_simulation=True))
    event('recovery_frozen',amendment_sha256=sha(HERE/'amendment.json'),recovered_pid=180,recovered_interception_validation=765)

def worker(bundle):
    name,task=bundle;pm,im=modules()
    return (pm.safe_episode if name=='pid' else im.safe_episode)(task)

def batch(name,stage,pending,recovered,total):
    destination=HERE/name/stage;rows=list(recovered);existing={key(r) for r in rows}
    tasks=[]
    for original in pending:
        assert key(original) not in existing
        task=dict(original);task['output']=str(destination/'new_runs');tasks.append(task)
        assert {k:v for k,v in task.items() if k!='output'}=={k:v for k,v in original.items() if k!='output'}
    robust_dump(destination/'continuation_tasks.json',tasks)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(worker,(name,t)):t for t in tasks}
        for f in as_completed(futures):
            row=f.result();rows.append(row);assert len({key(r) for r in rows})==len(rows)
            robust_dump(destination/'episodes.json',rows)
            event(name+'_'+stage,recorded=len(rows),planned=total,new_recorded=len(rows)-len(recovered))
    assert len(rows)==total
    robust_dump(destination/'accounting.json',dict(planned=total,recovered=len(recovered),newly_run=len(tasks),recorded=len(rows),complete=True,time=now(),continuation_beyond_deadline=True))
    return rows

def render(name,module):
    module.HERE=HERE/name;module.report()
    report=HERE/name/'REPORT.md';content=report.read_text(encoding='utf-8')
    label='**Continuation beyond the original overnight deadline, explicitly authorized by the user. Original partial reports and all failed/aborted episodes are preserved. JSON-writer repair only; no change to control mathematics or selection rules. See ../amendment.json.**\n\n'
    report.write_text(label+content,encoding='utf-8')
    freeze=HERE/name/'result_freeze.json';data=read(freeze);data['artifacts']['REPORT.md']=sha(report)
    robust_dump(freeze,data);(freeze.parent/'result_freeze.sha256').write_text(sha(freeze)+'\n')

def run():
    a=read(HERE/'amendment.json');assert sha(HERE/'amendment.json')==(HERE/'amendment.sha256').read_text().strip()
    assert sha(Path(__file__))==a['source_sha256']
    pm,im=modules();pm.verify();im.verify();inventory=read(HERE/'inventory.json')
    pid_rows=batch('pid','blind_test',inventory['pid']['missing_tasks'],read(HERE/'pid/blind_test/recovered_episodes.json'),200)
    refs={r['case_id']:r['reference_sha256'] for r in read(PID_ROOT/'blind_test/pairing_gate.json')['cases']}
    for row in pid_rows:
        if 'reference_sha256' in row:assert row['reference_sha256']==refs[row['case_id']]
    assert sha(PID_ROOT/'selection.json')==a['pid_selection_sha256']==sha(HERE/'pid/selection.json')
    render('pid',pm);event('pid_report_complete',report=str(HERE/'pid/REPORT.md'))
    rows=batch('interception','validation',inventory['interception']['missing_tasks'],read(HERE/'interception/validation/recovered_episodes.json'),768)
    # Equal full budget for every finalist before selection; keep failed flights.
    for arm in im.ARMS:
        rr=[r for r in rows if r['arm']==arm];assert len(rr)==192
        for cid in {r['candidate_id'] for r in rr}:assert sum(r['candidate_id']==cid for r in rr)==48
    ranking=im.rankings(rows);robust_dump(HERE/'interception/validation/ranking.json',ranking)
    candidates=read(INT_ROOT/'candidates.json')
    selected={arm:[next(c for c in candidates[arm] if c['candidate_id']==ranking[arm][0]['candidate_id'])] for arm in im.ARMS}
    robust_dump(HERE/'interception/selection.json',dict(created_utc=now(),selected=selected,protocol_sha256=sha(INT_ROOT/'protocol.json'),amendment_sha256=sha(HERE/'amendment.json'),ranking_sha256=sha(HERE/'interception/validation/ranking.json')))
    selection_hash=sha(HERE/'interception/selection.json');robust_dump(HERE/'interception/selection.sha256.json',dict(sha256=selection_hash))
    render('interception',im);event('interception_selection_frozen',selection_sha256=selection_hash,selected={k:v[0]['candidate_id'] for k,v in selected.items()})
    # make_tasks uses the original frozen seeds/configs; im.HERE points only to new output.
    tasks=im.make_tasks('evaluation',selected);assert len(tasks)==160
    assert sha(HERE/'interception/selection.json')==selection_hash
    robust_dump(HERE/'interception/evaluation/tasks.json',tasks)
    evaluation=batch('interception','evaluation',tasks,[],160)
    for cid in {r['case_id'] for r in evaluation}:
        assert {r['arm'] for r in evaluation if r['case_id']==cid}==set(im.ARMS)
        assert len({r['reference_sha256'] for r in evaluation if r['case_id']==cid and 'reference_sha256' in r})<=1
    render('interception',im)
    for file,h in a['preserved_input_hashes'].items():assert sha(Path(file))==h,file
    assert sha(PID_ROOT/'selection.json')==a['pid_selection_sha256']
    event('all_missing_work_completed',pid=200,interception_validation=768,interception_evaluation=160)
    text='''# Missing-run continuation — 30 September 2026

Completed after the original overnight deadline, explicitly authorized by the user. Original protocols, partial reports, traces, failures and selections were preserved. Only the JSON writer and execution bookkeeping changed; no controller/model/plant arithmetic changed.

- PID: recovered 180 saved blind episodes and ran exactly the 20 missing ones. The original c02 selection and hash are unchanged. [Full report](pid/REPORT.md).
- Interception: recovered 765 saved validation episodes including all aborts; ran exactly three missing W3 episodes. Selected every arm only after its full 192-episode finalist-validation budget, using the frozen rule. Selection hash precedes evaluation. Ran all 40 frozen scenarios across four arms (160 episodes). [Full report](interception/REPORT.md).
- Block 2 was already complete and was not rerun.

The writer now retries Windows permission/sharing errors. Its test injected three failures and confirmed the fourth replace succeeded. No tuning used evaluation results. All old recovered metadata and traces were hash-checked unchanged. Timing is not an isolated benchmark. Hardware untouched; nothing pushed.

Amendment and input hashes: `amendment.json`. Output hashes: `freeze.json` and `freeze.sha256`. New raw NPZ traces remain on disk and are hashed; source, metadata, reports and hashes are committed by explicit path.
'''
    (HERE/'REPORT.md').write_text(text,encoding='utf-8')
    files=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.jsonl','.md','.npz','.sha256') and p.name not in ('freeze.json','freeze.sha256','commit.json')]
    robust_dump(HERE/'freeze.json',dict(created_utc=now(),artifacts={str(p.relative_to(HERE)):sha(p) for p in files}))
    (HERE/'freeze.sha256').write_text(sha(HERE/'freeze.json')+'\n')
    commit()

def commit():
    repo=ROOT.parent.parent
    files=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.jsonl','.md','.sha256') and p.name!='commit.json']
    paths=sorted(p.relative_to(repo).as_posix() for p in files);base=['git','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    with tempfile.NamedTemporaryFile(delete=False) as f:f.write(('\0'.join(paths)+'\0').encode());name=f.name
    try:
        subprocess.run(base+['add','-f','--pathspec-from-file='+name,'--pathspec-file-nul'],check=True)
        subprocess.run(base+['commit','--only','--pathspec-from-file='+name,'--pathspec-file-nul','-m','Complete missing PID episodes and frozen known-future interception evaluation'],check=True)
    finally:Path(name).unlink()
    rev=subprocess.check_output(base+['rev-parse','HEAD'],text=True).strip()
    changed=subprocess.check_output(base+['diff-tree','--no-commit-id','--name-only','-r',rev],text=True).splitlines();assert sorted(changed)==paths
    robust_dump(HERE/'commit.json',dict(commit=rev,paths=paths,pushed=False,freeze_sha256=sha(HERE/'freeze.json')))
    print('COMPLETED',rev,flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    try:globals()[args.action]()
    except Exception:
        robust_dump(HERE/(args.action+'_failure_'+datetime.datetime.now().strftime('%H%M%S')+'.json'),dict(time=now(),error=traceback.format_exc()));raise
