"""Execute only the frozen task ledger. No tuning, retries or timing benchmarks."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import json,time,sys,datetime
from runtime import HERE,ROOT,digest,dump,safe_episode
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def execute(tasks,phase,workers):
    marker=HERE/(phase+'_STARTED.json');assert not marker.exists(),f'{phase} already opened; no automatic retries'
    dump(marker,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(tasks),'task_ids':[t['task_id'] for t in tasks]})
    rows=[];begin=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(safe_episode,t):t for t in tasks}
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            dump(HERE/(phase+'_PROGRESS.json'),{'recorded':len(rows),'planned':len(tasks),'exceptions':sum(r['stop_reason']=='execution_exception' for r in rows),'completed':sum(r['completed'] for r in rows),'elapsed_seconds':time.time()-begin,'last':row['task_id']})
            print(f'{phase} {len(rows)}/{len(tasks)} {row["task_id"]} {row["stop_reason"]}',flush=True)
    dump(HERE/(phase+'_EPISODES.json'),rows);return rows
def main():
    protocol=read(HERE/'PROTOCOL.json');assert digest(HERE/'PROTOCOL.json')==(HERE/'PROTOCOL.sha256').read_text().split()[0]
    assert digest(HERE/'TASKS.json')==protocol['tasks_sha256']
    amendment=read(HERE/'SMOKE_CORRECTION_FREEZE.json')
    for p,h in protocol['code_sha256'].items():assert digest(ROOT/p)==amendment['code_overrides'].get(p,h),p
    tasks=read(HERE/'TASKS.json');mode=sys.argv[1]
    if mode=='smoke2':
        tasks=read(HERE/'SMOKE2_TASKS.json')
        rows=execute(tasks,'smoke2',amendment['workers'])
        gate={'exceptions':[r['task_id'] for r in rows if r['stop_reason']=='execution_exception'],'nonfinite':[r['task_id'] for r in rows if r['stop_reason']=='nonfinite_state'],'replay':{},'physical_isolation':True,'paired_references':True,'control_aborts':[r['task_id'] for r in rows if not r['completed']]}
        for arm in ['W1','W2a']:
            rr=[r for r in rows if r['phase']=='replay' and r['cell'].startswith(arm+'_')];assert len(rr)==2
            aa,bb=[np.load(r['trace']) for r in rr]
            gate['replay'][arm]={k:bool(np.array_equal(aa[k],bb[k])) for k in ['states','U','solver_accepted']}
        for r in rows:
            if 'plant_mass' not in r:continue
            t=next(t for t in tasks if t['task_id']==r['task_id'] and t['phase']==r['phase'])
            gate['physical_isolation'] &= bool(np.isclose(r['plant_mass'],r['controller_mass']*t['mass_scale']) and np.allclose(r['plant_inertia'],np.array(r['controller_inertia'])*t['inertia_scale']))
            paired=[q for q in rows if q['group']==r['group'] and q['case_id']==r['case_id'] and 'reference_sha256'in q]
            gate['paired_references'] &= len({q['reference_sha256'] for q in paired})==1
        gate['passed']=bool(not gate['exceptions'] and not gate['nonfinite'] and all(all(v.values()) for v in gate['replay'].values()) and gate['physical_isolation'] and gate['paired_references'])
        dump(HERE/'SMOKE_GATE.json',gate);print(json.dumps(gate,indent=2))
    elif mode in ['A','B','C']:
        assert read(HERE/'SMOKE_GATE.json')['passed'],'Smoke gate blocked'
        freeze=HERE/'EVALUATION_FREEZE.json'
        if not freeze.exists():dump(freeze,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':digest(HERE/'PROTOCOL.json'),'correction_sha256':digest(HERE/'SMOKE_CORRECTION_FREEZE.json'),'smoke_gate_sha256':digest(HERE/'SMOKE_GATE.json'),'runner_sha256':digest(__file__),'count':1120,'no_correction_after_smoke':True})
        assert read(freeze)['runner_sha256']==digest(__file__)
        execute([t for t in tasks if t['phase']=='evaluation' and t['group']==mode],mode,amendment['workers'])
    else:raise ValueError(mode)
if __name__=='__main__':main()
