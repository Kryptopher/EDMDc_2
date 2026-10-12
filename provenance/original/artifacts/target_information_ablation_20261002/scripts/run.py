"""Execute only the frozen stage ledger; never tune or retry."""
from runtime import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import time,inspect,ast

def checks():
    sim=quad_sim();p=np.array([4.,5.,6.]);v=np.array([.2,-.3,.4])
    packet_checks=[]
    for mode,d in [('cv',0),('delayed_cv_100ms',10),('delayed_cv_250ms',25)]:
        for k in [0,10,25,100]:
            history=np.arange(2025*6,dtype=float).reshape(2025,6)*.001
            sensor=CausalSensor(history,d);packet=sensor.receive(k)
            ref,yaw=causal_reference(packet,k,50,mode,d,0.,sim)
            assert packet.sample==k-d
            poison=history.copy();poison[k-d+26:]=np.nan
            altered=CausalSensor(poison,d).receive(k)
            assert packet==altered
            ref2,_=causal_reference(altered,k,50,mode,d,0.,sim)
            assert np.array_equal(ref,ref2)
            packet_checks.append({'mode':mode,'k':k,'read_sample':packet.sample,'newer_samples_poisoned_reference_unchanged':True})
    ref,_=causal_reference(Measurement(0,tuple(p),None),0,50,'position_hold',0,.7,sim)
    assert np.array_equal(ref[:,:3],np.broadcast_to(p,(50,3))) and np.all(ref[:,3:8]==0) and np.all(ref[:,9:]==0)
    tree=ast.parse(inspect.getsource(causal_reference))
    assert not {'truth','history','target','scenario_family','seed'}.intersection({n.id for n in ast.walk(tree) if isinstance(n,ast.Name)})
    with np.load(HERE/'SCENARIOS.npz') as z:truth=z[f"truth_{read(HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.json')['scenarios'][0]['seed']}"]
    actual=truth[:21].copy();good=metrics(actual,truth)
    assert good['captured'] and good['qualified_time_s']==.2
    shifted=truth.copy();shifted[:,:3]+=100
    assert not metrics(actual,shifted)['captured']
    return {'causal_packet_tests':packet_checks,'position_only_velocity_absent':True,'policy_AST_no_truth_generator_seed':True,'scoring_true_target_independent':True}

def execute(stage):
    p=assert_freeze();tasks=[t for t in read(HERE/'TASKS.json') if t['stage']==stage];assert tasks
    marker=HERE/f'{stage}_STARTED.json';assert not marker.exists(),'No retries or duplicate stage'
    dump(marker,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(tasks),'protocol_sha256':sha(HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.json')})
    begin=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(safe_episode,t):t for t in tasks}
        for f in as_completed(futures):
            row=f.result();rows.append(row)
            dump(HERE/f'{stage}_EPISODES.json',rows)
            dump(HERE/f'{stage}_PROGRESS.json',{'recorded':len(rows),'planned':len(tasks),'elapsed_seconds':time.time()-begin,'exceptions':sum(r['stop_reason']=='execution_exception' for r in rows),'last_task':row['task_id']})
            print(f"{stage} {len(rows)}/{len(tasks)} {row['cell']} {row['mode']} {row['family']} {row['stop_reason']} capture={row['captured']}",flush=True)
    return rows

def smoke():
    assert_freeze();unit=checks();dump(HERE/'CAUSAL_UNIT_CHECKS.json',unit)
    reproduction=execute('oracle_reproduction');old=read(ROOT/'paper_gap_audit_20261001/C_EPISODES.json');replays=[]
    cells={'Linear-A':'W1_bundle_W1','EDMDc-B':'W2a_bundle_W2a'}
    for row in reproduction:
        assert row['stop_reason']!='execution_exception',row
        saved=next(r for r in old if r['seed']==row['seed'] and r['cell']==cells[row['cell']])
        with np.load(row['trace']) as a,np.load(tracepath(saved['trace'])) as b:
            refsame=np.array_equal(a['reference'],b['reference']);accepted=np.array_equal(a['solver_accepted'],b['solver_accepted'])
            state_diff=float(np.max(np.abs(a['states']-b['states']))) if a['states'].shape==b['states'].shape else float('inf')
            input_diff=float(np.max(np.abs(a['U']-b['U']))) if a['U'].shape==b['U'].shape else float('inf')
            passed=refsame and accepted and state_diff<=1e-8 and input_diff<=1e-8
            replays.append({'cell':row['cell'],'seed':row['seed'],'reference_exact':refsame,'accepted_exact':accepted,'state_max_abs_difference':state_diff if np.isfinite(state_diff) else None,'applied_input_max_abs_difference':input_diff if np.isfinite(input_diff) else None,'passed':passed})
    if not all(r['passed'] for r in replays):
        dump(HERE/'SMOKE_GATE.json',{'passed':False,'oracle_reproduction':replays,'stop_reason':'oracle reproduction failed materially'});raise RuntimeError('Oracle reproduction failed; no full execution')
    rows=execute('smoke')
    failures=[r for r in rows if r['stop_reason'] in ('execution_exception','nonfinite_state') or not r.get('causal_access_verified',False)]
    gate={'passed':not failures,'oracle_reproduction':replays,'causal_unit_checks':unit,'smoke_episodes':len(rows),'execution_or_leakage_failures':failures,'control_aborts_retained':[r['task_id'] for r in rows if not r['completed']]}
    dump(HERE/'SMOKE_GATE.json',gate);assert gate['passed'],'Smoke failed; no full execution'
    print(json.dumps(gate,indent=2),flush=True)

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='smoke':smoke()
    else:
        assert stage in ['primary','secondary','position_only']
        assert read(HERE/'SMOKE_GATE.json')['passed']
        if stage!='primary':
            assert (HERE/'PRIMARY_RESULTS_FREEZE.json').exists() and (HERE/'PRIMARY_REPORT.md').exists()
            frozen=read(HERE/'PRIMARY_RESULTS_FREEZE.json');assert sha(HERE/'primary_EPISODES.json')==frozen['episodes_sha256']
        execute(stage)
