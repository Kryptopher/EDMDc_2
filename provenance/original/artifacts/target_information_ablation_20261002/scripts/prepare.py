"""Certify exact saved scenarios and freeze the design; execute no controller."""
from runtime import *
import importlib.util,dataclasses,platform

def main():
    assert not (HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.json').exists(),'Never overwrite a freeze'
    gap=ROOT/'paper_gap_audit_20261001';old=read(gap/'PROTOCOL.json');correction=read(gap/'SMOKE_CORRECTION_FREEZE.json')['code_overrides']
    sources=['Simulation.py','quadcopter.py','PID_Mixer.py','wrench_revisit_mpc.py','wrench_interception_v1/targets.py','wrench_interception_v1/core.py','paper_gap_audit_20261001/runtime.py']
    hashes={p:sha(ROOT/p) for p in sources}
    for p,h in hashes.items():assert h==correction.get(p,old['code_sha256'][p]),p
    selection=ROOT/'overnight_campaign_20260929/continuation_20260930/interception/selection.json'
    assert sha(selection)==read(selection.with_name('selection.sha256.json'))['sha256']
    selected=read(selection)['selected'];configs={a:selected[a][0]['config'] for a in ['W1','W2a']}
    assert selected['W1'][0]['candidate_id']=='c04' and selected['W2a'][0]['candidate_id']=='c22'
    assert (configs['W1']['horizon_steps'],configs['W1']['block_steps'])==(50,10)
    assert (configs['W2a']['horizon_steps'],configs['W2a']['block_steps'])==(100,5)
    model=ROOT/'artifacts/wrench_revisit_2026-09-21/training_v2/models/wrench_physics42_dt001.pkl'
    assert sha(model)=='090b8dd3106dcf3646dcf991fe23c030d3fe27435443a5f8fae19c1c377b2db4'
    saved=read(gap/'C_EPISODES.json');tasks=read(gap/'TASKS.json');ct=[t for t in tasks if t['group']=='C' and t['phase']=='evaluation']
    assert len(saved)==len(ct)==160
    spec=importlib.util.spec_from_file_location('frozen_target_generator',ROOT/'wrench_interception_v1/targets.py');targets=importlib.util.module_from_spec(spec);sys.modules[spec.name]=targets;spec.loader.exec_module(targets)
    sim=quad_sim();cases=[];arrays={};seeds=sorted({r['seed'] for r in saved});assert len(seeds)==40
    for seed in seeds:
        rows=[r for r in saved if r['seed']==seed];ts=[t for t in ct if t['seed']==seed];assert len(rows)==len(ts)==4
        family=rows[0]['family'];target=targets.scenario_family(family,np.random.default_rng(seed))
        truth=physical_reference([target.reference(t) for t in np.arange(2202)*.01],sim,.01)
        history=np.asarray([np.r_[target.position(j*.01),target.velocity(j*.01)] for j in range(-25,2000)])
        assert np.array_equal(history[25:],truth[:2000,:6])
        x=np.zeros(12);x[8]=truth[0,8]
        for row,task in zip(sorted(rows,key=lambda x:x['cell']),sorted(ts,key=lambda x:x['cell'])):
            assert row['case_id']==task['case_id'] and task['duration']==20 and task['mass_scale']==task['inertia_scale']==1
            assert task['config']==configs['W1' if task['cell'].endswith('bundle_W1') else 'W2a']
            assert row['reference_sha256']==arrsha(truth) and np.array_equal(row['initial_state'],x)
            assert row['plant_mass']==sim.quad.m and row['plant_inertia']==np.diag(sim.quad.I).tolist()
            with np.load(tracepath(row['trace'])) as z:
                assert np.array_equal(z['states'][0],x) and np.array_equal(z['reference'],truth[:len(z['states'])])
                m=metrics(z['states'],truth)
                for key in ['captured','qualified_time_s','scored_time_s']:assert m[key]==row[key]
        params={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in dataclasses.asdict(target).items()}
        cases.append({'case_id':rows[0]['case_id'],'family':family,'seed':seed,'initial_state':x.tolist(),'reference_sha256':arrsha(truth),'target_posvel_20s_sha256':arrsha(truth[:2001,:6].copy()),'measurement_history_sha256':arrsha(history),'target_generator_parameters_simulator_only':params,'oracle_task_ids':[r['task_id'] for r in rows]})
        arrays[f'truth_{seed}']=truth;arrays[f'history_{seed}']=history
    np.savez_compressed(HERE/'SCENARIOS.npz',**arrays)
    cells={'Linear-A':('yaw_linear','W1'),'EDMDc-B':('edmdc','W2a'),'EDMDc-A':('edmdc','W1'),'Linear-B':('yaw_linear','W2a')}
    modes=[('cv',0),('delayed_cv_100ms',10),('delayed_cv_250ms',25)]
    ledger=[]
    def add(stage,cell,mode,delay,case):
        pred,bundle=cells[cell];ledger.append({'task_id':f"{stage}_{cell}_{mode}_{case['seed']}",'stage':stage,'cell':cell,'predictor':pred,'bundle':bundle,'mode':mode,'delay_samples':delay,'case_id':case['case_id'],'family':case['family'],'seed':case['seed'],'config':configs[bundle]})
    for cell in ['Linear-A','EDMDc-B']:
        add('oracle_reproduction',cell,'oracle',0,cases[0])
        for mode,delay in modes:add('smoke',cell,mode,delay,cases[20])
        add('smoke',cell,'position_hold',0,cases[20])
        for mode,delay in modes:
            for case in cases:add('primary',cell,mode,delay,case)
        for case in cases:add('position_only',cell,'position_hold',0,case)
    for cell in ['EDMDc-A','Linear-B']:
        for mode,delay in modes:
            for case in cases:add('secondary',cell,mode,delay,case)
    assert collections_counts(ledger)=={'oracle_reproduction':2,'smoke':8,'primary':240,'position_only':80,'secondary':240}
    dump(HERE/'TASKS.json',ledger)
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'classification':'Post-hoc frozen target-information ablation on previously opened crossed-study scenarios; descriptive evidence, not a fresh confirmatory scenario set.',
              'NO_RETUNING':True,'ORACLE_RESULTS_REUSED':True,'oracle_reproduction_only':2,'configurations':configs,'configuration_sha256':{a:objsha(c) for a,c in configs.items()},'controller_identities':cells,
              'selection_path':str(selection),'selection_sha256':sha(selection),'edmdc_model':{'path':str(model),'sha256':sha(model),'state_observables':42,'state_dependent_input_features':10,'physical_inputs':4,'recursion':'initial lift then latent propagation; unchanged'},
              'scenarios':cases,'scenarios_archive_sha256':sha(HERE/'SCENARIOS.npz'),'source_sha256':hashes,'script_sha256':{f'scripts/{name}':sha(HERE/'scripts'/name) for name in ['runtime.py','prepare.py','run.py']},'oracle_episodes_path':str(gap/'C_EPISODES.json'),'oracle_episodes_sha256':sha(gap/'C_EPISODES.json'),'original_tasks_sha256':sha(gap/'TASKS.json'),'tasks_sha256':sha(HERE/'TASKS.json'),
              'planned_counts':collections_counts(ledger),'planned_total':len(ledger),'stage_order':['oracle_reproduction and smoke gate','primary 240','freeze and report primary','secondary 240 if inexpensive','position_only 80 exploratory'],
              'dt':.01,'duration_seconds':20,'capture':{'distance_le_m':.75,'relative_speed_le_mps':3,'consecutive_samples':21,'qualification':'end of first qualifying dwell','noncapture_score_seconds':20,'uses':'true target position and velocity'},
              'aborts':{'abs_roll_pitch_gt_degrees':80,'position_error_gt_m':50,'nonfinite_state':True,'position_error_reference':'true target, not forecast','capture_before_later_abort':'retain capture plus report abort separately'},
              'plant':{'mass':float(sim.quad.m),'inertia':np.asarray(sim.quad.I).tolist(),'g':float(sim.quad.g),'kT':float(sim.quad.kT),'kD':float(sim.quad.kD),'arm_length':float(sim.quad.l),'linear_drag':float(sim.quad.k_drag_linear),'angular_drag':float(sim.quad.k_drag_angular),'prop_efficiency':np.asarray(sim.quad.prop_efficiency).tolist(),'max_motor_speed':float(sim.max_speed),'integration':'unchanged nominal nonlinear 12-state RK4 plant_step and allocator'},
              'information_modes':{'oracle':'Saved exact scripted future physical reference; reuse all160 saved results.','cv':'Only p(k),v(k). p_hat(k+i|k)=p(k)+i*dt*v(k), v_hat=v(k).','delayed_cv_100ms':'Only p(k-10),v(k-10). p_hat=p(k-10)+(10+i)*dt*v(k-10).','delayed_cv_250ms':'Only p(k-25),v(k-25). p_hat=p(k-25)+(25+i)*dt*v(k-25).','heading':'atan2 of available measurement velocity; hold preceding available heading if horizontal speed<=1e-8; unwrap relative to preceding causal heading. Never use future true heading.','physical_reference':'Forecast acceleration zero; unchanged physical_reference maps forecast velocity and zero acceleration to drag-compensating attitude; forecast body rates derived only from forecast sequence. No controller weights changed.','position_hold':'Separate exploratory stationary-target/pure-pursuit reference: measured current position held; desired velocity zero (not an estimate of actual target velocity); desired roll/pitch/body rates zero; yaw fixed to initial interceptor yaw. All weights retained. No position-only cost exists; suppressing velocity weights would change selected objective, so is not done.','delayed_hold':'Not planned; optional variant omitted.'},
              'startup_delay':'Simulator preloads target position/velocity measurements at -0.25s through -0.01s. At k=0 the delayed policy receives precisely sample -d, without any newer sample. History is simulator-owned and not exposed to policy. Initial interceptor state remains unchanged; initial yaw is the original scenario state, not a new future measurement.',
              'leakage_guards':'Pure policy signature contains only immutable Measurement, current sample, horizon, delay, mode, preceding causal heading and fixed nominal quad parameters. Sensor asserts exact sample=k-delay and logs every access. Policy never receives scenario seed, generator, ground-truth arrays or parameters. Forecasts are saved every50 updates. Capture/abort evaluator accepts true reference independently.',
              'smoke_gate':{'oracle':'Full20s two selected-controller replays on first straight seed; reference and initial state exact, accepted flags identical, max state/applied wrench difference<=1e-8. No numerical adjustment if gate fails.','causality':'Unit checks verify all three causal equations, delay ages including startup, forbidden-newer-history poisoning invariance, and position-only packet velocity absent. Eight smoke episodes on first helix seed; control-envelope aborts are retained but do not permit tuning; execution/nonfinite/leakage errors stop evaluation.','scoring':'Dwell rescored against true reference; predicted reference never supplied to evaluator.'},
              'analysis_plan':{'aggregates':'Capture, full completion, aborts, solver failures, allocator events; mean/median successful qualification and sphere entry, entry speed, minimum separation, per-episode maximum absolute roll/pitch; repeat by family.','paired':'EDMDc-B minus Linear-A for each mode; successful-time comparisons only joint captures. Separately score every scenario, noncapture=20. Within each controller use later information condition minus earlier: oracle->CV->100ms->250ms.','bootstrap':'20000 paired iid scenario resamples, percentile95 CI, seed2026100201; conditional comparisons resample joint-capture pairs only. Descriptive post-hoc intervals.','wins':'Negative difference is second-arm/EDMDc improvement; abs(diff)<=1e-12 tie.','representative':'CurrentCV both captured: choose absolute paired-time difference nearest median; ties by seed. If none, no successful representative is fabricated.','figures':'After results frozen, capture by condition; conditional and penalized time in separate panels; family outcomes; deterministic CurrentCV trace with true separation/speed.','secondary':'Only after primary report and result freeze exist. No tuning/selection on primary results. Position-only reported separately and excluded from main CV comparisons.'},'workers':4,'BLAS_threads_per_worker':1,'automatic_retries':0,'environment':{'python':sys.version,'platform':platform.platform()}}
    dump(HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.json',protocol)
    (HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.sha256').write_text(sha(HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.json')+'  TARGET_INFORMATION_ABLATION_PROTOCOL.json\n')
    print(json.dumps({'protocol_sha256':sha(HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.json'),'all160_scenario_prefixes_exact':True,'counts':protocol['planned_counts']},indent=2),flush=True)

def collections_counts(ledger):
    import collections
    return dict(collections.Counter(t['stage'] for t in ledger))

if __name__=='__main__':main()
