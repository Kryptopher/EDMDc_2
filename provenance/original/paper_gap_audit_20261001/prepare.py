"""Read-only evidence inventory and prospective freeze; never execute controllers."""
from pathlib import Path
import json,hashlib,csv,subprocess,datetime,collections
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def walk(x):
    if isinstance(x,dict):
        yield x
        for v in x.values():yield from walk(v)
    elif isinstance(x,list):
        for v in x:yield from walk(v)
def main():
    assert not (HERE/'PROTOCOL.json').exists(),'Never overwrite a freeze'
    dirs=['artifacts/wrench_controller_tuning_v1','artifacts/wrench_feedforward_amendment_F','artifacts/wrench_stage_r_addendum_P_v1','artifacts/wrench_stage_r_addendum_P2_v1','artifacts/wrench_stage_r_addendum_O_v1','artifacts/wrench_addendum_OE_v1','wrench_stage_r_controller_mismatch_v1','overnight_campaign_20260929/continuation_20260930','paper_prep_20260928/heldout_additions_v1','paper_prep_20260928/frozen_tracking_configs']
    inventory=[];existing_seeds=set();seed_sources=[];summaries=[]
    # Read saved summaries, aggregate episode lists and manifests, never the held-out raw identification NPZs.
    for d in dirs:
        for p in sorted((ROOT/d).rglob('*')):
            if not p.is_file() or p.suffix.lower() not in ['.json','.md']:continue
            if any(q in p.parts for q in ['traces','source_snapshot','screening','validation','timing_pass']):continue
            if p.suffix=='.md':
                p.read_text(encoding='utf-8-sig');inventory.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)});continue
            if p.stat().st_size>50_000_000:continue
            obj=read(p);seeds={v for r in walk(obj) for k,v in r.items() if k in ['seed','base_seed','lhs_seed'] and isinstance(v,int)}
            if seeds:existing_seeds.update(seeds);seed_sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'seeds':sorted(seeds)})
            if p.name=='episodes.json' and isinstance(obj,list) and obj and isinstance(obj[0],dict):
                groups=collections.defaultdict(list)
                for r in obj:groups[str(r.get('arm',r.get('controller','unknown')))].append(r)
                table={}
                for arm,rows in groups.items():
                    vals=[r['position_rmse_m'] for r in rows if r.get('completed') and r.get('position_rmse_m') is not None]
                    table[arm]={'recorded':len(rows),'completed':sum(bool(r.get('completed')) for r in rows),'mean_position_rmse_m':sum(vals)/len(vals) if vals else None,'solver_failures':sum(r.get('solver_failures',0) or 0 for r in rows),'allocator_altered_steps':sum(r.get('allocator_altered_steps',0) or 0 for r in rows)}
                summaries.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'summary':table})
            if any(t in p.name.lower() for t in ['protocol','selection','manifest','freeze','selected','gate']):inventory.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    for p in (ROOT/'handoff_20261001').glob('*'):
        if p.name in ['HANDOFF.md','REPORT_INDEX.md','CURRENT_NUMBERS.json','ALL_REPORTS.md']:
            p.read_text(encoding='utf-8-sig');inventory.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    # Tuning scenario seeds also occur in top-level frozen protocols; their contents were included above.
    source_paths=['wrench_revisit_mpc.py','feedforward_wrench_mpc.py','analytic_feedforward_reference.py','Simulation.py','quadcopter.py','PID_Mixer.py','edmdc_mpc.py','EDMDc_training.py','wrench_interception_v1/targets.py','wrench_interception_v1/core.py','paper_gap_audit_20261001/runtime.py','paper_gap_audit_20261001/cost_reference_mpc.py']
    source_hashes={p:sha(ROOT/p) for p in source_paths}
    selections={a:ROOT/f'artifacts/wrench_controller_tuning_v1/selected_configs/{a}.json' for a in ['W1','W2a','W3']}
    selections['W1f']=ROOT/'artifacts/wrench_feedforward_amendment_F/selected_configs/W1f.json'
    configs={a:read(p)['selected_candidate']['config'] for a,p in selections.items()}
    assert configs['W1']==configs['W2a']
    inter=ROOT/'overnight_campaign_20260929/continuation_20260930/interception/selection.json'
    bundles={a:read(inter)['selected'][a][0]['config'] for a in ['W1','W2a']}
    model=ROOT/'artifacts/wrench_revisit_2026-09-21/training_v2/models/wrench_physics42_dt001.pkl';modelhash=sha(model)
    assert modelhash=='090b8dd3106dcf3646dcf991fe23c030d3fe27435443a5f8fae19c1c377b2db4'
    tasks=[];families=['helix','figure8','lissajous','waypoint'];targets=['straight','accelerating','helix','weaving']
    def task(group,phase,cell,condition,family,seed,yaw,config,predictor,controller,duration=30.,ms=1.,ins=1.,inp='hover'):
        case=f'{family}_{seed}_{yaw}';tid=f'{group}_{cell}_{condition}_{case}'
        t=dict(task_id=tid,group=group,phase=phase,cell=cell,condition=condition,case_id=case,family=family,seed=seed,yaw_mode=yaw,config=config,predictor=predictor,controller=controller,duration=duration,mass_scale=ms,inertia_scale=ins,input_reference=inp)
        if predictor=='edmdc':t.update(model=str(model),model_sha256=modelhash)
        tasks.append(t)
    for phase,n,base in [('smoke',1,3104000000),('evaluation',10,3101000000)]:
        for fi,f in enumerate(families):
            for yi,y in enumerate(['fixed','path'] if phase=='evaluation' else ['path']):
                for i in range(n):
                    seed=base+fi*1000+yi*100+i
                    for arm,pred in [('W1','yaw_linear'),('W2a','edmdc')]:
                        for inp in ['hover','analytic_cost_only']:task('A',phase,arm+'_'+inp,'nominal',f,seed,y,configs['W1'],pred,'cost_reference',30. if phase=='evaluation' else 4.,inp=inp)
    conditions=[('nominal',1.,1.),('mass_plus15',1.15,1.),('mass_minus15',.85,1.),('inertia_plus20',1.,1.2)]
    for phase,n,base in [('smoke',1,3104100000),('evaluation',5,3102000000)]:
        for fi,f in enumerate(families if phase=='evaluation' else ['helix']):
            for yi,y in enumerate(['fixed','path'] if phase=='evaluation' else ['fixed']):
                for i in range(n):
                    seed=base+fi*1000+yi*100+i
                    for cond,ms,ins in conditions:
                        for arm,pred in [('W1','yaw_linear'),('W2a','edmdc'),('W3','ltv'),('W1f','yaw_linear')]:task('B',phase,arm,cond,f,seed,y,configs[arm],pred,'W1f' if arm=='W1f' else 'original',30. if phase=='evaluation' else 4.,ms,ins)
    for phase,n,base in [('smoke',1,3104200000),('evaluation',10,3103000000)]:
        for fi,f in enumerate(targets):
            for i in range(n):
                for bundle,config in bundles.items():
                    for arm,pred in [('W1','yaw_linear'),('W2a','edmdc')]:task('C',phase,arm+'_bundle_'+bundle,'nominal',f,base+fi*1000+i,'known_future',config,pred,'original',20.)
    for arm,pred in [('W1','yaw_linear'),('W2a','edmdc')]:
        for ctrl in ['original','cost_reference']:task('A','replay',arm+'_'+ctrl,'nominal','helix',3105000000,'path',configs['W1'],pred,ctrl,2.)
    assert len(tasks)==1172
    fresh={t['seed'] for t in tasks};assert not (fresh&existing_seeds),fresh&existing_seeds
    for t in tasks:assert t['seed']>=3101000000
    save(HERE/'EXISTING_EVIDENCE.json',{'inventory':inventory,'saved_episode_recomputation':summaries,'seed_sources':seed_sources,'existing_seeds':sorted(existing_seeds),'new_seed_intersection':[]})
    save(HERE/'TASKS.json',tasks)
    protocol={
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Post-hoc paper-gap audit; new prospective fixed evaluations, no tuning/model adoption/hardware; historical results unchanged.',
      'decisions':{'A':'RUN: F independently tunes predictors and changes reference/error formulation; it is not a common-configuration 2x2 intervention.','B':'RUN: Stage R and P change controller assumptions; no completed same-model matched physical mass/inertia block was found.','C':'RUN: cross frozen bundles to test predictor-specific interpretation; existing independently tuned application alone only supports descriptive claims.','dictionary':'SKIP: nominal held-out random42 comparison already exists; narrow causal closed-loop dictionary claims.','timing':'SKIP: reuse original nominal42 full-update isolated evidence, QP-only was not saved; new cost-reference/controller variants have no timing claim.'},
      'model':{'path':str(model),'sha256':modelhash,'observables':42,'input_features':10,'lambda':0},
      'config_sources':{**{a:{'path':str(p),'sha256':sha(p),'config':configs[a]} for a,p in selections.items()},'interception':{'path':str(inter),'sha256':sha(inter),'bundles':bundles}},
      'code_sha256':source_hashes,'git_head':subprocess.check_output(['git','-c','safe.directory=D:/drone_capture/edmdc/quadcopter_EDMD','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'tasks_file':'TASKS.json','tasks_sha256':sha(HERE/'TASKS.json'),'evidence_sha256':sha(HERE/'EXISTING_EVIDENCE.json'),
      'counts':{'A_evaluation':320,'B_evaluation':640,'C_evaluation':160,'smoke':48,'equivalence_replay':4,'planned_total':1172,'hard_limit':1200,'automatic_retries':0},
      'seeds':'Exact per-task seeds in TASKS.json. Disjoint from all integer seed/base_seed/lhs_seed values in archived aggregate manifests/protocols/results inventoried in EXISTING_EVIDENCE.json. No held-out identification traces reopened.',
      'A_intervention':'Only replace the hover wrench in the absolute-command R cost by analytical u_ref[k:k+N]. No additive command, nominal-rollout replacement, reference-defect cancellation, initial guess change, feedforward clipping or weights change. All four cells use identical analytical state references and N75/b5 c23.',
      'tracking_reference':'Existing analytic_tracking_reference at duration31.01s, speed1.75, common analytical 12-state desired trajectory; unwrap desired yaw once. Initial true position/velocity/yaw equal desired, roll/pitch/body rates zero. Same arrays across cells/physical conditions. References differ from the old numerically differentiated baseline; do not attach old numbers to new trajectories.',
      'B':'Nominal controller model/hover/limits fixed; change only physical quad.m or quad.I. Motor speed, kT/kD and allocation unchanged. W1/W2a/W3 frozen primary N75; W1f frozen F c17 N50/b2. Within-arm degradation paired; between-arm W1f comparison is as-selected, not common config.',
      'C':'Original scripted target distributions, exact state and known future trajectory; rest initial state and target-reference yaw. Frozen W1 N50/b10 and W2a N100/b5 entire bundles crossed. Distance<=0.75 and relative speed<=3 for21 samples; end-of-dwell qualification; noncapture20s. Original semantics retain a capture before a later abort, with both explicitly reported.',
      'metrics':{'tracking':'sqrt(mean(error_coordinates_and_time^2)), positions m, Euler attitude radians (all three wrapped), yaw radians wrapped. Mean/median/P95/max over completed episode metrics. Record all failures separately. Prefix RMSE never counted as successful full-episode RMSE.','C':'capture count, scored qualification time including20s noncapture, first sphere entry conditional on entry, speed at entry, minimum separation, completion/envelope aborts, solver failures.','constraints':'Requested wrench bounds and applied rotor margin violation tolerance1e-5; allocator requested-applied max component>1e-7; optimizer acceptance retains original1e-4.','pairing':'Scenario case_id within group, same reference hashes required. Continuous tracking paired stats only mutually completed flights, report excluded failures; no equivalence inference from nonsignificance.','summary':'mean, median, numpy default linear P95, max; family/heading subgroups descriptive.','bootstrap':'20000 paired scenario bootstrap replicates, 95% percentile CI, numpy default_rng seed2026100101 independently reset per contrast. Exact two-sided binomial sign-test excludes exact ties. No multiplicity-adjusted superiority claim.','contrasts':['A W2a-W1 within each input reference','A analytic-hover within each predictor','A paired difference-in-differences','B each perturbed-nominal within each arm','B W2a-W1,W2a-W3,W2a-W1f each condition','C W2a-W1 each bundle for penalized time and capture','C bundleW2a-bundleW1 each predictor']},
      'failure_handling':'Preserve all exceptions, nonfinite states, >80deg roll/pitch and >50m tracking error aborts. No replacement scenarios, retries or retuning. If smoke indicates execution/implementation invalidity, halt affected group; failed control robustness alone remains an outcome. For C captures and completion both reported.',
      'smoke_gate':'48 designated smoke episodes plus4 original-vs-isolated-hover replays. Require no execution exceptions/nonfinite state in smoke, exact equality of states/applied commands/solver accepted flags for the two replay pairs; confirm physical-only parameter changes and paired reference identity. Envelope aborts retained and do not permit retuning. No evaluation opens until gate and evaluation freeze saved.',
      'workers':8,'BLAS_threads_per_worker':1,'timing':'No timing benchmark during parallel work; wall durations are workload diagnostics only.'}
    save(HERE/'PROTOCOL.json',protocol);(HERE/'PROTOCOL.sha256').write_text(sha(HERE/'PROTOCOL.json')+'  PROTOCOL.json\n')
    print(json.dumps({'tasks':len(tasks),'protocol_sha256':sha(HERE/'PROTOCOL.json'),'archived_unique_seeds_checked':len(existing_seeds),'episode_files':len(summaries)},indent=2))
if __name__=='__main__':main()
