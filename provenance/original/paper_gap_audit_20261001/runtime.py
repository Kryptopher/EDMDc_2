"""Isolated paper-gap evaluation. No writes to prior campaigns or hardware imports."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,pickle,time,hashlib,traceback,uuid
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import numpy as np
from Simulation import quad_sim
from PID_Mixer import pid_mixer
from wrench_revisit_mpc import WrenchConfig,WrenchMPC,plant_step,physical_reference
from cost_reference_mpc import WrenchMPC as CostReferenceMPC
from feedforward_wrench_mpc import FeedforwardWrenchMPC
from analytic_feedforward_reference import analytic_tracking_reference,feedforward_wrench
import importlib.util
spec=importlib.util.spec_from_file_location('gap_target_definitions',ROOT/'wrench_interception_v1/targets.py')
targets=importlib.util.module_from_spec(spec);spec.loader.exec_module(targets)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    for i in range(20):
        try:os.replace(tmp,p);return
        except PermissionError:
            if i==19:raise
            time.sleep(min(.025*(i+1),.5))

def capture(states,ref,complete):
    sep=np.linalg.norm(states[:,:3]-ref[:len(states),:3],axis=1)
    speed=np.linalg.norm(states[:,3:6]-ref[:len(states),3:6],axis=1)
    sphere=np.flatnonzero(sep<=.75);eligible=(sep<=.75)&(speed<=3)
    hit=np.flatnonzero(np.convolve(eligible.astype(int),np.ones(21,dtype=int),'valid')==21) if len(states)>=21 else []
    end=int(hit[0]+20) if len(hit) else None;entry=int(sphere[0]) if len(sphere) else None
    return dict(captured=end is not None,qualified_time_s=None if end is None else end*.01,
        scored_time_s=20. if end is None else end*.01,
        first_entry_s=None if entry is None else entry*.01,
        relative_speed_at_first_entry_mps=None if entry is None else float(speed[entry]),
        minimum_separation_m=float(np.min(sep)),capture_before_later_abort=bool(end is not None and not complete))

def run_episode(task):
    started=time.time();cfg=WrenchConfig(**task['config']);dt=cfg.dt;N=cfg.horizon_steps
    nominal=quad_sim();plant=quad_sim()
    plant.quad.m*=task.get('mass_scale',1.);plant.quad.I=np.array(plant.quad.I)*task.get('inertia_scale',1.)
    assert plant.max_speed==nominal.max_speed and plant.quad.kT==nominal.quad.kT and plant.quad.kD==nominal.quad.kD
    if task['group']=='C':
        target=targets.scenario_family(task['family'],np.random.default_rng(task['seed']))
        ref=[target.reference(t) for t in np.arange(2202)*dt];desired=physical_reference(ref,nominal,dt);uref=None
    else:
        # Duration held at 31.01 for every arm/condition, including smoke prefixes.
        ref=analytic_tracking_reference(task['family'],task['seed'],31.01,1.75,task['yaw_mode'])
        desired,uref,_=feedforward_wrench(ref,nominal.quad)
        desired[:,8]=np.unwrap(desired[:,8])
    x=np.zeros(12)
    if task['group']!='C':x[:6]=desired[0,:6]
    x[8]=desired[0,8]
    model=None
    if task['predictor']=='edmdc':
        assert digest(task['model'])==task['model_sha256']
        with Path(task['model']).open('rb') as f:model=pickle.load(f)
    if task['controller']=='W1f':controller=FeedforwardWrenchMPC('linear_error_ff',nominal,cfg)
    elif task['controller']=='cost_reference':controller=CostReferenceMPC(task['predictor'],nominal,cfg,model,initial_yaw=x[8])
    else:controller=WrenchMPC(task['predictor'],nominal,cfg,model,initial_yaw=x[8])
    initial=x.copy();states=[x.copy()];requested=[];applied=[];forces=[];accepted=[];statuses=[];saturated=[]
    stop='completed';count=int(round(task['duration']/dt))
    for k in range(count):
        future=desired[k+1:k+N+1]
        if task['controller']=='W1f':u,info=controller.compute(x,future,uref[k:k+N],current_reference=desired[k])
        elif task['controller']=='cost_reference':
            u,info=controller.compute(x,future,input_reference=uref[k:k+N] if task['input_reference']=='analytic_cost_only' else None)
        else:u,info=controller.compute(x,future)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):xn,actual,f,sat=plant_step(plant,x,u,dt)
        requested.append(u.copy());applied.append(actual.copy());forces.append(f);accepted.append(info['accepted']);statuses.append(info['status']);saturated.append(bool(np.any(sat)))
        states.append(xn.copy());controller.previous_applied=actual.copy();x=xn
        if not np.isfinite(x).all():stop='nonfinite_state';break
        if np.max(np.abs(x[6:8]))>np.deg2rad(80):stop='euler_attitude_envelope_80deg';break
        if np.linalg.norm(x[:3]-desired[k+1,:3])>50:stop='position_error_over_50m';break
    states=np.array(states);req=np.array(requested);u=np.array(applied);f=np.array(forces)
    finite=np.all(np.isfinite(states),axis=1);valid=int(np.flatnonzero(~finite)[0]) if not finite.all() else len(states)
    err=states[:valid]-desired[:valid];err[:,6:9]=np.arctan2(np.sin(err[:,6:9]),np.cos(err[:,6:9]));complete=stop=='completed'
    force_max=np.asarray(controller.force_max,dtype=float)
    bad=np.any((req<controller.umin-1e-5)|(req>controller.umax+1e-5),axis=1)
    bad|=np.any((f<1e-4-1e-5)|(f>force_max-1e-4+1e-5),axis=1)
    row={k:task[k] for k in ['task_id','group','phase','cell','condition','case_id','family','seed','yaw_mode']}
    row.update(completed=complete,stop_reason=stop,simulated_seconds=(len(states)-1)*dt,
        position_rmse_m=float(np.sqrt(np.mean(err[:,:3]**2))) if complete else None,
        velocity_rmse_mps=float(np.sqrt(np.mean(err[:,3:6]**2))) if complete else None,
        attitude_rmse_rad=float(np.sqrt(np.mean(err[:,6:9]**2))) if complete else None,
        yaw_rmse_rad=float(np.sqrt(np.mean(err[:,8]**2))) if complete else None,
        prefix_position_rmse_m=float(np.sqrt(np.mean(err[:,:3]**2))),
        solver_failures=int(np.count_nonzero(~np.array(accepted))),constraint_violations=int(np.count_nonzero(bad)),
        allocator_altered_steps=int(np.count_nonzero(np.max(np.abs(req-u),axis=1)>1e-7)),rotor_saturated_steps=int(np.count_nonzero(saturated)),
        actual_max_abs_tilt_deg=float(np.degrees(np.max(np.abs(states[:valid,6:8])))),
        reference_sha256=hashlib.sha256(desired.tobytes()).hexdigest(),initial_state=initial.tolist(),
        plant_mass=float(plant.quad.m),plant_inertia=np.diag(plant.quad.I).tolist(),controller_mass=float(nominal.quad.m),
        controller_inertia=np.diag(nominal.quad.I).tolist(),rotor_force_max=force_max.tolist(),
        umin=controller.umin.tolist(),umax=controller.umax.tolist(),wall_seconds=time.time()-started,
        config_sha256=hashlib.sha256(json.dumps(task['config'],sort_keys=True).encode()).hexdigest(),
        task_sha256=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest())
    if task['group']=='C':row.update(capture(states[:valid],desired,complete))
    path=HERE/'runs'/task['phase']/task['task_id'];path.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(path/'trace.npz',states=states,reference=desired[:len(states)],U_requested=req,U=u,motor_forces=f,
        solver_status=np.array(statuses),solver_accepted=np.array(accepted),rotor_saturated=np.array(saturated),time=np.arange(len(states))*dt)
    row['trace']=str(path/'trace.npz');dump(path/'metrics.json',row);return row

def safe_episode(task):
    try:return run_episode(task)
    except Exception:
        row={k:task[k] for k in ['task_id','group','phase','cell','condition','case_id','family','seed','yaw_mode']}
        row.update(completed=False,stop_reason='execution_exception',exception=traceback.format_exc(),position_rmse_m=None)
        if task['group']=='C':row.update(captured=False,scored_time_s=20.)
        dump(HERE/'runs'/task['phase']/task['task_id']/'exception.json',row);return row
