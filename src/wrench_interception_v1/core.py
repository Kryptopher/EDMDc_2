"""Known-target direct-wrench interception; no estimator or guidance layer."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,pickle,time,traceback,hashlib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
from Simulation import quad_sim
from PID_Mixer import pid_mixer
from wrench_revisit_mpc import WrenchConfig,WrenchMPC,physical_reference,plant_step
from targets import scenario_family

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)

class CascadedPID:
    """Current p/v/yaw only. Freeze every integrator if any rotor saturates."""
    def __init__(self,sim,gains):
        self.sim=sim;self.gains=gains;self.integral=np.zeros(6)
    def allocate(self,u):
        q=self.sim.quad
        return pid_mixer.fct_allocate_wrench(u,q.kT,q.kD,q.l,max_omega=self.sim.max_speed,prop_efficiency=q.prop_efficiency)
    def command(self,x,p,v,yaw,dt=.01):
        q=self.sim.quad;g=self.gains;before=self.integral.copy()
        ep=p-x[:3];ev=v-x[3:6]
        self.integral[:3]=np.clip(before[:3]+dt*ep,-1.1,1.1)
        def calculate(update_attitude):
            acc=np.asarray(g['kp_pos'])*ep+np.asarray(g['kd_pos'])*ev+np.asarray(g['ki_pos'])*self.integral[:3]
            lateral=np.linalg.norm(acc[:2]);cap=q.g # fixed 45 degree authority
            if lateral>cap:acc[:2]*=cap/lateral
            acc[2]=np.clip(acc[2],-5,10)
            force=q.m*(acc+np.array([0,0,q.g]))
            force[2]=max(force[2],.1*q.m*q.g)
            lateral=np.linalg.norm(force[:2])
            if lateral>force[2]:force[:2]*=force[2]/lateral
            thrust=float(np.linalg.norm(force))
            phi=np.arcsin(np.clip((np.sin(yaw)*force[0]-np.cos(yaw)*force[1])/thrust,-1,1))
            theta=np.arctan2(np.cos(yaw)*force[0]+np.sin(yaw)*force[1],force[2])
            ea=np.arctan2(np.sin(np.array([phi,theta,yaw])-x[6:9]),np.cos(np.array([phi,theta,yaw])-x[6:9]))
            if update_attitude:self.integral[3:]=np.clip(before[3:]+dt*ea,-.1,.1)
            torque=np.diag(q.I)*(np.asarray(g['kp_ang'])*ea-np.asarray(g['kd_ang'])*x[9:12]+np.asarray(g['ki_ang'])*self.integral[3:])
            return np.r_[np.clip(thrust,.05*q.m*q.g,q.m*(q.g+10)),np.clip(torque,[-1.1,-1.1,-.35],[1.1,1.1,.35])]
        u=calculate(True);_,_,_,sat=self.allocate(u)
        held=bool(np.any(sat))
        if held:
            self.integral=before
            u=calculate(False)
        self.last_integral_held=held
        return u

def capture_metrics(states,target,dt=.01):
    sep=np.linalg.norm(states[:,:3]-target[:len(states),:3],axis=1)
    speed=np.linalg.norm(states[:,3:6]-target[:len(states),3:6],axis=1)
    sphere=np.flatnonzero(sep<=.75);eligible=(sep<=.75)&(speed<=3)
    hits=np.flatnonzero(np.convolve(eligible.astype(int),np.ones(21,dtype=int),'valid')==21) if len(states)>=21 else []
    capture=int(hits[0]+20) if len(hits) else None
    entry=int(sphere[0]) if len(sphere) else None
    qualified_entry=np.flatnonzero(eligible)
    return dict(captured=capture is not None,qualified_time_s=capture*dt if capture is not None else None,
        scored_time_s=capture*dt if capture is not None else 20.,
        first_entry_s=entry*dt if entry is not None else None,
        relative_speed_at_first_entry_mps=float(speed[entry]) if entry is not None else None,
        first_joint_region_entry_s=float(qualified_entry[0]*dt) if len(qualified_entry) else None,
        minimum_separation_m=float(np.min(sep[np.isfinite(sep)])),
        position_rmse_m=float(np.sqrt(np.mean((states[:,:3]-target[:len(states),:3])**2))))

def run_episode(task):
    start=time.time();sim=quad_sim();dt=.01
    target=scenario_family(task['family'],np.random.default_rng(task['seed']))
    # Identical extended reference for all arms, independent of selected horizon.
    times=np.arange(2202)*dt;ref=[target.reference(t) for t in times]
    desired=physical_reference(ref,sim,dt)
    x=np.zeros(12);x[8]=desired[0,8]
    cfg=task['config'];arm=task['arm']
    if arm=='PID':controller=CascadedPID(sim,cfg);N=0
    else:
        model=None
        if arm=='W2a':
            with Path(task['model']).open('rb') as f:model=pickle.load(f)
        conf=WrenchConfig(**cfg);N=conf.horizon_steps
        controller=WrenchMPC({'W1':'yaw_linear','W2a':'edmdc','W3':'ltv'}[arm],sim,conf,model=model,initial_yaw=x[8])
    states=[x.copy()];requested=[];applied=[];forces=[];statuses=[];accepted=[];durations=[];sats=[];held=[]
    stop='completed'
    for k in range(round(task['duration']/dt)):
        if arm=='PID':
            tic=time.perf_counter();u=controller.command(x,desired[k,:3],desired[k,3:6],desired[k,8]);elapsed=time.perf_counter()-tic
            info=dict(status='no_qp',accepted=True);held.append(controller.last_integral_held)
        else:
            u,info=controller.compute(x,desired[k+1:k+N+1]);elapsed=info['elapsed_s'];held.append(False)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):xn,actual,f,sat=plant_step(sim,x,u,dt)
        requested.append(u.copy());applied.append(actual);forces.append(f);sats.append(bool(np.any(sat)))
        statuses.append(info['status']);accepted.append(info['accepted']);durations.append(elapsed);states.append(xn.copy())
        if arm!='PID':controller.previous_applied=actual.copy()
        x=xn
        if not np.isfinite(x).all():stop='nonfinite_state';break
        if np.max(np.abs(x[6:8]))>np.deg2rad(80):stop='euler_attitude_envelope_80deg';break
        if np.linalg.norm(x[:3]-desired[k+1,:3])>50:stop='position_error_over_50m';break
    states=np.asarray(states);requested=np.asarray(requested);applied=np.asarray(applied)
    finite=np.all(np.isfinite(states),axis=1)
    # Only a nonfinite final sample can be omitted from descriptive prefixes; preserved in trace.
    valid_n=len(states) if finite.all() else int(np.flatnonzero(~finite)[0])
    metrics=capture_metrics(states[:valid_n],desired)
    row={k:task[k] for k in ('case_id','family','seed','arm','candidate_id','stage')}
    row.update(metrics,completed=stop=='completed',stop_reason=stop,simulated_seconds=(len(states)-1)*dt,
        horizon_steps=N,solver_failures=None if arm=='PID' else int(np.count_nonzero(~np.asarray(accepted))),
        allocator_altered_steps=int(np.count_nonzero(np.max(np.abs(requested-applied),axis=1)>1e-7)),
        rotor_saturated_steps=int(np.count_nonzero(sats)),integrator_held_steps=int(np.count_nonzero(held)),
        wall_seconds=time.time()-start,reference_sha256=hashlib.sha256(desired.tobytes()).hexdigest(),
        task_sha256=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest())
    path=Path(task['output'])/task['arm']/task['candidate_id']/task['case_id'];path.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(path/'trace.npz',time=np.arange(len(states))*dt,states=states,reference=desired[:len(states)],
        U_requested=requested,U=applied,motor_forces=np.asarray(forces),solver_status=np.asarray(statuses),solver_accepted=np.asarray(accepted),
        solve_seconds=np.asarray(durations),rotor_saturated=np.asarray(sats),integrator_held=np.asarray(held))
    row['trace']=str(path/'trace.npz');dump(path/'metrics.json',row)
    return row

def safe_episode(task):
    try:return run_episode(task)
    except Exception:
        path=Path(task['output'])/'exceptions'/f"{task['arm']}_{task['candidate_id']}_{task['case_id']}.json"
        dump(path,dict(task=task,error=traceback.format_exc()))
        return {**{k:task[k] for k in ('case_id','family','seed','arm','candidate_id','stage')},'completed':False,'captured':False,'scored_time_s':20.,'stop_reason':'execution_exception','exception':str(path),'position_rmse_m':None,'solver_failures':None,'rotor_saturated_steps':None,'allocator_altered_steps':None}
