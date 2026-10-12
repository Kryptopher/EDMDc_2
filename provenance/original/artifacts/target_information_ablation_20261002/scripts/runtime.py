"""Isolated frozen-controller target-information intervention. No tuning."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys,json,pickle,hashlib,datetime,traceback
from dataclasses import dataclass
import numpy as np
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from Simulation import quad_sim
from wrench_revisit_mpc import WrenchMPC,WrenchConfig,physical_reference,plant_step

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def objsha(a):return hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)
def tracepath(p):
    f=Path(p)
    return f if f.exists() else ROOT/str(p).replace('\\','/').split('EDMDC_MPC/')[-1]

@dataclass(frozen=True)
class Measurement:
    sample:int
    position:tuple
    velocity:tuple|None

class CausalSensor:
    """Simulator-owned buffer; policy only receives one immutable measurement."""
    def __init__(self,history,delay,position_only=False):
        self.__history=history
        self.delay=delay;self.position_only=position_only;self.reads=[]
    def receive(self,k):
        j=k-self.delay
        assert j<=k-self.delay and j>=-25
        pv=self.__history[j+25]
        self.reads.append((k,j))
        return Measurement(j,tuple(pv[:3]),None if self.position_only else tuple(pv[3:6]))

def causal_reference(packet,k,N,mode,delay,previous_yaw,sim):
    """Pure reference builder: no target, trajectory array, seed or generator argument."""
    assert isinstance(packet,Measurement)
    assert packet.sample==k-delay
    p=np.asarray(packet.position,dtype=float)
    if mode=='position_hold':
        assert packet.velocity is None and delay==0
        ref=np.zeros((N,12));ref[:,:3]=p;ref[:,8]=previous_yaw
        return ref,previous_yaw
    assert packet.velocity is not None
    v=np.asarray(packet.velocity,dtype=float)
    yaw=previous_yaw if np.linalg.norm(v[:2])<=1e-8 else float(np.arctan2(v[1],v[0]))
    yaw=previous_yaw+float(np.arctan2(np.sin(yaw-previous_yaw),np.cos(yaw-previous_yaw)))
    forecasts=[dict(pos=p+(delay+i)*.01*v,vel=v,acc=np.zeros(3),yaw=yaw) for i in range(1,N+1)]
    ref=physical_reference(forecasts,sim,.01)
    assert np.allclose(ref[:,:3],p+(delay+np.arange(1,N+1))[:,None]*.01*v,rtol=0,atol=1e-13)
    assert np.array_equal(ref[:,3:6],np.broadcast_to(v,(N,3)))
    return ref,yaw

def metrics(states,truth):
    finite=np.all(np.isfinite(states),axis=1)
    valid=len(states) if finite.all() else int(np.flatnonzero(~finite)[0])
    x=states[:valid];r=truth[:valid]
    sep=np.linalg.norm(x[:,:3]-r[:,:3],axis=1);speed=np.linalg.norm(x[:,3:6]-r[:,3:6],axis=1)
    entry=np.flatnonzero(sep<=.75);joint=(sep<=.75)&(speed<=3)
    hits=np.flatnonzero(np.convolve(joint.astype(int),np.ones(21,dtype=int),'valid')==21) if valid>=21 else []
    end=int(hits[0]+20) if len(hits) else None
    first=int(entry[0]) if len(entry) else None
    return dict(captured=end is not None,qualified_time_s=None if end is None else end*.01,scored_time_s=20. if end is None else end*.01,
                first_entry_s=None if first is None else first*.01,relative_speed_at_first_entry_mps=None if first is None else float(speed[first]),
                minimum_separation_m=float(np.min(sep)),maximum_abs_roll_deg=float(np.degrees(np.max(np.abs(x[:,6])))),
                maximum_abs_pitch_deg=float(np.degrees(np.max(np.abs(x[:,7])))),qualifying_sample_end=end,scoring_target='true scripted target, never forecast')

def assert_freeze():
    p=read(HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.json')
    assert sha(HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.json')==(HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.sha256').read_text().split()[0]
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in p['script_sha256'].items():assert sha(HERE/rel)==h,rel
    assert sha(HERE/'TASKS.json')==p['tasks_sha256']
    assert sha(HERE/'SCENARIOS.npz')==p['scenarios_archive_sha256']
    assert sha(p['edmdc_model']['path'])==p['edmdc_model']['sha256']
    return p

def episode(task):
    protocol=assert_freeze();nominal=quad_sim();plant=quad_sim()
    case=next(c for c in protocol['scenarios'] if c['seed']==task['seed'])
    with np.load(HERE/'SCENARIOS.npz') as archive:
        truth=archive[f"truth_{task['seed']}"].copy()
        history=archive[f"history_{task['seed']}"].copy()
    assert arrsha(truth)==case['reference_sha256']
    assert arrsha(history)==case['measurement_history_sha256']
    x=np.asarray(case['initial_state']);initial=x.copy();config=task['config'];N=config['horizon_steps']
    model=None
    if task['predictor']=='edmdc':
        with Path(protocol['edmdc_model']['path']).open('rb') as f:model=pickle.load(f)
    controller=WrenchMPC(task['predictor'],nominal,WrenchConfig(**config),model,initial_yaw=x[8])
    sensor=None if task['mode']=='oracle' else CausalSensor(history,task['delay_samples'],task['mode']=='position_hold')
    states=[x.copy()];requested=[];applied=[];forces=[];saturated=[];statuses=[];accepted=[];forecasts=[];forecast_indices=[];yaw=x[8]
    stop='completed'
    for k in range(2000):
        if task['mode']=='oracle':future=truth[k+1:k+N+1]
        else:
            packet=sensor.receive(k)
            future,yaw=causal_reference(packet,k,N,task['mode'],task['delay_samples'],yaw,nominal)
            if k%50==0:
                forecasts.append(future.copy());forecast_indices.append(k)
        u,info=controller.compute(x,future)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):xn,actual,f,sat=plant_step(plant,x,u,.01)
        states.append(xn.copy());requested.append(u.copy());applied.append(actual.copy());forces.append(f);saturated.append(bool(np.any(sat)));statuses.append(info['status']);accepted.append(info['accepted'])
        controller.previous_applied=actual.copy();x=xn
        if not np.isfinite(x).all():stop='nonfinite_state';break
        if np.max(np.abs(x[6:8]))>np.deg2rad(80):stop='euler_attitude_envelope_80deg';break
        # Abort evaluation uses the true target, never the limited reference.
        if np.linalg.norm(x[:3]-truth[k+1,:3])>50:stop='position_error_over_50m';break
    states=np.asarray(states);req=np.asarray(requested);ap=np.asarray(applied);force=np.asarray(forces)
    row={k:task[k] for k in ['task_id','stage','cell','mode','delay_samples','case_id','family','seed']}
    row.update(metrics(states,truth),completed=stop=='completed',stop_reason=stop,simulated_seconds=(len(states)-1)*.01,
               solver_failures=int(np.count_nonzero(~np.asarray(accepted))),allocator_events=int(np.count_nonzero(np.max(np.abs(req-ap),axis=1)>1e-7)),
               rotor_saturated_steps=int(np.count_nonzero(saturated)),reference_sha256=arrsha(truth),initial_state=initial.tolist(),configuration_sha256=objsha(config),
               plant_mass=float(plant.quad.m),plant_inertia=np.diag(plant.quad.I).tolist(),controller_information='oracle future' if sensor is None else 'one causal measurement packet only',
               causal_access_verified=sensor is None or all(j==k-task['delay_samples'] for k,j in sensor.reads))
    folder=HERE/'episodes'/task['stage']/task['task_id'];folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'trace.npz',states=states,reference=truth[:len(states)],time=np.arange(len(states))*.01,U_requested=req,U=ap,motor_forces=force,
                        solver_status=np.asarray(statuses),solver_accepted=np.asarray(accepted),rotor_saturated=np.asarray(saturated),
                        measurement_access=np.asarray([] if sensor is None else sensor.reads,dtype=int).reshape(-1,2),
                        predicted_target_reference=np.asarray(forecasts).reshape(-1,N,12),prediction_update_indices=np.asarray(forecast_indices,dtype=int))
    row['trace']=str(folder/'trace.npz');row['trace_sha256']=sha(folder/'trace.npz');dump(folder/'metrics.json',row)
    return row

def safe_episode(task):
    try:return episode(task)
    except Exception:
        folder=HERE/'episodes'/task['stage']/task['task_id'];folder.mkdir(parents=True,exist_ok=True)
        row={k:task[k] for k in ['task_id','stage','cell','mode','delay_samples','case_id','family','seed']}
        row.update(completed=False,captured=False,qualified_time_s=None,scored_time_s=20.,stop_reason='execution_exception',exception=traceback.format_exc())
        dump(folder/'exception.json',row);return row
