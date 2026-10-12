"""Preregister, verify, tune and evaluate known-future wrench interception."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from core import *
import datetime,argparse,subprocess
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.stats import binomtest
from publication_scenarios import latin_hypercube
import run_wrench_controller_tuning as tracking

ARMS=('PID','W1','W2a','W3');FAMILIES=('straight','accelerating','helix','weaving')
BASE=2909290000;LHS_SEED=2026092903;BOOT_SEED=2026092904
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def utc():return datetime.datetime.now(datetime.timezone.utc)
TUNE_DEADLINE=datetime.datetime(2026,9,30,11,tzinfo=datetime.timezone.utc)
EVAL_DEADLINE=datetime.datetime(2026,9,30,12,30,tzinfo=datetime.timezone.utc)

def cases(stage):
    count,offset={'screening':(3,0),'validation':(12,10000),'evaluation':(10,40000)}[stage]
    return [dict(case_id=f'{stage}_{f}_{i:02d}',family=f,seed=BASE+offset+1000*j+i) for j,f in enumerate(FAMILIES) for i in range(count)]

def candidates():
    units=latin_hypercube(25,12,LHS_SEED);allc={arm:[] for arm in ARMS}
    for i,u in enumerate(units):
        log=lambda col,lo,hi:float(np.exp(np.log(lo)+u[col]*np.log(hi/lo)))
        N=(30,50,75,100)[min(int(u[0]*4),3)]
        blocks={30:(5,6,10,15),50:(5,10,25),75:(5,15,25),100:(5,10,20,25)}[N]
        nb=blocks[min(int(u[1]*len(blocks)),len(blocks)-1)]
        cfg=asdict(WrenchConfig());cfg['horizon_steps']=N;cfg['block_steps']=N//nb
        # Capture-scaled p/v residuals: radius .75 m and relative speed 3 m/s.
        cfg['q']=[log(2,.25,8)/.75**2]*3+[log(3,.25,8)/3**2]*3+[.2*log(4,.25,8)]*3+[.02*log(5,.1,4)]*3
        cfg['r']=(np.array([.01,.4,.4,.2])*log(7,.05,5)).tolist()
        cfg['rd']=(np.array([.02,.8,.8,.4])*log(8,.1,10)).tolist()
        cfg['terminal_scale']=5*log(6,.4,4)
        cfg['trust']=[5+15*u[9],.2+.9*u[10],.2+.9*u[10],.05+.3*u[11]]
        for arm in ('W1','W2a','W3'):allc[arm].append(dict(candidate_id=f'c{i:02d}',config=cfg))
        base={'kp_pos':[2,2,3],'kd_pos':[2,2,2.5],'ki_pos':[.1,.1,.2],
              'kp_ang':[30,30,12],'kd_ang':[8,8,5],'ki_ang':[1,1,.5]}
        # Twelve independent log factors: xy and z/yaw for each of six gain groups.
        gains={}
        for j,(key,value) in enumerate(base.items()):
            lo,hi=(.05,4) if key.startswith('ki') else (.25,4)
            gains[key]=(np.array(value)*np.array([log(2*j,lo,hi),log(2*j,lo,hi),log(2*j+1,lo,hi)])).tolist()
        allc['PID'].append(dict(candidate_id=f'c{i:02d}',config=gains))
    return allc

def setup():
    assert not (HERE/'protocol.json').exists()
    # Search all available study text artifacts for the reserved 2,909,29x,xxx seed band.
    cmd=['rg','--pcre2','--no-ignore','--hidden','-l',r'(?<![A-Za-z0-9.])2909[23][0-9]{5}(?![A-Za-z0-9.])',str(ROOT),'-g','*.json','-g','*.py','-g','*.md','-g','*.csv','-g','!**/wrench_interception_v1/**','-g','!**/.git/**']
    search=subprocess.run(cmd,capture_output=True,text=True)
    dump(HERE/'seed_audit.json',dict(command=cmd,returncode=search.returncode,stdout=search.stdout,stderr=search.stderr,created_utc=now()))
    assert search.returncode==1 and not search.stdout,'Reserved seed band collision or search error'
    cand=candidates();dump(HERE/'candidates.json',cand)
    splits={s:cases(s) for s in ('screening','validation','evaluation')}
    sets=[{r['seed'] for r in rr} for rr in splits.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    for s,rows in splits.items():dump(HERE/f'{s}/scenarios.json',rows)
    files=[HERE/x for x in ('campaign.py','core.py','targets.py','candidates.json','seed_audit.json')]
    files += [ROOT/x for x in ('Simulation.py','quadcopter.py','PID_Mixer.py','wrench_revisit_mpc.py','edmdc_mpc.py','publication_scenarios.py')]
    model=tracking.MODEL_PATHS['training_v2'];assert digest(model)==tracking.MODEL_HASHES['training_v2']
    p=dict(created_utc=now(),classification='confirmatory new known-future protocol; development and tuning exploratory',hardware_actions=False,
        sources={str(f):digest(f) for f in files},model=str(model),model_sha256=digest(model),arms=list(ARMS),workers=4,dt=.01,
        scope='Nominal simulation plant and exact current interceptor state. Scripted analytic target p/v/yaw preview; target estimation/forecasting errors excluded. PID sees only current p/v/yaw, no preview or target acceleration. No EKF/IMM/guidance.',
        targets='Unmodified straight, accelerating, helix and weaving definitions copied from Intercept_comparison.py; source in targets.py. Initial interceptor state rest at [0,0,0], yaw equal current target yaw. Target parameters sampled once per seed; paired unchanged for every arm.',
        formulation='Direct WrenchMPC minimizes physical-reference tracking with position normalized by .75 m and velocity by 3 m/s, common terminal multiplier. Analytic target acceleration determines feasible reference attitude/body rates; actuator allocation/rotor saturation identical to tracking. No fallback guidance. Existing MPC previous-applied-command fallback retained and counted.',
        pid='Position/velocity PID into thrust vector, Euler attitude PID into body torque; 45 degree tilt and nominal wrench limits; all six integrators held on rotor saturation. No QP; solver-failure counts are not comparable.',
        tuning=dict(candidates_per_arm=25,design='25 Latin-hypercube candidates; identical predictive search space across W1/W2a/W3; 12-dimensional PID gain multipliers, equal episode budget/procedure',lhs_seed=LHS_SEED,
        screening_cases=12,screening_seconds=10,finalists=4,validation_cases=48,validation_seconds=20,
        selection_rule=['number incomplete (exceptions count incomplete)','sum solver failures + rotor-saturated steps (PID solver term zero, no QP)','mean coordinate position RMSE over complete runs, infinity if none','candidate index'],
        finalists_rule='best four; no forced incumbent',disclosure='48 validation trajectories each have the interception episode length 20 s (tracking used 30 s); all four arms identical here. Cost re-scaled before any run; every candidate retained.'),
        capture=dict(separation_m=.75,relative_speed_mps=3,dwell_seconds=.2,samples=21,qualification_time='end of dwell',episode_seconds=20,noncapture_score_seconds=20,
            first_entry='first separation <= .75 irrespective of relative speed; also log first joint p/v region entry'),
        scenarios=splits,selection_freeze='hash before evaluation; no evaluation feedback to tuning',
        statistics=dict(primary='W2a minus W1 failure-aware qualified time (noncaptures=20) over all 40 pairs',resamples=20000,bootstrap_seed_base=BOOT_SEED,bootstrap_offsets={'PID':0,'W2a':1,'W3':2},sign_test='exact two-sided excluding abs(diff)<=1e-12 ties',alpha=.05,
        outcome_rule='EDMDc faster/slower only if sign-test p<.05 and 95% paired mean-difference bootstrap CI strictly below/above zero; otherwise no measurable difference, not equivalence. Completion/capture counts always reported. No confirmatory inference from incomplete evaluation block.',
        mechanism_rule='Do not infer causality from ordering: report observed saturation, solver/attitude diagnostics; mechanism unresolved unless supported.'),
        outcomes=[
        'With known future target motion, EDMDc-MPC qualified [X] s sooner than the hover-linear baseline ([p]); the ordering matches the tracking result.',
        'Qualified interception time did not differ measurably across prediction models ([counts], [p]), even though position tracking differed by [factor].',
        'Despite tracking more accurately, EDMDc-MPC qualified [X] s later ([p]); the mechanism is [saturation / attitude error / other], not tracking accuracy.'],
        gates=['source/model hashes and disjoint seeds','21-sample dwell boundary checks; noncapture score 20','hover allocation and no-preview PID interface; forced rotor saturation freezes all integrators'],
        schedule=dict(tuning_deadline_utc=TUNE_DEADLINE.isoformat(),evaluation_deadline_utc=EVAL_DEADLINE.isoformat(),if_tuning_late='No evaluation; report protocol and all exploratory work retained',priority='Block 2 must finish/report or halt before any Block 3 run'))
    dump(HERE/'protocol.json',p);dump(HERE/'protocol.sha256.json',dict(sha256=digest(HERE/'protocol.json')))
    lines=['# Known-future wrench interception — preregistration','',f"Protocol SHA-256: `{digest(HERE/'protocol.json')}`",'',p['scope'],'',p['formulation'],'',p['pid'],'',
        'All four arms receive 25 LHS candidates, 12 screening trajectories × 10 s, then best four × 48 validation trajectories × 20 s. All failed and aborted attempts retained. Evaluation: 40 fresh paired scenarios, ten/class, 20 s. Select by incomplete count, solver/saturation failures, completed-run position RMSE, candidate index. Selection hashed before evaluation. Capture requires 21 consecutive 100 Hz samples satisfying 0.75 m and 3 m/s; noncaptures score 20 s.','',
        'Predictive horizons are selected from N=30/50/75/100 with identical candidate sets. PID has no preview/horizon. This block is not horizon-matched unless selections coincide.','',
        '## Frozen outcome sentences','']+['- '+s for s in p['outcomes']]+['',p['statistics']['outcome_rule'],'',p['statistics']['mechanism_rule'],'','Full seed lists, exact search space, gates and deadlines: `protocol.json` and `candidates.json`. No runs before this hash.']
    (HERE/'PROTOCOL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PROTOCOL FROZEN',digest(HERE/'protocol.json'),flush=True)

def verify():
    p=read(HERE/'protocol.json');assert digest(HERE/'protocol.json')==read(HERE/'protocol.sha256.json')['sha256']
    for file,h in p['sources'].items():assert digest(file)==h,file
    assert digest(p['model'])==p['model_sha256'];return p

def gate():
    p=verify();checks=[]
    try:
        truth=np.zeros((22,12));x=truth.copy()
        assert capture_metrics(x[:20],truth)['captured'] is False
        assert capture_metrics(x[:21],truth)['qualified_time_s']==.2
        x[:,3]=3.01;assert capture_metrics(x,truth)['scored_time_s']==20
        x=truth.copy();x[10,0]=.76;assert not capture_metrics(x,truth)['captured'];checks.append('21-sample dwell and speed/radius boundaries')
        sim=quad_sim();g=read(HERE/'candidates.json')['PID'][0]['config'];pid=CascadedPID(sim,g)
        u=pid.command(np.zeros(12),np.zeros(3),np.zeros(3),0.)
        np.testing.assert_allclose(u,[sim.quad.m*sim.quad.g,0,0,0],atol=1e-12)
        _,actual,_,_=plant_step(sim,np.zeros(12),u,.01);np.testing.assert_allclose(actual,u,atol=1e-12)
        checks.append('shared hover wrench allocation')
        constrained=quad_sim();constrained.max_speed=1.0;pid=CascadedPID(constrained,g)
        before=pid.integral.copy();pid.command(np.zeros(12),np.ones(3),np.ones(3),.5)
        assert pid.last_integral_held;np.testing.assert_array_equal(before,pid.integral);checks.append('all integrators held on forced rotor saturation')
        # command signature cannot accept future target arrays; no future access in PID object.
        assert set(pid.__dict__)<= {'sim','gains','integral','last_integral_held'};checks.append('PID receives only current p/v/yaw through command arguments')
        dump(HERE/'gate.json',dict(passed=True,checks=checks,created_utc=now()))
    except Exception:
        dump(HERE/'gate.json',dict(passed=False,checks=checks,error=traceback.format_exc(),created_utc=now()));raise
    print('BLOCK 3 VERIFICATION PASSED',flush=True)

def make_tasks(stage,chosen=None):
    p=verify();cand=read(HERE/'candidates.json');tasks=[]
    for arm in ARMS:
        choices=cand[arm] if chosen is None else chosen[arm]
        for c in choices:
            for case in p['scenarios'][stage]:tasks.append(dict(**case,arm=arm,**c,stage=stage,model=p['model'],duration=10 if stage=='screening' else 20,output=str(HERE/stage/'runs')))
    return tasks

def batch(stage,tasks,deadline):
    assert not (HERE/stage/'tasks.json').exists()
    dump(HERE/stage/'tasks.json',tasks);rows=[]
    # Bounded submission permits hard scheduling cutoff without silently dropping launched work.
    with ProcessPoolExecutor(max_workers=4) as pool:
        cursor=0;active={}
        while (cursor<len(tasks) or active):
            while cursor<len(tasks) and len(active)<4 and utc()<deadline:
                t=tasks[cursor];active[pool.submit(safe_episode,t)]=t;cursor+=1
            if not active:break
            future=next(as_completed(active));active.pop(future);rows.append(future.result())
            dump(HERE/stage/'episodes.json',rows)
            dump(HERE/'progress.json',dict(stage=stage,recorded=len(rows),expected=len(tasks),launched=cursor,time=now()))
            if len(rows)%20==0:print(stage,len(rows),'/',len(tasks),flush=True)
    dump(HERE/stage/'accounting.json',dict(planned=len(tasks),launched=cursor,recorded=len(rows),missing=[t for t in tasks[cursor:]],complete=len(rows)==len(tasks),time=now()))
    return rows

def rankings(rows):
    ranked={}
    for arm in ARMS:
        rr=[]
        for cid in sorted({r['candidate_id'] for r in rows if r['arm']==arm}):
            group=[r for r in rows if r['arm']==arm and r['candidate_id']==cid];complete=[r for r in group if r['completed']]
            failures=sum((r.get('solver_failures') or 0)+(r.get('rotor_saturated_steps') or 0) for r in group)
            mean=float(np.mean([r['position_rmse_m'] for r in complete])) if complete else None
            rr.append(dict(candidate_id=cid,incomplete=len(group)-len(complete),failure_steps=failures,mean_position_rmse_m=mean,episodes=len(group)))
        ranked[arm]=sorted(rr,key=lambda r:(r['incomplete'],r['failure_steps'],r['mean_position_rmse_m'] if r['mean_position_rmse_m'] is not None else float('inf'),r['candidate_id']))
    return ranked

def run():
    p=verify();assert read(HERE/'gate.json')['passed']
    r2=ROOT/'wrench_stage_r_controller_mismatch_v1'
    assert (r2/'STAGE_R_RESULTS.md').exists()
    if utc()>=TUNE_DEADLINE:dump(HERE/'status.json',dict(stage='protocol_only',reason='06:00 cutoff'));report();return
    cand=read(HERE/'candidates.json')
    rr=batch('screening',make_tasks('screening'),TUNE_DEADLINE)
    if len(rr)!=1200:report();return
    rank=rankings(rr);dump(HERE/'screening/ranking.json',rank)
    chosen={a:[next(c for c in cand[a] if c['candidate_id']==r['candidate_id']) for r in rank[a][:4]] for a in ARMS}
    dump(HERE/'validation/finalists.json',chosen);print('BOUNDARY SCREENING COMPLETE',flush=True)
    rr=batch('validation',make_tasks('validation',chosen),TUNE_DEADLINE)
    if len(rr)!=768 or utc()>=TUNE_DEADLINE:report();return
    rank=rankings(rr);dump(HERE/'validation/ranking.json',rank)
    selected={a:[next(c for c in cand[a] if c['candidate_id']==rank[a][0]['candidate_id'])] for a in ARMS}
    dump(HERE/'selection.json',dict(created_utc=now(),selected=selected,protocol_sha256=digest(HERE/'protocol.json'),ranking_sha256=digest(HERE/'validation/ranking.json')))
    dump(HERE/'selection.sha256.json',dict(sha256=digest(HERE/'selection.json')))
    print('BOUNDARY SELECTION FROZEN',digest(HERE/'selection.json'),flush=True);report()
    assert digest(HERE/'selection.json')==read(HERE/'selection.sha256.json')['sha256']
    rr=batch('evaluation',make_tasks('evaluation',selected),EVAL_DEADLINE)
    # All arm/reference hashes are identical within each completed scenario.
    for cid in {r['case_id'] for r in rr}:
        assert len({r['reference_sha256'] for r in rr if r['case_id']==cid and 'reference_sha256' in r})<=1
    report();print('BOUNDARY EVALUATION REPORTED',flush=True)

def distribution(x):
    a=[v for v in x if v is not None]
    return dict(mean=float(np.mean(a)),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)),n=len(a)) if a else dict(mean=None,median=None,p95=None,max=None,n=0)

def report():
    p=verify();summaries=[];paired=[];ep=HERE/'evaluation/episodes.json';rows=read(ep) if ep.exists() else []
    for family in ('all',)+FAMILIES:
        for arm in ARMS:
            rr=[r for r in rows if r['arm']==arm and (family=='all' or r['family']==family)]
            if not rr:continue
            s=dict(family=family,arm=arm,recorded=len(rr),planned=40 if family=='all' else 10,completed=sum(r['completed'] for r in rr),captures=sum(r['captured'] for r in rr),
                allocator_altered_steps=sum(r.get('allocator_altered_steps') or 0 for r in rr),solver_failures=None if arm=='PID' else sum(r.get('solver_failures') or 0 for r in rr),
                exceptions=sum(r['stop_reason']=='execution_exception' for r in rr))
            for m in ('scored_time_s','qualified_time_s','first_entry_s','first_joint_region_entry_s','relative_speed_at_first_entry_mps','minimum_separation_m','position_rmse_m'):s[m]=distribution([r.get(m) for r in rr])
            summaries.append(s)
    if len(rows)==160:
        lookup={(r['case_id'],r['arm']):r for r in rows}
        for j,arm in enumerate(('PID','W2a','W3')):
            pairs=[(lookup[(c['case_id'],arm)],lookup[(c['case_id'],'W1')]) for c in p['scenarios']['evaluation']]
            d=np.array([a['scored_time_s']-b['scored_time_s'] for a,b in pairs]);wins=int(np.sum(d < -1e-12));losses=int(np.sum(d>1e-12))
            rng=np.random.default_rng(BOOT_SEED+j);boot=np.mean(d[rng.integers(0,40,(20000,40))],axis=1)
            paired.append(dict(first=arm,second='W1',pairs=40,wins=wins,losses=losses,ties=40-wins-losses,mean_difference_s=float(d.mean()),p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,ci95=np.percentile(boot,[2.5,97.5]).tolist(),bootstrap_seed=BOOT_SEED+j))
    dump(HERE/'analysis/summary.json',summaries);dump(HERE/'analysis/paired.json',paired)
    lines=['# Known-future wrench interception — results','',f"Protocol SHA-256: `{digest(HERE/'protocol.json')}`",'',p['scope'],'',
        'Tuning and verification are exploratory. The complete 40-scenario evaluation is confirmatory under the new preregistration; an incomplete block is descriptive only. No target estimation/forecasting uncertainty is represented. PID has no preview and solves no QP, so its solver counts are N/A. All failures remain in accounting.','',
        '## Stage accounting','']
    for stage in ('screening','validation','evaluation'):
        path=HERE/stage/'accounting.json';lines.append('- '+stage+': '+(json.dumps({k:v for k,v in read(path).items() if k!='missing'},ensure_ascii=True) if path.exists() else 'not completed / not started'))
    if (HERE/'selection.json').exists():
        lines+=['','Selection SHA-256: `'+digest(HERE/'selection.json')+'`.','']
        for a,cc in read(HERE/'selection.json')['selected'].items():lines.append(f"- {a}: {cc[0]['candidate_id']}; N={cc[0]['config'].get('horizon_steps','N/A (no preview)')}")
    lines+=['','| Class | Arm | Capture / planned | Complete / recorded | Scored time mean [s] | Qualified time mean [s] | First entry mean [s] | Speed at entry mean [m/s] | Minimum separation mean [m] | Allocator altered | Solver failures |','|---|---|---|---|---|---|---|---|---|---|---|']
    fmt=lambda x:'NA' if x is None else f'{x:.7g}'
    for s in summaries:lines.append(f"| {s['family']} | {s['arm']} | {s['captures']}/{s['planned']} | {s['completed']}/{s['recorded']} | {fmt(s['scored_time_s']['mean'])} | {fmt(s['qualified_time_s']['mean'])} | {fmt(s['first_entry_s']['mean'])} | {fmt(s['relative_speed_at_first_entry_mps']['mean'])} | {fmt(s['minimum_separation_m']['mean'])} | {s['allocator_altered_steps']} | {s['solver_failures']} |")
    lines+=['','Qualified-only and first-entry means condition on those events; failure-aware scored time includes every noncapture at 20 s. Full distributions/sample counts are in `analysis/summary.json`.','',
        '| Comparison | Wins / losses / ties | Mean difference [s] | Sign p | 95% CI [s] |','|---|---|---|---|---|']
    for r in paired:lines.append(f"| {r['first']} − W1 | {r['wins']} / {r['losses']} / {r['ties']} | {r['mean_difference_s']:.7g} | {r['p']:.7g} | {r['ci95']} |")
    if paired:
        r=next(r for r in paired if r['first']=='W2a');faster=r['p']<.05 and r['ci95'][1]<0;slower=r['p']<.05 and r['ci95'][0]>0
        if faster:sentence=f"With known future target motion, EDMDc-MPC qualified {abs(r['mean_difference_s']):.4g} s sooner than the hover-linear baseline (p={r['p']:.4g}); the ordering matches the tracking result."
        elif slower:sentence=f"Despite tracking more accurately in the earlier baseline, EDMDc-MPC qualified {r['mean_difference_s']:.4g} s later (p={r['p']:.4g}); the mechanism remains unresolved from aggregate outcomes, and tracking accuracy alone did not predict this ordering."
        else:
            counts=', '.join(f"{s['arm']} {s['captures']}/40" for s in summaries if s['family']=='all')
            sentence=f"Qualified interception time did not differ measurably between W2a and W1 ({counts}; p={r['p']:.4g}), even though mean position tracking RMSE differed by 4.41413× in the earlier baseline. This is not proof of equivalence."
        lines+=['','## Preregistered outcome','',sentence,'','Times here are failure-aware scored means; they are not mean time among captures only. Other arm comparisons are reported separately.']
    else:lines+=['','No confirmatory outcome sentence: full paired evaluation is not available.']
    lines+=['','All source files, seeds, failed attempts and traces retained. Hardware untouched. `result_freeze.json` hashes available artifacts.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    paths=[f for f in HERE.rglob('*') if f.is_file() and f.suffix in ('.json','.md','.py','.npz') and f.name not in ('result_freeze.json','commit.json')]
    dump(HERE/'result_freeze.json',dict(created_utc=now(),artifacts={str(f.relative_to(HERE)):digest(f) for f in paths}))
    (HERE/'result_freeze.sha256').write_text(digest(HERE/'result_freeze.json')+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['setup','gate','run','report']);args=parser.parse_args();globals()[args.action]()
