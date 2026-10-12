"""Frozen-rule analysis of saved metrics. No controller or raw training-data calls."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,csv,datetime,collections,sys
import numpy as np
from scipy.stats import binomtest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
SEED=2026100101;REPS=20000
METRICS=['position_rmse_m','velocity_rmse_mps','attitude_rmse_rad','yaw_rmse_rad']
EVENTS=['solver_failures','allocator_altered_steps','constraint_violations','rotor_saturated_steps']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def distribution(vals):
    a=np.asarray([float(v) for v in vals if v is not None],dtype=float)
    if not len(a):return {'n':0,'mean':None,'median':None,'p95':None,'max':None}
    assert np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))
def summary(rows):
    out={'n':len(rows),'completed':sum(r['completed'] for r in rows),'failures':dict(collections.Counter(r['stop_reason'] for r in rows if not r['completed']))}
    for metric in METRICS:out[metric]=distribution([r.get(metric) for r in rows if r['completed']])
    for e in EVENTS:out[e]=sum(r.get(e,0) or 0 for r in rows)
    out['fallback_step_fraction']=sum(r.get('solver_failures',0) or 0 for r in rows)/max(1,sum(round(r.get('simulated_seconds',0)/.01) for r in rows))
    if rows and rows[0]['group']=='C':
        out['captured']=sum(r.get('captured',False) for r in rows);out['capture_before_later_abort']=sum(r.get('capture_before_later_abort',False) for r in rows)
        for m in ['scored_time_s','qualified_time_s','first_entry_s','relative_speed_at_first_entry_mps','minimum_separation_m']:out[m]=distribution([r.get(m) for r in rows])
        out['first_entry_count']=sum(r.get('first_entry_s') is not None for r in rows)
    return out
def differences(d):
    a=np.array(d,dtype=float);n=len(a)
    if not n:return {'n':0,'mean_difference':None,'ci95':None,'wins':0,'losses':0,'ties':0,'sign_p':None}
    rng=np.random.default_rng(SEED);means=np.empty(REPS)
    for start in range(0,REPS,1000):means[start:start+1000]=a[rng.integers(n,size=(1000,n))].mean(axis=1)
    wins=int(np.count_nonzero(a<0));losses=int(np.count_nonzero(a>0))
    return {'n':n,'mean_difference':float(a.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'wins':wins,'losses':losses,'ties':int(np.count_nonzero(a==0)),'sign_p':float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,'bootstrap_seed':SEED,'resamples':REPS,'lower_is_better':True}
def pair(lhs,rhs,metric,completion=True):
    a={r['case_id']:r for r in lhs};b={r['case_id']:r for r in rhs};assert a.keys()==b.keys()
    ids=sorted(a);good=[i for i in ids if (not completion or (a[i]['completed'] and b[i]['completed'])) and a[i].get(metric) is not None and b[i].get(metric) is not None]
    for i in ids:
        if 'reference_sha256'in a[i] and 'reference_sha256'in b[i]:assert a[i]['reference_sha256']==b[i]['reference_sha256']
    result=differences([a[i][metric]-b[i][metric] for i in good]);result.update(metric=metric,total_pairs=len(ids),excluded_pairs=len(ids)-len(good),lhs_incomplete=sum(not r['completed'] for r in lhs),rhs_incomplete=sum(not r['completed'] for r in rhs),conditional_on_mutual_completion=completion,case_ids=good)
    if metric=='scored_time_s':
        cd=[int(b[i].get('captured',False))-int(a[i].get('captured',False)) for i in ids]
        result['noncapture_indicator_difference']=differences(cd)
    return result
def select(rows,cell,condition=None):return [r for r in rows if r['cell']==cell and (condition is None or r['condition']==condition)]
def process_group(group,rows):
    cells=sorted({(r['cell'],r['condition']) for r in rows});out={'recorded':len(rows),'summaries':{},'by_family':{},'by_heading':{},'paired':{}}
    for cell,condition in cells:
        key=cell+' / '+condition;rr=select(rows,cell,condition);out['summaries'][key]=summary(rr)
        out['by_family'][key]={v:summary([r for r in rr if r['family']==v]) for v in sorted({r['family'] for r in rr})}
        out['by_heading'][key]={v:summary([r for r in rr if r['yaw_mode']==v]) for v in sorted({r['yaw_mode'] for r in rr})}
    comparisons=[]
    if group=='A':
        comparisons=[('W2a_'+s,None,'W1_'+s,None) for s in ['hover','analytic_cost_only']]+[(a+'_analytic_cost_only',None,a+'_hover',None) for a in ['W1','W2a']]
    elif group=='B':
        conditions=['nominal','mass_plus15','mass_minus15','inertia_plus20']
        comparisons=[(a,c,a,'nominal') for a in ['W1','W2a','W3','W1f'] for c in conditions[1:]]+ [('W2a',c,a,c) for c in conditions for a in ['W1','W3','W1f']]
    else:
        comparisons=[('W2a_bundle_'+b,None,'W1_bundle_'+b,None) for b in ['W1','W2a']]+[(a+'_bundle_W2a',None,a+'_bundle_W1',None) for a in ['W1','W2a']]
    for a,c,b,d in comparisons:
        key=f'{a}/{c or "nominal"} minus {b}/{d or "nominal"}';aa=select(rows,a,c);bb=select(rows,b,d)
        out['paired'][key]={m:pair(aa,bb,m,True) for m in METRICS}
        if group=='C':out['paired'][key]['scored_time_s']=pair(aa,bb,'scored_time_s',False)
    if group=='A':
        maps={c:{r['case_id']:r for r in select(rows,c)} for c in ['W1_hover','W2a_hover','W1_analytic_cost_only','W2a_analytic_cost_only']};ids=sorted(next(iter(maps.values())))
        for m in METRICS:
            good=[i for i in ids if all(d[i]['completed'] and d[i].get(m) is not None for d in maps.values())]
            ds=[maps['W2a_analytic_cost_only'][i][m]-maps['W2a_hover'][i][m]-maps['W1_analytic_cost_only'][i][m]+maps['W1_hover'][i][m] for i in good]
            out.setdefault('difference_in_differences',{})[m]={**differences(ds),'definition':'(W2a analytic-W2a hover)-(W1 analytic-W1 hover)','excluded_pairs':len(ids)-len(good),'conditional_on_all_four_complete':True}
    out['failure_episodes']=[{k:r.get(k) for k in ['task_id','case_id','family','cell','condition','stop_reason','simulated_seconds','prefix_position_rmse_m','solver_failures','captured','scored_time_s','trace']} for r in rows if not r['completed']]
    return out
def fmt(v,scale=1):return '—' if v is None else f'{v*scale:.8g}'
def make_report(numbers):
    pr=read(HERE/'PROTOCOL.json');am=read(HERE/'SMOKE_CORRECTION_FREEZE.json');gate=read(HERE/'SMOKE_GATE.json')
    lines=['# ACC targeted gap evaluations — results', '', '**Post-hoc-motivated follow-up; evaluation plan frozen before opening the fresh evaluation scenarios.** No retuning, model selection/adoption, new identification, hardware work, or timing benchmark. Old results are unchanged.', '',f'Original protocol SHA256: `{sha(HERE/"PROTOCOL.json")}`. Reporting-correction amendment: `{sha(HERE/"SMOKE_CORRECTION_FREEZE.json")}`. Evaluation freeze: `{sha(HERE/"EVALUATION_FREEZE.json")}`. Full configuration values, exact tasks and seeds: `PROTOCOL.json`, `TASKS.json`. Full precision and subgroup statistics: `NUMBERS.json`.', '',
      'The wrapper initially completed numerical smoke execution but failed to serialize a four-vector rotor limit as a scalar. All52 records remain as failed reporting attempts. Only the wrapper reporting was corrected; the original source is retained as `runtime_attempt1.py`. The28 replacement smoke/reproduction episodes used four workers; both original-versus-isolated-hover pairs matched states, applied inputs and solver acceptance flags bit-for-bit. Two weaving smoke flights under the longer bundle hit the80° envelope and are retained. This was an implementation-validity gate, not a success-rate selection gate. No evaluation cases were used for this repair. **Total planned ceiling:52+28+1120=1200 episodes; no retries or sample expansion.**', '',
      'Position RMSE is sqrt(mean over time and three coordinates of squared error), in metres in JSON and millimetres where labelled below. Attitude RMSE averages wrapped roll/pitch/yaw squared errors; yaw is wrapped separately. Tracking summaries use completed full episodes only. Paired continuous tracking contrasts use mutually completed cases and report exclusions. Interception penalized qualification time includes20s for every noncapture; captures before a later abort retain the original criterion but are flagged. First entry is only entry into the distance sphere, independent of speed/dwell, and is conditional on entry. Means/medians/P95/max are scenario summaries, not time samples treated as trials.', '',
      f'Paired bootstrap: {REPS:,} scenario resamples, NumPy default_rng seed{SEED},95% percentile intervals; exact two-sided sign test excludes ties. Contrasts and thresholds were fixed in the protocol. Intervals/p-values are descriptive multiple-contrast evidence, not multiplicity-adjusted global claims. Non-significance does not establish equivalence.', '']
    for g,data in numbers['new'].items():
        lines+=['## Group '+g,'']
        lines += [{'A':'Common c23 N75/b5 for all four cells. The intervention changes **only the R-cost wrench reference** from hover to analytical trajectory wrench. No additive command, no changed nominal rollout, no defect cancellation, no independent retuning. Every cell uses the same analytical state reference; these references differ from the historical baseline generator, so old/new absolute errors are not pooled.', 'B':'Actual plant mass/inertia changes while controller assumptions, nominal hover, learned model, mixer and motor capabilities remain fixed. Same40 scenarios across four conditions. W1/W2a/W3 use frozen primaryN75; existing W1f uses F c17 N50/b2. W1f between-arm comparisons are as-selected. Nominal-versus-perturbed differences within each arm are fully paired.', 'C':'Both predictors are crossed with the two complete frozen interception bundles: W1N50/Nc5/b10 and W2aN100/Nc20/b5. Exact interceptor state and scripted known target future, no estimator/IMM/guidance. Ten fresh scenarios per straight/accelerating/helix/weaving family. Qualification is end of21 consecutive10ms samples satisfying0.75m and3m/s.'}[g], '']
        if g=='B':lines+=['W1/W2a share c23; W3 retains its separately selected c11. In perturbed conditions W3 is a **mismatched nominal-physics LTV controller**, not an oracle supplied with true changed parameters.','']
        lines+=['| Cell / condition | Complete | Position mean / median / P95 / max [mm] | Attitude mean / median / P95 / max [rad] | Yaw mean / median / P95 / max [rad] | Solver fallback steps | Allocator / constraint events |','|---|---:|---|---|---|---:|---:|']
        for key,s in data['summaries'].items():
            vals=lambda m,sc:' / '.join(fmt(s[m][t],sc) for t in ['mean','median','p95','max'])
            lines.append(f'| {key} | {s["completed"]}/{s["n"]} | {vals("position_rmse_m",1000)} | {vals("attitude_rmse_rad",1)} | {vals("yaw_rmse_rad",1)} | {s["solver_failures"]} | {s["allocator_altered_steps"]} / {s["constraint_violations"]} |')
        lines+=['','Conditional continuous-error summaries exclude incomplete flights; all aborted prefixes remain in the traces and failure ledger. Rotor-saturated steps and fallback fractions are in JSON.','']
        if g=='C':
            lines+=['| Cell | Captures | Mean penalized qualification [s] | Entry n; mean [s] | Mean speed at entry [m/s] | Mean minimum separation [m] | Capture before later abort |','|---|---:|---:|---|---:|---:|---:|']
            for key,s in data['summaries'].items():lines.append(f'| {key} | {s["captured"]}/{s["n"]} | {fmt(s["scored_time_s"]["mean"])} | {s["first_entry_count"]}; {fmt(s["first_entry_s"]["mean"])} | {fmt(s["relative_speed_at_first_entry_mps"]["mean"])} | {fmt(s["minimum_separation_m"]["mean"])} | {s["capture_before_later_abort"]} |')
            lines+=['','| Cell | Family | Captures / n | Complete | Mean penalized time [s] | Envelope aborts | Solver fallback steps |','|---|---|---:|---:|---:|---:|---:|']
            for key,fams in data['by_family'].items():
                for fam,s in fams.items():lines.append(f'| {key} | {fam} | {s["captured"]}/{s["n"]} | {s["completed"]} | {fmt(s["scored_time_s"]["mean"])} | {s["failures"].get("euler_attitude_envelope_80deg",0)} | {s["solver_failures"]} |')
            lines+=['']
            lines+=['| Cell | Outcome | Available n | Mean | Median | P95 | Maximum |','|---|---|---:|---:|---:|---:|---:|']
            for key,s in data['summaries'].items():
                for metric,label in [('scored_time_s','Penalized qualification [s]'),('qualified_time_s','Qualification, captures only [s]'),('first_entry_s','First entry, entrants only [s]'),('relative_speed_at_first_entry_mps','Speed at first entry [m/s]'),('minimum_separation_m','Minimum separation [m]')]:
                    d=s[metric];lines.append(f'| {key} | {label} | {d["n"]} | '+ ' | '.join(fmt(d[k]) for k in ['mean','median','p95','max'])+' |')
            lines+=['']
        metric='scored_time_s' if g=='C' else 'position_rmse_m';scale=1 if g=='C' else 1000
        lines+=['| Paired contrast (left minus right) | Included / total | Left / right incomplete | Mean difference'+(' [s]' if g=='C' else ' [mm]')+' |95% CI| Left wins / losses / ties | Exact sign p |','|---|---:|---:|---:|---|---:|---:|']
        for key,allmetrics in data['paired'].items():
            p=allmetrics[metric];ci=p['ci95']
            lines.append(f'| {key} | {p["n"]}/{p["total_pairs"]} | {p["lhs_incomplete"]}/{p["rhs_incomplete"]} | {fmt(p["mean_difference"],scale)} | {"—" if ci is None else "["+fmt(ci[0],scale)+", "+fmt(ci[1],scale)+"]"} | {p["wins"]}/{p["losses"]}/{p["ties"]} | {fmt(p["sign_p"])} |')
        if g=='A':
            p=data['difference_in_differences']['position_rmse_m'];lines+=['',f'Paired difference-in-differences, (W2a analytical−hover)−(W1 analytical−hover): **{fmt(p["mean_difference"],1000)} mm**,95% CI{[float(v*1000) for v in p["ci95"]] if p["ci95"] else None}; n={p["n"]},excluded={p["excluded_pairs"]}. This is an interaction for the stated cost intervention, not evidence that EDMDc learned feedforward.']
        if g=='C':
            lines+=['','| Paired contrast | Noncapture fraction difference (left minus right) | 95% CI | Left-only / right-only capture / concordant | Exact discordant-pair p |','|---|---:|---|---:|---:|']
            for key,m in data['paired'].items():
                p=m['scored_time_s']['noncapture_indicator_difference'];lines.append(f'| {key} | {fmt(p["mean_difference"])} | [{fmt(p["ci95"][0])}, {fmt(p["ci95"][1])}] | {p["wins"]}/{p["losses"]}/{p["ties"]} | {fmt(p["sign_p"])} |')
        lines+=['','Failure ledger:']
        if data['failure_episodes']:
            for r in data['failure_episodes']:lines.append(f'- `{r["task_id"]}`: {r["stop_reason"]} at{r["simulated_seconds"]}s; captures={r["captured"]}; solver fallback steps={r["solver_failures"]}. Trace: `{r["trace"]}`.')
        else:lines.append('- None.')
        lines+=['']
    lines+=['## Interpretation and paper decisions','','See `CONCLUSIONS.md` for completed-block interpretation and claim decisions. `AUDIT.md` documents existing evidence, manuscript corrections and retained failed gates. The current available TeX drafts describe an older attitude-command system; the wrench manuscript itself was not supplied. `PAPER_READY_RESULTS.tex` provides scoped replacement text, not certification of an absent draft.','', 'No new benchmark timings are claimed. No conclusions apply to the separate realistic68/R5 estimator/guidance system. No outcome guarantees conference acceptance.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def main():
    result={'classification':'post-hoc motivated, prospectively frozen targeted evaluations','bootstrap_seed':SEED,'bootstrap_replicates':REPS,'existing_saved_episode_recomputation':read(HERE/'EXISTING_RECOMPUTED.json')['sources'],'inventory_warning':'The frozen pre-evaluation EXISTING_EVIDENCE.json also contains coarse source/arm inventory means. Those pool conditions/candidates and are NOT scientific performance comparisons. They are superseded by the condition/candidate-separated EXISTING_RECOMPUTED.json included here; initial outputs are retained.','new':{}}
    allrows=[]
    for g in ['A','B','C']:
        p=HERE/(g+'_EPISODES.json')
        if p.exists():
            rows=read(p);result['new'][g]=process_group(g,rows);allrows+=rows
    # Every pair must have identical reference and initial conditions; physical changes only in B.
    identity={}
    for g in result['new']:
        for case in sorted({r['case_id'] for r in allrows if r['group']==g}):
            rr=[r for r in allrows if r['group']==g and r['case_id']==case and 'reference_sha256'in r]
            assert len({r['reference_sha256'] for r in rr})==1
            assert all(r['initial_state']==rr[0]['initial_state'] for r in rr)
            assert all(r['controller_mass']==rr[0]['controller_mass'] and r['controller_inertia']==rr[0]['controller_inertia'] and r['rotor_force_max']==rr[0]['rotor_force_max'] and r['umin']==rr[0]['umin'] and r['umax']==rr[0]['umax'] for r in rr)
        identity[g]=True
    result['paired_identity_checks']=identity;result['new_evaluation_recorded']=len(allrows)
    dump(HERE/'NUMBERS.json',result);make_report(result)
    print(json.dumps({'groups':list(result['new']),'recorded':len(allrows),'paired_identity_checks':identity},indent=2))
if __name__=='__main__':main()
