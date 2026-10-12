"""Independent ledger, forecast, scoring, and preservation checks; no simulations."""
from runtime import *
from collections import Counter

def main():
    p=assert_freeze();tasks={t['task_id']:t for t in read(HERE/'TASKS.json')};cases={c['seed']:c for c in p['scenarios']};checked=[]
    archive=np.load(HERE/'SCENARIOS.npz')
    for stage in ['primary','secondary','position_only']:
        f=HERE/f'{stage}_EPISODES.json'
        if not f.exists():continue
        rows=read(f);assert len(rows)==p['planned_counts'][stage]
        for row in rows:
            task=tasks[row['task_id']];case=cases[row['seed']]
            if row['stop_reason']=='execution_exception':
                checked.append({'task_id':row['task_id'],'exception_preserved':True});continue
            assert row['configuration_sha256']==p['configuration_sha256'][task['bundle']]
            assert row['plant_mass']==p['plant']['mass'] and row['plant_inertia']==np.diag(p['plant']['inertia']).tolist()
            truth=archive[f"truth_{row['seed']}"];history=archive[f"history_{row['seed']}"]
            with np.load(row['trace']) as z:
                states=z['states'];reference=z['reference'];times=z['time'];access=z['measurement_access']
                assert np.array_equal(reference,truth[:len(states)])
                assert np.array_equal(states[0],np.asarray(case['initial_state']))
                assert np.array_equal(times,np.arange(len(states))*.01)
                assert np.array_equal(access[:,0],np.arange(len(states)-1))
                assert np.array_equal(access[:,1],access[:,0]-task['delay_samples'])
                assert np.isfinite(states).all() or row['stop_reason']=='nonfinite_state'
                valid=len(states) if np.isfinite(states).all() else int(np.flatnonzero(~np.all(np.isfinite(states),axis=1))[0])
                sep=np.linalg.norm(states[:valid,:3]-truth[:valid,:3],axis=1);speed=np.linalg.norm(states[:valid,3:6]-truth[:valid,3:6],axis=1)
                dwell=0;end=None
                for i,eligible in enumerate((sep<=.75)&(speed<=3)):
                    dwell=dwell+1 if eligible else 0
                    if dwell==21:end=i;break
                assert row['captured']==(end is not None)
                assert row['qualified_time_s']==(None if end is None else end*.01)
                assert row['scored_time_s']==(20. if end is None else end*.01)
                if row['completed']:assert len(states)==2001 and np.max(np.abs(states[:,6:8]))<=np.deg2rad(80)
                sim=quad_sim();last_yaw=case['initial_state'][8];sampled_refs={int(k):ref for k,ref in zip(z['prediction_update_indices'],z['predicted_target_reference'])}
                errors=[]
                # Rebuild every packet/heading in order; compare explicitly saved forecasts.
                for k in range(len(states)-1):
                    j=k-task['delay_samples'];pv=history[j+25]
                    packet=Measurement(j,tuple(pv[:3]),None if task['mode']=='position_hold' else tuple(pv[3:6]))
                    if k in sampled_refs:
                        rebuilt,last_yaw=causal_reference(packet,k,task['config']['horizon_steps'],task['mode'],task['delay_samples'],last_yaw,sim)
                        assert np.array_equal(rebuilt,sampled_refs[k])
                        err=rebuilt[:50,:3]-truth[k+1:k+51,:3]
                        errors.append(float(np.sqrt(np.mean(np.sum(err**2,axis=1)))))
                    elif task['mode']!='position_hold':
                        v=np.asarray(packet.velocity)
                        yaw=last_yaw if np.linalg.norm(v[:2])<=1e-8 else float(np.arctan2(v[1],v[0]))
                        last_yaw+=float(np.arctan2(np.sin(yaw-last_yaw),np.cos(yaw-last_yaw)))
                checked.append({'task_id':row['task_id'],'configuration_unchanged':True,'scenario_exact':True,'every_causal_packet_age_verified':True,'independent_21_sample_dwell_verified':True,'saved_forecast_reproduction_exact':True,'saved_forecast_count':len(sampled_refs),'common_0p5s_target_forecast_position_rmse_mean_m':float(np.mean(errors)) if errors else None,'trace_sha256':sha(row['trace'])})
    originals={rel:sha(ROOT/rel)==h for rel,h in p['source_sha256'].items()};originals['selection']=sha(p['selection_path'])==p['selection_sha256'];originals['oracle_episodes']=sha(p['oracle_episodes_path'])==p['oracle_episodes_sha256'];originals['original_tasks']=sha(ROOT/'paper_gap_audit_20261001/TASKS.json')==p['original_tasks_sha256'];assert all(originals.values())
    inventory=read(ROOT.parents[2]/'experiment_inventory_20261001/CROSSED_INTERCEPTION_VERIFICATION.json')
    oracle_hash_checks={r['task_id']:sha(tracepath(r['trace']))==r['trace_sha256'] for r in inventory['episodes']}
    assert len(oracle_hash_checks)==160 and all(oracle_hash_checks.values())
    report={'protocol_sha256':sha(HERE/'TARGET_INFORMATION_ABLATION_PROTOCOL.json'),'originals_unchanged':originals,'original_oracle_trace_hash_checks':oracle_hash_checks,'checked_evaluation_episodes':len(checked),'stage_counts':dict(Counter(tasks[r['task_id']]['stage'] for r in checked)),'episode_checks':checked,'forecast_error_scope':'Target extrapolation error over common0.5s horizon at explicitly saved updates. Not internal interceptor-MPC prediction error. Alive-prefix update sampling can differ between aborted episodes.'}
    dump(HERE/'INDEPENDENT_VALIDATION.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['episode_checks','original_oracle_trace_hash_checks']},indent=2),flush=True)

if __name__=='__main__':main()
