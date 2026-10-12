"""Finalize derived reporting after independent audit; original results untouched."""
from runtime import *
import collections

def main():
    assert_freeze();f=HERE/'TARGET_INFORMATION_ABLATION_RESULTS.json';result=read(f);validation=read(HERE/'INDEPENDENT_VALIDATION.json');assert validation['checked_evaluation_episodes']==560
    assert all(validation['originals_unchanged'].values()) and all(validation['original_oracle_trace_hash_checks'].values())
    primary=read(HERE/'PRIMARY_RESULTS_FREEZE.json')
    assert sha(HERE/'PRIMARY_RESULTS.json')==primary['results_sha256'] and sha(HERE/'PRIMARY_REPORT.md')==primary['report_sha256'] and sha(HERE/'primary_EPISODES.json')==primary['episodes_sha256']
    result['independent_validation']={'path':str(HERE/'INDEPENDENT_VALIDATION.json'),'sha256':sha(HERE/'INDEPENDENT_VALIDATION.json'),'new_evaluation_traces_verified':560,'old_oracle_raw_traces_unchanged':160,'true_dwell_independently_recounted':True,'saved_causal_forecasts_rebuilt_exactly':True}
    result['run_accounting']={'primary_evaluation':240,'secondary_evaluation':240,'position_only_exploratory':80,'nonoracle_smoke':8,'oracle_reproduction_only':2,'total_new_controller_episodes':570,'reused_existing_oracle_episodes':160,'automatic_retries':0,'new_tuning_or_fitting':0}
    bytask={r['task_id']:r for r in result['episodes'] if r['mode']!='oracle'}
    errors=collections.defaultdict(list)
    for check in validation['episode_checks']:
        r=bytask[check['task_id']]
        v=check.get('common_0p5s_target_forecast_position_rmse_mean_m')
        if v is not None:errors[r['cell'],r['mode'],r['family']].append(v)
    result['target_extrapolation_error_common_0p5s']=[{'cell':cell,'mode':mode,'family':family,'n':len(v),'mean_episode_forecast_rmse_m':float(np.mean(v)),'note':'True target versus explicitly saved target forecasts over common0.5s horizon at saved alive-prefix updates; not internal interceptor prediction error.'} for (cell,mode,family),v in sorted(errors.items())]
    dump(f,result)
    report=HERE/'TARGET_INFORMATION_ABLATION_REPORT.md'
    extra=['','## Final independent audit and secondary/exploratory interpretation',
           'All560 evaluation traces passed independent scenario/configuration checks, timestamp-age checks, saved-forecast regeneration, and true-target21-sample dwell recounting. All160 original oracle NPZ hashes still match the pre-existing inventory. Original model, plant/controller sources, selection, crossed episode index and task ledger remain unchanged. The primary report/results were frozen before secondary execution and still match their hashes.',
           'Execution accounting:240 primary CV +240 secondary CV +80 position-only evaluations, plus8 non-oracle smoke and2 oracle reproductions =570 new controller episodes. No automatic retries, model fitting, candidate search or retuning occurred.',
           'The secondary crossing preserves the configuration interaction: EDMDc-A captures33/32/31 under CurrentCV/100ms/250ms, matching Linear-A capture counts. Linear-B captures2/2/1 with38 attitude aborts under every CV mode, whereas EDMDc-B captures40/40/37 with no attitude aborts. Thus the selected-controller advantage should not be generalized to every configuration or attributed to horizon alone.',
           'The exploratory position-only stationary-reference mode captures4/40 for Linear-A and15/40 for EDMDc-B; EDMDc-B has one80-degree attitude abort. Both have zero solver/allocator events. This mode retains zero desired velocity for the held target reference and fixed initial yaw; it is not the primary CV frozen-controller comparison and does not establish what a separately designed/tuned position-only controller could achieve.',
           'Target-extrapolation errors from explicitly saved forecasts are recorded over a common0.5s horizon in the results JSON. They measure the separate kinematic target assumption, not EDMDc interceptor prediction error. Alive-prefix forecast sampling differs for early-aborted secondary episodes.',
           'Recommended paper placement: show Oracle/CurrentCV/100ms selected-controller capture and time sensitivity in the main paper; place250ms, secondary crossing, and position-only pursuit in supplementary material. CurrentCV retains40/40 EDMDc-B captures and only increases its mean qualification from2.23350s to2.29100s, while Linear-A loses7 helix captures. These are useful direct data on dependence on oracle preview, with exact noiseless target position/velocity and fixed previously selected controller bundles. Avoid calling this perception, estimation or learned target-prediction validation.','']
    report.write_text(report.read_text(encoding='utf-8')+'\n'.join(extra),encoding='utf-8')
    dump(HERE/'RESULTS_FREEZE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results_sha256':sha(f),'report_sha256':sha(report),'stages_complete':['primary','secondary','position_only'],'episode_hashes':{s:sha(HERE/f'{s}_EPISODES.json') for s in ['primary','secondary','position_only']},'independent_validation_sha256':sha(HERE/'INDEPENDENT_VALIDATION.json'),'analysis_scripts_sha256':{p.name:sha(p) for p in (HERE/'scripts').glob('*.py')}})
    print(json.dumps({'results_sha256':sha(f),'new_evaluation_episodes':560,'total_new_including_smoke':570,'original_oracle_traces_unchanged':160},indent=2),flush=True)

if __name__=='__main__':main()
