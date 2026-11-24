"""Run identity and result availability come from saved evidence, not folder names."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from heat_sim.run_catalog import read_run,list_runs,choose_result,playable_runs,list_cases,latest_full_print

CURRENT='a'*64
OTHER='b'*64


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def frames(folder,count):
    result=[]
    for index in range(count):
        relative=f'frames/{index:05d}.npz'
        path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,temperature_k=np.ones((2,2,2))*353.15)
        result.append(dict(file=relative,time_s=float(index),layer_index=0))
    return result


def build_run(root,name,state='complete',count=2,sha=CURRENT,created='2026-11-15T09:00:00+00:00',updated='2026-11-15T09:05:00+00:00',start=0,end=3,total=901,alloy='17-4PH'):
    folder=root/name;folder.mkdir(parents=True)
    job=dict(name='17-4 Test.htmjob',source_sha256=sha,layer_count=total)
    config=dict(alloy=alloy,layer_start=start,layer_end=end,recoat_time_s=3.)
    write_json(folder/'job/job-summary.json',job)
    write_json(folder/'request.json',dict(kind='build',job_path='job/job-summary.json',build_config=config,created_utc=created))
    write_json(folder/'manifest.json',dict(status=state,job=job,config=config,frames=frames(folder,count)))
    write_json(folder/'status.json',dict(state=state,message='Synthetic saved status',progress=1. if state=='complete' else .2,updated_utc=updated))
    return folder


def test_latest_finished_current_job_wins_over_folder_name_and_new_cancel(tmp_path):
    full=build_run(tmp_path,'zzz-full-job',created='2026-11-15T08:00:00+00:00',updated='2026-11-15T11:00:00+00:00',end=901)
    build_run(tmp_path,'000-newer-fast-preview',created='2026-11-15T10:00:00+00:00',updated='2026-11-15T10:05:00+00:00')
    cancel=build_run(tmp_path,'newest-cancelled',state='cancelled',count=0,created='2026-11-15T12:00:00+00:00',updated='2026-11-15T12:01:00+00:00')
    build_run(tmp_path,'different-job',sha=OTHER,created='2026-11-15T13:00:00+00:00',updated='2026-11-15T13:01:00+00:00')
    runs=list_runs(tmp_path,job_sha256=CURRENT)
    assert len(runs)==3 and all(run.job_sha256==CURRENT for run in runs)
    available=playable_runs(runs)
    assert cancel not in [run.folder for run in available]
    assert choose_result(available)==full
    assert choose_result(available,preferred=cancel)==full
    info=read_run(full)
    assert info.frame_count==2 and info.has_results
    assert info.layer_start==0 and info.layer_end==901 and info.total_layers==901
    assert '17-4 Test.htmjob'in info.label and '17-4PH'in info.label
    assert '901'in info.layer_label and 'full'in info.layer_label.lower()
    assert 'complete'in info.status_label.lower()
    assert info.run_id==full.name


def test_failed_partial_frames_require_explicit_history_toggle(tmp_path):
    complete=build_run(tmp_path,'complete')
    failed=build_run(tmp_path,'failed-with-frames',state='failed',count=2)
    cancelled=build_run(tmp_path,'cancelled-with-zero-frames',state='cancelled',count=0)
    runs=list_runs(tmp_path,job_sha256=CURRENT)
    assert [run.folder for run in playable_runs(runs)]==[complete]
    assert {run.folder for run in playable_runs(runs,include_incomplete=True)}=={complete,failed}
    assert not read_run(cancelled).has_results
    assert read_run(cancelled).frame_count==0


def test_only_existing_saved_frame_files_count_as_results(tmp_path):
    folder=build_run(tmp_path,'missing-frame',count=1)
    (folder/'frames/00000.npz').unlink()
    info=read_run(folder)
    assert info.frame_count==0 and not info.has_results
    assert playable_runs([info])==[]
    assert list_cases(info)==[]


def transfer_run(root,name,valid=True):
    folder=build_run(root,name,count=0)
    (folder/'manifest.json').unlink()
    request=json.loads((folder/'request.json').read_text());request.update(kind='transfer',options=dict(baseline_material='17-4PH',target_materials=['316L','Ti-6Al-4V']))
    write_json(folder/'request.json',request)
    good=folder/'cases/good';good.mkdir(parents=True)
    bad=folder/'cases/bad';bad.mkdir(parents=True)
    write_json(good/'manifest.json',dict(status='complete',config=dict(alloy='316L'),frames=frames(good,2 if valid else 0)))
    write_json(bad/'manifest.json',dict(status='failed',config=dict(alloy='Ti-6Al-4V'),frames=frames(bad,1)))
    rows=[dict(material='316L',kind='identical',power_scale=1.,speed_scale=1.,case_folder='cases/good',valid=True)]if valid else[]
    write_json(folder/'report.json',dict(status='complete',baseline_material='17-4PH',target_materials=['316L','Ti-6Al-4V'],
        job_source_sha256=CURRENT,rows=rows,failures=[dict(material='Ti-6Al-4V',case_folder='cases/bad',valid=False,error='Integration failed')]))
    return folder


def test_transfer_all_failures_is_not_a_finished_result(tmp_path):
    folder=transfer_run(tmp_path,'all-failed',valid=False)
    info=read_run(folder)
    assert info.completed_case_count==0 and info.failed_case_count>=1
    assert info.state!='complete'
    assert info.frame_count==1 and info.has_results
    assert info not in playable_runs([info])
    assert info in playable_runs([info],include_incomplete=True)
    assert [case for case,label in list_cases(info)]==[folder/'cases/bad']


def test_partial_transfer_identity_and_case_counts_preserve_failed_evidence(tmp_path):
    folder=transfer_run(tmp_path,'partially-completed')
    info=read_run(folder)
    assert info.completed_case_count==1 and info.failed_case_count==1
    assert info.state=='partial'
    assert info.has_results and info.frame_count>=2
    assert info.job_sha256==CURRENT
    assert set(info.target_materials)=={'316L','Ti-6Al-4V'}
    assert '316L'in info.material_label and '17-4PH'in info.material_label
    assert folder in [run.folder for run in playable_runs([info])]
    assert any(path==folder/'cases/good'for path,label in list_cases(info))


def test_preferred_finished_run_is_respected_and_active_frames_can_be_opened(tmp_path):
    old=build_run(tmp_path,'old',created='2026-11-15T08:00:00+00:00')
    new=build_run(tmp_path,'new',created='2026-11-15T10:00:00+00:00',updated='2026-11-15T10:05:00+00:00')
    active=build_run(tmp_path,'active',state='running',count=1,created='2026-11-15T11:00:00+00:00',updated='2026-11-15T11:05:00+00:00')
    runs=playable_runs(list_runs(tmp_path,job_sha256=CURRENT))
    assert choose_result(runs,preferred=old)==old
    assert choose_result(runs,active=active)==active
    assert choose_result(runs)==new


def test_empty_results_folder_has_no_fake_default(tmp_path):
    assert list_runs(tmp_path/'not-created-yet',job_sha256=CURRENT)==[]
    assert choose_result([])is None
    assert read_run(tmp_path/'missing-run')is None


def test_legacy_all_failed_transfer_keeps_failed_frames_but_not_orphan_success(tmp_path):
    folder=transfer_run(tmp_path,'legacy-all-failed',valid=False)
    orphan=folder/'cases/old-unrelated-complete';orphan.mkdir()
    write_json(orphan/'manifest.json',dict(status='complete',config=dict(alloy='316L'),frames=frames(orphan,4)))
    info=read_run(folder)
    assert info.state=='failed' and info.completed_case_count==0
    assert info.frame_count==1
    assert len(list_cases(info))==1
    assert playable_runs([info])==[]
    assert playable_runs([info],include_incomplete=True)==[info]


def test_named_preset_study_identity_and_case_labels_come_from_saved_report(tmp_path):
    folder=build_run(tmp_path,'preset-study',count=0)
    (folder/'manifest.json').unlink()
    request=json.loads((folder/'request.json').read_text())
    request.update(kind='presets',options=dict(study_name='Contour tuning',reference=dict(name='Known print',material='Ti-6Al-4V'),
        candidates=[dict(name='Cooler contour',material='Ti-6Al-4V')]))
    write_json(folder/'request.json',request)
    rows=[]
    for name,kind,scale in [('Known print','baseline',1.),('Cooler contour','candidate',.9)]:
        case=folder/'cases'/kind;case.mkdir(parents=True)
        write_json(case/'manifest.json',dict(status='complete',config=dict(alloy='Ti-6Al-4V',power_scale=scale),frames=frames(case,2)))
        rows.append(dict(preset_name=name,material='Ti-6Al-4V',kind=kind,power_scale=scale,speed_scale=1.,case_folder='cases/'+kind,valid=True))
    write_json(folder/'report.json',dict(status='complete',baseline_material='Ti-6Al-4V',target_materials=['Ti-6Al-4V'],
        job_source_sha256=CURRENT,rows=rows,failures=[]))
    info=read_run(folder)
    assert info.kind=='presets' and info.study_name=='Contour tuning'
    assert 'Contour tuning'in info.label
    assert info.material=='Ti-6Al-4V' and info.target_materials==()
    assert info.completed_case_count==2 and info.state=='complete'
    labels=[label for _,label in list_cases(info)]
    assert any('Reference: Known print'in label for label in labels)
    assert any('Preset: Cooler contour'in label and 'P×0.9'in label for label in labels)


def test_failed_preset_frames_keep_the_named_preset_identity(tmp_path):
    folder=transfer_run(tmp_path,'failed-preset',valid=False)
    report=json.loads((folder/'report.json').read_text())
    report['failures'][0]['preset_name']='Slower supports'
    write_json(folder/'report.json',report)
    assert 'Preset: Slower supports'in list_cases(read_run(folder))[0][1]


def test_named_presets_that_share_cached_physics_keep_both_labels(tmp_path):
    folder=transfer_run(tmp_path,'cached-preset-aliases')
    report=json.loads((folder/'report.json').read_text())
    report['failures']=[]
    report['rows'][0].update(preset_name='Reference',kind='baseline')
    report['rows'].append({**report['rows'][0], 'preset_name':'Unchanged candidate','kind':'candidate'})
    write_json(folder/'report.json',report)
    info=read_run(folder)
    assert info.completed_case_count==2 and info.frame_count==2
    cases=list_cases(info)
    assert len(cases)==2 and cases[0][0]==cases[1][0]
    assert 'Reference'in cases[0][1] and 'Unchanged candidate'in cases[1][1]


def complete_source_evidence(folder,total=901,**config_changes):
    manifest=json.loads((folder/'manifest.json').read_text())
    manifest['config'].update(dict(layer_start=0,layer_end=total,time_limit_s=None,max_vectors=None))
    manifest['config'].update(config_changes)
    manifest['physical_duration_s']=100.
    manifest['frames'][-1].update(time_s=100.,layer_index=total-1)
    write_json(folder/'manifest.json',manifest)


def test_latest_full_print_prefers_actual_complete_job_over_new_preview(tmp_path):
    full=build_run(tmp_path,'full',end=901,updated='2026-11-15T10:00:00+00:00')
    complete_source_evidence(full)
    newer=build_run(tmp_path,'new-preview',updated='2026-11-15T12:00:00+00:00')
    declared_only=build_run(tmp_path,'declared-full-but-not-finished',end=901,updated='2026-11-15T13:00:00+00:00')
    complete_source_evidence(declared_only)
    manifest=json.loads((declared_only/'manifest.json').read_text());manifest['frames'][-1]['layer_index']=5
    write_json(declared_only/'manifest.json',manifest)
    assert latest_full_print(list_runs(tmp_path,CURRENT))==full
    assert choose_result(playable_runs(list_runs(tmp_path,CURRENT)))==declared_only
    assert latest_full_print([read_run(newer)])is None


def test_full_print_rejects_caps_missing_last_frame_and_early_time(tmp_path):
    for name,changes in [('time-cap',{'time_limit_s':1000.}),('vector-cap',{'max_vectors':9999999})]:
        folder=build_run(tmp_path,name,end=901)
        complete_source_evidence(folder,**changes)
        assert latest_full_print([read_run(folder)])is None
    early=build_run(tmp_path,'early-time',end=901)
    complete_source_evidence(early)
    manifest=json.loads((early/'manifest.json').read_text());manifest['frames'][-1]['time_s']=99.
    write_json(early/'manifest.json',manifest)
    assert latest_full_print([read_run(early)])is None
    missing=build_run(tmp_path,'missing-last',end=901)
    complete_source_evidence(missing)
    (missing/'frames/00001.npz').unlink()
    assert latest_full_print([read_run(missing)])is None
    no_evidence=build_run(tmp_path,'no-duration',end=901)
    assert latest_full_print([read_run(no_evidence)])is None
    assert latest_full_print([])is None


def test_partial_batch_can_offer_completed_full_case_and_reject_failed_case(tmp_path):
    folder=transfer_run(tmp_path,'partial-full')
    request=json.loads((folder/'request.json').read_text());request['build_config']['layer_end']=901
    write_json(folder/'request.json',request)
    good=folder/'cases/good'
    complete_source_evidence(good)
    assert read_run(folder).state=='partial'
    assert latest_full_print([read_run(folder)])==folder
    manifest=json.loads((good/'manifest.json').read_text());manifest['status']='failed'
    write_json(good/'manifest.json',manifest)
    assert latest_full_print([read_run(folder)])is None
