"""Exercise the job→current material→target settings UI without starting workers."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def widget(elements,label):
    return next(item for item in elements if item.label==label)


def test_job_ui_preserves_user_layer_delay_and_independent_alloy_absorption(monkeypatch):
    from heat_sim import job_ui
    calls=[]
    def capture(kind,summary,config,options,provenance):
        calls.append(dict(kind=kind,summary=summary,config=config,options=options,provenance=provenance))
    monkeypatch.setattr(job_ui,'_start',capture)
    # A user's independent live calculation must not disable this isolated UI
    # dispatch check or turn it into a background-progress test.
    monkeypatch.setattr(job_ui,'_active',lambda:(None,None))
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py')).run(timeout=45)
    assert not app.exception
    assert widget(app.number_input,'Between layers, including recoating (s) • user estimate').value==3.
    widget(app.multiselect,'Simulate the same print in').set_value(['316L','Ti-6Al-4V','AlSi10Mg'])
    for name,value in [('17-4PH',.38),('316L',.29),('Ti-6Al-4V',.44),('AlSi10Mg',.17)]:
        widget(app.number_input,name).set_value(value)
    widget(app.button,'Compare selected materials at identical job settings').click().run(timeout=45)
    assert not app.exception and len(calls)==1
    request=calls[0]
    assert request['kind']=='transfer'
    assert request['config'].recoat_time_s==3.
    assert request['config'].absorptivity==.38
    assert request['options']['absorptivities']=={'17-4PH':.38,'316L':.29,'Ti-6Al-4V':.44,'AlSi10Mg':.17}
    assert request['options']['target_materials']==['316L','Ti-6Al-4V','AlSi10Mg']
    assert request['options']['evaluate_candidates'] is False
    widget(app.button,'Calculate bounded transfer candidates').click().run(timeout=45)
    assert not app.exception and len(calls)==2
    assert calls[-1]['options']['evaluate_candidates'] is True
    assert calls[-1]['options']['power_scales']==[1.]
    assert calls[-1]['options']['speed_scales']==pytest.approx([.9,1.,1.1])
    assert calls[-1]['config'].recoat_time_s==3.


def test_saved_real_job_transfer_results_load_all_materials_and_locked_celsius_scale():
    folder=ROOT/'results/builds/20261115-real-job-material-transfer'
    if not (folder/'report.json').exists():
        pytest.skip('Actual locally computed transfer artifact is not available')
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py')).run(timeout=45)
    imported=ROOT/'results/imports/17-4-actual/job-summary.json'
    if imported.exists():
        widget(app.selectbox,'Imported job').set_value(imported).run(timeout=45)
    widget(app.selectbox,'Simulation to view').set_value(folder).run(timeout=45)
    assert not app.exception
    cases=widget(app.selectbox,'Material / candidate playback')
    assert len(cases.options)==10
    frame=next(table.value for table in app.dataframe if 'material'in table.value.columns)
    assert len(frame)==10
    assert set(frame['material'])=={'17-4PH','316L','Ti-6Al-4V','AlSi10Mg'}
    assert widget(app.number_input,'Locked temperature minimum (°C)').value==20.
    assert widget(app.number_input,'Locked temperature maximum (°C)').value==1800.
    cases.set_value(7).run(timeout=45)
    assert not app.exception
    assert widget(app.number_input,'Locked temperature minimum (°C)').value==20.
    assert widget(app.number_input,'Locked temperature maximum (°C)').value==1800.


def test_legitimate_single_layer_import_keeps_main_ui_usable(tmp_path,monkeypatch):
    import json
    from heat_sim import job_ui
    from heat_sim.job_import import load_job,save_imported_job
    source=tmp_path/'one-layer.json'
    source.write_text(json.dumps(dict(format='heat-sim-job-v1',coordinate_units='m',speed_units='m/s',
        power_units='W_assumed',layer_thickness_m=60e-6,
        layers=[dict(index=0,z_m=0,vectors=[[0,0,.001,0,120,1.4]])])))
    results=tmp_path/'isolated-results';results.mkdir()
    save_imported_job(load_job(source),results/'imports'/'one-layer')
    monkeypatch.setattr(job_ui,'RESULTS',results)
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py')).run(timeout=45)
    assert not app.exception
    assert widget(app.number_input,'First source layer (zero based)').value==0
    assert widget(app.number_input,'Last source layer (inclusive)').value==0
    assert widget(app.button,'Simulate current material').disabled is False


def test_historical_cancellation_is_not_a_global_warning_after_success():
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py'))
    app.session_state['build_last_status']={'state':'cancelled','message':'HISTORICAL_TEST_CANCELLED_ZERO_FRAMES'}
    app.run(timeout=45)
    assert not app.exception
    assert not any('HISTORICAL_TEST_CANCELLED_ZERO_FRAMES'in str(w.value)for w in app.warning)
    assert not any('Last calculation cancelled'in str(w.value)for w in app.warning)


def _select_actual_dachshund(app):
    import json
    candidates=list((ROOT/'results/imports').glob('*/job-summary.json'))
    chosen=next((p for p in candidates if json.loads(p.read_text()).get('source_sha256')=='5cd20985e62718377dce8b5944725853820ec8806e5d7f398a559a21baf8b136'),None)
    if chosen is None:
        pytest.skip('Actual locally imported Dachshund job is unavailable')
    widget(app.selectbox,'Imported job').set_value(chosen).run(timeout=45)


def test_completion_notice_opens_the_named_finished_run_and_playback_tab(monkeypatch):
    from heat_sim import job_ui
    monkeypatch.setattr(job_ui,'_active',lambda:(None,None))
    folder=ROOT/'results/builds/20261115-133654-f1357a'
    if not (folder/'manifest.json').exists():
        pytest.skip('Actual locally completed job is unavailable')
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py')).run(timeout=45)
    _select_actual_dachshund(app)
    app.session_state['build_notice_folder']=str(folder)
    app.run(timeout=45)
    assert not app.exception
    assert any('Simulation finished'in message.value for message in app.success)
    widget(app.button,'View results').click().run(timeout=45)
    assert not app.exception
    assert app.session_state['job_active_tab']=='Build playback'
    assert widget(app.selectbox,'Simulation to view').value==folder


def test_zero_frame_stop_notice_never_claims_completed_frames_exist(monkeypatch):
    from heat_sim import job_ui
    monkeypatch.setattr(job_ui,'_active',lambda:(None,None))
    folder=ROOT/'results/builds/20261115-140700-99964e'
    if not (folder/'status.json').exists():
        pytest.skip('Actual locally stopped zero-frame example is unavailable')
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py')).run(timeout=45)
    _select_actual_dachshund(app)
    app.session_state['build_notice_folder']=str(folder)
    app.run(timeout=45)
    assert not app.exception
    assert any('No thermal frames were calculated'in str(c.value)for c in app.caption)
    assert not any(button.label=='View results'for button in app.button)
    notices=[str(item.value)for item in [*app.info,*app.warning,*app.caption]]
    assert not any('Completed converged frames remain available'in text for text in notices)
    widget(app.button,'Dismiss').click().run(timeout=45)
    assert not app.exception
    assert not any('This simulation was stopped'in str(item.value)for item in app.info)


def _isolated_normalized_job(tmp_path,monkeypatch,layers=1,slow=False):
    import json
    from heat_sim import job_ui
    from heat_sim.job_import import load_job,save_imported_job
    source=tmp_path/'isolated-job.json'
    first=[0,0,.01,0,120,.05] if slow else [0,0,.001,0,120,1.4]
    second=[first[2],0,0,.00006,100,1.]
    source.write_text(json.dumps(dict(format='heat-sim-job-v1',coordinate_units='m',speed_units='m/s',power_units='W_assumed',
        layer_thickness_m=60e-6,layers=[dict(index=i,z_m=i*60e-6,vectors=[first,second],process_ids=[0,6])for i in range(layers)])))
    results=tmp_path/'isolated-results';results.mkdir()
    save_imported_job(load_job(source),results/'imports'/'fixture')
    monkeypatch.setattr(job_ui,'RESULTS',results)
    return results


def test_full_preview_custom_scopes_and_step_guard_preset_are_explicit(tmp_path,monkeypatch):
    _isolated_normalized_job(tmp_path,monkeypatch,layers=6,slow=True)
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py')).run(timeout=45)
    assert not app.exception
    scope=widget(app.radio,'Simulation scope')
    assert scope.value=='Entire print (coarse)'
    assert widget(app.number_input,'First source layer (zero based)').value==0
    assert widget(app.number_input,'Last source layer (inclusive)').value==5
    assert widget(app.number_input,'Last source layer (inclusive)').disabled
    scope.set_value('Quick preview (up to 3 layers)').run(timeout=45)
    assert not app.exception
    assert widget(app.number_input,'Last source layer (inclusive)').value==2
    assert any('3 of 6 source layers'in str(item.value)for item in app.info)
    widget(app.radio,'Simulation scope').set_value('Custom layer range').run(timeout=45)
    assert not app.exception
    assert not widget(app.number_input,'Last source layer (inclusive)').disabled
    widget(app.radio,'Simulation scope').set_value('Entire print (coarse)').run(timeout=45)
    widget(app.number_input,'Maximum exposure grouping interval (ms)').set_value(.001).run(timeout=45)
    assert not app.exception
    assert widget(app.button,'Simulate current material').disabled
    assert any('step limit'in str(item.value)for item in app.error)
    widget(app.button,'Use entire-print coarse preset').click().run(timeout=45)
    assert not app.exception
    assert widget(app.number_input,'Maximum exposure grouping interval (ms)').value==100.
    assert widget(app.number_input,'Lateral thermal cells (mm)').value==2.
    assert not widget(app.button,'Simulate current material').disabled


def test_actual_tiny_worker_completion_opens_the_same_run(tmp_path,monkeypatch):
    import json,time
    results=_isolated_normalized_job(tmp_path,monkeypatch)
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py')).run(timeout=45)
    assert not app.exception
    widget(app.button,'Simulate current material').click().run(timeout=45)
    assert not app.exception
    requests=list((results/'builds').glob('*/request.json'))
    assert len(requests)==1
    folder=requests[0].parent
    deadline=time.monotonic()+10
    status={}
    while time.monotonic()<deadline:
        path=folder/'status.json'
        if path.exists():status=json.loads(path.read_text())
        if status.get('state')in ('complete','failed','cancelled','partial'):
            break
        time.sleep(.05)
    log=(folder/'worker.log').read_text()if(folder/'worker.log').exists()else''
    assert status.get('state')=='complete',f'{status}\n{log}'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert len(manifest['frames'])>=2
    app.run(timeout=45)
    assert not app.exception
    assert any('Simulation finished'in str(item.value)for item in app.success)
    widget(app.button,'View results').click().run(timeout=45)
    assert not app.exception
    assert app.session_state['job_active_tab']=='Build playback'
    assert widget(app.selectbox,'Simulation to view').value==folder
    assert app.get('iframe'),'The completed run should show its actual stored-field viewer'
