"""Exercise named physical presets through save, simulation and interpretation."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest
from streamlit.testing.v1 import AppTest

from heat_sim import job_ui
from heat_sim.job_import import load_job
from heat_sim.preset_ui import region_editor_rows,overrides_from_rows,effective_region_rows
from heat_sim.presets import load_presets
from heat_sim.run_catalog import read_run
from test_job_app import _isolated_normalized_job,widget,ROOT


def open_app():
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py')).run(timeout=45)
    assert not app.exception
    app.session_state['job_active_tab']='Preset lab'
    app.run(timeout=45)
    assert not app.exception
    return app


def save_draft(app,name,**numbers):
    widget(app.text_input,'Preset name').set_value(name)
    for label,value in numbers.items():widget(app.number_input,label).set_value(value)
    widget(app.button,'Save preset').click().run(timeout=45)
    assert not app.exception


def imported_job(results):
    return load_job(next((results/'imports').glob('*/job-summary.json')))


def test_edit_save_reload_and_dispatch_exact_named_physical_values(tmp_path,monkeypatch):
    results=_isolated_normalized_job(tmp_path,monkeypatch,layers=2)
    calls=[]
    monkeypatch.setattr(job_ui,'_start',lambda kind,summary,config,options,provenance:calls.append(
        dict(kind=kind,summary=summary,config=config,options=deepcopy(options),provenance=provenance)))
    app=open_app()
    widget(app.number_input,'Ti-6Al-4V').set_value(.47).run(timeout=45)
    widget(app.selectbox,'Preset material').set_value('Ti-6Al-4V').run(timeout=45)
    assert not app.exception
    assert widget(app.number_input,'Preset effective absorptivity • assumed').value==.47
    save_draft(app,'Titanium slower contour',**{
        'Power (% of imported / overridden recipe)':85.,
        'Speed (% of imported / overridden recipe)':115.,
        'Preset preheat (°C)':125.,
        'Preset between-layer time incl. recoating (s)':2.7,
        'Preset 1/e² beam diameter (µm)':95.})
    job=imported_job(results)
    library=load_presets(results/'presets',job.source_sha256)
    assert len(library)==1
    saved=library[0]
    assert saved['name']=='Titanium slower contour'
    assert saved['material']=='Ti-6Al-4V' and saved['absorptivity']==.47
    assert saved['power_scale']==.85 and saved['speed_scale']==1.15
    assert saved['initial_k']==pytest.approx(398.15) and saved['recoat_time_s']==2.7
    assert saved['beam_diameter_m']==pytest.approx(95e-6)
    widget(app.text_input,'Preset study name').set_value('Compare before printing')
    widget(app.button,'Simulate preset comparison').click().run(timeout=45)
    assert not app.exception and len(calls)==1
    call=calls[0]
    assert call['kind']=='presets' and call['options']['study_name']=='Compare before printing'
    assert call['options']['candidates']==[saved]
    assert call['options']['reference']['material']=='17-4PH'
    assert call['options']['reference']['initial_k']==353.15
    assert call['config'].recoat_time_s==3.
    # A new browser session restores the on-disk library rather than inheriting
    # the editor's transient session state.
    fresh=open_app()
    assert any('Titanium slower contour'in option for option in widget(fresh.selectbox,'Preset to edit').options)
    assert widget(fresh.multiselect,'Candidate presets to simulate').value==[saved['id']]
    widget(fresh.selectbox,'Preset to edit').set_value(saved['id']).run(timeout=45)
    assert not fresh.exception
    assert widget(fresh.number_input,'Preset preheat (°C)').value==125.
    assert widget(fresh.number_input,'Power (% of imported / overridden recipe)').value==85.


def test_save_as_new_keeps_original_and_requires_distinct_names(tmp_path,monkeypatch):
    results=_isolated_normalized_job(tmp_path,monkeypatch)
    app=open_app()
    save_draft(app,'Successful reference')
    job=imported_job(results)
    original=load_presets(results/'presets',job.source_sha256)[0]
    widget(app.number_input,'Power (% of imported / overridden recipe)').set_value(90.)
    widget(app.button,'Save as new preset').click().run(timeout=45)
    assert not app.exception
    assert any('already has this name'in str(item.value) for item in app.error)
    assert len(load_presets(results/'presets',job.source_sha256))==1
    widget(app.text_input,'Preset name').set_value('Lower contour power')
    widget(app.button,'Save as new preset').click().run(timeout=45)
    assert not app.exception
    library=load_presets(results/'presets',job.source_sha256)
    assert len(library)==2
    assert next(item for item in library if item['id']==original['id'])==original
    copy=next(item for item in library if item['id']!=original['id'])
    assert copy['name']=='Lower contour power' and copy['power_scale']==.9
    # Saved-library reference can differ from the shared base config. Identity
    # must be correct while a run is preparing, before its report exists.
    folder=results/'builds'/'preparing';folder.mkdir(parents=True)
    (folder/'request.json').write_text(json.dumps(dict(kind='presets',build_config=dict(alloy='17-4PH'),
        job_source_sha256=job.source_sha256,job_name=job.name,
        options=dict(study_name='Independent titanium reference',reference=dict(name='Ti reference',material='Ti-6Al-4V'),
            candidates=[dict(name='Al alternative',material='AlSi10Mg')]))))
    (folder/'status.json').write_text(json.dumps(dict(state='queued')))
    preparing=read_run(folder)
    assert preparing.material=='Ti-6Al-4V' and preparing.target_materials==('AlSi10Mg',)
    assert preparing.study_name=='Independent titanium reference'


def test_real_tiny_preset_worker_to_insights_and_exact_playback(tmp_path,monkeypatch):
    results=_isolated_normalized_job(tmp_path,monkeypatch)
    app=open_app()
    save_draft(app,'Gentler power',**{'Power (% of imported / overridden recipe)':90.})
    widget(app.button,'Simulate preset comparison').click().run(timeout=45)
    assert not app.exception
    requests=list((results/'builds').glob('*/request.json'))
    assert len(requests)==1
    folder=requests[0].parent
    deadline=time.monotonic()+15
    status={}
    while time.monotonic()<deadline:
        path=folder/'status.json'
        if path.exists():status=json.loads(path.read_text())
        if status.get('state')in ('complete','partial','failed','cancelled'):break
        time.sleep(.05)
    log=(folder/'worker.log').read_text()if(folder/'worker.log').exists()else''
    assert status.get('state')=='complete',f'{status}\n{log}'
    report=json.loads((folder/'report.json').read_text())
    assert report['rows'][1]['preset_name']=='Gentler power'
    assert report['rows'][1]['comparable'] and report['rows'][1]['score'] is not None
    app.run(timeout=45)
    assert not app.exception
    widget(app.button,'Interpret these results').click().run(timeout=45)
    assert not app.exception
    assert app.session_state['job_active_tab']=='Results insights'
    assert widget(app.selectbox,'Results to interpret').value==folder
    assert any('Gentler power'in str(item.value)for item in app.success)
    assert widget(app.metric,'Reference').value=='Current job settings'
    table=next(item.value for item in app.dataframe if 'Preset / case'in item.value.columns)
    assert set(table['Preset / case'])=={'Current job settings','Gentler power'}
    assert table.loc[table['Preset / case']=='Gentler power','Comparison'].iloc[0]=='Scored'
    widget(app.button,'Adjust this preset').click().run(timeout=45)
    assert not app.exception
    assert app.session_state['job_active_tab']=='Preset lab'
    assert widget(app.text_input,'Preset name').value=='Gentler power revision'
    assert widget(app.number_input,'Power (% of imported / overridden recipe)').value==90.
    job=imported_job(results)
    assert len(load_presets(results/'presets',job.source_sha256))==1
    widget(app.button,'Open results insights').click().run(timeout=45)
    assert not app.exception
    assert widget(app.selectbox,'Results to interpret').value==folder
    widget(app.button,'Open this simulation in playback').click().run(timeout=45)
    assert not app.exception
    assert app.session_state['job_active_tab']=='Build playback'
    assert widget(app.selectbox,'Simulation to view').value==folder
    assert any('Gentler power'in option for option in widget(app.selectbox,'Material / candidate playback').options)
    assert app.get('iframe'),'Actual saved computed fields should render in playback'


def test_region_edit_helpers_preserve_imported_units_and_apply_overrides_before_scales():
    regions=[dict(section=6,region='contour',power_min=100.,power_max=120.,speed_min=1000.,speed_max=1400.)]
    editor=region_editor_rows(regions,{'6':{'speed_m_s':.8}})
    assert editor[0]['Speed (mm/s)']==800.
    assert editor[0]['Override speed'] and not editor[0]['Override power']
    assert overrides_from_rows(editor)=={'6':{'speed_m_s':.8}}
    editor[0].update({'Override power':True,'Power (W)':75.,'Speed (mm/s)':900.})
    overrides=overrides_from_rows(editor)
    assert overrides=={'6':{'power_w':75.,'speed_m_s':.9}}
    effective=effective_region_rows(regions,dict(power_scale=1.2,speed_scale=.8,section_overrides=overrides),.9)[0]
    assert effective['Preset command power (W)']=='90'
    assert effective['Assumed delivered power (W)']=='81'
    assert effective['Preset speed (mm/s)']=='720'
    editor[0].update({'Override power':False,'Override speed':False})
    assert overrides_from_rows(editor)=={}
