"""Use the same viewport for import, thermal results, changes and comparison."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import time

import psutil
import pytest
from streamlit.testing.v1 import AppTest

from heat_sim import simulation_ui, workspace
from heat_sim.build_solver import run_build
from heat_sim.job_import import load_job, save_imported_job
from heat_sim.storage import atomic_json


ROOT=Path(__file__).resolve().parents[1]


def widget(elements,label):
    return next(item for item in elements if item.label==label)


def isolated_job(tmp_path,monkeypatch,layers=2):
    source=tmp_path/'real-normalized-paths.json'
    source.write_text(json.dumps(dict(format='heat-sim-job-v1',coordinate_units='m',speed_units='m/s',
        power_units='W_assumed',layer_thickness_m=60e-6,
        layers=[dict(index=i,z_m=i*60e-6,vectors=[[0.,0.,.001,0.,120.,1.4],
            [.001,0.,0.,.00006,100.,1.]],process_ids=[0,6])for i in range(layers)])))
    results=tmp_path/'results'
    summary=save_imported_job(load_job(source),results/'imports'/'fixture')
    monkeypatch.setattr(simulation_ui,'RESULTS',results)
    monkeypatch.setattr(workspace,'RESULTS',results)
    return results,summary,load_job(summary)


def open_app():
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=45)
    assert not app.exception
    return app


def viewport(app):
    frames=app.get('iframe')
    assert len(frames)==1,'One main viewport should remain in the same workspace'
    source=frames[0].proto.srcdoc
    match=re.search(r'const W=(.*?);const \$=',source,re.S)
    assert match,'The imported/saved viewport payload should be present'
    return json.loads(match.group(1))


def saved_run(results,summary,job,state='complete'):
    folder=results/'builds'/('saved-'+state);folder.mkdir(parents=True)
    shutil.copytree(summary.parent,folder/'job')
    config=workspace.default_config(job)
    manifest=run_build(job,config,folder)
    atomic_json(folder/'request.json',dict(kind='build',build_config=asdict(config),
        job_source_sha256=job.source_sha256,job_name=job.name))
    status=dict(state=state,progress=1. if state=='complete'else .4,message='Saved thermal fixture')
    if state=='running':
        status.update(pid=os.getpid(),process_started_epoch_s=psutil.Process().create_time())
        manifest={**manifest,'status':'running','frames':manifest['frames'][:2]}
        atomic_json(folder/'manifest.json',manifest)
    atomic_json(folder/'status.json',status)
    return folder,manifest


def test_imported_geometry_is_automatic_and_primary_controls_are_one_workspace(tmp_path,monkeypatch):
    _,summary,job=isolated_job(tmp_path,monkeypatch)
    app=open_app()
    assert not app.tabs and not app.sidebar.radio
    assert sum(item.label=='Run simulation'for item in app.button)==1
    assert widget(app.selectbox,'Material').options==list(simulation_ui.MATERIALS)
    assert widget(app.number_input,'Power · % of job').value==100.
    assert widget(app.number_input,'Speed · % of job').value==100.
    assert widget(app.selectbox,'Job').value==str(summary.resolve())
    data=viewport(app)
    assert data['mode']=='geometry'
    assert data['cases'][0]['source_sha256']==job.source_sha256
    assert data['cases'][0]['segments'],'Actual imported scan geometry must appear without a separate preview action'
    assert any('no calculated temperatures'in str(item.value).lower()for item in app.caption)
    assert not any('emitting vectors'in str(item.value).lower()for item in app.caption if 'Job dimensions'in str(item.value))


def test_material_or_speed_change_invalidates_saved_temperatures_in_same_viewport(tmp_path,monkeypatch):
    results,summary,job=isolated_job(tmp_path,monkeypatch)
    folder,_=saved_run(results,summary,job)
    app=AppTest.from_file(str(ROOT/'app.py'))
    app.session_state['ws_view_run']=str(folder)
    app.run(timeout=45)
    assert not app.exception and viewport(app)['mode']=='thermal'
    original=json.loads((folder/'manifest.json').read_text())['config']
    widget(app.number_input,'Speed · % of job').set_value(110.).run(timeout=45)
    assert not app.exception
    assert viewport(app)['mode']=='geometry'
    assert any('outdated'in str(item.value)for item in app.warning)
    widget(app.number_input,'Speed · % of job').set_value(100.).run(timeout=45)
    assert viewport(app)['mode']=='thermal'
    widget(app.number_input,'Beam diameter · 1/e² (µm)').set_value(85.).run(timeout=45)
    assert viewport(app)['mode']=='geometry'
    assert any('outdated'in str(item.value)for item in app.warning)
    widget(app.number_input,'Beam diameter · 1/e² (µm)').set_value(80.).run(timeout=45)
    assert viewport(app)['mode']=='thermal'
    widget(app.number_input,'Absorptivity · 17-4PH').set_value(.4).run(timeout=45)
    assert viewport(app)['mode']=='geometry'
    assert any('outdated'in str(item.value)for item in app.warning)
    widget(app.number_input,'Absorptivity · 17-4PH').set_value(.35).run(timeout=45)
    assert viewport(app)['mode']=='thermal'
    widget(app.selectbox,'Material').set_value('316L').run(timeout=45)
    assert not app.exception and viewport(app)['mode']=='geometry'
    assert any('outdated'in str(item.value)for item in app.warning)
    assert json.loads((folder/'manifest.json').read_text())['config']==original
    widget(app.button,'Load saved run settings').click().run(timeout=45)
    assert not app.exception
    assert widget(app.selectbox,'Material').value=='17-4PH'
    assert viewport(app)['mode']=='thermal'


def test_colour_range_changes_reuse_unchanged_solver_results(tmp_path,monkeypatch):
    results,summary,job=isolated_job(tmp_path,monkeypatch)
    folder,_=saved_run(results,summary,job)
    app=AppTest.from_file(str(ROOT/'app.py'))
    app.session_state['ws_view_run']=str(folder)
    app.run(timeout=45)
    assert not app.exception
    original=viewport(app)
    assert original['auto_temperature_range'] is True
    assert widget(app.number_input,'Shared temperature maximum (°C)').disabled
    widget(app.radio,'Colour scale').set_value('Custom fixed range').run(timeout=45)
    widget(app.number_input,'Shared temperature minimum (°C)').set_value(75.)
    widget(app.number_input,'Shared temperature maximum (°C)').set_value(150.).run(timeout=45)
    assert not app.exception
    fixed=viewport(app)
    assert fixed['mode']=='thermal' and fixed['auto_temperature_range'] is False
    assert (fixed['vmin'],fixed['vmax'])==(75.,150.)
    assert fixed['cases'][0]['frames']==original['cases'][0]['frames']
    assert not any('outdated' in str(item.value) for item in app.warning)
    widget(app.radio,'Colour scale').set_value('Fit current temperatures').run(timeout=45)
    assert not app.exception and viewport(app)['auto_temperature_range'] is True


def test_one_run_action_returns_real_worker_temperatures_in_same_viewport(tmp_path,monkeypatch):
    results,_,_=isolated_job(tmp_path,monkeypatch)
    app=open_app()
    assert viewport(app)['mode']=='geometry'
    widget(app.button,'Run simulation').click().run(timeout=45)
    assert not app.exception
    folder=Path(app.session_state['ws_view_run'])
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        status=workspace.read_status(folder)
        if status.get('state')not in workspace.ACTIVE_STATES:break
        time.sleep(.05)
    assert status['state']=='complete',(folder/'worker.log').read_text()
    app.run(timeout=45)
    assert not app.exception and not app.tabs
    data=viewport(app)
    assert data['mode']=='thermal'and len(data['cases'])==1
    manifest=json.loads((folder/'manifest.json').read_text())
    assert data['cases'][0]['frames'][-1]['t']==manifest['frames'][-1]['time_s']
    assert any('Thermal results'in str(item.value)for item in app.markdown)
    assert sum(item.label=='Run simulation'for item in app.button)==1


def test_optional_comparison_runs_two_independent_materials_with_shared_scale(tmp_path,monkeypatch):
    _,_,_=isolated_job(tmp_path,monkeypatch,layers=1)
    app=open_app()
    widget(app.checkbox,'Compare material').check().run(timeout=45)
    assert not app.exception
    assert widget(app.selectbox,'Second material').value=='316L'
    widget(app.button,'Run simulation').click().run(timeout=45)
    assert not app.exception
    folder=Path(app.session_state['ws_view_run'])
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        status=workspace.read_status(folder)
        if status.get('state')not in workspace.ACTIVE_STATES:break
        time.sleep(.05)
    assert status['state']=='complete',(folder/'worker.log').read_text()
    app.run(timeout=45)
    assert not app.exception
    data=viewport(app)
    assert len(data['cases'])==2
    assert {case['alloy']for case in data['cases']}=={'17-4PH','316L'}
    assert all(case['frames']for case in data['cases'])
    assert data['vmin']==20. and data['vmax']==1800.
    assert all(case['vmin']==data['vmin']and case['vmax']==data['vmax']for case in data['cases'])
    assert not app.tabs
    widget(app.checkbox,'Compare material').uncheck().run(timeout=45)
    assert not app.exception and viewport(app)['mode']=='geometry'
    assert any('outdated'in str(item.value)for item in app.warning)


def test_running_frames_are_visible_and_cancelled_case_keeps_actual_partial_fields(tmp_path,monkeypatch):
    results,summary,job=isolated_job(tmp_path,monkeypatch)
    folder,manifest=saved_run(results,summary,job,'running')
    app=open_app()
    data=viewport(app)
    assert data['mode']=='thermal'and data['live']is True
    assert widget(app.button,'Run simulation').disabled
    widget(app.button,'Cancel').click().run(timeout=45)
    assert not app.exception
    marker=json.loads((folder/'CANCEL').read_text())
    assert marker['run_id']==folder.name
    atomic_json(folder/'status.json',dict(state='cancelled',progress=.4,message='Cancelled after converged thermal step'))
    atomic_json(folder/'manifest.json',{**manifest,'status':'cancelled'})
    app.run(timeout=45)
    assert not app.exception
    assert any('Calculation cancelled'in str(item.value)for item in app.warning)
    assert viewport(app)['cases'][0]['frames'][-1]['t']==manifest['frames'][-1]['time_s']
    assert not widget(app.button,'Run simulation').disabled


def test_launch_failure_is_useful_and_never_displays_fabricated_results(tmp_path,monkeypatch):
    isolated_job(tmp_path,monkeypatch)
    def failed(*args,**kwargs):raise RuntimeError('CPU worker could not start: deliberate test failure')
    monkeypatch.setattr(workspace,'start_run',failed)
    app=open_app()
    widget(app.button,'Run simulation').click().run(timeout=45)
    assert not app.exception
    assert any('CPU worker could not start'in str(item.value)for item in app.error)
    assert viewport(app)['mode']=='geometry'


def test_failed_calculation_message_is_shown_with_geometry_instead_of_fake_heat(tmp_path,monkeypatch):
    results,summary,job=isolated_job(tmp_path,monkeypatch)
    folder=results/'builds'/'failed-before-first-field';folder.mkdir(parents=True)
    shutil.copytree(summary.parent,folder/'job')
    atomic_json(folder/'request.json',dict(kind='build',build_config=asdict(workspace.default_config(job)),
        job_source_sha256=job.source_sha256,job_name=job.name))
    atomic_json(folder/'status.json',dict(state='failed',progress=.1,message='Thermal integration did not converge'))
    app=AppTest.from_file(str(ROOT/'app.py'))
    app.session_state['ws_view_run']=str(folder)
    app.run(timeout=45)
    assert not app.exception
    assert any('Thermal integration did not converge'in str(item.value)for item in app.warning)
    assert viewport(app)['mode']=='geometry'
