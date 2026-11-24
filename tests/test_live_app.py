"""Live and full-print routes display actual fields from the intended run."""
from dataclasses import asdict
import json
import os
import re
import shutil
import time

from streamlit.testing.v1 import AppTest

from heat_sim.build_solver import BuildConfig,run_build
from heat_sim.job_import import load_job
from heat_sim.storage import atomic_json
from test_job_app import _isolated_normalized_job,widget,ROOT


def iframe_data(app):
    frames=app.get('iframe')
    assert frames,'The view must render an actual saved thermal field'
    source=frames[0].proto.srcdoc
    match=re.search(r'const D=(.*?);const \$=',source,re.S)
    assert match,'The stored viewer payload is missing'
    return json.loads(match.group(1))


def saved_build(tmp_path,monkeypatch,running=False):
    results=_isolated_normalized_job(tmp_path,monkeypatch,layers=2)
    summary=next((results/'imports').glob('*/job-summary.json'))
    job=load_job(summary)
    folder=results/'builds'/'live-fixture'
    folder.mkdir(parents=True)
    shutil.copytree(summary.parent,folder/'job')
    config=BuildConfig(absorptivity=.35,layer_end=2,recoat_time_s=.01,
        final_cooldown_s=.01,spacing_m=.0005,z_spacing_m=.00012,
        plate_depth_m=.00024,padding_m=.0005,max_event_s=.002,
        frame_interval_s=.002,max_frames=8)
    manifest=run_build(job,config,folder)
    atomic_json(folder/'request.json',dict(kind='build',build_config=asdict(config),
        job_source_sha256=job.source_sha256,job_name=job.name))
    if running:
        partial={**manifest,'status':'running','frames':manifest['frames'][:2]}
        atomic_json(folder/'manifest.json',partial)
    atomic_json(folder/'status.json',dict(state='running'if running else'complete',
        pid=os.getpid(),phase='integrating'if running else'finished',
        progress=.2 if running else 1.,message='Saved-field UI fixture'))
    return folder,manifest


def open_view(folder,tab):
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py'))
    app.session_state['job_active_tab']=tab
    app.session_state['build_last_started_folder']=str(folder)
    app.run(timeout=45)
    assert not app.exception
    return app


def test_running_live_view_advances_to_new_real_frame_and_retains_final_state(tmp_path,monkeypatch):
    folder,manifest=saved_build(tmp_path,monkeypatch,running=True)
    app=open_view(folder,'Live simulation')
    assert app.session_state['job_active_tab']=='Live simulation'
    assert widget(app.metric,'Calculated frames available').value=='2'
    initial=iframe_data(app)
    assert initial['live']is True
    assert initial['frames'][-1]['t']==manifest['frames'][1]['time_s']
    atomic_json(folder/'manifest.json',{**manifest,'status':'running','frames':manifest['frames'][:4]})
    app.run(timeout=45)
    assert not app.exception
    assert widget(app.metric,'Calculated frames available').value=='4'
    assert iframe_data(app)['frames'][-1]['t']==manifest['frames'][3]['time_s']
    atomic_json(folder/'manifest.json',manifest)
    atomic_json(folder/'status.json',dict(state='complete',progress=1.,message='Build completed'))
    app.run(timeout=45)
    assert not app.exception
    assert app.session_state['job_active_tab']=='Live simulation'
    assert any('showing its final calculated field'in str(item.value)for item in app.success)
    assert iframe_data(app)['frames'][-1]['t']==manifest['frames'][-1]['time_s']
    assert widget(app.metric,'Current source layer').value=='1'
    widget(app.button,'Open full playback for this run').click().run(timeout=45)
    assert not app.exception
    assert app.session_state['job_active_tab']=='Build playback'
    assert widget(app.selectbox,'Simulation to view').value==folder
    assert widget(app.radio,'Playback mode').value=='Full simulated print'
    assert iframe_data(app)['trajectory_scope']is None


def test_full_print_playback_is_default_and_layer_detail_is_explicit(tmp_path,monkeypatch):
    folder,manifest=saved_build(tmp_path,monkeypatch)
    app=open_view(folder,'Build playback')
    assert widget(app.radio,'Playback mode').value=='Full simulated print'
    assert not any(item.label=='Detailed source layer'for item in app.number_input)
    data=iframe_data(app)
    assert data['trajectory_scope']is None
    assert [entry['t']for entry in data['frames']]==[entry['time_s']for entry in manifest['frames']]
    assert data['frames'][-1]['layer']==1
    widget(app.radio,'Playback mode').set_value('One layer: detailed scan paths').run(timeout=45)
    widget(app.number_input,'Detailed source layer').set_value(1).run(timeout=45)
    assert not app.exception
    assert iframe_data(app)['trajectory_scope']['layer_index']==1
    widget(app.button,'Open latest full print').click().run(timeout=45)
    assert not app.exception
    assert widget(app.radio,'Playback mode').value=='Full simulated print'
    assert iframe_data(app)['trajectory_scope']is None


def test_starting_real_worker_opens_live_simulation_and_keeps_its_result(tmp_path,monkeypatch):
    results=_isolated_normalized_job(tmp_path,monkeypatch)
    app=AppTest.from_file(str(ROOT/'tests'/'legacy_app.py')).run(timeout=45)
    assert not app.exception
    widget(app.button,'Simulate current material').click().run(timeout=45)
    assert not app.exception
    assert app.session_state['job_active_tab']=='Live simulation'
    request=next((results/'builds').glob('*/request.json'))
    folder=request.parent
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        status=json.loads((folder/'status.json').read_text())
        if status.get('state')in('complete','partial','failed','cancelled'):break
        time.sleep(.05)
    assert status['state']=='complete',(folder/'worker.log').read_text()
    app.run(timeout=45)
    assert not app.exception
    assert app.session_state['job_active_tab']=='Live simulation'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert iframe_data(app)['frames'][-1]['t']==manifest['frames'][-1]['time_s']
