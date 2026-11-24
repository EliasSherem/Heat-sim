"""Live view reads published solver fields without inventing progress."""
import json
from pathlib import Path
import re

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from heat_sim import live_ui
from heat_sim.storage import atomic_json


def saved_case(folder,state="running",times=(0.,1.),material="17-4PH"):
    folder.mkdir(parents=True,exist_ok=True)
    shape=(2,2,3)
    z=np.array([-.00012,.00012,.00036])
    entries=[]
    for index,time in enumerate(times):
        path=folder/f"frames/{index:05d}.npz";path.parent.mkdir(exist_ok=True)
        active=np.ones(shape,dtype="f4")
        temperature=np.full(shape,353.15+index*100,dtype="f4")
        part=np.zeros(shape,bool);part[0,0,1:]=True
        plate=np.broadcast_to(z[None,None,:]<0,shape).copy()
        np.savez_compressed(path,temperature_k=temperature,active_fraction=active,
            consolidated_fraction=part.astype("f4"),part_mask=part,plate_mask=plate,
            support_mask=np.zeros(shape,bool),x_m=np.array([0.,.001]),
            y_m=np.array([0.,.001]),z_m=z,time_s=time)
        entries.append(dict(file=f"frames/{index:05d}.npz",time_s=time,
            layer_index=index,laser_position_m=[0.,0.,.00012],laser_on=True))
    manifest=dict(status=state,frames=entries,config=dict(alloy=material,
        layer_start=0,layer_end=3,power_scale=1.,speed_scale=1.,initial_k=353.15),
        job=dict(name="part.mjob",source_sha256="live-test"),physical_duration_s=120.,
        timeline_event_count=0)
    atomic_json(folder/"manifest.json",manifest)
    return manifest


def payload(html):
    match=re.search(r"const D=(.*?);const \$=",html,re.S)
    assert match
    return json.loads(match.group(1))


def test_latest_view_uses_only_published_existing_fields_and_skips_unsafe_paths(tmp_path):
    manifest=saved_case(tmp_path,times=(0.,1.,3.))
    manifest["frames"].extend([dict(file="frames/not-yet.npz",time_s=4.,layer_index=4),
        dict(file="../outside.npz",time_s=5.,layer_index=5)])
    (tmp_path.parent/"outside.npz").write_bytes(b"not a result")
    atomic_json(tmp_path/"manifest.json",manifest)
    case=live_ui.latest_live_case(tmp_path)
    assert case["latest_time_s"]==3.
    assert case["frame_count"]==3 and case["layer_index"]==2
    assert case["name"]=="17-4PH" and case["job_name"]=="part.mjob"


def test_live_follows_running_named_case_before_newer_finished_output(tmp_path):
    root=tmp_path
    atomic_json(root/"request.json",dict(kind="presets",options=dict(
        reference=dict(name="Successful reference",material="17-4PH"),
        candidates=[dict(name="Titanium trial",material="Ti-6Al-4V")])) )
    saved_case(root/"cases/titanium",material="Ti-6Al-4V")
    saved_case(root/"cases/reference",state="complete")
    live=live_ui.latest_live_case(root)
    assert live["folder"]==root/"cases/titanium"
    assert live["name"]=="Titanium trial"
    manifest=live["manifest"];manifest["status"]="complete"
    atomic_json(live["manifest_path"],manifest)
    final=live_ui.latest_live_case(root)
    assert final["folder"]==root/"cases/titanium"
    assert final["latest_time_s"]==1.


def test_new_active_case_without_frames_does_not_mislabel_old_temperatures(tmp_path):
    saved_case(tmp_path/"cases/old",state="complete")
    saved_case(tmp_path/"cases/new",times=(),material="316L")
    case=live_ui.latest_live_case(tmp_path)
    assert case["material"]=="316L" and case["frames"]==[]
    app=AppTest.from_string(f'''
from heat_sim.live_ui import render_live_simulation
render_live_simulation({str(tmp_path)!r},{{"state":"running"}})
''').run(timeout=30)
    assert not app.exception
    assert not app.get("iframe")
    assert any("first converged temperature field" in item.value for item in app.info)


def test_live_html_starts_at_latest_saved_time_and_keeps_exact_values(tmp_path):
    manifest=saved_case(tmp_path,times=(0.,1.,3.,8.))
    manifest={**manifest,"frames":manifest["frames"][-2:]}
    html=live_ui._live_html(str(tmp_path),1,((1,1),),json.dumps(manifest),20.,1800.)
    data=payload(html)
    assert data["live"] is True
    assert [frame["t"] for frame in data["frames"]]==[3.,8.]
    assert set(data["frames"][-1]["temperature"])=={380.}
    assert "time=D.live?end:begin" in html


def test_live_renderer_publishes_latest_field_then_retains_it_after_finish(tmp_path):
    manifest=saved_case(tmp_path,times=(0.,1.,3.))
    atomic_json(tmp_path/"status.json",dict(state="running"))
    source=f'''
from heat_sim.live_ui import render_live_simulation
render_live_simulation({str(tmp_path)!r})
'''
    app=AppTest.from_string(source).run(timeout=30)
    assert not app.exception and app.get("iframe")
    by_label={item.label:item.value for item in app.metric}
    assert by_label["Calculated physical time"]=="3.000 s"
    assert by_label["Current source layer"]=="2"
    assert by_label["Calculated frames available"]=="3"
    atomic_json(tmp_path/"status.json",dict(state="complete"))
    manifest["status"]="complete";atomic_json(tmp_path/"manifest.json",manifest)
    app.run(timeout=30)
    assert not app.exception and app.get("iframe")
    assert any("final calculated field" in item.value for item in app.success)
    assert next(item.value for item in app.metric if item.label=="Calculated physical time")=="3.000 s"


def test_preparation_waits_for_real_frame_and_cache_avoids_status_tick_reencoding(tmp_path,monkeypatch):
    app=AppTest.from_string(f'''
from heat_sim.live_ui import render_live_simulation
render_live_simulation({str(tmp_path)!r},{{"state":"running","message":"Reconstructing scan timing"}})
''').run(timeout=30)
    assert not app.exception and not app.get("iframe")
    assert any("first converged" in item.value for item in app.info)
    calls=[]
    monkeypatch.setattr(live_ui,"playback_html",lambda *args,**kwargs:calls.append((args,kwargs)) or "cached")
    live_ui._live_html.clear()
    assert live_ui._live_html(str(tmp_path),1,(),"{}",20.,1800.)=="cached"
    assert live_ui._live_html(str(tmp_path),1,(),"{}",20.,1800.)=="cached"
    assert len(calls)==1 and calls[0][1]["live"]
