"""Live rendering of converged worker fields while a calculation continues.

The worker publishes its manifest only after a frame is saved. This reader
never predicts the next temperature or advances the physical clock itself.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import streamlit as st

from .build_viewer import playback_html
from .run_catalog import read_json


def _child(root,relative):
    try:
        target=(root/relative).resolve()
        return target if target.is_relative_to(root) else None
    except (TypeError,ValueError,OSError):
        return None


def _stamp(path):
    try:
        value=Path(path).stat()
        return value.st_mtime_ns,value.st_size
    except OSError:
        return 0,0


def _same(left,right):
    if isinstance(left,(int,float)) and isinstance(right,(int,float)):
        return math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-12)
    return left==right


def _case_name(root,folder,manifest,request,report):
    for row in [*report.get("rows",[]),*report.get("failures",[])]:
        if _child(root,row.get("case_folder"))==folder:
            name=row.get("preset_name") or row.get("name")
            if name:return str(name)
    cfg=manifest.get("config",{})
    options=report.get("options") or request.get("options",{})
    if request.get("kind")=="presets":
        # Current cases have a manifest before their first completed report row.
        # Match only the physical fields stored by each complete named preset.
        fields={"material":"alloy","substrate":"substrate","power_scale":"power_scale",
                "speed_scale":"speed_scale","initial_k":"initial_k","recoat_time_s":"recoat_time_s",
                "beam_diameter_m":"beam_diameter_m","absorptivity":"absorptivity",
                "powder_packing":"powder_packing","powder_k_w_mk":"powder_k_w_mk",
                "jump_speed_m_s":"jump_speed_m_s","jump_delay_s":"jump_delay_s",
                "section_overrides":"section_overrides"}
        presets=[options.get("reference") or {},*options.get("candidates",[])]
        for preset in presets:
            if preset.get("name") and all(_same(cfg.get(target),preset[source]) for source,target in fields.items() if source in preset):
                return str(preset["name"])
    material=cfg.get("alloy","Current material")
    if request.get("kind")=="transfer":
        return f"{material} · power ×{cfg.get('power_scale',1):g} / speed ×{cfg.get('speed_scale',1):g}"
    return str(material)


def latest_live_case(folder):
    """Return the active case and only published, existing temperature frames.

    Running case manifests take precedence over an earlier completed case,
    including while a new material/preset prepares its first field. Otherwise
    the newest manifest keeps the final result visible after the worker stops.
    """
    root=Path(folder).resolve()
    request=read_json(root/"request.json")
    report=read_json(root/"report.json")
    paths=[root/"manifest.json",*root.glob("cases/*/manifest.json")]
    candidates=[]
    for path in paths:
        manifest=read_json(path)
        if not manifest:continue
        candidates.append((manifest.get("status")=="running",_stamp(path)[0],path,manifest))
    if not candidates:return None
    _,modified,path,manifest=max(candidates,key=lambda value:(value[0],value[1]))
    case_folder=path.parent
    published=[]
    for entry in manifest.get("frames",[]):
        target=_child(case_folder,entry.get("file"))
        try:time=float(entry["time_s"])
        except (TypeError,ValueError,KeyError):continue
        if target is not None and _stamp(target)[1]>0 and math.isfinite(time):
            published.append(entry)
    published.sort(key=lambda entry:float(entry["time_s"]))
    return dict(folder=case_folder,manifest_path=path,manifest=manifest,
        manifest_stamp=modified,frames=published,frame_count=len(published),
        name=_case_name(root,case_folder,manifest,request,report),
        material=manifest.get("config",{}).get("alloy","Material not recorded"),
        job_name=manifest.get("job",{}).get("name") or request.get("job_name") or "Imported print",
        state=manifest.get("status","unknown"),
        latest_time_s=float(published[-1]["time_s"]) if published else None,
        layer_index=int(published[-1].get("layer_index",0)) if published else None)


@st.cache_data(show_spinner=False,max_entries=8)
def _live_html(folder,manifest_stamp,frame_stamps,manifest_json,minimum,maximum):
    """Do not reread/reencode arrays on status-only polling ticks."""
    return playback_html(Path(folder),json.loads(manifest_json),minimum,maximum,
                         max_points=9000,trajectory_layer=None,live=True)


def render_live_simulation(folder,status=None):
    """Render latest actual fields; caller supplies its periodic fragment."""
    st.subheader("Live simulation")
    st.caption("The 3D view follows the newest calculated temperature field while the solver works.")
    if folder is None:
        st.info("Start a simulation to watch the print being calculated. This window opens automatically for a new run.")
        return
    root=Path(folder).resolve()
    status=status or read_json(root/"status.json")
    state=status.get("state","preparing")
    if state in ("queued","running"):
        st.caption("Updating automatically while this calculation runs.")
    elif state=="complete":st.success("Calculation complete — showing its final calculated field.")
    elif state in ("failed","cancelled"):
        st.warning("This calculation stopped. Showing its most recent calculated field, if available.")
    elif state=="partial":st.warning("Some comparison cases failed. Showing the latest available calculated case.")
    case=latest_live_case(root)
    if case is None:
        st.info("Preparing scan timing and the thermal grid. The live print will appear as soon as the first converged temperature field is saved.")
        if status.get("message"):st.caption(status["message"])
        return
    st.write(f"**{case['job_name']}** · **{case['name']}** · {case['material']}")
    st.caption(f"Run {root.name} · case {case['folder'].name} · case status {case['state']}")
    if not case["frames"]:
        st.info("Preparing this case’s first converged temperature field. Completed earlier cases remain available in Build playback.")
        return
    left,middle,right=st.columns(3)
    left.metric("Calculated physical time",f"{case['latest_time_s']:,.3f} s")
    middle.metric("Current source layer",str(case["layer_index"]))
    right.metric("Calculated frames available",str(case["frame_count"]))
    latest=case["frames"][-2:]
    manifest={**case["manifest"],"frames":latest}
    stamps=tuple((_stamp(_child(case["folder"],entry["file"]))) for entry in latest)
    minimum=float(st.session_state.get("build_view_min",20.))
    maximum=float(st.session_state.get("build_view_max",1800.))
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum<=minimum:
        minimum,maximum=20.,1800.
    try:
        viewer=_live_html(str(case["folder"]),case["manifest_stamp"],stamps,
            json.dumps(manifest,sort_keys=True),minimum,maximum)
        st.iframe(viewer,height=1000,alt="Live calculated print temperatures")
    except (OSError,ValueError,KeyError,EOFError) as exc:
        st.info("A new temperature field is being published. The next live update will retry.")
        st.caption(str(exc))
    st.caption("The image updates when the solver saves a converged field; no future temperatures are generated. Coarse whole-print runs can save fields tens of simulated seconds apart. The completed timeline remains available in Build playback.")
