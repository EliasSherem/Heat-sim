"""The primary, single-screen print simulation workspace.

Import/physics/lifecycle/rendering stay in their existing numerical modules.
Only matching frozen configurations can populate the thermal viewport.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import workspace
from .build_solver import BuildConfig, estimate_build
from .job_import import load_job, save_imported_job, PROCESS_NAMES
from .materials import MATERIALS, ILLUSTRATIVE_ABSORPTIVITY
from .preset_ui import region_defaults, region_editor_rows, overrides_from_rows
from .run_catalog import list_runs, read_json
from .run_planning import estimate_run_plan
from .storage import ROOT, RESULTS, bundle_run
from .workspace_viewer import geometry_preview_html, thermal_workspace_html

UI_REVISION="single-workspace-1"

_STYLE="""<style>
.block-container{max-width:none;padding:3.2rem 1.3rem 2rem}h1{font-size:1.55rem!important;letter-spacing:-.04em;line-height:1.2!important;margin-bottom:0!important;padding:0!important}h3{font-size:1.05rem!important}
[data-testid="stHeader"]{height:2.3rem}[data-testid="stAppDeployButton"]{display:none}
.st-key-workspace_layout [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-controls){flex-wrap:nowrap!important;gap:1rem!important;align-items:flex-start}
.st-key-workspace_layout [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-controls)>[data-testid="stColumn"]:first-child{flex:0 0 clamp(180px,21vw,270px)!important;min-width:180px!important}
.st-key-workspace_layout [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-controls)>[data-testid="stColumn"]:nth-child(2){flex:1 1 0!important;min-width:230px!important}
.st-key-controls{border:1px solid #2b3848;border-radius:12px;background:#151d28;padding:.8rem;gap:.4rem!important}
.st-key-controls [data-testid="stVerticalBlock"]{gap:.55rem}.st-key-controls label p{font-size:.8rem!important}.st-key-controls [data-testid="stCaptionContainer"] p{font-size:.72rem;line-height:1.35}
.st-key-controls [data-testid="stFileUploaderDropzone"]{padding:.35rem;min-height:0}.st-key-controls [data-testid="stFileUploaderDropzoneInstructions"]{display:none}
.st-key-controls [data-testid="stFileUploaderFile"]{display:none}.st-key-controls [data-testid="stFileUploader"] section{padding:.25rem}
.st-key-controls [data-testid="stNumberInput"] input{font-size:.87rem}.st-key-controls button p{font-size:.8rem!important}
.st-key-stage [data-testid="stVerticalBlock"]{gap:.4rem}.st-key-stage iframe{border:1px solid #2b3a4c;border-radius:12px}
.ws-subtle{color:#99adc4;font-size:.79rem}.ws-assumptions{color:#e8bd85;font-size:.76rem;line-height:1.4}.ws-heading{display:flex;align-items:center;gap:12px;margin-bottom:8px}.ws-heading span{color:#97acc3;font-size:.8rem}
@media(max-width:700px){.block-container{padding:3rem .7rem 1.5rem}.st-key-workspace_layout [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-controls){gap:.6rem!important}.st-key-workspace_layout [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-controls)>[data-testid="stColumn"]:first-child{flex-basis:165px!important;min-width:165px!important}.st-key-workspace_layout [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-controls)>[data-testid="stColumn"]:nth-child(2){min-width:200px!important}.st-key-controls{padding:.55rem}}
</style>"""


def _job_changed():
    st.session_state.ws_view_run=None
    st.session_state.ws_import_error=None
    st.session_state.pop("ws_loaded_base",None)


def _upload():
    uploaded=st.session_state.get("ws_upload")
    if uploaded is None:return
    try:
        data=uploaded.getvalue()
        if len(data)>200*1024**2:raise ValueError("This file exceeds the 200 MiB browser upload limit.")
        location=RESULTS/"uploads"/sha256(data).hexdigest()[:20]
        location.mkdir(parents=True,exist_ok=True)
        source=location/Path(uploaded.name).name
        source.write_bytes(data)
        job=load_job(source)
        summary=save_imported_job(job,RESULTS/"imports"/job.source_sha256[:20])
        st.session_state.ws_job=str(summary)
        _job_changed()
    except Exception as exc:
        st.session_state.ws_import_error=f"Import failed: {exc}"


def _import_local(location):
    job=load_job(Path(location).expanduser())
    summary=save_imported_job(job,RESULTS/"imports"/job.source_sha256[:20])
    st.session_state.ws_job=str(summary)
    _job_changed()


def _open_advanced():
    st.session_state.ws_advanced=True


def _range(values):
    if not values or values[0] is None:return "not supplied"
    return f"{values[0]:g}" if abs(values[0]-values[1])<1e-9 else f"{values[0]:g}–{values[1]:g}"


def _advanced(job,base,material,prefix,comparison=None):
    """One collapsible place for all assumed physical and numerical inputs."""
    with st.expander("Advanced settings",expanded=False,key="ws_advanced",on_change="rerun"):
        st.caption("Whole-part heat history uses a coarse thermal grid. Refined local regions are separate calculations; neither mode establishes a qualified recipe or defect prediction.")
        a,b,c=st.columns(3)
        beam=a.number_input("Beam diameter · 1/e² (µm)",10.,1000.,base.beam_diameter_m*1e6,5.,key=prefix+"beam")*1e-6
        alpha=b.number_input(f"Absorptivity · {material}",.001,1.,float(ILLUSTRATIVE_ABSORPTIVITY[material]),.01,key=prefix+"alpha_"+material)
        delivered=c.number_input("Delivered / imported power",.01,2.,base.power_factor,.01,key=prefix+"delivered")
        a,b,c=st.columns(3)
        preheat=a.number_input("Preheat (°C)",0.,1000.,base.initial_k-273.15,5.,key=prefix+"preheat")+273.15
        substrate=b.selectbox("Substrate material",list(MATERIALS),index=list(MATERIALS).index(base.substrate),key=prefix+"substrate")
        recoat=c.number_input("Between layers incl. recoating (s)",0.,120.,3.,.1,key=prefix+"recoat")
        st.caption("The 3-second pause is your approximate inter-layer estimate. Jump timing, beam, delivered power, coupling and powder inputs remain assumed unless independently measured. Substrate is fixed during material comparison.")
        comparison_alpha=None
        if comparison:
            comparison_alpha=st.number_input(f"Absorptivity · {comparison}",.001,1.,float(ILLUSTRATIVE_ABSORPTIVITY[comparison]),.01,key=prefix+"compare_alpha_"+comparison)
            st.caption("The second material has its own independent optical input. Both runs use the same job, process settings and substrate.")
        a,b,c=st.columns(3)
        packing=a.number_input("Powder packing fraction",.2,1.,base.powder_packing,.01,key=prefix+"packing")
        powder_k=b.number_input("Powder conductivity (W/m·K)",.05,30.,base.powder_k_w_mk,.1,key=prefix+"powder_k")
        wave=c.number_input("Wavelength (nm)",100.,20000.,base.wavelength_nm,10.,key=prefix+"wavelength")
        a,b,c=st.columns(3)
        jump_speed=a.number_input("Jump speed (m/s)",.01,100.,base.jump_speed_m_s,.5,key=prefix+"jump_speed")
        jump_delay=b.number_input("Jump settling delay (µs)",0.,10000.,base.jump_delay_s*1e6,10.,key=prefix+"jump_delay")*1e-6
        cooldown=c.number_input("Cooling after print (s)",0.,120.,base.final_cooldown_s,.1,key=prefix+"cooldown")
        with st.expander("Region recipes and boundary properties"):
            st.caption("Unchecked fields retain each imported region recipe. Enabled absolute replacements apply before the main power/speed percentages. Original job files and geometric scan paths remain unchanged.")
            regions=region_defaults(job.source_sha256,job)
            table=st.data_editor(pd.DataFrame(region_editor_rows(regions,base.section_overrides)),hide_index=True,width="stretch",key=prefix+"regions_"+str(st.session_state.get("ws_load_generation",0)),
                disabled=["Section","Region"],column_config={
                    "Override power":st.column_config.CheckboxColumn("Replace power"),
                    "Override speed":st.column_config.CheckboxColumn("Replace speed"),
                    "Power (W)":st.column_config.NumberColumn(min_value=.001,max_value=10000.),
                    "Speed (mm/s)":st.column_config.NumberColumn(min_value=.001,max_value=100000.)})
            a,b,c=st.columns(3)
            convection=a.number_input("Convection (W/m²·K)",0.,1000.,base.convection_w_m2k,1.,key=prefix+"convection")
            emissivity=b.number_input("Emissivity",0.,1.,base.emissivity,.05,key=prefix+"emissivity")
            footprint=c.number_input("Subgrid scan footprint (µm)",1.,1000.,base.hatch_m*1e6,5.,key=prefix+"footprint")*1e-6
            st.caption("The footprint is a subgrid coverage assumption. Changing actual hatch spacing, layer thickness or scan strategy requires a newly sliced job.")
        st.markdown("**Calculation domain and resolution**")
        a,b=st.columns(2)
        extent=a.selectbox("Simulation extent",["Entire print","Selected layers"],key=prefix+"extent")
        intent=b.selectbox("Model scope",["Whole-part thermal history","Refined local region"],key=prefix+"intent")
        start=0;end=len(job.layers);warm=False
        if extent=="Selected layers":
            a,b=st.columns(2)
            start=int(a.number_input("First source layer",0,len(job.layers)-1,0,1,key=prefix+"first"))
            end=int(b.number_input("Last source layer (inclusive)",start,len(job.layers)-1,max(start,min(19,len(job.layers)-1)),1,key=prefix+"last"))+1
            if start:
                warm=st.checkbox("Assume earlier layers already built at uniform preheat",key=prefix+"warm")
                st.caption("A later start omits the previous thermal history. Use layer 0 for continuous accumulation.")
        a,b,c=st.columns(3)
        numerical=prefix+("local_" if intent=="Refined local region" else "global_")
        spacing=a.number_input("Lateral cells (mm)",.05,5.,.1 if intent=="Refined local region" else base.spacing_m*1000,.05,key=numerical+"spacing")*.001
        zspacing=b.number_input("Vertical cells (mm)",.02,2.,.03 if intent=="Refined local region" else base.z_spacing_m*1000,.02,key=numerical+"zspacing")*.001
        plate=c.number_input("Simulated plate depth (mm)",.05,20.,base.plate_depth_m*1000,.25,key=prefix+"plate")*.001
        a,b,c=st.columns(3)
        grouping=a.number_input("Exposure grouping (ms)",.001,1000.,1. if intent=="Refined local region" else base.max_event_s*1000,1.,key=numerical+"grouping",format="%.3f")*.001
        interval=b.number_input("Saved-frame interval (s)",.0001,120.,base.frame_interval_s,.1,key=prefix+"interval",format="%.4f")
        frames=int(c.number_input("Maximum saved frames",2,1000,base.max_frames,10,key=prefix+"frames"))
        region=None
        if intent=="Refined local region":
            bounds=job.bounds_m
            a,b,c=st.columns(3)
            cx=a.number_input("Region centre x (mm)",value=(bounds[0]+bounds[3])*500,step=.1,key=prefix+"region_x")
            cy=b.number_input("Region centre y (mm)",value=(bounds[1]+bounds[4])*500,step=.1,key=prefix+"region_y")
            width=c.number_input("Region width (mm)",.2,30.,2.,.2,key=prefix+"region_width")
            region=((cx-width/2)*.001,(cy-width/2)*.001,(cx+width/2)*.001,(cy+width/2)*.001)
            st.info("This local grid omits surrounding heat transfer. It remains a coarse homogenised model, rather than a resolved melt pool. Use the local track laboratory below for separate track-scale studies.")
        with st.expander("Limits and supplemental geometry"):
            a,b,c=st.columns(3)
            time_limit=a.number_input("Physical-time cap (s; 0 = none)",0.,100000.,0.,1.,key=prefix+"time_limit") or None
            vector_limit=int(b.number_input("Vector cap (0 = none)",0,2_000_000,0,100,key=prefix+"vector_limit")) or None
            steps=int(c.number_input("Thermal-step guard",1000,2_000_000,base.max_steps,1000,key=prefix+"max_steps"))
            a,b=st.columns(2)
            padding=a.number_input("Domain padding (mm)",0.,10.,base.padding_m*1000,.25,key=prefix+"padding")*.001
            supports=b.multiselect("Support-like source sections",sorted({int(pid) for layer in job.layers for pid in np.unique(layer.process_ids)}),default=list(base.support_sections),format_func=lambda i:f"{i}: {PROCESS_NAMES.get(i,'unknown')}",key=prefix+"supports")
            mesh=st.file_uploader("Optional aligned STL",type=["stl"],key=prefix+"stl")
            units=["mm","m","inch"]
            unit=st.selectbox("STL declared units",units,index=units.index({.001:"mm",1.:"m",.0254:"inch"}.get(base.geometry_unit_scale,"mm")),key=prefix+"stl_units")
            offsets=st.columns(3)
            translation=tuple(col.number_input(f"STL {axis} offset (mm)",value=float(offset)*1000,step=1.,key=prefix+"stl_"+axis)*.001 for col,axis,offset in zip(offsets,"xyz",base.geometry_translation_m))
            geometry=base.geometry_file
            if mesh:
                data=mesh.getvalue();folder=RESULTS/"geometry";folder.mkdir(parents=True,exist_ok=True)
                path=folder/(sha256(data).hexdigest()[:20]+".stl");path.write_bytes(data);geometry=str(path)
            st.caption("Supplemental geometry supplies a planned footprint; it never replaces the imported exposure paths. Alignment and units must be declared.")
        st.markdown("**Specialist tools**")
        st.caption("These tools retain the earlier calibration, uncertainty and preset studies. Ordinary printing and material comparison use the Run simulation action beside the viewer.")
        a,b=st.columns(2)
        a.link_button("Local track laboratory",url="http://127.0.0.1:8501/?tool=single-track",help="Separate refined single/two-track calculations, calibration and uncertainty.")
        b.link_button("Detailed preset studies",url="http://127.0.0.1:8501/?tool=studies",help="Saved preset laboratory and bounded multi-candidate/uncertainty studies.")
        st.caption("Material property tables, source references, import evidence, saved results and detailed exports are in Diagnostics.")
    return replace(base,absorptivity=alpha,beam_diameter_m=beam,power_factor=delivered,initial_k=preheat,substrate=substrate,
        recoat_time_s=recoat,powder_packing=packing,powder_k_w_mk=powder_k,wavelength_nm=wave,jump_speed_m_s=jump_speed,jump_delay_s=jump_delay,
        final_cooldown_s=cooldown,section_overrides=overrides_from_rows(table.to_dict("records")),convection_w_m2k=convection,emissivity=emissivity,hatch_m=footprint,
        layer_start=start,layer_end=end,warm_start_assumed=warm,spacing_m=spacing,z_spacing_m=zspacing,plate_depth_m=plate,
        max_event_s=grouping,frame_interval_s=interval,max_frames=frames,region_xy_m=region,time_limit_s=time_limit,max_vectors=vector_limit,
        max_steps=steps,padding_m=padding,support_sections=tuple(supports),geometry_file=geometry,geometry_unit_scale={"mm":.001,"m":1.,"inch":.0254}[unit],geometry_translation_m=translation),comparison_alpha


@st.cache_data(show_spinner=False,max_entries=8)
def _preview(summary,modified):
    job=workspace.cached_job(summary)
    return geometry_preview_html(job,view_key="job-"+job.source_sha256)


@st.cache_data(show_spinner=False,max_entries=6)
def _thermal(case_json,stamps,minimum,maximum,live,view_key,probe,auto_temperature_range):
    cases=json.loads(case_json)
    return thermal_workspace_html(cases,temperature_min_c=minimum,temperature_max_c=maximum,live=live,view_key=view_key,show_probe=probe,canvas_height=430,auto_temperature_range=auto_temperature_range)


def _cancel(folder):
    workspace.cancel_run(folder)


def _render_stage(summary,config,comparison,comparison_alpha,folder,active_folder,initial_busy,probe,minimum,maximum,auto_temperature_range):
    """Only this compact output region polls while a worker is alive."""
    job=workspace.cached_job(summary)
    status=workspace.read_status(active_folder) if active_folder else workspace.read_status(folder) if folder else {}
    state=status.get("state")
    busy=state in workspace.ACTIVE_STATES
    if initial_busy and not busy:
        st.rerun()
    if busy:
        active_request=read_json(Path(active_folder or folder)/"request.json")
        label=active_request.get("job_name",job.name)
        st.progress(float(np.clip(status.get("progress",0),0,1)),text=f"Solver progress · {status.get('progress',0):.0%}")
        a,b=st.columns([4,1])
        a.caption(f"{label} · {status.get('message','Preparing thermal calculation')}")
        b.button("Cancel",key="ws_cancel_"+Path(active_folder or folder).name,on_click=_cancel,args=(active_folder or folder,),disabled=(Path(active_folder or folder)/"CANCEL").exists())
    valid=bool(folder and workspace.matching_result(folder,job,config,comparison,comparison_alpha))
    view_status=workspace.read_status(folder) if folder else {}
    view_state=view_status.get("state")
    view_busy=view_state in workspace.ACTIVE_STATES
    if folder and not valid:
        st.warning("Settings changed — previous thermal results are outdated. Run simulation to calculate these settings.")
    cases=workspace.read_view_cases(folder) if valid else []
    available=any(case.get("available") for case in cases)
    if valid and view_state in ("failed","cancelled","partial"):
        notice="Calculation cancelled" if view_state=="cancelled" else "Calculation failed" if view_state=="failed" else "Comparison partly completed"
        st.warning(f"{notice}: {view_status.get('message','Inspect Diagnostics for details.')}")
        if available:st.caption("Showing only the calculated frames retained by this attempt.")
    if available:
        st.markdown("**Thermal results**"+(" · calculating" if view_busy else " · completed" if view_state=="complete" else " · available frames"))
        st.caption(f"{config.alloy}"+(f" / {comparison}" if comparison else "")+f" · layers {config.layer_start}–{(config.layer_end or len(job.layers))-1} of {len(job.layers)} · {config.spacing_m*1000:g} mm thermal grid")
        serial=[];stamps=[]
        for case in cases:
            serial.append({"folder":str(case["folder"]) if case.get("available") else None,
                "manifest":case["manifest"] if case.get("available") else None,"label":case["label"],
                "state":case.get("state"),"message":case.get("message")})
            if case.get("available"):
                path=Path(case["folder"])/"manifest.json"
                stamps.append((str(path),path.stat().st_mtime_ns,path.stat().st_size))
        try:
            viewer=_thermal(json.dumps(serial,sort_keys=True),tuple(stamps),minimum,maximum,view_busy,
                "job-"+job.source_sha256,probe,auto_temperature_range)
            st.iframe(viewer,height=860 if probe else 730,alt="Print simulation viewport")
        except (OSError,ValueError,KeyError,EOFError) as exc:
            st.info("A calculated field is being published. The view will retry automatically.")
            st.caption(str(exc))
    else:
        st.markdown("**Geometry preview**")
        scope=f"Entire print · {len(job.layers)} layers" if config.layer_start==0 and (config.layer_end or len(job.layers))==len(job.layers) else f"Run layers {config.layer_start}–{(config.layer_end or len(job.layers))-1} of {len(job.layers)}"
        st.caption("No calculated temperatures · "+scope+(" · preparing first field" if valid and view_busy else ""))
        st.iframe(_preview(str(summary),Path(summary).stat().st_mtime_ns),height=640,alt="Imported print geometry preview")


def _show_result(folder):
    """Restore frozen inputs before displaying a historical result as current."""
    request=read_json(Path(folder)/"request.json")
    cfg=dict(request.get("build_config") or read_json(Path(folder)/"manifest.json").get("config") or {})
    if not cfg:return
    if cfg.get("geometry_file") and not Path(cfg["geometry_file"]).is_absolute():cfg["geometry_file"]=str((Path(folder)/cfg["geometry_file"]).resolve())
    st.session_state.ws_loaded_base=cfg
    st.session_state.ws_load_generation=st.session_state.get("ws_load_generation",0)+1
    prefix="ws_"+st.session_state.ws_job_hash[:16]+"_"
    st.session_state.ws_material=cfg.get("alloy","17-4PH")
    for key,field,scale in [("power","power_scale",100.),("speed","speed_scale",100.),("beam","beam_diameter_m",1e6),
        ("delivered","power_factor",1.),("packing","powder_packing",1.),("powder_k","powder_k_w_mk",1.),
        ("wavelength","wavelength_nm",1.),("recoat","recoat_time_s",1.),("jump_speed","jump_speed_m_s",1.),
        ("jump_delay","jump_delay_s",1e6),("cooldown","final_cooldown_s",1.),("plate","plate_depth_m",1000.),
        ("interval","frame_interval_s",1.),("frames","max_frames",1.),("convection","convection_w_m2k",1.),
        ("emissivity","emissivity",1.),("footprint","hatch_m",1e6),("padding","padding_m",1000.),("max_steps","max_steps",1.)]:
        if field in cfg:st.session_state[prefix+key]=cfg[field]*scale if key not in ("frames","max_steps") else int(cfg[field])
    st.session_state[prefix+"preheat"]=cfg.get("initial_k",353.15)-273.15
    st.session_state[prefix+"substrate"]=cfg.get("substrate","17-4PH")
    st.session_state[prefix+"alpha_"+st.session_state.ws_material]=cfg.get("absorptivity",.35)
    st.session_state[prefix+"extent"]="Entire print" if cfg.get("layer_start",0)==0 and cfg.get("layer_end")==st.session_state.ws_layer_count else "Selected layers"
    st.session_state[prefix+"first"]=int(cfg.get("layer_start",0))
    st.session_state[prefix+"last"]=int(cfg.get("layer_end") or st.session_state.ws_layer_count)-1
    st.session_state[prefix+"warm"]=bool(cfg.get("warm_start_assumed"))
    st.session_state[prefix+"intent"]="Refined local region" if cfg.get("region_xy_m") else "Whole-part thermal history"
    numeric=prefix+("local_" if cfg.get("region_xy_m") else "global_")
    for key,field,scale in [("spacing","spacing_m",1000.),("zspacing","z_spacing_m",1000.),("grouping","max_event_s",1000.)]:
        if field in cfg:st.session_state[numeric+key]=cfg[field]*scale
    st.session_state[prefix+"time_limit"]=float(cfg.get("time_limit_s") or 0.)
    st.session_state[prefix+"vector_limit"]=int(cfg.get("max_vectors") or 0)
    st.session_state[prefix+"supports"]=list(cfg.get("support_sections",[]))
    st.session_state.ws_compare=False
    candidates=request.get("options",{}).get("candidates",[])
    if len(candidates)==1 and candidates[0].get("material")!=cfg.get("alloy"):
        st.session_state.ws_compare=True
        st.session_state.ws_comparison_material=candidates[0]["material"]
        st.session_state[prefix+"compare_alpha_"+candidates[0]["material"]]=candidates[0].get("absorptivity",ILLUSTRATIVE_ABSORPTIVITY[candidates[0]["material"]])
    st.session_state.ws_view_run=str(folder)


def render():
    st.markdown(_STYLE,unsafe_allow_html=True)
    st.title("Tracklab")
    st.markdown('<div class="ws-subtle">Print simulation · local thermal calculation</div>',unsafe_allow_html=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    imports=sorted((RESULTS/"imports").glob("*/job-summary.json"),key=lambda p:p.stat().st_mtime,reverse=True)
    imports=[str(p.resolve()) for p in imports if "unconfirmed" not in str(p)]
    with st.container(key="workspace_layout"):
        controls,stage_column=st.columns([1,3.4],gap="medium",wrap=False)
        with controls:
            with st.container(key="controls"):
                st.markdown("**Print setup**")
                if imports:
                    if st.session_state.get("ws_job") not in imports:st.session_state.ws_job=imports[0]
                    summary=st.selectbox("Job",imports,format_func=lambda p:read_json(p).get("name",Path(p).parent.name),key="ws_job",on_change=_job_changed)
                else:summary=None
                st.file_uploader("Import .htmjob / .mjob",type=["htmjob","mjob","json"],key="ws_upload",on_change=_upload,label_visibility="collapsed")
                if st.session_state.get("ws_import_error"):st.error(st.session_state.ws_import_error)
                if summary:
                    try:job=workspace.cached_job(summary);info=workspace.process_summary(job)
                    except Exception as exc:st.error(f"Cannot read this imported job: {exc}");return
                    st.session_state.ws_job_hash=job.source_sha256
                    st.session_state.ws_layer_count=len(job.layers)
                    st.caption("Job dimensions · "+" × ".join(f"{n:.1f}" for n in info["dimensions_mm"])+f" mm · {len(job.layers):,} layers")
                else:job=None
                material=st.selectbox("Material",list(MATERIALS),key="ws_material")
                prefix="ws_"+(job.source_sha256[:16] if job else "empty")+"_"
                power=st.number_input("Power · % of job",1.,300.,100.,5.,key=prefix+"power",help="100% retains every imported region's power. Other percentages scale those recipes together.")/100
                speed=st.number_input("Speed · % of job",10.,500.,100.,5.,key=prefix+"speed",help="100% retains imported scan speeds. Changing this percentage recalculates exposure duration.")/100
                if job:st.caption(f"Job recipes: {_range(info['power_w'])} W · {_range(info['speed_mm_s'])} mm/s")
                compare=st.checkbox("Compare material",key="ws_compare")
                comparison=None
                if compare:
                    targets=[name for name in MATERIALS if name!=material]
                    if st.session_state.get("ws_comparison_material") not in targets:st.session_state.ws_comparison_material=targets[0]
                    comparison=st.selectbox("Second material",targets,key="ws_comparison_material")
                action=st.empty()
                st.markdown('<div class="ws-assumptions">Beam, optical coupling and timing use assumed inputs.</div>',unsafe_allow_html=True)
                st.button("Edit Advanced ↓",key="ws_edit_advanced",on_click=_open_advanced)
        with stage_column:stage=st.container(key="stage")
    if job is None:
        with action.container():st.button("Run simulation",type="primary",disabled=True,width="stretch")
        with stage:
            st.markdown("**Import a print job to start**")
            st.info("Choose an .htmjob or .mjob file. Its geometry will appear here before calculating temperatures.")
        with st.expander("Diagnostics"):
            path=st.text_input("Existing local job path")
            if st.button("Import local job",disabled=not path):
                try:_import_local(path);st.rerun()
                except Exception as exc:st.error(str(exc))
        return
    base=BuildConfig(**st.session_state.ws_loaded_base) if st.session_state.get("ws_loaded_base") else workspace.default_config(job,material)
    config,comparison_alpha=_advanced(job,replace(base,alloy=material,power_scale=power,speed_scale=speed),material,prefix,comparison)
    active_folder,active_status=workspace.active_run(results_root=RESULTS)
    busy=bool(active_folder)
    if active_folder and read_json(active_folder/"request.json").get("job_source_sha256")==job.source_sha256:
        st.session_state.ws_view_run=str(active_folder)
    selected=st.session_state.get("ws_view_run")
    error=None;estimate=None;plan=None
    try:
        estimate=estimate_build(job,config);plan=estimate_run_plan(job,config)
        if estimate["cells"]>2_000_000 or estimate["max_saved_frame_bytes"]>8*1024**3:
            raise ValueError("Grid/frame selection exceeds the memory guard. Use coarser cells or a shorter range in Advanced.")
        if plan["step_limit_exceeded"]:raise ValueError(plan["reason"])
        if comparison:workspace.comparison_config(config,comparison,comparison_alpha).validate(job)
    except (ValueError,TypeError) as exc:error=str(exc)
    with action.container():
        if st.button("Run simulation",type="primary",disabled=busy or bool(error),width="stretch"):
            try:
                folder=workspace.start_run(summary,config,comparison,comparison_alpha,results_root=RESULTS,
                    provenance={"inter_layer_pause":"User estimate; approximately3seconds unless edited", "other_inputs":"Assumed until measured","workspace":"single simulation workspace"})
                st.session_state.ws_view_run=str(folder)
                st.rerun()
            except Exception as exc:st.error(f"Could not run simulation: {exc}")
        if busy:st.caption("Calculation running · progress and cancellation beside the print.")
        if error:st.error(error)
    with st.expander("Diagnostics",expanded=False,key="ws_diagnostics",on_change="rerun"):
        st.markdown("**Imported data**")
        st.caption(f"{info['vector_count']:,} emitting vectors · source SHA-256 {job.source_sha256}")
        st.dataframe(pd.DataFrame([{"Region":name,"Emitting vectors":count} for name,count in job.summary()["process_vector_counts"].items()]),hide_index=True,width="stretch")
        with st.expander("Decoding, units and format assumptions"):
            st.json(job.metadata)
        with st.expander("Import an existing local file"):
            location=st.text_input("Existing local job path",key="ws_local_path")
            if st.button("Import local job",disabled=not location):
                try:_import_local(location);st.rerun()
                except Exception as exc:st.error(f"Import failed: {exc}")
        with st.expander("Viewer and point-temperature options"):
            scale_mode=st.radio("Colour scale",["Fit current temperatures","Custom fixed range"],horizontal=True,key="ws_scale_mode")
            auto_temperature_range=scale_mode=="Fit current temperatures"
            a,b=st.columns(2)
            minimum=a.number_input("Shared temperature minimum (°C)",value=20.,step=5.,key="ws_temp_min",disabled=auto_temperature_range)
            maximum=b.number_input("Shared temperature maximum (°C)",value=1800.,step=10.,key="ws_temp_max",disabled=auto_temperature_range)
            st.caption("Automatic colours fit the current calculated field, then stay locked during playback. Fit colours in the viewport refits another moment. Both comparison materials always share the same range.")
            probe=False
            st.caption("Click the print for its stored cell temperature. Open View options in the viewport for the point-temperature graph. Both material viewers use one physical clock and this same fixed scale. Lines between saved samples are reading aids, not calculated intermediate temperatures.")
        st.markdown("**Saved simulations**")
        runs=list_runs(RESULTS/"builds",job.source_sha256)
        if runs:
            choices={str(run.folder):run for run in runs}
            run_choice=st.selectbox("Saved run",list(choices),format_func=lambda p:choices[p].label,key="ws_saved_run")
            st.button("Load saved run settings",on_click=_show_result,args=(run_choice,))
        if selected:
            status=workspace.read_status(selected)
            st.json(status,expanded=False)
            failures=read_json(Path(selected)/"report.json").get("failures",[])
            if failures:
                st.dataframe(pd.DataFrame([{"Case":row.get("preset_name") or row.get("name") or row.get("material","Case"),
                    "Error":row.get("error","Calculation failed")} for row in failures]),hide_index=True,width="stretch")
            if status.get("state")=="failed":
                log=Path(selected)/"worker.log"
                if log.exists():st.code(log.read_text()[-6000:],language="text")
            st.download_button("Download run configuration",json.dumps(read_json(Path(selected)/"request.json"),indent=2),"simulation-request.json","application/json")
            if st.button("Prepare detailed result export"):
                st.session_state.ws_bundle=(str(selected),bundle_run(Path(selected)))
            bundle=st.session_state.get("ws_bundle")
            if bundle and bundle[0]==str(selected):st.download_button("Download study ZIP",bundle[1],Path(selected).name+".zip","application/zip")
            with st.expander("Detailed result interpretation"):
                show=st.checkbox("Open saved-result analysis",key="ws_analyse")
                if show:
                    from .results_ui import render_results_insights
                    render_results_insights(runs,selected_folder=selected)
        with st.expander("Numerical configuration and convergence"):
            st.json(asdict(config))
            if estimate:st.json(estimate)
            if plan:st.json(plan)
            st.caption("Conservation checks do not establish mesh independence or experimental accuracy. Whole-part predictions remain sensitive to thermal grid, event grouping and plate depth.")
        with st.expander("Material properties, sources and model evidence"):
            for name in [material]+([comparison] if comparison else []):
                st.markdown(f"**{name}**")
                st.dataframe(pd.DataFrame(MATERIALS[name].table()),hide_index=True)
                st.caption(MATERIALS[name].notes)
            for filename in ("WORKSPACE_AUDIT.md","BUILD_MODEL.md","JOB_FORMAT.md","MATERIALS.md","VERIFICATION.md"):
                path=ROOT/"docs"/filename
                if path.exists():
                    with st.expander(filename):st.markdown(path.read_text())
    if auto_temperature_range:
        minimum,maximum=20.,1800.  # Safe fallback before the client fits the actual calculated fields.
    elif maximum<=minimum:
        maximum=minimum+1
        with stage:st.error("Temperature scale maximum must exceed minimum.")
    with stage:
        st.fragment(run_every="2s" if busy else None)(_render_stage)(summary,config,comparison,comparison_alpha,selected,active_folder,busy,probe,minimum,maximum,auto_temperature_range)
