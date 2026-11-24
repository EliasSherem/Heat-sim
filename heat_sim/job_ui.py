"""Local job import, material-transfer studies and numerical build playback."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import streamlit as st

from .build_solver import BuildConfig, estimate_build
from .build_viewer import playback_html, probe_history, read_build_manifest
from .job_import import load_job, save_imported_job, JobImportError, PROCESS_NAMES
from .materials import MATERIALS, ILLUSTRATIVE_ABSORPTIVITY
from .storage import ROOT, RESULTS, atomic_json, snapshot_source, bundle_run
from .run_catalog import read_run, list_runs, playable_runs, choose_result, list_cases, read_json, latest_full_print
from .run_planning import estimate_run_plan
from .preset_ui import render_preset_lab, region_defaults, effective_region_rows
from .presets import plan_presets, preset_from_config
from .results_ui import render_results_insights
from .live_ui import render_live_simulation

UI_REVISION = "live-full-playback-5"
ACTIVE_STATES = {"queued", "running"}


def _queue_view_run(folder):
    st.session_state.pending_view_run = str(folder)
    st.session_state.pending_job_tab = "Build playback"
    st.session_state["viewer_mode_"+Path(folder).name]="Full simulated print"


def _queue_live_run(folder):
    st.session_state.build_last_started_folder=str(folder)
    st.session_state.pending_job_tab="Live simulation"


def _queue_insights_run(folder=None):
    if folder is not None:st.session_state.pending_insights_run=str(folder)
    st.session_state.pending_job_tab="Results insights"


def _queue_preset_from_result(case):
    prefix="preset_"+st.session_state.current_job_sha[:16]+"_"
    draft=preset_from_config(BuildConfig(**case["config"]),(case["name"]+" revision")[:80])
    st.session_state[prefix+"imported_draft"]=draft
    st.session_state.pending_job_tab="Preset lab"


def _job_picker_key(job_hash):
    return "playback_run_" + job_hash[:16]


def _use_whole_job_preset(job_hash):
    st.session_state["run_scope_"+job_hash]="Entire print (coarse)"
    st.session_state["resolution_"+job_hash]="Whole-build thermal history"
    for field,value in {"spacing":2.,"zspacing":1.5,"plate":3.,"event":100.,"capture":30.,"frames":80}.items():
        st.session_state[f"{field}_Entire print (coarse)_Whole-build thermal history_{job_hash}"]=value


@st.cache_resource(show_spinner=False)
def cached_job(path, modified):
    return load_job(path)


def _job_from_cache(path):
    return cached_job(str(path), Path(path).stat().st_mtime_ns)


def _import(path):
    progress = st.progress(0., text="Inspecting job format")
    try:
        job = load_job(path, progress=lambda f,msg: progress.progress(float(f),text=msg))
        folder = RESULTS / "imports" / job.source_sha256[:20]
        summary = save_imported_job(job, folder)
        st.session_state.job_summary_path = str(summary)
        st.success(f"Imported {job.name}: {sum(len(layer.vectors) for layer in job.layers):,} emitting vectors")
        return summary
    finally:
        progress.empty()


def _active():
    value = st.session_state.get("build_active_folder")
    if not value:
        runs=list_runs(RESULTS/"builds",st.session_state.get("current_job_sha"))
        live=next((run for run in runs if run.state in ACTIVE_STATES),None)
        if live is None:return None,None
        value=str(live.folder)
        st.session_state.build_active_folder=value
    folder = Path(value)
    path = folder / "status.json"
    status = read_json(path) if path.exists() else {"state":"queued","progress":0.,"message":"Starting local calculation"}
    return folder, status


def _start(kind, summary, config, options, provenance):
    old, status = _active()
    if status and status["state"] in ACTIVE_STATES:
        return
    # Guard every entry point, including callbacks and replayed configurations.
    if kind=="presets":
        plans=plan_presets(_job_from_cache(summary),config,options)
        blocked=next((item for item in plans if item["plan"]["step_limit_exceeded"]),None)
        plan=blocked["plan"] if blocked else plans[0]["plan"]
    else:plan=estimate_run_plan(_job_from_cache(summary),config)
    if plan["step_limit_exceeded"]:
        st.error(plan.get("reason") or plan.get("failure_reason") or "The selected inputs exceed the solver step budget. Use the whole-job coarse preset or reduce the range.")
        return
    folder = RESULTS / "builds" / (datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    # Freeze the decoded source once per study, shared by all material cases.
    input_folder = folder / "job"
    shutil.copytree(Path(summary).parent, input_folder)
    geometry_file = getattr(config,"geometry_file",None)
    if geometry_file:
        target = folder / "supplement.stl"
        shutil.copy2(geometry_file,target)
        config = replace(config,geometry_file="supplement.stl")
    request = {"kind":kind, "job_path": "job/job-summary.json", "build_config":asdict(config),
               "options":options, "input_provenance":provenance,"created_utc":datetime.now(timezone.utc).isoformat(),
               "job_name":_job_from_cache(summary).name,"job_source_sha256":_job_from_cache(summary).source_sha256}
    atomic_json(folder/"request.json",request)
    atomic_json(folder/"status.json",{"state":"queued","progress":0.,"message":"Starting CPU enthalpy simulation"})
    with (folder/"worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable,"-m","heat_sim.worker",str(folder/"request.json")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    st.session_state.build_active_folder = str(folder)
    st.session_state.build_active_pid = process.pid
    st.session_state.build_selected_folder = str(folder)
    st.session_state.build_last_started_folder = str(folder)
    st.session_state.pop("build_notice_folder",None)
    st.session_state.pop("build_last_status",None)
    st.session_state.pending_job_tab = "Live simulation"
    st.rerun()


def _progress():
    folder,status = _active()
    if not folder:
        notice=st.session_state.get("build_notice_folder")
        if not notice:return
        info=read_run(notice)
        if not info:return
        current_job=st.session_state.get("current_job_sha")
        if current_job and info.job_sha256!=current_job:
            st.session_state.pop("build_notice_folder",None)
            return
        with st.container(border=True):
            if info.state=="complete":st.success("Simulation finished — results are ready")
            elif info.state=="partial":st.warning("Comparison partly finished — some cases failed")
            elif info.state=="cancelled":st.info("This simulation was stopped")
            else:st.error("This simulation did not finish")
            st.write(f"**{info.job_name}** · {info.material_label} · {info.layer_label}")
            if info.study_name:st.write(f"**{info.study_name}**")
            st.caption(info.message)
            if info.has_results:
                st.caption(f"{info.frame_count} saved thermal frames · {info.completed_case_count} completed material cases")
                st.button("View results",key="notice_view_"+info.run_id,on_click=_queue_view_run,args=(info.folder,),type="primary")
                st.button("Interpret these results",key="notice_insights_"+info.run_id,on_click=_queue_insights_run,args=(info.folder,))
            else:st.caption("No thermal frames were calculated. Your previous completed results are unchanged.")
            if st.button("Dismiss",key="dismiss_notice_"+info.run_id):
                st.session_state.pop("build_notice_folder",None)
                st.rerun()
        return
    if status["state"] in ACTIVE_STATES:
        info=read_run(folder)
        st.markdown("**Current calculation**")
        if info:st.write(f"{info.job_name} · {info.material_label} · {info.layer_label}")
        if info and info.study_name:st.write(f"**{info.study_name}**")
        phase=status.get("phase", "preparing")
        st.caption(f"{phase.replace('_',' ').capitalize()} · run {folder.name}")
        st.progress(float(np.clip(status.get("progress",0),0,1)),text=status.get("message","Calculating"))
        pid = status.get("pid") or st.session_state.get("build_active_pid")
        try:
            gone = pid and (not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            gone = True
        if gone:
            # A worker can finish between the read and the process check.
            fresh=read_json(folder/"status.json")
            if fresh.get("state") not in ACTIVE_STATES:st.rerun()
            else:
                atomic_json(folder/"status.json",{**fresh,"state":"failed","phase":"failed","message":"Worker exited unexpectedly; inspect worker.log"})
                st.rerun()
        if info and info.has_results:
            st.button("View calculated frames",key="active_view_"+info.run_id,on_click=_queue_view_run,args=(folder,))
        st.button("Open live simulation",key="active_live_"+folder.name,on_click=_queue_live_run,args=(folder,))
        stopping=(folder/"CANCEL").exists()
        if st.button("Stop this run",key="stop_build_"+folder.name,disabled=stopping):
            atomic_json(folder/"CANCEL",{"run_id":folder.name,"requested_utc":datetime.now(timezone.utc).isoformat(),"source":"user_interface","action":"Stop this run"})
            st.info("Stopping this run. Completed frames, if any, will be kept.")
        elif stopping:st.caption("Stop requested — waiting for the calculation to stop safely.")
    else:
        st.session_state.build_notice_folder = str(folder)
        info=read_run(folder)
        if info and info.has_results and info.job_sha256==st.session_state.get("current_job_sha"):
            st.session_state.pending_result_selection = str(folder)
        st.session_state.pop("build_active_folder",None)
        st.session_state.pop("build_active_pid",None)
        st.rerun()


def render():
    # Migrate the previous unscoped warning; its original run remains in history.
    st.session_state.pop("build_last_status",None)
    pending_tab=st.session_state.pop("pending_job_tab",None)
    if pending_tab:st.session_state.job_active_tab=pending_tab
    elif "job_active_tab" not in st.session_state and st.session_state.get("build_selected_folder"):
        st.session_state.job_active_tab="Build playback"
    st.markdown("""<style>.block-container{padding-top:2rem;max-width:1480px}h1{letter-spacing:-.045em}.stMetric{background:#182332;border:1px solid #2b3a4e;border-radius:10px;padding:13px}.note{color:#a5b8d0;font-size:16px;line-height:1.6}.eyebrow{color:#ffac79;font-size:11px;letter-spacing:.18em;font-weight:700}</style>""",unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">TRACKLAB / JOB-DRIVEN MATERIAL TRANSFER</div>',unsafe_allow_html=True)
    st.title("One print job. Compare materials.")
    st.markdown('<div class="note">Import your scan paths, save print presets, and compare their heat response on the same geometry. Keep heat history across tracks, pauses and layers, then use Results insights to choose the next settings to test.</div>',unsafe_allow_html=True)
    st.warning("Exploratory thermal simulation. The whole-build model resolves heat accumulation and cooling; it cannot establish microscopic melt-pool dimensions, porosity, keyhole stability or a qualified material conversion.")
    activity_slot=st.empty()
    RESULTS.mkdir(exist_ok=True)
    with st.sidebar:
        st.markdown("### ◈ Tracklab")
        st.caption("LOCAL CPU • NO PRINTER CONNECTION")
    import_tab,setup_tab,preset_tab,calculate_tab,live_tab,playback_tab,insights_tab,comparison_tab,evidence_tab = st.tabs(["Import job","Materials & settings","Preset lab","Calculate","Live simulation","Build playback","Results insights","Transfer comparison","Evidence"],key="job_active_tab",on_change="rerun")
    with import_tab:
        st.subheader("Upload the job you already print")
        uploaded = st.file_uploader("Print job",type=["htmjob","mjob","json"],help="Supported binary/JSON variants are identified by contents. Unsupported variants are reported; imported objects are never executed.")
        if st.button("Import uploaded job",type="primary",disabled=uploaded is None):
            try:
                data = uploaded.getvalue()
                if len(data)>200*1024**2:
                    raise ValueError("Browser uploads are limited to 200 MiB")
                folder = RESULTS/"uploads"/sha256(data).hexdigest()[:20]
                folder.mkdir(parents=True,exist_ok=True)
                source = folder/Path(uploaded.name).name
                source.write_bytes(data)
                _import(source)
                st.rerun()
            except Exception as exc:
                st.error(f"Import stopped: {exc}")
        with st.expander("Import an existing local file"):
            location = st.text_input("Local job path",value="/Users/elias/Downloads/Metal_printer/17-4 Test.htmjob")
            if st.button("Inspect and import local job"):
                try:
                    _import(Path(location).expanduser())
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import stopped: {exc}")
        imports = sorted((RESULTS/"imports").glob("*/job-summary.json"),key=lambda p:p.stat().st_mtime,reverse=True)
        # The unrelated format fixture is never silently substituted for the user's job.
        imports = [p for p in imports if "unconfirmed" not in str(p)]
        current = st.session_state.get("job_summary_path")
        if imports:
            default = next((i for i,p in enumerate(imports) if str(p)==current),0)
            summary = st.selectbox("Imported job",imports,index=default,format_func=lambda p: read_json(p)["name"]+" · "+str(read_json(p).get("layer_count","?"))+" layers")
            st.session_state.job_summary_path = str(summary)
        else:
            st.info("Import a job to start. A normalized geometry/path JSON can supplement an unsupported job variant; see Evidence for the schema.")
            return
    try:
        job = _job_from_cache(summary)
        info = job.summary()
    except Exception as exc:
        st.error(f"Cannot read imported job: {exc}")
        return
    st.session_state.current_job_sha=job.source_sha256
    # Poll only while a calculation is active. Idle polling made the app flash
    # its generic Running/Stop indicators even after a simulation had finished.
    _,poll_status=_active()
    poll_every="1s" if poll_status and poll_status.get("state") in ACTIVE_STATES else None
    with activity_slot.container():st.fragment(run_every=poll_every)(_progress)()
    current_runs=list_runs(RESULTS/"builds",job.source_sha256)
    pending_insights=st.session_state.pop("pending_insights_run",None)
    if pending_insights:
        insight_run=read_run(pending_insights)
        if insight_run and insight_run.job_sha256==job.source_sha256:
            st.session_state["insights_run_"+job.source_sha256[:16]]=Path(pending_insights).resolve()
            if insight_run.state not in ("complete","partial"):
                st.session_state["insights_failed_"+job.source_sha256[:16]]=True
    pending_result=st.session_state.pop("pending_result_selection",None)
    pending_view=st.session_state.pop("pending_view_run",None)
    if pending_view or pending_result:
        desired=Path(pending_view or pending_result).resolve()
        chosen_info=read_run(desired)
        if chosen_info and chosen_info.has_results:
            st.session_state.build_selected_folder=str(desired)
            st.session_state[_job_picker_key(job.source_sha256)]=desired
            if chosen_info.state in ("failed","cancelled"):st.session_state["show_stopped_"+job.source_sha256]=True
            if chosen_info.job_sha256!=job.source_sha256:st.session_state["show_other_"+job.source_sha256]=True
    with import_tab:
        a,b,c,d = st.columns(4)
        a.metric("Layers",f"{info['layer_count']:,}")
        b.metric("Emitting vectors",f"{info['vector_count']:,}")
        c.metric("Layer thickness",f"{job.layer_thickness_m*1e6:g} µm")
        d.metric("Laser-on time only",f"{info['mark_time_s']/60:.2f} min")
        st.caption("Laser-on duration is computed from actual vector length / imported speed. It excludes laser-off motion, delays and recoating. Source units and reconstructed order remain subject to the import notes.")
        st.dataframe(pd.DataFrame([{"region":name,"emitting_vectors":count} for name,count in info["process_vector_counts"].items()]),hide_index=True)
        st.info("Legacy section 16 is labelled support by the recovered software mapping; its physical role in your 17-4 job is not confirmed. Reserved/disabled sections are reported and excluded from exposure.")
        with st.expander("Exactly what was imported, inferred or assumed",expanded=True):
            st.json(job.metadata.get("provenance",{}))
            for warning in job.metadata.get("warnings",[]):
                st.write("• "+str(warning))
            st.json({"source_sha256":job.source_sha256,"format":job.format,"parts":job.metadata.get("parts"),"recipes":job.metadata.get("recipes"),"reserved_ignored_points":job.metadata.get("reserved_ignored_points"),"zero_length_vectors_omitted":job.metadata.get("zero_length_vectors_omitted")},expanded=False)
        with st.expander("Actual scan geometry preview"):
            layer_id = st.slider("Preview source layer",0,len(job.layers)-1,0) if len(job.layers)>1 else 0
            layer = job.layers[layer_id]
            from matplotlib.collections import LineCollection
            fig,ax = plt.subplots(figsize=(9,4.5))
            if len(layer.vectors):
                segments = layer.vectors[:,:4].reshape((-1,2,2))*1000
                ax.add_collection(LineCollection(segments,linewidths=.35,colors="#ffb070"));ax.autoscale()
            ax.set(xlabel="Job x (mm)",ylabel="Job y (mm)",title=f"Imported exposure vectors • layer {layer_id} • decoded vector order")
            ax.set_aspect("equal",adjustable="datalim");st.pyplot(fig);plt.close(fig)

    with setup_tab:
        st.subheader("Establish the successful baseline, then choose target alloys")
        a,b=st.columns(2)
        names=list(MATERIALS)
        baseline = a.selectbox("Current material used for this print",names,index=names.index("17-4PH"))
        targets = b.multiselect("Simulate the same print in",[n for n in names if n!=baseline],default=["316L"] if baseline!="316L" else ["17-4PH"])
        successful = st.checkbox("This job has printed successfully in the current material",value=False)
        st.caption("A successful baseline is useful evidence; measured tracks/temperatures are still needed for calibration. Selecting a material does not establish its optical coupling or powder properties.")
        st.markdown("**Effective absorptivity by alloy • independent assumed values**")
        cols=st.columns(4);alphas={}
        for column,(name,value) in zip(cols,ILLUSTRATIVE_ABSORPTIVITY.items()):
            alphas[name]=column.number_input(name,.01,.99,float(value),.01,key="build_alpha_"+name)
        measured_abs = st.multiselect("Alloys with measured/calibrated absorptivity",names,default=[])
        a,b,c=st.columns(3)
        wavelength = a.number_input("Wavelength (nm) • assumed",100.,20000.,1070.,10.)
        beam = b.number_input("Physical 1/e² intensity diameter (µm) • assumed",10.,1000.,80.,5.)*1e-6
        power_factor = c.number_input("Delivered / imported power • assumed",.01,2.,1.,.01)
        a,b,c=st.columns(3)
        preheat = a.number_input("Preheat / initial temperature (°C) • assumed",0.,1000.,80.,5.)+273.15
        substrate = b.selectbox("Build plate / substrate material • assumed",names,index=names.index(baseline))
        follow_substrate = c.checkbox("Use each print alloy as its substrate",value=False)
        st.caption("The selected substrate is held constant across comparisons unless the checkbox is selected. Wavelength is recorded; absorptivity is supplied independently. Beam profile is Gaussian.")
        a,b=st.columns(2)
        packing = a.number_input("Powder packing fraction • assumed",.2,1.,.55,.01)
        powder_k = b.number_input("Effective powder k (W/m·K) • assumed",.05,30.,1.,.1)
        a,b,c=st.columns(3)
        recoat = a.number_input("Between layers, including recoating (s) • user estimate",0.,120.,3.,.1)
        jump_speed = b.number_input("Laser-off jump speed (m/s) • assumed",.01,100.,5.,.5)
        jump_delay = c.number_input("Jump settling delay (µs) • assumed",0.,10000.,100.,10.)*1e-6
        st.caption("Your approximately 3-second inter-layer estimate is applied once per layer transition, including empty source layers. It is not added again as a separate recoater time.")
        process_ids = sorted({int(pid) for layer in job.layers for pid in np.unique(layer.process_ids)})
        supports = st.multiselect("Regions to display as supports • interpretation",process_ids,default=[16] if 16 in process_ids else [],format_func=lambda i:f"{i}: {PROCESS_NAMES.get(i,'unknown')}")
        geometry_file=None;geometry_unit=1e-3;geometry_translation=(0.,0.,0.)
        with st.expander("Optional supplemental STL geometry"):
            st.caption("STL supplies the planned body footprint; it does not replace the imported exposure paths or consolidate unscanned powder. Closed, aligned meshes are required; thin features below the thermal grid remain unresolved.")
            mesh_upload=st.file_uploader("Supplemental geometry",type=["stl"],key="supplement_stl")
            unit=st.selectbox("STL coordinate units (declared)",["mm","m","inch"])
            geometry_unit={"mm":.001,"m":1.,"inch":.0254}[unit]
            offsets=st.columns(3)
            geometry_translation=tuple(col.number_input(f"STL {axis} translation (mm)",value=0.,step=1.)*.001 for col,axis in zip(offsets,"xyz"))
            if mesh_upload:
                data=mesh_upload.getvalue();folder=RESULTS/"geometry";folder.mkdir(exist_ok=True)
                path=folder/(sha256(data).hexdigest()[:20]+".stl");path.write_bytes(data);geometry_file=str(path)

    active_folder,active_status=_active()
    busy=bool(active_status and active_status["state"] in ACTIVE_STATES)
    with calculate_tab:
        st.subheader("Choose what to calculate")
        scope=st.radio("Simulation scope",["Entire print (coarse)","Quick preview (up to 3 layers)","Custom layer range"],key="run_scope_"+job.source_sha256,horizontal=True)
        study_mode=st.radio("Resolution intent",["Whole-build thermal history","Refined local region"],horizontal=True,key="resolution_"+job.source_sha256)
        a,b=st.columns(2)
        scope_fixed=scope!="Custom layer range"
        start_layer=a.number_input("First source layer (zero based)",0,len(job.layers)-1,0,1,disabled=scope_fixed,key="first_"+scope+job.source_sha256)
        end_default=len(job.layers)-1 if scope=="Entire print (coarse)" else min(len(job.layers)-1,int(start_layer)+2)
        end_layer=b.number_input("Last source layer (inclusive)",int(start_layer),len(job.layers)-1,end_default,1,disabled=scope_fixed,key="last_"+scope+job.source_sha256)
        selected_layers=int(end_layer)-int(start_layer)+1
        st.info(f"This run will calculate **{selected_layers} of {len(job.layers)} source layers** ({int(start_layer)}–{int(end_layer)})."+(" This is a short preview." if scope.startswith("Quick") else ""))
        warm_start=st.checkbox("Assume earlier layers already built at uniform preheat",value=False,disabled=start_layer==0)
        if start_layer:
            st.warning("Starting after layer 0 omits the earlier thermal history. Enable the explicit warm-start assumption, or start at layer 0 for a continuous history.")
        a,b,c=st.columns(3)
        whole_preset=scope=="Entire print (coarse)" and study_mode=="Whole-build thermal history"
        numerical_key=scope+"_"+study_mode+"_"+job.source_sha256
        spacing=a.number_input("Lateral thermal cells (mm)",.05,5.,2. if whole_preset else 1.5 if study_mode=="Whole-build thermal history" else .1,.05,key="spacing_"+numerical_key)
        zspacing=b.number_input("Vertical thermal cells (mm)",.02,2.,1.5 if whole_preset else .24 if study_mode=="Whole-build thermal history" else .03,.02,key="zspacing_"+numerical_key)
        plate_depth=c.number_input("Simulated plate thickness (mm) • assumed",.05,20.,3. if whole_preset else 1.5,.25,key="plate_"+numerical_key)
        if whole_preset:
            st.caption("Entire-print preset: coarse heat accumulation and cooling. It uses 2 mm lateral cells, 1.5 mm vertical cells and 100 ms exposure groups; these are numerical approximations, not measured machine specifications.")
        region=None
        if study_mode=="Refined local region":
            bounds=job.bounds_m
            first = next((layer.vectors[0] for layer in job.layers[int(start_layer):int(end_layer)+1] if len(layer.vectors)),None)
            cx=float(first[0]*1000) if first is not None else (bounds[0]+bounds[3])*500
            cy=float(first[1]*1000) if first is not None else (bounds[1]+bounds[4])*500
            cols=st.columns(3)
            region_x=cols[0].number_input("Region centre x (mm)",value=float(cx),step=.1)
            region_y=cols[1].number_input("Region centre y (mm)",value=float(cy),step=.1)
            width=cols[2].number_input("Region width and height (mm)",.2,20.,2.,.2)
            region=((region_x-width/2)*.001,(region_y-width/2)*.001,(region_x+width/2)*.001,(region_y+width/2)*.001)
            st.warning("A cropped local region omits heat transfer to the surrounding part. Source events remain in order; only their in-domain energy is deposited. Inspect the captured-energy fraction and boundary sensitivity.")
        a,b,c=st.columns(3)
        max_event=a.number_input("Maximum exposure grouping interval (ms)",.001,1000.,100. if whole_preset else 5.,1.,key="event_"+numerical_key)*.001
        capture=b.number_input("Thermal-frame interval (s)",.001,60.,30. if whole_preset else .1 if study_mode=="Whole-build thermal history" else .002,.001,key="capture_"+numerical_key)
        max_frames=c.number_input("Maximum stored frames",2,1000,80,10,key="frames_"+numerical_key)
        a,b,c=st.columns(3)
        time_limit=a.number_input("Physical-time limit (s; 0 = no limit)",0.,100000.,0.,1.)
        vector_limit=b.number_input("Exposure-vector limit (0 = all selected)",0,2_000_000,0,100)
        cooldown=c.number_input("Cooling after selected print (s)",0.,120.,.5,.1)
        config_values=dict(alloy=baseline,substrate=substrate,absorptivity=alphas[baseline],power_factor=power_factor,
            beam_diameter_m=beam,wavelength_nm=wavelength,initial_k=preheat,powder_packing=packing,powder_k_w_mk=powder_k,
            spacing_m=spacing*.001,z_spacing_m=zspacing*.001,plate_depth_m=plate_depth*.001,
            layer_start=int(start_layer),layer_end=int(end_layer)+1,warm_start_assumed=bool(warm_start),
            jump_speed_m_s=jump_speed,jump_delay_s=jump_delay,recoat_time_s=recoat,final_cooldown_s=cooldown,
            max_event_s=max_event,frame_interval_s=capture,max_frames=int(max_frames),support_sections=tuple(supports),
            time_limit_s=time_limit or None,max_vectors=int(vector_limit) or None,region_xy_m=region)
        if geometry_file:
            config_values.update(geometry_file=geometry_file,geometry_unit_scale=geometry_unit,geometry_translation_m=geometry_translation)
        config=BuildConfig(**config_values)
        estimate=None
        plan=None
        try:
            estimate=estimate_build(job,config)
            a,b,c=st.columns(3)
            a.metric("Thermal grid cells",f"{estimate['cells']:,}")
            b.metric("Working arrays • estimate",f"{estimate['working_array_bytes']/1024**2:.1f} MiB")
            c.metric("Maximum raw saved frames",f"{estimate['max_saved_frame_bytes']/1024**2:.1f} MiB")
            st.caption(estimate["note"])
            st.write(f"Selected exposures: {estimate['selected_vector_count']:,}. Laser-on time estimate: {estimate['nominal_mark_time_s']:.3f}s, before jumps/recoating and any physical-time truncation. Full-job runtime is not inferred from this count.")
            regions=sorted({int(pid) for layer in job.layers[int(start_layer):int(end_layer)+1] for pid in np.unique(layer.process_ids)})
            st.caption("Regions in this source-layer window: "+", ".join(f"{pid} ({PROCESS_NAMES.get(pid,'unknown')})" for pid in regions))
            if regions==[16]:
                st.info("This initial window contains only section 16 exposures. It tests the early support-like region, not the main-body infill or full part.")
            plan=estimate_run_plan(job,config)
            if plan["step_limit_exceeded"]:
                st.error(f"This configuration needs at least {plan['minimum_thermal_steps']:,} thermal steps, above the {config.max_steps:,}-step limit. Laser-off jumps and recoating are included in this check. Use the entire-print coarse preset or calculate a shorter range.")
                st.button("Use entire-print coarse preset",on_click=_use_whole_job_preset,args=(job.source_sha256,))
            else:
                st.caption(f"Step-budget check passed: at least {plan['minimum_thermal_steps']:,} steps; limit {config.max_steps:,}. Adaptive solving can add work.")
        except Exception as exc:
            st.error(str(exc))
        provenance={"recoating":{"value_s":recoat,"source":"User estimate: around 3 seconds between layers including recoating","accuracy":"approximate, editable"},
            "baseline_reported_successful":successful,"absorptivity_measured_materials":measured_abs,
            "other_machine_inputs":"assumed until independently measured","study_mode":study_mode,
            "supplemental_geometry_units":geometry_unit,"supplemental_translation_m":geometry_translation}
        a,b=st.columns(2)
        ready=estimate is not None and plan is not None and not plan["step_limit_exceeded"]
        if a.button("Simulate current material",type="primary",disabled=busy or not ready):
            _start("build",summary,config,{},provenance)
        if b.button("Compare selected materials at identical job settings",disabled=busy or not ready or not targets):
            _start("transfer",summary,config,{"baseline_material":baseline,"target_materials":targets,"absorptivities":alphas,"evaluate_candidates":False,"substrate_follows_material":follow_substrate},provenance)
        with st.expander("Screen power / speed adjustments"):
            st.caption("Multipliers apply to each region's imported power and speed, preserving region differences. Speed changes recalculate exposure duration and heat accumulation. The closest simulated thermal history is a screening candidate, not an optimal printer recipe.")
            vary_power=st.checkbox("Vary power as well as speed",value=False)
            power_bounds=st.slider("Power multiplier bounds",.5,1.5,(.9,1.1),.05,disabled=not vary_power)
            speed_bounds=st.slider("Speed multiplier bounds",.5,1.5,(.9,1.1),.05)
            uncertainty=st.checkbox("Also test target-absorptivity endpoints",value=False)
            alpha_fraction=st.slider("Relative absorptivity range (± fraction)",.05,.5,.2,.05,disabled=not uncertainty)
            p_scales=np.linspace(*power_bounds,3).tolist() if vary_power else [1.]
            s_scales=np.linspace(*speed_bounds,3).tolist()
            candidate_ready=ready
            if ready:
                slow_plan=estimate_run_plan(job,replace(config,speed_scale=min([1.,*s_scales])))
                if slow_plan["step_limit_exceeded"]:
                    candidate_ready=False
                    st.error("The slowest speed candidate exceeds the step budget. Increase exposure grouping or reduce the layer range before running the comparison.")
            if st.button("Calculate bounded transfer candidates",disabled=busy or not candidate_ready or not targets):
                _start("transfer",summary,config,{"baseline_material":baseline,"target_materials":targets,"absorptivities":alphas,
                    "evaluate_candidates":True,"power_scales":p_scales,"speed_scales":s_scales,"substrate_follows_material":follow_substrate,
                    "evaluate_uncertainty":uncertainty,"uncertainty_absorptivity_ranges":{n:[max(.01,alphas[n]*(1-alpha_fraction)),min(.99,alphas[n]*(1+alpha_fraction))] for n in targets}},provenance)
        latest=choose_result(playable_runs(current_runs))
        if latest:
            latest_info=read_run(latest)
            with st.container(border=True):
                st.markdown("**Most recent completed results for this job**")
                st.write(f"{latest_info.material_label} · {latest_info.layer_label} · {latest_info.frame_count} saved frames")
                st.caption(latest_info.label)
                st.button("View latest completed simulation",on_click=_queue_view_run,args=(latest,),type="secondary")
                st.button("Interpret latest completed simulation",on_click=_queue_insights_run,args=(latest,))

    with preset_tab:
        render_preset_lab(job,config,RESULTS/"presets",alphas,estimate is not None,busy,
            on_start=lambda options:_start("presets",summary,config,options,provenance),
            on_insights=_queue_insights_run)

    with live_tab:
        if live_tab.open:
            live_folder=active_folder or st.session_state.get("build_last_started_folder")
            live_info=read_run(live_folder) if live_folder else None
            if live_info and live_info.job_sha256!=job.source_sha256:
                if live_info.state not in ACTIVE_STATES:live_folder=None
                else:st.info("The active calculation uses another imported job. Its live view identifies that source below.")
            if not live_folder:live_folder=choose_result(playable_runs(current_runs))
            if live_folder:
                live_status=read_json(Path(live_folder)/"status.json")
                live_poll="2s" if live_status.get("state") in ACTIVE_STATES else None
                st.fragment(run_every=live_poll)(render_live_simulation)(live_folder)
                st.button("Open full playback for this run",on_click=_queue_view_run,args=(live_folder,),key="live_playback_"+Path(live_folder).name)
            else:
                st.subheader("Live simulation")
                st.info("Start a simulation in Calculate or Preset lab. This view opens automatically and follows its calculated 3D temperatures while the solver runs.")

    selected_folder=None;transfer_report=None;cases=[]
    with playback_tab:
        st.subheader("View a simulation")
        st.caption("Results for the imported job appear here, newest completed first. Stopped and failed attempts are kept separately in run history.")
        a,b=st.columns(2)
        include_incomplete=a.checkbox("Show incomplete runs",key="show_stopped_"+job.source_sha256)
        include_other=b.checkbox("Show results from other jobs",key="show_other_"+job.source_sha256)
        catalog=list_runs(RESULTS/"builds",None if include_other else job.source_sha256)
        available=playable_runs(catalog,include_incomplete=include_incomplete)
        latest_finished=choose_result(playable_runs(current_runs))
        full_print=latest_full_print(current_runs)
        if full_print:
            st.button("Open latest full print",on_click=_queue_view_run,args=(full_print,),key="open_latest_full_playback",type="primary")
        elif latest_finished:
            st.button("Open latest completed run",on_click=_queue_view_run,args=(latest_finished,),key="open_latest_playback")
        if available:
            preferred=st.session_state.get("build_selected_folder")
            chosen=choose_result(available,preferred=preferred,active=active_folder)
            folders=[item.folder for item in available]
            by_folder={item.folder:item for item in available}
            if chosen is None:chosen=folders[0]
            picker_key=_job_picker_key(job.source_sha256)
            if st.session_state.get(picker_key) not in folders:st.session_state[picker_key]=chosen
            selected_folder=st.selectbox("Simulation to view",folders,key=picker_key,format_func=lambda p:by_folder[p].label)
            st.session_state.build_selected_folder=str(selected_folder)
            selected_run=by_folder[selected_folder]
            with st.container(border=True):
                st.markdown(f"**Viewing: {selected_run.job_name}**")
                st.write(f"{selected_run.material_label} · {selected_run.layer_label}")
                st.caption(f"{selected_run.status_label} · {selected_run.frame_count} saved frames · run {selected_run.run_id}")
                if selected_run.study_name:st.write(f"**{selected_run.study_name}**")
                if selected_run.layer_start or selected_run.layer_end and selected_run.total_layers and selected_run.layer_end<selected_run.total_layers:
                    st.info("This saved run covers only a selected layer range. Use Open latest full print above to view the completed entire print." if full_print else "This saved run covers only a selected layer range. Calculate the entire print to create a full-print playback.")
                st.button("Interpret this simulation",on_click=_queue_insights_run,args=(selected_folder,),key="playback_insights_"+selected_run.run_id)
                if selected_run.state in ("failed","cancelled"):
                    st.warning("Partial results from an unfinished calculation. These frames do not represent a completed print.")
                    st.caption(selected_run.message)
                elif selected_run.state=="partial":
                    st.warning(f"{selected_run.completed_case_count} cases completed; {selected_run.failed_case_count} cases failed.")
                elif selected_run.state in ACTIVE_STATES:st.info("Calculation is still running. Only already calculated frames are shown.")
            report_path=selected_folder/"report.json"
            if report_path.exists():
                transfer_report=read_json(report_path)
            cases=list_cases(selected_run)
            if st.button("Refresh completed frames"):
                st.rerun()
            if cases:
                case_index=st.selectbox("Material / candidate playback",list(range(len(cases))),format_func=lambda i:cases[i][1],key="case_"+selected_run.run_id)
                case_folder=cases[case_index][0];manifest=read_build_manifest(case_folder)
                if manifest.get("job",{}).get("source_sha256")!=job.source_sha256:
                    st.warning("This saved calculation belongs to a different imported job. Its playback uses its own saved geometry and temperatures; the current upload settings apply only to a new calculation.")
                times=[entry["time_s"] for entry in manifest.get("frames",[])]
                saved_config=manifest.get("config",{})
                layer_low=int(saved_config.get("layer_start",0))
                layer_high=int(saved_config.get("layer_end") or selected_run.total_layers or len(job.layers))-1
                if times:
                    st.write(f"**Saved playback:** source layers {layer_low}–{int(manifest['frames'][-1].get('layer_index',layer_high))} · {times[0]:,.1f}–{times[-1]:,.1f} physical seconds · {len(times)} calculated frames")
                if len(times)>1 and np.median(np.diff(times))>1.:
                    st.caption(f"Stored temperatures are about {np.median(np.diff(times)):.1f}s apart. Full playback spans the entire saved print; temperatures update at those calculated frames. Single-layer scan detail uses the same stored temperatures.")
                trajectory_layer=None
                playback_mode=st.radio("Playback mode",["Full simulated print","One layer: detailed scan paths"],horizontal=True,key="viewer_mode_"+selected_run.run_id)
                if playback_mode=="One layer: detailed scan paths":
                    trajectory_layer=st.number_input("Detailed source layer",layer_low,layer_high,layer_low,1,key="viewer_detail_"+selected_run.run_id)
                    st.warning(f"Zoomed into source layer {trajectory_layer}. Select Full simulated print above to watch every saved layer from start to finish.")
                a,b=st.columns(2)
                tmin=a.number_input("Locked temperature minimum (°C)",value=20.,step=20.,key="build_view_min")
                tmax=b.number_input("Locked temperature maximum (°C)",value=1800.,step=100.,key="build_view_max")
                if tmax<=tmin:st.error("Temperature maximum must exceed minimum")
                elif playback_tab.open:
                    viewer=playback_html(case_folder,manifest,tmin,tmax,trajectory_layer=trajectory_layer)
                    st.iframe(viewer,height=1040,alt="Stored build temperatures and actual scan-event playback")
                    st.download_button("Download standalone interactive playback",viewer,"build-playback.html","text/html")
                st.caption("The temperature limits persist across material cases. Playback speed changes viewing time only. The timeline exposes calculated frames incrementally; refresh while a calculation is running. Laser paths follow the imported/reconstructed order, with assumed laser-off timing.")
                st.write(f"Case status: {manifest.get('status','unknown')} • {len(manifest.get('frames',[]))} completed thermal frames")
                if manifest.get("error"):st.error(manifest["error"])
                if manifest.get("frames") and playback_tab.open:
                    with st.expander("Export a calculated frame as PNG"):
                        selected_frame=st.slider("Stored frame index",0,len(manifest["frames"])-1,len(manifest["frames"])-1) if len(manifest["frames"])>1 else 0
                        from .build_render import frame_figure
                        figure=frame_figure(case_folder,manifest,selected_frame,tmin,tmax)
                        st.pyplot(figure)
                        binary=io.BytesIO();figure.savefig(binary,format="png",dpi=160);plt.close(figure)
                        st.download_button("Download frame PNG",binary.getvalue(),"build-thermal-frame.png","image/png")
                with st.expander("Temperature probe • nearest stored cell"):
                    bounds=job.bounds_m;cols=st.columns(3)
                    probe=(cols[0].number_input("Probe x (mm)",value=float((bounds[0]+bounds[3])*500),step=.1)*.001,
                           cols[1].number_input("Probe y (mm)",value=float((bounds[1]+bounds[4])*500),step=.1)*.001,
                           cols[2].number_input("Probe z above plate (mm)",value=float(job.layer_thickness_m*500),step=.03)*.001)
                    history=probe_history(case_folder,manifest,probe) if playback_tab.open else pd.DataFrame()
                    if len(history):
                        st.line_chart(history.set_index("time_s")[["temperature_c"]],x_label="Physical print time (s)",y_label="Stored cell temperature (°C)")
                        st.caption("Nearest-cell samples at saved solver times; heating/cooling peaks between saved frames may be missed.")
                        st.dataframe(history,hide_index=True)
                        st.download_button("Download probe CSV",history.to_csv(index=False),"probe.csv","text/csv")
                with st.expander("Assumptions, solver metrics and numerical limits"):
                    st.json(manifest)
                st.download_button("Download case configuration",json.dumps(manifest.get("config",{}),indent=2),"build-config.json","application/json")
            else:
                st.info("The worker has not saved its first converged frame yet. Refresh completed frames to view progress.")
        else:
            st.info("No completed thermal results are available for this job yet. Start a calculation, then use View results when it finishes.")
            if active_folder:st.caption("Your current calculation is preparing or has not saved its first frame yet.")
        with st.expander("Run history and diagnostics"):
            if catalog:
                st.dataframe(pd.DataFrame([{"Simulation":item.job_name,"Materials":item.material_label,"Layer range":item.layer_label,
                    "Status":item.status_label,"Saved frames":item.frame_count,"Completed cases":item.completed_case_count,
                    "Failed cases":item.failed_case_count,"Run":item.run_id,"Details":item.message} for item in catalog]),hide_index=True,width="stretch")
                stopped=[item for item in catalog if item.state in ("failed","cancelled","partial")]
                for item in stopped[:5]:
                    st.caption(f"{item.run_id}: {item.message}")
            else:st.caption("No saved runs for this job.")

    with insights_tab:
        if insights_tab.open:
            render_results_insights(current_runs,selected_folder=selected_folder,on_view=_queue_view_run,on_edit_preset=_queue_preset_from_result)

    with comparison_tab:
        st.subheader("Saved comparison details" if transfer_report and transfer_report.get("kind")=="presets" else "What changes when the same job uses another material?")
        if transfer_report and transfer_report.get("rows"):
            saved_input_matches=transfer_report.get("job_source_sha256")==job.source_sha256
            if not saved_input_matches:
                st.warning("Saved comparison uses a different source-job hash. Upload controls do not alter these saved results; region-setting arithmetic for the current upload is suppressed.")
            frame=pd.DataFrame(transfer_report["rows"])
            if "peak_temperature_k" in frame:
                frame["peak_temperature_c"]=frame["peak_temperature_k"]-273.15
            columns=[n for n in ("preset_name","material","kind","power_scale","speed_scale","score","comparable","peak_temperature_c","mean_peak_enthalpy_fraction","peak_enthalpy_fraction","physical_duration_s","runtime_s","energy_error_fraction") if n in frame]
            st.dataframe(frame[columns],hide_index=True,width="stretch",column_config={
                "peak_temperature_c":st.column_config.NumberColumn("Peak voxel temperature (°C)",format="%.1f"),
                "score":st.column_config.NumberColumn("Thermal distance",format="%.3f"),
                "runtime_s":st.column_config.NumberColumn("Measured runtime (s)",format="%.2f"),
                "physical_duration_s":st.column_config.NumberColumn("Physical print time (s)",format="%.3f")})
            st.caption("Power and speed multipliers apply after any saved region overrides. A thermal-history match is a candidate for measurement. Coarse voxel temperatures and subgrid consolidation are separate quantities.")
            rankings=transfer_report.get("rankings",[])
            if rankings:
                st.markdown("**Closest sampled thermal responses in this bounded window**")
                st.dataframe(pd.DataFrame([{k:v for k,v in row.items() if k in ("material","closest_preset_name","closest_power_scale","closest_speed_scale","score","ranked_candidates_count")} for row in rankings]),hide_index=True)
                st.warning("These matches are conditional on the saved absorption, beam, powder, substrate, geometry and timing assumptions. Peak temperature and retained heat are not yet numerically converged for the representative mesh/plate checks. Use them to choose measurements and a small trial bracket.")
                chosen=st.selectbox("Inspect region settings for a sampled match",list(range(len(rankings))),format_func=lambda i:rankings[i]["material"])
                ranking=rankings[chosen]
                recipe_rows=[]
                if transfer_report.get("kind")=="presets" and saved_input_matches:
                    matched=next((row for row in transfer_report["rows"] if row.get("case_folder")==ranking.get("closest_case_folder") and row.get("kind")=="candidate"),None)
                    if matched and matched.get("preset"):
                        st.dataframe(pd.DataFrame(effective_region_rows(region_defaults(job.source_sha256,job),matched["preset"],matched.get("config",{}).get("power_factor",1.))),hide_index=True,width="stretch")
                        st.caption("These are this saved candidate's region settings, including its absolute overrides and global factors. Regions outside the saved layer range remain untested.")
                elif ranking.get("closest_power_scale") is not None and saved_input_matches:
                    for part,recipes in job.metadata.get("recipes",{}).items():
                        for pid,recipe in recipes.items():
                            if not isinstance(recipe,dict) or "power" not in recipe:continue
                            used=any(np.any(layer.process_ids==int(pid)) for layer in job.layers)
                            if not used:continue
                            speed_source=recipe.get("speed",recipe.get("speed_mm_s"))
                            if speed_source is None:continue
                            recipe_rows.append({"part":part,"section":pid,"region":PROCESS_NAMES.get(int(pid),"unknown"),
                                "imported_power_provisional_W":recipe["power"],"sampled_power_provisional_W":recipe["power"]*ranking["closest_power_scale"],
                                "imported_speed_mm_s":speed_source,"sampled_speed_mm_s":speed_source*ranking["closest_speed_scale"]})
                    if recipe_rows:st.dataframe(pd.DataFrame(recipe_rows),hide_index=True)
                    st.caption("This arithmetic applies the tested multipliers to used imported presets. Regions outside the simulated window have not been verified by that window's score.")
            st.json({k:v for k,v in transfer_report.items() if k not in ("rows",)},expanded=False)
            if transfer_report.get("failures"):st.error("Some candidates failed and are excluded from valid comparisons. Inspect the recorded failures.")
            st.download_button("Download material comparison CSV",frame.drop(columns=["metrics"],errors="ignore").to_csv(index=False),"material-transfer.csv","text/csv")
        else:
            st.info("Run a material comparison, then select it in Build playback to inspect baseline, targets and bounded adjustments.")
        st.write("To improve confidence: calibrate delivered laser power and spot size; measure baseline track width/depth and heating/cooling where available; determine independent target-alloy coupling and powder behavior. Your 3-second layer delay is an estimate. The simulation helps narrow trials; it cannot remove the need for physical validation.")
        if selected_folder and (selected_folder/"request.json").exists():
            if st.button("Prepare reproducible study ZIP"):
                st.session_state.build_bundle=(str(selected_folder),bundle_run(selected_folder))
            bundle=st.session_state.get("build_bundle")
            if bundle and bundle[0]==str(selected_folder):
                st.download_button("Download reproducible study ZIP",bundle[1],selected_folder.name+".zip","application/zip")

    with evidence_tab:
        build_check=ROOT/"examples"/"build-sensitivity-m4.json"
        if build_check.exists():
            st.warning("Representative whole-build/local checks remain mesh- and plate-sensitive. Numerical conservation passes; thermal predictions are not experimentally validated.")
            with st.expander("Actual Mac build-model sensitivity results",expanded=False):st.json(json.loads(build_check.read_text()))
        for filename,title in [("PRESETS.md","Editing presets and interpreting their results"),("JOB_FORMAT.md","Imported formats, path semantics and missing timing"),("BUILD_MODEL.md","Whole-build equations, activation and approximation"),("MATERIAL_TRANSFER.md","Material comparison and ranking limits"),("MATERIALS.md","Material sources, units and validity ranges"),("VERIFICATION.md","Numerical verification and measured Mac performance")]:
            path=ROOT/"docs"/filename
            if path.exists():
                with st.expander(title):st.markdown(path.read_text())
        st.caption("Preserved single-track geometry/temperature rendering and local numerical studies remain available in Workspace → Single-track laboratory.")
        st.json({"source_job":job.source_path,"source_sha256":job.source_sha256,"local_only":True,"printer_connection":False})
