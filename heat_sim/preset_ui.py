"""Editable local preset library and fixed-job comparison workflow."""
from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from .job_import import PROCESS_NAMES
from .materials import MATERIALS
from .presets import (load_presets, save_preset, preset_from_config,
                      normalize_preset, plan_presets)


@st.cache_data(show_spinner=False)
def region_defaults(job_hash, _job):
    """Summarize actual vector settings, including differences between parts."""
    values={}
    for layer in _job.layers:
        for pid in np.unique(layer.process_ids):
            vectors=layer.vectors[layer.process_ids==pid]
            if not len(vectors):continue
            low=np.min(vectors[:,4:6],axis=0);high=np.max(vectors[:,4:6],axis=0)
            if int(pid) in values:
                old=values[int(pid)];low=np.minimum(low,old[0]);high=np.maximum(high,old[1])
            values[int(pid)]=(low,high)
    return [dict(section=pid,region=PROCESS_NAMES.get(pid,"unknown"),
                 power_min=float(low[0]),power_max=float(high[0]),
                 speed_min=float(low[1])*1000,speed_max=float(high[1])*1000)
            for pid,(low,high) in sorted(values.items())]


def region_editor_rows(regions, overrides):
    rows=[]
    for region in regions:
        value=overrides.get(str(region["section"]),overrides.get(region["section"],{}))
        rows.append({"Section":region["section"],"Region":region["region"],
            "Override power": "power_w" in value,
            "Power (W)":float(value.get("power_w",region["power_min"])),
            "Override speed": "speed_m_s" in value,
            "Speed (mm/s)":float(value.get("speed_m_s",region["speed_min"]*.001))*1000})
    return rows


def overrides_from_rows(rows):
    """Unchecked cells retain the exact imported setting on every vector."""
    overrides={}
    for row in rows:
        values={}
        if row.get("Override power"):values["power_w"]=float(row["Power (W)"])
        if row.get("Override speed"):values["speed_m_s"]=float(row["Speed (mm/s)"])*.001
        if values:overrides[str(int(row["Section"]))]=values
    return overrides


def _range_text(low,high):
    return f"{low:g}" if abs(high-low)<1e-9 else f"{low:g}–{high:g}"


def effective_region_rows(regions,preset,power_factor):
    rows=[]
    for region in regions:
        value=preset.get("section_overrides",{}).get(str(region["section"]),{})
        p0=value.get("power_w",region["power_min"])*preset["power_scale"]
        p1=value.get("power_w",region["power_max"])*preset["power_scale"]
        v0=value.get("speed_m_s",region["speed_min"]*.001)*1000*preset["speed_scale"]
        v1=value.get("speed_m_s",region["speed_max"]*.001)*1000*preset["speed_scale"]
        rows.append({"Region":f"{region['section']}: {region['region']}",
            "Imported power (W)":_range_text(region["power_min"],region["power_max"]),
            "Preset command power (W)":_range_text(p0,p1),
            "Assumed delivered power (W)":_range_text(p0*power_factor,p1*power_factor),
            "Imported speed (mm/s)":_range_text(region["speed_min"],region["speed_max"]),
            "Preset speed (mm/s)":_range_text(v0,v1)})
    return rows


def _seed_editor(prefix,preset):
    mappings={"name":"name","material":"material","substrate":"substrate", "power":"power_scale",
              "speed":"speed_scale","alpha":"absorptivity","preheat":"initial_k",
              "recoat":"recoat_time_s","beam":"beam_diameter_m","packing":"powder_packing",
              "powder_k":"powder_k_w_mk","jump_speed":"jump_speed_m_s","jump_delay":"jump_delay_s"}
    for key,source in mappings.items():
        value=preset[source]
        if key in ("power","speed"):value*=100
        elif key=="preheat":value-=273.15
        elif key in ("beam","jump_delay"):value*=1e6
        st.session_state[prefix+key]=value
    st.session_state[prefix+"draft"]=preset
    st.session_state[prefix+"generation"]=st.session_state.get(prefix+"generation",0)+1


def _material_changed(prefix,alphas):
    st.session_state[prefix+"alpha"]=float(alphas[st.session_state[prefix+"material"]])


def render_preset_lab(job,config,library_root,alphas,ready,busy,on_start,on_insights):
    st.subheader("Tune print presets before printing")
    st.write("Save named settings, choose a reference, and simulate several variations on the same imported print. Compare the results before choosing a physical trial.")
    end=config.layer_end or len(job.layers)
    st.info(f"Shared calculation scope: **{end-config.layer_start} of {len(job.layers)} layers** ({config.layer_start}–{end-1}). "
            f"Thermal cells {config.spacing_m*1000:g} × {config.z_spacing_m*1000:g} mm; exposure groups {config.max_event_s*1000:g} ms. Change these in Calculate.")
    st.caption("Physical presets are separate from the coarse/preview calculation presets. Every comparison freezes its settings; later edits do not change saved results.")
    prefix="preset_"+job.source_sha256[:16]+"_"
    try:library=load_presets(library_root,job.source_sha256)
    except ValueError as exc:
        st.error(str(exc));return
    by_id={item["id"]:item for item in library}
    pending=st.session_state.pop(prefix+"pending_edit",None)
    imported_draft=st.session_state.pop(prefix+"imported_draft",None)
    if imported_draft:
        st.session_state[prefix+"edit"]="__new__"
        _seed_editor(prefix,imported_draft)
        st.session_state[prefix+"origin"]="__new__"
    if pending:
        st.session_state[prefix+"edit"]=pending
        st.session_state.pop(prefix+"origin",None)
    pending_candidate=st.session_state.pop(prefix+"pending_candidate",None)
    if pending_candidate:
        chosen=list(dict.fromkeys([*st.session_state.get(prefix+"candidates",[]),pending_candidate]))
        st.session_state[prefix+"candidates"]=chosen[-12:]
    if st.session_state.get(prefix+"edit") not in ["__new__",*by_id]:
        st.session_state[prefix+"edit"]="__new__"
    notice=st.session_state.pop(prefix+"notice",None)
    if notice:st.success(notice)
    regions=region_defaults(job.source_sha256,job)
    st.markdown("**1 · Create or edit a named preset**")
    selection=st.selectbox("Preset to edit",["__new__",*by_id],key=prefix+"edit",
        format_func=lambda value:"Create a new preset" if value=="__new__" else by_id[value]["name"]+" · "+by_id[value]["material"])
    if st.session_state.get(prefix+"origin")!=selection:
        seed=by_id[selection] if selection!="__new__" else preset_from_config(config,f"Trial preset {len(library)+1}")
        _seed_editor(prefix,seed)
        st.session_state[prefix+"origin"]=selection
    draft=st.session_state[prefix+"draft"]
    a,b,c=st.columns([2,1,1])
    name=a.text_input("Preset name",key=prefix+"name",max_chars=80)
    material=b.selectbox("Preset material",list(MATERIALS),key=prefix+"material",on_change=_material_changed,args=(prefix,alphas))
    substrate=c.selectbox("Preset substrate",list(MATERIALS),key=prefix+"substrate")
    a,b,c=st.columns(3)
    power=a.number_input("Power (% of imported / overridden recipe)",1.,300.,step=5.,key=prefix+"power")
    speed=b.number_input("Speed (% of imported / overridden recipe)",10.,500.,step=5.,key=prefix+"speed")
    alpha=c.number_input("Preset effective absorptivity • assumed",.001,1.,step=.01,key=prefix+"alpha",format="%.3f")
    a,b,c=st.columns(3)
    preheat=a.number_input("Preset preheat (°C)",0.,1000.,step=5.,key=prefix+"preheat")
    recoat=b.number_input("Preset between-layer time incl. recoating (s)",0.,120.,step=.1,key=prefix+"recoat")
    beam=c.number_input("Preset 1/e² beam diameter (µm)",10.,1000.,step=5.,key=prefix+"beam")
    st.caption("Changing material selects its independent optical assumption. Verify absorptivity and substrate for each preset. Your starting inter-layer estimate is 3 seconds.")
    with st.expander("Tune individual region recipes (infill, contour, supports)"):
        st.write("Enable only the power or speed values you want to replace. Other values retain their imported settings. The percentage controls above apply after these replacements.")
        table=st.data_editor(pd.DataFrame(region_editor_rows(regions,draft.get("section_overrides",{}))),
            key=prefix+"regions_"+str(st.session_state[prefix+"generation"]),hide_index=True,width="stretch",
            disabled=["Section","Region"],column_config={
                "Override power":st.column_config.CheckboxColumn("Replace power"),
                "Override speed":st.column_config.CheckboxColumn("Replace speed"),
                "Power (W)":st.column_config.NumberColumn(min_value=.001,max_value=10000.,step=5.),
                "Speed (mm/s)":st.column_config.NumberColumn(min_value=.001,max_value=100000.,step=10.)})
        st.caption("Region replacements apply to all source parts using that section. If their imported settings differ, the effective-setting table below shows the original range. Regions outside the calculated layer range remain untested.")
    with st.expander("Additional powder and motion inputs"):
        a,b=st.columns(2)
        packing=a.number_input("Preset powder packing fraction",.2,1.,step=.01,key=prefix+"packing")
        powder_k=b.number_input("Preset powder conductivity (W/m·K)",.05,30.,step=.1,key=prefix+"powder_k")
        a,b=st.columns(2)
        jump_speed=a.number_input("Preset jump speed (m/s)",.01,100.,step=.5,key=prefix+"jump_speed")
        jump_delay=b.number_input("Preset jump settling delay (µs)",0.,10000.,step=10.,key=prefix+"jump_delay")
    edited=None
    try:
        edited=normalize_preset({**draft,"name":name,"material":material,"substrate":substrate,
            "power_scale":power/100,"speed_scale":speed/100,"absorptivity":alpha,"initial_k":preheat+273.15,
            "recoat_time_s":recoat,"beam_diameter_m":beam*1e-6,"powder_packing":packing,
            "powder_k_w_mk":powder_k,"jump_speed_m_s":jump_speed,"jump_delay_s":jump_delay*1e-6,
            "section_overrides":overrides_from_rows(table.to_dict("records"))},config)
    except (ValueError,TypeError,KeyError) as exc:st.error("Preset needs attention: "+str(exc))
    a,b=st.columns(2)
    save=a.button("Save preset",type="primary",disabled=edited is None,key=prefix+"save")
    copy=b.button("Save as new preset",disabled=edited is None or selection=="__new__",key=prefix+"copy",help="Change the name first; saved presets have unique names.")
    if save or copy:
        try:
            value=dict(edited)
            if copy:value["id"]=uuid.uuid4().hex[:16]
            saved=save_preset(library_root,job.source_sha256,value,config)
            st.session_state[prefix+"pending_edit"]=saved["id"]
            st.session_state[prefix+"pending_candidate"]=saved["id"]
            st.session_state[prefix+"notice"]=f"Saved {saved['name']}. Choose it below to calculate its actual thermal response."
            st.rerun()
        except ValueError as exc:st.error(str(exc))
    if edited:
        with st.expander("Preview this preset's actual region settings"):
            st.dataframe(pd.DataFrame(effective_region_rows(regions,edited,config.power_factor)),hide_index=True,width="stretch")
            st.caption("W and mm/s use the importer's provisional source-unit interpretation. Delivered power also applies the shared machine factor from Materials & settings.")
    st.markdown("**2 · Compare saved presets against a reference**")
    current=preset_from_config(config,"Current job settings",preset_id="current-reference-"+job.source_sha256[:16])
    refs={"__current__":current,**by_id}
    if st.session_state.get(prefix+"reference") not in refs:st.session_state[prefix+"reference"]="__current__"
    reference_id=st.selectbox("Reference preset",list(refs),key=prefix+"reference",format_func=lambda value:refs[value]["name"]+" · "+refs[value]["material"])
    reference=refs[reference_id]
    options=[value for value in by_id if value!=reference_id]
    if prefix+"candidates" in st.session_state:
        st.session_state[prefix+"candidates"]=[value for value in st.session_state[prefix+"candidates"] if value in options]
    candidate_ids=st.multiselect("Candidate presets to simulate",options,default=options[:3],max_selections=12,
        key=prefix+"candidates",format_func=lambda value:by_id[value]["name"]+" · "+by_id[value]["material"])
    study_name=st.text_input("Preset study name",value="Print preset comparison",key=prefix+"study",max_chars=120)
    request={"study_name":study_name,"reference":reference,"candidates":[by_id[value] for value in candidate_ids]}
    planned=[];can_run=ready and bool(candidate_ids)
    if not ready:st.info("Review the shared calculation settings in Calculate before starting this study.")
    if candidate_ids:
        try:
            planned=plan_presets(job,config,request)
            st.dataframe(pd.DataFrame([{"Role":"Reference" if i==0 else "Candidate","Preset":row["preset"]["name"],
                "Material":row["preset"]["material"],"Power (%)":row["config"]["power_scale"]*100,
                "Speed (%)":row["config"]["speed_scale"]*100,"Preheat (°C)":row["config"]["initial_k"]-273.15,
                "Between layers (s)":row["config"]["recoat_time_s"],"Region overrides":len(row["config"].get("section_overrides",{})),
                "Estimated print time (s)":round(row["plan"]["estimated_physical_time_s"],3),
                "Minimum steps":row["plan"]["minimum_thermal_steps"]} for i,row in enumerate(planned)]),hide_index=True,width="stretch")
            bad=[row for row in planned if row["plan"]["step_limit_exceeded"]]
            if bad:
                can_run=False
                st.error("Step budget exceeded for "+", ".join(row["preset"]["name"] for row in bad)+". Adjust the shared calculation scope or grouping in Calculate.")
        except ValueError as exc:can_run=False;st.error(str(exc))
    else:st.info("Save at least one candidate preset, then choose it here. You can keep the current job settings as your reference or select a saved successful preset.")
    if edited and selection in by_id:
        stored={k:v for k,v in by_id[selection].items() if k!="saved_utc"}
        unsaved={k:v for k,v in edited.items() if k!="saved_utc"}
        if stored!=unsaved:st.warning("The editor has unsaved changes. Comparisons use saved presets; save your changes first.")
    elif edited and selection=="__new__":st.caption("The editor is a draft until you save it. The comparison list uses saved presets.")
    a,b=st.columns(2)
    if a.button("Simulate preset comparison",disabled=busy or not can_run,key=prefix+"run",type="primary"):
        on_start(request)
    b.button("Open results insights",on_click=on_insights,key=prefix+"insights")
    st.caption("The score asks which sampled preset produces the closest resolved thermal response to your reference. It does not establish a print-quality optimum. Hatch spacing, layer thickness, and scan strategy require a newly sliced job.")
    if library:
        st.download_button("Export saved presets",json.dumps({"job_source_sha256":job.source_sha256,"presets":library},indent=2),
            "print-presets.json","application/json",key=prefix+"export")
