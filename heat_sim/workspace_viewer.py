"""One viewport for imported geometry and actual saved thermal fields.

The same clock, temperature scale and camera drive comparison scenes. Frame
selection holds the last calculated field; it never interpolates temperature.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .build_viewer import playback_data,load_build_frame,_local_path
from .path_geometry import simplify_scan_paths
from .viewer_geometry_transport import publish_path_geometry


@lru_cache(maxsize=4)
def _timeline_paths(path,modified,size):
    """Keep every emitting chain, simplifying joined curves within 0.02 mm."""
    path=Path(path)
    if path.suffix==".npz":
        with np.load(path,allow_pickle=False) as archive:raw=archive["events"]
        starts=raw["start_m"]*1000;ends=raw["end_m"]*1000
        t0=raw["t0_s"];t1=raw["t1_s"];layers=raw["layer_index"];sections=raw["section"]
        enabled=(raw["kind"]==1)&(raw["power_w"]>0)
    else:
        raw=json.loads(path.read_text());raw=raw.get("events",[]) if isinstance(raw,dict) else raw
        starts=np.asarray([e.get("start_m",[0,0,0]) for e in raw],float).reshape(-1,3)*1000
        ends=np.asarray([e.get("end_m",e.get("start_m",[0,0,0])) for e in raw],float).reshape(-1,3)*1000
        t0=np.asarray([e.get("t0_s",e.get("start_s",0)) for e in raw],float)
        t1=np.asarray([e.get("t1_s",e.get("end_s",0)) for e in raw],float)
        layers=np.asarray([e.get("layer_index",0) for e in raw]);sections=np.asarray([e.get("section",0) for e in raw])
        enabled=np.asarray([bool(e.get("laser_on",e.get("kind")=="scan")) and float(e.get("power_w",1))>0 for e in raw])
    first,last=simplify_scan_paths(starts,ends,layers,sections,t0,t1,enabled)
    paths=np.empty(len(first),dtype=[("t","f8"),("p","f8",(3,)),("q","f8",(3,)),("section","i4")])
    paths["t"]=t1[last];paths["p"]=starts[first];paths["q"]=ends[last];paths["section"]=sections[last]
    return paths,int(np.count_nonzero(enabled))


def _nearest_indices(axis,values):
    right=np.clip(np.searchsorted(axis,values),0,len(axis)-1)
    left=np.maximum(right-1,0)
    return np.where(abs(axis[left]-values)<=abs(axis[right]-values),left,right)


def _attach_part_paths(data,folder,manifest):
    """Map imported scans to exact saved midpoint-cell temperatures.

    This is a visual projection of the coarse field onto source scan geometry,
    not a new fine-scale solution or a reconstructed watertight surface.
    """
    data["paths"]=[]
    if manifest.get("config",{}).get("region_xy_m"):return
    path=_local_path(folder,manifest.get("timeline_file","timeline.json"))
    if not path.is_file():return
    stat=path.stat();paths,total=_timeline_paths(str(path),stat.st_mtime_ns,stat.st_size)
    if not len(paths):return
    entries=[entry for entry in manifest.get("frames",[]) if _local_path(folder,entry["file"]).is_file()]
    final=load_build_frame(folder,entries[-1]);axes=[final[k] for k in ("x_m","y_m","z_m")]
    midpoints=(paths["p"]+paths["q"])/2000
    indices=tuple(_nearest_indices(axis,midpoints[:,a]) for a,axis in enumerate(axes))
    shape=final["temperature_k"].shape;ravel=np.ravel_multi_index(indices,shape)
    shown=np.asarray(data["points"])/1000
    old_indices=tuple(_nearest_indices(axis,shown[:,a]) for a,axis in enumerate(axes))
    lookup={int(value):i for i,value in enumerate(np.ravel_multi_index(old_indices,shape))}
    missing=list(dict.fromkeys(int(value) for value in ravel if int(value) not in lookup))
    if missing:
        positions=np.unravel_index(missing,shape)
        start=len(data["points"])
        for i,value in enumerate(missing):lookup[value]=start+i
        data["points"].extend(np.column_stack([axis[index]*1000 for axis,index in zip(axes,positions)]).tolist())
        plate=final.get("plate_mask",np.zeros(shape,bool)).ravel()[missing]
        support=final.get("support_mask",np.zeros(shape,bool)).ravel()[missing]
        part=final.get("part_mask",final["consolidated_fraction"]>.001).ravel()[missing]
        data["types"].extend(np.where(plate,2,np.where(support,1,np.where(part,0,3))).astype(int).tolist())
        for field,entry in zip(data["frames"],entries):
            raw=load_build_frame(folder,entry);active=raw.get("active_fraction",raw.get("deposited_fraction"))
            field["temperature"].extend((np.asarray(raw["temperature_k"].ravel()[missing],float)-273.15).tolist())
            for key,values in (("active",active),("solid",raw["consolidated_fraction"]),
                               ("part",raw.get("part_mask",raw["consolidated_fraction"]>.001)),
                               ("support",raw.get("support_mask",np.zeros(shape,bool)))):
                field[key].extend(values.ravel()[missing].tolist())
    cells=np.asarray([lookup[int(value)] for value in ravel],dtype=np.uint32)
    support=paths["section"]==16
    if len(paths)>20000:
        data["path_geometry"]=publish_path_geometry(paths["t"],paths["p"],paths["q"],cells,support)
    else:
        data["paths"]=[dict(t=float(end),p=p.tolist(),q=q.tolist(),support=bool(flag),cell=int(cell))
                       for end,p,q,flag,cell in zip(paths["t"],paths["p"],paths["q"],support,cells)]
    data["source_path_count"]=total
    data["display_path_count"]=len(paths)
    data["geometry_coverage"]="complete"
    data["geometry_tolerance_mm"]=.02
    data["path_temperature_mapping"]="Exact stored thermal cell nearest each displayed scan segment's midpoint"


def geometry_preview_data(job, max_segments=None, view_key=None):
    """Display complete source geometry, or an explicitly requested sample."""
    if max_segments is not None and (not isinstance(max_segments,int) or max_segments<1):
        raise ValueError("Geometry display budget must be a positive integer")
    layers=[layer for layer in job.layers if len(layer.vectors)]
    source_layers=[int(layer.index) for layer in layers]
    segments=[];original=sum(len(layer.vectors) for layer in layers);geometry=None
    if max_segments is None and original:
        starts=np.empty((original,3));ends=np.empty((original,3))
        sections=np.empty(original,np.int64);indices=np.empty(original,np.int64);offset=0
        for layer in layers:
            last=offset+len(layer.vectors)
            starts[offset:last,:2]=layer.vectors[:,:2]*1000
            ends[offset:last,:2]=layer.vectors[:,2:4]*1000
            starts[offset:last,2]=ends[offset:last,2]=(layer.z_m+job.layer_thickness_m)*1000
            sections[offset:last]=layer.process_ids;indices[offset:last]=layer.index;offset=last
        first,last=simplify_scan_paths(starts,ends,indices,sections,max_span_s=None)
        displayed=len(first)
        if displayed>20000:
            geometry=publish_path_geometry(indices[last].astype(float),starts[first],ends[last],
                                           sections[last].astype(np.uint32),sections[last]==16)
        else:
            segments=np.column_stack((starts[first],ends[last],sections[last],indices[last])).tolist()
    else:
        if max_segments is not None and len(layers)>max_segments:
            layers=[layers[i] for i in np.linspace(0,len(layers)-1,max_segments,dtype=int)]
        quota=max(1,(max_segments or 1)//max(1,len(layers)))
        for layer in layers:
            # Explicitly sampled views still stratify supports and the part.
            groups=[np.flatnonzero(layer.process_ids!=16),np.flatnonzero(layer.process_ids==16)]
            counts=([max(1,int(quota*.75)),quota-max(1,int(quota*.75))] if quota>1 else [1,0]) if all(len(g) for g in groups) else [quota,quota]
            for selected,limit in zip(groups,counts):
                if not len(selected) or limit<1:continue
                step=max(1,math.ceil(len(selected)/limit))
                for index in selected[::step]:
                    if len(segments)>=max_segments:break
                    vector=layer.vectors[index]
                    z=(layer.z_m+job.layer_thickness_m)*1000
                    segments.append([float(vector[0]*1000),float(vector[1]*1000),float(z),
                                     float(vector[2]*1000),float(vector[3]*1000),float(z),
                                     int(layer.process_ids[index]),int(layer.index)])
        displayed=len(segments)
    bounds=np.asarray(job.bounds_m,float).reshape(2,3).T*1000
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Imported geometry bounds must be finite")
    scene=dict(label=job.name,alloy=None,segments=segments,bounds=bounds.tolist(),
        source_sha256=job.source_sha256,layer_count=len(job.layers),
        displayed_segments=displayed,source_segments=original,source_layers=source_layers,
        geometry_coverage="complete" if max_segments is None or displayed==original else "sampled",
        geometry_tolerance_mm=.02 if max_segments is None else 0.)
    if geometry is not None:scene["path_geometry"]=geometry
    return dict(mode="geometry",cases=[scene],live=False,show_probe=False,
                vmin=None,vmax=None,view_key=view_key or job.source_sha256,
                canvas_height=440,initial_time=None,run_key=None)


def _html(data):
    serialized=json.dumps(data,allow_nan=False,separators=(",",":"),ensure_ascii=True).replace("<","\\u003c")
    return _HTML.replace("__DATA__",serialized)


def geometry_preview_html(job,max_segments=None,view_key=None,canvas_height=440):
    data=geometry_preview_data(job,max_segments,view_key)
    data["canvas_height"]=max(250,min(800,int(canvas_height)))
    return _html(data)


def thermal_workspace_data(cases,temperature_min_c=20.,temperature_max_c=1800.,live=False,
                           view_key=None,initial_time=None,max_points=14000,
                           show_probe=False,canvas_height=440,part_geometry=True,auto_temperature_range=False):
    """Read one or two independent calculations of the same source geometry."""
    if not 1<=len(cases)<=2:
        raise ValueError("The simulation viewport needs one or two cases")
    scenes=[]
    for case in cases:
        if not case.get("manifest") or not case.get("folder"):
            scenes.append(dict(pending=True,label=str(case.get("label") or "Comparison material"),frames=[],
                               state=str(case.get("state") or "queued"),message=str(case.get("message") or "")))
            continue
        folder=Path(case["folder"])
        manifest=case["manifest"]
        data=playback_data(folder,manifest,temperature_min_c,temperature_max_c,
                           max_points=max_points,live=live,temperature_decimals=None,
                           include_inactive=len(cases)==2)
        if data is None:
            raise ValueError("This case has no available calculated temperature frames")
        if part_geometry:_attach_part_paths(data,folder,manifest)
        else:data["paths"]=[]
        data["label"]=str(case.get("label") or data["alloy"])
        scenes.append(data)
    available=[scene for scene in scenes if not scene.get("pending")]
    if not available:raise ValueError("No calculated temperature fields are available yet")
    for scene in scenes:
        if scene.get("pending"):
            scene.update(bounds=available[0]["bounds"],source_sha256=available[0].get("source_sha256"))
    hashes={scene["source_sha256"] for scene in scenes if scene.get("source_sha256")}
    if len(hashes)>1:
        raise ValueError("Material comparison requires the same imported source job")
    if len(available)==2:
        if any(not scene.get("source_sha256") for scene in available):
            raise ValueError("Material comparison needs the saved source-job identity for both cases")
        # A common numerical mesh is required for a direct point comparison.
        if available[0]["points"]!=available[1]["points"] or available[0]["cell_mm"]!=available[1]["cell_mm"]:
            raise ValueError("Synchronized material comparison requires matching saved thermal grids")
    if initial_time is not None and not math.isfinite(float(initial_time)):
        raise ValueError("Initial physical time must be finite")
    primary=Path(available[0]["view_key"])
    study=primary.parent.parent if primary.parent.name=="cases" else primary
    return dict(mode="thermal",cases=scenes,live=bool(live),show_probe=bool(show_probe),run_key=str(study),
        vmin=float(temperature_min_c),vmax=float(temperature_max_c),auto_temperature_range=bool(auto_temperature_range),
        view_key=str(view_key or available[0].get("source_sha256") or available[0]["view_key"]),
        initial_time=None if initial_time is None else float(initial_time),
        canvas_height=max(250,min(800,int(canvas_height))))


def thermal_workspace_html(cases,temperature_min_c=20.,temperature_max_c=1800.,live=False,
                           view_key=None,initial_time=None,max_points=14000,
                           show_probe=False,canvas_height=440,part_geometry=True,auto_temperature_range=False):
    return _html(thermal_workspace_data(cases,temperature_min_c,temperature_max_c,live,
                                       view_key,initial_time,max_points,show_probe,canvas_height,part_geometry,auto_temperature_range))


_HTML=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}body{margin:0;background:#10151d;color:#e4edf9;font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,select{background:#1e2c3e;color:#e3edf9;border:1px solid #3b506b;border-radius:6px;padding:7px 10px;font:inherit}button{cursor:pointer}button:disabled,input:disabled{opacity:.4;cursor:default}button.active{background:#b05a36;border-color:#e58b5c}.scenes{display:grid;grid-template-columns:repeat(var(--count),minmax(0,1fr));gap:12px}.scene{min-width:0;border:1px solid #2b3b50;border-radius:10px;overflow:hidden;background:#121e2d}.scene-head{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:9px 12px;border-bottom:1px solid #29384b}.scene-label{font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.frame-label{color:#afc2d8;font-size:11px;text-align:right}canvas.viewport{width:100%;height:var(--canvas-height);display:block;touch-action:none}.scene-note{color:#a4b8d0;font-size:11px;padding:5px 12px 9px;min-height:26px}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:10px 0 7px}.controls label{display:inline-flex;align-items:center;gap:5px;color:#b8cadf}input[type=range]{accent-color:#fa9a68;min-width:120px}.timeline{width:100%;margin:3px 0 7px}.clock{color:#8fe3d8;font-variant-numeric:tabular-nums}.time-row{display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;color:#adbed3;font-size:12px}.legend-row{display:flex;align-items:center;gap:8px;margin-top:10px;font-size:11px;color:#b8c9dc}.gradient{height:9px;width:180px;max-width:45%;border-radius:5px;background:linear-gradient(90deg,#397fba,#559fc5,#c17594,#ed8d51,#fce65a)}.notice{font-size:11px;color:#a8bad1;line-height:1.4;margin-top:8px}.badge{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#edbc91;margin:0 0 8px}.point{font-size:12px;line-height:1.5;color:#c3d6e9;margin-top:8px;min-height:20px}.graph{margin-top:8px;display:none;border:1px solid #2b3b50;background:#121e2d;border-radius:8px}.graph canvas{width:100%;height:130px;display:block}.graph .notice{padding:0 10px 8px}.geometry .thermal-only,.thermal .geometry-only{display:none}.view-options{margin-top:10px;border-top:1px solid #28394d;padding-top:8px}.view-options summary{cursor:pointer;color:#b8cadf;font-size:12px}.view-options .controls{padding-top:8px}.controls .live-only{display:none}.live .controls .live-only{display:inline-flex}@media(max-width:650px){canvas.viewport{height:300px}.scenes.compare{gap:7px}.scene-head{padding:7px;font-size:11px;align-items:flex-start;flex-direction:column;gap:3px}.frame-label{text-align:left}.controls{gap:6px}.controls label{font-size:11px}button,select{padding:6px 8px}.notice{font-size:10px}}
</style></head><body>
<div class="badge" id="kind"></div>
<div class="scenes" id="scenes"></div>
<div class="controls">
 <button id="play" class="thermal-only replay-only">▶ Play</button><button id="restart" class="thermal-only replay-only">Restart</button><button id="finish" class="thermal-only replay-only" aria-label="Show finished print" title="Show finished print">End</button>
 <label class="thermal-only replay-only">Speed <select id="speed" aria-label="Playback speed"><option value="auto">Auto (~1 min)</option><option value=".1">0.1×</option><option value=".5">0.5×</option><option value="1">1×</option><option value="4">4×</option><option value="16">16×</option><option value="64">64×</option><option value="256">256×</option></select></label>
 <label class="live-only"><input type="checkbox" id="followcalculation" checked>Follow calculation</label><button id="view3d" class="active">3D</button><button id="viewlayer">Layer cross-section</button><label><input type="checkbox" id="supports" checked>Supports</label>
 <label id="plane-wrap" style="display:none">Plane <select id="plane" aria-label="Layer cross-section plane"></select></label>
 <label class="thermal-only" id="follow-wrap" style="display:none"><input type="checkbox" id="follow" checked>Follow build</label>
</div>
<input class="timeline thermal-only replay-only" type="range" id="timeline" aria-label="Physical print time" step="any">
<div class="time-row thermal-only"><span class="clock" id="clock"></span><span id="layers"></span><span id="availability"></span></div>
<div class="legend-row thermal-only"><span id="minimum"></span><div class="gradient"></div><span id="maximum"></span><span id="scale-label">Temperature · °C · shared scale</span><button id="fitcolors" style="padding:4px 7px;font-size:11px">Fit colours</button></div>
<div class="notice thermal-only" id="scale-note" style="color:#e8bd85;display:none"></div>
<details class="view-options"><summary>View options</summary>
<div class="controls">
 <label class="thermal-only">Display <select id="representation" aria-label="Thermal display"><option value="part">Part geometry</option><option value="cells">Thermal cells</option></select></label><button id="reset">Reset camera</button>
 <label class="thermal-only"><input type="checkbox" id="plate" checked>Plate</label><label class="thermal-only"><input type="checkbox" id="powder">Unexposed powder</label>
 <label class="thermal-only"><input type="checkbox" id="pointgraph">Point temperature graph</label><span class="geometry-only notice" id="geometry-sampling"></span>
</div>
</details>
<div class="point thermal-only" id="point">Click a calculated cell to inspect its temperature.</div>
<div class="graph thermal-only" id="graph"><canvas id="history"></canvas><div class="notice">Actual stored cell samples. Lines connect samples for reading; temperatures between saved frames are not calculated here.</div></div>
<div class="notice" id="notice"></div>
<script>const W=__DATA__;const $=id=>document.getElementById(id);const thermal=W.mode==='thermal',scenes=W.cases;document.body.classList.add(W.mode);if(W.live)document.body.classList.add('live');document.documentElement.style.setProperty('--count',String(scenes.length));document.documentElement.style.setProperty('--canvas-height',W.canvas_height+'px');if(scenes.length===2)$('scenes').classList.add('compare');
const savedKey='tracklab.workspace.'+W.view_key;let yaw=-.7,pitch=.65,zoom=1,view='3d',drag=null,selected=null,playing=false,last=0,picks=[],savedState=null,following=!!W.live;
try{const s=JSON.parse(sessionStorage.getItem(savedKey));savedState=s;if(s){if([s.yaw,s.pitch,s.zoom].every(Number.isFinite)){yaw=s.yaw;pitch=s.pitch;zoom=s.zoom}if(s.view==='layer')view='layer';if(s.selected)selected=s.selected;if(W.live&&s.run===W.run_key&&s.following===false)following=false}}catch{}
function saveCamera(){try{sessionStorage.setItem(savedKey,JSON.stringify({yaw,pitch,zoom,view,selected,run:W.run_key,mode:W.mode,plane:Number($('plane').value),followBuild:$('follow').checked,representation:$('representation').value,pointgraph:$('pointgraph').checked,time:thermal?time:null,playing,following,supports:$('supports').checked,plate:$('plate').checked,powder:$('powder').checked,temperatureRange:thermal&&W.auto_temperature_range?{version:1,min:scaleMinimum,max:scaleMaximum,locked:scaleLocked}:null}))}catch{}}
scenes.forEach((s,i)=>{const pane=document.createElement('div');pane.className='scene';const head=document.createElement('div');head.className='scene-head';const label=document.createElement('span');label.className='scene-label';label.textContent=s.label;head.appendChild(label);const frame=document.createElement('span');frame.id='frame-'+i;frame.className='frame-label';head.appendChild(frame);pane.appendChild(head);const canvas=document.createElement('canvas');canvas.id='viewport-'+i;canvas.className='viewport';canvas.setAttribute('aria-label',(thermal?'Calculated thermal field: ':'Imported geometry: ')+s.label);canvas.setAttribute('role','img');pane.appendChild(canvas);const note=document.createElement('div');note.id='note-'+i;note.className='scene-note';pane.appendChild(note);$('scenes').appendChild(pane)});
const bounds=[0,1,2].map(axis=>[Math.min(...scenes.map(s=>s.bounds[axis][0])),Math.max(...scenes.map(s=>s.bounds[axis][1]))]),lo=bounds.map(b=>b[0]),hi=bounds.map(b=>b[1]),centre=lo.map((v,i)=>(v+hi[i])/2),span=Math.max(...lo.map((v,i)=>hi[i]-v),.001);
function physicalBounds(data){return{begin:Math.min(...data.filter(s=>s.frames.length).map(s=>s.frames[0].t)),end:Math.min(...data.filter(s=>s.frames.length).map(s=>s.frames[s.frames.length-1].t))}}
const limits=thermal?physicalBounds(scenes):{begin:0,end:0};let time=thermal?Math.max(limits.begin,Math.min(limits.end,following?limits.end:W.initial_time!==null?W.initial_time:savedState&&savedState.run===W.run_key&&Number.isFinite(savedState.time)?savedState.time:limits.end)):0;if(savedState&&savedState.run===W.run_key&&savedState.playing&&!following&&thermal)playing=true;
let scaleMinimum=W.vmin,scaleMaximum=W.vmax,scaleReady=!W.auto_temperature_range,scaleLocked=!W.live||!following;
if(thermal&&W.auto_temperature_range&&savedState&&savedState.run===W.run_key&&savedState.temperatureRange){const r=savedState.temperatureRange;if(r.version===1&&Number.isFinite(r.min)&&Number.isFinite(r.max)&&r.max>r.min){scaleMinimum=r.min;scaleMaximum=r.max;scaleReady=true;scaleLocked=!W.live||!following||r.locked===true}}
function frameIndex(s,t){if(!s.frames||!s.frames.length||t<s.frames[0].t-1e-9)return-1;let l=0,r=s.frames.length-1;while(l<r){let m=Math.ceil((l+r)/2);if(s.frames[m].t<=t)l=m;else r=m-1}return l}
function layerAt(s,t,f){const a=s.layer_spans||[];let l=0,r=a.length-1,found=null;while(l<=r){let m=(l+r)>>1;if(a[m].a<=t){found=a[m];l=m+1}else r=m-1}return found&&t<=found.b+1e-9?found.layer:f.layer}
function playbackRate(value){return value==='auto'?Math.max(1,(limits.end-limits.begin)/60):Number(value)}
function pointTemperature(s,f,index){return f&&f.active[index]>0?f.temperature[index]:null}
function regionAt(s,f,i){if(s.types[i]===2)return'plate';if(f.support?f.support[i]:s.types[i]===1&&f.solid[i]>.001)return'supports';return(f.part?f.part[i]:f.solid[i]>.001)?'part':'powder'}
function isVisible(s,f,i){if(f.active[i]<=0)return false;const r=regionAt(s,f,i);return r==='part'||$(r).checked}
function currentTemperatureBounds(){let low=Infinity,high=-Infinity,fallbackLow=Infinity,fallbackHigh=-Infinity;for(const s of scenes){if(s.pending)continue;const k=frameIndex(s,time);if(k<0)continue;const f=s.frames[k];for(let i=0;i<f.temperature.length;i++){const t=f.temperature[i];if(f.active[i]<=0||!Number.isFinite(t))continue;const r=regionAt(s,f,i);if(isVisible(s,f,i)){fallbackLow=Math.min(fallbackLow,t);fallbackHigh=Math.max(fallbackHigh,t)}if(r==='part'||r==='supports'&&$('supports').checked){low=Math.min(low,t);high=Math.max(high,t)}}}return Number.isFinite(low)?{low,high}:Number.isFinite(fallbackLow)?{low:fallbackLow,high:fallbackHigh}:null}
function roundedTemperatureBounds(low,high){if(high-low<10){const centre=(low+high)/2;low=centre-5;high=centre+5;return{min:Math.floor(low/5)*5,max:Math.ceil(high/5)*5}}const target=(high-low)/6,power=10**Math.floor(Math.log10(target)),step=([1,2,5,10].find(n=>n*power>=target)||10)*power;return{min:Math.floor(low/step)*step,max:Math.ceil(high/step)*step}}
function fitTemperatureScale(lock=true){const bounds=currentTemperatureBounds();if(bounds){const range=roundedTemperatureBounds(bounds.low,bounds.high);scaleMinimum=range.min;scaleMaximum=range.max;scaleReady=true}if(lock)scaleLocked=true}
function updateTemperatureScale(){if(W.auto_temperature_range&&(!scaleReady||W.live&&following&&!scaleLocked))fitTemperatureScale(false);$('minimum').textContent=scaleMinimum.toFixed(0)+' °C';$('maximum').textContent=scaleMaximum.toFixed(0)+' °C';$('scale-label').textContent='Temperature · °C · shared scale · '+(W.auto_temperature_range?(W.live&&following&&!scaleLocked?'auto live':'locked'):'custom');const bounds=currentTemperatureBounds(),clipped=[];if(bounds&&bounds.high>scaleMaximum+1e-8)clipped.push('Above '+scaleMaximum.toFixed(0)+' °C uses the top colour');if(bounds&&bounds.low<scaleMinimum-1e-8)clipped.push('Below '+scaleMinimum.toFixed(0)+' °C uses the bottom colour');$('scale-note').textContent=clipped.join('. ')+(clipped.length&&W.auto_temperature_range?'. Fit colours to rescale.':clipped.length?'.':'');$('scale-note').style.display=clipped.length?'block':'none'}
function displayPoint(s,p,i,f){if(s.types[i]===2||f.active[i]<=0)return p;return[p[0],p[1],p[2]-(1-f.active[i])*s.dz_mm/2]}
let projectionCache=null;
function rotated(p){const a=p.map((v,i)=>(v-centre[i])/span),u=Math.cos(yaw)*a[0]-Math.sin(yaw)*a[1],v=Math.sin(yaw)*a[0]+Math.cos(yaw)*a[1];return[u,Math.sin(pitch)*v+Math.cos(pitch)*a[2],Math.cos(pitch)*v-Math.sin(pitch)*a[2]]}
function projectionSize(w,h){const key=[w,h,yaw,pitch,zoom].join(',');if(projectionCache&&projectionCache.key===key)return projectionCache;const corners=Array.from({length:8},(_,k)=>rotated(lo.map((v,a)=>(k>>a)&1?hi[a]:v))),xs=corners.map(p=>p[0]),ys=corners.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);projectionCache={key,u:(xmin+xmax)/2,v:(ymin+ymax)/2,scale:Math.min(Math.max(30,w-64)/Math.max(xmax-xmin,1e-9),Math.max(30,h-54)/Math.max(ymax-ymin,1e-9))*zoom};return projectionCache}
function project(p,w,h){const q=rotated(p),fit=projectionSize(w,h);return[w/2+(q[0]-fit.u)*fit.scale,h/2-(q[1]-fit.v)*fit.scale,q[2]]}
function prepare(id){const c=$(id),w=c.clientWidth,h=c.clientHeight,d=window.devicePixelRatio||1;if(c.width!==Math.round(w*d)||c.height!==Math.round(h*d)){c.width=Math.round(w*d);c.height=Math.round(h*d)}const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);if(!id.startsWith('viewport-'))x.clearRect(0,0,w,h);return{c,x,w,h}}
const stops=[[57,127,186],[85,159,197],[193,117,148],[237,141,81],[252,230,90]];
function color(t){const a=Math.max(0,Math.min(.999999,(t-scaleMinimum)/(scaleMaximum-scaleMinimum)))*4,i=Math.floor(a),u=a-i;return'rgb('+stops[i].map((v,j)=>Math.round(v+(stops[i+1][j]-v)*u)).join(',')+')'}
function axes(ctx,w,h){const origin=[lo[0],lo[1],lo[2]],p=project(origin,w,h);['x','y','z'].forEach((axis,i)=>{const q=origin.slice();q[i]=hi[i];const r=project(q,w,h);ctx.strokeStyle='#486178';ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(r[0],r[1]);ctx.stroke();ctx.fillStyle='#91a9bf';ctx.font='10px sans-serif';ctx.fillText(axis+' (mm)',r[0]+3,r[1])})}
const faceSpecs=[{n:[-1,0,0],a:[0,2,6,4]},{n:[1,0,0],a:[1,5,7,3]},{n:[0,-1,0],a:[0,4,5,1]},{n:[0,1,0],a:[2,3,7,6]},{n:[0,0,-1],a:[0,1,3,2]},{n:[0,0,1],a:[4,6,7,5]}];
function voxelFaces(s,p,i,f,w,h){const c=displayPoint(s,p,i,f),fraction=s.types[i]===2?1:Math.max(0,Math.min(1,f.active[i])),half=s.cell_mm.map((v,a)=>v*(a===2?fraction:1)/2),corners=Array.from({length:8},(_,k)=>project(c.map((v,a)=>v+((k>>a)&1?1:-1)*half[a]),w,h)),towards=[-Math.cos(pitch)*Math.sin(yaw),-Math.cos(pitch)*Math.cos(yaw),Math.sin(pitch)];return faceSpecs.filter(face=>face.n.reduce((v,n,a)=>v+n*towards[a],0)>1e-9).map(face=>{const points=face.a.map(k=>corners[k]);return{points,depth:points.reduce((v,q)=>v+q[2],0)/4,index:i}})}
function planeZ(){return Number($('plane').value)}
function currentTop(s,f){let top=-Infinity;s.points.forEach((p,i)=>{if(f.active[i]>0&&['part','supports'].includes(regionAt(s,f,i)))top=Math.max(top,p[2])});return top}
function xy(p,w,h){const scale=Math.min((w-55)/Math.max(hi[0]-lo[0],.001),(h-45)/Math.max(hi[1]-lo[1],.001))*zoom;return[w/2+(p[0]-centre[0])*scale,h/2-(p[1]-centre[1])*scale,scale]}
function hasPaths(s){return !!(s.paths&&s.paths.length||s.path_geometry&&s.path_geometry.count)}
function firstGeometryFrame(s,t,cell){let low=0,high=s.frames.length;while(low<high){const mid=(low+high)>>1;if(s.frames[mid].t+1e-9<t)low=mid+1;else high=mid}while(low<s.frames.length&&s.frames[low].active[cell]<=0)low++;return low}
function buildGeometryBatches(s,w,h){const stride=drag&&drag.moved?Math.max(1,Math.ceil(s.path_geometry.count/60000)):1,key=[w,h,yaw,pitch,zoom,view,planeZ(),stride].join(',');if(s.geometryCache&&s.geometryCache.key===key)return s.geometryCache.groups;const groups=new Map(),data=s.geometryData;if(!data)return[];for(let n=0;n<s.path_geometry.count;n+=stride){const offset=n*40,t=data.getFloat64(offset,true),cell=data.getUint32(offset+32,true),support=!!data.getUint32(offset+36,true),p=[data.getFloat32(offset+8,true),data.getFloat32(offset+12,true),data.getFloat32(offset+16,true)],q=[data.getFloat32(offset+20,true),data.getFloat32(offset+24,true),data.getFloat32(offset+28,true)];if(view==='layer'&&(thermal?Math.abs((p[2]+q[2])/2-planeZ())>s.dz_mm*.5:t!==Number($('plane').value)))continue;const first=thermal?firstGeometryFrame(s,t,cell):0;if(thermal&&first>=s.frames.length)continue;const id=[first,cell,support].join(':'),a=view==='3d'?project(p,w,h):xy(p,w,h),b=view==='3d'?project(q,w,h):xy(q,w,h);let group=groups.get(id);if(!group){group={shape:new Path2D(),lines:[],bounds:[Infinity,Infinity,-Infinity,-Infinity],index:cell,support,firstFrame:first,depth:0,count:0};groups.set(id,group)}group.shape.moveTo(a[0],a[1]);group.shape.lineTo(b[0],b[1]);group.lines.push(a[0],a[1],b[0],b[1]);group.bounds[0]=Math.min(group.bounds[0],a[0],b[0]);group.bounds[1]=Math.min(group.bounds[1],a[1],b[1]);group.bounds[2]=Math.max(group.bounds[2],a[0],b[0]);group.bounds[3]=Math.max(group.bounds[3],a[1],b[1]);group.depth+=view==='3d'?(a[2]+b[2])/2:0;group.count++}const result=Array.from(groups.values());for(const group of result){group.depth/=group.count;group.lines=Float32Array.from(group.lines)}result.sort((a,b)=>b.depth-a.depth);s.geometryCache={key,groups:result};return result}
function drawCompleteGeometry(s,i,f,ctx,w,h){
 if(thermal)drawContextCells(s,i,f,ctx,w,h);
 if(!s.geometryData){ctx.fillStyle='#a7bbd1';ctx.font='13px sans-serif';ctx.fillText(s.geometryError||'Loading complete print geometry…',14,h/2,Math.max(50,w-28));return}
 const index=thermal?frameIndex(s,f.t):0,groups=buildGeometryBatches(s,w,h).filter(g=>g.firstFrame<=index),width=thermal?1.1:.65;
 // Composite the support layer once: overlapping tiny scans must not make it opaque.
 if($('supports').checked&&groups.some(g=>g.support)){
  const layer=document.createElement('canvas'),d=window.devicePixelRatio||1;layer.width=Math.round(w*d);layer.height=Math.round(h*d);const x=layer.getContext('2d');x.setTransform(d,0,0,d,0,0);x.lineWidth=width;
  for(const group of groups){if(!group.support)continue;x.strokeStyle=thermal?color(f.temperature[group.index]):'#a0bad1';x.stroke(group.shape);if(thermal)picks[i].push({geometry:group,index:group.index})}
  ctx.globalAlpha=.35;ctx.drawImage(layer,0,0,w,h);ctx.globalAlpha=1;
 }
 ctx.lineWidth=width;
 for(const group of groups){if(group.support)continue;ctx.strokeStyle=thermal?color(f.temperature[group.index]):'#c0d5e8';ctx.stroke(group.shape);if(thermal)picks[i].push({geometry:group,index:group.index})}
 if(view==='3d')axes(ctx,w,h);
 if(thermal&&selected){const k=nearestPoint(s,selected),p=view==='3d'?project(s.points[k],w,h):xy(s.points[k],w,h);ctx.strokeStyle='#f5e7ae';ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.stroke()}
}
function geometryHit(point,group){const b=group.bounds;if(point[0]<b[0]-6||point[0]>b[2]+6||point[1]<b[1]-6||point[1]>b[3]+6)return false;const a=group.lines;for(let n=0;n<a.length;n+=4)if(distanceToLine(point,[[a[n],a[n+1]],[a[n+2],a[n+3]]])<6)return true;return false}
async function loadCompleteGeometry(){const requests=new Map();await Promise.all(scenes.filter(s=>s.path_geometry).map(async s=>{try{const asset=s.path_geometry;if(asset.format!=='tracklab-paths-v1'||asset.record_bytes!==40)throw Error('Unsupported geometry data');if(!requests.has(asset.url))requests.set(asset.url,fetch(asset.url).then(response=>{if(!response.ok)throw Error('Geometry could not be loaded');return response.arrayBuffer()}));const buffer=await requests.get(asset.url);if(buffer.byteLength!==asset.count*40)throw Error('Incomplete geometry data');s.geometryData=new DataView(buffer)}catch(error){s.geometryError='Complete geometry unavailable. Reopen this result to retry.';console.error(error)}}));draw()}
function drawPreview(s,i,ctx,w,h){ctx.clearRect(0,0,w,h);if(s.path_geometry){drawCompleteGeometry(s,i,null,ctx,w,h);$('frame-'+i).textContent=s.layer_count+' source layers';$('note-'+i).textContent='Complete imported geometry · no thermal results';return}let count=0;const layer=Number($('plane').value);ctx.lineWidth=.65;for(const a of s.segments){if(a[6]===16&&!$('supports').checked)continue;if(view==='layer'&&a[7]!==layer)continue;const p=view==='3d'?project(a.slice(0,3),w,h):xy(a.slice(0,3),w,h),q=view==='3d'?project(a.slice(3,6),w,h):xy(a.slice(3,6),w,h);ctx.strokeStyle=a[6]===16?'rgba(110,135,157,.24)':'rgba(177,205,228,.55)';ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();count++}if(view==='3d')axes(ctx,w,h);$('frame-'+i).textContent=s.layer_count+' source layers';$('note-'+i).textContent='Imported scan geometry · no thermal results';}
function drawThermal(s,i,f,ctx,w,h){const key=[w,h,window.devicePixelRatio||1,yaw,pitch,zoom,view,planeZ(),f.t,$('supports').checked,$('plate').checked,$('powder').checked,$('representation').value,scaleMinimum,scaleMaximum,selected&&selected.join(','),!!s.geometryData,s.geometryError,!!(drag&&drag.moved)].join(':');if(s.renderKey===key)return;s.renderKey=key;ctx.clearRect(0,0,w,h);const faces=[];picks[i]=[];if($('representation').value==='part'&&hasPaths(s)){if(s.path_geometry)drawCompleteGeometry(s,i,f,ctx,w,h);else drawPartPaths(s,i,f,ctx,w,h);return}if(view==='3d'){s.points.forEach((p,k)=>{if(isVisible(s,f,k))faces.push(...voxelFaces(s,p,k,f,w,h))});faces.sort((a,b)=>b.depth-a.depth);faces.forEach(face=>{const r=regionAt(s,f,face.index);ctx.globalAlpha=r==='powder'?.16:r==='plate'?.85:1;ctx.fillStyle=color(f.temperature[face.index]);ctx.strokeStyle='rgba(13,26,42,.38)';ctx.lineWidth=.4;ctx.beginPath();face.points.forEach((p,k)=>k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();ctx.stroke();picks[i].push(face)});ctx.globalAlpha=1;axes(ctx,w,h);const beam=f.laser&&Math.abs(time-f.t)<1e-9?f.laser.map(v=>v*1000):null;if(beam){const p=project(beam,w,h);ctx.fillStyle=f.on?'#a4ffe7':'#afbdc9';ctx.beginPath();ctx.arc(p[0],p[1],4,0,Math.PI*2);ctx.fill()}}else{s.points.forEach((p,k)=>{if(!isVisible(s,f,k)||Math.abs(p[2]-planeZ())>s.dz_mm*.1)return;const q=xy(p,w,h),cw=s.cell_mm[0]*q[2],ch=s.cell_mm[1]*q[2],points=[[q[0]-cw/2,q[1]-ch/2],[q[0]+cw/2,q[1]-ch/2],[q[0]+cw/2,q[1]+ch/2],[q[0]-cw/2,q[1]+ch/2]];ctx.fillStyle=color(f.temperature[k]);ctx.fillRect(points[0][0],points[0][1],cw,ch);ctx.strokeStyle='rgba(13,26,42,.25)';ctx.strokeRect(points[0][0],points[0][1],cw,ch);picks[i].push({points,index:k})});ctx.fillStyle='#8fa6bd';ctx.fillText('x (mm)',w-50,h-8);ctx.fillText('y (mm)',6,13)}if(selected){const k=nearestPoint(s,selected),p=view==='3d'?project(displayPoint(s,s.points[k],k,f),w,h):xy(s.points[k],w,h);ctx.strokeStyle='#f5e7ae';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.stroke()}}
function distanceToLine(p,line){const a=line[0],b=line[1],dx=b[0]-a[0],dy=b[1]-a[1],u=Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/Math.max(dx*dx+dy*dy,1e-12)));return Math.hypot(p[0]-a[0]-u*dx,p[1]-a[1]-u*dy)}
function completedPath(path,frame){return path.t<=frame.t+1e-9&&frame.active[path.cell]>0}
function contextCellVisible(s,f,i){if(f.active[i]<=0)return false;const region=regionAt(s,f,i);return(region==='plate'&&$('plate').checked)||(region==='powder'&&$('powder').checked)}
function drawContextCells(s,i,f,ctx,w,h){if(view==='3d'){const faces=[];s.points.forEach((p,k)=>{if(contextCellVisible(s,f,k))faces.push(...voxelFaces(s,p,k,f,w,h))});faces.sort((a,b)=>b.depth-a.depth);for(const face of faces){const region=regionAt(s,f,face.index);ctx.globalAlpha=region==='powder'?.12:.85;ctx.fillStyle=color(f.temperature[face.index]);ctx.strokeStyle='rgba(13,26,42,.32)';ctx.lineWidth=.4;ctx.beginPath();face.points.forEach((p,k)=>k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();ctx.stroke();picks[i].push(face)}}else{s.points.forEach((p,k)=>{if(!contextCellVisible(s,f,k)||Math.abs(p[2]-planeZ())>s.dz_mm*.1)return;const q=xy(p,w,h),cw=s.cell_mm[0]*q[2],ch=s.cell_mm[1]*q[2],points=[[q[0]-cw/2,q[1]-ch/2],[q[0]+cw/2,q[1]-ch/2],[q[0]+cw/2,q[1]+ch/2],[q[0]-cw/2,q[1]+ch/2]];ctx.globalAlpha=regionAt(s,f,k)==='powder'?.12:.85;ctx.fillStyle=color(f.temperature[k]);ctx.fillRect(points[0][0],points[0][1],cw,ch);picks[i].push({points,index:k})})}ctx.globalAlpha=1}
function drawPartPaths(s,i,f,ctx,w,h){drawContextCells(s,i,f,ctx,w,h);const lines=[];for(const path of s.paths){if(!completedPath(path,f)||path.support&&!$('supports').checked)continue;if(view==='layer'&&Math.abs((path.p[2]+path.q[2])/2-planeZ())>s.dz_mm*.5)continue;const p=view==='3d'?project(path.p,w,h):xy(path.p,w,h),q=view==='3d'?project(path.q,w,h):xy(path.q,w,h);lines.push({path,p,q,depth:view==='3d'?(p[2]+q[2])/2:0})}lines.sort((a,b)=>b.depth-a.depth);ctx.lineWidth=1.1;for(const line of lines){ctx.globalAlpha=line.path.support?.55:1;ctx.strokeStyle=color(f.temperature[line.path.cell]);ctx.beginPath();ctx.moveTo(line.p[0],line.p[1]);ctx.lineTo(line.q[0],line.q[1]);ctx.stroke();picks[i].push({line:[line.p,line.q],index:line.path.cell})}ctx.globalAlpha=1;if(view==='3d')axes(ctx,w,h);if(selected){const k=nearestPoint(s,selected),p=view==='3d'?project(s.points[k],w,h):xy(s.points[k],w,h);ctx.strokeStyle='#f5e7ae';ctx.beginPath();ctx.arc(p[0],p[1],6,0,Math.PI*2);ctx.stroke()}}
function nearestPoint(s,position){let best=0,d=Infinity;s.points.forEach((p,i)=>{const q=p.reduce((v,n,a)=>v+(n-position[a])**2,0);if(q<d){best=i;d=q}});return best}
function updatePoint(){if(!selected){$('point').textContent='Click a calculated cell to inspect its temperature.';return}const primary=scenes.find(s=>!s.pending);selected=primary.points[nearestPoint(primary,selected)];const labels=scenes.map(s=>{if(s.pending)return s.label+': calculation pending';const k=nearestPoint(s,selected),index=frameIndex(s,time),f=index<0?null:s.frames[index],t=pointTemperature(s,f,k);return s.label+': '+(t===null?'not deposited at this time':t.toFixed(3)+' °C at saved '+f.t.toFixed(3)+' s')});$('point').textContent='Cell '+selected.map(v=>v.toFixed(3)).join(', ')+' mm · '+labels.join(' | ')}
function drawHistory(){if(!$('pointgraph').checked)return;const {x:ctx,w,h}=prepare('history');if(!selected){ctx.fillStyle='#b6c9dd';ctx.fillText('Click a cell in the print to inspect its saved temperature history.',12,25);return}const histories=scenes.map(s=>{if(s.pending)return[];const k=nearestPoint(s,selected);return s.frames.map(f=>({t:f.t,v:pointTemperature(s,f,k)}))}),values=histories.flat().filter(p=>p.v!==null).map(p=>p.v);if(!values.length)return;const low=Math.min(...values),high=Math.max(low+1,...values),x=t=>36+(t-limits.begin)/Math.max(limits.end-limits.begin,1e-12)*(w-50),y=v=>h-22-(v-low)/(high-low)*(h-37);ctx.strokeStyle='#5b728a';ctx.beginPath();ctx.moveTo(34,10);ctx.lineTo(34,h-20);ctx.lineTo(w-10,h-20);ctx.stroke();histories.forEach((points,i)=>{ctx.strokeStyle=['#9ce7dd','#ffa579'][i];ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=1.7;ctx.beginPath();let started=false;for(const p of points){if(p.v===null||p.t>limits.end){started=false;continue}if(started)ctx.lineTo(x(p.t),y(p.v));else ctx.moveTo(x(p.t),y(p.v));started=true}ctx.stroke();for(const p of points){if(p.v===null||p.t>limits.end)continue;ctx.beginPath();ctx.arc(x(p.t),y(p.v),2,0,Math.PI*2);ctx.fill()}});ctx.fillStyle='#b6c9dd';ctx.font='10px sans-serif';ctx.fillText(high.toFixed(1)+' °C',3,10);ctx.fillText(low.toFixed(1)+' °C',3,h-20);ctx.fillText('Physical time (s)',Math.max(40,w-112),h-5);ctx.strokeStyle='#e6dcad';ctx.beginPath();ctx.moveTo(x(time),10);ctx.lineTo(x(time),h-20);ctx.stroke()}
function exposureKind(s,f){let support=false;if(s.paths&&s.paths.length){for(const path of s.paths){if(!completedPath(path,f))continue;if(!path.support)return'part';support=true}}else{for(let i=0;i<s.points.length;i++){if(f.active[i]<=0)continue;const region=regionAt(s,f,i);if(region==='part')return'part';if(region==='supports')support=true}}return support?'supports':'none'}
function pendingDescription(s){if(['queued','running','preparing'].includes(s.state||'queued'))return{title:'Calculating '+s.label+'…',field:'No calculated field yet'};if(['cancelled','canceled'].includes(s.state))return{title:'Not calculated — cancelled',field:'Cancelled · no calculated field'};if(s.state==='failed')return{title:'Calculation failed',field:'Failed · no calculated field'};return{title:'No calculated result',field:'No calculated result'}}
function draw(){if(thermal)updateTemperatureScale();const layerLabels=[];if(thermal&&$('follow').checked&&view==='layer'){const first=scenes.find(s=>!s.pending),f=first.frames[frameIndex(first,time)];if(f){const top=currentTop(first,f);if(Number.isFinite(top))$('plane').value=String(first.z.reduce((a,z)=>Math.abs(z-top)<Math.abs(a-top)?z:a,first.z[0]))}}scenes.forEach((s,i)=>{const {x:ctx,w,h}=prepare('viewport-'+i);if(!thermal){drawPreview(s,i,ctx,w,h);return}if(s.pending){const pending=pendingDescription(s);ctx.fillStyle='#a7bbd1';ctx.font='14px sans-serif';ctx.fillText(pending.title,18,h/2,Math.max(60,w-36));$('frame-'+i).textContent=pending.field;$('note-'+i).textContent=s.message||'No results from the other material are substituted.';return}const k=frameIndex(s,time);if(k<0){ctx.fillStyle='#9eb4cb';ctx.fillText('No calculated temperature frame is available at this physical time.',12,h/2);$('frame-'+i).textContent='Awaiting calculated field';$('note-'+i).textContent='No future temperature is shown.';return}const f=s.frames[k];drawThermal(s,i,f,ctx,w,h);const layer=layerAt(s,time,f);layerLabels.push(s.label+': layer '+layer);$('frame-'+i).textContent='Field '+(k+1)+' / '+s.frames.length+' · '+f.t.toFixed(3)+' s';$('note-'+i).textContent=(time>s.frames[s.frames.length-1].t+1e-9?'Holding final available field. ':'')+(($('representation').value==='part'&&hasPaths(s))?(s.geometry_coverage==='complete'?'Complete scan coverage · nearest calculated cell temperature':'Imported scan geometry · nearest calculated cell temperature'):'Calculated grid cells · drag to rotate, scroll to zoom');if(exposureKind(s,f)==='supports'&&!$('supports').checked){$('note-'+i).textContent='Only support-like regions are exposed at this time; enable Supports to display them.';ctx.fillStyle='#c2d5e8';ctx.font='12px sans-serif';ctx.textAlign='center';['Only support-like regions','are exposed at this time.','Enable Supports below the viewport.'].forEach((line,n)=>ctx.fillText(line,w/2,h/2+(n-1)*18,Math.max(60,w-24)));ctx.textAlign='left'}});if(thermal){$('clock').textContent='Physical print time '+time.toFixed(3)+' s';$('layers').textContent=layerLabels.join(' · ');$('availability').textContent=(W.live?(following?'Following calculation · ':'Live calculation · viewing saved history · '):'')+(scenes.length===2?'Synchronized comparison · ':'')+'Available through '+limits.end.toFixed(3)+' s';$('timeline').value=time;updatePoint();drawHistory();saveCamera()}}
function updateView(){const layer=view==='layer';$('viewlayer').classList.toggle('active',layer);$('view3d').classList.toggle('active',!layer);$('plane-wrap').style.display=layer?'inline-flex':'none';$('follow-wrap').style.display=layer&&thermal?'inline-flex':'none';saveCamera();draw()}
function seek(t){scaleLocked=true;following=false;$('followcalculation').checked=false;time=Math.max(limits.begin,Math.min(limits.end,t));playing=false;last=0;$('play').textContent='▶ Play';draw()}
$('kind').textContent=thermal?(W.live?'Live thermal results':'Thermal results')+(scenes.length===2?' · synchronized material comparison':''):'Geometry preview · no thermal temperatures';
if(thermal){$('fitcolors').style.display=W.auto_temperature_range?'inline-block':'none';$('fitcolors').disabled=!W.auto_temperature_range;$('minimum').textContent=W.vmin.toFixed(0)+' °C';$('maximum').textContent=W.vmax.toFixed(0)+' °C';$('timeline').min=limits.begin;$('timeline').max=limits.end;$('timeline').disabled=limits.end<=limits.begin;scenes.find(s=>!s.pending).z.forEach(z=>{const o=document.createElement('option');o.value=z;o.textContent=z.toFixed(3)+' mm';$('plane').appendChild(o)});$('plane').value=String(scenes.find(s=>!s.pending).z.find(z=>z>=0)||scenes.find(s=>!s.pending).z[0]);$('notice').textContent='Colours and point values come from stored solver temperatures. Between calculated frames the previous field is held. Playback speed changes viewing time only. Coarse full-build cells do not resolve microscopic melt pools.';if(scenes.some(s=>!s.pending&&s.sample_stride>1))$('notice').textContent+=' Display cells are sampled by height; complete arrays remain in the saved results.';if(savedState&&savedState.mode==='thermal')for(const id of ['supports','plate','powder'])if(typeof savedState[id]==='boolean')$(id).checked=savedState[id];$('followcalculation').checked=following;if(!scenes.some(hasPaths)){$('representation').value='cells';$('representation').disabled=true}if(savedState&&savedState.representation==='cells')$('representation').value='cells';$('representation').onchange=()=>{saveCamera();draw()};if(savedState&&savedState.mode==='thermal'&&Number.isFinite(savedState.plane)&&scenes.find(s=>!s.pending).z.includes(savedState.plane)){$('plane').value=String(savedState.plane);$('follow').checked=savedState.followBuild!==false}$('pointgraph').checked=W.show_probe||!!(savedState&&savedState.pointgraph);$('graph').style.display=$('pointgraph').checked?'block':'none'}else{$('supports').checked=savedState&&typeof savedState.supports==='boolean'?savedState.supports:true;const indices=scenes[0].source_layers||[...new Set(scenes[0].segments.map(a=>a[7]))];indices.forEach(layer=>{const o=document.createElement('option');o.value=layer;o.textContent='Source layer '+layer;$('plane').appendChild(o)});$('plane').value=String(indices[indices.length-1]||0);$('geometry-sampling').textContent=scenes[0].geometry_coverage==='complete'?'Complete imported path coverage':scenes[0].displayed_segments<scenes[0].source_segments?'Display-sampled imported paths':'Imported scan paths';$('notice').textContent='Geometry is drawn from imported scan vectors. This neutral preview contains no simulated temperatures. Support-like section 16 can be shown separately. No scan paths or hatch have been invented.'}
$('play').onclick=()=>{scaleLocked=true;following=false;$('followcalculation').checked=false;if(time>=limits.end)time=limits.begin;playing=!playing;last=0;$('play').textContent=playing?'❚❚ Pause':'▶ Play';draw()};$('restart').onclick=()=>seek(limits.begin);$('finish').onclick=()=>seek(limits.end);$('timeline').oninput=e=>seek(Number(e.target.value));$('reset').onclick=()=>{yaw=-.7;pitch=.65;zoom=1;saveCamera();draw()};$('view3d').onclick=()=>{view='3d';updateView()};$('viewlayer').onclick=()=>{view='layer';updateView()};$('plane').onchange=()=>{$('follow').checked=false;saveCamera();draw()};for(const id of ['supports','plate','powder','follow'])$(id).onchange=()=>{saveCamera();draw()};$('followcalculation').onchange=()=>{following=$('followcalculation').checked;scaleLocked=!following;if(following){time=limits.end;playing=false;$('play').textContent='▶ Play'}draw()};$('pointgraph').onchange=()=>{$('graph').style.display=$('pointgraph').checked?'block':'none';saveCamera();drawHistory()};
function contains(p,polygon){let inside=false;for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){const a=polygon[i],b=polygon[j];if(((a[1]>p[1])!==(b[1]>p[1]))&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])inside=!inside}return inside}
scenes.forEach((s,i)=>{const c=$('viewport-'+i);c.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,moved:false};c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(!drag||view!=='3d')return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>3)drag.moved=true;yaw+=dx*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+dy*.008));drag.x=e.clientX;drag.y=e.clientY;saveCamera();draw()};c.onpointerup=e=>{if(thermal&&!s.pending&&drag&&!drag.moved){const box=c.getBoundingClientRect(),p=[e.clientX-box.left,e.clientY-box.top],list=picks[i]||[];for(let j=list.length-1;j>=0;j--)if(list[j].geometry?geometryHit(p,list[j].geometry):list[j].line?distanceToLine(p,list[j].line)<6:contains(p,list[j].points)){selected=s.points[list[j].index];saveCamera();break}}drag=null;draw()};c.onpointercancel=()=>{drag=null;draw()};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));saveCamera();draw()}});
$('fitcolors').onclick=()=>{if(!W.auto_temperature_range)return;fitTemperatureScale(true);draw()};if(playing)$('play').textContent='❚❚ Pause';window.onresize=draw;function tick(stamp){if(playing&&thermal){if(last)time=Math.min(limits.end,time+(stamp-last)/1000*playbackRate($('speed').value));last=stamp;draw();if(time>=limits.end){playing=false;$('play').textContent='▶ Play';saveCamera()}}requestAnimationFrame(tick)}updateView();if(scenes.some(s=>s.path_geometry))loadCompleteGeometry();requestAnimationFrame(tick);
</script></body></html>'''
