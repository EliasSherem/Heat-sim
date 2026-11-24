"""Offline interactive playback of stored build temperatures, without physics interpolation."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def read_build_manifest(folder: Path):
    path = Path(folder) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def _local_path(folder, relative):
    root = Path(folder).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Build result paths must stay inside the result folder")
    return path


def load_build_frame(folder: Path, entry):
    path = _local_path(folder, entry["file"])
    with np.load(path, allow_pickle=False) as data:
        frame = {key: data[key].copy() for key in data.files}
    if "time_s" in entry and not np.isfinite(float(entry["time_s"])):
        raise ValueError("Stored frame timestamps must be finite")
    if "time_s" in frame and (not np.isfinite(float(frame["time_s"])) or not np.isclose(float(frame["time_s"]),float(entry["time_s"]),rtol=0,atol=1e-9)):
        raise ValueError("Stored frame timestamp differs from its manifest")
    shape = tuple(len(frame[key]) for key in ("x_m", "y_m", "z_m"))
    if not all(shape) or frame["temperature_k"].shape != shape:
        raise ValueError("Stored thermal-frame shape does not match its coordinate axes")
    for key in ("x_m", "y_m", "z_m", "temperature_k", "consolidated_fraction"):
        if not np.all(np.isfinite(frame[key])):
            raise ValueError(f"Non-finite stored thermal field: {key}")
    for key in ("x_m", "y_m", "z_m"):
        if frame[key].ndim != 1 or np.any(np.diff(frame[key]) <= 0):
            raise ValueError("Stored thermal axes must be increasing finite cell centres")
    active = frame.get("active_fraction", frame.get("deposited_fraction"))
    if active is None or active.shape != shape or not np.all(np.isfinite(active)):
        raise ValueError("Stored thermal frame needs a finite deposited/active fraction")
    if frame["consolidated_fraction"].shape != shape:
        raise ValueError("Stored consolidation shape does not match temperature")
    for key in ("part_mask","support_mask","plate_mask"):
        if key in frame and frame[key].shape != shape:
            raise ValueError("Stored geometry mask shape does not match temperature")
    return frame


def probe_history(folder: Path, manifest, position_m):
    """Exact stored nearest-cell samples; no invented intermediate temperatures."""
    position_m = np.asarray(position_m,dtype=float)
    if position_m.shape != (3,) or not np.all(np.isfinite(position_m)):
        raise ValueError("Probe position needs three finite coordinates in metres")
    rows = []
    for entry in manifest.get("frames", []):
        frame = load_build_frame(folder, entry)
        indices = tuple(int(np.argmin(abs(frame[key]-value))) for key, value in zip(("x_m", "y_m", "z_m"), position_m))
        active = frame.get("active_fraction", frame.get("deposited_fraction"))
        rows.append({"time_s": float(entry["time_s"]), "temperature_c": float(frame["temperature_k"][indices])-273.15 if active[indices] > 0 else None,
                     "stored_temperature_c": float(frame["temperature_k"][indices])-273.15,
                     "active_fraction": float(active[indices]),
                     "consolidated_fraction": float(frame["consolidated_fraction"][indices]),
                     "cell_x_mm": float(frame["x_m"][indices[0]])*1000,
                     "cell_y_mm": float(frame["y_m"][indices[1]])*1000,
                     "cell_z_mm": float(frame["z_m"][indices[2]])*1000})
    return pd.DataFrame(rows)


@lru_cache(maxsize=24)
def _npz_layer_spans(path, modified, size):
    """Retain only layer clock boundaries, never millions of event objects."""
    with np.load(path,allow_pickle=False) as archive:
        raw=archive["events"]
    if not len(raw):return ()
    cuts=np.r_[0,np.flatnonzero(np.diff(raw["layer_index"])!=0)+1,len(raw)]
    return tuple((float(raw["t0_s"][a]),float(raw["t1_s"][b-1]),int(raw["layer_index"][a]))
                 for a,b in zip(cuts[:-1],cuts[1:]))


def playback_data(folder: Path, manifest, temperature_min_c=20., temperature_max_c=1800., max_points=14000, trajectory_layer=None, live=False, temperature_decimals=2, include_inactive=False):
    """Read stored display cells; sampling never changes numerical results.

    ``temperature_decimals=None`` preserves each stored cell's exact float
    value converted to Celsius, for point inspection in the main workspace.
    """
    entries = [entry for entry in manifest.get("frames", []) if _local_path(folder,entry["file"]).exists()]
    if not entries:
        return None
    times = np.array([float(entry["time_s"]) for entry in entries])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):
        raise ValueError("Stored playback frames must have increasing finite timestamps")
    reference = load_build_frame(folder, entries[-1])
    shape = reference["temperature_k"].shape
    mesh = np.meshgrid(reference["x_m"], reference["y_m"], reference["z_m"], indexing="ij")
    active = reference.get("active_fraction", reference.get("deposited_fraction"))
    eligible = np.flatnonzero(active.ravel() > 0)
    # Include inactive cells in the planned material footprint for earlier views.
    planned = reference.get("part_mask", np.zeros(shape, bool)) | reference.get("support_mask", np.zeros(shape, bool)) | reference.get("plate_mask", np.zeros(shape, bool))
    eligible = np.union1d(eligible, np.flatnonzero(planned.ravel()))
    if include_inactive:eligible=np.arange(np.prod(shape))
    if not len(eligible):
        eligible = np.arange(np.prod(shape))
    if not isinstance(max_points,int) or max_points < 1:
        raise ValueError("Display point budget must be a positive integer")
    if not np.isfinite(temperature_min_c) or not np.isfinite(temperature_max_c) or temperature_max_c <= temperature_min_c:
        raise ValueError("Temperature scale must have finite increasing Celsius bounds")
    # Sample separately within each z plane: flat-array strides can omit whole
    # selectable planes when the stride and z dimension share a divisor.
    occupied_planes = np.unique(eligible % shape[2])
    if len(occupied_planes) > max_points:
        raise ValueError("Display point budget must cover each occupied z plane")
    per_plane = max(1,max_points//len(occupied_planes))
    sampled=[];strides=[]
    for plane in occupied_planes:
        candidates=eligible[eligible%shape[2]==plane]
        step=max(1,int(np.ceil(len(candidates)/per_plane)))
        sampled.append(candidates[::step]);strides.append(step)
    chosen = np.sort(np.concatenate(sampled))
    stride = max(strides)
    points = np.column_stack([axis.ravel()[chosen]*1000 for axis in mesh])
    plate = reference.get("plate_mask", np.zeros(shape, bool)).ravel()[chosen]
    support = reference.get("support_mask", np.zeros(shape, bool)).ravel()[chosen]
    final_part=reference.get("part_mask",reference["consolidated_fraction"]>.001).ravel()[chosen]
    types = np.where(plate,2,np.where(support,1,np.where(final_part,0,3))).astype(int)
    frames = []
    for entry in entries:
        frame = load_build_frame(folder, entry)
        active = frame.get("active_fraction", frame.get("deposited_fraction"))
        if frame["temperature_k"].shape != shape or any(not np.array_equal(frame[key],reference[key]) for key in ("x_m","y_m","z_m")):
            raise ValueError("Playback requires the same stored mesh in every frame")
        laser = entry.get("laser_position_m")
        if laser is not None and (np.asarray(laser).shape != (3,) or not np.all(np.isfinite(laser))):
            if entry.get("laser_on"):
                raise ValueError("An active stored beam position must contain three finite coordinates")
            laser = None
        temperature=np.asarray(frame["temperature_k"].ravel()[chosen],dtype=float)-273.15
        if temperature_decimals is not None:temperature=np.round(temperature,temperature_decimals)
        frames.append({"t": float(entry["time_s"]), "layer": int(entry.get("layer_index", 0)),
                       "temperature": temperature.tolist(),
                       "active": np.round(active.ravel()[chosen], 4).tolist(),
                       "solid": np.round(frame["consolidated_fraction"].ravel()[chosen], 4).tolist(),
                       "part":frame.get("part_mask",frame["consolidated_fraction"]>.001).ravel()[chosen].astype(bool).tolist(),
                       "support":frame.get("support_mask",np.zeros(shape,bool)).ravel()[chosen].astype(bool).tolist(),
                       "laser": laser, "on": bool(entry.get("laser_on", False))})
    timeline_path = _local_path(folder,manifest.get("timeline_file", "timeline.json"))
    events = []
    trajectory_status = "not_embedded"
    trajectory_scope = None
    layer_spans=[]
    if trajectory_layer is not None and (not isinstance(trajectory_layer,(int,np.integer)) or trajectory_layer<0):
        raise ValueError("Detailed trajectory layer must be a nonnegative integer")
    if not live and timeline_path.exists() and timeline_path.suffix==".npz":
        stamp=timeline_path.stat()
        layer_spans=[{"a":a,"b":b,"layer":layer} for a,b,layer in
                     _npz_layer_spans(str(timeline_path),stamp.st_mtime_ns,stamp.st_size)]
    if not live and timeline_path.exists() and (int(manifest.get("timeline_event_count",0)) <= 40000 or trajectory_layer is not None):
        if timeline_path.suffix == ".npz":
            with np.load(timeline_path,allow_pickle=False) as archive:
                raw=archive["events"]
            if trajectory_layer is not None:
                selected=raw[raw["layer_index"]==trajectory_layer]
                del raw
            else:
                selected=raw
            if trajectory_layer is not None and len(selected):
                trajectory_scope={"layer_index":int(trajectory_layer),
                                  "start_s":float(selected["t0_s"].min()),"end_s":float(selected["t1_s"].max())}
            timeline=[]
            if len(selected)<=40000:
                for event in selected:
                    timeline.append({"t0_s":float(event["t0_s"]),"t1_s":float(event["t1_s"]),
                                     "start_m":event["start_m"].tolist(),"end_m":event["end_m"].tolist(),
                                     "laser_on":bool(event["kind"]==1 and event["power_w"]>0),
                                     "layer_index":int(event["layer_index"])})
            else:
                trajectory_status="too_many_events_in_layer" if trajectory_layer is not None else "not_embedded"
        else:
            timeline = json.loads(timeline_path.read_text())
            timeline = timeline.get("events", []) if isinstance(timeline, dict) else timeline
            for event in timeline:
                a=float(event.get("t0_s",event.get("start_s",0)))
                b=float(event.get("t1_s",event.get("end_s",0)))
                layer=int(event.get("layer_index",0))
                if layer_spans and layer_spans[-1]["layer"]==layer:layer_spans[-1]["b"]=b
                else:layer_spans.append({"a":a,"b":b,"layer":layer})
            if trajectory_layer is not None:
                timeline=[event for event in timeline if int(event.get("layer_index",0))==trajectory_layer]
                if timeline:
                    trajectory_scope={"layer_index":int(trajectory_layer),
                                      "start_s":min(float(event.get("t0_s",event.get("start_s",0)))for event in timeline),
                                      "end_s":max(float(event.get("t1_s",event.get("end_s",0)))for event in timeline)}
        # Never decimate path geometry and pretend it is an exact trajectory.
        if len(timeline) <= 40000:
            for event in timeline:
                start = event.get("start_m", event.get("p0_m"))
                end = event.get("end_m", event.get("p1_m"))
                if start is None or end is None:
                    continue
                p,q=np.asarray(start,float),np.asarray(end,float)
                if p.shape!=(3,) or q.shape!=(3,) or not np.all(np.isfinite(np.r_[p,q])):
                    raise ValueError("Beam events require finite three-dimensional coordinates")
                events.append({"a": float(event.get("t0_s", event.get("start_s", 0))),
                               "b": float(event.get("t1_s", event.get("end_s", 0))),
                               "p": (p*1000).tolist(), "q": (q*1000).tolist(),
                               "on": bool(event.get("laser_on", event.get("kind") == "scan")),
                               "layer": int(event.get("layer_index", 0))})
            if events:
                trajectory_status = "exact_stored_layer_events" if trajectory_layer is not None else "exact_stored_events"
        else:
            trajectory_status = "too_many_events_in_layer" if trajectory_layer is not None else "not_embedded"
    cells=[float(np.diff(reference[key])[0])*1000 if len(reference[key])>1 else
           float(manifest.get("config",{}).get("z_spacing_m" if key=="z_m" else "spacing_m",0.))*1000
           for key in ("x_m","y_m","z_m")]
    extents = np.array([[axis.min()*1000-cell/2,axis.max()*1000+cell/2]
                        for axis,cell in zip((reference["x_m"],reference["y_m"],reference["z_m"]),cells)])
    data = {"points": np.round(points, 5).tolist(), "types": types.tolist(), "frames": frames,
            "events": events, "bounds": extents.tolist(), "z": (reference["z_m"]*1000).tolist(),
            "layer_spans":layer_spans,"case_status":manifest.get("status","unknown"),
            "live":bool(live),"view_key":str(Path(folder).resolve()),
            "requested_duration_s":manifest.get("physical_duration_s",float(times[-1])),
            "simulated_layer_start":int(manifest.get("config",{}).get("layer_start",min((f["layer"] for f in frames),default=0))),
            "simulated_layer_end":int(manifest.get("config",{}).get("layer_end") or max((f["layer"] for f in frames),default=0)+1),
            "vmin": temperature_min_c, "vmax": temperature_max_c,
            "sample_stride": stride, "sample_policy":"stratified by z plane",
            "dz_mm":cells[2],
            "trajectory_status":trajectory_status,"trajectory_scope":trajectory_scope,
            "trajectory_waiting":bool(trajectory_scope and trajectory_scope["start_s"]>float(entries[-1]["time_s"])),
            "cell_mm":cells,"alloy": manifest.get("config", {}).get("alloy", "material")}
    data["source_sha256"]=manifest.get("job",{}).get("source_sha256")
    data["job_name"]=manifest.get("job",{}).get("name")
    return data


def playback_html(folder: Path, manifest, temperature_min_c=20., temperature_max_c=1800., max_points=14000, trajectory_layer=None, live=False):
    """A Canvas viewer; point sampling affects display only, never solver results."""
    data=playback_data(folder,manifest,temperature_min_c,temperature_max_c,max_points,trajectory_layer,live)
    if data is None:return "<p style='color:#b7c6d8'>No completed thermal frames yet.</p>"
    serialized = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return _HTML.replace("__DATA__", serialized)


_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#101722;color:#dce7f5;font:13px -apple-system,BlinkMacSystemFont,Arial,sans-serif}.tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 0}button,select,input[type=number]{background:#233348;border:1px solid #43546a;color:#e6eef9;border-radius:6px;padding:6px 9px}button{cursor:pointer}label{color:#b6c7dd}input[type=range]{accent-color:#ff915c}.views{display:grid;grid-template-columns:1.2fr 1fr;gap:10px}.panel{background:#111e30;border:1px solid #293b50;border-radius:9px;overflow:hidden}canvas{width:100%;height:380px;display:block}.caption{font-size:11px;color:#9db1c9;padding:7px 10px}.time{font-variant-numeric:tabular-nums;color:#80e8db}.scale{height:10px;background:linear-gradient(90deg,#397fba,#559fc5,#c17594,#ed8d51,#fce65a);border-radius:4px}.limits{display:flex;justify-content:space-between;font-size:11px;color:#a9bad0;margin:5px 0}.status{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;padding:8px 0}.timeline{width:100%}.notice{color:#e8bb81;line-height:1.5;font-size:11px;margin-top:9px}.live .playback-only{display:none}@media(max-width:740px){.views{grid-template-columns:1fr}canvas{height:270px}}</style></head><body>
<div id="coverage" class="status"></div>
<div class="tools"><button class="playback-only" id="play">▶ Play</button><button class="playback-only" id="start">Start</button><button class="playback-only" id="finish">Show finished print</button><label class="playback-only">Playback speed <select id="speed"><option value="auto" selected>Auto (~1 minute)</option><option value="0.1">0.1×</option><option value="0.5">0.5×</option><option value="1">1×</option><option value="4">4×</option><option value="16">16×</option><option value="64">64×</option><option value="256">256×</option><option value="1024">1024×</option></select></label><button id="reset">Reset view</button></div>
<input class="timeline playback-only" id="timeline" type="range" min="0" max="1" step="any" value="0" aria-label="Physical print time">
<div class="status"><span id="physical" class="time"></span><span id="thermal"></span><span id="layer"></span><span id="laser"></span></div>
<div class="tools"><label><input id="powder" type="checkbox">Unexposed powder</label><label><input id="solid" type="checkbox" checked>Exposed part</label><label><input id="support" type="checkbox" checked>Supports</label><label><input id="plate" type="checkbox" checked>Build plate</label><label><input id="paths" type="checkbox" checked>Exposure paths</label><label><input id="lock" type="checkbox" checked>Lock temperature scale</label></div>
<div class="views"><div class="panel"><canvas id="volume"></canvas><div class="caption">3D calculated grid cells • drag to rotate, wheel to zoom • axes in mm</div></div><div class="panel"><canvas id="section"></canvas><div class="caption"><label><input id="follow" type="checkbox" checked>Follow build height</label> <label>Layer / plane: <select id="plane"></select></label> • stored cell temperatures, °C</div></div></div>
<div class="scale"></div><div class="limits"><span id="low"></span><span>Temperature (°C)</span><span id="high"></span></div>
<div class="notice">Temperatures are held at the latest calculated frame at or before the cursor; no thermal interpolation is used. Full-print mode spans every saved thermal frame. Exact layer-clock boundaries come from the stored schedule; large event schedules remain in the download and beam position is only shown where known. Unexposed powder is hidden initially so the exposed part remains visible. This footprint and its consolidation proxy do not establish density or successful fusion. The 3D point height follows the deposited bottom fraction of a vertical cell; plane choices retain fixed finite-volume centres and temperatures are unchanged. Coarse cells and consolidation proxies do not resolve microscopic melt pools. <span id="sampling"></span> <span id="trajectory"></span></div>
<script>const D=__DATA__;const $=s=>document.getElementById(s);let time=D.frames[0].t,playing=false,last=0,yaw=-.7,pitch=.65,zoom=1,drag=null;
function playbackBounds(data){const first=data.frames[0].t,last=data.frames[data.frames.length-1].t,end=data.trajectory_scope?Math.min(data.trajectory_scope.end_s,last):last,begin=data.trajectory_scope?Math.min(end,Math.max(data.trajectory_scope.start_s,first)):first;return{begin,end}}
const {begin,end}=playbackBounds(D);time=D.live?end:begin;$("timeline").min=begin;$("timeline").max=end;$("timeline").value=time;let currentIndex=0;
function playbackRate(value){return value==='auto'?Math.max(1,(end-begin)/60):Number(value)}
if(D.live)document.body.classList.add('live');$("coverage").textContent=D.live?'Latest calculated field · source layer '+D.frames[D.frames.length-1].layer+' · time '+end.toFixed(2)+' s':(D.trajectory_scope?'One source layer: '+D.trajectory_scope.layer_index:'All simulated layers: '+D.simulated_layer_start+'–'+(D.simulated_layer_end-1))+' · '+begin.toFixed(2)+'–'+end.toFixed(2)+' s · '+D.frames.length+' saved thermal frames';$("finish").textContent=D.trajectory_scope?'Show layer end':D.case_status==='complete'?'Show finished print':'Show latest calculated frame';
try{const saved=JSON.parse(sessionStorage.getItem('tracklab.camera.'+D.view_key));if(saved&&[saved.yaw,saved.pitch,saved.zoom].every(Number.isFinite)){yaw=saved.yaw;pitch=saved.pitch;zoom=saved.zoom}}catch{}
function saveView(){try{sessionStorage.setItem('tracklab.camera.'+D.view_key,JSON.stringify({yaw,pitch,zoom}))}catch{}}
D.z.forEach((z,i)=>{let o=document.createElement('option');o.value=i;o.textContent=z.toFixed(3)+' mm';$("plane").appendChild(o)});$("plane").value=String(D.z.findIndex(z=>z>=0));
const lo=D.bounds.map(b=>b[0]),hi=D.bounds.map(b=>b[1]),center=lo.map((v,i)=>(v+hi[i])/2),span=Math.max(...lo.map((v,i)=>hi[i]-v),.001);
function lerp(a,b,t){return a+(b-a)*t}const stops=[[57,127,186],[85,159,197],[193,117,148],[237,141,81],[252,230,90]];
function color(t,v0,v1){let s=Math.max(0,Math.min(.999999,(t-v0)/Math.max(v1-v0,1e-8)))*4,i=Math.floor(s),f=s-i;return 'rgb('+stops[i].map((v,j)=>Math.round(lerp(v,stops[i+1][j],f))).join(',')+')'}
function frameAt(t){let l=0,r=D.frames.length-1;while(l<r){let m=Math.ceil((l+r)/2);if(D.frames[m].t<=t)l=m;else r=m-1}return l}
function eventAt(t){let l=0,r=D.events.length-1;while(l<=r){let m=(l+r)>>1,e=D.events[m];if(t<e.a)r=m-1;else if(t>e.b)l=m+1;else return e}return null}
function layerAt(t,f){const spans=D.layer_spans||[];let l=0,r=spans.length-1,found=null;while(l<=r){let m=(l+r)>>1,e=spans[m];if(e.a<=t){found=e;l=m+1}else r=m-1}return found&&t<=found.b+1e-9?found.layer:f.layer}
function beamAt(t,f){let e=eventAt(t);if(e){let u=(t-e.a)/Math.max(e.b-e.a,1e-15);return{p:e.p.map((v,i)=>lerp(v,e.q[i],u)),on:e.on,layer:e.layer,known:true}}if(Math.abs(t-f.t)<1e-9)return{p:f.laser?f.laser.map(v=>v*1000):null,on:f.on,layer:layerAt(t,f),known:true};return{p:null,on:false,layer:layerAt(t,f),known:false}}
function displayPoint(p,i,f){if(D.types[i]===2||f.active[i]<=0)return p;return[p[0],p[1],p[2]-(1-f.active[i])*D.dz_mm/2]}
function voxelCorners(p,i,f){const c=displayPoint(p,i,f),fraction=D.types[i]===2?1:Math.max(0,Math.min(1,f.active[i])),half=[D.cell_mm[0]/2,D.cell_mm[1]/2,D.cell_mm[2]*fraction/2];return Array.from({length:8},(_,k)=>c.map((v,axis)=>v+((k>>axis)&1?1:-1)*half[axis]))}
function viewPoint(p,w,h){let a=p.map((v,i)=>(v-center[i])/span),u=Math.cos(yaw)*a[0]-Math.sin(yaw)*a[1],v=Math.sin(yaw)*a[0]+Math.cos(yaw)*a[1],z=Math.sin(pitch)*v+Math.cos(pitch)*a[2];return[w/2+u*w*.78*zoom,h*.53-z*w*.78*zoom,Math.cos(pitch)*v-Math.sin(pitch)*a[2]]}
function voxelFaces(p,i,f,w,h){const corners=voxelCorners(p,i,f).map(q=>viewPoint(q,w,h)),towardCamera=[-Math.cos(pitch)*Math.sin(yaw),-Math.cos(pitch)*Math.cos(yaw),Math.sin(pitch)],specs=[{n:[-1,0,0],ids:[0,2,6,4]},{n:[1,0,0],ids:[1,5,7,3]},{n:[0,-1,0],ids:[0,4,5,1]},{n:[0,1,0],ids:[2,3,7,6]},{n:[0,0,-1],ids:[0,1,3,2]},{n:[0,0,1],ids:[4,6,7,5]}];return specs.filter(face=>face.n.reduce((dot,v,axis)=>dot+v*towardCamera[axis],0)>1e-9).map(face=>{const vertices=face.ids.map(k=>corners[k]);return{vertices,depth:vertices.reduce((sum,q)=>sum+q[2],0)/4,i}})}
function canvas(id){let c=$(id),w=c.clientWidth,h=c.clientHeight,d=window.devicePixelRatio||1;if(c.width!==Math.round(w*d)||c.height!==Math.round(h*d)){c.width=Math.round(w*d);c.height=Math.round(h*d)}let ctx=c.getContext('2d');ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);return[c,ctx,w,h]}
function regionAt(i,f){if(D.types[i]===2)return'plate';if(f.support?f.support[i]:D.types[i]===1&&f.solid[i]>.001)return'support';return(f.part?f.part[i]:f.solid[i]>.001)?'solid':'powder'}
function visible(i,f){return f.active[i]>0&&$(regionAt(i,f)).checked}
function followPlane(f){let top=-Infinity;D.points.forEach((p,i)=>{if(f.active[i]>0&&regionAt(i,f)!=='plate'&&regionAt(i,f)!=='powder')top=Math.max(top,p[2])});if(!Number.isFinite(top))D.points.forEach((p,i)=>{if(f.active[i]>0&&D.types[i]!==2)top=Math.max(top,p[2])});if(Number.isFinite(top)){let index=D.z.reduce((best,z,i)=>Math.abs(z-top)<Math.abs(D.z[best]-top)?i:best,0);$("plane").value=String(index)}}
function draw(){currentIndex=frameAt(time);let f=D.frames[currentIndex],b=beamAt(time,f),values=f.temperature.filter((v,i)=>f.active[i]>0),v0=$("lock").checked?D.vmin:Math.min(...values),v1=$("lock").checked?D.vmax:Math.max(...values,v0+1);$("low").textContent=v0.toFixed(0)+' °C';$("high").textContent=v1.toFixed(0)+' °C';$("physical").textContent='Physical time '+time.toFixed(5)+' s';$("thermal").textContent='Calculated frame '+currentIndex+' at '+f.t.toFixed(5)+' s';$("layer").textContent='Layer '+b.layer;$("laser").textContent=b.p?(b.on?'Laser on • ':'Laser off • ')+b.p.map(v=>v.toFixed(3)).join(', ')+' mm':b.known?'Laser off':'Beam position unavailable between calculated frames';
if($("follow").checked)followPlane(f);let [c,ctx,w,h]=canvas('volume');let faces=[];D.points.forEach((p,i)=>{if(visible(i,f))faces.push(...voxelFaces(p,i,f,w,h))});faces.sort((a,b)=>b.depth-a.depth);faces.forEach(face=>{let region=regionAt(face.i,f);ctx.globalAlpha=region==='powder'?.15:region==='plate'?.92:1;ctx.fillStyle=color(f.temperature[face.i],v0,v1);ctx.strokeStyle='rgba(15,29,48,.42)';ctx.lineWidth=.55;ctx.beginPath();face.vertices.forEach((q,k)=>{if(k)ctx.lineTo(q[0],q[1]);else ctx.moveTo(q[0],q[1])});ctx.closePath();ctx.fill();ctx.stroke()});ctx.globalAlpha=1;
let origin=[lo[0],lo[1],lo[2]],p0=viewPoint(origin,w,h);['x','y','z'].forEach((axis,i)=>{let q=origin.slice();q[i]=hi[i];let p=viewPoint(q,w,h);ctx.strokeStyle='#69849e';ctx.beginPath();ctx.moveTo(p0[0],p0[1]);ctx.lineTo(p[0],p[1]);ctx.stroke();ctx.fillStyle='#b8cde2';ctx.fillText(axis+' '+hi[i].toFixed(1)+' mm',p[0]+4,p[1])});
if($("paths").checked){ctx.strokeStyle='#95c6d0';ctx.globalAlpha=.30;let shown=0;for(let e of D.events){if(e.layer===b.layer&&e.on&&e.a<=time){let a=viewPoint(e.p,w,h),q=viewPoint(e.q,w,h);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();if(++shown>=10000)break}}ctx.globalAlpha=1}if(b.p){let q=viewPoint(b.p,w,h);ctx.fillStyle=b.on?'#a5fff0':'#a2b0c4';ctx.beginPath();ctx.arc(q[0],q[1],5,0,7);ctx.fill();ctx.fillStyle='#c7f7ec';ctx.fillText(b.on?'LASER':'JUMP / PAUSE',q[0]+8,q[1]-6)}
let [s,sc,sw,sh]=canvas('section');let z=D.z[Number($("plane").value)],zstep=D.z.length>1?Math.abs(D.z[1]-D.z[0]):1;function xy(p){return[30+(p[0]-lo[0])/Math.max(hi[0]-lo[0],.001)*(sw-55),sh-28-(p[1]-lo[1])/Math.max(hi[1]-lo[1],.001)*(sh-45)]}let cellw=Math.max(1,D.cell_mm[0]/Math.max(hi[0]-lo[0],.001)*(sw-55)),cellh=Math.max(1,D.cell_mm[1]/Math.max(hi[1]-lo[1],.001)*(sh-45));D.points.forEach((p,i)=>{if(visible(i,f)&&Math.abs(p[2]-z)<zstep*.1){let q=xy(p);sc.fillStyle=color(f.temperature[i],v0,v1);sc.fillRect(q[0]-cellw/2,q[1]-cellh/2,cellw,cellh)}});sc.strokeStyle='#6f879f';sc.beginPath();sc.moveTo(28,10);sc.lineTo(28,sh-25);sc.lineTo(sw-15,sh-25);sc.stroke();sc.fillStyle='#b8cde2';sc.fillText('x (mm)',sw-65,sh-8);sc.fillText('y (mm)',4,15);if(b.p&&Math.abs(b.p[2]-z)<zstep){let q=xy(b.p);sc.fillStyle=b.on?'#a5fff0':'#a2b0c4';sc.beginPath();sc.arc(q[0],q[1],4,0,7);sc.fill()}
$("sampling").textContent=D.sample_stride>1?'Display samples eligible cells separately in each z plane (largest stride '+D.sample_stride+'); complete arrays remain in the downloaded results.':'Display uses stored mesh cell values.';$("trajectory").textContent=D.trajectory_waiting?' Awaiting calculated thermal frames for this detailed source layer; showing the latest available calculated time.':D.trajectory_scope?' Detailed trajectory scope: source layer '+D.trajectory_scope.layer_index+' ('+D.trajectory_scope.start_s.toFixed(3)+'–'+D.trajectory_scope.end_s.toFixed(3)+' s). All stored thermal frames are retained; the latest calculated frame is held. '+(D.trajectory_status==='too_many_events_in_layer'?'This layer exceeds the event display limit; exact events remain in the download.':''):''}
function seekTo(value){time=Math.max(begin,Math.min(end,value));playing=false;last=0;$("play").textContent='▶ Play';$("timeline").value=time;draw()}
$("play").onclick=()=>{if(time>=end)time=begin;playing=!playing;$("play").textContent=playing?'❚❚ Pause':'▶ Play';last=0;draw()};$("timeline").oninput=e=>seekTo(Number(e.target.value));$("start").onclick=()=>seekTo(begin);$("finish").onclick=()=>seekTo(end);$("reset").onclick=()=>{yaw=-.7;pitch=.65;zoom=1;saveView();draw()};for(let id of ['powder','solid','support','plate','paths','lock','follow'])$(id).onchange=draw;$("plane").onchange=()=>{$("follow").checked=false;draw()};
$("volume").onpointerdown=e=>{drag=[e.clientX,e.clientY];e.currentTarget.setPointerCapture(e.pointerId)};$("volume").onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];saveView();draw()}};$("volume").onpointerup=()=>drag=null;$("volume").onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));saveView();draw()};window.onresize=draw;
function tick(stamp){if(playing){if(last)time=Math.min(end,time+(stamp-last)/1000*playbackRate($("speed").value));last=stamp;$("timeline").value=time;draw();if(time>=end){playing=false;$("play").textContent='▶ Play'}}requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);
</script></body></html>'''
