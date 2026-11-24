"""Read-only run identities and result availability, including older saved runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import math


@lru_cache(maxsize=512)
def _json_cached(path, modified, size):
    return json.loads(Path(path).read_text())


def read_json(path, default=None):
    try:
        p=Path(path);s=p.stat()
        value=_json_cached(str(p.resolve()),s.st_mtime_ns,s.st_size)
        return value if isinstance(value,dict) else (default or {})
    except (OSError,ValueError,TypeError):
        return default or {}


def _timestamp(value, fallback):
    if value:
        try:
            dt=datetime.fromisoformat(str(value).replace("Z","+00:00"))
            return dt.replace(tzinfo=timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp()
        except (ValueError,TypeError):
            pass
    return float(fallback)


def _mtime(path):
    try:return Path(path).stat().st_mtime
    except OSError:return 0.


def _child(folder, value):
    try:
        root=Path(folder).resolve();target=(root/value).resolve()
        return target if target.is_relative_to(root) else None
    except (TypeError,ValueError,OSError):return None


def _frames(folder, manifest):
    count=0
    for entry in manifest.get("frames",[]):
        if not isinstance(entry,dict):continue
        path=_child(folder,entry.get("file"))
        if path is not None and path.is_file():count+=1
    return count


def _case_data(folder, report):
    cases=[];seen=set()
    for row in report.get("rows",[]):
        if not isinstance(row,dict) or row.get("valid") is False:continue
        case=_child(folder,row.get("case_folder"))
        if case is None:continue
        manifest=read_json(case/"manifest.json")
        count=_frames(case,manifest)
        if count:
            cases.append((case,row,manifest,count));seen.add(case)
    # A batch can expose its currently calculated frames before the first case
    # finishes. Completed orphan cases from earlier code versions stay hidden.
    legacy_all_failed=(report.get("status")=="complete" and not report.get("rows") and bool(report.get("failures")))
    if report.get("status") in ("running","cancelled","failed","partial") or not report or legacy_all_failed:
        for path in sorted(Path(folder).glob("cases/*/manifest.json")):
            if path.parent.resolve() in seen:continue
            manifest=read_json(path)
            if legacy_all_failed and manifest.get("status") not in ("failed","cancelled"):
                continue
            if manifest.get("status")=="complete" and report:continue
            count=_frames(path.parent,manifest)
            if count:
                failed_row=next((row for row in report.get("failures",[]) if isinstance(row,dict) and _child(folder,row.get("case_folder"))==path.parent.resolve()),{})
                cases.append((path.parent.resolve(),failed_row,manifest,count));seen.add(path.parent.resolve())
    return cases


@dataclass(frozen=True)
class RunInfo:
    folder: Path
    run_id: str
    job_name: str
    job_sha256: str | None
    material: str
    target_materials: tuple
    kind: str
    layer_start: int
    layer_end: int | None
    total_layers: int | None
    created_utc: str
    created_timestamp: float
    updated_timestamp: float
    state: str
    frame_count: int
    completed_case_count: int
    failed_case_count: int
    message: str
    phase: str
    progress: float
    study_name: str = ""

    @property
    def has_results(self):return self.frame_count>0

    @property
    def material_label(self):
        return self.material+(" → "+", ".join(self.target_materials) if self.target_materials else "")

    @property
    def layer_label(self):
        end=self.layer_end if self.layer_end is not None else self.total_layers
        if end is None:
            return f"layers {self.layer_start} onward"
        count=max(0,end-self.layer_start)
        if self.total_layers is not None and self.layer_start==0 and end==self.total_layers:
            return f"Full job · {self.total_layers} layers (0–{end-1})"
        total=f" of {self.total_layers}" if self.total_layers is not None else ""
        return f"layers {self.layer_start}–{end-1} ({count}{total})"

    @property
    def status_label(self):
        labels={"complete":"Completed","partial":"Partly completed","running":"Running","queued":"Preparing","failed":"Failed","cancelled":"Cancelled","unknown":"Unknown status"}
        value=labels.get(self.state,self.state.title())
        if self.state in ("failed","cancelled"):
            value+=" · partial results" if self.has_results else " · no results"
        return value

    @property
    def label(self):
        when=datetime.fromtimestamp(self.created_timestamp,ZoneInfo("Asia/Hebron")).strftime("%d %b %H:%M")
        study=f" · {self.study_name}" if self.study_name else ""
        return f"{self.job_name}{study} · {self.material_label} · {self.layer_label} · {when} · {self.status_label} · {self.run_id[-6:]}"


def read_run(folder):
    folder=Path(folder).resolve()
    request=read_json(folder/"request.json");status=read_json(folder/"status.json")
    manifest=read_json(folder/"manifest.json");report=read_json(folder/"report.json")
    if not any((request,status,manifest,report)):return None
    summary=read_json(folder/"job"/"job-summary.json")
    cfg=request.get("build_config") or manifest.get("config") or report.get("config") or {}
    job=manifest.get("job",{})
    metadata=job.get("metadata",{})
    cases=_case_data(folder,report)
    if not job and cases:job=cases[0][2].get("job",{});metadata=job.get("metadata",{})
    own_frames=_frames(folder,manifest)
    # Distinct named presets may share a cached calculation. Preserve their
    # comparison identities, while counting the underlying frame files once.
    frame_count=own_frames+sum({item[0]:item[3] for item in cases}.values())
    completed=(int(manifest.get("status")=="complete" and own_frames>0)
               +sum(item[2].get("status")=="complete" and item[3]>0 for item in cases))
    failures=len(report.get("failures",[]))
    if not report and manifest.get("status")=="failed":failures=1
    state=status.get("state") or report.get("status") or manifest.get("status") or "unknown"
    if report:
        if report.get("status")=="cancelled":state="cancelled"
        elif report.get("status") in ("complete","partial"):
            if not completed:state="failed"
            elif failures:state="partial"
            elif state not in ("running","queued","cancelled"):state="complete"
    elif manifest.get("status") in ("failed","cancelled"):
        state=manifest["status"]
    elif state=="complete" and not own_frames:
        state="failed"
    job_hash=(summary.get("source_sha256") or report.get("job_source_sha256") or job.get("source_sha256") or request.get("job_source_sha256"))
    name=(summary.get("name") or job.get("name") or request.get("job_name") or "Imported job")
    options=request.get("options",{})
    kind=request.get("kind") or ("transfer" if report else "build")
    targets=tuple(report.get("target_materials") or options.get("target_materials",[]))
    material=cfg.get("alloy") or report.get("baseline_material") or "Material not recorded"
    study_name=""
    if kind=="presets":
        reference=options.get("reference",{})
        material=report.get("baseline_material") or reference.get("material") or material
        if not targets:targets=tuple(dict.fromkeys(p.get("material",material) for p in options.get("candidates",[])))
        targets=tuple(item for item in targets if item!=material)
        study_name=str(report.get("study_name") or options.get("study_name") or "Print preset comparison")
    total=(summary.get("layer_count") or job.get("layer_count"))
    if total is None:
        maximum=metadata.get("source_header",{}).get("max_count")
        if isinstance(maximum,int):total=maximum+1
    created_raw=request.get("created_utc") or status.get("started_utc") or manifest.get("created_utc")
    fallback=_mtime(folder/"request.json") or _mtime(folder/"manifest.json") or _mtime(folder)
    created=_timestamp(created_raw,fallback)
    updated=_timestamp(status.get("finished_utc") or status.get("updated_utc"),max(_mtime(folder/"status.json"),_mtime(folder/"report.json"),_mtime(folder/"manifest.json"),created))
    message=status.get("message") or manifest.get("error") or ""
    if state=="failed" and report and not completed:
        messages=[str(item.get("error","")) for item in report.get("failures",[]) if isinstance(item,dict)]
        message=f"No simulation completed ({failures} failed cases)."+(" "+messages[0] if messages else "")
    if state=="cancelled" and not frame_count:
        message="Cancelled before any thermal frames were calculated."
    return RunInfo(folder,folder.name,str(name),job_hash,str(material),targets,str(kind),
        int(cfg.get("layer_start",0)),cfg.get("layer_end"),total,
        datetime.fromtimestamp(created,timezone.utc).isoformat(),created,updated,state,frame_count,int(completed),failures,
        message,str(status.get("phase",state)),float(status.get("progress",0.)),study_name)


def list_runs(root,job_sha256=None):
    values=[]
    root=Path(root)
    if root.exists():
        for folder in root.iterdir():
            if not folder.is_dir():continue
            info=read_run(folder)
            if info and (job_sha256 is None or info.job_sha256==job_sha256):values.append(info)
    return sorted(values,key=lambda run:(run.updated_timestamp,run.created_timestamp),reverse=True)


def playable_runs(runs,include_incomplete=False):
    return [run for run in runs if run.has_results and (include_incomplete or run.state in ("complete","partial","running"))]


def choose_result(runs,preferred=None,active=None):
    by_path={run.folder.resolve():run for run in runs}
    for value in (preferred,active):
        if value is not None:
            run=by_path.get(Path(value).resolve())
            if run and run.has_results:return run.folder
    finished=[run for run in runs if run.state in ("complete","partial") and run.has_results]
    return max(finished,key=lambda run:run.updated_timestamp).folder if finished else None


def latest_full_print(runs):
    """Return the newest run with evidence of a finished, untruncated source job.

    A full requested range alone is insufficient: a completed case must retain
    its final source-layer frame through the saved physical duration. Any
    time/vector cap is conservatively excluded, even if it could exceed the
    job size. Partly failed batches qualify only through a complete full case.
    Missing metadata, unfinished cases and missing final frames never qualify.
    The caller supplies the intended source-job scope (normally list_runs with
    a source hash); this function does not combine or infer different jobs.
    """
    candidates=[]
    for run in runs:
        total=run.total_layers
        end=run.layer_end if run.layer_end is not None else total
        if (run.state not in ("complete","partial") or not run.has_results
                or not isinstance(total,int) or total<1 or run.layer_start!=0
                or end is None or end<total):
            continue
        for folder,_ in list_cases(run):
            manifest=read_json(folder/"manifest.json")
            cfg=manifest.get("config") or {}
            case_end=cfg.get("layer_end")
            if (manifest.get("status")!="complete" or cfg.get("layer_start")!=0
                    or (case_end is not None and case_end<total)
                    or cfg.get("time_limit_s") is not None
                    or cfg.get("max_vectors") is not None):
                continue
            entries=manifest.get("frames") or []
            if not entries:continue
            last=entries[-1]
            path=_child(folder,last.get("file"))
            if path is None or not path.is_file():continue
            try:
                duration=float(manifest.get("physical_duration_s",manifest.get("metrics",{}).get("physical_duration_s")))
                final_time=float(last["time_s"])
                final_layer=int(last["layer_index"])
            except (KeyError,ValueError,TypeError,OverflowError):
                continue
            if (math.isfinite(duration) and duration>=0 and math.isfinite(final_time)
                    and final_time>=duration-max(1e-8,abs(duration)*1e-10)
                    and final_layer>=total-1):
                candidates.append(run)
                break
    return max(candidates,key=lambda run:(run.updated_timestamp,run.created_timestamp)).folder if candidates else None


def list_cases(run):
    folder=run.folder if isinstance(run,RunInfo) else Path(run)
    report=read_json(folder/"report.json");manifest=read_json(folder/"manifest.json")
    cases=[]
    own_count=_frames(folder,manifest)
    if own_count:
        cfg=manifest.get("config",{})
        cases.append((folder,f"{cfg.get('alloy','Current material')} · {own_count} saved frames · {manifest.get('status','unknown')}"))
    for case,row,meta,count in _case_data(folder,report):
        cfg=meta.get("config",{})
        material=row.get("material",cfg.get("alloy","Material"))
        p=row.get("power_scale",cfg.get("power_scale",1.));v=row.get("speed_scale",cfg.get("speed_scale",1.))
        preset_name=row.get("preset_name") or row.get("name")
        role="Reference" if row.get("kind")=="baseline" else "Preset"
        prefix=f"{role}: {preset_name} · " if preset_name else ""
        label=f"{prefix}{material} · P×{p:g} / v×{v:g} · {count} frames"
        if meta.get("status")!="complete":label+=f" · {meta.get('status','unknown')}"
        cases.append((case,label))
    return cases
