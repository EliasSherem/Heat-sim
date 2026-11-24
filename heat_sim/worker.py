"""Background calculation lifecycle, explicit cancellation and honest outcomes."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback

import numpy as np
import psutil

from .storage import atomic_json, snapshot_source


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError,TypeError):return None


class WorkerStatus:
    """Serialize progress and heartbeat writes to one small atomic status file."""
    def __init__(self,folder,kind,heartbeat_interval=2.):
        self.folder=Path(folder);self.path=self.folder/"status.json"
        self.lock=threading.RLock();self.stop=threading.Event()
        self.started=time.perf_counter();now=_utc();self.final=False
        self.report_stamp=None
        self.data=dict(schema=2,run_id=self.folder.name,kind=kind,pid=os.getpid(),
            process_started_epoch_s=psutil.Process().create_time(),state="running",phase="starting",progress=0.,
            started_utc=now,updated_utc=now,heartbeat_utc=now,last_progress_utc=now,
            elapsed_wall_s=0.,message="Starting calculation",completed_case_count=0,failed_case_count=0,
            frame_count=0,completed_frame_count=0,
            cancellation=dict(requested=False,acknowledged=False,marker_path=str(self.folder/"CANCEL"),requested_utc=None,metadata=None))
        self._write()
        self.thread=threading.Thread(target=self._heartbeat,args=(max(.01,heartbeat_interval),),daemon=True)
        self.thread.start()

    def _counts(self):
        count=0
        for folder in [self.folder/"frames",*self.folder.glob("cases/*/frames")]:
            if folder.is_dir():count+=sum(bool(re.fullmatch(r"\d{5,}\.npz",p.name)) for p in folder.iterdir())
        if self.data["kind"]=="render" and (self.folder/"frames.npz").is_file():
            manifest=_read_json(self.folder/"manifest.json") or {}
            count+=int(manifest.get("frame_count",0))
        self.data["frame_count"]=self.data["completed_frame_count"]=count
        report_path=self.folder/"report.json"
        try:stamp=report_path.stat().st_mtime_ns
        except OSError:stamp=None
        if stamp is not None and stamp!=self.report_stamp:
            report=_read_json(report_path)
            if isinstance(report,dict):
                self.report_stamp=stamp
                self.data["completed_case_count"]=sum(row.get("valid",True) is not False for row in report.get("rows",[]))
                self.data["failed_case_count"]=len(report.get("failures",[]))

    def _write(self):
        now=_utc();self.data.update(updated_utc=now,heartbeat_utc=now,elapsed_wall_s=time.perf_counter()-self.started)
        atomic_json(self.path,self.data)

    def _heartbeat(self,interval):
        while not self.stop.wait(interval):
            with self.lock:
                if self.final:return
                self._counts();self._write()

    def publish(self,fraction=None,message=None,phase=None,**fields):
        with self.lock:
            if self.final:return
            if fraction is not None:
                self.data["progress"]=float(np.clip(fraction,0.,1.));self.data["last_progress_utc"]=_utc()
            if message is not None:self.data["message"]=str(message)
            if phase is not None:self.data["phase"]=phase
            self.data.update(fields);self._write()

    def progress(self,fraction,message):
        lower=message.lower()
        phase="reconstructing_timing" if "reconstruct" in lower else "integrating" if ("layer" in lower or "thermal integration" in lower) else "comparing" if self.data["kind"] in ("transfer","presets") else self.data["phase"]
        self.publish(fraction,message,phase)

    def cancel_requested(self):
        marker=self.folder/"CANCEL"
        if not marker.is_file():return False
        try:stat=marker.stat();text=marker.read_text()[:16384]
        except OSError:return False
        try:metadata=json.loads(text) if text.strip() else None
        except ValueError:metadata=None
        if not isinstance(metadata,dict):metadata={"source":"legacy_cancel_marker","actor":"not recorded"}
        if metadata.get("run_id") not in (None,self.data["run_id"]):
            self.publish(cancellation_rejected_reason="Cancellation marker belongs to a different run")
            return False
        with self.lock:
            if not self.data["cancellation"]["requested"]:
                self.data["cancellation"].update(requested=True,
                    requested_utc=metadata.get("requested_utc") or datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),
                    requested_during_phase=self.data["phase"],metadata=metadata)
                if not self.final:self.data["phase"]="cancelling";self._write()
        return True

    def finish(self,state,message,**fields):
        with self.lock:
            self._counts();self.data.update(state=state,last_phase=self.data["phase"],phase="finished",message=message,finished_utc=_utc(),**fields)
            if state in ("complete","partial"):self.data["progress"]=1.
            if state=="cancelled":self.data["cancellation"]["acknowledged"]=True
            self.final=True;self._write()
            return copy.deepcopy(self.data)

    def close(self):
        self.stop.set();self.thread.join(timeout=3.)


def _outcome(kind,result):
    """Processing every plan differs from obtaining valid calculated cases."""
    if kind in ("transfer","presets") or isinstance(result,dict) and "rows" in result:
        rows=[row for row in result.get("rows",[]) if row.get("valid",True) is not False]
        good=len(rows);bad=len(result.get("failures",[]))
        if not good and bad:return "failed",f"All {bad} requested cases failed",good,bad
        if good and bad:return "partial",f"Completed {good} cases; {bad} cases failed",good,bad
        if kind in ("transfer","presets") and not good:return "failed","No valid simulation cases completed",0,bad
        return "complete",f"Completed {good} calculated cases",good,bad
    if kind=="build":
        if result.get("status")!="complete":return "failed","Build did not complete",0,1
        return "complete","Build completed",1,0
    return "complete","Calculation completed",1,0


def build_preflight(job,config):
    """Shared planning implementation also used by the interface."""
    from .run_planning import workload_plan
    return workload_plan(job,config)


def execute_request(request_path,heartbeat_interval=2.):
    """Run one immutable request; return terminal status, never false success."""
    request_path=Path(request_path).resolve();folder=request_path.parent
    request=_read_json(request_path)
    kind=request.get("kind","unknown") if isinstance(request,dict) else "unknown"
    status=WorkerStatus(folder,kind,heartbeat_interval);result=None
    try:
        if not isinstance(request,dict) or not isinstance(request.get("kind"),str):
            raise ValueError("The saved request is missing or invalid; a typed calculation kind is required")
        snapshot_source(folder)
        from .solver import SimulationCancelled
        if status.cancel_requested():raise SimulationCancelled("Cancellation requested before calculation started")
        status.publish(0,"Loading numerical model (first use may compile CPU kernels)","loading_model")
        if kind in ("build","transfer","presets"):
            from .job_import import load_job
            from .build_solver import BuildConfig,run_build
            job_path=Path(request["job_path"])
            if not job_path.is_absolute():job_path=folder/job_path
            status.publish(0,"Loading imported job","loading_job")
            job=load_job(job_path,progress=lambda f,m:status.publish(.02*f,m,"loading_job"))
            values=dict(request["build_config"])
            if values.get("geometry_file") and not Path(values["geometry_file"]).is_absolute():values["geometry_file"]=str((folder/values["geometry_file"]).resolve())
            config=BuildConfig(**values)
            status.publish(message="Checking grid, timing and step limits",phase="preflight")
            planning_config=config
            if kind=="transfer" and request.get("options",{}).get("evaluate_candidates",False):
                from dataclasses import replace
                # The baseline/identical settings are always speed×1. Candidate
                # slowdown can exceed the cap even when that baseline fits.
                speeds=request.get("options",{}).get("speed_scales",[.9,1.,1.1])
                planning_config=replace(config,speed_scale=min([1.,*(float(s) for s in speeds)]))
            if kind=="presets":
                from .presets import preset_config,preset_from_config
                options=request.get("options",{})
                cases=[options.get("reference") or preset_from_config(config,"Reference"),*options.get("candidates",[])]
                checks=[]
                for index,preset in enumerate(cases):
                    if status.cancel_requested():raise SimulationCancelled("Cancelled while checking print presets")
                    case_config=preset_config(config,preset)
                    check=build_preflight(job,case_config)
                    check.update(preset_name=preset.get("name") or f"Preset {index+1}",planning_speed_scale=case_config.speed_scale)
                    checks.append(check)
                worst=max(checks,key=lambda item:item.get("minimum_thermal_steps",0))
                preflight={**worst,"planning_scope":"every requested print preset","cases":checks}
                failed=next((item for item in checks if item.get("step_limit_feasible",item.get("feasible",not item.get("step_limit_exceeded",False))) is False),None)
                if failed is not None:
                    preflight.update(step_limit_feasible=False,failure_reason=f"{failed['preset_name']}: "+(failed.get("failure_reason") or failed.get("reason") or "Preset exceeds the thermal-step limit"))
            else:
                preflight=build_preflight(job,planning_config)
                preflight["planning_speed_scale"]=planning_config.speed_scale
                preflight["planning_scope"]="slowest requested transfer case" if planning_config is not config else "requested configuration"
            status.publish(preflight=preflight)
            # The shared planner supplies a guaranteed lower bound. Never
            # reject a time-limited prefix using an untrimmed full-job count.
            if preflight.get("step_limit_feasible",preflight.get("feasible",not preflight.get("step_limit_exceeded",False))) is False:
                raise ValueError(preflight.get("failure_reason") or preflight.get("reason") or "Selected workload exceeds the thermal-step limit; choose the coarse full-build preset or a shorter range")
            if kind=="build":result=run_build(job,config,folder,progress=status.progress,cancel=status.cancel_requested)
            elif kind=="transfer":
                from .transfer import run_transfer
                result=run_transfer(job,config,request.get("options",{}),folder,progress=status.progress,cancel=status.cancel_requested)
            else:
                from .presets import run_presets
                result=run_presets(job,config,request.get("options",{}),folder,progress=status.progress,cancel=status.cancel_requested)
        elif kind=="render":
            from dataclasses import replace
            from .solver import SimulationConfig
            from .render import render_simulation
            alloy=request.get("options",{}).get("render_alloy","17-4PH")
            alpha=request.get("options",{}).get("absorptivity_316" if alloy=="316L" else "absorptivity_17",.35)
            config=replace(SimulationConfig(**request["config"]),alloy=alloy,absorptivity=alpha)
            manifest=render_simulation(config,folder,progress=status.progress,cancel=status.cancel_requested)
            row=dict(label=f"{alloy} instantaneous playback",scenario="nominal",alloy=alloy,power_w=config.power_w,
                     speed_mm_s=config.speed_m_s*1000,cache_key=manifest["cache_key"],**manifest["metrics"])
            result=dict(kind="render",rows=[row],extra=manifest,fingerprint=manifest["model_fingerprint"])
            atomic_json(folder/"report.json",result)
        elif kind=="verification":
            from .verification import run_verification
            from .storage import RESULTS
            verification=run_verification(include_sensitivity=True,progress=status.progress,cancel=status.cancel_requested)
            atomic_json(RESULTS/"verification.json",verification)
            result=dict(kind="verification",rows=[],extra=verification)
            atomic_json(folder/"report.json",result)
            if not verification.get("numerical_checks_passed",False):raise ValueError("Numerical verification failed; inspect the saved report")
        else:
            from .workflows import run_workflow
            result=run_workflow(request,folder,status.progress,status.cancel_requested)
        # A request arriving after the final solver poll is recorded without
        # relabelling a successfully finished calculation as cancelled.
        status.cancel_requested()
        state,message,good,bad=_outcome(kind,result)
        if kind in ("transfer","presets"):
            result.update(status=state,processing_complete=True,completed_case_count=good,failed_case_count=bad)
            atomic_json(folder/"report.json",result)
        return status.finish(state,message,completed_case_count=good,failed_case_count=bad)
    except BaseException as exc:
        cancelled=exc.__class__.__name__=="SimulationCancelled" and status.cancel_requested()
        state="cancelled" if cancelled else "failed";message=str(exc)
        if exc.__class__.__name__=="SimulationCancelled" and not cancelled:message=f"Unexpected cancellation without a request for this run: {exc}"
        if not cancelled:(folder/"traceback.txt").write_text(traceback.format_exc())
        with status.lock:
            status._counts();frames=status.data["frame_count"];cases=status.data["completed_case_count"]
            if cancelled:
                suffix=f"{frames} saved frames and {cases} completed cases remain available" if frames or cases else "No completed frames or cases are available"
                message=f"{message}. {suffix}."
            if kind=="build" and state=="failed":status.data["failed_case_count"]=1
        return status.finish(state,message,exception=type(exc).__name__)
    finally:status.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("request",type=Path)
    terminal=execute_request(parser.parse_args(argv).request)
    return 1 if terminal["state"]=="failed" else 0


if __name__=="__main__":raise SystemExit(main())
