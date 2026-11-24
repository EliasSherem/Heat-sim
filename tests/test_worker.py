"""Worker lifecycle integration: outcomes, explicit cancellation and heartbeat."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pytest

from heat_sim import worker
from heat_sim.build_solver import BuildConfig
from heat_sim.job_import import ImportedJob,ImportedLayer,save_imported_job
from heat_sim.solver import SimulationCancelled,SimulationError
from heat_sim.storage import atomic_json


def request(tmp_path,kind="build"):
    vectors=np.array([[0.,0.,.001,0.,120.,1.4],[.001,0.,0.,.00006,100.,1.]])
    layer=ImportedLayer(0,0.,vectors,np.array([0,6]),np.array([0,0]),np.array([0,1]))
    job=ImportedJob("worker-fixture","none","worker-fixture-hash","test",60e-6,[layer],{})
    save_imported_job(job,tmp_path/"job",copy_source=False)
    config=BuildConfig(absorptivity=.35,layer_end=1,final_cooldown_s=.002,
        spacing_m=.0005,z_spacing_m=.00012,plate_depth_m=.00024,padding_m=.0005,
        max_event_s=.001,frame_interval_s=.0005,max_frames=12)
    path=tmp_path/"request.json"
    atomic_json(path,dict(kind=kind,job_path="job/job-summary.json",build_config=asdict(config),options=dict(target_materials=["316L"])))
    return path


def test_real_build_records_complete_lifecycle_and_saved_frame_count(tmp_path):
    terminal=worker.execute_request(request(tmp_path),heartbeat_interval=.01)
    assert terminal["state"]=="complete"
    assert terminal["run_id"]==tmp_path.name and terminal["pid"]>0
    assert terminal["phase"]=="finished" and terminal["progress"]==1.
    assert terminal["completed_case_count"]==1 and terminal["failed_case_count"]==0
    manifest=json.loads((tmp_path/"manifest.json").read_text())
    assert terminal["frame_count"]==len(manifest["frames"])>1
    assert datetime.fromisoformat(terminal["finished_utc"])>=datetime.fromisoformat(terminal["started_utc"])
    assert terminal["elapsed_wall_s"]>0
    assert terminal["cancellation"]["requested"] is False


def test_all_failed_transfer_is_failed_not_complete_and_cli_exits_nonzero(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise SimulationError("deliberate integration failure")
    monkeypatch.setattr("heat_sim.transfer.run_build",fail)
    terminal=worker.execute_request(request(tmp_path,"transfer"))
    assert terminal["state"]=="failed"
    assert terminal["completed_case_count"]==0 and terminal["failed_case_count"]==2
    assert terminal["frame_count"]==0
    report=json.loads((tmp_path/"report.json").read_text())
    assert report["status"]=="failed" and report["processing_complete"] is True
    assert "All 2" in terminal["message"]
    monkeypatch.setattr(worker,"execute_request",lambda path:terminal)
    assert worker.main([str(tmp_path/"request.json")])==1


def test_mixed_transfer_outcome_is_partial_with_computed_case_available(tmp_path,monkeypatch):
    from heat_sim import transfer
    real=transfer.run_build
    def mixed(job,config,*args,**kwargs):
        if config.alloy=="316L":raise SimulationError("target failed")
        return real(job,config,*args,**kwargs)
    monkeypatch.setattr(transfer,"run_build",mixed)
    terminal=worker.execute_request(request(tmp_path,"transfer"))
    assert terminal["state"]=="partial"
    assert terminal["completed_case_count"]==terminal["failed_case_count"]==1
    assert terminal["frame_count"]>0
    assert json.loads((tmp_path/"report.json").read_text())["status"]=="partial"


def test_legacy_explicit_cancel_before_start_does_not_claim_frames_exist(tmp_path):
    path=request(tmp_path);(tmp_path/"CANCEL").touch()
    terminal=worker.execute_request(path)
    assert terminal["state"]=="cancelled"
    assert terminal["frame_count"]==terminal["completed_case_count"]==0
    assert "No completed frames" in terminal["message"]
    assert terminal["cancellation"]["acknowledged"] is True
    assert terminal["cancellation"]["metadata"]["actor"]=="not recorded"
    assert terminal["cancellation"]["requested_utc"]


def test_explicit_cancel_preserves_progress_and_only_complete_frame_files(tmp_path,monkeypatch):
    def cancel_midway(job,config,folder,progress,cancel):
        frames=Path(folder)/"frames";frames.mkdir()
        np.savez(frames/"00000.npz",temperature_k=np.array([400.]))
        np.savez(frames/"00001.tmp.npz",temperature_k=np.array([500.]))
        progress(.37,"Layer0 thermal integration")
        atomic_json(Path(folder)/"CANCEL",dict(run_id=Path(folder).name,requested_by="cancel_button",requested_utc=worker._utc()))
        assert cancel()
        raise SimulationCancelled("Requested during integration")
    monkeypatch.setattr("heat_sim.build_solver.run_build",cancel_midway)
    terminal=worker.execute_request(request(tmp_path))
    assert terminal["state"]=="cancelled" and terminal["progress"]==.37
    assert terminal["frame_count"]==1
    assert terminal["cancellation"]["metadata"]["requested_by"]=="cancel_button"
    assert "1 saved frames" in terminal["message"]


def test_unrequested_cancellation_exception_is_a_failure(tmp_path,monkeypatch):
    def unexpected(*args,**kwargs):raise SimulationCancelled("no marker was supplied")
    monkeypatch.setattr("heat_sim.build_solver.run_build",unexpected)
    terminal=worker.execute_request(request(tmp_path))
    assert terminal["state"]=="failed"
    assert "without a request" in terminal["message"]
    assert terminal["cancellation"]["requested"] is False


def test_cancel_marker_for_another_run_cannot_cancel_this_run(tmp_path):
    path=request(tmp_path)
    atomic_json(tmp_path/"CANCEL",dict(run_id="different-study",requested_by="cancel_button"))
    terminal=worker.execute_request(path)
    assert terminal["state"]=="complete"
    assert terminal["cancellation"]["requested"] is False
    assert "different run" in terminal["cancellation_rejected_reason"]


def test_heartbeat_changes_while_solver_progress_is_unchanged(tmp_path,monkeypatch):
    observations=[]
    def stalled(job,config,folder,progress,cancel):
        progress(.2,"Thermal integration")
        observations.append(json.loads((Path(folder)/"status.json").read_text()))
        time.sleep(.06)
        observations.append(json.loads((Path(folder)/"status.json").read_text()))
        return dict(status="complete")
    monkeypatch.setattr("heat_sim.build_solver.run_build",stalled)
    terminal=worker.execute_request(request(tmp_path),heartbeat_interval=.01)
    first,second=observations
    assert second["heartbeat_utc"]>first["heartbeat_utc"]
    assert second["last_progress_utc"]==first["last_progress_utc"]
    assert first["progress"]==second["progress"]==.2
    assert terminal["state"]=="complete"


def test_preflight_blocks_impossible_work_before_starting_solver(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,"build_preflight",lambda *a:dict(step_limit_feasible=False,failure_reason="At least108471steps exceeds100000limit"))
    def must_not_run(*args,**kwargs):raise AssertionError("Impossible workload reached solver")
    monkeypatch.setattr("heat_sim.build_solver.run_build",must_not_run)
    terminal=worker.execute_request(request(tmp_path))
    assert terminal["state"]=="failed"
    assert terminal["frame_count"]==0
    assert "108471" in terminal["message"]
    assert terminal["preflight"]["step_limit_feasible"] is False


def test_numerical_failure_preserves_real_progress_and_available_frames(tmp_path,monkeypatch):
    def fail_midway(job,config,folder,progress,cancel):
        frames=Path(folder)/"frames";frames.mkdir()
        np.savez(frames/"00000.npz",temperature_k=np.array([400.]))
        progress(.61,"Layer5 thermal integration")
        raise SimulationError("Build step limit reached")
    monkeypatch.setattr("heat_sim.build_solver.run_build",fail_midway)
    terminal=worker.execute_request(request(tmp_path))
    assert terminal["state"]=="failed" and terminal["progress"]==.61
    assert terminal["frame_count"]==1 and terminal["failed_case_count"]==1
    assert terminal["last_phase"]=="integrating"
    assert terminal["cancellation"]["requested"] is False


def test_malformed_saved_request_still_records_failed_worker_status(tmp_path):
    path=tmp_path/"request.json";path.write_text('{"options":{}}')
    terminal=worker.execute_request(path)
    assert terminal["state"]=="failed"
    assert terminal["run_id"]==tmp_path.name and terminal["kind"]=="unknown"
    assert "typed calculation kind" in terminal["message"]
    assert json.loads((tmp_path/"status.json").read_text())["state"]=="failed"


def test_transfer_preflight_checks_slowest_actual_candidate_without_changing_base(tmp_path,monkeypatch):
    path=request(tmp_path,"transfer")
    values=json.loads(path.read_text())
    values["options"].update(evaluate_candidates=True,speed_scales=[.9,1.1])
    atomic_json(path,values)
    captured=[]
    def candidate_plan(job,config):
        captured.append(config.speed_scale)
        return dict(step_limit_feasible=config.speed_scale>=1.,failure_reason="Slow candidate exceeds step limit")
    monkeypatch.setattr(worker,"build_preflight",candidate_plan)
    def must_not_run(*args,**kwargs):raise AssertionError("Unsafe candidate batch reached solver")
    monkeypatch.setattr("heat_sim.transfer.run_transfer",must_not_run)
    terminal=worker.execute_request(path)
    assert captured==[.9] and terminal["state"]=="failed"
    assert terminal["preflight"]["planning_speed_scale"]==.9
    assert json.loads(path.read_text())["build_config"]["speed_scale"]==1.


def test_unused_candidate_speeds_do_not_affect_transfer_preflight(tmp_path,monkeypatch):
    path=request(tmp_path,"transfer");values=json.loads(path.read_text())
    values["options"].update(evaluate_candidates=False,speed_scales=[.1])
    atomic_json(path,values);planned=[];actual=[]
    monkeypatch.setattr(worker,"build_preflight",lambda job,cfg:planned.append(cfg.speed_scale) or dict(step_limit_feasible=True))
    def completed(job,config,options,folder,**kwargs):
        actual.append(config.speed_scale)
        return dict(status="complete",rows=[dict(valid=True)],failures=[])
    monkeypatch.setattr("heat_sim.transfer.run_transfer",completed)
    terminal=worker.execute_request(path)
    assert terminal["state"]=="complete" and planned==actual==[1.]


def preset_request(tmp_path):
    path=request(tmp_path,"presets")
    values=json.loads(path.read_text())
    values["options"]=dict(study_name="Tune contour heat",reference=dict(name="Successful print"),
        candidates=[dict(name="Lower power",power_scale=.9)])
    atomic_json(path,values)
    return path


def test_real_preset_study_saves_names_and_completed_case_counts(tmp_path):
    terminal=worker.execute_request(preset_request(tmp_path))
    assert terminal["state"]=="complete"
    assert terminal["completed_case_count"]==2 and terminal["failed_case_count"]==0
    assert terminal["frame_count"]>2
    report=json.loads((tmp_path/"report.json").read_text())
    assert report["status"]=="complete" and report["processing_complete"]
    assert [row["preset_name"] for row in report["rows"]]==["Successful print","Lower power"]
    assert report["rows"][1]["power_scale"]==.9
    assert (tmp_path/"source/heat_sim/presets.py").exists()


def test_preset_preflight_checks_every_actual_configuration(tmp_path,monkeypatch):
    path=preset_request(tmp_path);values=json.loads(path.read_text())
    values["options"]["candidates"][0].update(speed_scale=.8,recoat_time_s=4.)
    atomic_json(path,values)
    checked=[]
    def plan(job,config):
        checked.append((config.speed_scale,config.recoat_time_s))
        return dict(step_limit_feasible=config.recoat_time_s<4.,failure_reason="Long recoat exceeds budget")
    monkeypatch.setattr(worker,"build_preflight",plan)
    def must_not_run(*args,**kwargs):raise AssertionError("Infeasible batch reached solver")
    monkeypatch.setattr("heat_sim.presets.run_presets",must_not_run)
    terminal=worker.execute_request(path)
    assert terminal["state"]=="failed"
    assert checked==[(1.,3.),(.8,4.)]
    assert terminal["preflight"]["planning_scope"]=="every requested print preset"
    assert len(terminal["preflight"]["cases"])==2
    assert "Lower power" in terminal["message"]
    assert json.loads(path.read_text())["build_config"]["speed_scale"]==1.


def test_all_failed_preset_study_is_failed(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise SimulationError("Preset integration failed")
    monkeypatch.setattr("heat_sim.presets.run_build",fail)
    terminal=worker.execute_request(preset_request(tmp_path))
    assert terminal["state"]=="failed" and terminal["completed_case_count"]==0
    assert terminal["failed_case_count"]==2 and terminal["frame_count"]==0
    assert json.loads((tmp_path/"report.json").read_text())["status"]=="failed"
