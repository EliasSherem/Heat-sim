"""Planner follows actual imported paths without constructing their event table."""
from dataclasses import replace

import numpy as np
import pytest

from heat_sim.build_solver import BuildConfig,build_timeline
from heat_sim.job_import import ImportedJob,ImportedLayer
from heat_sim.run_planning import estimate_run_plan,workload_plan


def fixture_job():
    layers=[]
    for index in range(2):
        x=index*.002
        vectors=np.array([[x,0.,x+.001,0.,120.,.01],
                          [x+.001,0.,x+.002,0.,100.,.01]])
        layers.append(ImportedLayer(index,index*60e-6,vectors,np.array([0,6]),
                                   np.array([0,0]),np.array([0,1])))
    return ImportedJob("planner-fixture","none","planner-hash","test",60e-6,layers,{})


def config(**changes):
    return replace(BuildConfig(absorptivity=.35,layer_end=2,jump_speed_m_s=5.,
        jump_delay_s=.02,recoat_time_s=.6,final_cooldown_s=.3,
        max_event_s=.05,max_steps=100_000),**changes)


def assert_matches_timeline(job,cfg):
    plan=estimate_run_plan(job,cfg)
    gold=build_timeline(job,cfg)
    durations=gold["t1_s"]-gold["t0_s"]
    for key,kind in (("scan_time_s",1),("jump_time_s",2),("interlayer_time_s",3),("cooldown_time_s",4)):
        assert plan[key]==pytest.approx(float(durations[gold["kind"]==kind].sum()),abs=1e-11)
    assert plan["estimated_physical_time_s"]==pytest.approx(float(gold[-1]["t1_s"]),abs=1e-11)
    assert plan["vector_count"]==int(np.count_nonzero(gold["kind"]==1))
    return plan


def test_known_scan_jump_recoat_and_grouping_lower_bound():
    plan=assert_matches_timeline(fixture_job(),config())
    assert plan["scan_time_s"]==pytest.approx(.4)
    assert plan["jump_time_s"]==pytest.approx(.06)
    assert plan["interlayer_time_s"]==pytest.approx(.6)
    assert plan["estimated_physical_time_s"]==pytest.approx(1.36)
    # Two flushed exposure blocks .22/.24s:5+5; recoat .6/.25:3;
    # cooldown .3/.25:2. Frames/nonlinear retries can add steps.
    assert plan["minimum_thermal_steps"]==15
    assert plan["feasible"] and not plan["step_limit_exceeded"]
    assert plan["numerical_convergence_guaranteed"] is False


@pytest.mark.parametrize("limit",[.025,.1,.11,.22,.3,.82,.835,1.1,2.])
def test_time_cutoffs_include_partial_scan_jump_recoat_and_cooldown(limit):
    plan=assert_matches_timeline(fixture_job(),config(time_limit_s=limit))
    assert plan["estimated_physical_time_s"]<=limit+1e-12
    assert plan["truncated_estimate"]==(limit<1.36)


@pytest.mark.parametrize("cap",[1,2,3,4,9])
def test_vector_limit_matches_existing_timeline_boundary_convention(cap):
    plan=assert_matches_timeline(fixture_job(),config(max_vectors=cap))
    assert plan["truncated_estimate"]==(cap<4)
    assert plan["vector_count"]==min(cap,4)


def test_speed_scaling_and_continuous_paths_change_timing_correctly():
    job=fixture_job()
    job.layers[0].path_ids[:]=0
    plan=assert_matches_timeline(job,config(speed_scale=2.))
    assert plan["scan_time_s"]==pytest.approx(.2)
    assert plan["jump_time_s"]==pytest.approx(.04)


def test_cross_layer_jump_uses_xy_not_platform_z_distance():
    job=fixture_job()
    # Keep adjacent XY ends/starts identical while increasing layer height.
    job.layers[1].z_m=.01
    plan=assert_matches_timeline(job,config())
    assert plan["jump_time_s"]==pytest.approx(.06)


def test_empty_layers_and_nonzero_start_ranges_match_timeline():
    job=fixture_job()
    empty=ImportedLayer(1,60e-6,np.empty((0,6)),np.array([],int),np.array([],int),np.array([],int))
    job.layers[1].index=2;job.layers[1].z_m=120e-6
    job.layers.insert(1,empty)
    assert_matches_timeline(job,config(layer_end=3))
    assert_matches_timeline(job,config(layer_start=1,layer_end=3,warm_start_assumed=True))


def test_step_guard_and_millisecond_suggestion_leave_planning_margin():
    job=fixture_job();cfg=config(max_steps=10)
    plan=estimate_run_plan(job,cfg)
    assert plan["step_limit_exceeded"] and not plan["step_limit_feasible"]
    assert plan["minimum_thermal_steps"]==plan["min_steps"]==15
    assert "exceed" in plan["reason"] and plan["reason"]==plan["failure_reason"]
    suggestion=plan["suggested_max_event_s"]
    assert suggestion>=cfg.max_event_s
    assert suggestion*1000==pytest.approx(round(suggestion*1000))
    revised=estimate_run_plan(job,replace(cfg,max_event_s=suggestion))
    assert revised["minimum_thermal_steps"]<=.75*cfg.max_steps
    impossible=estimate_run_plan(job,config(max_steps=1))
    assert impossible["step_limit_exceeded"] and impossible["suggested_max_event_s"] is None


def test_planner_never_calls_the_global_timeline_allocator(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("Preflight allocated the full timeline")
    monkeypatch.setattr("heat_sim.build_solver.build_timeline",forbidden)
    job=fixture_job();cfg=config()
    assert estimate_run_plan(job,cfg)["selected_vector_count"]==4
    assert workload_plan(job,cfg)["minimum_thermal_steps"]==15


@pytest.mark.parametrize("budget",[0,-1,1.5])
def test_invalid_step_budget_rejected(budget):
    with pytest.raises(ValueError,match="max_steps"):
        estimate_run_plan(fixture_job(),config(max_steps=budget))


def test_section_recipe_overrides_change_only_selected_exposure_and_plan():
    job=fixture_job()
    original=[layer.vectors.copy() for layer in job.layers]
    cfg=config(power_scale=1.1,speed_scale=2.,power_factor=.8,
               section_overrides={"6":{"power_w":180.,"speed_m_s":.005}})
    plan=assert_matches_timeline(job,cfg)
    events=build_timeline(job,cfg);scans=events[events["kind"]==1]
    np.testing.assert_allclose(scans["power_w"],[120.*1.1*.8,180.*1.1*.8]*2)
    np.testing.assert_allclose(scans["speed_m_s"],[.02,.01]*2)
    assert plan["scan_time_s"]==pytest.approx(.3)
    for layer,copy in zip(job.layers,original):np.testing.assert_array_equal(layer.vectors,copy)
    clipped=assert_matches_timeline(job,replace(cfg,time_limit_s=.06))
    assert clipped["time_limit_reached"]


@pytest.mark.parametrize("overrides",[{"-1":{"power_w":100}}, {"6":{"speed_m_s":0}},
    {"6":{"power_w":float("nan")}}, {"6":{"hatch_m":.001}}, {"6":{"power_w":True}}])
def test_invalid_section_recipe_overrides_rejected(overrides):
    with pytest.raises(ValueError,match="Region"):
        estimate_run_plan(fixture_job(),config(section_overrides=overrides))
