from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from heat_sim.build_solver import BuildConfig
from heat_sim.job_import import ImportedJob,ImportedLayer
from heat_sim.materials import MATERIALS
from heat_sim.solver import SimulationCancelled,SimulationError
from heat_sim.transfer import run_transfer,enthalpy_fraction,thermal_similarity_score,SCORE_FEATURES


def fixture():
    vectors=np.array([[0.,0.,.001,0.,120.,1.4],
                      [.001,0.,0.,.00006,100.,1.]])
    layer=ImportedLayer(0,0.,vectors,np.array([0,6]),np.array([0,0]),np.array([0,1]))
    job=ImportedJob("transfer-fixture","none","transfer-fixture-hash","test",60e-6,[layer],{})
    cfg=BuildConfig(absorptivity=.35,layer_end=1,final_cooldown_s=.002,
        spacing_m=.0005,z_spacing_m=.00012,plate_depth_m=.00024,padding_m=.0005,
        max_event_s=.001,frame_interval_s=.0005,max_frames=12)
    return job,cfg


@pytest.mark.parametrize("alloy",MATERIALS)
def test_enthalpy_screen_is_zero_at_initial_one_at_liquidus(alloy):
    m=MATERIALS[alloy]
    actual=enthalpy_fraction([353.15,m.liquidus],alloy,353.15)
    np.testing.assert_allclose(actual,[0.,1.],atol=1e-14)


def test_thermal_score_has_known_dimensionless_relative_solution():
    baseline=dict(zip(SCORE_FEATURES,[.1,.3,.5]))
    assert thermal_similarity_score(baseline,baseline)==0.
    candidate={key:2*value for key,value in baseline.items()}
    assert thermal_similarity_score(candidate,baseline)==pytest.approx(1.)


def test_real_transfer_preserves_paths_and_section_ratios_and_caches(tmp_path,monkeypatch):
    job,cfg=fixture()
    # The numerical workflow must not overwrite the worker's replayable request.
    worker_request={"kind":"transfer","job_path":"job/job-summary.json","build_config":{"recoat_time_s":3.}}
    (tmp_path/"request.json").write_text(json.dumps(worker_request))
    options=dict(baseline_material="17-4PH",target_materials=["316L"],
        absorptivities={"17-4PH":.35,"316L":.27},evaluate_candidates=True,
        power_scales=[1.1],speed_scales=[.9])
    report=run_transfer(job,cfg,options,tmp_path)
    assert json.loads((tmp_path/"request.json").read_text())==worker_request
    assert (tmp_path/"transfer-request.json").exists()
    assert report["status"]=="complete" and not report["failures"]
    assert len(report["rows"])==3
    baseline,identical,candidate=report["rows"]
    assert baseline["power_scale"]==baseline["speed_scale"]==1.
    assert identical["absorptivity"]==candidate["absorptivity"]==.27
    assert candidate["valid"] and candidate["comparable"] and candidate["score"] is not None
    with np.load(tmp_path/baseline["case_folder"]/"timeline.npz") as saved:
        b=saved["events"];b=b[b["kind"]==1]
    with np.load(tmp_path/candidate["case_folder"]/"timeline.npz") as saved:
        c=saved["events"];c=c[c["kind"]==1]
    np.testing.assert_array_equal(b["start_m"],c["start_m"])
    np.testing.assert_array_equal(b["end_m"],c["end_m"])
    np.testing.assert_allclose(c["power_w"],b["power_w"]*1.1)
    np.testing.assert_allclose(c["speed_m_s"],b["speed_m_s"]*.9)
    assert (tmp_path/"comparison.csv").exists()
    assert json.loads((tmp_path/"report.json").read_text())["experimental_validation"] is False
    def must_not_run(*args,**kwargs):raise AssertionError("Completed case should have been cached")
    monkeypatch.setattr("heat_sim.transfer.run_build",must_not_run)
    cached=run_transfer(job,cfg,options,tmp_path)
    assert not cached["failures"] and cached["cache_hits"]==3
    # A copied/exported study must resolve its own artifacts after relocation,
    # including ranked cases and probe CSVs, without invoking the solver again.
    moved=tmp_path.parent/(tmp_path.name+"-copy")
    shutil.copytree(tmp_path,moved)
    moved_report=json.loads((moved/"report.json").read_text())
    for row in moved_report["rows"]:
        assert not Path(row["case_folder"]).is_absolute()
        assert not Path(row["probe_csv"]).is_absolute()
        assert (moved/row["case_folder"]/"manifest.json").exists()
        assert (moved/row["probe_csv"]).exists()
    for ranking in moved_report["rankings"]:
        assert not Path(ranking["closest_case_folder"]).is_absolute()
        assert (moved/ranking["closest_case_folder"]/"manifest.json").exists()
        assert all(not Path(row["case_folder"]).is_absolute() for row in ranking["candidates"])
    moved_cached=run_transfer(job,cfg,options,moved)
    assert not moved_cached["failures"] and moved_cached["cache_hits"]==3


def test_different_exposed_prefixes_are_not_ranked(tmp_path):
    job,cfg=fixture();cfg=replace(cfg,time_limit_s=.0004)
    options=dict(target_materials=["316L"],evaluate_candidates=True,
                 power_scales=[1.],speed_scales=[.5])
    report=run_transfer(job,cfg,options,tmp_path)
    candidate=next(row for row in report["rows"] if row["kind"]=="candidate")
    assert candidate["valid"] and not candidate["comparable"]
    assert candidate["score"] is None
    assert report["rankings"][0]["ranked_candidates_count"]==1


def test_failed_builds_are_recorded_and_excluded_from_rankings(tmp_path,monkeypatch):
    from heat_sim import transfer
    job,cfg=fixture();real=transfer.run_build
    def sometimes_fail(job,config,*args,**kwargs):
        if config.speed_scale==1.1:raise SimulationError("deliberate integration failure")
        return real(job,config,*args,**kwargs)
    monkeypatch.setattr(transfer,"run_build",sometimes_fail)
    report=run_transfer(job,cfg,dict(target_materials=["AlSi10Mg"],evaluate_candidates=True,
        power_scales=[1.],speed_scales=[1.1]),tmp_path)
    assert len(report["failures"])==1
    assert not report["failures"][0]["valid"]
    assert not Path(report["failures"][0]["case_folder"]).is_absolute()
    assert len(report["rows"])==2
    assert report["rankings"][0]["ranked_candidates_count"]==1


def test_uncertainty_replays_independent_target_absorptivity_endpoints(tmp_path):
    job,cfg=fixture()
    report=run_transfer(job,cfg,dict(target_materials=["316L"],evaluate_candidates=True,
        power_scales=[1.],speed_scales=[.9],evaluate_uncertainty=True,
        absorptivities={"17-4PH":.35,"316L":.3},
        uncertainty_absorptivity_ranges={"316L":[.2,.4]}),tmp_path)
    assert not report["failures"]
    assert len(report["rows"])==7
    scenarios=report["uncertainty"]["316L"]["scenarios"]
    assert [s["absorptivity"] for s in scenarios]==[.2,.3,.4]
    assert all(s["complete"] for s in scenarios)
    assert report["uncertainty"]["316L"]["baseline_absorptivity_fixed"]==.35
    # Real computed thermal responses change the closest speed as absorption
    # moves across the supplied range; the report must expose that sensitivity.
    assert scenarios[0]["closest"]!=scenarios[-1]["closest"]
    assert report["uncertainty"]["316L"]["closest_consistent"] is False
    assert report["uncertainty"]["316L"]["ranking_evaluable"] is True


def test_cancellation_keeps_report_and_never_invents_partial_scores(tmp_path):
    job,cfg=fixture()
    with pytest.raises(SimulationCancelled):
        run_transfer(job,cfg,dict(target_materials=["316L"]),tmp_path,cancel=lambda:True)
    report=json.loads((tmp_path/"report.json").read_text())
    assert report["status"]=="cancelled" and report["rows"]==[]
