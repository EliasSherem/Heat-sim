"""The single workspace must never mix current inputs with old solver fields."""
from dataclasses import asdict, replace
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil
import pytest

from heat_sim import workspace
from heat_sim.build_solver import BuildConfig, run_build
from heat_sim.job_import import ImportedJob, ImportedLayer, save_imported_job
from heat_sim.presets import preset_from_config
from heat_sim.storage import atomic_json


def fixture(tmp_path):
    vectors = np.array([[0., 0., .001, 0., 120., 1.4],
                        [.001, 0., 0., .00006, 100., 1.]])
    layer = ImportedLayer(0, 0., vectors, np.array([0, 6]), np.array([0, 0]), np.array([0, 1]))
    job = ImportedJob("workspace-part.mjob", "none", "workspace-fixture-hash", "test", 60e-6, [layer], {})
    summary = save_imported_job(job, tmp_path/"import", copy_source=False)
    cfg = BuildConfig(absorptivity=.35, layer_end=1, final_cooldown_s=.002,
        spacing_m=.0005, z_spacing_m=.00012, plate_depth_m=.00024, padding_m=.0005,
        max_event_s=.001, frame_interval_s=.0005, max_frames=12)
    return job, summary, cfg


def wait_complete(folder):
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        status = workspace.read_status(folder)
        if status.get("state") not in workspace.ACTIVE_STATES:
            return status
        time.sleep(.025)
    raise AssertionError(f"Worker did not finish: {status}\n{(folder/'worker.log').read_text()}")


def test_compact_defaults_preserve_region_recipes_and_cached_import(tmp_path):
    job, summary, _ = fixture(tmp_path)
    loaded = workspace.cached_job(summary)
    assert workspace.cached_job(summary) is loaded
    compact = workspace.process_summary(loaded)
    assert compact["layer_count"] == 1
    assert compact["dimensions_mm"] == pytest.approx([1., .06, .06])
    assert compact["power_w"] == [100., 120.]
    assert compact["speed_mm_s"] == [1000., 1400.]
    cfg = workspace.default_config(job, "Ti-6Al-4V")
    assert cfg.layer_end == 1 and cfg.recoat_time_s == 3.
    assert cfg.power_scale == cfg.speed_scale == 1.
    assert cfg.absorptivity == .4 and cfg.substrate == cfg.alloy
    assert cfg.section_overrides == {}
    assert cfg.spacing_m == .002 and cfg.max_event_s == .1


def test_identity_invalidates_physical_numerical_job_and_comparison_changes(tmp_path):
    job, _, cfg = fixture(tmp_path)
    original = workspace.configuration_id(job, cfg)
    assert workspace.configuration_id(job.source_sha256, cfg) == original
    assert workspace.configuration_id(job, replace(cfg, beam_diameter_m=80.*1e-6)) == original
    for changes in (dict(alloy="316L"), dict(power_scale=.9), dict(speed_scale=1.1),
                    dict(initial_k=373.15), dict(frame_interval_s=.001),
                    dict(section_overrides={"6": {"power_w": 80.}})):
        assert workspace.configuration_id(job, replace(cfg, **changes)) != original
    assert workspace.configuration_id("another-source", cfg) != original
    paired = workspace.configuration_id(job, cfg, "AlSi10Mg")
    assert paired != original
    assert paired == workspace.configuration_id(job, cfg, "AlSi10Mg", .15)
    assert paired != workspace.configuration_id(job, cfg, "AlSi10Mg", .2)
    with pytest.raises(ValueError, match="different"):
        workspace.configuration_id(job, cfg, "17-4PH")


def test_real_single_worker_freezes_request_and_fields_without_mutating_import(tmp_path):
    job, summary, cfg = fixture(tmp_path)
    source_bytes = {path.name: path.read_bytes() for path in summary.parent.iterdir()}
    input_cfg = replace(cfg, section_overrides={"6": {"power_w": 80.}})
    provenance = {"timing": {"source": "user estimate"}}
    folder = workspace.start_run(summary, input_cfg, provenance=provenance, results_root=tmp_path/"results")
    input_cfg.section_overrides["6"]["power_w"] = 999.
    provenance["timing"]["source"] = "changed after launch"
    status = wait_complete(folder)
    assert status["state"] == "complete", (status, (folder/"worker.log").read_text())
    saved = json.loads((folder/"request.json").read_text())
    assert saved["build_config"]["section_overrides"] == {"6": {"power_w": 80.}}
    assert saved["input_provenance"]["timing"]["source"] == "user estimate"
    assert saved["workspace_configuration_id"] == workspace.configuration_id(job, replace(cfg, section_overrides={"6": {"power_w": 80.}}))
    for name, data in source_bytes.items():
        assert (summary.parent/name).read_bytes() == data
    cases = workspace.read_view_cases(folder)
    assert len(cases) == 1 and cases[0]["available"]
    assert cases[0]["manifest"]["frames"][-1]["time_s"] > 0
    assert cases[0]["manifest"]["config"]["section_overrides"]["6"]["power_w"] == 80.
    exact = replace(cfg, section_overrides={"6": {"power_w": 80.}})
    assert workspace.matching_result(folder, job, exact)
    assert not workspace.matching_result(folder, job, replace(exact, speed_scale=.9))
    assert not workspace.matching_result(folder, job, replace(exact, alloy="316L"))
    assert not workspace.matching_result(folder, job, exact, "316L")
    assert workspace.cancel_run(folder) is False


def test_real_comparison_runs_both_materials_and_has_exact_configuration_identity(tmp_path):
    job, summary, cfg = fixture(tmp_path)
    folder = workspace.start_run(summary, cfg, "316L", .27, results_root=tmp_path/"results")
    status = wait_complete(folder)
    assert status["state"] == "complete", (status, (folder/"worker.log").read_text())
    saved = json.loads((folder/"request.json").read_text())
    assert saved["kind"] == "presets"
    cases = workspace.read_view_cases(folder)
    assert [case["material"] for case in cases] == ["17-4PH", "316L"]
    assert all(case["available"] for case in cases)
    assert cases[1]["manifest"]["config"]["absorptivity"] == .27
    assert cases[1]["manifest"]["config"]["substrate"] == "17-4PH"
    assert cases[0]["manifest"]["config"]["power_scale"] == cases[1]["manifest"]["config"]["power_scale"] == 1.
    assert workspace.matching_result(folder, job, cfg, "316L", .27)
    assert not workspace.matching_result(folder, job, cfg, "316L", .35)
    assert not workspace.matching_result(folder, job, cfg)
    with np.load(cases[0]["folder"]/"timeline.npz") as source:
        first = source["events"].copy()
    with np.load(cases[1]["folder"]/"timeline.npz") as source:
        second = source["events"].copy()
    np.testing.assert_array_equal(first, second)
    assert cases[0]["manifest"]["material_data"].keys() != cases[1]["manifest"]["material_data"].keys()


def test_comparison_pending_slot_never_reuses_primary_temperature(tmp_path):
    job, summary, cfg = fixture(tmp_path)
    folder = tmp_path/"study"
    save_imported_job(job, folder/"job", copy_source=False)
    second = workspace.comparison_config(cfg, "316L", .27)
    atomic_json(folder/"request.json", dict(kind="presets", build_config=asdict(cfg), job_source_sha256=job.source_sha256,
        options=dict(reference=preset_from_config(cfg, "17-4PH"), candidates=[preset_from_config(second, "316L")])))
    manifest = run_build(job, cfg, folder/"cases"/"first")
    slots = workspace.read_view_cases(folder)
    assert len(slots) == 2
    assert slots[0]["available"] and slots[0]["frame_count"] == len(manifest["frames"])
    assert slots[1]["material"] == "316L" and slots[1]["manifest"]["frames"] == []
    assert slots[1]["state"] == "queued" and not slots[1]["available"]
    assert len(workspace.read_view_cases(folder, include_pending=False)) == 1
    assert workspace.matching_result(folder, job, cfg, "316L", .27)


def test_cancelled_partial_reference_and_unstarted_comparison_are_both_terminal(tmp_path):
    job, _, cfg = fixture(tmp_path)
    folder = tmp_path/"cancelled-comparison"
    second = workspace.comparison_config(cfg, "316L", .27)
    atomic_json(folder/"request.json", dict(kind="presets", build_config=asdict(cfg), job_source_sha256=job.source_sha256,
        options=dict(reference=preset_from_config(cfg, "17-4PH"), candidates=[preset_from_config(second, "316L")])))
    case_folder = folder/"cases/first"
    manifest = run_build(job, cfg, case_folder)
    manifest.update(status="running", frames=manifest["frames"][:2])
    atomic_json(case_folder/"manifest.json", manifest)
    atomic_json(folder/"status.json", dict(state="cancelled", message="Cancelled by user after two saved fields"))
    atomic_json(folder/"report.json", dict(status="cancelled", rows=[], failures=[]))
    slots = workspace.read_view_cases(folder)
    assert [case["state"] for case in slots] == ["cancelled", "cancelled"]
    assert slots[0]["available"] and slots[0]["frame_count"] == 2
    assert "partial" in slots[0]["message"]
    assert slots[1]["manifest"]["frames"] == [] and not slots[1]["available"]
    assert "not calculated" in slots[1]["message"]
    assert all(case["manifest"]["status"] == case["state"] for case in slots)
    assert all(case["error"] for case in slots)
    # Outcome reconciliation is an in-memory reading policy, not a rewrite of
    # worker evidence or a side effect of PID reconciliation.
    assert json.loads((case_folder/"manifest.json").read_text())["status"] == "running"
    assert json.loads((folder/"status.json").read_text())["state"] == "cancelled"


@pytest.mark.parametrize("overall", ["complete", "partial", "failed"])
def test_terminal_missing_slot_never_claims_to_be_calculating(tmp_path, overall):
    job, _, cfg = fixture(tmp_path)
    folder = tmp_path/"missing-case"
    second = workspace.comparison_config(cfg, "316L", .27)
    atomic_json(folder/"request.json", dict(kind="presets", build_config=asdict(cfg), job_source_sha256=job.source_sha256,
        options=dict(reference=preset_from_config(cfg, "17-4PH"), candidates=[preset_from_config(second, "316L")])))
    run_build(job, cfg, folder/"cases/first")
    atomic_json(folder/"status.json", dict(state=overall, message="Missing case evidence"))
    slots = workspace.read_view_cases(folder)
    assert slots[0]["state"] == "complete"
    assert slots[1]["state"] == "not_calculated"
    assert slots[1]["message"] and not slots[1]["available"]


def test_report_failure_belongs_only_to_its_own_material_slot(tmp_path):
    job, _, cfg = fixture(tmp_path)
    folder = tmp_path/"failed-comparison"
    second = workspace.comparison_config(cfg, "316L", .27)
    atomic_json(folder/"request.json", dict(kind="presets", build_config=asdict(cfg), job_source_sha256=job.source_sha256,
        options=dict(reference=preset_from_config(cfg, "17-4PH"), candidates=[preset_from_config(second, "316L")])))
    run_build(job, cfg, folder/"cases/first")
    atomic_json(folder/"status.json", dict(state="partial", message="One case failed"))
    atomic_json(folder/"report.json", dict(status="partial", failures=[
        dict(preset_name="316L", material="316L", config=asdict(second), error="Nonlinear integration could not converge")]))
    slots = workspace.read_view_cases(folder)
    assert slots[0]["state"] == "complete" and slots[0]["error"] == ""
    assert slots[1]["state"] == "failed"
    assert slots[1]["message"] == slots[1]["error"] == "Nonlinear integration could not converge"


def test_mismatched_frozen_artifact_and_job_hash_cannot_claim_current_inputs(tmp_path):
    job, _, cfg = fixture(tmp_path)
    folder = tmp_path/"study"
    save_imported_job(job, folder/"job", copy_source=False)
    atomic_json(folder/"request.json", dict(kind="build", build_config=asdict(cfg), job_source_sha256=job.source_sha256))
    assert workspace.matching_result(folder, job, cfg)
    atomic_json(folder/"manifest.json", dict(config=asdict(replace(cfg, speed_scale=.9)),
        job=dict(source_sha256=job.source_sha256), frames=[]))
    assert not workspace.matching_result(folder, job, cfg)
    (folder/"manifest.json").unlink()
    summary = json.loads((folder/"job/job-summary.json").read_text())
    summary["source_sha256"] = "another-job"
    atomic_json(folder/"job/job-summary.json", summary)
    assert not workspace.matching_result(folder, job, cfg)


def test_legacy_standalone_build_keeps_its_frozen_fields_accessible(tmp_path):
    job, _, cfg = fixture(tmp_path)
    folder = tmp_path/"legacy-build"
    manifest = run_build(job, cfg, folder)
    assert not (folder/"request.json").exists()
    assert workspace.matching_result(folder, job, cfg)
    cases = workspace.read_view_cases(folder)
    assert len(cases) == 1 and cases[0]["available"]
    assert cases[0]["manifest"]["frames"] == manifest["frames"]


def test_geometry_identity_follows_frozen_file_contents_not_relocation(tmp_path):
    job, _, cfg = fixture(tmp_path)
    geometry = tmp_path/"part.stl"; geometry.write_bytes(b"closed aligned mesh fixture")
    folder = tmp_path/"study"; folder.mkdir()
    (folder/"supplement.stl").write_bytes(geometry.read_bytes())
    source = replace(cfg, geometry_file=str(geometry))
    frozen = replace(cfg, geometry_file="supplement.stl")
    atomic_json(folder/"request.json", dict(kind="build", build_config=asdict(frozen), job_source_sha256=job.source_sha256))
    assert workspace.matching_result(folder, job, source)
    (folder/"supplement.stl").write_bytes(b"different mesh")
    assert not workspace.matching_result(folder, job, source)


def test_run_scoped_cancellation_and_pid_reuse_detection(tmp_path):
    job, summary, cfg = fixture(tmp_path)
    folder = workspace.start_run(summary, cfg, results_root=tmp_path/"results")
    assert workspace.cancel_run(folder)
    marker = json.loads((folder/"CANCEL").read_text())
    assert marker["run_id"] == folder.name and marker["source"] == "simulation_workspace"
    terminal = wait_complete(folder)
    assert terminal["state"] == "cancelled"
    assert terminal["cancellation"]["acknowledged"]
    stale = tmp_path/"results/builds/reused-pid"
    atomic_json(stale/"request.json", dict(kind="build", build_config=asdict(cfg), job_source_sha256=job.source_sha256))
    atomic_json(stale/"status.json", dict(state="running", pid=os.getpid(), process_started_epoch_s=psutil.Process().create_time()-60))
    status = workspace.read_status(stale)
    assert status["state"] == "failed" and "exited" in status["message"]
    assert workspace.active_run(tmp_path/"results") == (None, None)


def test_global_live_guard_and_impossible_step_budget_fail_before_launch(tmp_path, monkeypatch):
    _, summary, cfg = fixture(tmp_path)
    monkeypatch.setattr(workspace, "active_run", lambda *args: (tmp_path/"another-job", {"state": "running"}))
    with pytest.raises(RuntimeError, match="already running"):
        workspace.start_run(summary, cfg, results_root=tmp_path/"results")
    monkeypatch.setattr(workspace, "active_run", lambda *args: (None, None))
    with pytest.raises(ValueError, match="step"):
        workspace.start_run(summary, replace(cfg, max_steps=1), results_root=tmp_path/"results")
    assert not (tmp_path/"results/builds").exists()
