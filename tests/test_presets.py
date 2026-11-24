"""Physical-setting presets execute real solver cases and preserve identities."""
from dataclasses import replace
import json

import numpy as np
import pytest

from heat_sim import presets
from heat_sim.build_solver import BuildConfig
from heat_sim.job_import import ImportedJob, ImportedLayer
from heat_sim.solver import SimulationCancelled, SimulationError


def fixture():
    vectors = np.array([[0., 0., .001, 0., 120., 1.4],
                        [.001, 0., 0., .00006, 100., 1.]])
    layer = ImportedLayer(0, 0., vectors, np.array([0, 6]), np.array([0, 0]), np.array([0, 1]))
    job = ImportedJob("preset-fixture", "none", "preset-fixture-hash", "test", 60e-6, [layer], {})
    cfg = BuildConfig(absorptivity=.35, layer_end=1, final_cooldown_s=.002,
        spacing_m=.0005, z_spacing_m=.00012, plate_depth_m=.00024, padding_m=.0005,
        max_event_s=.001, frame_interval_s=.0005, max_frames=12)
    return job, cfg


def test_library_roundtrip_update_duplicate_names_and_job_isolation(tmp_path):
    _, cfg = fixture()
    source = presets.preset_from_config(cfg, "My successful steel", "reference")
    saved = presets.save_preset(tmp_path, "job-a", source, cfg)
    source["section_overrides"]["6"] = {"power_w": 155.}
    assert presets.load_presets(tmp_path, "job-a")[0]["section_overrides"] == {}
    assert saved["saved_utc"]
    assert presets.load_presets(tmp_path, "job-b") == []
    updated = presets.save_preset(tmp_path, "job-a", {**saved, "power_scale": 1.12}, cfg)
    assert len(presets.load_presets(tmp_path, "job-a")) == 1
    assert updated["power_scale"] == 1.12
    with pytest.raises(ValueError, match="already has this name"):
        presets.save_preset(tmp_path, "job-a", {**saved, "id": "other"}, cfg)
    presets.delete_preset(tmp_path, "job-a", saved["id"])
    assert presets.load_presets(tmp_path, "job-a") == []


def test_minimal_material_change_uses_independent_optical_default_and_fixes_numerics():
    _, cfg = fixture()
    target = presets.preset_config(cfg, dict(name="Titanium trial", material="Ti-6Al-4V"))
    assert target.alloy == "Ti-6Al-4V"
    assert target.absorptivity == presets.ILLUSTRATIVE_ABSORPTIVITY["Ti-6Al-4V"]
    assert target.spacing_m == cfg.spacing_m
    with pytest.raises(ValueError, match="Unsupported preset fields"):
        presets.preset_config(cfg, dict(name="Different mesh", spacing_m=.002))
    with pytest.raises(ValueError, match="finite"):
        presets.preset_config(cfg, dict(name="Bad power", power_scale=float("nan")))
    with pytest.raises(ValueError, match="below both melting"):
        presets.preset_config(cfg, dict(name="Bad preheat", initial_k=4000.))


def test_real_named_presets_preserve_paths_and_apply_region_settings(tmp_path):
    job, cfg = fixture()
    (tmp_path / "request.json").write_text('{"kind":"presets"}')
    options = dict(study_name="Contour adjustment", reference={"name": "Current successful settings"},
        candidates=[dict(name="Gentler contour", power_scale=1.1, speed_scale=.9,
            section_overrides={"6": {"power_w": 80., "speed_m_s": .8}}),
            dict(name="316L trial", material="316L", absorptivity=.27)])
    report = presets.run_presets(job, cfg, options, tmp_path)
    assert report["status"] == "complete" and not report["failures"]
    assert len(report["rows"]) == 3
    assert report["study_name"] == report["options"]["study_name"] == "Contour adjustment"
    assert json.loads((tmp_path / "request.json").read_text()) == {"kind": "presets"}
    reference, candidate, target = report["rows"]
    assert reference["kind"] == "baseline"
    assert reference["preset_name"] == "Current successful settings"
    assert candidate["comparable"] and candidate["score"] is not None
    assert target["comparable"] and target["material"] == "316L"
    assert candidate["config"]["section_overrides"] == {"6": {"power_w": 80., "speed_m_s": .8}}
    with np.load(tmp_path / reference["case_folder"] / "timeline.npz") as data:
        before = data["events"][data["events"]["kind"] == 1]
    with np.load(tmp_path / candidate["case_folder"] / "timeline.npz") as data:
        after = data["events"][data["events"]["kind"] == 1]
    np.testing.assert_array_equal(before["start_m"], after["start_m"])
    np.testing.assert_array_equal(before["end_m"], after["end_m"])
    np.testing.assert_allclose(after["power_w"], [132., 88.])
    np.testing.assert_allclose(after["speed_m_s"], [1.26, .72])
    assert candidate["metrics"]["peak_enthalpy_fraction"] > 0
    # The override must reach the heat source, rather than only relabel a
    # timeline. This expected energy follows P*length/speed independently.
    expected = .35 * (132. * .001 / 1.26 + 88. * np.hypot(.001, .00006) / .72)
    assert candidate["metrics"]["nominal_absorbed_energy_j"] == pytest.approx(expected)
    assert candidate["metrics"]["absorbed_energy_j"] == pytest.approx(expected, rel=1e-8)
    assert candidate["metrics"]["peak_temperature_k"] != pytest.approx(reference["metrics"]["peak_temperature_k"])
    assert (tmp_path / candidate["probe_csv"]).is_file()
    assert (tmp_path / "comparison.csv").is_file()
    assert report["experimental_validation"] is False


def test_same_settings_have_zero_score_and_reusable_computed_cases(tmp_path, monkeypatch):
    job, cfg = fixture()
    options = dict(reference={"name": "Reference"}, candidates=[{"name": "Unchanged check"}])
    report = presets.run_presets(job, cfg, options, tmp_path)
    assert report["rows"][1]["score"] == pytest.approx(0.)
    assert report["rows"][0]["case_folder"] == report["rows"][1]["case_folder"]
    assert report["closest_preset_name"] == "Unchanged check"
    def must_not_run(*a, **k):
        raise AssertionError("Computed matching case should be reused")
    monkeypatch.setattr(presets, "run_build", must_not_run)
    cached = presets.run_presets(job, cfg, options, tmp_path)
    assert cached["status"] == "complete" and cached["cache_hits"] == 2


def test_changed_preheat_is_recorded_and_actual_fields_use_that_preheat(tmp_path):
    job, cfg = fixture()
    report = presets.run_presets(job, cfg,
        dict(reference={"name": "80 C"}, candidates=[{"name": "120 C", "initial_k": 393.15}]), tmp_path)
    assert report["status"] == "complete"
    row = report["rows"][1]
    assert row["config"]["initial_k"] == 393.15
    manifest = json.loads((tmp_path / row["case_folder"] / "manifest.json").read_text())
    assert manifest["config"]["initial_k"] == 393.15
    assert row["score"] is not None
    assert "own material and initial temperature" in report["normalization"]


def test_time_clipped_different_paths_are_excluded_from_ranking(tmp_path):
    job, cfg = fixture()
    cfg = replace(cfg, time_limit_s=.0004)
    report = presets.run_presets(job, cfg,
        dict(reference={"name": "Reference"}, candidates=[{"name": "Slower", "speed_scale": .5}]), tmp_path)
    candidate = report["rows"][1]
    assert candidate["valid"] and not candidate["comparable"]
    assert candidate["score"] is None
    assert report["rankings"][0]["ranked_candidates_count"] == 0


def test_reference_failure_preserves_candidate_without_fabricating_score(tmp_path, monkeypatch):
    job, cfg = fixture()
    real = presets.run_build
    def fail_reference(job, config, *a, **k):
        if config.power_scale == 1.:
            raise SimulationError("Deliberate reference failure")
        return real(job, config, *a, **k)
    monkeypatch.setattr(presets, "run_build", fail_reference)
    report = presets.run_presets(job, cfg, dict(reference={"name": "Reference"},
        candidates=[{"name": "Candidate", "power_scale": 1.1}]), tmp_path)
    assert report["status"] == "partial"
    assert len(report["failures"]) == len(report["rows"]) == 1
    assert report["rows"][0]["score"] is None
    assert not report["comparison_available"]


def test_all_failed_is_failed_and_early_cancel_does_not_invent_results(tmp_path, monkeypatch):
    job, cfg = fixture()
    def fail(*a, **k):
        raise SimulationError("Deliberate numerical failure")
    monkeypatch.setattr(presets, "run_build", fail)
    options = dict(reference={"name": "Reference"}, candidates=[{"name": "Candidate", "power_scale": 1.1}])
    report = presets.run_presets(job, cfg, options, tmp_path / "failed")
    assert report["status"] == "failed" and report["completed_case_count"] == 0
    assert report["failed_case_count"] == 2
    # Cancellation is polled before any simulation, so no thermal outputs are invented.
    with pytest.raises(SimulationCancelled):
        presets.run_presets(job, cfg, options, tmp_path / "cancelled", cancel=lambda: True)
    saved = json.loads((tmp_path / "cancelled" / "report.json").read_text())
    assert saved["status"] == "cancelled" and saved["rows"] == []


def test_all_preset_step_plans_and_unique_names_are_validated():
    job, cfg = fixture()
    options = dict(reference={"name": "Reference"}, candidates=[{"name": "Slow", "speed_scale": .1}])
    plans = presets.plan_presets(job, cfg, options)
    assert len(plans) == 2
    assert plans[1]["plan"]["scan_time_s"] == pytest.approx(plans[0]["plan"]["scan_time_s"] * 10)
    with pytest.raises(ValueError, match="different name"):
        presets.plan_presets(job, cfg, dict(reference={"name": "Ref"}, candidates=[{"name": "ref"}]))
    with pytest.raises(ValueError, match="distinct id"):
        presets.plan_presets(job, cfg, dict(reference={"name": "Ref", "id": "same"},
            candidates=[{"name": "Other", "id": "same"}]))


def test_normalization_and_config_snapshots_do_not_alias_editor_region_dicts():
    _, cfg = fixture()
    draft = dict(name="Edit me", section_overrides={"6": {"power_w": 80.}})
    normalized = presets.normalize_preset(draft, cfg)
    frozen_config = presets.preset_config(cfg, draft)
    draft["section_overrides"]["6"]["power_w"] = 190.
    normalized["section_overrides"]["6"]["power_w"] = 150.
    assert frozen_config.section_overrides == {"6": {"power_w": 80.}}
    assert cfg.section_overrides == {}


def test_invalid_saved_energy_check_cannot_be_ranked(tmp_path, monkeypatch):
    job, cfg = fixture()
    monkeypatch.setattr(presets, "run_build", lambda *a, **k:
        dict(status="complete", metrics={"energy_error_fraction": -1e-6}))
    report = presets.run_presets(job, cfg, dict(reference={"name": "Ref"},
        candidates=[{"name": "Trial", "power_scale": 1.1}]), tmp_path)
    assert report["status"] == "failed"
    assert report["rows"] == []
    assert all("energy check" in row["error"] for row in report["failures"])
