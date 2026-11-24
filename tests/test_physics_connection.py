"""Audit the import → process schedule → enthalpy fields → viewer connection.

Real-job checks are optional when local user files are absent. The synthetic
heat/no-new-heat experiment isolates state retention and plate conduction.
"""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pytest

from heat_sim.build_solver import BuildConfig, build_timeline, run_build, _conductances
from heat_sim.build_viewer import playback_html
from heat_sim.job_import import ImportedJob, ImportedLayer, load_job
from heat_sim.materials import MATERIALS
from heat_sim.run_planning import estimate_run_plan


ROOT = Path(__file__).resolve().parents[1]
REAL_MJOB = ROOT / "results/imports/5cd20985e62718377dce/source.mjob"
REAL_HTM = ROOT / "results/imports/17-4-actual/job-summary.json"


@pytest.fixture(scope="module")
def real_job():
    if not REAL_MJOB.is_file():
        pytest.skip("The user's actual Dachshund source job is not present")
    before = sha256(REAL_MJOB.read_bytes()).hexdigest()
    job = load_job(REAL_MJOB)
    assert sha256(REAL_MJOB.read_bytes()).hexdigest() == before == job.source_sha256
    return job


def test_actual_job_import_is_inert_and_short_preview_scope_is_honest(real_job):
    assert len(real_job.layers) == 725
    assert real_job.metadata["verified_layer_hashes"] == 725
    assert {int(pid) for layer in real_job.layers[:100] for pid in layer.process_ids} == {16}
    used = {int(pid) for layer in real_job.layers[:110] for pid in layer.process_ids}
    assert used == {0, 1, 2, 6, 7, 8, 16}
    # Geometry is imported, while machine schedule and optical units remain
    # qualified explicitly rather than being promoted to machine measurements.
    assert "imported" in real_job.metadata["provenance"]["sequence"]
    assert "not" in real_job.metadata["provenance"]["sequence"]
    assert "not recorded" in real_job.metadata["provenance"]["delays"]


def test_actual_mixed_region_scan_order_energy_and_timing_are_vector_derived(real_job):
    cfg = BuildConfig(absorptivity=.35, layer_start=200, layer_end=203,
        warm_start_assumed=True, power_scale=.9, speed_scale=1.13, power_factor=.8,
        recoat_time_s=3., final_cooldown_s=.2,
        section_overrides={"0": {"power_w": 150., "speed_m_s": 1.2}})
    events = build_timeline(real_job, cfg)
    scans = events[events["kind"] == 1]
    selected = real_job.layers[200:203]
    vectors = np.concatenate([layer.vectors for layer in selected])
    sections = np.concatenate([layer.process_ids for layer in selected])
    z = np.concatenate([np.full(len(layer.vectors), layer.z_m + real_job.layer_thickness_m)
                        for layer in selected])
    np.testing.assert_array_equal(scans["start_m"][:, :2], vectors[:, :2])
    np.testing.assert_array_equal(scans["end_m"][:, :2], vectors[:, 2:4])
    np.testing.assert_array_equal(scans["start_m"][:, 2], z)
    np.testing.assert_array_equal(scans["section"], sections)
    np.testing.assert_array_equal(scans["vector_index"], np.arange(len(vectors)))
    power = np.where(sections == 0, 150., vectors[:, 4]) * .9 * .8
    speed = np.where(sections == 0, 1.2, vectors[:, 5]) * 1.13
    duration = np.linalg.norm(vectors[:, 2:4] - vectors[:, :2], axis=1) / speed
    np.testing.assert_allclose(scans["power_w"], power, rtol=0, atol=1e-13)
    np.testing.assert_allclose(scans["speed_m_s"], speed, rtol=0, atol=1e-13)
    np.testing.assert_allclose(scans["t1_s"] - scans["t0_s"], duration, rtol=0, atol=2e-15)
    assert .35 * np.sum(scans["power_w"] * (scans["t1_s"] - scans["t0_s"])) == pytest.approx(.35 * np.sum(power * duration))
    recoats = events[events["kind"] == 3]
    np.testing.assert_allclose(recoats["t1_s"] - recoats["t0_s"], [3., 3.])
    assert estimate_run_plan(real_job, cfg)["estimated_physical_time_s"] == pytest.approx(events[-1]["t1_s"])


def test_actual_disabled_and_reserved_legacy_regions_are_not_exposed():
    if not REAL_HTM.is_file():
        pytest.skip("The user's legacy 17-4 job is not present")
    job = load_job(REAL_HTM)
    used = {int(pid) for layer in job.layers for pid in layer.process_ids}
    assert 12 not in used
    assert not used.intersection({9, 10, 11, 13, 14, 15})
    # This source declares the disabled 380 W core recipe without storing any
    # core points at all. Fixture tests separately exercise discarded records.
    assert not any(np.any(layer.vectors[:, 4] == 380.) for layer in job.layers)
    assert any(part.get("disabled_core") is True for part in job.metadata["parts"].values())


def state_experiment():
    layers = []
    for index in range(2):
        # The second layer's zero-power paths isolate retained heat: there is
        # deposition and laser-off transport, but no newly absorbed energy.
        vectors = np.array([[0., 0., .001, 0., 120. if index == 0 else 0., 1.4],
                            [.001, 0., 0., .00006, 100. if index == 0 else 0., 1.]])
        layers.append(ImportedLayer(index, index * 60e-6, vectors,
                                    np.array([0, 16]), np.array([0, 0]), np.array([0, 1])))
    job = ImportedJob("state-retention-experiment", "none", "state-audit", "test", 60e-6, layers, {})
    cfg = BuildConfig(absorptivity=.35, layer_end=2, recoat_time_s=.02, final_cooldown_s=.03,
        spacing_m=.0005, z_spacing_m=.00024, plate_depth_m=.00048, padding_m=.0005,
        max_event_s=.001, frame_interval_s=.001, max_frames=80, support_sections=(16,))
    return job, cfg


def test_heat_persists_across_deposition_and_cools_through_plate_without_new_heat(tmp_path):
    job, cfg = state_experiment()
    timeline = build_timeline(job, cfg)
    manifest = run_build(job, cfg, tmp_path)
    stored = []
    for entry in manifest["frames"]:
        with np.load(tmp_path / entry["file"], allow_pickle=False) as saved:
            stored.append((entry, {key: saved[key].copy() for key in saved.files}))
    second_activation = float(timeline[(timeline["kind"] == 0) & (timeline["layer_index"] == 1)][0]["t0_s"])
    after_activation = next(field for entry, field in stored if entry["time_s"] >= second_activation)
    assert after_activation["temperature_k"].max() > cfg.initial_k + 5.
    assert after_activation["temperature_k"][after_activation["plate_mask"]].max() > cfg.initial_k + 5.
    # During recoating the top cools while energy reaches initially cold plate
    # cells. This cannot be obtained by resetting each track/layer to preheat.
    recoat = timeline[timeline["kind"] == 3][0]
    off = [field for entry, field in stored if recoat["t0_s"] < entry["time_s"] < recoat["t1_s"]]
    assert len(off) >= 4
    assert off[-1]["temperature_k"].max() < off[0]["temperature_k"].max()
    assert max(float(field["temperature_k"][field["plate_mask"]].max()) for field in off) > float(off[0]["temperature_k"][off[0]["plate_mask"]].max())
    for _, field in stored:
        np.testing.assert_allclose(field["temperature_k"][field["active_fraction"] <= 0], cfg.initial_k, atol=1e-4)
        assert np.all(field["consolidated_fraction"] <= field["active_fraction"] + 1e-7)
    assert stored[-1][1]["part_mask"].any() and stored[-1][1]["support_mask"].any()
    expected = .35 * sum(float(np.sum(layer.vectors[:, 4] * np.linalg.norm(layer.vectors[:, 2:4] - layer.vectors[:, :2], axis=1) / layer.vectors[:, 5])) for layer in job.layers)
    metrics = manifest["metrics"]
    assert metrics["absorbed_energy_j"] == pytest.approx(expected, rel=1e-10)
    assert metrics["stored_energy_j"] + metrics["boundary_loss_j"] == pytest.approx(expected, abs=1e-9)
    assert metrics["boundary_loss_j"] > 0 and metrics["energy_error_fraction"] < 1e-8


def test_future_material_has_no_conduction_and_powder_differs_from_consolidated_metal():
    _, cfg = state_experiment()
    material = MATERIALS[cfg.alloy]
    shape = (2, 1, 3)
    temperature = np.full(shape, cfg.initial_k)
    active = np.ones(shape); active[:, :, 2] = 0
    plate = np.zeros(shape, bool); plate[:, :, 0] = True
    consolidated = plate.astype(float); consolidated[1, :, 1] = 1.
    _, _, vertical, _ = _conductances(temperature, active, consolidated, plate,
                                       [material, material], cfg, 1)
    assert np.all(vertical[:, :, 0] > 0)
    assert np.all(vertical[:, :, 1] == 0)
    assert vertical[1, 0, 0] > vertical[0, 0, 0]
    bulk = float(material.conductivity(cfg.initial_k))
    expected_powder_plate = cfg.spacing_m ** 2 / (cfg.z_spacing_m / 2 / bulk + cfg.z_spacing_m / 2 / cfg.powder_k_w_mk)
    assert vertical[0, 0, 0] == pytest.approx(expected_powder_plate)


def test_material_selection_changes_solved_fields_and_viewer_colours_are_saved_temperatures(tmp_path):
    job, cfg = state_experiment()
    results = []
    for alloy in MATERIALS:
        folder = tmp_path / alloy
        manifest = run_build(job, replace(cfg, alloy=alloy), folder)
        assert manifest["material_data"][alloy]["rho_kg_m3"] == MATERIALS[alloy].rho
        with np.load(folder / manifest["frames"][-1]["file"], allow_pickle=False) as saved:
            last = {key: saved[key].copy() for key in saved.files}
        results.append((manifest["metrics"]["peak_temperature_k"], last["temperature_k"]))
        html = playback_html(folder, manifest, live=True)
        payload = json.loads(re.search(r"const D=(.*?);const \$=", html, re.S).group(1))
        for point, value in zip(payload["points"], payload["frames"][-1]["temperature"]):
            indices = tuple(int(np.argmin(abs(last[key] - position / 1000)))
                            for key, position in zip(("x_m", "y_m", "z_m"), point))
            assert value == pytest.approx(float(last["temperature_k"][indices]) - 273.15, abs=.0051)
    assert len({round(peak, 3) for peak, _ in results}) == len(MATERIALS)
    assert not np.allclose(results[0][1], results[1][1], rtol=0, atol=.001)
