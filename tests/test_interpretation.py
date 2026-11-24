"""The interpretation window must explain stored evidence without false ranks."""
from pathlib import Path
import json

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from heat_sim.interpretation import (config_changes, insights_text, load_result_insights,
                                     load_temperature_history)


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def saved_case(folder, config=None, state="complete", temperatures=(353.15, 453.15, 393.15),
               expose_from=0, coords_shift=False):
    config = config or dict(alloy="17-4PH", initial_k=353.15, power_scale=1., speed_scale=1.,
                            recoat_time_s=3., beam_diameter_m=80e-6, absorptivity=.35)
    frames = []
    for i, value in enumerate(temperatures):
        frame = folder/f"frames/{i:05d}.npz"
        frame.parent.mkdir(parents=True, exist_ok=True)
        active = np.ones((2, 2, 2))
        part = np.ones((2, 2, 2), dtype=bool)
        if i < expose_from:
            part[:] = False
        t = np.full((2, 2, 2), value)
        np.savez_compressed(frame, temperature_k=t, active_fraction=active, part_mask=part,
                            plate_mask=np.zeros_like(part),
                            x_m=np.array([0., .001])+(.01 if coords_shift and i == 0 else 0),
                            y_m=np.array([0., .001]), z_m=np.array([0., .001]))
        frames.append(dict(file=f"frames/{i:05d}.npz", time_s=float(i), layer_index=0))
    manifest = dict(status=state, config=config, frames=frames,
                    job=dict(name="part.mjob", source_sha256="a"*64, layer_count=725),
                    metrics=dict(peak_temperature_k=max(temperatures), stored_energy_j=8.,
                                 absorbed_energy_j=20., physical_duration_s=2., energy_error_fraction=1e-10,
                                 mean_material_peak_enthalpy_ratio=.08))
    put(folder/"manifest.json", manifest)
    return manifest


def run_setup(folder, kind="build"):
    put(folder/"request.json", dict(kind=kind, job_name="part.mjob", job_source_sha256="a"*64,
                                    build_config=dict(alloy="17-4PH", layer_start=0, layer_end=3)))
    put(folder/"job/job-summary.json", dict(name="part.mjob", source_sha256="a"*64, layer_count=725))
    put(folder/"status.json", dict(state="complete"))


def preset_study(folder):
    run_setup(folder, "presets")
    reference = dict(alloy="17-4PH", initial_k=353.15, power_scale=1., speed_scale=1.,
                     recoat_time_s=3., beam_diameter_m=80e-6, absorptivity=.35)
    candidate = {**reference, "initial_k": 373.15, "power_scale": .9,
                 "section_overrides": {"0": {"power_w": 110., "speed_m_s": 1.5}}}
    saved_case(folder/"cases/base", reference)
    saved_case(folder/"cases/lower", candidate)
    saved_case(folder/"cases/mismatch", candidate)
    rows = [dict(preset_name="Successful reference", material="17-4PH", kind="baseline",
                 case_folder="cases/base", comparable=True, score=0., mean_peak_enthalpy_fraction=.1),
            dict(preset_name="Less power", material="17-4PH", kind="candidate",
                 case_folder="cases/lower", comparable=True, score=.04, mean_peak_enthalpy_fraction=.09),
            dict(preset_name="Different prefix", material="17-4PH", kind="candidate",
                 case_folder="cases/mismatch", comparable=False, score=.001,
                 comparison_exclusion="Actual exposure signatures differ")]
    put(folder/"report.json", dict(status="complete", baseline_material="17-4PH", study_name="Lower-power trial bracket",
                                    rows=rows, failures=[], config=reference))
    return rows


def test_single_build_reads_frozen_inputs_and_never_invents_a_comparison(tmp_path):
    run_setup(tmp_path)
    saved_case(tmp_path)
    data = load_result_insights(tmp_path)
    assert "3 of 725" in data["layer_label"]
    case = data["cases"][0]
    assert case["peak_temperature_c"] == pytest.approx(180.)
    assert case["config"]["initial_k"] == 353.15
    assert case["score"] is None and data["ranked"] == []
    assert case["mean_index_weighted"]
    assert case["energy_check_passed"]
    assert "part/powder/plate" in " ".join(case["interpretation"])
    assert "not an optimal printer recipe" in insights_text(data)


def test_only_complete_comparable_candidates_rank_and_physical_changes_are_explicit(tmp_path):
    preset_study(tmp_path)
    data = load_result_insights(tmp_path)
    assert data["reference"]["name"] == "Successful reference"
    assert [case["name"] for case in data["ranked"]] == ["Less power"]
    selected = data["ranked"][0]
    by_setting = {row["Setting"]: row for row in selected["changes"]}
    assert by_setting["Preheat (°C)"]["Selected preset"] == "100"
    assert by_setting["Imported power multiplier"]["Selected preset"] == "0.9"
    assert by_setting["Region 0 speed (mm/s) before global multipliers"]["Selected preset"] == "1500"
    text = " ".join(selected["interpretation"])
    assert "higher power or slower scans" in text
    assert "own alloy and preheat" in text
    mismatch = data["cases"][2]
    assert mismatch["score"] is None
    assert "Actual exposure signatures differ" in " ".join(mismatch["interpretation"])


def test_shared_case_artifacts_still_keep_distinct_preset_identities(tmp_path):
    rows = preset_study(tmp_path)
    rows.append({**rows[0], "kind": "candidate", "preset_name": "Reference clone"})
    report = json.loads((tmp_path/"report.json").read_text())
    report["rows"] = rows
    put(tmp_path/"report.json", report)
    data = load_result_insights(tmp_path)
    assert len(set(case["id"] for case in data["cases"])) == 4
    assert data["ranked"][0]["name"] == "Reference clone"


def test_failed_or_nonfinite_results_never_acquire_a_thermal_rank(tmp_path):
    rows = preset_study(tmp_path)
    manifest_path = tmp_path/"cases/lower/manifest.json"
    manifest = json.loads(manifest_path.read_text()); manifest["status"] = "failed"
    put(manifest_path, manifest)
    report = json.loads((tmp_path/"report.json").read_text())
    report["rows"][2].update(comparable=True, score=float("nan"))
    report["failures"] = [dict(preset_name="Less power", error="Deliberate integration failure")]
    put(tmp_path/"report.json", report)
    data = load_result_insights(tmp_path)
    assert data["ranked"] == []
    assert "partial frames" in " ".join(data["cases"][1]["interpretation"])
    assert "Deliberate integration failure" in insights_text(data)


def test_case_paths_cannot_escape_the_study(tmp_path):
    folder = tmp_path/"study"
    preset_study(folder)
    saved_case(tmp_path/"outside")
    report = json.loads((folder/"report.json").read_text())
    report["rows"].append(dict(preset_name="Outside", case_folder="../outside", comparable=True, score=0.))
    put(folder/"report.json", report)
    data = load_result_insights(folder)
    assert len(data["cases"]) == 3
    assert data["failures"][-1]["error"] == "Saved case path is invalid"


def test_fixed_cell_history_uses_actual_stored_samples_and_keeps_unexposed_values_missing(tmp_path):
    saved_case(tmp_path, temperatures=(353.15, 453.15, 393.15, 383.15, 373.15), expose_from=1)
    history = load_temperature_history(tmp_path, max_points=3)
    assert history["available_count"] == 5 and history["sampled_count"] == 3
    assert [row["time_s"] for row in history["rows"]] == [0., 2., 4.]
    assert history["rows"][0]["probe_temperature_c"] is None
    assert history["rows"][-1]["probe_temperature_c"] == pytest.approx(100.)
    assert history["position_m"] == [0., 0., 0.]
    assert history["rows"][1]["part_max_temperature_c"] == pytest.approx(120.)


def test_temperature_history_refuses_inconsistent_grid_coordinates(tmp_path):
    saved_case(tmp_path, coords_shift=True)
    with pytest.raises(ValueError, match="inconsistent grid"):
        load_temperature_history(tmp_path)


def test_results_window_renders_saved_named_presets_and_selected_physical_inputs(tmp_path):
    preset_study(tmp_path)
    source = f'''
from heat_sim.run_catalog import read_run
from heat_sim.results_ui import render_results_insights
render_results_insights([read_run({str(tmp_path)!r})])
'''
    app = AppTest.from_string(source).run(timeout=30)
    assert not app.exception
    assert any("Closest sampled response: Less power" in item.value for item in app.success)
    assert any("Interpreting Less power" in item.value for item in app.markdown)
    assert any("Lower-power trial bracket" in item.value for item in app.markdown)
    assert any("100" in str(table.value) and "Preheat" in str(table.value) for table in app.dataframe)
    assert any("Frozen" in str(item.value) or "frozen inputs" in str(item.value) for item in app.caption)
    assert len(app.get("vega_lite_chart")) == 2
    assert any("each case's own saved physical times" in item.value for item in app.caption)


def test_adjust_selected_preset_passes_frozen_physical_inputs_to_callback(tmp_path):
    preset_study(tmp_path)
    app = AppTest.from_string(f'''
import streamlit as st
from heat_sim.run_catalog import read_run
from heat_sim.results_ui import render_results_insights
def adjust(case):
    st.session_state.adjusted_case = case
render_results_insights([read_run({str(tmp_path)!r})], on_edit_preset=adjust)
''').run(timeout=30)
    assert not app.exception
    next(button for button in app.button if button.label == "Adjust this preset").click().run(timeout=30)
    assert not app.exception
    assert app.session_state["adjusted_case"]["name"] == "Less power"
    assert app.session_state["adjusted_case"]["config"]["initial_k"] == 373.15
    assert app.session_state["adjusted_case"]["config"]["section_overrides"] == {"0": {"power_w": 110., "speed_m_s": 1.5}}


def test_results_window_handles_existing_single_build_without_preset_features(tmp_path):
    run_setup(tmp_path)
    saved_case(tmp_path)
    app = AppTest.from_string(f'''
from heat_sim.run_catalog import read_run
from heat_sim.results_ui import render_results_insights
render_results_insights([read_run({str(tmp_path)!r})])
''').run(timeout=30)
    assert not app.exception
    assert not app.success
    assert any("reference heat response" in item.value for item in app.markdown)
    assert any("weighted by scan coverage" in item.value for item in app.caption)
