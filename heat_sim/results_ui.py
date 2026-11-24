"""A focused window for reading frozen simulation results and preset tradeoffs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .interpretation import load_result_insights, load_temperature_history, insights_text


@st.cache_data(show_spinner=False, max_entries=24)
def _saved_history(folder, manifest_stamp):
    return load_temperature_history(folder)


def _metric(value, suffix="", precision=3):
    return "Unavailable" if value is None else f"{value:,.{precision}g}{suffix}"


def _case_table(cases):
    return pd.DataFrame([{
        "Preset / case": case["name"], "Material": case["material"],
        "Thermal difference ↓": case["score"],
        "Mean peak enthalpy index": case["mean_peak_enthalpy_index"],
        "Peak voxel temperature (°C)": case["peak_temperature_c"],
        "Selected print duration (s)": case["duration_s"],
        "Heat retained at end (J)": case["retained_energy_j"],
        "Comparison": "Reference" if case["kind"] in ("baseline", "reference") else
                      "Scored" if case["score"] is not None else "Excluded / not scored",
    } for case in cases])


def render_results_insights(runs, selected_folder=None, on_view=None, on_edit_preset=None):
    """Render saved evidence from the current job's RunInfo values.

    The caller filters runs to the imported source job and only calls this when
    the tab is open. on_view(folder) may navigate to that exact study's playback.
    on_edit_preset(case) may open a new draft from the frozen physical inputs.
    """
    st.subheader("Understand the simulation")
    st.caption("Compare the response of saved presets, see what changed, and choose the next settings to test.")
    if not runs:
        st.info("Run a simulation or a preset comparison first. Its saved results will appear here.")
        return
    namespace = str(runs[0].job_sha256 or "unknown")[:16]
    show_failed = st.checkbox("Include stopped and failed simulations", value=False,
                              key="insights_failed_"+namespace)
    available = [run for run in runs if run.state in ("complete", "partial") or show_failed]
    if not available:
        st.info("No completed results for this job yet. Include stopped and failed simulations to review diagnostics.")
        return
    by_folder = {run.folder.resolve(): run for run in available}
    preferred = Path(selected_folder).resolve() if selected_folder else None
    default = preferred if preferred in by_folder else next((run.folder.resolve() for run in available if run.state == "complete"), available[0].folder.resolve())
    folders = list(by_folder)
    key = "insights_run_"+namespace
    if key in st.session_state and st.session_state[key] not in folders:
        st.session_state.pop(key)
    folder = st.selectbox("Results to interpret", folders, index=folders.index(default),
                          format_func=lambda value: by_folder[value].label, key=key)
    try:
        data = load_result_insights(folder)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Cannot interpret this saved result: {exc}")
        return
    with st.container(border=True):
        if data.get("study_name"):
            st.markdown(f"**{data['study_name']}**")
        st.markdown(f"**{data['job_name']}**")
        st.write(f"{data['material_label']} · {data['layer_label']} · {data['status_label']}")
        st.caption(f"Saved run {data['run_id']}. Values below come from this run's frozen inputs.")
        if on_view and by_folder[folder].has_results:
            st.button("Open this simulation in playback", key="insights_play_"+data["run_id"],
                      on_click=on_view, args=(folder,))
    if data["state"] not in ("complete", "partial"):
        st.warning(data["message"] or "This simulation did not complete; its results are incomplete.")
    config = data["scope_config"]
    if config.get("layer_start", 0):
        st.info("This run starts after layer 0. Earlier layers were assumed to be at preheat; their earlier heating was omitted.")
    if config.get("region_xy_m"):
        st.info("This is a cropped region. The simulated sides omit heat transfer to the surrounding part.")
    if config.get("time_limit_s"):
        st.info("A physical-time limit was used. Scores are available only for cases with the same actual exposed path footprint.")
    cases = data["cases"]
    if not cases:
        st.info("No completed case metrics were saved. Review the failed-case details below, then adjust the run configuration.")
    else:
        st.markdown("**What should I look for?**")
        if data["ranked"]:
            st.write("Use thermal difference to find the preset closest to your reference response. Then compare print duration and retained heat, and inspect the temperature history before narrowing a physical trial.")
        elif len(cases) == 1:
            st.write("This run establishes a reference heat response. Add named alternatives in Preset lab and simulate the same range to see how your adjustments change it.")
        else:
            st.write("Compare only cases marked Scored. Unfinished cases or different exposure footprints cannot establish a closest preset.")
        reference = data["reference"]
        a, b, c = st.columns(3)
        a.metric("Reference", data["reference_name"])
        b.metric("Cases with finished metrics", sum(case["completed"] for case in cases))
        c.metric("Failed cases", len(data["failures"]))
        materials = list(dict.fromkeys(case["material"] for case in cases))
        material = st.selectbox("Material to compare", ["All simulated materials", *materials],
                                key="insights_material_"+data["run_id"]) if len(materials) > 1 else materials[0]
        visible = cases if material == "All simulated materials" else [case for case in cases if case["material"] == material or case is reference]
        ranked = [case for case in data["ranked"] if material == "All simulated materials" or case["material"] == material]
        if ranked:
            best = ranked[0]
            st.success(f"Closest sampled response: {best['name']} · difference {best['score']:.4g}")
            if len({case['material'] for case in ranked}) > 1:
                st.caption("This is the smallest score across the selected materials. Choose one material above to compare its presets separately.")
            st.caption("A lower score measures thermal similarity to the saved reference. It does not establish an optimal recipe, acceptable defects, or successful printing.")
        st.dataframe(_case_table(visible), hide_index=True, width="stretch")
        scored = [case for case in visible if case["score"] is not None and case["kind"] not in ("reference", "baseline", "uncertainty")]
        if scored:
            ordered = sorted(scored, key=lambda case: case["score"])
            fig, ax = plt.subplots(figsize=(10, max(2.2, min(7.5, .38*len(ordered)+1.1))))
            labels = [case["name"] for case in ordered]
            ax.barh(range(len(ordered)), [case["score"] for case in ordered], color="#ff925c")
            ax.set_yticks(range(len(labels)), labels)
            ax.invert_yaxis()
            ax.set_xlabel("Thermal difference from the reference · lower is closer")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        case_by_id = {case["id"]: case for case in visible}
        ids = list(case_by_id)
        default_case = ranked[0]["id"] if ranked and ranked[0]["id"] in ids else ids[0]
        case_key = "insights_case_"+data["run_id"]
        if case_key in st.session_state and st.session_state[case_key] not in ids:
            st.session_state.pop(case_key)
        chosen_id = st.selectbox("Preset / case to inspect", ids, index=ids.index(default_case),
                                 format_func=lambda value: case_by_id[value]["name"], key=case_key)
        chosen = case_by_id[chosen_id]
        st.markdown(f"**Interpreting {chosen['name']}**")
        if on_edit_preset:
            st.button("Adjust this preset", key="insights_adjust_"+data["run_id"],
                      on_click=on_edit_preset, args=(chosen,),
                      help="Open these saved settings as a new draft in Preset lab, then simulate the adjusted version.")
        if chosen["changes"]:
            st.dataframe(pd.DataFrame(chosen["changes"]), hide_index=True, width="stretch")
        elif reference and chosen is not reference:
            st.caption("No changed physical settings were recorded relative to the reference.")
        x, y, z = st.columns(3)
        x.metric("Peak voxel temperature", _metric(chosen["peak_temperature_c"], " °C", 4))
        y.metric("Selected print + cooldown", _metric(chosen["duration_s"], " s", 4))
        z.metric("Heat retained at end", _metric(chosen["retained_energy_j"], " J", 4))
        st.caption("Peak is the maximum resolved cell temperature during integration. Retained heat includes the simulated part, powder and plate. Duration includes this selected range and its cooldown.")
        for text in chosen["interpretation"]:
            st.write(text)
        if chosen["mean_index_weighted"]:
            st.caption("This single-build mean enthalpy index is weighted by scan coverage; preset comparison uses a separately saved unweighted exposed-part statistic. Do not mix these two mean statistics.")
        with st.expander("Temperature history at a fixed location", expanded=True):
            try:
                case_folder = Path(chosen["folder"])
                stamp = (case_folder/"manifest.json").stat().st_mtime_ns
                history = _saved_history(str(case_folder), stamp)
                points = pd.DataFrame(history["rows"])
                if points.empty:
                    st.info("No exposed-part temperature history was saved for this case.")
                else:
                    points = points.rename(columns={"time_s": "Physical time (s)",
                        "probe_temperature_c": "Fixed exposed-cell temperature (°C)",
                        "part_max_temperature_c": "Exposed-part spatial maximum (°C)"})
                    st.line_chart(points, x="Physical time (s)", y=["Fixed exposed-cell temperature (°C)", "Exposed-part spatial maximum (°C)"], width="stretch")
                    position = history["position_m"]
                    st.caption(f"Fixed cell at x={position[0]*1000:.3g}, y={position[1]*1000:.3g}, z={position[2]*1000:.3g} mm, selected near the final exposed-part centre. Showing {history['sampled_count']} of {history['available_count']} saved frames. No probe value is drawn before that cell is exposed.")
                    st.caption("These are actual stored samples. The connecting line is a reading aid; intermediate temperatures are not calculated here. Brief peaks between frames can be missed. The spatial maximum can move as new regions are scanned.")
                    if reference and chosen["id"] != reference["id"]:
                        try:
                            reference_folder = Path(reference["folder"])
                            reference_stamp = (reference_folder/"manifest.json").stat().st_mtime_ns
                            reference_history = _saved_history(str(reference_folder), reference_stamp)
                            if reference_history["rows"] and reference_history["position_m"] == history["position_m"]:
                                overlay = []
                                for label, saved_history in (("Selected: "+chosen["name"], history),
                                                             ("Reference: "+reference["name"], reference_history)):
                                    overlay.extend({"Physical time (s)": row["time_s"],
                                                    "Fixed-cell temperature (°C)": row["probe_temperature_c"],
                                                    "Preset": label} for row in saved_history["rows"]
                                                   if row["probe_temperature_c"] is not None)
                                if overlay:
                                    st.markdown("**Same location: selected preset versus reference**")
                                    st.line_chart(pd.DataFrame(overlay), x="Physical time (s)",
                                                  y="Fixed-cell temperature (°C)", color="Preset", width="stretch")
                                    st.caption("Both traces use the same stored cell coordinates and each case's own saved physical times. Samples are not resampled or aligned by frame number; speed and pause changes can shift when this cell is heated.")
                            else:
                                st.caption("A reference overlay is unavailable because the automatically selected exposed cells differ. Use matching coordinates in Build playback to compare a specific location.")
                        except (OSError, ValueError, KeyError, IndexError):
                            st.caption("The saved reference samples are unavailable for a same-location overlay.")
            except (OSError, ValueError, KeyError, IndexError) as exc:
                st.info(f"Temperature-history samples are unavailable: {exc}")
        with st.expander("Definitions and numerical checks"):
            st.write("Thermal difference is the relative RMS difference of the exposed-part mean, 95th percentile and maximum peak enthalpy indices, with a denominator floor of 0.02. It is a comparison screen; a score of 0.1 is not a 10% defect risk.")
            st.write("The enthalpy index measures excess heat relative to the heat needed to reach that alloy's liquidus from its own initial temperature. If preheat changes, the normalizer changes too; compare absolute temperatures and retained heat alongside the score.")
            if chosen["energy_error_fraction"] is not None:
                st.write(f"Saved energy-balance relative error: {chosen['energy_error_fraction']:.3g}. " + ("Passed the model's 2×10⁻⁶ check." if chosen["energy_check_passed"] else "Does not pass the model's 2×10⁻⁶ check."))
            else:
                st.write("No saved energy-balance check is available for this case.")
            st.write("Energy conservation checks numerical bookkeeping. It does not establish that the mesh, timing grouping, assumed inputs or physical model predict your machine accurately. Refine promising settings and compare measured tracks/parts before selecting a printer preset.")
            st.json(chosen["config"], expanded=False)
    if data["failures"]:
        with st.expander(f"Failed cases and exclusions ({len(data['failures'])})", expanded=not bool(cases)):
            for failure in data["failures"]:
                name = failure.get("preset_name") or failure.get("material") or "Case"
                st.write(f"**{name}:** {failure.get('error', 'This case failed and is excluded from ranking.')}")
    if data["uncertainty"]:
        with st.expander("Sensitivity to assumed optical coupling"):
            for material, result in data["uncertainty"].items():
                st.write(f"**{material}:** {result.get('interpretation', 'No interpretation was saved.')}")
                st.caption(result.get("omitted_uncertainty", ""))
    with st.expander("Saved study limitations"):
        for warning in data["warnings"] or ["Coarse whole-build temperatures do not resolve microscopic melt pools, keyhole stability, porosity or part quality."]:
            st.write("• "+str(warning))
        st.write("Numerical resolution assessed for this study: " + ("yes" if data["numerical_resolution_assessed"] else "no"))
    st.download_button("Download interpretation", insights_text(data), file_name=data["run_id"]+"-interpretation.txt",
                       mime="text/plain", key="download_insights_"+data["run_id"])
