"""Local macOS laser single-track thermal laboratory."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import streamlit as st

from heat_sim.solver import SimulationConfig
from heat_sim.storage import ROOT, RESULTS, atomic_json, load_simulation, bundle_run, model_fingerprint
from heat_sim.plots import thermal_figure, sweep_figure
from heat_sim.materials import MATERIALS, MATERIAL_WARNINGS

st.set_page_config(page_title="Tracklab · alloy transfer", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem;max-width:1500px}
h1 {letter-spacing:-.055em;font-weight:650!important}
h2,h3 {letter-spacing:-.025em}
.eyebrow {color:#ff9a6e;letter-spacing:.19em;font-size:11px;font-weight:700;margin-bottom:8px}
.intro {color:#9caec5;max-width:860px;font-size:16px;line-height:1.6;margin-bottom:22px}
.stMetric {background:#192332;border:1px solid #2a384b;border-radius:12px;padding:16px}
.stMetric label {color:#9caec5}
div[data-testid="stSidebar"] {border-right:1px solid #293447}
div[data-testid="stTabs"] button {font-size:13px}
.pill {display:inline-block;border:1px solid #4d4238;color:#ffc598;padding:5px 10px;border-radius:20px;font-size:11px;margin-right:6px}
</style>""", unsafe_allow_html=True)

defaults = asdict(SimulationConfig())
loaded = st.session_state.get("loaded_config", {})
defaults.update(loaded)
generation = st.session_state.get("input_generation", 0)

def number(label, field, scale=1., low=None, high=None, step=1., **kwargs):
    value = float(defaults[field]) * scale
    return st.number_input(label, min_value=low, max_value=high, value=value, step=step,
                           key=f"input_{generation}_{field}", **kwargs) / scale


with st.sidebar:
    st.markdown("### ◈ Tracklab")
    st.caption("LOCAL THERMAL LABORATORY")
    with st.expander("Load a saved configuration"):
        upload = st.file_uploader("Configuration JSON", type=["json"])
        if st.button("Load configuration", disabled=upload is None):
            try:
                data = json.loads(upload.getvalue())
                incoming = data.get("config", data)
                SimulationConfig(**incoming).validate()
                st.session_state.loaded_config = incoming
                st.session_state.loaded_assumptions = data.get("assumptions", {})
                st.session_state.loaded_options = data.get("options", {})
                st.session_state.input_generation = generation + 1
                st.rerun()
            except Exception as exc:
                st.error(f"Cannot load configuration: {exc}")
    st.markdown("#### Baseline · main body infill")
    power = number("Commanded power (W, provisional)", "power_w", low=1., high=1000., step=5.)
    speed = number("Scan speed (mm/s, provisional)", "speed_m_s", scale=1000., low=100., high=10000., step=20.)
    a, b = st.columns(2)
    with a:
        layer = number("Layer (µm)", "layer_m", scale=1e6, low=20., high=120., step=10.)
    with b:
        hatch = number("Hatch (µm)", "hatch_m", scale=1e6, low=10., high=200., step=5.)
    mode = st.selectbox("Numerical resolution", ["coarse", "refined"], index=["coarse", "refined"].index(defaults["mode"]),
                        help="Coarse for exploration; refined for comparison. Check sensitivity before interpreting geometry.", key=f"mode_{generation}")
    with st.expander("Optics & delivered power · assumed", expanded=True):
        beam = number("Actual beam diameter (µm)", "beam_diameter_m", scale=1e6, low=40., high=200., step=5.,
                      help="Gaussian: full 1/e² intensity diameter. Top-hat: physical illuminated disk diameter. The job's raduis field is never used.")
        profile = st.selectbox("Beam profile", ["gaussian", "top_hat"], index=["gaussian", "top_hat"].index(defaults["profile"]), key=f"profile_{generation}")
        old_assumptions = st.session_state.get("loaded_assumptions", {})
        wavelength = st.number_input("Laser wavelength (nm)", 300., 12000., float(old_assumptions.get("wavelength_nm", 1070.)), 10., key=f"wave_{generation}")
        st.caption("Wavelength is recorded metadata. Effective absorptivity is supplied separately; no optical spectrum model is applied.")
        power_factor = number("Delivered / commanded power", "power_factor", low=.1, high=1.5, step=.01)
        old_options = st.session_state.get("loaded_options", {})
        abs17 = st.number_input("17-4PH effective absorptivity", .05, .90, float(old_options.get("absorptivity_17", defaults["absorptivity"])), .01, key=f"abs17_{generation}")
        abs316 = st.number_input("316L effective absorptivity", .05, .90, float(old_options.get("absorptivity_316", .35)), .01, key=f"abs316_{generation}")
        if st.session_state.get("fitted_absorptivity_17") is not None:
            st.caption(f"Last fitted 17-4PH absorptivity: {st.session_state.fitted_absorptivity_17:.3f}. Enter it above to apply;316L remains independent.")
    with st.expander("Powder, substrate & environment · assumed"):
        initial = st.number_input("Initial / build plate temperature (°C)", 20., 600., float(defaults["initial_k"]-273.15), 5., key=f"temp_{generation}")+273.15
        substrate = st.selectbox("Substrate material", ["17-4PH", "316L"], index=["17-4PH", "316L"].index(defaults["substrate"]), key=f"substrate_{generation}")
        packing = number("Powder packing fraction", "powder_packing", low=.3, high=.75, step=.01)
        powder_k = number("Unmelted powder conductivity (W/m·K)", "powder_k_w_mk", low=.1, high=8., step=.1)
        convection = number("Top convection coefficient (W/m²·K)", "convection_w_m2k", low=0., high=100., step=1.)
        emissivity = number("Top emissivity", "emissivity", low=0., high=1., step=.05)
        tracks = st.selectbox("Scan pattern", [1, 2], index=[1, 2].index(defaults["tracks"]), format_func=lambda n: "Single main-body track" if n==1 else "Two adjacent tracks (same direction)", key=f"tracks_{generation}")
        delay = number("Delay between adjacent tracks (µs)", "intertrack_delay_s", scale=1e6, low=0., high=5000., step=10.)
        st.caption("Substrate stays the same when changing powder alloy. Two-track results are thermal reheating screens only.")
    with st.expander("Numerical controls"):
        domain_scale = number("Domain size multiplier", "domain_scale", low=1., high=2., step=.25)
        dt_scale = number("Time-step multiplier", "dt_scale", low=.25, high=1., step=.25)
        cooldown = number("Cooling after scan (µs)", "cooldown_s", scale=1e6, low=0., high=2000., step=50.)
    known = st.multiselect("Inputs supported by your measurements", ["beam diameter", "beam profile", "wavelength", "delivered power", "17-4PH absorptivity", "316L absorptivity", "plate temperature", "substrate", "packing fraction", "powder conductivity"],
                           default=old_assumptions.get("measured_inputs", []), key=f"known_{generation}")
    st.caption("All unselected machine / powder inputs are ASSUMED. Job power and speed units remain provisional.")

config = SimulationConfig(power_w=power, speed_m_s=speed, layer_m=layer, hatch_m=hatch, mode=mode,
        beam_diameter_m=beam, profile=profile, power_factor=power_factor, absorptivity=abs17,
        initial_k=initial, substrate=substrate, powder_packing=packing, powder_k_w_mk=powder_k,
        convection_w_m2k=convection, emissivity=emissivity, tracks=tracks,
        intertrack_delay_s=delay, domain_scale=domain_scale, dt_scale=dt_scale, cooldown_s=cooldown)
assumptions = {"wavelength_nm": wavelength, "measured_inputs": known,
    "unmeasured_inputs_status": "ASSUMED", "power_unit": "W (provisional)", "speed_unit": "mm/s (provisional)",
    "raduis": "0.12, meaning unknown; not used", "demonstration_only": True,
    "beam_convention": "Gaussian full 1/e² intensity diameter; top-hat physical disk diameter",
    "substrate_comparison": "Same selected substrate for both powders"}

st.markdown('<div class="eyebrow">LASER POWDER BED FUSION / SINGLE-TRACK STUDY</div>', unsafe_allow_html=True)
st.title("From 17-4PH to 316L")
st.markdown('<div class="intro">Compare thermal melt footprints, test the assumptions, and narrow a bracket for physical trials. A moving heat source in a 3D powder-and-substrate domain, running on your Mac.</div>', unsafe_allow_html=True)
st.markdown('<span class="pill">CONDUCTION MODEL</span><span class="pill">ILLUSTRATIVE INPUTS</span><span class="pill">LOCALHOST ONLY</span>', unsafe_allow_html=True)
st.write("")
st.warning("Demonstration outputs are not printer settings. Unknown beam, delivered power, optical coupling and powder properties remain assumed. Numerical verification does not establish experimental validity.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("17-4PH baseline", f"{power:g} W", f"{speed*1000:,.0f} mm/s", delta_color="off")
c2.metric("Powder layer / hatch", f"{layer*1e6:g} / {hatch*1e6:g} µm")
c3.metric("Beam diameter · assumed" if "beam diameter" not in known else "Measured beam diameter", f"{beam*1e6:g} µm")
c4.metric("Delivered power · assumed" if "delivered power" not in known else "Delivered power", f"{power*power_factor:g} W")

RESULTS.mkdir(exist_ok=True)

def active_status():
    folder = st.session_state.get("active_folder")
    if not folder:
        return None, None
    folder = Path(folder)
    status_path = folder / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "queued", "progress": 0, "message": "Starting process"}
    return folder, status


folder, status = active_status()
busy = status is not None and status["state"] in ("queued", "running")

def start(kind, options=None):
    if busy:
        return
    folder = RESULTS / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    request = {"kind": kind, "config": asdict(config), "options": {"absorptivity_17": abs17, "absorptivity_316": abs316, **(options or {})},
               "assumptions": assumptions, "created_utc": datetime.now(timezone.utc).isoformat(), "model_fingerprint": model_fingerprint()}
    atomic_json(folder / "request.json", request)
    atomic_json(folder / "status.json", {"state": "queued", "progress": 0., "message": "Starting local CPU process"})
    with (folder / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "heat_sim.worker", str(folder / "request.json")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    st.session_state.active_folder = str(folder)
    st.session_state.active_pid = process.pid
    st.rerun()


@st.fragment(run_every="1s")
def progress_panel():
    folder, status = active_status()
    if status is None:
        return
    if status["state"] in ("running", "queued"):
        pid = st.session_state.get("active_pid")
        if pid and (not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE):
            atomic_json(folder / "status.json", {"state": "failed", "progress": 0, "message": "Worker exited unexpectedly; see worker.log"})
            st.rerun()
        st.progress(max(0., min(1., status["progress"])), text=status["message"])
        if st.button("Cancel after current solver step", key="cancel_job"):
            (folder / "CANCEL").touch()
            st.info("Cancellation requested. Finished simulations remain in the local cache.")
    else:
        st.session_state.selected_run = str(folder)
        st.session_state.last_status = status
        st.session_state.pop("active_folder", None)
        st.rerun()


progress_panel()
last_status = st.session_state.get("last_status")
if last_status and last_status["state"] in ("failed", "cancelled"):
    st.error(f"Last run {last_status['state']}: {last_status['message']}")

tabs = st.tabs(["Compare", "Thermal fields", "Power × speed", "Uncertainty", "Calibration", "Model & verification", "Job reference"])

report = None
run_folders = sorted([p.parent for p in (RESULTS / "runs").glob("*/report.json")], reverse=True)
with tabs[0]:
    st.subheader("A test bracket, with the baseline beside it")
    st.write("Run 17-4PH first, then 316L at the same machine settings. The exploratory bracket stays at 120 W / 1,260, 1,400 and 1,540 mm/s.")
    if st.button("Run baseline + 316L bracket", type="primary", disabled=busy):
        start("comparison")
    st.caption("Same chosen substrate for both alloys. Absorptivity is independent for each alloy. Cached identical simulations are reused.")
    if run_folders:
        current = Path(st.session_state.get("selected_run", str(run_folders[0])))
        index = run_folders.index(current) if current in run_folders else 0
        selected = st.selectbox("Saved local run", run_folders, index=index, format_func=lambda p: p.name)
        report = json.loads((selected / "report.json").read_text())
        st.caption(f"Saved workflow: {report['kind']} · source fingerprint {report.get('fingerprint', 'see request')[:12]}. Results reflect saved inputs; sidebar edits apply to the next run.")
        if report.get("rows"):
            frame = pd.DataFrame(report["rows"])
            show = [c for c in ["label", "scenario", "power_w", "speed_mm_s", "width_um", "depth_um", "penetration_um", "width_to_hatch", "geometry_distance", "energy_error_fraction", "runtime_s", "valid"] if c in frame]
            st.dataframe(frame[show], hide_index=True, width="stretch", column_config={
                "width_um": st.column_config.NumberColumn("Width (µm)", format="%.1f"),
                "depth_um": st.column_config.NumberColumn("Depth (µm)", format="%.1f"),
                "penetration_um": st.column_config.NumberColumn("Penetration (µm)", format="%.1f"),
                "width_to_hatch": st.column_config.NumberColumn("Width / hatch", format="%.2f"),
                "geometry_distance": st.column_config.NumberColumn("Geometry distance", format="%.3f"),
                "runtime_s": st.column_config.NumberColumn("Runtime (s)", format="%.2f"),
                "energy_error_fraction": st.column_config.NumberColumn("Energy closure", format="%.1e")})
            st.caption("Width/depth are a peak-over-time liquidus envelope at the central transverse section; depth starts at the original powder top. Penetration subtracts the layer thickness. Width/hatch >1 is an overlap screen and cannot establish lack-of-fusion avoidance. Two-track width is a combined footprint that may bridge an unmelted gap; its width/hatch ratio is suppressed.")
            warning_list = sorted({str(w) for r in report["rows"] for w in r.get("warnings", [])})
            if any(r["peak_temperature_k"] > 2900 for r in report["rows"]):
                st.warning("Saved cases exceed 2900 K, where omitted vaporisation, recoil and melt flow make the conduction geometry unreliable. Use these outputs to examine assumptions; they cannot justify printer settings.")
            if all(r["penetration_um"] <= 0 for r in report["rows"]):
                st.warning("No saved candidate predicts liquidus penetration through the powder layer into the substrate. Width / hatch alone cannot establish adequate fusion.")
            with st.expander("Numerical and material warnings", expanded=False):
                for warning in warning_list:
                    st.write("• " + warning)
        if report.get("failures"):
            st.error(f"{len(report['failures'])} failed simulations were recorded and excluded from plots.")
            st.json(report["failures"])
        if report["kind"] == "calibration":
            st.json(report["extra"])
            st.session_state.fitted_absorptivity_17 = report["extra"].get("fitted_absorptivity_17")
        columns = st.columns(3)
        if (selected / "comparison.csv").exists():
            columns[0].download_button("Download CSV", (selected / "comparison.csv").read_bytes(), "comparison.csv", "text/csv")
        columns[1].download_button("Configuration JSON", (selected / "request.json").read_bytes(), "configuration.json", "application/json")
        columns[2].download_button("Reproducible run ZIP", bundle_run(selected), f"{selected.name}.zip", "application/zip")
        with st.expander("Saved inputs, assumptions and run details"):
            st.json(json.loads((selected / "request.json").read_text()))
            st.json({k:v for k,v in report.items() if k not in ("rows", "assumptions")})
    else:
        st.info("Ready for the first run. All results will be saved under results/ on this Mac.")

with tabs[1]:
    st.subheader("Where the conduction model predicts melting")
    if report and report.get("rows"):
        options = list(range(len(report["rows"])))
        row_id = st.selectbox("View a saved simulation", options, format_func=lambda i: report["rows"][i]["label"]+" · "+report["rows"][i].get("scenario", "nominal"))
        row = report["rows"][row_id]
        meta, fields = load_simulation(row["cache_key"])
        fig = thermal_figure(meta, fields)
        st.pyplot(fig, width="stretch")
        image = io.BytesIO(); fig.savefig(image, format="png", dpi=180); plt.close(fig)
        st.download_button("Download this PNG", image.getvalue(), "melt-pool-sections.png", "image/png")
        st.caption("Each cell shows its maximum over time; this is a thermal envelope, not an instantaneous molten pool. Cyan denotes the liquidus of the powder alloy. The dashed line is the initial powder/substrate interface. A different substrate alloy may have a different liquidus; see model notes.")
        st.json(meta["metrics"], expanded=False)
    else:
        st.info("Run a comparison to view labelled temperature maps and liquidus sections.")

with tabs[2]:
    st.subheader("Sample a local power–speed window")
    st.write("Rank candidates by relative width/depth distance from the simulated 17-4PH baseline. This is geometry matching, not an optimal recipe or a defect prediction.")
    c1,c2,c3 = st.columns(3)
    p_bounds = c1.slider("Power bounds (W)", 40, 300, (100, 140), 5)
    v_bounds = c2.slider("Speed bounds (mm/s)", 300, 3000, (1200, 1600), 50)
    count = c3.selectbox("Points per axis", [3, 5, 7, 9])
    if st.button(f"Run {count*count}-point 316L sweep", disabled=busy):
        start("sweep", {"powers_w": np.linspace(*p_bounds, count).tolist(), "speeds_mm_s": np.linspace(*v_bounds, count).tolist()})
    if report and report["kind"] == "sweep" and report.get("rows"):
        metric = st.selectbox("Map colour", ["geometry_distance", "width_um", "depth_um"])
        fig = sweep_figure(pd.DataFrame([r for r in report["rows"] if r["alloy"] == "316L"]), metric)
        st.pyplot(fig); plt.close(fig)
    else:
        st.caption("Select a saved sweep in Compare to display its map.")

with tabs[3]:
    st.subheader("Does the bracket ranking survive the assumptions?")
    st.write("Vary beam size, each alloy's absorptivity and unmelted powder conductivity independently at the range endpoints. Nine scenarios require 36 simulations before caching.")
    a,b = st.columns(2)
    beam_bounds = a.slider("Beam diameter range (µm)", 40., 160., (60., 100.), 5.)
    k_bounds = b.slider("Powder conductivity range (W/m·K)", .1, 6., (.5, 2.), .1)
    alpha17_bounds = a.slider("17-4PH absorptivity range", .05, .9, (.25, .5), .01)
    alpha316_bounds = b.slider("316L absorptivity range", .05, .9, (.25, .5), .01)
    if st.button("Run uncertainty envelope", disabled=busy):
        start("uncertainty", {"bounds": {"beam_um": beam_bounds, "absorptivity_17": alpha17_bounds,
                "absorptivity_316": alpha316_bounds, "powder_k_w_mk": k_bounds}})
    if report and report["kind"] == "uncertainty":
        extra = report["extra"]
        st.dataframe(pd.DataFrame(extra["rankings"]), hide_index=True, width="stretch")
        if extra["rankable_scenarios"] < extra["scenario_count"]:
            st.warning("Some scenarios have no resolved liquidus pool or failed simulations; rankings are incomplete.")
        if extra["common_closest_speeds_mm_s"]:
            st.info(f"Common closest-geometry speed(s) across rankable endpoint scenarios: {extra['common_closest_speeds_mm_s']} mm/s. This does not establish suitability for printing.")
        else:
            st.warning("No common closest candidate across the rankable scenarios. Measure beam diameter and delivered power first, then effective optical coupling / melt-pool cross-sections for each alloy before narrowing the bracket.")
        st.caption(extra["interpretation"])
    st.caption("Input uncertainty: this endpoint study. Numerical error: mesh / time-step / domain checks. Model limitations: no flow, recoil, vaporisation, spatter or keyhole dynamics. Endpoint sweeps do not cover coupled extremes or provide statistical confidence intervals.")

with tabs[4]:
    st.subheader("Optional17-4PH geometry calibration")
    st.write("Use a successful 17-4PH single-track cross-section at these baseline settings. Fit only 17-4PH effective absorptivity within physical bounds while keeping the measured beam and delivered power fixed.")
    st.caption("Depth must be measured from the original powder top. If your measurement starts at the original substrate surface, add the layer thickness. A multi-track/full-layer section is not directly equivalent to this single-track envelope.")
    a,b,c = st.columns(3)
    measured_w = a.number_input("Measured width (µm)", 1., 1000., 100., 1.)
    measured_d = b.number_input("Measured total depth (µm)", 1., 1000., 80., 1.)
    absorption_bounds = c.slider("17-4PH fitting bounds", .05, .9, (.1, .8), .01)
    measured_confirmed = st.checkbox("I have entered actual 17-4PH width and depth measurements")
    if st.button("Fit 17-4PH absorptivity", disabled=busy or not measured_confirmed):
        start("calibration", {"measured_width_um": measured_w, "measured_depth_um": measured_d,
                              "absorption_bounds": absorption_bounds})
    st.caption("A geometry fit can absorb errors in conductivity, beam size, powder mass and omitted flow. It does not validate the model, and does not determine 316L absorptivity. Results include measured/predicted dimensions and residuals.")

with tabs[5]:
    st.subheader("Evidence before interpretation")
    verification_path = RESULTS / "verification.json"
    if not verification_path.exists():
        verification_path = ROOT / "examples" / "verification-m4.json"
    if verification_path.exists():
        st.caption(f"Recorded numerical checks: {verification_path.name}. See recorded configuration and hardware; results are specific to those cases.")
        verification_report = json.loads(verification_path.read_text())
        st.json(verification_report, expanded=False)
        sensitivity = verification_report.get("laser_sensitivity", {})
        if sensitivity.get("comparisons_to_coarse"):
            st.dataframe(pd.DataFrame(sensitivity["comparisons_to_coarse"]), hide_index=True)
        if sensitivity and not sensitivity.get("geometry_convergence_claimed", False):
            st.warning("Geometry convergence has not been established. Review the mesh changes before interpreting close candidate scores; energy conservation alone does not establish geometry accuracy.")
    else:
        st.warning("Numerical verification report has not been generated yet. Run the checks before interpreting geometry.")
    if st.button("Run numerical verification + sensitivity", disabled=busy):
        start("verification")
    st.caption("Verification includes known-solution conduction, Gaussian power, enthalpy/latent energy and representative mesh/time/domain comparisons. It is separate from validation against physical samples.")
    for name, label in [("MODEL.md", "Equations, geometry and limitations"), ("MATERIALS.md", "Thermophysical data and primary sources"), ("VERIFICATION.md", "Verification methods and measured performance")]:
        path = ROOT / "docs" / name
        if path.exists():
            with st.expander(label):
                st.markdown(path.read_text())
    with st.expander("Inspect the exact material interpolation tables"):
        for name, material in MATERIALS.items():
            st.markdown(f"**{name}** · density {material.rho:g} kg/m³ · solidus/liquidus {material.solidus:g}/{material.liquidus:g} K · fusion latent heat {material.latent/1000:g} kJ/kg")
            st.caption(material.notes)
            st.dataframe(pd.DataFrame(material.table()), hide_index=True)
        st.write(MATERIAL_WARNINGS)

with tabs[6]:
    st.subheader("Presets retained as reference")
    reference = json.loads((ROOT / "examples" / "job_reference.json").read_text())
    st.dataframe(pd.DataFrame(reference["presets"]), hide_index=True, width="stretch")
    st.info("Only main body infill is initially simulated. Contour keep_same=true may override separate surface contour values. The 380 W /888 mm/s core preset is DISABLED.")
    st.warning("raduis=0.12 has unknown meaning and is excluded from all beam calculations. Units in the vendor job are provisionally interpreted as W, mm/s and mm.")
    st.caption("The supplied job was inspected as inert data. This app has no printer connection, control or job-writing functions.")
    st.json({k: v for k, v in reference.items() if k not in ("presets", "raw_parameter_header")}, expanded=False)

st.divider()
st.caption("TRACKLAB / CPU thermal exploration · local files only · no qualified process recipe · no part-density or fatigue predictions")
