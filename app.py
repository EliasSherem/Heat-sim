"""Tracklab's single print-simulation workspace, with optional expert tools."""
import runpy
import streamlit as st

st.set_page_config(page_title="Tracklab · Print simulation",page_icon="◈",layout="wide")
tool=st.query_params.get("tool")
if tool in ("single-track","studies"):
    st.link_button("← Back to print simulation","http://127.0.0.1:8501/")
    st.caption("Specialist numerical studies · separate from the whole-part simulation workspace")
    if tool=="single-track":runpy.run_module("heat_sim.single_track_ui",run_name="__main__")
    else:
        from heat_sim import job_ui
        job_ui.render()
else:
    from heat_sim.simulation_ui import render
    render()
