"""Tracklab: local job-driven material-transfer workspace."""
import runpy
import streamlit as st

st.set_page_config(page_title="Tracklab · job material transfer", page_icon="◈", layout="wide")
with st.sidebar:
    workspace = st.radio("Workspace", ["Job material transfer", "Single-track laboratory"], key="workspace")
if workspace == "Single-track laboratory":
    runpy.run_module("heat_sim.single_track_ui", run_name="__main__")
else:
    from heat_sim import job_ui
    # Refresh the UI once in an already running local server while preserving
    # browser session inputs; numerical workers always import their own source.
    if getattr(job_ui,"UI_REVISION",None)!="live-full-playback-5":
        import importlib
        importlib.reload(job_ui)
    job_ui.render()
