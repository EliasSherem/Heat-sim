"""The application opens a single print workspace; specialist modes are optional."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_local_app_opens_one_simulation_workspace_without_top_level_navigation():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=45)
    assert not app.exception
    assert app.title[0].value == 'Tracklab'
    assert not app.tabs
    assert not app.sidebar.radio
    assert not any(item.label == 'Workspace' for item in app.radio)
    assert sum(button.label == 'Run simulation' for button in app.button) == 1
    material = next(item for item in app.selectbox if item.label == 'Material')
    assert material.options == ['17-4PH', '316L', 'Ti-6Al-4V', 'AlSi10Mg']
    assert any(item.label == 'Advanced settings' for item in app.expander)
    assert any(item.label == 'Diagnostics' for item in app.expander)
    assert app.get('iframe'), 'An existing imported job should show its geometry immediately'
