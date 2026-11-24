"""A hidden support display must remain a reversible visual choice."""
import json

import numpy as np
import pytest

from heat_sim.workspace_viewer import thermal_workspace_html
from test_workspace_viewer import case_fixture, execute_viewer


@pytest.mark.parametrize("format", ["json", "npz"])
def test_support_paths_render_when_enabled_and_restore_without_recalculating(tmp_path, format):
    case = case_fixture(tmp_path / format)
    events = [
        dict(t0_s=0., t1_s=.5, start_m=[.00025, .00025, .00025],
             end_m=[.00025, .00075, .00025], laser_on=True,
             power_w=80., kind="scan", section=16, layer_index=0),
        dict(t0_s=.5, t1_s=1., start_m=[.00075, .00075, .00025],
             end_m=[.00075, .00025, .00025], laser_on=True,
             power_w=120., kind="scan", section=0, layer_index=0),
        dict(t0_s=3., t1_s=4., start_m=[.00025, .00025, .00025],
             end_m=[.00075, .00025, .00025], laser_on=True,
             power_w=80., kind="scan", section=16, layer_index=1),
    ]
    timeline = case["folder"] / ("timeline." + format)
    if format == "json":
        timeline.write_text(json.dumps(events))
    else:
        dtype = np.dtype([("t0_s", "f8"), ("t1_s", "f8"),
                          ("start_m", "f8", (3,)), ("end_m", "f8", (3,)),
                          ("kind", "u1"), ("power_w", "f8"),
                          ("section", "i2"), ("layer_index", "i4")])
        values = np.zeros(len(events), dtype=dtype)
        for i, event in enumerate(events):
            for key in values.dtype.names:
                values[key][i] = 1 if key == "kind" else event[key]
        np.savez_compressed(timeline, events=values)
    case["manifest"].update(timeline_file=timeline.name, timeline_event_count=len(events))
    result = execute_viewer(thermal_workspace_html([case]), """
const before=JSON.stringify(scenes[0].frames);
$('supports').checked=true;draw();
const enabled=picks[0].filter(p=>p.line).length;
$('supports').checked=false;$('supports').onchange();
const hidden=picks[0].filter(p=>p.line).length;
$('supports').checked=true;$('supports').onchange();
({enabled,hidden,restored:picks[0].filter(p=>p.line).length,
  unchanged:before===JSON.stringify(scenes[0].frames),
  sourceSupportPaths:scenes[0].paths.filter(p=>p.support).length})
""")["result"]
    assert result == dict(enabled=2, hidden=1, restored=2,
                          unchanged=True, sourceSupportPaths=2)
