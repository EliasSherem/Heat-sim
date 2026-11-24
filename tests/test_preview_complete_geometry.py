"""The neutral preview must retain every imported disconnected path."""
from pathlib import Path

import numpy as np

from heat_sim.job_import import ImportedJob, ImportedLayer
from heat_sim.viewer_geometry_transport import PATH_RECORD_DTYPE, publish_path_geometry
from heat_sim.workspace_viewer import geometry_preview_data


def test_default_preview_keeps_paths_beyond_previous_budget_and_all_support_layers(tmp_path, monkeypatch):
    monkeypatch.setattr("heat_sim.workspace_viewer.publish_path_geometry",
                        lambda *args: publish_path_geometry(*args, static_dir=tmp_path))
    layers = []
    for index in range(3):
        count = 15001
        x = np.arange(count) * .000001
        vectors = np.column_stack((x, np.full(count, .001), x + .0000001,
                                   np.full(count, .0010001), np.full(count, 100.), np.ones(count)))
        sections = np.zeros(count, dtype=int)
        sections[-1] = 16
        layers.append(ImportedLayer(index, index * .00003, vectors, sections,
                                    np.zeros(count, int), np.arange(count)))
    job = ImportedJob("many-disconnected.mjob", "unchanged.mjob", "complete-source",
                      "test", .00003, layers, {})
    original = [layer.vectors.copy() for layer in layers]
    data = geometry_preview_data(job)
    scene = data["cases"][0]
    assert scene["geometry_coverage"] == "complete"
    assert scene["source_segments"] == scene["displayed_segments"] == 45003
    assert scene["segments"] == [] and scene["source_layers"] == [0, 1, 2]
    descriptor = scene["path_geometry"]
    records = np.fromfile(tmp_path / Path(descriptor["url"]).name, dtype=PATH_RECORD_DTYPE)
    assert len(records) == 45003 and np.array_equal(np.unique(records["t"]), [0, 1, 2])
    assert records["support"].sum() == 3
    assert np.array_equal(records["cell"][records["support"] == 1], [16, 16, 16])
    assert np.allclose(records["p"][0], [0, 1, .03], atol=1e-6)
    assert np.allclose(records["q"][-1], [15.0001, 1.0001, .09], atol=1e-6)
    for layer, before in zip(layers, original):
        np.testing.assert_array_equal(layer.vectors, before)


def test_complete_preview_only_simplifies_connected_chains_and_reports_source_count():
    vectors = np.array([[0, 0, .001, 0, 100, 1], [.001, 0, .002, 0, 100, 1],
                        [.01, 0, .011, 0, 100, 1], [.011, 0, .012, 0, 100, 1]])
    layer = ImportedLayer(7, .00021, vectors, np.array([0, 0, 16, 16]),
                          np.zeros(4, int), np.arange(4))
    job = ImportedJob("joined.mjob", "unchanged.mjob", "joined-source", "test", .00003, [layer], {})
    scene = geometry_preview_data(job)["cases"][0]
    assert scene["geometry_coverage"] == "complete"
    assert scene["source_segments"] == 4 and scene["displayed_segments"] == 2
    segments = scene["segments"]
    assert np.allclose(np.asarray(segments)[:, :6], [[0, 0, .24, 2, 0, .24],
                                                   [10, 0, .24, 12, 0, .24]])
    assert [segment[6:] for segment in segments] == [[0, 7], [16, 7]]
