import numpy as np
import pytest

from heat_sim.path_geometry import simplify_scan_paths


def endpoints(points):
    points = np.asarray(points, dtype=float)
    return points[:-1], points[1:]


def test_simplification_retains_every_connected_chain_and_bounds_geometric_error():
    x = np.linspace(0, 8, 81)
    points = np.column_stack((x, 0.07 * np.sin(x * 2), np.zeros(len(x))))
    starts, ends = endpoints(points)
    # An isolated, tiny support must survive even beside a much larger part.
    starts = np.vstack((starts, [20, 0, 1]))
    ends = np.vstack((ends, [20.0001, 0, 1]))
    sections = np.r_[np.zeros(80, dtype=int), 16]
    before = starts.copy(), ends.copy()
    a, b = simplify_scan_paths(starts, ends, np.zeros(81), sections, tolerance_mm=0.02)
    assert len(a) < 81
    assert (a[-1], b[-1]) == (80, 80)
    assert a[0] == 0 and b[-2] == 79
    assert np.array_equal(np.concatenate([np.arange(i, j + 1) for i, j in zip(a, b)]),
                          np.arange(81))
    for first, last in zip(a, b):
        original = np.vstack((starts[first:last + 1], ends[first:last + 1]))
        delta = ends[last] - starts[first]
        offset = original - starts[first]
        along = np.clip(offset @ delta / (delta @ delta), 0, 1)
        error = np.linalg.norm(offset - along[:, None] * delta, axis=1)
        assert error.max() <= 0.02 + 1e-12
    assert np.array_equal(starts, before[0]) and np.array_equal(ends, before[1])


def test_jumps_disabled_events_layers_and_sections_remain_chain_boundaries():
    starts, ends = endpoints([[x, 0, 0] for x in range(10)])
    # Disabled event 2 is a real gap; event 5 has a disconnected start.
    starts = starts.copy()
    starts[5, 0] = 5.1
    enabled = np.ones(9, dtype=bool)
    enabled[2] = False
    layers = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    sections = np.array([0, 0, 0, 0, 0, 0, 16, 16, 16])
    a, b = simplify_scan_paths(starts, ends, layers, sections, enabled=enabled)
    assert list(zip(a, b)) == [(0, 1), (3, 3), (4, 4), (5, 5), (6, 8)]
    assert np.array_equal(np.concatenate([np.arange(i, j + 1) for i, j in zip(a, b)]),
                          np.flatnonzero(enabled))


def test_temporal_span_is_bounded_without_splitting_original_scans():
    starts, ends = endpoints([[x, 0, 0] for x in range(8)])
    t0 = np.array([0, .01, .02, .03, .04, .05, .1])
    t1 = np.array([.01, .02, .03, .04, .05, .06, .2])
    a, b = simplify_scan_paths(starts, ends, np.zeros(7), np.full(7, 16), t0, t1)
    assert list(zip(a, b)) == [(0, 1), (2, 3), (4, 5), (6, 6)]
    assert np.all((t1[b] - t0[a] <= .025) | (a == b))
    assert (a[-1], b[-1]) == (6, 6)


def test_zero_distance_disabled_jump_and_off_time_still_break_paths():
    a, b = simplify_scan_paths([[0, 0, 0], [1, 0, 0], [1, 0, 0]],
                               [[1, 0, 0], [1, 0, 0], [2, 0, 0]],
                               [0, 0, 0], [16, 16, 16], enabled=[True, False, True])
    assert list(zip(a, b)) == [(0, 0), (2, 2)]
    starts, ends = endpoints([[x, 0, 0] for x in range(5)])
    a, b = simplify_scan_paths(starts, ends, [0] * 4, [16] * 4,
                               [0, .005, .03, .035], [.005, .01, .035, .04],
                               max_span_s=1)
    assert list(zip(a, b)) == [(0, 1), (2, 3)]


def test_zero_length_support_and_retraced_paths_are_retained():
    # Finite-segment distance prevents losing an excursion beyond endpoints.
    starts, ends = endpoints([[0, 0, 0], [1, 0, 0], [.5, 0, 0]])
    starts = np.vstack((starts, [8, 0, 0]))
    ends = np.vstack((ends, [8, 0, 0]))
    a, b = simplify_scan_paths(starts, ends, np.zeros(3), np.full(3, 16))
    assert list(zip(a, b)) == [(0, 0), (1, 1), (2, 2)]


def test_empty_and_all_disabled_paths_have_no_display_segments():
    a, b = simplify_scan_paths(np.empty((0, 3)), np.empty((0, 3)), [], [])
    assert a.dtype == b.dtype == np.int64 and not len(a) and not len(b)
    a, b = simplify_scan_paths([[0, 0, 0]], [[1, 0, 0]], [0], [16], enabled=[False])
    assert not len(a) and not len(b)


@pytest.mark.parametrize("kwargs", [{"tolerance_mm": -1}, {"max_span_s": 0},
                                    {"t0": [0]}, {"t0": [1], "t1": [0]}])
def test_invalid_display_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        simplify_scan_paths([[0, 0, 0]], [[1, 0, 0]], [0], [16], **kwargs)
