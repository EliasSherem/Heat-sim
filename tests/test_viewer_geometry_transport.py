"""Large geometry transport preserves solver identities and immutable content."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from heat_sim.viewer_geometry_transport import (
    PATH_RECORD_BYTES, PATH_RECORD_DTYPE, pack_path_geometry, publish_path_geometry,
)


def columns():
    return (np.array([.000000123456789, 2671.401546450745]),
            np.array([[51.96234567, 44.92245678, .03], [0., 1234.123456, 21.75]]),
            np.array([[51.96298765, 45.90012345, .03], [0., 1234.987654, 21.75]]),
            np.array([1733, 2**32 - 1], dtype=np.int64), np.array([True, False]))


def test_records_keep_exact_times_cells_support_and_precise_coordinates():
    source = columns()
    payload = pack_path_geometry(*source)
    assert len(payload) == 2 * PATH_RECORD_BYTES == 80
    decoded = np.frombuffer(payload, dtype=PATH_RECORD_DTYPE)
    assert np.array_equal(decoded["t"], source[0])
    assert np.array_equal(decoded["cell"], source[3])
    assert np.array_equal(decoded["support"], source[4])
    np.testing.assert_allclose(decoded["p"], source[1], rtol=0, atol=.00006)
    np.testing.assert_allclose(decoded["q"], source[2], rtol=0, atol=.00006)
    assert {name: PATH_RECORD_DTYPE.fields[name][1] for name in PATH_RECORD_DTYPE.names} == {
        "t": 0, "p": 8, "q": 20, "cell": 32, "support": 36}


def test_identical_geometry_is_not_rewritten_and_changed_times_get_new_asset(tmp_path):
    source = columns()
    descriptor = publish_path_geometry(*source, static_dir=tmp_path)
    target = tmp_path / descriptor["url"].split("/")[-1]
    modified = target.stat().st_mtime_ns
    assert descriptor["format"] == "tracklab-paths-v1"
    assert descriptor["count"] == 2 and descriptor["record_bytes"] == 40
    assert publish_path_geometry(*source, static_dir=tmp_path) == descriptor
    assert target.stat().st_mtime_ns == modified
    changed = (source[0] + 1, *source[1:])
    assert publish_path_geometry(*changed, static_dir=tmp_path)["sha256"] != descriptor["sha256"]
    assert target.read_bytes() == pack_path_geometry(*source)


def test_concurrent_publishers_leave_one_complete_file_and_no_temporary_files(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        descriptors = list(pool.map(lambda _: publish_path_geometry(*columns(), static_dir=tmp_path), range(12)))
    assert all(descriptor == descriptors[0] for descriptor in descriptors)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].read_bytes() == pack_path_geometry(*columns())


@pytest.mark.parametrize("column,value", [(0, [float('nan'), 1.]), (1, [[0., 0., 0.]]),
                                         (3, [-1, 0]), (3, [0., 1.]), (4, [False, 2])])
def test_invalid_geometry_does_not_create_partial_assets(tmp_path, column, value):
    source = list(columns())
    source[column] = value
    with pytest.raises(ValueError):
        publish_path_geometry(*source, static_dir=tmp_path)
    assert not list(tmp_path.iterdir())
