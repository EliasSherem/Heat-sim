"""Content-addressed local transport for complete, large scan geometry.

The browser reads a small descriptor from its iframe and fetches this immutable
binary once.  Keeping geometry out of the iframe's JSON avoids repeatedly
embedding hundreds of thousands of paths when live temperature fields update.
All times and thermal-cell identities are retained exactly. Coordinates are
Float32 millimetres (far finer than the display curve simplification tolerance).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np


PATH_RECORD_DTYPE = np.dtype([
    ("t", "<f8"),
    ("p", "<f4", (3,)),
    ("q", "<f4", (3,)),
    ("cell", "<u4"),
    ("support", "<u4"),
])
PATH_RECORD_BYTES = PATH_RECORD_DTYPE.itemsize
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent.parent / "static" / "viewer"


def pack_path_geometry(times, starts_mm, ends_mm, cells, support):
    """Return little-endian 40-byte records in original path order.

    Offsets: time Float64 at 0; start XYZ Float32 at 8,12,16; end XYZ
    Float32 at 20,24,28; thermal cell Uint32 at 32; support Uint32 at 36.
    The record layout deliberately aligns both Float32 and Uint32 views.
    """
    times = np.asarray(times, dtype=np.float64)
    starts = np.asarray(starts_mm, dtype=np.float64)
    ends = np.asarray(ends_mm, dtype=np.float64)
    cell_values = np.asarray(cells)
    supports = np.asarray(support)
    count = len(times)
    if times.shape != (count,) or starts.shape != (count, 3) or ends.shape != (count, 3):
        raise ValueError("Path times and XYZ coordinates must have matching lengths")
    if cell_values.shape != (count,) or supports.shape != (count,):
        raise ValueError("Path cells and support flags must have matching lengths")
    if not all(np.isfinite(values).all() for values in (times, starts, ends)):
        raise ValueError("Path times and coordinates must be finite")
    if (not np.issubdtype(cell_values.dtype, np.integer)
            or np.any(cell_values < 0) or np.any(cell_values > np.iinfo(np.uint32).max)):
        raise ValueError("Thermal cell indices must be unsigned 32-bit integers")
    if not np.isin(supports, [False, True]).all():
        raise ValueError("Support flags must be boolean")
    if np.any(abs(starts) > np.finfo(np.float32).max) or np.any(abs(ends) > np.finfo(np.float32).max):
        raise ValueError("Display coordinates exceed Float32 range")
    records = np.empty(count, dtype=PATH_RECORD_DTYPE)
    records["t"] = times
    records["p"] = starts
    records["q"] = ends
    records["cell"] = cell_values
    records["support"] = supports
    return records.tobytes()


def publish_path_geometry(times, starts_mm, ends_mm, cells, support, *, static_dir=None):
    """Atomically publish a local binary and return its browser descriptor.

    Identical geometry has an identical filename and is never rewritten.
    The default directory is served only by the app's existing localhost server.
    ``static_dir`` is provided for isolated tests or equivalent app installations.
    """
    payload = pack_path_geometry(times, starts_mm, ends_mm, cells, support)
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(static_dir) if static_dir is not None else DEFAULT_STATIC_DIR
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (digest + ".bin")
    if not destination.is_file() or destination.stat().st_size != len(payload):
        # A unique temporary file also makes concurrent material comparisons safe.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".paths-", suffix=".tmp", dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(url="/app/static/viewer/" + destination.name,
                count=len(times), record_bytes=PATH_RECORD_BYTES,
                format="tracklab-paths-v1", sha256=digest)
