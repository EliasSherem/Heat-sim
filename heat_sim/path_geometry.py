"""Bounded-error display simplification of complete imported scan geometry.

Every enabled source event belongs to a retained span.  A span can replace
several connected scan segments, but cannot cross a jump, disabled event,
layer, or process region.  Returned indices always refer to original events.
"""
from __future__ import annotations

import math

import numpy as np


def _retained_vertices(points: np.ndarray, tolerance_mm: float) -> np.ndarray:
    """Iterative Douglas-Peucker simplification with finite-segment distance."""
    keep = [0, len(points) - 1]
    stack = [(0, len(points) - 1)]
    tolerance_squared = tolerance_mm * tolerance_mm
    while stack:
        first, last = stack.pop()
        if last - first <= 1:
            continue
        delta = points[last] - points[first]
        relative = points[first + 1:last] - points[first]
        squared_length = float(delta @ delta)
        if squared_length:
            along = np.clip((relative @ delta) / squared_length, 0.0, 1.0)
            offset = relative - along[:, None] * delta
        else:
            offset = relative
        distances = np.einsum("ij,ij->i", offset, offset)
        farthest = int(np.argmax(distances))
        if distances[farthest] > tolerance_squared:
            index = first + 1 + farthest
            keep.append(index)
            stack.extend(((first, index), (index, last)))
    return np.asarray(sorted(keep), dtype=np.int64)


def simplify_scan_paths(start_mm, end_mm, layer, section, t0=None, t1=None,
                        enabled=None, tolerance_mm=0.02, max_span_s=0.025):
    """Return original start/end event indices for complete display geometry.

    Input coordinates have shape ``(N, 3)`` and use millimetres.  The other
    arrays have length N.  The output segment endpoints are
    ``start_mm[start_indices]`` and ``end_mm[end_indices]``; its actual finish
    time is ``t1[end_indices]``.  Non-scan, laser-off and jump events must be
    marked disabled.  They remain boundaries even when their coordinates
    coincide with their neighbours.

    Only exactly joined endpoints merge.  Optional physical times must also
    be continuous within 1 ns.  Each merged span lasts at most max_span_s;
    an original single event longer than this is retained unchanged.  With
    times omitted, only the geometric error bound applies.  Endpoints and
    every disconnected chain are retained, without a global display budget.
    """
    starts = np.asarray(start_mm, dtype=float)
    ends = np.asarray(end_mm, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 3 or ends.shape != starts.shape:
        raise ValueError("Scan endpoints must have matching (N, 3) shapes")
    count = len(starts)
    layers, sections = np.asarray(layer), np.asarray(section)
    on = np.ones(count, dtype=bool) if enabled is None else np.asarray(enabled, dtype=bool)
    if any(values.shape != (count,) for values in (layers, sections, on)):
        raise ValueError("Scan metadata must have one value per event")
    tolerance_mm = float(tolerance_mm)
    if not math.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError("Display tolerance must be finite and nonnegative")
    if max_span_s is not None:
        max_span_s = float(max_span_s)
        if not math.isfinite(max_span_s) or max_span_s <= 0:
            raise ValueError("Maximum display span must be finite and positive")
    if not np.all(np.isfinite(starts[on])) or not np.all(np.isfinite(ends[on])):
        raise ValueError("Enabled scan coordinates must be finite")
    if (t0 is None) != (t1 is None):
        raise ValueError("Both scan start and end times are required together")
    times0 = times1 = None
    if t0 is not None:
        times0, times1 = np.asarray(t0, dtype=float), np.asarray(t1, dtype=float)
        if times0.shape != (count,) or times1.shape != (count,):
            raise ValueError("Scan times must have one value per event")
        if (not np.all(np.isfinite(times0[on])) or not np.all(np.isfinite(times1[on]))
                or np.any(times1[on] < times0[on])):
            raise ValueError("Enabled scan times must be finite and ordered")
    if not count or not np.any(on):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    joined = (on[:-1] & on[1:] & (layers[:-1] == layers[1:])
              & (sections[:-1] == sections[1:])
              & np.all(ends[:-1] == starts[1:], axis=1))
    if times0 is not None:
        joined &= np.abs(times0[1:] - times1[:-1]) <= 1e-9
    chain_starts = np.flatnonzero(on & np.r_[True, ~joined])
    chain_ends = np.flatnonzero(on & np.r_[~joined, True])
    capacity = int(np.count_nonzero(on))
    result_starts, result_ends = np.empty(capacity, np.int64), np.empty(capacity, np.int64)
    written = 0
    for chain_first, chain_last in zip(chain_starts, chain_ends):
        first = int(chain_first)
        chain_last = int(chain_last)
        while first <= chain_last:
            last = chain_last
            if times0 is not None and max_span_s is not None:
                # A connected chain has nondecreasing times, so this also
                # avoids adding an O(N) search for every retained segment.
                within = int(np.searchsorted(times1[first:chain_last + 1],
                                             times0[first] + max_span_s, side="right"))
                last = first + max(1, within) - 1
            if first == last:
                result_starts[written], result_ends[written] = first, last
                written += 1
            else:
                points = np.concatenate((starts[first:first + 1], ends[first:last + 1]))
                vertices = _retained_vertices(points, tolerance_mm)
                length = len(vertices) - 1
                result_starts[written:written + length] = first + vertices[:-1]
                result_ends[written:written + length] = first + vertices[1:] - 1
                written += length
            first = last + 1
    return result_starts[:written].copy(), result_ends[:written].copy()
