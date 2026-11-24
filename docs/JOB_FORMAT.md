# Offline job import and timing provenance

`heat_sim.job_import.load_job(path)` imports existing exposure paths. It neither imports printer-controller Python modules nor opens any hardware interface. `save_imported_job(job, folder)` saves compact arrays, a readable summary, hashes and an exact copy of the source. Importing or previewing a job never executes pickle objects or job-provided code.

## Normalized API and units

`ImportedJob` contains `name`, `source_path`, `source_sha256`, `format`, `layer_thickness_m`, `layers`, and `metadata`. Each `ImportedLayer` contains `index`, `z_m`, `vectors`, `process_ids`, `part_ids`, `path_ids`, and `metadata`.

`vectors` is a float64 array with six columns:

| Column | Meaning | Unit |
|---|---|---|
| 0, 1 | Laser-on vector start x, y | m |
| 2, 3 | Laser-on vector end x, y | m |
| 4 | Source power value provisionally interpreted as commanded power | W **assumed** |
| 5 | Marking speed | m/s |

The companion integer arrays have one row per vector. Consecutive vectors sharing a path ID belong to an imported polyline; each flat hatch pair has its own path ID. No connection between unrelated paths is added as a laser-on segment. `layer.mark_time_s` is the sum of vector lengths divided by marking speeds. It excludes all laser-off travel and delays.

`z_m` retains the source layer's base/offset convention. The thermal model's surface at `z_m + layer_thickness_m` is an explicitly stated model-coordinate convention, not a measurement of the physical build-plane height. A header declaring 54 mm and a final layer offset of 54 mm should not be relabelled as a measured 54.06 mm part. Empty layers remain present and still need the user's cooling/recoating-time assumptions.

## Legacy HTMJOB

The inspected `17-4 Test.htmjob` is a ZIP containing `para.head` and 901 `.epone` members. Layer filenames are `index * thickness`, where `thickness=6` is a filename increment and `thickness_real=0.06` is the layer height in provisionally interpreted millimetres. The filename increment is not 6 mm.

`para.head` is a primitive protocol-4 pickle. The importer accepts only an explicit data-opcode whitelist and uses an unpickler that rejects all class and persistent references. The format does not require object construction or `REDUCE`. ZIP members are read in place; paths are never extracted. Duplicate/unsafe names, encryption, excessive sizes, invalid counts, non-finite coordinates and truncated records are rejected.

Each `.epone` record is little endian:

```
uint16 part_id
uint16 process_id
uint32 point_count
point_count × (float32 x_mm, float32 y_mm)
```

The following interpretation is inferred from the read-only recovered controller source already on this Mac. This is a software-format interpretation, not a capture of a printer run:

- Processes 0, 1, 2 and 12 are flat jump/mark pairs. Their point lists are concatenated within each part/section in first-seen order, then paired. Odd point counts are rejected.
- Processes 3, 4, 5, 6, 7, 8 and 16 are polylines: jump to the first point, mark to each subsequent point. A polygon is not implicitly closed unless its supplied points close it.
- Processes 9, 10, 11, 13, 14 and 15 are reserved and skipped by the recovered runtime. Their raw point counts remain in diagnostics. They are not silently drawn as exposed material.
- Part and process groups execute in their first-seen group order in the recovered code. This can differ from simply replaying every binary record in archive order. The provenance label is therefore **inferred sequence**, not measured execution order.
- The imported `part.section` table supplies the flattened recipe actually used by that code. The richer property tree, disabled core flag and `contour.keep_same` flag are preserved. Explicit disabled core/outer-contour flags are respected. Property-tree settings do not overwrite the flattened section table silently.
- Imported operations apply translation, rotation around the header bounding-box centre, then axis scaling around that centre. Source point coordinates are already millimetres; the header `unit` scales the bounding-box centre, not the `.epone` coordinates again.
- Zero-length marking segments are omitted from the moving-source vector list and counted. Any stationary laser-on dwell at those points is unknown and is not fabricated.

The exact source power values remain vendor source units until calibrated. Treating them as watts is a visible, editable assumption. The source field `raduis` remains uninterpreted and never establishes beam size.

Local primary format evidence inspected without executing it:

- `/Users/elias/Downloads/Metal_printer/controller/htmjob.py`: bounded record decoder and process identities.
- `/Users/elias/Downloads/Metal_printer/tools/phase3_rtc4_call_recorder.py`: grouping, jump/mark selection, transformations and flattened section recipe interpretation.
- `/Users/elias/Downloads/Metal_printer/controller/slicing/package.py`, `layer_codec.py` and `processes.py`: portable MJOB schemas and stored path semantics.

These files are reference evidence, not runtime dependencies of Heat / Sim. This application does not import or run them.

## Portable MJOB

The importer supports recognized ZIP manifests with `format` equal to:

- `metal-compiled-job`: ordered layer references, either JSON paths or the `paths-f64le-v1` binary encoding. Binary layers start with `<4sIddI` (`MLY1`, index, z_mm, slice_z_mm, path_count). Each path has `<BBI` process, kind and point count, followed by float64 XY points. Supplied layer/source hashes are checked.
- `metal-slc-job` or `metal-cli-job`: existing `compiled.json` paths are read and checked against `compiled_sha256`. Original source-member hashes are checked when supplied. Source contour files are not re-sliced or converted into new hatch lines during import.

Stored path and point sequence is **imported**. Physical scanner scheduling, calibration, command quantization and execution delays are not captured by these packages. The imported preset determines each process recipe. A contour-only job remains contour-only; no interior exposure is invented to make a fuller-looking part.

The missing `/Users/elias/Downloads/job.mjob` was not substituted with a similarly named retained example. A separately inspected 557,688-byte retained `metal-slc-job` sample is a format example only; its identity as the user's newly supplied job has not been established.

## Supplemental explicit JSON

If another proprietary MJOB variant cannot be decoded, export actual vectors to this schema. Unknown variants fail with a diagnostic rather than generating a plausible-looking exposure path:

```json
{
  "format": "heat-sim-job-v1",
  "coordinate_units": "m",
  "speed_units": "m/s",
  "power_units": "W_assumed",
  "layer_thickness_m": 0.00006,
  "layers": [
    {
      "index": 0,
      "z_m": 0,
      "vectors": [[0, 0, 0.001, 0, 120, 1.4]],
      "process_ids": [0],
      "part_ids": [0],
      "path_ids": [0]
    }
  ],
  "metadata": {
    "sequence_basis": "Describe the actual export or ordering evidence here"
  }
}
```

The example illustrates the schema; its simple line is not substituted for any imported part. Explicit SI units are required. Layers must be ordered and uniformly spaced. Optional integer arrays default to infill, part 0 and separate vector paths; describe the source provenance in `metadata`.

## Timing limitations

Mark time is derived from imported length and source speed. Full elapsed build time additionally requires jump speed, laser on/off and scanner delays, list/section handoff delays, recoating/layer preparation, and any pauses or retries. These remain assumed/editable until supplied. A playback can preserve all imported vectors and persistent thermal state while still having an **assumed timeline**. No parser result establishes part density, melt-pool accuracy or a qualified printer recipe.

## Recorded actual HTMJOB import

`examples/actual-17-4-job-manifest.json` records the source SHA-256 and per-layer diagnostic summary. The canonical compact import is generated locally under `results/imports/17-4-actual` with:

```
.venv/bin/python -m heat_sim.job_import '/Users/elias/Downloads/Metal_printer/17-4 Test.htmjob' --output results/imports/17-4-actual
```

The inspected source has 901 layers, 866 populated layers, 3,216,686 raw points, and 1,595,099 nonzero active marking vectors. Process 16 supports account for 503,449 vectors, normal infill for 862,932, and the remaining active vectors are surface fills/contours. Reserved process 9 contains 159,516 points that the recovered runtime skips. 415,717 zero-length segments are counted but omitted from moving-source integration.

The derived laser-on marking time is 1,340.886 seconds (22.35 minutes). This is **not full build duration**. Layer 0 is support-only with 3,349 vectors; layer 1 is empty; layer 2 is support-only with 3,481 vectors. The early empty layers are preserved, not removed or replaced by invented geometry.
