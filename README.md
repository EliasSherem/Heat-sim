# Tracklab — local print simulation

A working Python / Streamlit application for comparing **17-4PH, 316L, Ti-6Al-4V and AlSi10Mg** using imported print jobs and a 3D moving-source enthalpy model. Runs on the Mac CPU and binds to **127.0.0.1:8501**. There is no printer interface, cloud compute, CUDA, or paid dependency.

This is an exploratory engineering tool. Its outputs are **not a qualified process recipe**, a defect prediction, or a guarantee of density, fatigue strength or part quality. The illustrative machine inputs need measurement before selecting printer settings.

## One print-simulation workspace

The default screen keeps print setup beside one large viewport. Import or select an `.htmjob` or `.mjob`, choose **Material**, adjust **Power · % of job** and **Speed · % of job**, then use the single **Run simulation** action. Imported per-region recipes supply the starting powers and speeds; percentages preserve their differences. Filename, job-coordinate dimensions and source-layer count stay compact. Unsupported variants fail explicitly; original job bytes are preserved.

After import the viewport immediately shows **Geometry preview**, with neutral styling from actual imported scan paths. This is not a thermal result. Running the calculation progressively replaces it with published solver temperatures in the same viewport. The geometry view projects coarse calculated cell temperatures onto complete source scan geometry. Connected curves are simplified within a 0.02 mm display tolerance while retaining every disconnected chain, layer and support region; there is no default path-count cutoff. Large geometry is cached as an immutable local asset and fetched by the browser from the same local server. The thermal-grid view shows the computed cells directly. Neither display increases the spatial accuracy of the thermal model.

Play/pause, restart, physical-time scrubbing, playback speed, source layer, 3D/layer view and the labelled temperature scale sit with the viewport. Rotate by dragging and zoom with the wheel. A point readout uses the selected stored cell temperature; its optional graph shows actual saved samples. Playback speed changes viewing time only. Camera position persists when frames or settings update.

The default colour range fits the current calculated temperatures rather than using a fixed 20–1800 °C span. It stays locked during completed playback, with one shared range for both comparison materials. **Fit colours** refits the selected moment; temperatures outside a locked range use its endpoint colours and are identified. Following a live calculation can update the shared range as new fields arrive. Choose **Custom fixed range** in Diagnostics for explicit bounds. Colour-range changes leave all solver temperatures unchanged.

Imported supports are shown by default. The **Supports** checkbox sits beside the main viewport controls, so hiding or restoring them does not require opening View options. Visibility changes leave support geometry and thermal calculations unchanged.

During a calculation, solver progress and **Cancel** appear beside the print. The view follows the latest calculated field; pause or scrub only among available frames. Failed or stopped runs retain their completed fields and show a useful status. A complete full-source result spans every saved frame, rather than silently zooming into its first layer. Unexposed powder starts hidden so it does not obscure the part.

Changing material, power, speed, beam/coupling or other physical/numerical settings marks the old result **outdated** and returns the viewport to geometry preview. Saved fields are never relabelled as belonging to the new settings. Diagnostics retains the earlier simulations. **Load saved run settings** restores a run's frozen configuration before showing its temperatures as current.

### Optional material comparison

Enable **Compare material** beside the viewport and choose a second grade. The same **Run simulation** action launches two independent thermal solves of the imported job with the same process settings, grid, timing assumptions and substrate, plus independent alloy properties and optical coupling. The result uses side-by-side scenes driven by one physical clock, one camera and one shared temperature scale. Each scene selects its stored field by physical time, not by frame number. A material whose first frame is still being calculated is labelled pending; another alloy's temperatures are never substituted for it.

### Advanced and Diagnostics

**Advanced settings** contains beam diameter, independent absorptivity, preheat, substrate, the between-layer pause, powder and motion assumptions, region recipe replacements, boundary properties, selected layer ranges, refined local regions, numerical controls and optional aligned STL. The compact assumed-input notice beside Run links directly to this section. The approximately **3-second pause including recoating** is the user's editable estimate, not a captured machine timestamp.

Specialist **Local track laboratory** and **Detailed preset studies** links live in Advanced. They retain the earlier calibration, uncertainty, preset and transfer tools without making them part of ordinary import → material → run → view use. The laboratory is a separate numerical configuration, not a second mandatory workspace mode.

**Diagnostics** contains vector counts, region tables, import/format provenance, property tables and sources, saved-run selection, detailed exports, result interpretation, convergence/planning information and worker errors. [The physics connection audit](docs/WORKSPACE_AUDIT.md) documents what reaches the solver and rendered fields. [The preset guide](docs/PRESETS.md) describes the specialist named-preset workflow.

The whole-part model retains enthalpy across tracks and layers, uses temperature-dependent bulk properties and fusion enthalpy, activates powder at preheat, conducts through active material and substrate, and cools during laser-off events. Its consolidation state is a subgrid conductivity/history proxy. Coarse full-build temperatures do not resolve microscopic melt pools, melt flow, keyhole instability, porosity or qualified print quality. A cropped refined region also omits surrounding heat transfer. Numerical energy checks do not establish mesh independence or experimental validity.

### Actual-job workspace verification

A continuous **0–109 layer prefix** of the available 725-layer Dachshund job was calculated separately in **17-4PH and 316L** using the new main workspace. The prefix contains **428,955 imported vectors** and all seven used regions. Both cases saved 80 calculated fields through **398.995 seconds of conditional physical time**; the worker finished both in **77.99 seconds** on this Mac. Their stored exposure schedules are identical, while their final temperature fields differ by up to **7.331 K**, with the same source settings and assumed absorptivity. Relative energy-ledger errors are approximately **2.48×10⁻¹⁴ and 2.73×10⁻¹⁴**.

Independent inspection checked the retained source hashes, every selected scan endpoint/order/region power/speed, vector-derived timing, chosen material tables, and every viewport temperature sample against its NPZ field. A separate cancelled attempt retained **67 actual published frames**. See [the numeric verification record](examples/workspace-verification.json) and [the connection audit](docs/WORKSPACE_AUDIT.md). This verifies a representative selected prefix and the data connection, not a full-job or mesh-converged prediction. The first 100 source layers contain only support-like exposures; body regions enter at layer 100.

### Reproduce a real-job example

```sh
.venv/bin/python -m heat_sim.example '/Users/elias/Downloads/Metal_printer/17-4 Test.htmjob' --output results/builds/my-first-study --layers 3 --all-materials --run
```

Choose a new output directory to preserve prior evidence. This freezes the real source job, normalized arrays, numerical source, configuration, computed frames, case metrics and comparison CSV. In the app, prepare the study ZIP and follow its `REPRODUCE.txt`. Numerical source/configuration are local; no printer connection or cloud calculation is involved.

`examples/actual-17-4-job-manifest.json` records the available real job: 901 source layers and 1,595,099 active nonzero vectors. The newly referenced `/Users/elias/Downloads/job.mjob` became unavailable during inspection; an unrelated format fixture was used only to test MJOB decoding and is not presented as that missing job.

### Resolution, performance and credibility

The entire available 901-layer / 1,595,099-vector 17-4 job completed on this M4 Mac in **10 min 46 s**, with **760 MiB external maximum RSS** and 60 saved 3D fields. See `examples/full-build-benchmark-m4.json`. This coarse run uses 2 mm lateral / 1.5 mm vertical cells, 100 ms exposure groups, the user-estimated 3 s inter-layer pause and assumed jump timing. Its calculated physical time is 86 min 19 s; that is conditional model timing, not a measured machine duration. `examples/full-build-grouping-m4.json` verifies exact exposure energy/order/timing preservation while showing substantial short-transient temperature differences from 5 ms grouping.

`docs/BUILD_MODEL.md` describes whole-build equations and event grouping; `docs/JOB_FORMAT.md` records exact decoding/provenance; `docs/MATERIAL_TRANSFER.md` explains the similarity score and limits. `examples/build-sensitivity-m4.json` contains actual Mac measurements, including a preserved earlier failed attempt and successful refined reruns.

Energy conservation and known-solution conduction checks pass. Representative voxel peaks are **not mesh-converged**, and retained heat is sensitive to plate depth. Whole-build matching therefore screens thermal accumulation and measurement priorities; it does not establish microscopic melt-pool shape, porosity, keyhole stability or a qualified alloy conversion. Measure delivered power, physical beam profile/diameter, baseline track geometry and independently justified target-alloy coupling before selecting printer settings. Ti/Al property extensions and source-data gaps are explicit in `docs/MATERIALS.md`.

## Start on this Mac

The local environment was inspected before installation: Apple M4 MacBook Air, 16GB RAM, macOS 27.2.1, arm64, Python 3.13.7. Dependencies are isolated in `.venv`; measured numerical performance is recorded in `examples/verification-m4.json` after verification.

```sh
cd /Users/elias/Desktop/Heat-sim
./run.command
```

Open [Tracklab](http://127.0.0.1:8501). Double-clicking `run.command` in Finder also starts the app. Stop the server with Ctrl-C in its terminal. Run cancellation in the browser cancels thermal integration and preserves already finished cached runs.

## Recreate the environment

Use Python 3.13 (Python 3.14 support depends on compatible Numba wheels). On this machine `/usr/local/bin/python3` is 3.13.7; select an equivalent Python 3.13 interpreter on another Mac.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

`requirements.txt` lists compatible direct dependencies; `requirements-lock.txt` records the actual installed versions for reproduction. Development checks additionally need `pytest` (included in the recorded lock or install `requirements-dev.txt`). The first numerical run includes compilation of Numba CPU kernels.

## Specialist single-track workflow

Open **Advanced → Local track laboratory** for this separate study workflow.

1. Review all laboratory sidebar assumptions. Job power and speed are provisionally W and mm/s. Enter the real beam diameter and profile, delivered-power factor, wavelength, each alloy's effective absorptivity, plate temperature, substrate alloy, packing and powder conductivity. Mark measured inputs explicitly.
2. Run **baseline + 316L bracket**. It first simulates17-4PH at 120W / 1,400mm/s, then 316L at identical settings and the requested120W / 1,260,1,400,1,540mm/s bracket. Editing the baseline changes the baseline comparison; the requested bracket stays anchored at 120W.
3. Optionally fit only 17-4PH absorptivity to measured successful 17-4PH width and depth. Bounds are constrained to0.05–0.9. The fit does not set 316L absorptivity. Apply its value manually after reviewing the fit and other assumptions.
4. Sweep editable power/speed bounds; compare geometry to the simulated 17-4PH baseline. Undefined/no-liquidus cases have no geometry rank.
5. Run the uncertainty endpoint study. It varies beam diameter, each alloy's absorptivity independently and powder conductivity. It reports whether a common closest candidate remains; it does not calculate an optimal recipe or statistical confidence intervals.
6. Review the saved numerical verification and sensitivity before interpreting small geometry differences. Use refined mode when comparing promising cases. Optional two-track mode runs adjacent same-direction passes in the same 3D thermal field.

Depth is measured **down from the original powder top**. Penetration is depth minus layer thickness, clipped at zero. Width is the largest transverse liquidus footprint in the central section. These are **maximum-over-time thermal envelopes**, not instantaneous molten-pool shapes. Width greater than hatch spacing and positive penetration are plausibility screens; they do not establish lack-of-fusion avoidance.

## Reproducible local example and tests

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m heat_sim.verification --sensitivity --output results/verification.json
.venv/bin/python -m heat_sim.worker examples/demo/request.json
```

The demonstration request records the assumed 80µm Gaussian full 1/e² intensity diameter, independent absorptivities 0.35, delivered factor 1, 80°C initial temperature,17-4PH substrate, packing 0.55, powder conductivity 1 W/(m·K),60µm layer/hatch and 120W / 1,400mm/s baseline. It is not derived from the job's `raduis` field. The example worker writes beside its request; browser runs write under `results/runs/`.

Actual job metadata was inspected as inert data from `/Users/elias/Downloads/Metal_printer/17-4 Test.htmjob`; `examples/job_reference.json` preserves source paths, units assumptions, source hash and disabled state. To inspect another compatible job without executing embedded objects:

```sh
.venv/bin/python -m heat_sim.job_reference '/path/to/example.htmjob' --output results/job-reference.json
```

Other presets are reference only. The 380W /888mm/s core is disabled. `keep_same=true` may override separate contour settings. `raduis=0.12` has unknown semantics and is never interpreted as beam size. The additional 200W / 1,900mm/s section may be supports; its purpose/exposure remains uncertain.

## Actual time-resolved playback

In the specialist single-track laboratory, the **Playback** tab renders actual instantaneous 3D temperature snapshots from the thermal integrator. It shows liquidus voxels, the moving beam position, top temperatures and a longitudinal section, with physical time in microseconds. The GIF is slowed for viewing. This is distinct from the maximum-over-time maps in Thermal fields.

```sh
.venv/bin/python -m heat_sim.render --alloy 17-4PH --output results/rendered/17-4PH
.venv/bin/python -m heat_sim.render --alloy 316L --output results/rendered/316L
```

Each rendering includes a GIF, PNG, actual temperature frames in NPZ, configuration and numerical metrics. Default machine and powder inputs remain illustrative; animated liquidus geometry is not melt-flow or keyhole simulation.

## Files and exports

- `app.py`: local interface, responsive background progress and cancellation.
- `heat_sim/solver.py`: conservative3D finite-volume enthalpy integration; Gaussian cell-face integrals, temperature-dependent properties and irreversible powder conductivity state.
- `heat_sim/materials.py`: exact property knots, sources and interpolation/extension notes.
- `heat_sim/workflows.py`: comparisons, sweeps, one-factor uncertainty envelopes and constrained17-4PH calibration.
- `docs/MODEL.md`, `docs/MATERIALS.md`, `docs/VERIFICATION.md`: model equations, primary references and verification details.
- `results/cache/`: content-addressed field arrays and metrics. Keys include source and configuration fingerprints.
- `results/runs/`: request JSON, status, CSV, figures, failure records and reports. No partial field from a failed/cancelled simulation is plotted.

The UI exports CSV, selected PNG figures, configuration JSON and a reproducible ZIP with numerical arrays, material/solver source, docs and dependency versions. Loading an exported configuration restores model fields and each alloy's absorptivity; a saved result always displays its own inputs rather than current sidebar values. Remove `results/cache/` to clear the cache while the app is idle.

## Evidence and remaining measurements

The primary sources are documented with URLs, units and validity ranges in `docs/MATERIALS.md`.316L uses NIST SRM 1155a measurements;17-4PH high-temperature graph readings/estimates and 270kJ/kg fusion heat have explicit uncertainty. Room-temperature properties are not silently used throughout.

Numerical verification checks units, source normalization, a known conduction solution through the production operator, enthalpy/latent consistency, boundary flux conservation, actual laser energy closure and representative mesh/time/domain sensitivity. It is distinct from experimental validation. The fixed-grid powder mass/consolidation approximation and omitted melt flow, recoil, vaporisation, spatter and keyhole dynamics are substantial limitations. Peaks above 2900K receive a model-validity warning.

Before narrowing physical trials, obtain the beam profile/full 1/e² diameter at the work plane, delivered power at these commands, actual scan-speed units, layer/hatch calibration, plate/substrate temperature/material, powder packing and conductivity, and measured17-4PH single-track cross-sections. Independent316L coupling or316L track measurements are needed; a 17-4PH fit cannot establish316L absorptivity. If ranking changes with uncertain inputs, measure the influential inputs before choosing a trial bracket.
