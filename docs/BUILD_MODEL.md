# Imported-job whole-build heat history

`heat_sim.build_solver` extends the existing single-track solver with a
persistent coarse 3D finite-volume thermal state driven by decoded job vectors.
It supports each material in the same extensible `materials.py` database.
This mode is intended for comparing **heat accumulation, diffusion and cooling**
over a selected build range. It does not resolve microscopic melt-pool width,
depth, keyholes, porosity, consolidation density, or final mechanical properties.

## What is replayed and what is assumed

The parser supplies ordered vectors, layer-base Z positions, section/process IDs,
and section power and speed. Source coordinates are converted to SI; power is
still provisionally interpreted as W until calibrated. The recovered controller
logic establishes a reconstructed section/vector ordering, rather than a record
of captured machine timestamps. Enabled/disabled source regions are handled by
the parser. The ambiguous `raduis` field is never a physical beam measurement.

Every nonzero scan duration is `vector_length/(imported_speed*speed_scale)`.
The delivered power is `imported_power*power_scale*power_factor`. All section
ratios are retained. A change in speed therefore changes print time, jumps'
relative importance, heat diffusion during exposure, and subsequent cooling.
Missing laser-off jumps use an editable jump speed and settle delay; continuous
segments of one polyline have no added settle delay. The initial unknown laser
position is omitted rather than invented. The inter-layer total pause defaults
to **approximately 3 seconds including recoating**, from the user's estimate.
It is applied once per layer transition, including transitions through empty
layers. Jump timing is additional only where an actual off-path repositioning
is required. These assumptions do not constitute exact machine replay.

`timeline.npz` stores every event used by the calculation in sequence (activation,
scan, jump, recoat, cooldown), its start/end physical time and position, power,
speed, layer, section and vector ID. Up to 30,000 events are also saved as readable
`timeline.json`. A time-limited run truncates its last path proportionally.
A vector-limited run executes that exact prefix and an optional final cooldown.
Events outside a selected XY thermal crop still consume their physical time;
only the heat intersecting the crop is deposited there.

## Domain, activation and geometry

Coordinates use metres, with **z positive upward from the plate interface**.
The plate occupies z<0. X/Y bounds come from the imported scan bounds plus
editable powder/plate padding, or an explicit XY region. Plate depth, plate
material and powder-bed extent are assumptions until supplied. This is a finite
portion of a plate, not the printer's measured entire plate.

The default lateral cell size is 0.5 mm; the default vertical grouping is
0.24 mm. These are much larger than a physical 60 µm layer or 80 µm beam. A
new layer activates **powder mass**, not solid metal. For a vertical cell spanning
several physical layers, its active fraction increases by the deposited layer
thickness divided by cell height. Its mass is

    m_cell = rho_bulk * packing_fraction * active_fraction * dx*dy*dz

and the plate uses full bulk density. Added powder starts at the initial/preheat
temperature. We store **total excess enthalpy per cell**, relative to that common
preheat. Added preheated powder has zero excess enthalpy: existing cell energy
is retained while its mass increases. This causes a documented mixture-temperature
change in a partially filled cell; it does not reset the previously built volume.
All already-deposited material, powder and plate remain thermally coupled.
Only not-yet-deposited upper cells are omitted from the linear solve.

Without supplemental CAD, material geometry is a coarse scan-vector raster:
path length times the editable hatch-width proxy is accumulated in each crossed
cell and capped by its active fraction. This is inferred geometry, not a recovered
watertight CAD surface. A supplied, explicitly scaled and aligned STL instead
provides a planned part mask. It does **not** activate or solidify unscanned
material and does not change the imported exposure paths. Support masks only use
the process IDs explicitly selected by the user; section 16's physical purpose
must be confirmed rather than inferred solely from its presence.

`scanned_fraction` is scan-history occupancy. `thermal_liquid_fraction` is the
liquid fraction computed from the actual coarse voxel temperature. The
`consolidated_fraction` used for homogenised conductivity is explicitly an
**energy-constrained subgrid scan-history approximation**: the swept material
fraction is multiplied by at most

    absorbed segment energy /
    [rho_bulk*packing*swept_volume*(h_liquidus-h_preheat)]

capped at one. Resolved coarse melting can also increase this irreversible
conductivity state. There is no claim that this proxy proves actual fusion,
pore closure, dimensional accuracy, or density. A coarse voxel can average below
solidus even when a sub-voxel track locally melts. The proxy represents this
unresolved history without adding energy a second time. Its uncertainty includes
omitted conduction during subgrid melting. Cell mass is never increased by
switching its conductivity from powder to bulk. Shrinkage and moving surfaces
are not modelled.

## Moving source and temporal grouping

The surface source is the same Gaussian with an explicit **1/e² intensity
diameter** as the local solver. Its cell-face integral uses error functions;
its infinite-plane integral is the independently entered absorptivity times
delivered power. Wavelength is recorded, with no unsupported optical law mapping
wavelength to absorptivity. Stainless-steel absorption is never automatically
established for titanium or aluminium.

The actual vector is split in space into midpoint integration segments no longer
than the smaller of one third of a lateral cell and half a beam diameter. These
integrated face weights deposit energy into the active cell containing the
current physical powder layer. Their energy and duration are accumulated in
finite thermal blocks bounded by `max_event_s` (default 5 ms), with layer changes
forcing a flush. Laser-off jumps add cooling time but no heat. Long recoating
and final cooling intervals use implicit steps up to 0.25 s. Saved-frame boundaries
can further shorten a step.

This **homogenises the timing within a thermal block**. It retains ordered path
geometry, applied energy and total physical duration, but temperatures at
sub-block times and sub-voxel scales are unresolved. It is not a decorative laser
trail or an instantaneous heating of a whole layer. Reducing block duration and
spatial spacing is required to assess numerical sensitivity. The coarse mode
must not be used to measure melt-pool dimensions; the existing resolved
single-track mode is available for local investigations with measured inputs.

## Heat equation and conservative implicit solution

For each active cell the model solves

    E_i = m_i [h(T_i)-h(T_preheat)]
    dE_i/dt = sum_j G_ij (T_j-T_i) + Q_i - Q_environment

where `h` is the temperature-dependent sensible enthalpy plus fusion latent heat
from the material database. Lateral conductance uses a harmonic conductivity and
the active height of the neighbouring cells. Vertical conductance uses the two
half-cell thermal resistances with their active heights. Powder conductivity
blends to the bulk material's temperature-dependent conductivity according to
the irreversible homogenised consolidation state. The substrate has its own
material properties and phase-change range.

Sidewalls are insulated. The bottom plate face is fixed at preheat, through a
half-cell resistance. The exposed top loses heat through editable convection
and radiation to an enclosure assumed at preheat. Radiation uses the exact
factorisation `(T⁴-Ta⁴)=(T-Ta)(T+Ta)(T²+Ta²)` with its coefficient lagged during
the nonlinear iteration. This model omits melt flow, recoil, vaporisation,
spatter, keyholes, gas dynamics and changing surface absorption.

A backward-Euler finite-volume step solves the nonlinear enthalpy equation.
Damped Newton iterations use the apparent heat capacity, including latent heat,
with lagged conductivity. A residual-decreasing line search handles crossing
the fusion interval. Each linear system is symmetric positive definite and
solved by conjugate gradients with a diagonal preconditioner. The final update
uses converged equal-and-opposite face fluxes to update stored enthalpy directly.
Thus internal heat transfers cancel in the global balance.

The audit is

    cumulative applied absorbed energy - boundary losses - stored excess enthalpy

Fresh-powder activation has zero excess enthalpy at the selected preheat.
Sensible and latent stored energy are reported separately. Unconverged steps
are retried as two shorter steps with proportionally divided source energy;
repeated failures stop the calculation and write a failed manifest. Non-finite
states, temperatures above the numerical 14000 K guard, unphysical undershoots,
CG/Newton failures and a failed global energy balance are never plotted as
completed calculated frames. A high-temperature/model-range flag is separate
from numerical convergence. Numerical conservation is not experimental validation.

## Stored playback and comparison

Each completed frame is atomically saved as an NPZ file containing the actual
instantaneous temperature, deposited/scanned/consolidated fractions, liquid
fraction, part/support/plate masks, coordinates, physical time, layer and laser
position. Float32 temperatures reduce storage; this is storage quantisation,
not a change to the float64 calculation. The manifest is updated only after the
frame file is complete. Cancellation retains completed frames and records its
status; incomplete numerical states are not exported as valid frames.

Playback speed only changes the viewer clock. Timeline scrubbing selects stored
thermal results; it does not fabricate interpolated temperature physics. Where
the viewer displays a beam between thermal frames, its position comes from the
actual timed vector, while the thermal-frame time remains separately labelled.
Selected probes use actual stored voxel values and therefore do not resolve
transients between saved frames. 3D point sampling or colour interpolation in
the viewer is visual only and does not refine the numerical solution.

Each material comparison must use the same job prefix, domain, discretisation
and assumptions, with independent absorptivity for each alloy. The solver exports
material-relative peak enthalpy ratios, normalised by the energy required to heat
that material from preheat to liquidus. These are **coarse thermal-history
screening indicators**, not melting-point-ratio recipes or defect predictions.
Power/speed candidate matching can reduce an exploratory test bracket; it cannot
establish a printer recipe without measurements and local/experimental checks.

## Verification and Mac operation

Tests cover ordered vector timing and per-section power scaling, new-powder mass
activation without energy reset, Gaussian path-energy integration, the actual
implicit conduction operator against an insulated cosine-decay benchmark,
latent-enthalpy heating through an aluminium melting interval, cooling across
laser-off intervals, saved-frame consistency and cancellation. Mesh, event-time
and plate-depth sensitivity results are recorded separately; failed refined
cases remain failures rather than appearing as converged fields.

`estimate_build` reports actual grid/work counts and uncompressed array-byte
estimates. It does not promise a whole-job runtime. The full imported reference
job has 901 layers and about 1.595 million scan vectors, so initial runs should
use a short layer/vector/time range. A selected nonzero starting layer requires
explicit consent to initialise the earlier build at uniform preheat without its
missing thermal history. Whole-job studies require a coarser grid and larger
finite event blocks, followed by sensitivity checks on a smaller region.

### Measured example and sensitivity on this Mac

The real imported job's first three layers (including its empty layer 1) were
calculated on the actual Apple M4 MacBook Air, 16 GB, macOS environment. The
run in `results/builds/actual-17-4-first3-newton` used 104×64×8 cells,
0.5 mm lateral/0.24 mm vertical spacing, 10 ms exposure blocks, 30 stored frames,
and a 0.2 s final cooldown. It represented 6,830 vectors and 9,083 schedule
events over 9.2216 s physical time, including two approximate 3 s recoating
pauses. It took **13.12 s measured computation time**, 345 thermal steps and
about **343 MiB sampled process RSS** including imported-job and JIT memory.
Its relative energy closure error was 3.69e-13. These measurements do not
establish a whole-job runtime or physical accuracy.

The separate reproducible `examples/build-sensitivity-m4.json` uses the first
150 actual support-section vectors and records:

| Case | Peak voxel temperature (K) | Final stored excess energy (J) | Measured run time (s) |
|---|---:|---:|---:|
| 0.5 mm lateral / 0.24 mm vertical | 1056.78 | 0.39509 | 0.52 |
| Half exposure-block duration | 1097.09 | 0.39629 | 0.74 |
| 0.25 mm lateral / 0.12 mm vertical | 2137.18 | 0.39705 | 3.73 |
| Plate depth doubled, 0.72→1.44 mm | 1056.81 | 0.55498 | 0.74 |

All four final cases converged and conserved energy to about 1e-13 relative.
The **peak temperature is not mesh-converged**: spatial refinement changes it
by roughly 1080 K. Doubling plate depth changes retained energy by about 40.5%,
so that case is also **sensitive to the assumed lower plate boundary**. These
are limitations requiring further checks, not validation successes. The prior
secant-iteration failure was retained as `previous-secant-report.json` in the
sensitivity results directory; the current damped-Newton solver completed the
same refined case. The app must not turn these coarse peaks into microscopic
melt-pool predictions or printer settings without additional evidence.

The **complete real 901-layer job also ran successfully** on this Mac. The
frozen, reproducible `results/builds/full-17-4-coarse` run processes all
1,595,099 decoded scan vectors and 2,634,151 timed events on a 26×16×39 grid
(2 mm lateral/1.5 mm vertical), with 100 ms finite exposure blocks. It computes
5,179.20 s of assumed physical print time in **646.37 s measured computation
time (10 min 46 s)**, using 36,027 thermal steps and saving 60 complete 3D frames.
Sampled process RSS was 670.84 MiB; external maximum resident set was
797,065,216 bytes (about 760 MiB), with zero swaps. Its energy audit is
57,423.79438 J absorbed, 52,195.53144 J lost at boundaries and 5,228.26294 J
stored, with 1.61e-13 relative residual. The maximum **coarse voxel average**
was 1008.66 K (735.51 °C); this is not a resolved track peak or evidence that
the real process does not melt powder. `examples/full-build-benchmark-m4.json`
records the configuration, exact job/source hashes, environment and metrics.

The first full-run attempt stopped at layer124: equivalent expressions for
7.5 mm differed by one floating-point ulp and activated a negligible-mass ghost
cell above the physical powder surface. The solver now snaps activation
fractions within1e-12 of zero or one, with a regression test for that exact
actual-job condition. The failure, partial frames, original frozen source and
diagnosis remain in `full-17-4-coarse-failed-ghost-layer124`. The successful
run restarted from the beginning with every vector and identical physical inputs.

A separate first-150-vector comparison on the full-build grid, saved in
`examples/full-build-grouping-m4.json`, checks 100 ms against 5 ms blocks. Both
runs have a byte-identical timed exposure schedule, the same physical duration,
the same source order and 0.5731172174 J absorbed energy. Energy closure is about
1e-13 in both. However, peak voxel temperature changes from 397.56 K to 488.12 K
(+90.55 K), and final stored heat increases by 2.14%. The material-relative peak
enthalpy indicator also changes substantially. This quantifies the grouping
approximation: **100 ms full-build results are not converged local transients**.
They support viewing coarse heat history, while selected features require
shorter blocks and spatial/local convergence checks before physical trials.

Local use (after installing the project's isolated environment):

```python
from heat_sim.job_import import load_job
from heat_sim.build_solver import BuildConfig, run_build

job = load_job("results/imports/17-4-actual/job-summary.json")
config = BuildConfig(absorptivity=0.35, layer_end=3,
                     recoat_time_s=3.0, max_event_s=0.01,
                     max_frames=30, final_cooldown_s=0.2,
                     support_sections=(16,))  # support identity remains assumed
manifest = run_build(job, config, "results/builds/my-replay")
```

The source fingerprint, full material tables and references, configuration,
job hash, exact event schedule, numerical failures and runtime/memory measurements
are retained locally. The app does not communicate with the printer.
