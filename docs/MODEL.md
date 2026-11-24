# Thermal model and interpretation

This is a local, CPU-only exploratory model of a **0.5 mm straight scan** on a
powder layer over a solid substrate. It is numerical verification software, not
an experimentally validated parameter conversion or qualified print recipe.
Only body infill is initially simulated; the other job sections are reference
data. Power is provisionally interpreted as W and speed as mm/s.

## Equations and units

All solver quantities use SI units. The conserved quantity in each fixed
finite-volume cell is specific enthalpy `h` [J/kg]:

    rho_fixed * dh/dt = div(k(T,state) grad(T)) + Q
    h(T) = integral(cp_sensible(T) dT) + L * f_liquid(T)
    f_liquid = clip((T - T_solidus)/(T_liquidus-T_solidus), 0, 1)

Reference enthalpy is zero at 298.15 K. `cp` is piecewise linear; its integral
is piecewise quadratic. Fusion latent heat is added **once**, linearly over
the solidus-to-liquidus interval. Solidification returns that latent energy.
Material tables, provenance, missing ranges, and endpoint extrapolations are
in [MATERIALS.md](MATERIALS.md). The solver inverts a uniform-enthalpy lookup
table (50 J/kg spacing), built from this enthalpy law; this is approximately
0.13 K or less in sensible-temperature interpolation scale.

For the default Gaussian surface source, with r the radial distance from the
moving beam centre and w the **1/e² intensity radius**:

    q_laser(r,t) = (2 A P_delivered / (pi w²)) exp(-2 r² / w²)
    P_delivered = P_job * calibration_factor
    diameter = 2 w

The integral over the infinite plane is `A*P_delivered`. Cell-face integrals
use differences of the error function, rather than sampling at cell centres
and renormalising. The reported applied energy accounts for the fraction
actually intersecting the domain. A selectable uniform disk profile uses a
**geometric disk diameter**, integrating circle/rectangle overlap by 24-point
Gauss quadrature per x cell. Equal numerical diameters of these profiles do
not imply equal second moments. Wavelength is recorded as an input; no
optical wavelength-to-absorptivity law is inferred.

The source travels at constant speed, x = -0.25 mm + v*t, at y=0. There is no
source outside the scan interval. Time-step overlaps with the laser-on
interval are integrated exactly in duration; spatial position uses its
midpoint. For an optional second same-direction track, y=hatch, with an
editable intertrack delay, both tracks share the same temperature/history
field. This checks limited local reheating, not a rastered full layer.

## Powder, consolidation and substrate

The top layer initially has fixed mass density `packing_fraction*rho_bulk`
and an editable effective conductivity in W/(m K). During partial melting,
conductivity blends linearly from powder k to bulk k with liquid fraction;
after a cell first reaches liquidus, bulk conductivity is retained on cooling.
Enthalpy and cell mass are unchanged by this switch. This deliberately avoids
creating mass or energy when the conductivity state changes.

The fixed grid **does not move, collapse, or resolve pore closure**. A melted
powder cell retains its initial mass per cell volume despite the consolidated
conductivity approximation. It therefore represents an effective thermal
medium, not a mass-conserving free surface with correct local densification.
This is a substantial uncertainty in surface geometry and temperature.
The solid substrate uses its own selectable alloy, full reference density,
temperature-dependent cp and k, solidus/liquidus, and latent heat. Fixed density
omits thermal expansion, shrinkage and fluid transport.

## Domain and boundaries

Initial temperature is uniform and editable. Default domain is 0.9 mm in x,
0.6 mm in y, and 0.6 mm substrate depth **plus the powder layer**. Domain-scale
checks enlarge all outer extents while preserving the 0.5 mm track and its
central measurement plane. Lateral and bottom boundary **faces** are held at
initial temperature using half-cell face distances. Their heat loss is
integrated and reported. The top boundary loses heat by

    q_loss = h_conv (T-T_initial) + emissivity sigma (T⁴-T_initial⁴)

and receives the laser flux. Convective and radiative surroundings are assumed
at initial/build-plate temperature; this is a simplified enclosure boundary.
The default h_conv=15 W/(m² K), emissivity=0.35, initial 353.15 K, beam 80 µm,
absorptivity 0.35, packing 0.55, powder k=1 W/(m K), and delivered power factor=1
are **illustrative assumptions**, not specifications of the user's printer.
The file field `raduis=0.12` is never used for beam dimensions.

At 60 µm layer thickness, coarse mode has 20 µm x/y nominal spacing and 20 µm
z spacing (three cells through the layer). Refined mode uses 10 µm nominal
x/y and z spacing (six cells through the layer). For changed layer thickness,
z spacing becomes layer/3 or layer/6; the substrate depth is rounded upward.
Domain enlargement preserves lateral cell spacing and centring parity; outer
extents round upward to whole cell counts, so nominal scaled dimensions can
differ by up to two cells. This isolates boundary effects from grid-phase
changes in the domain comparison.
Short scans are followed by an editable cooldown, default 0.3 ms. The
temperature rise at outermost cells is reported as a boundary warning,
but a domain comparison is needed even when that warning is absent.

## Numerical scheme and failure handling

Cell-centred finite volumes use a harmonic conductivity at each interior
face. Equal/opposite interior heat transfers cancel exactly in the global
energy sum. A forward Euler step updates cell enthalpy, then looks up
temperature and state-dependent conductivity. The global conservative
time-step limit is

    dt <= 0.35 / [2 alpha_max (dx^-2 + dy^-2 + dz^-2)]
    dt <= dx/(4 v)

where alpha_max covers all sampled bulk cp/k values and the lower fixed
powder mass density; the initial powder conductivity is included separately.
The factor is conservative for the fixed-temperature half-cell boundary
faces. `dt_scale` can only reduce this step. Top radiation is explicit.
Finite-state checks reject NaN/Inf, unphysical undershoots >1 K below initial,
and excursions above 14000 K (a numerical guard, not a physical boiling model).

The energy audit is `applied absorbed energy - outward boundary losses -
change in stored enthalpy`. Stored enthalpy includes latent heat; the final
sensible/latent split is also recorded. Runs with relative closure error
above 1e-6 (with an absolute 1e-10 J floor for zero/tiny input energy) raise
a failure and do not return valid plotted geometry. A small
closure error is a **conservation check**, not proof of spatial or temporal
accuracy. Material-lookup interpolation, grid, source discretisation, finite
domain and omitted transport remain separate errors. Cancellation raises
and discards partial results. First use includes JIT compilation; runtime
and sampled process RSS are measured on the actual executing machine.

## Geometry definitions

Coordinates are x along the track, y transverse, and **z positive downward
from the original powder top surface**. The geometry measurement plane is
the cell centre nearest x=0; its exact coordinate is exported. At that plane,
we retain each cell's **maximum temperature over the complete run**. Its
liquidus footprint estimates the fully molten cross-section swept by the
short track; it is not an instantaneous melt-pool shape. Peak maps similarly
combine maxima reached at different times and must be labelled accordingly.

Width is the largest transverse liquidus-to-liquidus span at any depth in
that plane. Depth is the deepest liquidus crossing measured from z=0.
Substrate penetration is `max(0, depth-layer_thickness)`. Threshold crossings
are linearly interpolated between cell-centre temperatures. With unlike
substrate and powder alloys, each cell uses the local alloy's liquidus.
At the material interface this piecewise threshold is itself an approximation.
An unresolved crossing reports zero with a warning. Reaching domain edges
means truncation and requires a larger domain. Width/depth precision is
limited by cell size and must be assessed through refinement.

`width/hatch > 1` suggests geometrical overlap may be plausible; it does not
guarantee lack-of-fusion avoidance. Positive penetration only establishes
a thermal footprint below the prior solid surface under these assumptions.
Mushy material can exist outside the liquidus footprint. No output estimates
part density, fatigue strength or a guaranteed defect-free process.

## What the model excludes

There is no melt convection/Marangoni flow, free-surface motion, recoil
pressure, vaporisation cooling, spatter, multiple scattering, dynamic
absorptivity, keyhole geometry, gas flow, residual stress, or microstructure
evolution. High temperatures, especially above about 2900 K, are a reason
to distrust conduction-only geometry, not a validated keyhole prediction.
Any thermal aspect-ratio/overlap proxy is only a **screening indicator**.
Latent heat of vaporisation is not included. A 17-4PH fit of absorptivity or
beam size does not establish 316L absorptivity.

Numerical checks (source integration, analytic conduction benchmark,
conservation, mesh/time/domain sensitivity) are distinct from experimental
validation against measured single-track metallography. Rankings should be
compared across uncertain beam size, independently chosen absorptivities,
and powder conductivity. If they change, measurements of the influential
input and actual 17-4PH and 316L track geometry are needed before selecting
settings for physical trials.
