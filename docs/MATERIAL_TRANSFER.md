# Transfer the same imported job between alloys

The material-transfer workflow replays the **actual imported vector geometry**
with the whole-build enthalpy solver. Select the current material, independent
absorptivity for each material, one or more targets, and a bounded selection of
power/speed multipliers. The current case always uses multipliers 1/1; each
target is first run at identical commanded settings. Optional candidates scale
the imported power and speed of every section by the selected multipliers.
Existing contour/infill/support differences are preserved. The software does
not manufacture a target recipe by scaling melting temperatures or strength.

For example, power×1.1 and speed×0.9 turn an imported 120 W / 1400 mm/s section
into 132 W / 1260 mm/s, and an imported 100 W / 1000 mm/s section into
110 W / 900 mm/s. Delivered-power calibration still applies separately. The
original file and scan vectors are unchanged. No printer connection occurs.

## What the comparison measures

The coarse whole-build model does not resolve the beam-scale instantaneous
melt pool. It supplies persistent temperature/enthalpy histories, heat
accumulation, cooling, and approximate conduction response of the imported
scan footprint and surrounding powder/substrate. Even if the coarse voxel
temperature stays below liquidus, microscopic melting may have occurred below
the grid/time resolution. Do not use these whole-build results as melt-pool
width/depth, fusion, keyhole, porosity or fatigue predictions.

For each alloy, the transfer module converts the computed peak temperature in
each exposed part voxel into an enthalpy index:

    I = [h(T_peak) - h(T_initial)] / [h(T_liquidus) - h(T_initial)]

Here `h` is the material's temperature-dependent sensible-plus-fusion enthalpy.
The reference uses that material's own liquidus and latent heat. **I is not a
liquid fraction.** It describes the coarse excess enthalpy relative to the
enthalpy needed to reach liquidus. No physical setting is calculated from a
ratio of melting points.

The three features are the mean, 95th percentile, and maximum of I over the
common exposed part-voxel set; the transfer module's mean is an unweighted
voxel mean. Its similarity score is

    score = sqrt(mean_j(((feature_target_j - feature_current_j)
                         / max(abs(feature_current_j), 0.02))**2))

The dimensionless 0.02 denominator floor is a disclosed numerical scaling
choice for nearly unheated reference cases. Lower means a closer **sampled
coarse thermal response** under the supplied assumptions. There is no universal
acceptable score, defect threshold or validated optimum. Small score
differences have no established significance without refinement and physical
measurements. The result names the closest sampled case and retains every
computed alternative for review.

Physical peak temperature rise is reported independently. The saved probe
history contains physical time in seconds, temperature in K, temperature rise
in K, enthalpy index, and active fraction at one common nearest-cell position.
It comes from converged stored frames; no intermediate temperatures are
invented. Probe histories are not included in the similarity score because
changing speed changes physical timing, while frame sampling also has finite
resolution. The whole-build peak field covers actual solver integration steps
rather than just movie frames.

## Keeping the comparison meaningful

The same configuration preserves geometry, mesh, initial conditions, laser
profile and optical wavelength, layer range, timing assumptions and surrounding
domain. By default the selected substrate stays fixed; an explicit option
makes it follow each material, in which case substrate-property changes also
affect the comparison. Powder conductivity remains an editable assumption.

Only cases with matching grid coordinates, exposed voxel footprints and actual
scan-vector exposure signatures receive a score. A physical-time cutoff can
stop different speed cases halfway through different vectors. Those cases may
still have valid computed temperatures, but their scores are unavailable and
they are excluded from rankings. Prefer a common layer range or maximum
vector count to compare the same geometry. A cropped region has the model's
documented artificial side boundaries and cannot represent heat transfer from
omitted surrounding geometry.

Each case must finish integration with its energy check passing. Failures are
written to `failures.json`, not converted into zero temperatures or ranked as
good settings. Cancellation retains completed artifacts and marks the transfer
report cancelled. A completed batch can contain failed cases; inspect the
failure count and comparison availability.

## Independent absorptivity and uncertainty

17-4PH, 316L, Ti-6Al-4V and AlSi10Mg each have an independent editable effective
absorptivity. The supplied values 0.35, 0.35, 0.40 and 0.15 are illustrative
assumptions. A fit to a successful stainless track does not identify another
alloy's absorption. Even a fitted current-alloy absorption may compensate for
single-track model error; it is not experimental validation of this coarse
whole-build model.

Optional uncertainty runs hold the current-alloy case fixed and replay every
target candidate at user-supplied low and high target absorptivities. Along
with the nominal run, these are three assumption scenarios, not confidence
intervals or a probability distribution. The report states whether the
closest sampled case and full ordering remain consistent. Missing/failed
scenarios make consistency unevaluable. A single candidate cannot establish a
stable ranking.

If rankings change, measure target effective absorptivity and delivered power,
check the beam profile/diameter, and obtain successful current-alloy and target
single-track cross-sections before narrowing printer trials. Stable rankings
across these scenarios do not account for beam size, powder conductivity,
packing, gas, timing, grid/time/domain error or omitted physics. These remain
separate uncertainty sources. Use the local single-track laboratory and
experimental measurements when track-scale width, penetration and overlap are
the decision criteria; numerical verification is separate from validation.

## Local artifacts and reproducibility

`report.json` contains requested options, material assumptions, successful rows,
rankings, uncertainty scenarios, failure details and fingerprints.
`comparison.csv` is the flat results table. Each row's `case_folder` links to
the build `manifest.json`, frame NPZs, `peak-fields.npz`, timeline, configuration,
`transfer-features.json` and `transfer-probe.csv`.
Case folders, ranked-case links, and report-level probe CSV links are relative
to the study root, so they remain usable after copying or extracting an archive.

Completed cases are reused only by a key containing the imported normalized
vector data, source job hash, full build configuration, model source content
and optional supplemental geometry file content. Changing an STL at the same
path therefore changes its cache key. Each candidate still needs review of its
explicit material-source limits in [MATERIALS.md](MATERIALS.md) and whole-build
limitations in [BUILD_MODEL.md](BUILD_MODEL.md).
