# Material inputs and provenance

Retrieved 15 November 2026. Values describe reference bulk alloys, not the user's
powder lot. `heat_sim/materials.py` contains every interpolation knot. The app's
material table and exports expose those knots. All temperatures below are kelvin
unless labelled °C. No absorptivity is inferred from these thermal properties.

## 316L: primary measurements on NIST SRM 1155a

[Pichler, Simonds, Sowards and Pottlacher (2020), *Measurements of thermophysical
properties of solid and liquid NIST SRM 316L stainless steel*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8193647/)
provides the following inputs (Tables 2–3): reference density 7904 kg/m³ at room
temperature; solidus 1675 ±15 K; liquidus 1708 ±30 K; fusion enthalpy 290 kJ/kg.
Sensible heat capacity, J/(kg·K), uses the derivative of their enthalpy fits:

| Interval K | Implemented cp |
|---|---|
| 500–1250 | 459 + 0.1432 T |
| 1350–1675 | 714 |
| 1708–2900 | 847 |

The 1250–1350 K gap and mushy interval are linearly bridged. Below 500 K the
low-temperature fit is extrapolated to 298.15 K; below that cp is held fixed.
Above 2900 K cp is held fixed. These extensions are assumptions. Constant density
replaces the paper's temperature-dependent density because this fixed grid does
not solve expansion or shrinkage. Reported source uncertainties are measurement
uncertainties, not uncertainty bounds on this model's predictions.

## 316L conductivity

[Pichler et al. (2022), *Surface Tension and Thermal Conductivity of NIST SRM
1155a*](https://doi.org/10.1007/s10765-022-02991-5), equations 11–13, gives k in
W/(m·K):

| Interval K | Fit sampled into the implementation |
|---|---|
| 500–1250 | 8.98 + 0.0157 T − 1.50×10⁻⁶ T² |
| 1300–1675 | 6.38 + 0.0190 T − 2.45×10⁻⁶ T² |
| 1708–2900, liquid | 2.27 + 0.0176 T − 1.39×10⁻⁶ T² |

The sentence introducing equation 13 specifies 1708–2900 K; the printed equation
repeats the solid interval. The implementation follows the stated liquid
interval. Knots at intervals no larger than 50 K approximate the fits by linear
interpolation. Missing intervals are bridged; the low-temperature polynomial is
extrapolated to 298.15 K. Outside the knot range k is constant. The liquid fits
rely on an electrical-resistivity relation and are not direct conductivity
measurements.

## 17-4PH: a less certain illustrative property set

[Cleveland-Cliffs, *17-4 PH Stainless Steel*, Table 9, page 8](https://d1io3yog0oux5.cloudfront.net/_2cb86048768458ca9e4bc49ff6cfa0d4/clevelandcliffs/db/1190/10503/file/CLF_ProductData_17-4-PHSS_052021.pdf)
supplies density 7780 kg/m³ (Condition A), cp 460 J/(kg·K) over 0–100 °C,
and conductivity anchors 17.9, 19.5, 22.5, 22.6 W/(m·K) at 149, 260, 460,
482 °C (H900). Mixing conditions introduces uncertainty.

[Cannell and Sabau, ORNL/TM-2007/204 (2007), section 4.2.1, Figure 4.2,
printed page 14 / PDF page 28](https://digital.library.unt.edu/ark:/67531/metadc897473/m2/1/high_res_d/932641.pdf)
provides plotted cp/k measurements and high-temperature estimates. The app uses
coarse approximate visual readings of these curves through 1600 °C, combined
with the manufacturer anchors. They are **not precision digitized measurements**.
The cp transformation peak and subsequent decrease are included approximately;
hysteresis is ignored. High-temperature estimates remain assumptions in this
tool. Obtain lot-specific thermal analysis before quantitative interpretation.

[Sandmeyer Steel, *Alloy 17-4PH specification sheet*](https://www.sandmeyersteel.com/wp-content/uploads/17-4PH-Spec-Sheet.pdf)
gives melting range 1404–1440 °C, used as solidus/liquidus 1677.15–1713.15 K.
This differs from the ORNL composition's calculated melting interval: the
combination is an explicit illustrative approximation.

The fusion latent heat **270 kJ/kg is an assumed model input**, not a newly
measured or certified 17-4PH value. Endpoint cp/k values are held constant outside
the implemented range. Replace the table when better data are available.

## Powder, machine and phase-change assumptions

[Zhang, Lane and Chou (2020), NIST, *Powder Thermal Conductivity Measurements
in L-PBF using Powder-Included Build Specimens*](https://www.nist.gov/publications/powder-thermal-conductivity-measurements-l-pbf-using-powder-included-build-specimens)
reports effective 17-4PH powder conductivity 0.67–1.34 W/(m·K) over 100–500 °C,
with specimen-geometry dependence. This motivates exploring powder conductivity;
it does not establish its value at melting, for another powder/gas, or for 316L.

The user's powder conductivity and packing fraction are editable assumptions.
The solver retains the original powder mass on the fixed grid while changing
conductivity after consolidation. It does not fill shrinkage voids or move the
free surface. The substrate uses the selected alloy's properties; a measured
different substrate requires its own property set.

Both materials use h(T) = integral(cp dT) + L f(T), with f linear between solidus
and liquidus. Thus the total mushy-zone enthalpy rise includes sensible heat in
addition to L; it is not an exact reproduction of the source's measured enthalpy
curve. There is no latent heat of vaporisation. Outputs at extreme temperatures
are a warning about missing physics, not evaporation predictions.

Wavelength is recorded, but the tool has no optical model to calculate
absorptivity from wavelength. Effective absorptivity must be supplied or explored
separately for each alloy. A fit for 17-4PH cannot identify the 316L value.
