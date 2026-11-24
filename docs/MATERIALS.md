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

## Ti-6Al-4V: primary high-temperature data and fitted extensions

[Cezairliyan, McClure and Taylor (1977), *Thermophysical Measurements on
90Ti-6Al-4V Alloy Above 1450 K*, NBS](https://nvlpubs.nist.gov/nistpubs/jres/081/2/V81.N02.A02.pdf)
supplies cp = 1000(1.3833 − 9.943×10⁻⁴ T + 3.745×10⁻⁷ T²)
J/(kg·K), valid 1450–1900 K, with estimated 3% inaccuracy. Its reference density
4422 kg/m³ is held fixed here. Their measured melting plateau of 1943 K differs
from the illustrative interval below; chemistry and heating rate matter.

[McClure and Cezairliyan (1992), *Measurement of the heat of fusion of titanium
and a titanium alloy*, NIST author abstract reproduced in its publication
catalog](https://www.govinfo.gov/content/pkg/GOVPUB-C13-022f4b1dcd580b4fb910d307f6745517/pdf/GOVPUB-C13-022f4b1dcd580b4fb910d307f6745517.pdf)
reports 286 J/g = **286000 J/kg** for 90Ti-6Al-4V, with estimated maximum
uncertainty ±6%; this is the implemented fusion latent heat.

[Bartsch et al. (2021), *Material modeling of Ti–6Al–4V alloy processed by laser
powder bed fusion*, equations 10, 14, 20–22](https://tore.tuhh.de/dspace-cris-server/api/core/bitstreams/ef255f46-e996-4039-87d8-dd7010965df6/content)
provides fitted branches informed by literature and their DSC/LFA work:

| Property | Implemented fitted extension |
|---|---|
| cp below 1268 K, J/(kg·K) | 483 + 0.25 T |
| Liquid cp, J/(kg·K) | 930 (model median) |
| k below 1268 K, W/(m·K) | 3.3 + 0.012 T |
| Solid k above 1268 K | −3 + 0.016 T |
| Liquid k | −4.5 + 0.0175 T |

The model interval 1878–1928 K is adopted. Solid-state transformation enthalpy
and hysteresis are omitted. Gaps between branches and fusion conductivity are
bridged. This is a simplified mixed-source constitutive law, not their complete
history-dependent model. No Marangoni conductivity multiplier is applied.
Properties are sampled to 2300 K and held constant beyond it.

## AlSi10Mg: explicitly illustrative high-temperature thermal model

[Pelevin et al. (2025), *Influence of AlSi10Mg Thermophysical Properties on the
Melt Pool Morphology*, Table 2](https://doi.org/10.3390/modelling6040135)
provides a published simulation input model, **not a measured full-range data
set**. The app converts its volumetric properties consistently:

| Quantity | Published model / implemented conversion |
|---|---|
| Solidus / liquidus | 831 / 867 K |
| Volumetric sensible capacity Cvol | 2.3, 2.5, 2.8 MJ/(m³·K) below 600, 800, 867 K; 2.6 above |
| Solid diffusivity | 7.40×10⁻⁵ − 3.00×10⁻⁸ T m²/s |
| Liquid diffusivity | 1.88×10⁻⁵ + 6.43×10⁻⁹ T m²/s |
| Fusion enthalpy per volume | 1.06×10⁹ J/m³ |
| cp, k, L for fixed grid | Cvol/rho, diffusivity×Cvol, 1.06×10⁹/rho |

[EOS's AlSi10Mg material sheet](https://www.eos.info/05-datasheet-images/Assets_MDS_Metal/EOS_Aluminium_AlSi10Mg/Material_Datasheet_EOS_Aluminium_AlSi10Mg_EN.pdf)
reports average part density ≥2.67 g/cm³. The app assumes a fixed reference
2670 kg/m³. Thus L is approximately 397 kJ/kg; its computed decimal places reflect
unit conversion, not measurement precision. Heat-capacity steps are smoothed by
the exported interpolation knots; conductivity is bridged through fusion.
Liquid properties remain assumptions; values above 2100 K are held constant.

[Fetni et al. (2024), primary DSC, dilatometry and laser-flash measurements,
section 2 and Figure 4](https://arxiv.org/html/2407.11580v1)
demonstrates that AlSi10Mg properties depend on thermal history and
microstructure. The available solid-state measurements do not establish the
selected high-temperature model for the user's powder. This app does not claim
to reproduce their measured curves or phase-field model. It flags temperatures
above 867 K as outside its conservative property-confidence limit.

## Optical assumptions for all four alloys

`ILLUSTRATIVE_ABSORPTIVITY` stores independent demonstration values: 17-4PH 0.35,
316L 0.35, Ti-6Al-4V 0.40, AlSi10Mg 0.15. **All are assumed**; none is an optical
measurement or a transferable calibrated value. Wavelength, beam size,
surface/oxide state, powder distribution, gas and temperature can affect the
effective absorbed fraction. A successful stainless calibration establishes
neither Ti nor aluminium absorption. Measure or explore each separately.
