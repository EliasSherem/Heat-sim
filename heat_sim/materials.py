"""Auditable thermal properties. SI throughout; see docs/MATERIALS.md.

Sensible heat capacity is interpolated linearly. Its exact integral plus a
linear mushy-zone latent contribution gives a monotone enthalpy. Density is a
constant reference density: the fixed grid does not model volume shrinkage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

REFERENCE_T = 298.15
SOURCE_316_ENTHALPY = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8193647/"
SOURCE_316_CONDUCTIVITY = "https://doi.org/10.1007/s10765-022-02991-5"
SOURCE_174_REPORT = "https://digital.library.unt.edu/ark:/67531/metadc897473/m2/1/high_res_d/932641.pdf"
SOURCE_174_MANUFACTURER = "https://d1io3yog0oux5.cloudfront.net/_2cb86048768458ca9e4bc49ff6cfa0d4/clevelandcliffs/db/1190/10503/file/CLF_ProductData_17-4-PHSS_052021.pdf"
SOURCE_174_MELTING = "https://www.sandmeyersteel.com/wp-content/uploads/17-4PH-Spec-Sheet.pdf"
SOURCE_POWDER = "https://www.nist.gov/publications/powder-thermal-conductivity-measurements-l-pbf-using-powder-included-build-specimens"
SOURCE_TI_NBS = "https://nvlpubs.nist.gov/nistpubs/jres/081/2/V81.N02.A02.pdf"
SOURCE_TI_MODEL = "https://tore.tuhh.de/dspace-cris-server/api/core/bitstreams/ef255f46-e996-4039-87d8-dd7010965df6/content"
SOURCE_TI_FUSION = "https://www.govinfo.gov/content/pkg/GOVPUB-C13-022f4b1dcd580b4fb910d307f6745517/pdf/GOVPUB-C13-022f4b1dcd580b4fb910d307f6745517.pdf"
SOURCE_AL_MODEL = "https://doi.org/10.3390/modelling6040135"
SOURCE_AL_DENSITY = "https://www.eos.info/05-datasheet-images/Assets_MDS_Metal/EOS_Aluminium_AlSi10Mg/Material_Datasheet_EOS_Aluminium_AlSi10Mg_EN.pdf"
SOURCE_AL_MEASUREMENTS = "https://arxiv.org/html/2407.11580v1"

# Independent demonstration values, not measurements or alloy conversions.
# In particular, no stainless-steel calibration establishes the Ti/Al values.
ILLUSTRATIVE_ABSORPTIVITY = {"17-4PH": .35, "316L": .35,
                            "Ti-6Al-4V": .40, "AlSi10Mg": .15}


@dataclass(frozen=True)
class Material:
    name: str
    rho: float  # kg/m3, constant reference density
    solidus: float  # K
    liquidus: float  # K
    latent: float  # J/kg
    temperature_knots: tuple[float, ...]
    cp_values: tuple[float, ...]  # J/(kg K), excludes fusion latent heat
    k_values: tuple[float, ...]  # W/(m K)
    notes: str = ""
    sources: tuple[str, ...] = ()
    valid_temperature_max: float = 2900.0
    _nodes: np.ndarray = field(init=False, repr=False, compare=False)
    _cp: np.ndarray = field(init=False, repr=False, compare=False)
    _slope: np.ndarray = field(init=False, repr=False, compare=False)
    _h: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        t = np.asarray(self.temperature_knots, dtype=float)
        cp = np.asarray(self.cp_values, dtype=float)
        k = np.asarray(self.k_values, dtype=float)
        if len(t) < 2 or len(cp) != len(t) or len(k) != len(t):
            raise ValueError("Property arrays need matching lengths of at least two.")
        if (not np.all(np.isfinite(np.r_[t, cp, k, self.rho, self.solidus, self.liquidus, self.latent]))
                or np.any(np.diff(t) <= 0) or np.any(cp <= 0) or np.any(k <= 0)
                or self.rho <= 0 or self.solidus <= 0 or self.liquidus <= self.solidus or self.latent < 0):
            raise ValueError("Material properties must be finite, positive, and ordered.")
        nodes = np.unique(np.r_[t, self.solidus, self.liquidus, REFERENCE_T])
        sensible = np.interp(nodes, t, cp)
        slope = np.diff(sensible) / np.diff(nodes)
        apparent = sensible[:-1].copy()
        mid = (nodes[:-1] + nodes[1:]) / 2
        apparent += np.where((mid > self.solidus) & (mid < self.liquidus),
                             self.latent / (self.liquidus - self.solidus), 0.0)
        dh = apparent * np.diff(nodes) + .5 * slope * np.diff(nodes)**2
        h = np.r_[0.0, np.cumsum(dh)]
        h -= h[np.searchsorted(nodes, REFERENCE_T)]
        for key, value in [("_nodes", nodes), ("_cp", apparent), ("_slope", slope), ("_h", h)]:
            object.__setattr__(self, key, value)

    def conductivity(self, temperature):
        return np.interp(temperature, self.temperature_knots, self.k_values)

    def heat_capacity(self, temperature):
        """Sensible cp, excluding fusion; constant endpoint extrapolation."""
        return np.interp(temperature, self.temperature_knots, self.cp_values)

    def liquid_fraction(self, temperature):
        return np.clip((np.asarray(temperature) - self.solidus) /
                       (self.liquidus - self.solidus), 0.0, 1.0)

    def enthalpy(self, temperature):
        """Specific enthalpy [J/kg] relative to 298.15 K, including fusion."""
        t = np.asarray(temperature, dtype=float)
        j = np.clip(np.searchsorted(self._nodes, t, side="right") - 1, 0, len(self._cp) - 1)
        delta = t - self._nodes[j]
        h = self._h[j] + self._cp[j] * delta + .5 * self._slope[j] * delta**2
        h = np.where(t < self._nodes[0], self._h[0] + (t - self._nodes[0]) * self.cp_values[0], h)
        return np.where(t > self._nodes[-1], self._h[-1] + (t - self._nodes[-1]) * self.cp_values[-1], h)

    def temperature_from_enthalpy(self, enthalpy):
        """Exact inverse of the piecewise quadratic enthalpy, including tails."""
        h = np.asarray(enthalpy, dtype=float)
        j = np.clip(np.searchsorted(self._h, h, side="right") - 1, 0, len(self._cp) - 1)
        delta_h = np.maximum(h - self._h[j], 0.0)
        root = np.sqrt(np.maximum(self._cp[j]**2 + 2 * self._slope[j] * delta_h, 0.0))
        temperature = self._nodes[j] + 2 * delta_h / (self._cp[j] + root)
        temperature = np.where(h < self._h[0], self._nodes[0] + (h - self._h[0]) / self.cp_values[0], temperature)
        return np.where(h > self._h[-1], self._nodes[-1] + (h - self._h[-1]) / self.cp_values[-1], temperature)

    def table(self):
        """Serializable source/interpolation knots for inspection/export."""
        return [{"temperature_K": t, "cp_J_kgK": cp, "k_W_mK": k}
                for t, cp, k in zip(self.temperature_knots, self.cp_values, self.k_values)]


def _make_316():
    # Sample primary fits at <=50 K intervals. Gaps and tails are assumptions.
    t = np.unique(np.r_[298.15, np.arange(500., 1251., 50.), 1300., 1350.,
                        np.arange(1400., 1651., 50.), 1675., 1708.,
                        np.arange(1750., 2901., 50.)])
    cp = np.where(t <= 1250., 459. + .1432*t, 714.)
    # Linear bridge from end of DSC fit to high-temperature pulse-heating cp.
    bridge = (t > 1250.) & (t < 1350.)
    cp[bridge] = np.interp(t[bridge], [1250., 1350.], [638., 714.])
    cp[t >= 1708.] = 847.
    cp[(t > 1675.) & (t < 1708.)] = np.interp(t[(t > 1675.) & (t < 1708.)], [1675., 1708.], [714., 847.])
    k = np.where(t <= 1250., 8.98 + .0157*t - 1.5e-6*t*t,
                 6.38 + .019*t - 2.45e-6*t*t)
    k[(t > 1250.) & (t < 1300.)] = np.interp(t[(t > 1250.) & (t < 1300.)], [1250., 1300.],
        [8.98+.0157*1250-1.5e-6*1250**2, 6.38+.019*1300-2.45e-6*1300**2])
    k[t >= 1708.] = 2.27 + .0176*t[t >= 1708.] - 1.39e-6*t[t >= 1708.]**2
    return Material("316L", 7904., 1675., 1708., 290000., tuple(t), tuple(cp), tuple(k),
        notes="NIST SRM1155a data; cp/k fitted curves sampled. Below500K extrapolated; gaps bridged; above2900K constant tails. Reference density fixed. Fusion latent290kJ/kg is distributed in addition to sensible cp across33K.",
        sources=(SOURCE_316_ENTHALPY, SOURCE_316_CONDUCTIVITY), valid_temperature_max=2900.)


def _make_ti64():
    """Primary high-T cp plus an explicitly simplified published material model."""
    t = np.unique(np.r_[298.15, np.arange(300., 1201., 100.), 1268., 1300.,
                        1400., 1450., np.arange(1500., 1901., 50.), 1878.,
                        1928., 2000., 2100., 2200., 2300.])
    cp = 483. + .25*t  # Bartsch model, low-temperature sensible branch.
    nbs = 1000*(1.3833 - 9.943e-4*t + 3.745e-7*t*t)
    cp[(t >= 1450.) & (t <= 1900.)] = nbs[(t >= 1450.) & (t <= 1900.)]
    bridge = (t > 1268.) & (t < 1450.)
    cp[bridge] = np.interp(t[bridge], [1268., 1450.],
                         [483.+.25*1268., 1000*(1.3833-9.943e-4*1450.+3.745e-7*1450.**2)])
    hot = (t > 1900.) & (t < 1928.)
    cp[hot] = np.interp(t[hot], [1900., 1928.],
                      [1000*(1.3833-9.943e-4*1900.+3.745e-7*1900.**2), 930.])
    cp[t >= 1928.] = 930.  # Published model median, not a lot-specific measurement.
    k = np.where(t <= 1268., .012*t+3.3, .016*t-3.)
    mushy = (t > 1878.) & (t < 1928.)
    k[mushy] = np.interp(t[mushy], [1878., 1928.],
                        [.016*1878.-3., .0175*1928.-4.5])
    k[t >= 1928.] = .0175*t[t >= 1928.]-4.5
    return Material("Ti-6Al-4V", 4422., 1878., 1928., 286000., tuple(t), tuple(cp), tuple(k),
        notes="Illustrative mixed-source Ti64: measured NBS cp at1450–1900K; published fitted cp/k branches elsewhere. Solid-state transformation enthalpy/hysteresis omitted; phase interval1878–1928K is a model choice, not the NBS1943K plateau. Liquid cp930J/(kgK) is a fitted-model median; beyond2300K constant tails. Density fixed; independent absorptivity0.40 is ASSUMED.",
        sources=(SOURCE_TI_NBS, SOURCE_TI_MODEL, SOURCE_TI_FUSION), valid_temperature_max=2300.)


def _make_alsi10mg():
    """Convert a published volumetric thermal model to the fixed mass grid.

    This is not a claim that the high-temperature properties were measured.
    Sparse knots bridge the paper's heat-capacity jumps and the mushy interval.
    """
    rho = 2670.
    t = np.array([298.15, 400., 500., 600., 625., 700., 775., 800., 815.,
                  831., 850., 867., 900., 1000., 1200., 1500., 1800., 2100.])
    volumetric_cp = np.select([t <= 600., t <= 800., t <= 867.],
                             [2.3e6, 2.5e6, 2.8e6], default=2.6e6)
    alpha = 7.40e-5 - 3.00e-8*t
    mushy = (t > 831.) & (t < 867.)
    alpha[mushy] = np.interp(t[mushy], [831., 867.],
                            [7.40e-5-3.00e-8*831., 1.88e-5+6.43e-9*867.])
    alpha[t >= 867.] = 1.88e-5 + 6.43e-9*t[t >= 867.]
    return Material("AlSi10Mg", rho, 831., 867., 1.06e9/rho, tuple(t),
        tuple(volumetric_cp/rho), tuple(alpha*volumetric_cp),
        notes="Illustrative AlSi10Mg published thermal model, not a measured full-range curve: cp=Cvol/2670, k=diffusivity*Cvol, L=1.06e9/2670J/kg. Density2670kg/m3 is a fixed EOS reference. Heat-capacity steps and mushy conductivity are bridged; liquid/high-T properties assumed, with constant tails above2100K. Heat treatment/oxide effects unresolved; absorptivity0.15 is independently ASSUMED.",
        sources=(SOURCE_AL_MODEL, SOURCE_AL_DENSITY, SOURCE_AL_MEASUREMENTS),
        valid_temperature_max=867.)


# Approximate graph readings, not a precision transcription. Manufacturer anchors
# the low-temperature cp/density; high-T estimates are explicitly not measurements.
_t174_c = np.array([20., 149., 200., 260., 320., 460., 482., 600., 680., 720.,
                    850., 1050., 1200., 1350., 1440., 1500., 1600.])
_cp174 = [460., 460., 460., 500., 560., 680., 710., 840., 900., 550.,
          600., 650., 730., 835., 830., 820., 830.]
_k174 = [17.9, 17.9, 18.65, 19.5, 20.4, 22.5, 22.6, 23., 24., 24.5,
         25.5, 27., 29., 34., 35.5, 26., 27.]

MATERIALS = {
    "17-4PH": Material("17-4PH", 7780., 1677.15, 1713.15, 270000.,
        tuple(_t174_c+273.15), tuple(_cp174), tuple(_k174),
        notes="Mixed manufacturer/ORNL illustrative table: cp/k high-T values are approximate graph readings/estimates; latent270kJ/kg is an assumed model input. Manufacturer melting range differs from ORNL estimated range. Constant density and endpoint tails; hysteresis ignored.",
        sources=(SOURCE_174_MANUFACTURER, SOURCE_174_REPORT, SOURCE_174_MELTING),
        valid_temperature_max=1873.15),
    "316L": _make_316(),
    "Ti-6Al-4V": _make_ti64(),
    "AlSi10Mg": _make_alsi10mg(),
}

MATERIAL_WARNINGS = [
    "17-4PH fusion latent heat (270 kJ/kg) is assumed; high-temperature cp/k are approximate ORNL graph readings with estimated extensions.",
    "316L inputs represent NIST SRM1155a, not a certification of your powder chemistry. Source-data gaps and temperatures below500K are bridged/extrapolated.",
    "All alloys use fixed reference density; powder shrinkage, solid-state hysteresis and fluid transport are omitted.",
    "Effective absorptivity, powder conductivity and beam size are machine/powder inputs, not established by these bulk material sources.",
    "Ti-6Al-4V combines primary high-temperature cp with fitted-model branches; solid-state transformation enthalpy and hysteresis are omitted. Its absorptivity is independently assumed.",
    "AlSi10Mg uses a published volumetric thermal model converted to the fixed reference density; liquid/high-temperature properties are assumptions, not certified measurements. Its absorptivity is independently assumed.",
]
