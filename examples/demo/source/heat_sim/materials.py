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
}

MATERIAL_WARNINGS = [
    "17-4PH fusion latent heat (270 kJ/kg) is assumed; high-temperature cp/k are approximate ORNL graph readings with estimated extensions.",
    "316L inputs represent NIST SRM1155a, not a certification of your powder chemistry. Source-data gaps and temperatures below500K are bridged/extrapolated.",
    "Both alloys use fixed reference density; powder shrinkage, solid-state hysteresis and fluid transport are omitted.",
    "Effective absorptivity, powder conductivity and beam size are machine/powder inputs, not established by these bulk material sources.",
]
