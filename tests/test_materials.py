"""Material invariants that protect phase change and unit conventions."""
import numpy as np
import pytest

from heat_sim.materials import MATERIALS, REFERENCE_T, ILLUSTRATIVE_ABSORPTIVITY


@pytest.mark.parametrize("name", MATERIALS)
def test_enthalpy_inverse_through_melting_and_extrapolation(name):
    material = MATERIALS[name]
    temperature = np.unique(np.r_[200., np.linspace(298., 4000., 777),
                                  material.solidus, material.liquidus, 10000.])
    h = material.enthalpy(temperature)
    assert np.all(np.diff(h) > 0)
    np.testing.assert_allclose(material.temperature_from_enthalpy(h), temperature,
                               rtol=1e-12, atol=1e-9)
    assert material.enthalpy(REFERENCE_T) == 0.


@pytest.mark.parametrize("name", MATERIALS)
def test_phase_change_includes_latent_heat_exactly_once(name):
    material = MATERIALS[name]
    # Integrate the exact piecewise-linear sensible cp in the mushy interval.
    t = np.unique(np.r_[material.solidus, material.liquidus,
                       [v for v in material.temperature_knots
                        if material.solidus < v < material.liquidus]])
    sensible = np.sum(np.diff(t) * (material.heat_capacity(t[:-1]) +
                                   material.heat_capacity(t[1:])) / 2)
    total = material.enthalpy(material.liquidus) - material.enthalpy(material.solidus)
    assert total - sensible == pytest.approx(material.latent)
    np.testing.assert_allclose(material.liquid_fraction(
        [material.solidus - 1., material.solidus,
         (material.solidus + material.liquidus) / 2,
         material.liquidus, material.liquidus + 1.]), [0., 0., .5, 1., 1.])


def test_primary_316l_phase_inputs_and_si_units():
    m = MATERIALS["316L"]
    assert m.rho == 7904.
    assert m.solidus == 1675.
    assert m.liquidus == 1708.
    assert m.latent == 290000.
    assert m.heat_capacity(1708.) == 847.
    assert m.conductivity(500.) == pytest.approx(8.98 + .0157 * 500 - 1.5e-6 * 500**2)


@pytest.mark.parametrize("name", MATERIALS)
def test_properties_positive_and_tails_explicitly_constant(name):
    m = MATERIALS[name]
    t = np.linspace(100., 10000., 2001)
    assert np.all(m.heat_capacity(t) > 0.)
    assert np.all(m.conductivity(t) > 0.)
    assert m.heat_capacity(10000.) == m.cp_values[-1]
    assert m.conductivity(10000.) == m.k_values[-1]


def test_titanium_primary_heat_capacity_and_fusion_unit_conversions():
    m = MATERIALS["Ti-6Al-4V"]
    # NBS equation reports J/(g K); the model must use J/(kg K).
    assert m.heat_capacity(1450.) == pytest.approx(
        1000*(1.3833 - 9.943e-4*1450. + 3.745e-7*1450.**2))
    assert m.heat_capacity(1900.) == pytest.approx(
        1000*(1.3833 - 9.943e-4*1900. + 3.745e-7*1900.**2))
    assert m.latent == 286000.  # Primary result is 286 J/g, not 286 J/kg.
    assert m.rho == 4422.
    assert m.solidus == 1878. and m.liquidus == 1928.
    assert m.heat_capacity(2300.) == 930.


def test_aluminium_volumetric_to_mass_thermal_model_is_consistent():
    m = MATERIALS["AlSi10Mg"]
    assert m.rho*m.heat_capacity(298.15) == pytest.approx(2.3e6)
    assert m.rho*m.heat_capacity(1000.) == pytest.approx(2.6e6)
    assert m.rho*m.latent == pytest.approx(1.06e9)
    diffusivity = m.conductivity(1000.)/(m.rho*m.heat_capacity(1000.))
    assert diffusivity == pytest.approx(1.88e-5 + 6.43e-9*1000.)
    assert m.solidus == 831. and m.liquidus == 867.
    assert "assum" in m.notes.lower()


def test_every_alloy_has_independent_illustrative_absorptivity():
    assert set(MATERIALS) == {"17-4PH", "316L", "Ti-6Al-4V", "AlSi10Mg"}
    assert set(ILLUSTRATIVE_ABSORPTIVITY) == set(MATERIALS)
    assert all(0 < a < 1 for a in ILLUSTRATIVE_ABSORPTIVITY.values())
    assert ILLUSTRATIVE_ABSORPTIVITY["Ti-6Al-4V"] == .4
    assert ILLUSTRATIVE_ABSORPTIVITY["AlSi10Mg"] == .15
