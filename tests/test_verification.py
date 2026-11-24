"""Tests exercise the production flux operator and material enthalpy tables."""
from heat_sim.verification import (
    conduction_benchmark, enthalpy_checks, flux_conservation, gaussian_normalization,
)


def test_gaussian_delivers_absorbed_power_without_renormalizing_boundaries():
    result = gaussian_normalization()
    assert result['passed'], result


def test_material_enthalpy_includes_latent_and_is_reversible():
    result = enthalpy_checks()
    assert all(case['passed'] for case in result.values()), result


def test_actual_conduction_operator_matches_known_solution_under_refinement():
    result = conduction_benchmark()
    assert result['passed'], result


def test_internal_fluxes_cancel_and_boundary_losses_match_energy_change():
    result = flux_conservation()
    assert result['passed'], result
