import math
import pytest

from heat_sim.workflows import geometry_distance, uncertainty_scenarios


def test_no_ranking_for_unresolved_or_nonfinite_pool():
    assert geometry_distance(0, 40, 100, 50) is None
    assert geometry_distance(100, 40, 0, 50) is None
    assert geometry_distance(float("nan"), 40, 100, 50) is None
    assert geometry_distance(100, 50, 100, 50) == 0
    assert geometry_distance(110, 60, 100, 50) == pytest.approx(math.hypot(.1, .2))


def test_uncertainty_changes_alloy_absorptivity_independently():
    scenarios = dict(uncertainty_scenarios({"absorptivity_17": [.2,.4], "absorptivity_316": [.3,.5], "beam_um": [60,100], "powder_k_w_mk": [.5,2]}))
    assert len(scenarios) == 9
    assert scenarios["absorptivity_17 low"]["absorptivity_316"] == .4
    assert scenarios["absorptivity_316 high"]["absorptivity_17"] == pytest.approx(.3)
    assert scenarios["beam_um high"]["beam_um"] == 100
