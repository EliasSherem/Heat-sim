"""Physical invariants, geometry definitions, and cancellation contract."""
from dataclasses import replace
import numpy as np
import pytest

from heat_sim.solver import (SimulationConfig, SimulationCancelled, simulate,
                             gaussian_cell_weights, top_hat_cell_weights,
                             _geometry)


def test_gaussian_power_and_radius_convention():
    w = 40e-6
    e = np.linspace(-8*w,8*w,101)
    weights = gaussian_cell_weights(e,e,0,0,w)
    assert weights.sum() == pytest.approx(1.,abs=1e-14)
    # Disk r=w contains1-exp(-2) analytically; a centreline exponent is -2.
    narrow = np.array([-1e-9,1e-9])
    centre = gaussian_cell_weights(narrow,narrow,0,0,w)[0,0]
    shifted = gaussian_cell_weights(narrow,narrow,w,0,w)[0,0]
    assert shifted/centre == pytest.approx(np.exp(-2),rel=1e-6)


def test_top_hat_integrates_physical_disk():
    e=np.linspace(-150e-6,150e-6,61)
    weights=top_hat_cell_weights(e,e,3e-6,-2e-6,40e-6)
    assert weights.sum() == pytest.approx(1,rel=1e-4)
    assert np.all(weights>=0)


def test_geometry_interpolates_depth_from_powder_top():
    y=np.array([-40.,-20.,0.,20.,40.])*1e-6
    z=np.array([10.,30.,50.,70.,90.])*1e-6
    section=np.full((5,5),300.)
    section[1:4,:3]=2000.
    width,depth,penetration=_geometry(section,y,z,1500.,60e-6)
    assert width==pytest.approx(51.76470588235)
    assert depth==pytest.approx(55.88235294118)
    assert penetration==0.


def test_config_rejects_unsupported_or_nonfinite_inputs():
    for fields in [dict(beam_diameter_m=0),dict(profile="unknown"),
                   dict(absorptivity=1.1),dict(power_w=np.nan),
                   dict(dt_scale=1.2),dict(tracks=3)]:
        with pytest.raises(ValueError):
            replace(SimulationConfig(),**fields).validate()


def test_cancel_discards_partial_run():
    with pytest.raises(SimulationCancelled):
        simulate(SimulationConfig(),cancel=lambda:True)


def test_zero_power_preserves_initial_and_energy():
    r=simulate(SimulationConfig(power_w=0,cooldown_s=0))
    assert np.max(abs(r.temperature_k-r.config.initial_k))<0.2
    assert r.metrics["width_um"]==0
    assert r.metrics["absorbed_energy_j"]==0
    assert abs(r.metrics["stored_energy_j"])<1e-15


def test_adjacent_tracks_deliver_full_scan_dwell():
    c=SimulationConfig(power_w=0,tracks=2,intertrack_delay_s=0,cooldown_s=0)
    r=simulate(c)
    assert r.metrics["laser_active_s"]==pytest.approx(2*0.5e-3/c.speed_m_s,abs=1e-15)


def test_snapshot_samples_actual_chronological_states():
    frames=[]
    r=simulate(SimulationConfig(),snapshot=frames.append,snapshot_count=7)
    reference=simulate(SimulationConfig())
    np.testing.assert_array_equal(r.temperature_k,reference.temperature_k)
    np.testing.assert_array_equal(r.peak_temperature_k,reference.peak_temperature_k)
    assert r.metrics["width_um"]==reference.metrics["width_um"]
    assert r.metrics["depth_um"]==reference.metrics["depth_um"]
    assert len(frames)==7
    assert frames[0]["time_s"]==0.
    assert np.all(frames[0]["temperature_k"]==r.config.initial_k)
    assert all(a["time_s"] < b["time_s"] for a,b in zip(frames,frames[1:]))
    expected_duration=0.5e-3/r.config.speed_m_s+r.config.cooldown_s
    assert frames[-1]["time_s"]==pytest.approx(expected_duration)
    np.testing.assert_array_equal(frames[-1]["temperature_k"],r.temperature_k)
    assert any(f["laser_on"] for f in frames)
    assert frames[-1]["laser_on"] is False
    assert frames[-1]["laser_x_m"] is None
    assert not frames[0]["x_m"].flags.writeable
    assert frames[0]["liquidus_by_z_k"].shape==r.z_m.shape
    # Captured states are independent arrays; they are neither the final field
    # repeated nor the maximum-over-time footprint used for geometry.
    assert any(np.max(f["temperature_k"]) > np.max(r.temperature_k) for f in frames[1:-1])
    assert not np.shares_memory(frames[0]["temperature_k"],frames[-1]["temperature_k"])
