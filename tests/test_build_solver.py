from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator,cg

from heat_sim.build_solver import (BuildConfig,build_timeline,estimate_build,run_build,
                                   _activate,_network_matvec,_source_segment,_implicit_step)
from heat_sim.job_import import ImportedJob,ImportedLayer
from heat_sim.materials import MATERIALS
from heat_sim.solver import SimulationCancelled


def fixture_job():
    layers=[]
    for i in range(2):
        v=np.array([[0,0,.001,0,120,1.4],[.001,0,0,.00006,100,1.]])
        layers.append(ImportedLayer(i,i*60e-6,v,np.array([0,16]),np.array([0,0]),np.array([0,1])))
    return ImportedJob("fixture","none","testhash","test",60e-6,layers,{"order":"fixture-authoritative"})


def small_config(**kwargs):
    return replace(BuildConfig(absorptivity=.35,layer_end=2,recoat_time_s=.005,final_cooldown_s=.01,
                   spacing_m=.0005,z_spacing_m=.00024,plate_depth_m=.00048,padding_m=.0005,
                   max_event_s=.001,frame_interval_s=.001,max_frames=40,support_sections=(16,)),**kwargs)


def test_timeline_preserves_scan_sequence_and_speed_changes_physical_time():
    job=fixture_job();c=small_config(recoat_time_s=3.)
    events=build_timeline(job,c)
    scans=events[events["kind"]==1]
    assert scans["section"].tolist()==[0,16,0,16]
    np.testing.assert_array_equal(scans["power_w"],[120,100,120,100])
    expected=sum(layer.mark_time_s for layer in job.layers)
    assert np.sum(scans["t1_s"]-scans["t0_s"])==pytest.approx(expected)
    recoat=events[events["kind"]==3]
    assert len(recoat)==1
    assert recoat[0]["t1_s"]-recoat[0]["t0_s"]==pytest.approx(3.)
    faster=build_timeline(job,replace(c,speed_scale=2,power_scale=1.1))
    fs=faster[faster["kind"]==1]
    assert np.sum(fs["t1_s"]-fs["t0_s"])==pytest.approx(expected/2)
    np.testing.assert_allclose(fs["power_w"],scans["power_w"]*1.1)
    assert all(a<=b for a,b in zip(events["t1_s"][:-1],events["t0_s"][1:]))


def test_activation_adds_powder_mass_without_resetting_prior_energy():
    c=small_config();shape=(2,2,3)
    active=np.zeros(shape);mass=np.zeros(shape);energy=np.zeros(shape)
    z=np.array([-.00024,0,.00024,.00048])
    _activate(active,mass,energy,60e-6,z,1,c,MATERIALS[c.alloy])
    old_mass=mass.copy();energy[:,:,1]=.01
    old_energy=energy.copy()
    _activate(active,mass,energy,120e-6,z,1,c,MATERIALS[c.alloy])
    np.testing.assert_array_equal(energy,old_energy)
    np.testing.assert_allclose(mass[:,:,1],old_mass[:,:,1]*2)
    assert active[0,0,1]==pytest.approx(.5)


def test_activation_on_group_boundary_does_not_create_roundoff_ghost_layer():
    c=small_config(spacing_m=.002,z_spacing_m=.0015)
    shape=(2,2,8);active=np.zeros(shape);mass=np.zeros(shape);energy=np.zeros(shape)
    edges=(np.arange(9)-2)*c.z_spacing_m
    # Actual reference layer124: the two equivalent physical expressions differ
    # by one floating-point ulp and previously created activefraction5.8e-16.
    top=125*60e-6
    assert top>edges[7]
    _activate(active,mass,energy,top,edges,2,c,MATERIALS[c.alloy])
    assert np.all(active[:,:,7]==0)
    assert np.all(mass[:,:,7]==0)
    assert np.all(active[:,:,6]==1)


def test_moving_source_preserves_absorbed_energy_and_marks_path_only():
    c=small_config();shape=(8,8,1)
    arrays=[np.zeros(shape) for _ in range(4)]
    edges=np.linspace(-.002,.002,9)
    start=np.array([-.001,0,0]);end=np.array([.001,0,0])
    absorbed,nominal=_source_segment(*arrays,start,end,.002,120,16,c,edges,edges,0,.25,MATERIALS[c.alloy])
    assert absorbed==pytest.approx(c.absorptivity*120*.002,rel=1e-12)
    assert arrays[0].sum()==pytest.approx(absorbed)
    assert nominal==pytest.approx(absorbed)
    assert np.count_nonzero(arrays[1])<shape[0]*shape[1]
    assert arrays[3].sum()>0


def test_implicit_network_matches_insulated_cosine_conduction():
    nx,ny,nz=32,3,3;shape=(nx,ny,nz);dx=1e-4
    conductivity=20.;rho=7800.;cp=500.;alpha=conductivity/(rho*cp)
    mass_capacity=np.full(shape,rho*cp*dx**3)
    gx=np.full((nx-1,ny,nz),conductivity*dx)
    gy=np.full((nx,ny-1,nz),conductivity*dx)
    gz=np.full((nx,ny,nz-1),conductivity*dx)
    env=np.zeros(shape)
    x=(np.arange(nx)+.5)*dx;length=nx*dx
    initial=300+20*np.cos(np.pi*x/length)
    temp=np.broadcast_to(initial[:,None,None],shape).copy()
    dt=2e-5;steps=10
    operator=LinearOperator((temp.size,temp.size),matvec=lambda v:_network_matvec(v,mass_capacity,gx,gy,gz,env,dt,shape),dtype=float)
    for _ in range(steps):
        value,info=cg(operator,(mass_capacity*temp).ravel(),x0=temp.ravel(),rtol=1e-12,atol=1e-14)
        assert info==0;temp=value.reshape(shape)
    exact=300+20*np.exp(-alpha*np.pi**2*dt*steps/length**2)*np.cos(np.pi*x/length)
    assert np.max(abs(temp[:,1,1]-exact))<2e-4
    assert temp.mean()==pytest.approx(300,abs=1e-9)


def test_implicit_heating_resolves_fusion_enthalpy_instead_of_skipping_latent_heat():
    c=small_config(alloy="AlSi10Mg",substrate="AlSi10Mg",absorptivity=.15,initial_k=800.,convection_w_m2k=0,emissivity=0)
    material=MATERIALS[c.alloy];shape=(2,2,2)
    active=np.ones(shape);plate=np.zeros(shape,dtype=bool)
    mass=np.full(shape,material.rho*c.powder_packing*c.spacing_m**2*c.z_spacing_m)
    h0=np.full(shape,float(material.enthalpy(c.initial_k)))
    source=mass*(material.enthalpy(850.)-material.enthalpy(c.initial_k))
    energy,temperature,loss,_=_implicit_step(np.zeros(shape),source,1e-9,mass,active,np.zeros(shape),plate,[material,material],h0,c,1)
    np.testing.assert_allclose(temperature,850.,atol=1e-5)
    assert 0<float(material.liquid_fraction(temperature).mean())<1
    latent=float(np.sum(mass*material.latent*material.liquid_fraction(temperature)))
    assert latent>source.sum()*.5
    assert energy.sum()+loss==pytest.approx(source.sum(),abs=1e-12)


def test_build_energy_cooling_progressive_activation_and_saved_states(tmp_path):
    manifest=run_build(fixture_job(),small_config(),tmp_path)
    metrics=manifest["metrics"]
    assert manifest["status"]=="complete"
    assert metrics["energy_error_fraction"]<1e-8
    assert metrics["boundary_loss_j"]>0
    assert metrics["activation_excess_enthalpy_j"]==0
    assert metrics["microscopic_geometry_resolved"] is False
    fields=[]
    for entry in manifest["frames"]:
        with np.load(tmp_path/entry["file"],allow_pickle=False)as source:
            fields.append({k:source[k].copy() for k in source.files})
    assert len(fields)>=6
    assert fields[-1]["temperature_k"].max()<max(f["temperature_k"].max() for f in fields)
    for before,after in zip(fields,fields[1:]):
        assert np.all(after["deposited_fraction"]>=before["deposited_fraction"])
        assert np.all(after["consolidated_fraction"]>=before["consolidated_fraction"]-1e-7)
    assert fields[-1]["support_mask"].any()
    assert fields[-1]["part_mask"].any()
    saved=json.loads((tmp_path/"manifest.json").read_text())
    assert saved["frames"]==manifest["frames"]
    assert saved["frames"][-1]["time_s"]==pytest.approx(metrics["physical_duration_s"])


def test_build_cancel_keeps_only_completed_frames(tmp_path):
    with pytest.raises(SimulationCancelled):
        run_build(fixture_job(),small_config(),tmp_path,cancel=lambda:any((tmp_path/"frames").glob("*.npz")))
    manifest=json.loads((tmp_path/"manifest.json").read_text())
    assert manifest["status"]=="cancelled"
    assert len(manifest["frames"])==1


def test_build_requires_independent_absorptivity_and_explicit_warm_start():
    with pytest.raises(ValueError):BuildConfig().validate(fixture_job())
    with pytest.raises(ValueError):small_config(layer_start=1).validate(fixture_job())
    with pytest.raises(ValueError):small_config(alloy="AlSi10Mg",initial_k=900.).validate(fixture_job())
    estimate=estimate_build(fixture_job(),small_config(max_vectors=1))
    assert estimate["selected_vector_count"]==1
    assert estimate["nominal_mark_time_s"]==pytest.approx(.001/1.4)


def test_build_rejects_negative_or_decreasing_layer_heights():
    job=fixture_job()
    job.layers[0].z_m=-60e-6
    with pytest.raises(ValueError,match="layer Z"):
        small_config().validate(job)
    job.layers[0].z_m=120e-6
    with pytest.raises(ValueError,match="layer Z"):
        small_config().validate(job)
