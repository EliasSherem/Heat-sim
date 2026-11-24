"""Persistent whole-build finite-volume enthalpy replay of imported scan vectors.

Coordinates are SI and z is positive UP from the plate interface. This coarse
homogenised model does not resolve microscopic melt pools. See BUILD_MODEL.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import math
from pathlib import Path
import time

import numpy as np
from numba import njit
import psutil
from scipy.sparse.linalg import LinearOperator, cg

from .materials import MATERIALS
from .solver import SimulationCancelled, SimulationError, gaussian_cell_weights
from .storage import atomic_json


@dataclass(frozen=True)
class BuildConfig:
    alloy: str = "17-4PH"
    substrate: str = "17-4PH"
    absorptivity: float | None = None
    power_factor: float = 1.0
    power_scale: float = 1.0
    speed_scale: float = 1.0
    section_overrides: dict = field(default_factory=dict)
    beam_diameter_m: float = 80e-6
    wavelength_nm: float = 1070.
    initial_k: float = 353.15
    powder_packing: float = .55
    powder_k_w_mk: float = 1.
    spacing_m: float = .0005
    z_spacing_m: float = .00024
    plate_depth_m: float = .0015
    padding_m: float = .001
    layer_start: int = 0
    layer_end: int | None = 3
    warm_start_assumed: bool = False
    jump_speed_m_s: float = 5.
    jump_delay_s: float = .0001
    recoat_time_s: float = 3.
    final_cooldown_s: float = .1
    max_event_s: float = .005
    frame_interval_s: float = .1
    max_frames: int = 120
    convection_w_m2k: float = 15.
    emissivity: float = .35
    support_sections: tuple[int, ...] = ()
    hatch_m: float = 60e-6
    time_limit_s: float | None = None
    region_xy_m: tuple[float,float,float,float] | None = None
    max_vectors: int | None = None
    max_steps: int = 100_000
    geometry_file: str | None = None
    geometry_unit_scale: float = .001
    geometry_translation_m: tuple[float,float,float] = (0.,0.,0.)

    def validate(self, job=None):
        numeric=[v for v in asdict(self).values() if isinstance(v,(int,float))]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("Build inputs must be finite")
        if self.alloy not in MATERIALS or self.substrate not in MATERIALS:
            raise ValueError("Unsupported layer/substrate material")
        if self.absorptivity is None or not 0 < self.absorptivity <= 1:
            raise ValueError("Supply an independent assumed/measured absorptivity for the selected alloy")
        if not 0 < self.power_scale <= 3 or not .1 <= self.speed_scale <= 5 or not 0 < self.power_factor <= 2:
            raise ValueError("Invalid power/speed/delivered-power multiplier")
        if not isinstance(self.section_overrides,dict):
            raise ValueError("Region overrides must be a dictionary")
        for section,settings in self.section_overrides.items():
            text=str(section)
            if not text.isascii() or not text.isdigit() or not 0<=int(text)<=32767 or text!=str(int(text)):
                raise ValueError("Region override IDs must be integers from 0 to 32767")
            if not isinstance(settings,dict) or set(settings)-{"power_w","speed_m_s"}:
                raise ValueError("Region overrides support power_w and speed_m_s only")
            for key,value in settings.items():
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError("Region power and speed must be finite numbers")
                valid=(0 < value <= 10_000) if key=="power_w" else (1e-6 <= value <= 100)
                if not valid:
                    raise ValueError("Region power must be positive and at most 10000 W; speed must be 0.000001–100 m/s")
        if not 10e-6 <= self.beam_diameter_m <= .001 or not 100 <= self.wavelength_nm <= 20000:
            raise ValueError("Invalid physical spot diameter or wavelength")
        if not 273.15 <= self.initial_k < min(MATERIALS[self.alloy].solidus,MATERIALS[self.substrate].solidus):
            raise ValueError("Initial temperature must be below both melting ranges and at least 273.15 K")
        if not .2 <= self.powder_packing <= 1 or not .05 <= self.powder_k_w_mk <= 30:
            raise ValueError("Invalid powder properties")
        if not 50e-6 <= self.spacing_m <= .005 or not 20e-6 <= self.z_spacing_m <= .002:
            raise ValueError("Whole-build spacings must be 50 µm–5 mm lateral, 20 µm–2 mm vertical")
        if not 50e-6 <= self.plate_depth_m <= .02 or not 0 <= self.padding_m <= .01:
            raise ValueError("Invalid plate depth or lateral padding")
        if self.layer_start < 0 or (self.layer_end is not None and self.layer_end <= self.layer_start):
            raise ValueError("Invalid layer range (end is exclusive)")
        if self.layer_start and not self.warm_start_assumed:
            raise ValueError("A nonzero starting layer requires explicit warm_start_assumed: earlier material starts at preheat without its history")
        if job and (self.layer_start >= len(job.layers) or (self.layer_end is not None and self.layer_end > len(job.layers))):
            raise ValueError("Layer range exceeds imported job")
        if job:
            heights=np.asarray([layer.z_m for layer in job.layers],dtype=float)
            if not np.all(np.isfinite(heights)) or np.any(heights<0) or np.any(np.diff(heights)<-1e-12):
                raise ValueError("Whole-build activation requires nonnegative, monotonically increasing layer Z from the plate interface")
        if not .01 <= self.jump_speed_m_s <= 100 or min(self.jump_delay_s,self.recoat_time_s,self.final_cooldown_s) < 0:
            raise ValueError("Invalid laser-off timing")
        if not 1e-6 <= self.max_event_s <= 1 or self.frame_interval_s <= 0 or not 2 <= self.max_frames <= 1000:
            raise ValueError("Invalid event resolution/frame controls")
        if not 0 <= self.convection_w_m2k <= 1000 or not 0 <= self.emissivity <= 1:
            raise ValueError("Invalid exposed-surface boundary inputs")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("Time limit must be positive or None")
        if self.max_vectors is not None and self.max_vectors <= 0:
            raise ValueError("Vector limit must be positive or None")
        if self.region_xy_m is not None:
            b=self.region_xy_m
            if len(b)!=4 or not all(math.isfinite(x) for x in b) or b[2]<=b[0] or b[3]<=b[1]:
                raise ValueError("Region must be (xmin,ymin,xmax,ymax), SI")
        if self.geometry_file is not None and (self.geometry_unit_scale<=0 or len(self.geometry_translation_m)!=3):
            raise ValueError("Invalid supplemental geometry units/translation")


_EVENT_DTYPE=np.dtype([("t0_s","f8"),("t1_s","f8"),("start_m","f8",(3,)),("end_m","f8",(3,)),
                       ("power_w","f8"),("speed_m_s","f8"),("layer_index","i4"),("section","i2"),
                       ("vector_index","i8"),("kind","u1")])
EVENT_NAMES={0:"activation",1:"scan",2:"jump",3:"recoat",4:"cooldown"}


def vector_settings(vectors,process_ids,config):
    """Apply explicit section recipes before common command/delivery factors.

    Absolute section settings apply to every source part using that section;
    omitted settings retain each imported vector's original value. Source
    arrays and geometric scan order are never modified.
    """
    powers=np.asarray(vectors[:,4],float).copy()
    speeds=np.asarray(vectors[:,5],float).copy()
    for section,settings in config.section_overrides.items():
        mask=np.asarray(process_ids)==int(section)
        if "power_w" in settings:powers[mask]=settings["power_w"]
        if "speed_m_s" in settings:speeds[mask]=settings["speed_m_s"]
    return powers*config.power_scale*config.power_factor,speeds*config.speed_scale


def build_timeline(job, config: BuildConfig, progress=None, cancel=None):
    """Controller-reconstructed vector order; missing laser-off times assumed.

    No laser vectors are resampled/reordered here. Thermal grouping is separate.
    """
    config.validate(job)
    selected=job.layers[config.layer_start:config.layer_end]
    count=sum(len(layer.vectors) for layer in selected)
    if config.max_vectors is not None:count=min(count,config.max_vectors)
    events=np.empty(2*count+2*len(selected)+2,dtype=_EVENT_DTYPE)
    n=0;t=0.;last=None;seen=0;stop=False
    def append(kind,duration,start,end,layer,section=-1,vector=-1,power=0.,speed=0.):
        nonlocal n,t,stop
        limit=config.time_limit_s
        if limit is not None and t>=limit:stop=True;return
        if limit is not None and t+duration>limit:
            fraction=(limit-t)/duration if duration else 0
            end=np.asarray(start)+(np.asarray(end)-start)*fraction
            duration=limit-t;stop=True
        events[n]=(t,t+duration,start,end,power,speed,layer,section,vector,kind)
        n+=1;t+=duration
    for ordinal,layer in enumerate(selected):
        if cancel and cancel():raise SimulationCancelled("Cancelled while reconstructing job timing")
        if progress and ordinal%max(1,len(selected)//100)==0:
            progress(ordinal/max(1,len(selected)),f"Reconstructing layer{layer.index} · {seen:,}/{count:,} actual vectors")
        z=layer.z_m+job.layer_thickness_m
        point=np.array([0.,0.,z]) if last is None else np.array([last[0],last[1],z])
        if ordinal and config.recoat_time_s:
            append(3,config.recoat_time_s,last,point,layer.index)
            if stop:break
        if last is not None:
            # Layer-height motion belongs to recoating/platform movement,
            # not an invented 3-D galvo flight at jump_speed_m_s.
            last=point.copy()
        append(0,0.,point,point,layer.index)
        if stop:break
        powers,speeds=vector_settings(layer.vectors,layer.process_ids,config)
        for vi,vec in enumerate(layer.vectors):
            if config.max_vectors is not None and seen>=config.max_vectors:stop=True;break
            start=np.array([vec[0],vec[1],z]);end=np.array([vec[2],vec[3],z])
            if last is not None:
                gap=float(np.linalg.norm(start[:2]-last[:2]))
                # Continuous segments in the same polyline do not acquire an
                # invented settle delay. Other jumps remain explicit assumptions.
                same_path=vi>0 and layer.path_ids[vi]==layer.path_ids[vi-1] and gap<1e-12
                delay=0. if same_path else config.jump_delay_s
                duration=gap/config.jump_speed_m_s+delay
                if duration:append(2,duration,last,start,layer.index)
                if stop:break
            speed=float(speeds[vi])
            length=float(np.linalg.norm(end-start))
            append(1,length/speed,start,end,layer.index,int(layer.process_ids[vi]),seen,
                   float(powers[vi]),speed)
            seen+=1;last=end
            if stop:break
        if stop:break
    if n and config.final_cooldown_s:
        point=events[n-1]["end_m"].copy()
        append(4,config.final_cooldown_s,point,point,int(events[n-1]["layer_index"]))
    return events[:n].copy()


def _grid(job,c):
    bounds=job.bounds_m
    if c.region_xy_m is None:xmin,ymin,xmax,ymax=bounds[0]-c.padding_m,bounds[1]-c.padding_m,bounds[3]+c.padding_m,bounds[4]+c.padding_m
    else:xmin,ymin,xmax,ymax=c.region_xy_m
    nx=max(2,math.ceil((xmax-xmin)/c.spacing_m));ny=max(2,math.ceil((ymax-ymin)/c.spacing_m))
    end=len(job.layers) if c.layer_end is None else c.layer_end
    height=max(layer.z_m+job.layer_thickness_m for layer in job.layers[:end])
    plate_nz=max(1,math.ceil(c.plate_depth_m/c.z_spacing_m))
    build_nz=max(1,math.ceil(height/c.z_spacing_m))
    nz=plate_nz+build_nz
    xe=xmin+np.arange(nx+1)*c.spacing_m;ye=ymin+np.arange(ny+1)*c.spacing_m
    ze=(np.arange(nz+1)-plate_nz)*c.z_spacing_m
    return xe,ye,ze,plate_nz


def estimate_build(job,config:BuildConfig):
    config.validate(job)
    xe,ye,ze,plate_nz=_grid(job,config)
    shape=[len(xe)-1,len(ye)-1,len(ze)-1];cells=math.prod(shape)
    selected=job.layers[config.layer_start:config.layer_end]
    vectors=sum(len(layer.vectors) for layer in selected)
    if config.max_vectors is not None:vectors=min(vectors,config.max_vectors)
    remaining=config.max_vectors
    mark=0.
    for layer in selected:
        take=len(layer.vectors) if remaining is None else min(remaining,len(layer.vectors))
        v=layer.vectors[:take]
        if len(v):
            _,speeds=vector_settings(v,layer.process_ids[:take],config)
            mark+=float(np.sum(np.linalg.norm(v[:,2:4]-v[:,:2],axis=1)/speeds))
        if remaining is not None:
            remaining-=take
            if remaining<=0:break
    # These are byte/work counts, not a promise of full-build runtime.
    return dict(grid_shape=shape,cells=cells,working_array_bytes=cells*8*30,
                raw_frame_bytes=cells*(4*6+1*3),max_saved_frame_bytes=cells*27*config.max_frames,
                selected_vector_count=vectors,selected_layer_count=len(selected),
                nominal_mark_time_s=mark,thermal_exposure_step_lower_bound=math.ceil(mark/config.max_event_s),
                plate_cells=plate_nz,spacing_m=config.spacing_m,z_spacing_m=config.z_spacing_m,
                note="Working-memory byte estimate excludes Python/import arrays, CG/JIT overhead, compression and geometry. Runtime must be benchmarked; off-time, adaptive retries and frames add work.")


@njit(cache=True)
def _network_matvec(flat,capacity,gx,gy,gz,environment,dt,shape):
    nx,ny,nz=shape
    out=np.empty_like(flat)
    a=flat.reshape((nx,ny,nz));b=out.reshape((nx,ny,nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                value=a[i,j,k];v=environment[i,j,k]*value
                if i>0:v+=gx[i-1,j,k]*(value-a[i-1,j,k])
                if i<nx-1:v+=gx[i,j,k]*(value-a[i+1,j,k])
                if j>0:v+=gy[i,j-1,k]*(value-a[i,j-1,k])
                if j<ny-1:v+=gy[i,j,k]*(value-a[i,j+1,k])
                if k>0:v+=gz[i,j,k-1]*(value-a[i,j,k-1])
                if k<nz-1:v+=gz[i,j,k]*(value-a[i,j,k+1])
                b[i,j,k]=capacity[i,j,k]*value+dt*v
    return out


def _material_field(materials,plate,temperature,method):
    result=np.asarray(getattr(materials[0],method)(temperature),dtype=float)
    result[plate]=getattr(materials[1],method)(temperature[plate])
    return result


def _temperature(energy,mass,materials,plate,h0,initial):
    specific=np.divide(energy,mass,out=np.zeros_like(energy),where=mass>0)+h0
    result=_material_field(materials,plate,specific,"temperature_from_enthalpy")
    result[mass==0]=initial
    return result


def _conductances(T,active,consolidated,plate,materials,c,top_index):
    bulk=_material_field(materials,plate,T,"conductivity")
    state=np.divide(consolidated,active,out=np.zeros_like(active),where=active>0)
    k=c.powder_k_w_mk+(bulk-c.powder_k_w_mk)*np.clip(state,0,1)
    k[plate]=bulk[plate]
    k[active<=0]=0.
    dx=c.spacing_m;dz=c.z_spacing_m
    def harmonic(a,b):return np.divide(2*a*b,a+b,out=np.zeros_like(a),where=(a+b)>0)
    gx=harmonic(k[:-1],k[1:])*dz*np.minimum(active[:-1],active[1:])
    gy=harmonic(k[:,:-1],k[:,1:])*dz*np.minimum(active[:,:-1],active[:,1:])
    denom=np.divide(active[:,:,:-1],k[:,:,:-1],out=np.zeros_like(k[:,:,:-1]),where=k[:,:,:-1]>0)+np.divide(active[:,:,1:],k[:,:,1:],out=np.zeros_like(k[:,:,1:]),where=k[:,:,1:]>0)
    gz=np.divide(2*dx*dx,dz*denom,out=np.zeros_like(denom),where=(denom>0)&(active[:,:,:-1]>0)&(active[:,:,1:]>0))
    env=np.zeros_like(T)
    env[:,:,0]=2*k[:,:,0]*dx*dx/dz # fixed preheated lower plate face
    surface=T[:,:,top_index]
    radiation=c.emissivity*5.670374419e-8*(surface+c.initial_k)*(surface**2+c.initial_k**2)
    env[:,:,top_index]+=(c.convection_w_m2k+radiation)*dx*dx
    return gx,gy,gz,env


def _implicit_step(energy,source_j,dt,mass,active,consolidated,plate,materials,h0,c,top_index):
    """Damped enthalpy Newton/lagged-k backward Euler, conservative flux audit."""
    oldT=_temperature(energy,mass,materials,plate,h0,c.initial_k)
    guess=oldT.copy()
    oldh=_material_field(materials,plate,oldT,"enthalpy")
    shape=energy.shape
    for iteration in range(30):
        guessh=_material_field(materials,plate,guess,"enthalpy")
        cp=_material_field(materials,plate,guess,"heat_capacity")
        for material,mask in ((materials[0],~plate),(materials[1],plate)):
            cp[mask]+=material.latent/(material.liquidus-material.solidus)*((guess[mask]>material.solidus)&(guess[mask]<material.liquidus))
        capacity=mass*np.maximum(cp,1.)
        capacity[mass==0]=1.
        gx,gy,gz,env=_conductances(guess,active,consolidated,plate,materials,c,top_index)
        diagonal=capacity+dt*env
        diagonal[:-1]+=dt*gx;diagonal[1:]+=dt*gx
        diagonal[:,:-1]+=dt*gy;diagonal[:,1:]+=dt*gy
        diagonal[:,:,:-1]+=dt*gz;diagonal[:,:,1:]+=dt*gz
        rhs=(capacity*guess-mass*(guessh-oldh)+source_j+dt*env*c.initial_k).ravel()
        operator=LinearOperator((energy.size,energy.size),matvec=lambda x:_network_matvec(x,capacity,gx,gy,gz,env,dt,shape),dtype=float)
        pre=LinearOperator(operator.shape,matvec=lambda x:x/diagonal.ravel(),dtype=float)
        flat,info=cg(operator,rhs,x0=guess.ravel(),M=pre,rtol=2e-10,atol=1e-10,maxiter=400)
        if info!=0:raise SimulationError(f"Whole-build CG failed to converge (code {info})")
        proposed=flat.reshape(shape)
        zero=np.zeros_like(capacity)
        def residual_norm(value):
            net=_network_matvec(value.ravel(),zero,gx,gy,gz,env,1.,shape).reshape(shape)-env*c.initial_k
            residual=mass*(_material_field(materials,plate,value,"enthalpy")-oldh)+dt*net-source_j
            return float(np.linalg.norm(residual.ravel()))
        before=residual_norm(guess)
        damping=1.
        for _ in range(16):
            newT=guess+damping*(proposed-guess)
            if (np.all(np.isfinite(newT)) and newT.max()<=14000 and newT.min()>=c.initial_k-.1
                    and residual_norm(newT)<=before*(1-1e-4*damping)+1e-12):
                break
            damping*=.5
        else:raise SimulationError("Whole-build enthalpy Newton line search failed")
        if np.max(abs(newT-guess))<.015:
            # Use converged face fluxes directly for stored enthalpy. Internal
            # transfers cancel exactly; conservation does not depend on CG sum.
            net=_network_matvec(newT.ravel(),zero,gx,gy,gz,env,1.,shape).reshape(shape)-env*c.initial_k
            new_energy=energy+source_j-dt*net
            corrected=_temperature(new_energy,mass,materials,plate,h0,c.initial_k)
            if np.max(abs(corrected-newT))<.05:
                loss=dt*float(np.sum(env*(newT-c.initial_k)))
                return new_energy,corrected,loss,iteration+1
        guess=newT
    raise SimulationError("Whole-build nonlinear enthalpy iteration failed; reduce max_event_s")


def _activate(active,mass,energy,top_z,z_edges,plate_nz,c,material):
    previous=mass.sum()
    fraction=np.clip((top_z-z_edges[plate_nz:-1])/c.z_spacing_m,0.,1.)
    # A physical layer exactly on a grouped-cell boundary can be a few ulps
    # above it (e.g.125*60µm vs5*1.5mm). Do not activate a zero-mass ghost
    # layer and place the exposed-surface boundary on that ill-conditioned cell.
    fraction[fraction<1e-12]=0.
    fraction[fraction>1.-1e-12]=1.
    active[:,:,plate_nz:]=fraction[None,None,:]
    mass[:,:,plate_nz:]=material.rho*c.powder_packing*c.spacing_m**2*c.z_spacing_m*fraction[None,None,:]
    # Added powder has zero excess enthalpy relative to the common preheat;
    # existing stored energy is deliberately retained, never reset.
    return float(mass.sum()-previous)


def _source_segment(source,coverage,consolidation,support_coverage,start,end,duration,power,section,c,xe,ye,z_index,layer_fraction,material):
    """Integrate actual moving Gaussian using short spatial midpoint segments."""
    length=float(np.linalg.norm(end[:2]-start[:2]))
    samples=max(1,math.ceil(length/min(c.spacing_m/3,c.beam_diameter_m/2)))
    absorbed=0.;nominal=c.absorptivity*power*duration
    # Subgrid consolidation is limited by the incident energy needed to bring
    # initially preheated powder to liquidus; it is NOT resolved fusion.
    needed=max(float(material.enthalpy(material.liquidus)-material.enthalpy(c.initial_k)),1.)
    swept_volume=length*c.hatch_m*(layer_fraction*c.z_spacing_m)
    energy_limit=min(1.,nominal/max(material.rho*c.powder_packing*swept_volume*needed,1e-30)) if power>0 else 0.
    for s in range(samples):
        point=start+(end-start)*((s+.5)/samples)
        if nominal:
            weights=gaussian_cell_weights(xe,ye,point[0],point[1],c.beam_diameter_m/2)
            addition=nominal/samples*weights
            source[:,:,z_index]+=addition
            absorbed+=float(addition.sum())
        ix=int(np.searchsorted(xe,point[0])-1);iy=int(np.searchsorted(ye,point[1])-1)
        if 0<=ix<source.shape[0] and 0<=iy<source.shape[1]:
            area_fraction=length/samples*c.hatch_m/c.spacing_m**2*layer_fraction
            coverage[ix,iy,z_index]+=area_fraction
            consolidation[ix,iy,z_index]+=area_fraction*energy_limit
            if section in c.support_sections:support_coverage[ix,iy,z_index]+=area_fraction
    return absorbed,nominal


def _timeline_json(events):
    return [dict(t0_s=float(e["t0_s"]),t1_s=float(e["t1_s"]),start_m=e["start_m"].tolist(),end_m=e["end_m"].tolist(),
                 laser_on=bool(e["kind"]==1 and e["power_w"]>0),layer_index=int(e["layer_index"]),section=int(e["section"]),
                 power_w=float(e["power_w"]),speed_m_s=float(e["speed_m_s"]),kind=EVENT_NAMES[int(e["kind"])],vector_index=int(e["vector_index"])) for e in events]


def run_build(job,config:BuildConfig,folder,progress=None,cancel=None):
    """Calculate and atomically persist actual states for incremental viewing.

    progress(fraction,message); cancel()->bool. Failures write manifest status
    and raise; only fully converged frames appear in the manifest.
    """
    c=config;c.validate(job);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    frames_dir=folder/"frames";frames_dir.mkdir(exist_ok=True)
    progress=progress or (lambda f,m:None);cancel=cancel or (lambda:False)
    started=time.perf_counter();rss=psutil.Process();peak_rss=rss.memory_info().rss
    estimate=estimate_build(job,c)
    if estimate["cells"]>2_000_000:raise ValueError("Whole-build grid exceeds2millioncells; choose coarser spacing or a smaller range/region")
    if estimate["max_saved_frame_bytes"]>8*1024**3:raise ValueError("Raw saved frames could exceed8GiB; reduce max_frames/grid size")
    progress(0.,"Reconstructing scan timing and allocating persistent thermal grid")
    try:
        events=build_timeline(job,c,progress=lambda f,m:progress(.02*f,m),cancel=cancel)
    except Exception as exc:
        atomic_json(folder/"manifest.json",dict(schema=1,status="cancelled" if isinstance(exc,SimulationCancelled) else "failed",
                    error=f"{type(exc).__name__}:{exc}",frames=[],config=asdict(c),estimate=estimate,current_time_s=0.))
        raise
    if not len(events):raise ValueError("Selected range has no events")
    np.savez_compressed(folder/"timeline.npz",events=events)
    if len(events)<=30_000:atomic_json(folder/"timeline.json",_timeline_json(events))
    xe,ye,ze,plate_nz=_grid(job,c)
    xyz=[(e[:-1]+e[1:])/2 for e in (xe,ye,ze)]
    shape=tuple(len(v) for v in xyz);volume=c.spacing_m**2*c.z_spacing_m
    active=np.zeros(shape);active[:,:,:plate_nz]=1.
    plate=np.zeros(shape,dtype=bool);plate[:,:,:plate_nz]=True
    mats=[MATERIALS[c.alloy],MATERIALS[c.substrate]]
    mass=np.zeros(shape);mass[plate]=mats[1].rho*volume
    energy=np.zeros(shape);T=np.full(shape,c.initial_k)
    h0=np.full(shape,float(mats[0].enthalpy(c.initial_k)));h0[plate]=float(mats[1].enthalpy(c.initial_k))
    scanned=np.zeros(shape);consolidated=np.zeros(shape);consolidated[plate]=1.
    support=np.zeros(shape)
    planned_geometry=None
    if c.geometry_file:
        from .geometry import rasterize_stl
        planned_geometry=rasterize_stl(c.geometry_file,*xyz,unit_scale=c.geometry_unit_scale,translation_m=c.geometry_translation_m)
    activation_mass=0.
    if c.layer_start:
        prior_top=job.layers[c.layer_start].z_m
        activation_mass+=_activate(active,mass,energy,prior_top,ze,plate_nz,c,mats[0])
        # Starting later explicitly assumes a cold/preheated prior build, with
        # no invented thermal history. Prior geometry uses actual vector raster.
        zero=np.zeros(shape)
        for layer in job.layers[:c.layer_start]:
            zi=min(shape[2]-1,max(plate_nz,int(np.searchsorted(ze,layer.z_m+job.layer_thickness_m*.5)-1)))
            for vec,pid in zip(layer.vectors,layer.process_ids):
                a=np.r_[vec[:2],layer.z_m];b=np.r_[vec[2:4],layer.z_m]
                _source_segment(zero,scanned,consolidated,support,a,b,0.,0.,int(pid),c,xe,ye,zi,job.layer_thickness_m/c.z_spacing_m,mats[0])
        consolidated[:]=np.minimum(scanned,active);consolidated[plate]=1.
    total_time=float(events[-1]["t1_s"])
    frame_interval=max(c.frame_interval_s,total_time/max(1,c.max_frames-1))
    digest=sha256()
    for source_name in ("build_solver.py","solver.py","materials.py","job_import.py","geometry.py"):
        code=Path(__file__).with_name(source_name)
        if code.exists():digest.update(code.read_bytes())
    material_data={m.name:dict(rho_kg_m3=m.rho,solidus_k=m.solidus,liquidus_k=m.liquidus,latent_j_kg=m.latent,
                              notes=m.notes,sources=m.sources,table=m.table()) for m in mats}
    manifest=dict(schema=1,status="running",source_fingerprint=digest.hexdigest(),config=asdict(c),material_data=material_data,
                  job=dict(name=job.name,source_path=job.source_path,source_sha256=job.source_sha256,format=job.format,metadata=job.metadata),
                  estimate=estimate,frames=[],timeline_file="timeline.json" if len(events)<=30000 else "timeline.npz",
                  timeline_event_count=len(events),timeline_json_available=len(events)<=30000,physical_duration_s=total_time,
                  coordinate_convention="SI; z positive upward from plate interface at0; platez<0",
                  fidelity="Coarse homogenised heat history; vector order reconstructed; missing laser-off timing assumed. Not a microscopic melt-pool/porosity prediction.",
                  consolidation_definition="Energy-constrained subgrid scan-history proxy; not resolved fusion or validated density.",
                  timing_provenance=dict(recoat_time_s=c.recoat_time_s,source="user approximate estimate around3seconds including recoating" if c.recoat_time_s==3. else "editable assumed inter-layer total pause",jumps="assumed jump speed and settle delay; no machine timestamp capture"),
                  geometry_provenance="supplemental user-aligned STL planned material; exposure controls consolidation" if c.geometry_file else "inferred scan-vector raster; not watertight CAD",
                  assumptions=["Job W,mm/s,mm unit interpretation provisional unless verified.","Gaussian1/e²spot diameter, absorptivity, delivered power, wavelength, powder, plate and environmental inputs remain assumed until measured.",
                               "Jump speed/delay and recoating time are editable assumptions, not captured machine timestamps.","Geometry inferred from scan raster; no recovered watertight CAD or guaranteed support identity.",
                               "New powder mixed into partially filled grouped-z cells; prior stored enthalpy is retained.","Sidewalls insulated; bottom fixed at preheat; exposed top convection/radiation to preheat environment.",
                               "Energy-constrained consolidation is a subgrid history approximation; coarse voxel temperature is not track peak temperature."])
    if c.layer_start:manifest["assumptions"].append("Earlier layers assumed present at uniform preheat; their actual thermal history is omitted.")
    if c.region_xy_m:manifest["assumptions"].append("Cropped region has insulated sides; surrounding-part heat transfer is omitted.")
    atomic_json(folder/"config.json",asdict(c));atomic_json(folder/"manifest.json",manifest)
    source=np.zeros(shape);pending_scan=np.zeros(shape);pending_cons=np.zeros(shape);pending_support=np.zeros(shape)
    absorbed_j=nominal_j=loss_j=0.;steps=iterations=0;max_temperature=c.initial_k;sum_peak_temperature=0.;temp_peak_count=0
    next_frame=0.;current_time=0.;pending_dt=0.;current_layer=int(events[0]["layer_index"]);last_event=0
    top_index=max(0,int(np.flatnonzero(active[0,0]>0)[-1]));laser_position=None;laser_on=False
    frame_peak=np.full(shape,c.initial_k)

    def save_frame(force=False):
        nonlocal next_frame,peak_rss
        if not force and current_time+1e-12<next_frame:return
        if len(manifest["frames"])>=c.max_frames and not force:return
        if manifest["frames"] and abs(manifest["frames"][-1]["time_s"]-current_time)<1e-12:return
        index=len(manifest["frames"]);relative=f"frames/{index:05d}.npz"
        target=folder/relative;temporary=target.with_suffix(".tmp.npz")
        liquid=_material_field(mats,plate,T,"liquid_fraction")
        np.savez_compressed(temporary,temperature_k=T.astype("f4"),deposited_fraction=active.astype("f4"),active_fraction=active.astype("f4"),
                            consolidated_fraction=consolidated.astype("f4"),scanned_fraction=scanned.astype("f4"),thermal_liquid_fraction=liquid.astype("f4"),
                            part_mask=(((planned_geometry if planned_geometry is not None else scanned>1e-7)&(active>0))|((scanned>1e-7)&~plate)),support_mask=(support>1e-7)&~plate,plate_mask=plate,
                            x_m=xyz[0],y_m=xyz[1],z_m=xyz[2],time_s=current_time,layer_index=current_layer,
                            laser_position_m=np.asarray(laser_position if laser_position is not None else [np.nan]*3),laser_on=laser_on)
        temporary.replace(target)
        manifest["frames"].append(dict(file=relative,time_s=current_time,layer_index=current_layer,laser_position_m=laser_position,laser_on=laser_on,event_index=last_event))
        next_frame=current_time+frame_interval
        manifest["current_time_s"]=current_time;manifest["solver_steps"]=steps
        peak_rss=max(peak_rss,rss.memory_info().rss)
        atomic_json(folder/"manifest.json",manifest)

    def integrate(dt,heat,retry=0):
        nonlocal energy,T,loss_j,steps,iterations,max_temperature,sum_peak_temperature,temp_peak_count,frame_peak
        if cancel():raise SimulationCancelled("Build simulation cancelled; completed frames retained")
        if steps>=c.max_steps:raise SimulationError("Build step limit reached; reduce range or increase grouped event duration")
        try:
            # Future layers remain allocated for stable frame geometry but do
            # not enter the linear solve. All already-active cells do enter.
            built=np.s_[:,:,:top_index+1]
            energy_new,Tnew,loss,niter=_implicit_step(energy[built].copy(),heat[built].copy(),dt,
                mass[built].copy(),active[built].copy(),consolidated[built].copy(),
                plate[built].copy(),mats,h0[built].copy(),c,top_index)
        except SimulationError:
            if retry>=8 or dt<2e-6:raise
            integrate(dt/2,heat/2,retry+1);integrate(dt/2,heat/2,retry+1);return
        energy[built]=energy_new;T[built]=Tnew;loss_j+=loss;steps+=1;iterations+=niter
        if not np.all(np.isfinite(energy_new)):raise SimulationError("Non-finite whole-build enthalpy")
        max_temperature=max(max_temperature,float(Tnew.max()));sum_peak_temperature+=float(Tnew.max());temp_peak_count+=1
        np.maximum(frame_peak[built],Tnew,out=frame_peak[built])

    def flush():
        nonlocal pending_dt,source,pending_scan,pending_cons,pending_support
        if pending_dt<=0:return
        integrate(pending_dt,source)
        scanned[:]=np.minimum(active,scanned+pending_scan)
        consolidated[:]=np.minimum(active,consolidated+pending_cons)
        consolidated[plate]=1.
        # Coarse resolved melting also irreversibly increases conductivity state.
        consolidated[:]=np.maximum(consolidated,active*_material_field(mats,plate,T,"liquid_fraction"))
        support[:]=np.minimum(active,support+pending_support)
        pending_dt=0.;source.fill(0);pending_scan.fill(0);pending_cons.fill(0);pending_support.fill(0)

    save_frame(True)
    try:
        for ei,event in enumerate(events):
            last_event=ei;current_layer=int(event["layer_index"])
            kind=int(event["kind"]);t0=float(event["t0_s"]);t1=float(event["t1_s"])
            if kind==0:
                flush();current_time=t0
                activation_mass+=_activate(active,mass,energy,float(event["end_m"][2]),ze,plate_nz,c,mats[0])
                T=_temperature(energy,mass,mats,plate,h0,c.initial_k)
                top_index=max(0,int(np.flatnonzero(active[0,0]>0)[-1]))
                continue
            remaining=t1-t0;segment_start=t0
            if kind in (3,4):flush()
            while remaining>1e-12:
                # Long laser-off periods use implicit cooling steps up to0.25s;
                # exposure/jump blocks are bounded by max_event_s.
                maximum=max(c.max_event_s,.25) if kind in (3,4) else c.max_event_s
                available=maximum-pending_dt
                part=min(remaining,max(available,1e-12))
                if next_frame>segment_start+1e-12:
                    part=min(part,next_frame-segment_start)
                f0=(segment_start-t0)/(t1-t0);f1=(segment_start+part-t0)/(t1-t0)
                a=event["start_m"]+(event["end_m"]-event["start_m"])*f0
                b=event["start_m"]+(event["end_m"]-event["start_m"])*f1
                if kind==1:
                    zi=min(shape[2]-1,max(plate_nz,int(np.searchsorted(ze,float(a[2])-job.layer_thickness_m*.5)-1)))
                    absorbed,nominal=_source_segment(source,pending_scan,pending_cons,pending_support,a,b,part,float(event["power_w"]),int(event["section"]),c,xe,ye,zi,job.layer_thickness_m/c.z_spacing_m,mats[0])
                    absorbed_j+=absorbed;nominal_j+=nominal
                pending_dt+=part;current_time=segment_start+part
                laser_on=kind==1 and event["power_w"]>0
                laser_position=b.tolist() if kind in (1,2) else None
                if pending_dt>=maximum-1e-12 or current_time>=next_frame-1e-12:
                    flush();save_frame()
                segment_start+=part;remaining-=part
            if ei%max(1,len(events)//200)==0:
                progress(min(.999,current_time/max(total_time,1e-20)),f"Layer{current_layer} · {current_time:.3f}/{total_time:.3f}s physical · event{ei:,}/{len(events):,} · {steps:,} thermal steps")
                if cancel():raise SimulationCancelled("Build simulation cancelled; completed frames retained")
        flush();current_time=total_time;save_frame(True)
        stored=float(energy.sum());residual=absorbed_j-loss_j-stored
        if abs(residual)>max(1e-7,absorbed_j*2e-6):raise SimulationError(f"Build global energy balance failed:{residual:g}J")
        np.savez_compressed(folder/"peak-fields.npz",temperature_k=frame_peak,x_m=xyz[0],y_m=xyz[1],z_m=xyz[2])
        active_cells=active>0
        material_cells=(scanned>1e-9)&~plate
        if not np.any(material_cells):material_cells=active_cells&~plate
        normalizer=max(float(mats[0].enthalpy(mats[0].liquidus)-mats[0].enthalpy(c.initial_k)),1.)
        normalized_peak=(np.asarray(mats[0].enthalpy(frame_peak))-float(mats[0].enthalpy(c.initial_k)))/normalizer
        latent_field=mass*_material_field(mats,plate,T,"liquid_fraction")
        latent_field[~plate]*=mats[0].latent;latent_field[plate]*=mats[1].latent
        stored_latent=float(latent_field.sum())
        history_weights=scanned[material_cells]
        if not np.any(history_weights):history_weights=np.ones(int(material_cells.sum()))
        metrics=dict(peak_temperature_k=max_temperature,mean_step_peak_temperature_k=sum_peak_temperature/max(1,temp_peak_count),
                     mean_peak_temperature_k=float(frame_peak[active_cells].mean()),stored_energy_j=stored,absorbed_energy_j=absorbed_j,
                     stored_latent_energy_j=stored_latent,stored_sensible_energy_j=stored-stored_latent,
                     nominal_absorbed_energy_j=nominal_j,source_captured_fraction=absorbed_j/nominal_j if nominal_j else 1.,boundary_loss_j=loss_j,
                     energy_residual_j=residual,energy_error_fraction=abs(residual)/max(absorbed_j,1e-12),
                     activated_powder_mass_kg=activation_mass,activation_excess_enthalpy_j=0.,
                     thermal_steps=steps,nonlinear_iterations=iterations,runtime_s=time.perf_counter()-started,
                     peak_memory_mb=max(peak_rss,rss.memory_info().rss)/1024**2,
                     physical_duration_s=total_time,frame_count=len(manifest["frames"]),grid_shape=list(shape),
                     mean_material_peak_enthalpy_ratio=float(np.average(normalized_peak[material_cells],weights=history_weights)) if np.any(material_cells) else 0.,
                     max_material_peak_enthalpy_ratio=float(normalized_peak[material_cells].max()) if np.any(material_cells) else 0.,
                     high_temperature_caution=max_temperature>min(mats[0].valid_temperature_max,2900.),
                     experimental_validation=False,microscopic_geometry_resolved=False)
        manifest.update(status="complete",metrics=metrics)
        atomic_json(folder/"manifest.json",manifest)
        progress(1.,f"Build complete · {steps:,} thermal steps · {metrics['runtime_s']:.2f}s measured runtime")
        return manifest
    except (SimulationCancelled,Exception) as exc:
        manifest.update(status="cancelled" if isinstance(exc,SimulationCancelled) else "failed",error=f"{type(exc).__name__}:{exc}",current_time_s=current_time)
        atomic_json(folder/"manifest.json",manifest)
        raise
