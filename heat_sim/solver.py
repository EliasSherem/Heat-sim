"""Conservative, cell-centred 3-D finite-volume enthalpy solver (SI units)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import time
from typing import Callable

import numpy as np
import psutil
from numba import njit
from scipy.special import erf

from .materials import MATERIALS


class SimulationError(RuntimeError):
    """A run failed numerical checks; no geometry should be reported."""


class SimulationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class SimulationConfig:
    alloy: str = "17-4PH"
    substrate: str = "17-4PH"
    power_w: float = 120.0
    speed_m_s: float = 1.4
    beam_diameter_m: float = 80e-6
    profile: str = "gaussian"
    absorptivity: float = 0.35
    power_factor: float = 1.0
    layer_m: float = 60e-6
    hatch_m: float = 60e-6
    initial_k: float = 353.15
    powder_packing: float = 0.55
    powder_k_w_mk: float = 1.0
    mode: str = "coarse"
    domain_scale: float = 1.0
    dt_scale: float = 1.0
    convection_w_m2k: float = 15.0
    emissivity: float = 0.35
    tracks: int = 1
    intertrack_delay_s: float = 50e-6
    cooldown_s: float = 300e-6

    def validate(self):
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.alloy not in MATERIALS or self.substrate not in MATERIALS:
            raise ValueError("Choose a supported layer/substrate alloy")
        if self.mode not in ("coarse", "refined"):
            raise ValueError("Mode must be coarse or refined")
        if self.profile not in ("gaussian", "top_hat"):
            raise ValueError("Profile must be gaussian or top_hat")
        if not 0 < self.absorptivity <= 1 or not 0.2 <= self.powder_packing <= 1:
            raise ValueError("Absorptivity must be (0,1]; packing fraction [0.2,1]")
        if not 0 < self.power_factor <= 2 or not 0 <= self.power_w <= 1000:
            raise ValueError("Power must be 0–1000 W; delivered power factor (0,2]")
        if not 0.05 <= self.speed_m_s <= 10:
            raise ValueError("Speed must be 0.05–10 m/s")
        if not 20e-6 <= self.beam_diameter_m <= 250e-6:
            raise ValueError("Beam diameter must be 20–250 µm")
        if not 20e-6 <= self.layer_m <= 120e-6:
            raise ValueError("Layer thickness must be 20–120 µm")
        if not 10e-6 <= self.hatch_m <= 200e-6:
            raise ValueError("Hatch spacing must be 10–200 µm")
        if not 273.15 <= self.initial_k <= 1200:
            raise ValueError("Initial temperature must be 273.15–1200 K")
        if not 0.1 <= self.powder_k_w_mk <= 20:
            raise ValueError("Powder conductivity must be 0.1–20 W/(m K)")
        if not 0.75 <= self.domain_scale <= 2:
            raise ValueError("Domain scale must be 0.75–2")
        if not 0 < self.dt_scale <= 1:
            raise ValueError("Time step scale must be (0,1]")
        if self.tracks not in (1, 2):
            raise ValueError("Only one or two tracks are implemented")
        if not 0 <= self.convection_w_m2k <= 1000 or not 0 <= self.emissivity <= 1:
            raise ValueError("Invalid top convection or emissivity")
        if not 0 <= self.cooldown_s <= 0.005 or not 0 <= self.intertrack_delay_s <= 0.005:
            raise ValueError("Cooldown/delay must be 0–5 ms")


@dataclass
class SimulationResult:
    config: SimulationConfig
    metrics: dict
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    peak_temperature_k: np.ndarray
    temperature_k: np.ndarray
    cross_section_peak_k: np.ndarray
    top_peak_k: np.ndarray


def gaussian_cell_weights(x_edges, y_edges, xc, yc, radius):
    """Fraction of incident P in each xy cell; radius is the 1/e² radius.

    No discrete renormalisation: the sum is the fraction actually inside domain.
    """
    a = math.sqrt(2.0) / radius
    wx = 0.5 * np.diff(erf((np.asarray(x_edges) - xc) * a))
    wy = 0.5 * np.diff(erf((np.asarray(y_edges) - yc) * a))
    return wx[:, None] * wy[None, :]


_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


def top_hat_cell_weights(x_edges, y_edges, xc, yc, radius):
    """Disk-area fractions by 24-point Gauss quadrature in each x cell.

    The small quadrature error is not renormalised; recorded input energy uses
    precisely the flux actually applied to the mesh.
    """
    x0 = np.maximum(np.asarray(x_edges[:-1]) - xc, -radius)
    x1 = np.minimum(np.asarray(x_edges[1:]) - xc, radius)
    half = np.maximum(x1 - x0, 0.0) / 2
    qx = (x0 + x1)[:, None] / 2 + half[:, None] * _GL_X
    circle_y = np.sqrt(np.maximum(radius * radius - qx * qx, 0))
    lower = np.maximum(np.asarray(y_edges[:-1])[None, :, None] - yc, -circle_y[:, None, :])
    upper = np.minimum(np.asarray(y_edges[1:])[None, :, None] - yc, circle_y[:, None, :])
    area = np.sum(np.maximum(upper - lower, 0) * _GL_W, axis=-1) * half[:, None]
    return area / (np.pi * radius * radius)


@njit(cache=True)
def _harmonic(a, b):
    return 2.0 * a * b / (a + b)


@njit(cache=True)
def face_flux_rhs(T, conductivity, dx, dy, dz, ambient_k, convection=0.0,
                  emissivity=0.0, boundary="fixed"):
    """Return conservative q [W/m³], outward boundary loss [W].

    Exposed top has convection+radiation. Other boundaries are either fixed
    ambient at the face (half-cell distance), or insulated for verification.
    """
    nx, ny, nz = T.shape
    fixed_boundaries = boundary != "insulated"
    q = np.zeros_like(T)
    loss_w = 0.0
    for i in range(nx):
        for j in range(ny):
            for l in range(nz):
                t = T[i, j, l]
                k = conductivity[i, j, l]
                v = 0.0
                if i > 0:
                    v += _harmonic(k, conductivity[i-1, j, l]) * (T[i-1, j, l]-t) / dx**2
                elif fixed_boundaries:
                    f = 2*k*(ambient_k-t)/dx**2
                    v += f
                    loss_w -= f*dx*dy*dz
                if i < nx-1:
                    v += _harmonic(k, conductivity[i+1, j, l]) * (T[i+1, j, l]-t) / dx**2
                elif fixed_boundaries:
                    f = 2*k*(ambient_k-t)/dx**2
                    v += f
                    loss_w -= f*dx*dy*dz
                if j > 0:
                    v += _harmonic(k, conductivity[i, j-1, l]) * (T[i, j-1, l]-t) / dy**2
                elif fixed_boundaries:
                    f = 2*k*(ambient_k-t)/dy**2
                    v += f
                    loss_w -= f*dx*dy*dz
                if j < ny-1:
                    v += _harmonic(k, conductivity[i, j+1, l]) * (T[i, j+1, l]-t) / dy**2
                elif fixed_boundaries:
                    f = 2*k*(ambient_k-t)/dy**2
                    v += f
                    loss_w -= f*dx*dy*dz
                if l > 0:
                    v += _harmonic(k, conductivity[i, j, l-1]) * (T[i, j, l-1]-t) / dz**2
                else:
                    f = convection*(t-ambient_k) + emissivity*5.670374419e-8*(t**4-ambient_k**4)
                    v -= f/dz
                    loss_w += f*dx*dy
                if l < nz-1:
                    v += _harmonic(k, conductivity[i, j, l+1]) * (T[i, j, l+1]-t) / dz**2
                elif fixed_boundaries:
                    f = 2*k*(ambient_k-t)/dz**2
                    v += f
                    loss_w -= f*dx*dy*dz
                q[i, j, l] = v
    return q, loss_w


@njit(cache=True)
def _table_interp(t, table, origin, interval):
    f = (t-origin)/interval
    p = min(max(int(f), 0), table.size-2)
    return table[p] + (f-p)*(table[p+1]-table[p])


@njit(cache=True)
def _advance(h, T, k, melted, rho, material_id, q, laser_weights,
             absorbed_power, dx, dy, dz, dt, inverse_h, h_origins, dh,
             k_tables, t_origin, t_interval, solidus, liquidus, powder_nz,
             powder_k):
    nx, ny, nz = T.shape
    peak = 0.0
    smallest = 1e20
    for i in range(nx):
        for j in range(ny):
            for l in range(nz):
                src = 0.0
                if l == 0:
                    src = absorbed_power*laser_weights[i, j]/(dx*dy*dz)
                h[i, j, l] += dt*(q[i, j, l]+src)/rho[i, j, l]
                m = material_id[l]
                t = _table_interp(h[i, j, l], inverse_h[m], h_origins[m], dh)
                if not math.isfinite(t) or not math.isfinite(h[i, j, l]):
                    return math.nan, math.nan
                T[i, j, l] = t
                peak = max(peak, t)
                smallest = min(smallest, t)
                bulk_k = _table_interp(t, k_tables[m], t_origin, t_interval)
                if l < powder_nz:
                    if t >= liquidus[m]:
                        melted[i, j, l] = True
                    # Retain powder mass in fixed cell volume. Consolidated
                    # conductivity is an effective fixed-grid approximation.
                    fraction = 1.0 if melted[i, j, l] else max(0.0, min(1.0, (t-solidus[m])/(liquidus[m]-solidus[m])))
                    k[i, j, l] = powder_k*(1-fraction)+bulk_k*fraction
                else:
                    k[i, j, l] = bulk_k
    return peak, smallest


def _geometry(section, y, z, liquidus, layer):
    """Linear interpolation of threshold crossing between cell centres."""
    widths = []
    for l in range(len(z)):
        values = section[:, l]
        hot = np.flatnonzero(values >= liquidus)
        if not len(hot):
            continue
        lo, hi = hot[0], hot[-1]
        left = y[lo] if lo == 0 else np.interp(liquidus, values[lo-1:lo+1], y[lo-1:lo+1])
        right = y[hi] if hi == len(y)-1 else np.interp(liquidus, values[hi:hi+2][::-1], y[hi:hi+2][::-1])
        widths.append(right-left)
    depths = []
    for j in range(len(y)):
        values = section[j]
        hot = np.flatnonzero(values >= liquidus)
        if not len(hot):
            continue
        hi = hot[-1]
        depth = z[hi] if hi == len(z)-1 else np.interp(liquidus, values[hi:hi+2][::-1], z[hi:hi+2][::-1])
        depths.append(depth)
    width = max(widths, default=0.0)
    depth = max(depths, default=0.0)
    return width*1e6, depth*1e6, max(0, depth-layer)*1e6


def simulate(config: SimulationConfig, progress: Callable | None = None,
             cancel: Callable | None = None, snapshot: Callable | None = None,
             snapshot_count: int = 60) -> SimulationResult:
    """Run a short track; progress(fraction, message), cancel()->bool.

    Failures/cancellation raise and never return a misleading valid geometry.
    Runtime includes first-use JIT compilation. Peak RSS is sampled process RSS.
    Optional snapshot(payload) receives approximately uniformly spaced actual
    instantaneous fields, including the initial and final states. Temperature
    is copied; coordinate and local-liquidus arrays are shared read-only.
    """
    config.validate()
    if snapshot is not None and (not isinstance(snapshot_count, int) or not 2 <= snapshot_count <= 1000):
        raise ValueError("Snapshot count must be an integer from 2 to 1000")
    start = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    if progress:
        progress(0.0, "Preparing material enthalpy tables and 3D mesh")
    c = config
    materials = [MATERIALS[c.alloy], MATERIALS[c.substrate]]
    nominal_xy = 20e-6 if c.mode == "coarse" else 10e-6
    powder_nz = 3 if c.mode == "coarse" else 6
    dz = c.layer_m/powder_nz
    # Track remains 0.5 mm long under domain enlargement. End margin and
    # depth are varied independently of the measurement cross-section.
    nx = math.ceil(0.9e-3*c.domain_scale/nominal_xy)
    ny = math.ceil(0.6e-3*c.domain_scale/nominal_xy)
    # Preserve cell size and centring phase when enlarging the domain;
    # otherwise a domain check is confounded by mesh/parity changes.
    nx += (math.ceil(0.9e-3/nominal_xy)-nx) % 2
    ny += (math.ceil(0.6e-3/nominal_xy)-ny) % 2
    substrate_nz = math.ceil(0.6e-3*c.domain_scale/dz)
    nz = powder_nz+substrate_nz
    if nx*ny*nz > 4_000_000:
        raise ValueError("Requested grid exceeds four million cells; reduce domain or use coarse mode")
    dx = nominal_xy
    dy = nominal_xy
    x_edges = np.linspace(-nx*dx/2, nx*dx/2, nx+1)
    y_center = c.hatch_m/2 if c.tracks == 2 else 0.0
    y_edges = np.linspace(-ny*dy/2+y_center, ny*dy/2+y_center, ny+1)
    x = (x_edges[:-1]+x_edges[1:])/2
    y = (y_edges[:-1]+y_edges[1:])/2
    z = (np.arange(nz)+0.5)*dz
    material_id = np.ones(nz, dtype=np.int64)
    material_id[:powder_nz] = 0
    rho = np.empty((nx, ny, nz))
    rho[:, :, :powder_nz] = materials[0].rho*c.powder_packing
    rho[:, :, powder_nz:] = materials[1].rho
    T = np.full((nx, ny, nz), c.initial_k)
    h = np.empty_like(T)
    k = np.empty_like(T)
    initial_h = np.empty(nz)
    for m, sl in [(materials[0], slice(0, powder_nz)), (materials[1], slice(powder_nz, None))]:
        initial_h[sl] = float(m.enthalpy(c.initial_k))
        h[:, :, sl] = float(m.enthalpy(c.initial_k))
        k[:, :, sl] = float(m.conductivity(c.initial_k))
    k[:, :, :powder_nz] = c.powder_k_w_mk
    melted = np.zeros_like(T, dtype=np.bool_)
    peakT = T.copy()
    # Uniform-in-enthalpy inverse tables avoid nonlinear iterations. 50J/kg
    # spacing gives <=0.13K interpolation scale away from latent plateau.
    tt = np.arange(100.0, 15001.0, 1.0)
    dh = 50.0
    tables_h = [np.asarray(m.enthalpy(tt)) for m in materials]
    # Align each inverse grid with the initial enthalpy so an unheated domain
    # preserves its exact initial temperature, including zero-power checks.
    h_origins = np.array([float(m.enthalpy(c.initial_k))-math.ceil((float(m.enthalpy(c.initial_k))-tables_h[i][0])/dh)*dh for i,m in enumerate(materials)])
    n_h = int(max((a[-1]-a[0])/dh for a in tables_h))+2
    inverse_h = np.array([m.temperature_from_enthalpy(h_origins[i]+np.arange(n_h)*dh) for i,m in enumerate(materials)])
    k_tables = np.array([m.conductivity(tt) for m in materials])
    solidus = np.array([m.solidus for m in materials])
    liquidus = np.array([m.liquidus for m in materials])
    # Maximum thermal diffusivity over the table and minimum fixed mass
    # density; factor2 at Dirichlet half faces is included in the bound.
    max_alpha = max(float(np.max(k_tables[i]/(materials[i].rho*np.asarray(materials[i].heat_capacity(tt))))) for i in range(2))
    max_alpha = max(max_alpha/c.powder_packing, c.powder_k_w_mk/(materials[0].rho*c.powder_packing*min(materials[0].heat_capacity(tt))))
    dt_conduction = 0.35/(2*max_alpha*(1/dx**2+1/dy**2+1/dz**2))
    # Resolve transit with <=quarter-cell travel per step.
    dt_max = min(dt_conduction, dx/(4*c.speed_m_s))*c.dt_scale
    track_length = 0.5e-3
    track_time = track_length/c.speed_m_s
    duration = c.tracks*track_time+(c.tracks-1)*c.intertrack_delay_s+c.cooldown_s
    steps = math.ceil(duration/dt_max)
    if steps > 250_000:
        raise ValueError("Requested simulation exceeds 250,000 steps")
    dt = duration/steps
    absorbed_power = c.power_w*c.power_factor*c.absorptivity
    absorbed_j = 0.0
    boundary_j = 0.0
    laser_active_s = 0.0
    weight_sum_min = 1.0
    weight_sum_max = 0.0
    report_every = max(1, steps//100)
    zero_weights = np.zeros((nx, ny))
    beam_weights = gaussian_cell_weights if c.profile == "gaussian" else top_hat_cell_weights
    snapshot_steps = set(np.rint(np.linspace(0, steps, min(snapshot_count, steps+1))).astype(int)) if snapshot else set()
    local_liquidus = np.where(z < c.layer_m, materials[0].liquidus, materials[1].liquidus)

    def emit_snapshot(frame_time):
        laser_x, laser_y = None, None
        if absorbed_power > 0:
            for track in range(c.tracks):
                begin = track*(track_time+c.intertrack_delay_s)
                if begin <= frame_time < begin+track_time:
                    laser_x = -track_length/2+c.speed_m_s*(frame_time-begin)
                    laser_y = track*c.hatch_m
                    break
        snapshot(dict(time_s=float(frame_time), temperature_k=T.copy(),
                      x_m=x, y_m=y, z_m=z,
                      laser_x_m=laser_x, laser_y_m=laser_y,
                      laser_on=laser_x is not None,
                      liquidus_by_z_k=local_liquidus))

    if snapshot:
        for coordinate in (x, y, z, local_liquidus):
            coordinate.setflags(write=False)
        emit_snapshot(0.0)
    for n in range(steps):
        if n % report_every == 0:
            if cancel and cancel():
                raise SimulationCancelled("Simulation cancelled; partial results discarded")
            peak_rss = max(peak_rss, process.memory_info().rss)
            if progress:
                progress(n/steps, f"Thermal integration · {n:,}/{steps:,} steps · {nx}×{ny}×{nz} cells")
        # Split steps at laser on/off boundaries. Midpoint source samples
        # within each active interval, so delivered energy uses exact dwell.
        t0, t1 = n*dt, (n+1)*dt
        weights = np.zeros_like(zero_weights)
        active_dt = 0.0
        for track in range(c.tracks):
            begin = track*(track_time+c.intertrack_delay_s)
            end = begin+track_time
            a, b = max(t0, begin), min(t1, end)
            if b > a:
                segment_dt = b-a
                active_dt += segment_dt
                tm = (a+b)/2-begin
                xc = -track_length/2+c.speed_m_s*tm
                weights += segment_dt*beam_weights(x_edges, y_edges, xc, track*c.hatch_m, c.beam_diameter_m/2)
        power_this_step = absorbed_power
        if active_dt:
            ws = float(weights.sum()/active_dt)
            weight_sum_min = min(weight_sum_min, ws)
            weight_sum_max = max(weight_sum_max, ws)
            absorbed_j += absorbed_power*active_dt*ws
            laser_active_s += active_dt
        weights /= dt
        q, loss_w = face_flux_rhs(T, k, dx, dy, dz, c.initial_k,
                                 c.convection_w_m2k, c.emissivity, "fixed")
        boundary_j += dt*loss_w
        hottest, coldest = _advance(h, T, k, melted, rho, material_id, q, weights,
                                   power_this_step, dx, dy, dz, dt, inverse_h,
                                   h_origins, dh, k_tables, 100.0, 1.0,
                                   solidus, liquidus, powder_nz, c.powder_k_w_mk)
        if not math.isfinite(hottest) or not math.isfinite(coldest) or hottest > 14000 or coldest < c.initial_k-1:
            raise SimulationError(f"Invalid temperature at step {n}: [{coldest:g},{hottest:g}] K; refine mesh/time step or lower input heat")
        np.maximum(peakT, T, out=peakT)
        if snapshot and n+1 in snapshot_steps:
            emit_snapshot(duration if n+1 == steps else (n+1)*dt)
    volume = dx*dy*dz
    stored_j = float(np.sum(rho*(h-initial_h[None, None, :]))*volume)
    latent_j = 0.0
    for m, sl in [(materials[0], slice(0,powder_nz)), (materials[1],slice(powder_nz,None))]:
        latent_j += float(np.sum(rho[:,:,sl]*m.latent*(m.liquid_fraction(T[:,:,sl])-m.liquid_fraction(c.initial_k)))*volume)
    error_j = absorbed_j-boundary_j-stored_j
    error_fraction = abs(error_j)/absorbed_j if absorbed_j > 1e-12 else (0.0 if abs(error_j) <= 1e-10 else math.inf)
    if abs(error_j) > max(1e-10, absorbed_j*1e-6):
        raise SimulationError(f"Enthalpy energy balance failed ({error_fraction:.3g} relative)")
    ix = int(np.argmin(abs(x)))
    section = peakT[ix].copy()
    # If substrate differs, each depth uses its local liquidus threshold.
    # Transform into a common threshold for extraction without changing T.
    equivalent_section = section-local_liquidus[None, :]+materials[0].liquidus
    if (np.any(equivalent_section[[0,-1],:] >= materials[0].liquidus)
            or np.any(equivalent_section[:,-1] >= materials[0].liquidus)):
        raise SimulationError("Liquidus footprint touches transverse or lower domain boundary; increase domain size before reporting geometry")
    width, depth, penetration = _geometry(equivalent_section, y, z, materials[0].liquidus, c.layer_m)
    maxT = float(peakT.max())
    boundary_rise = float(max(peakT[0].max(),peakT[-1].max(),peakT[:,0].max(),peakT[:,-1].max(),peakT[:,:,-1].max())-c.initial_k)
    warnings = ["Conduction-only exploratory prediction; unvalidated against your printer."]
    if c.alloy == "17-4PH" or c.substrate == "17-4PH":
        warnings.append("17-4PH high-temperature cp/k are approximate graph samples/estimates; latent heat 270 kJ/kg is assumed.")
    if c.mode == "coarse":
        warnings.append("Coarse mesh: geometry requires refined-grid comparison before interpreting differences.")
    if c.beam_diameter_m/min(dx,dy) < 4:
        warnings.append("Beam spans fewer than four lateral cells; refine the mesh before interpreting geometry.")
    if width and (width < 3*dy*1e6 or depth < 2*dz*1e6):
        warnings.append("Liquidus geometry spans very few cells; interpolated width/depth may be poorly resolved.")
    if weight_sum_min < .999:
        warnings.append(f"Source is partly clipped by the domain (minimum captured fraction {weight_sum_min:.5f}); enlarge domain.")
    if boundary_rise > 20:
        warnings.append(f"Outer-cell temperature rose {boundary_rise:.1f} K; assess larger-domain sensitivity.")
    if maxT > 2900:
        warnings.append("Peak temperature exceeds 2900 K: vaporisation/recoil/flow are omitted; geometry may be unreliable.")
    if c.alloy == "17-4PH" and maxT > materials[0].valid_temperature_max:
        warnings.append("17-4PH temperature exceeds tabulated estimated range; constant endpoint cp/k tails are used.")
    if c.alloy == "316L" and maxT > 2900:
        warnings.append("316L temperature exceeds measured high-temperature range; endpoint extrapolation used.")
    if depth == 0:
        warnings.append("No liquidus crossing resolved at the central section; zero means no resolved fully molten region, not proof of no melt.")
    if c.tracks == 2:
        warnings.append("Geometry is the combined two-track thermal footprint; single-track geometry definitions differ.")
    metrics = dict(width_um=width, depth_um=depth, penetration_um=penetration,
                   width_to_hatch=width/(c.hatch_m*1e6), peak_temperature_k=maxT,
                   energy_error_fraction=error_fraction, energy_residual_j=error_j,
                   absorbed_energy_j=absorbed_j, boundary_loss_j=boundary_j,
                   stored_energy_j=stored_j, stored_latent_energy_j=latent_j,
                   stored_sensible_energy_j=stored_j-latent_j, runtime_s=time.perf_counter()-start,
                   peak_memory_mb=peak_rss/1024**2, steps=steps,
                   cells=nx*ny*nz, grid_shape=[nx, ny, nz], dx_um=dx*1e6,
                   dy_um=dy*1e6,dz_um=dz*1e6,dt_s=dt,
                   boundary_max_rise_k=boundary_rise,
                   layer_cells=powder_nz, section_x_um=float(x[ix]*1e6),
                   liquidus_k=materials[0].liquidus, substrate_liquidus_k=materials[1].liquidus,
                   laser_active_s=laser_active_s, source_fraction_min=weight_sum_min,
                   source_fraction_max=weight_sum_max,
                   high_temperature_caution=maxT > 2900,
                   experimentally_validated=False,
                   valid=True, warnings=warnings)
    if progress:
        progress(1.0, f"Complete · {metrics['runtime_s']:.2f} s · energy closure {error_fraction:.2e}")
    return SimulationResult(c, metrics, x, y, z, peakT, T.copy(), section, peakT[:,:,0].copy())
