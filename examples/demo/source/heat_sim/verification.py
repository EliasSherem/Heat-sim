"""Reproducible numerical checks and optional representative laser sensitivity runs.

Execute `python -m heat_sim.verification --sensitivity` in the project environment.
This verifies numerical behavior; it does not experimentally validate the model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time
import resource
import threading

import numpy as np
import psutil


def environment() -> dict:
    def sysctl(key):
        try:
            return subprocess.check_output(['sysctl', '-n', key], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        'system': platform.platform(), 'machine': platform.machine(),
        'python': platform.python_version(), 'cpu': sysctl('machdep.cpu.brand_string'),
        'memory_bytes': psutil.virtual_memory().total,
        'logical_cpu_count': psutil.cpu_count(),
    }


def gaussian_normalization() -> dict:
    from .solver import gaussian_cell_weights
    radius = 40e-6
    edges = np.linspace(-8 * radius, 8 * radius, 115)
    weights = gaussian_cell_weights(edges, edges, 0.0, 0.0, radius)
    full = float(np.sum(weights))
    # A beam centered exactly at an x boundary deposits one half into the domain.
    half = float(np.sum(gaussian_cell_weights(edges, edges, edges[0], 0.0, radius)))
    return {'integral_full_domain': full, 'integral_boundary_center': half,
            'radius_m': radius, 'diameter_convention': '1/e^2 intensity diameter = 2 radius',
            'passed': abs(full - 1.0) < 1e-12 and abs(half - .5) < 1e-12}


def enthalpy_checks() -> dict:
    from .materials import MATERIALS
    result = {}
    for name, material in MATERIALS.items():
        temperatures = np.linspace(298.15, 3500, 3000)
        h = material.enthalpy(temperatures)
        recovered = material.temperature_from_enthalpy(h)
        err = float(np.max(np.abs(recovered - temperatures)))
        grid = np.linspace(material.solidus, material.liquidus, 10001)
        sensible = float(np.trapezoid(material.heat_capacity(grid), grid))
        total = float(material.enthalpy(material.liquidus) - material.enthalpy(material.solidus))
        inferred_latent = total - sensible
        relative = abs(inferred_latent - material.latent) / material.latent
        result[name] = {
            'max_temperature_roundtrip_error_k': err,
            'inferred_latent_j_kg': inferred_latent, 'configured_latent_j_kg': material.latent,
            'relative_latent_error': relative,
            'passed': bool(np.all(np.diff(h) > 0) and err < .5 and relative < 1e-3),
        }
    return result


def conduction_benchmark() -> dict:
    """Filled against the exported production face operator; no alternative solver."""
    from .solver import face_flux_rhs
    rho, cp, k = 7800., 500., 20.
    alpha, length, final_t = k / (rho * cp), 1e-3, .002
    runs = []
    for nx in (16, 32, 64):
        dx = length / nx
        x = (np.arange(nx) + .5) * dx
        initial = 300 + 20 * np.cos(np.pi * x / length)
        temperature = np.broadcast_to(initial[:, None, None], (nx, 3, 3)).copy()
        conductivity = np.full_like(temperature, k)
        nominal_dt = .1 * dx * dx / alpha
        steps = int(np.ceil(final_t / nominal_dt))
        dt = final_t / steps
        for _ in range(steps):
            rhs, losses = face_flux_rhs(temperature, conductivity, dx, dx, dx, 300., 0., 0., boundary='insulated')
            temperature += dt * rhs / (rho * cp)
        exact = 300 + 20 * np.exp(-alpha * (np.pi / length) ** 2 * final_t) * np.cos(np.pi * x / length)
        error = float(np.max(np.abs(temperature[:, 0, 0] - exact)))
        runs.append({'nx': nx, 'steps': steps, 'dx_m': dx, 'dt_s': dt,
                     'max_abs_temperature_error_k': error})
    errors = [run['max_abs_temperature_error_k'] for run in runs]
    return {'solution': '300+20*cos(pi*x/L)*exp(-alpha*(pi/L)^2*t)',
            'L_m': length, 'time_s': final_t, 'alpha_m2_s': alpha, 'runs': runs,
            'passed': errors[-1] < .01 and errors[2] < errors[1] < errors[0]}


def flux_conservation() -> dict:
    from .solver import face_flux_rhs
    rng = np.random.default_rng(17316)
    temperature = 350 + 100 * rng.random((8, 7, 6))
    conductivity = 5 + 20 * rng.random(temperature.shape)
    dx, dy, dz = 20e-6, 25e-6, 10e-6
    rhs, loss = face_flux_rhs(temperature, conductivity, dx, dy, dz, 300., 0., 0., boundary='insulated')
    residual = float(abs(np.sum(rhs) * dx * dy * dz))
    scale = float(np.sum(np.abs(rhs)) * dx * dy * dz)
    relative = residual / max(scale, 1e-30)
    uniform = np.full((8, 7, 6), 500.)
    rhs_cool, loss_cool = face_flux_rhs(uniform, np.full_like(uniform, 20.), dx, dy, dz, 300., 20., .3)
    net_w = float(-np.sum(rhs_cool) * dx * dy * dz)
    cooling_relative = abs(net_w - loss_cool) / max(abs(loss_cool), 1e-30)
    rho_cp = 7800. * 500.
    cooling_dt = .2 / (2 * (20. / rho_cp) * (1 / dx**2 + 1 / dy**2 + 1 / dz**2))
    cooled = uniform + cooling_dt * rhs_cool / rho_cp
    cooling_step_energy_j = float(np.sum((cooled - uniform) * rho_cp) * dx * dy * dz)
    cooling_energy_residual = abs(cooling_step_energy_j + cooling_dt * loss_cool) / (cooling_dt * loss_cool)
    return {'insulated_net_w': residual, 'insulated_relative_residual': relative,
            'outward_boundary_loss_w': float(loss_cool), 'integrated_cooling_w': net_w,
            'cooling_relative_residual': float(cooling_relative),
            'zero_input_cooling_dt_s': cooling_dt,
            'zero_input_cooling_min_k': float(cooled.min()),
            'zero_input_cooling_max_k': float(cooled.max()),
            'zero_input_cooling_energy_residual': float(cooling_energy_residual),
            'passed': bool(relative < 1e-12 and loss_cool > 0 and cooling_relative < 1e-12
                           and cooled.min() >= 300. and cooled.max() <= 500.
                           and cooling_energy_residual < 1e-12)}


def laser_sensitivity(progress=None, cancel=None) -> dict:
    from .solver import SimulationConfig, SimulationCancelled, simulate
    base = SimulationConfig()
    cases = [
        ('coarse', base),
        ('coarse_half_dt', replace(base, dt_scale=.5)),
        ('refined', replace(base, mode='refined')),
        ('larger_domain', replace(base, domain_scale=1.5)),
    ]
    result = []
    for index, (name, config) in enumerate(cases):
        if cancel and cancel():
            raise SimulationCancelled('Verification cancelled; completed report not published')
        if progress:
            progress(index / len(cases), name)
        started = time.perf_counter()
        process = psutil.Process()
        before_mb = process.memory_info().rss / 1024 ** 2
        samples = [before_mb]
        stop = threading.Event()
        def sample_memory():
            while not stop.wait(.02):
                samples.append(process.memory_info().rss / 1024 ** 2)
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        try:
            run = simulate(config, progress=(lambda f, message: progress((index + f) / len(cases), f'{name}: {message}')) if progress else None, cancel=cancel)
            result.append({'case': name, 'config': asdict(config), 'status': 'complete',
                           'metrics': run.metrics,
                           'elapsed_s': time.perf_counter() - started,
                           'resident_memory_mb_before': before_mb,
                           'resident_memory_mb_after': process.memory_info().rss / 1024 ** 2,
                           'sampled_peak_resident_memory_mb': max(samples),
                           'process_lifetime_peak_resident_memory_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if platform.system() == 'Darwin' else 1024)})
        except SimulationCancelled:
            raise
        except Exception as exc:
            result.append({'case': name, 'config': asdict(config), 'status': 'failed',
                           'error': f'{type(exc).__name__}: {exc}',
                           'elapsed_s': time.perf_counter() - started})
        finally:
            stop.set()
            sampler.join(timeout=1)
    comparisons = []
    reference = next((row for row in result if row['case'] == 'coarse' and row['status'] == 'complete'), None)
    if reference:
        for row in result:
            if row['case'] == 'coarse' or row['status'] != 'complete':
                continue
            changes = {'case': row['case']}
            for metric in ('width_um', 'depth_um', 'penetration_um'):
                before = reference['metrics'][metric]
                after = row['metrics'][metric]
                changes[metric + '_change'] = after - before
                changes[metric + '_relative_change'] = (after - before) / before if before else None
            changes['within_5_percent_width_and_depth'] = all(
                changes[key + '_relative_change'] is not None and abs(changes[key + '_relative_change']) <= .05
                for key in ('width_um', 'depth_um'))
            comparisons.append(changes)
    return {'cases': result, 'comparisons_to_coarse': comparisons,
            'numerically_verified_energy': bool(result and all(
                row['status'] == 'complete' and row['metrics']['energy_error_fraction'] < 1e-6 for row in result)),
            'geometry_convergence_claimed': False,
            'interpretation': 'Two mesh levels provide a sensitivity diagnostic, not an asymptotic convergence proof. Differences comparable to mesh cells are unresolved. A >5% width/depth change needs further refinement. Failure cases are excluded. No experimental validation is implied.'}


def run_verification(include_sensitivity=False, progress=None, cancel=None) -> dict:
    started = time.perf_counter()
    from .storage import model_fingerprint
    report = {
        'model_fingerprint': model_fingerprint(),
        'created_utc': datetime.now(timezone.utc).isoformat(), 'environment': environment(),
        'unit_conversions': {'1400_mm_s_in_m_s': 1400 * 1e-3,
                             '0.06_mm_in_m': .06 * 1e-3,
                             '60_um_in_m': 60 * 1e-6,
                             'passed': bool(np.isclose(.06 * 1e-3, 60 * 1e-6))},
    }
    for name, function in [('gaussian', gaussian_normalization), ('enthalpy', enthalpy_checks),
                           ('conduction_benchmark', conduction_benchmark), ('flux_conservation', flux_conservation)]:
        try:
            report[name] = function()
        except Exception as exc:
            report[name] = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
    if include_sensitivity:
        report['laser_sensitivity'] = laser_sensitivity(progress, cancel)
    enthalpy_ok = report['enthalpy'].get('passed') is not False and all(
        value.get('passed', False) for value in report['enthalpy'].values() if isinstance(value, dict))
    report['numerical_checks_passed'] = bool(report['unit_conversions']['passed'] and enthalpy_ok and all(
        report[key].get('passed', False) for key in ('gaussian', 'conduction_benchmark', 'flux_conservation')))
    if include_sensitivity:
        report['numerical_checks_passed'] &= report['laser_sensitivity']['numerically_verified_energy']
    report['elapsed_s'] = time.perf_counter() - started
    report['resident_memory_mb_after'] = psutil.Process().memory_info().rss / 1024 ** 2
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/verification.json'))
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()
    report = run_verification(args.sensitivity, lambda fraction, message: print(f'{fraction:.0%} {message}', flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    print(args.output)
    if not report['numerical_checks_passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
