"""Render actual transient solver fields to a local GIF and reproducible NPZ.

No temperature field is fabricated or interpolated for the animation. The 3-D
volume consists of actual mesh cells at or above each cell's local liquidus.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

from .solver import SimulationCancelled, SimulationConfig, simulate
from .storage import atomic_json, cache_key, model_fingerprint, save_simulation, snapshot_source


BACKGROUND = '#0b1220'
PANEL = '#111e30'
CYAN = '#65eee0'
INK = '#e5edf8'
MUTED = '#a8bad0'


def _edges(centers):
    centers = np.asarray(centers)
    delta = centers[1] - centers[0]
    return np.r_[centers[0] - delta / 2, centers + delta / 2]


def _check_cancel(cancel):
    if cancel and cancel():
        raise SimulationCancelled('Rendering cancelled; incomplete animation not published')


def _frame_figure(config, snapshot, vmax, roi, frame_index, frame_count):
    """Draw only instantaneous physical fields; 3-D values are voxel-resolved."""
    T = snapshot['temperature_k']
    x = snapshot['x_m'] * 1e6
    y = snapshot['y_m'] * 1e6
    z = snapshot['z_m'] * 1e6
    threshold = snapshot['liquidus_by_z_k']
    xm, ym, zm = roi
    ix, iy, iz = np.flatnonzero(xm), np.flatnonzero(ym), np.flatnonzero(zm)
    xr, yr, zr = x[ix], y[iy], z[iz]
    crop = T[np.ix_(ix, iy, iz)]
    molten = crop >= threshold[iz][None, None, :]
    norm = Normalize(vmin=config.initial_k, vmax=vmax)
    cmap = plt.get_cmap('inferno')

    with plt.rc_context({'font.size': 9, 'text.color': INK, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': '#37506a',
                         'figure.facecolor': BACKGROUND, 'axes.facecolor': PANEL,
                         'savefig.facecolor': BACKGROUND}):
        fig = plt.figure(figsize=(12.4, 7.0), dpi=100)
        grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1), left=.035, right=.93,
                               top=.82, bottom=.19, hspace=.50, wspace=.23)
        ax3 = fig.add_subplot(grid[:, 0], projection='3d')
        top = fig.add_subplot(grid[0, 1])
        section = fig.add_subplot(grid[1, 1])
        ax3.set_facecolor(PANEL)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.pane.set_facecolor(PANEL)
            axis.pane.set_edgecolor('#33485f')
            axis._axinfo['grid']['color'] = '#263b52'
            axis.line.set_color('#47617c')
        ax3.tick_params(colors=MUTED, labelsize=7, pad=0)
        if np.any(molten):
            xe, ye, ze = np.meshgrid(_edges(xr), _edges(yr), _edges(zr), indexing='ij')
            colors = cmap(norm(crop))
            colors[..., 3] = .94
            ax3.voxels(xe, ye, ze, molten, facecolors=colors,
                       edgecolors=(1., .82, .55, .30), linewidth=.18, shade=True)
        else:
            ax3.text2D(.12, .84, 'No fully molten cells at this instant',
                       transform=ax3.transAxes, color=MUTED, fontsize=9)
        # Reference plane marks the actual original powder/substrate interface.
        px, py = np.meshgrid([_edges(xr)[0], _edges(xr)[-1]],
                             [_edges(yr)[0], _edges(yr)[-1]])
        ax3.plot_surface(px, py, np.full_like(px, config.layer_m * 1e6),
                         color='#8ba2ba', alpha=.13, shade=False, linewidth=0)
        ax3.plot_wireframe(px, py, np.full_like(px, config.layer_m * 1e6),
                           color='#92a9c0', alpha=.55, linewidth=.6)
        ax3.set(xlim=(_edges(xr)[0], _edges(xr)[-1]),
                ylim=(_edges(yr)[0], _edges(yr)[-1]),
                zlim=(_edges(zr)[-1], 0), xlabel='Scan x (µm)',
                ylabel='Transverse y (µm)', zlabel='Depth z (µm)')
        ax3.set_box_aspect((np.ptp(_edges(xr)), np.ptp(_edges(yr)), np.ptp(_edges(zr))))
        ax3.view_init(elev=26, azim=-63)
        ax3.set_title('3D liquidus cells • actual mesh geometry', color=INK, pad=8, fontsize=11)
        ax3.text2D(.03, .03, f'Grey plane: original substrate at {config.layer_m*1e6:g} µm\n'
                   'Depth measured from original powder top', transform=ax3.transAxes,
                   color=MUTED, fontsize=8)

        mesh = top.pcolormesh(xr, yr, T[np.ix_(ix, iy, [0])][:, :, 0].T,
                              shading='nearest', cmap=cmap, norm=norm)
        top_excess = T[np.ix_(ix, iy, [0])][:, :, 0].T - threshold[0]
        if top_excess.min() < 0 < top_excess.max():
            top.contour(xr, yr, top_excess, levels=[0], colors=[CYAN], linewidths=1.05)
        top.set(title=f'Top-cell temperature • z = {z[0]:g} µm',
                xlabel='Scan x (µm)', ylabel='Transverse y (µm)')
        top.set_xlim(_edges(xr)[0], _edges(xr)[-1])
        top.set_ylim(_edges(yr)[0], _edges(yr)[-1])

        mid_y = int(np.argmin(np.abs(y)))
        values = T[np.ix_(ix, [mid_y], iz)][:, 0, :].T
        section.pcolormesh(xr, zr, values, shading='nearest', cmap=cmap, norm=norm)
        excess = values - threshold[iz, None]
        if excess.min() < 0 < excess.max():
            section.contour(xr, zr, excess, levels=[0], colors=[CYAN], linewidths=1.05)
        section.axhline(config.layer_m * 1e6, color='#adbed2', ls='--', lw=.8)
        section.set(title=f'Longitudinal section • y = {y[mid_y]:g} µm',
                    xlabel='Scan x (µm)', ylabel='Depth z (µm)')
        section.set_xlim(_edges(xr)[0], _edges(xr)[-1])
        section.set_ylim(_edges(zr)[-1], 0)
        if snapshot['laser_on']:
            lx, ly = snapshot['laser_x_m'] * 1e6, snapshot['laser_y_m'] * 1e6
            top.plot(lx, ly, marker='+', ms=10, mew=1.6, color='#b5ffff')
            # This marker is the laser's x position projected onto the section.
            section.plot(lx, 0, marker='v', ms=5, color='#b5ffff', clip_on=False)
        color_ax = fig.add_axes([.951, .27, .014, .47])
        bar = fig.colorbar(mesh, cax=color_ax)
        bar.set_label('Instantaneous temperature (K)', color=MUTED)
        bar.ax.tick_params(colors=MUTED, labelsize=8)
        fig.text(.045, .945, 'HEAT / SIM', color=CYAN, fontsize=12, weight='bold')
        fig.text(.045, .894, f'{config.alloy}  ·  moving heat-source simulation',
                 color=INK, fontsize=20, weight='bold')
        actual_us = snapshot['time_s'] * 1e6
        phase = 'LASER ON' if snapshot['laser_on'] else ('INITIAL STATE' if actual_us == 0 else 'COOLING')
        fig.text(.955, .94, f't = {actual_us:7.1f} µs  ·  {phase}', ha='right', color=INK, fontsize=10)
        fig.text(.955, .885, f'Frame {frame_index+1}/{frame_count}  ·  playback slowed',
                 ha='right', color=MUTED, fontsize=9)
        fig.text(.045, .126,
                 f'{config.power_w:g} W × {config.power_factor:g} delivered factor  |  '
                 f'{config.speed_m_s*1000:g} mm/s  |  {config.mode} grid  |  '
                 f'cyan = local liquidus  |  layer {config.layer_m*1e6:g} µm',
                 color=INK, fontsize=9)
        diameter = '1/e²' if config.profile == 'gaussian' else 'disk'
        fig.text(.045, .089,
                 f'ASSUMED: {config.beam_diameter_m*1e6:g} µm {diameter} beam, '
                 f'absorptivity {config.absorptivity:g}, packing {config.powder_packing:g}, '
                 f'powder k {config.powder_k_w_mk:g} W/(m K), '
                 f'initial {config.initial_k-273.15:g} °C', color=MUTED, fontsize=9)
        validity_note = 'peaks exceed physical validity' if vmax > 2900 else 'physical inputs unvalidated'
        fig.text(.045, .048,
                 f'ILLUSTRATIVE CONDUCTION MODEL • flow / vaporisation omitted • {validity_note} '
                 '• no printer recommendation', color='#ffc17d', fontsize=8.5)
        return fig


def render_simulation(config: SimulationConfig, folder: Path, progress=None, cancel=None) -> dict:
    """Simulate and render 60 actual instantaneous fields; return file manifest.

    progress(fraction,message) and cancel()->bool use the same contract as the
    thermal solver. Source/configuration snapshots and raw frames stay local.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config.validate()
    started = time.perf_counter()
    progress = progress or (lambda f, message: None)
    snapshot_source(folder)
    atomic_json(folder / 'config.json', asdict(config))
    snapshots = []
    def capture(payload):
        _check_cancel(cancel)
        # Float32 storage has <0.001 K resolution at the temperatures plotted.
        # The solver itself integrates in float64. Coordinates remain float64.
        payload = dict(payload)
        payload['temperature_k'] = payload['temperature_k'].astype(np.float32)
        snapshots.append(payload)
    result = simulate(config,
                      progress=lambda f, message: progress(.18*f, message),
                      cancel=cancel, snapshot=capture, snapshot_count=60)
    _check_cancel(cancel)
    key = cache_key(config)
    save_simulation(result, key)
    frames = np.stack([s['temperature_k'] for s in snapshots])
    times = np.array([s['time_s'] for s in snapshots])
    threshold = snapshots[0]['liquidus_by_z_k']
    x, y, z = snapshots[0]['x_m'], snapshots[0]['y_m'], snapshots[0]['z_m']
    molten = frames >= threshold[None, None, None, :]
    still_index = int(np.argmax(molten.sum(axis=(1, 2, 3))))
    if not np.any(molten):
        still_index = int(np.argmax(frames.max(axis=(1, 2, 3))))
    # Keep the plotted ROI fixed through time; expand it if liquidus cells need it.
    hot_locations = np.argwhere(np.any(molten, axis=0))
    dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
    x_bound, y_low, y_high = 310e-6, -150e-6, max(150e-6, config.hatch_m+90e-6 if config.tracks == 2 else 150e-6)
    z_bound = max(180e-6, config.layer_m*2.5)
    if len(hot_locations):
        x_bound = max(x_bound, float(np.max(np.abs(x[hot_locations[:, 0]])))+2*dx)
        y_low = min(y_low, float(y[hot_locations[:, 1]].min())-2*dy)
        y_high = max(y_high, float(y[hot_locations[:, 1]].max())+2*dy)
        z_bound = max(z_bound, float(z[hot_locations[:, 2]].max())+2*dz)
    roi = (abs(x) <= x_bound, (y >= y_low) & (y <= y_high), z <= z_bound)
    vmax = float(max(frames.max(), 1.1 * threshold.max()))
    pil_frames = []
    try:
        for index, snapshot in enumerate(snapshots):
            _check_cancel(cancel)
            progress(.18 + .77 * index / len(snapshots),
                     f'Rendering actual thermal field {index+1}/{len(snapshots)}')
            fig = _frame_figure(config, snapshot, vmax, roi, index, len(snapshots))
            try:
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                rgb = Image.fromarray(rgba[:, :, :3].copy())
                pil_frames.append(rgb.quantize(colors=192, method=Image.Quantize.MEDIANCUT))
                if index == still_index:
                    rgb.save(folder / '.final.tmp.png')
            finally:
                plt.close(fig)
        _check_cancel(cancel)
        duration_ms = [max(20, int(round(dt / (times[-1]-times[0]) * 6000 / 10)) * 10)
                       for dt in np.r_[np.diff(times), times[-1]-times[-2]]]
        temp_gif = folder / '.simulation.tmp.gif'
        pil_frames[0].save(temp_gif, save_all=True, append_images=pil_frames[1:],
                           duration=duration_ms, loop=0, disposal=2, optimize=False)
        _check_cancel(cancel)
        temp_npz = folder / '.frames.tmp.npz'
        np.savez_compressed(temp_npz, temperature_k=frames, time_s=times,
                            x_m=x, y_m=y, z_m=z, liquidus_by_z_k=threshold,
                            laser_on=np.array([s['laser_on'] for s in snapshots]),
                            laser_x_m=np.array([np.nan if s['laser_x_m'] is None else s['laser_x_m'] for s in snapshots]),
                            laser_y_m=np.array([np.nan if s['laser_y_m'] is None else s['laser_y_m'] for s in snapshots]))
        temp_gif.replace(folder / 'simulation.gif')
        temp_npz.replace(folder / 'frames.npz')
        (folder / '.final.tmp.png').replace(folder / 'final.png')
        manifest = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'alloy': config.alloy, 'config': asdict(config), 'metrics': result.metrics,
            'cache_key': key, 'model_fingerprint': model_fingerprint(),
            'frame_count': len(snapshots), 'simulation_duration_s': float(times[-1]),
            'playback_duration_s': sum(duration_ms)/1000,
            'render_wall_s': time.perf_counter()-started,
            'files': {'gif': 'simulation.gif', 'png': 'final.png', 'raw_frames': 'frames.npz',
                      'configuration': 'config.json'},
            'still_frame_index': still_index, 'still_time_s': float(times[still_index]),
            'still_selection': 'Snapshot with the largest number of fully molten mesh cells; hottest snapshot if none melt.',
            'field_definition': 'Instantaneous finite-volume cell temperatures; not a maximum-over-time envelope.',
            'three_d_definition': 'Actual cells at or above local alloy liquidus; coarse voxels are not a smooth liquid surface.',
            'temperature_storage': 'Solver float64; archived snapshot temperatures float32; coordinates/time float64.',
            'depth_definition': 'z=0 at original powder top; substrate begins at layer_m.',
            'assumptions_status': 'Illustrative editable inputs; beam, delivered power, absorptivity and powder properties are not printer measurements.',
            'physical_limitations': 'Conduction only. Flow, recoil and vaporisation omitted. ' + ('Predicted peaks exceed physical validity. ' if result.metrics.get('high_temperature_caution') else 'Physical inputs remain experimentally unvalidated. ') + 'No printer recommendation.',
        }
        atomic_json(folder / 'manifest.json', manifest)
        progress(1., f'Rendered {len(snapshots)} actual thermal fields; saved animation and source frames')
        return manifest
    finally:
        for frame in pil_frames:
            frame.close()
        for path in folder.glob('.*.tmp.*'):
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alloy', choices=['17-4PH', '316L'], default='17-4PH')
    parser.add_argument('--config', type=Path, help='Saved config.json or workflow request.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.config:
        data = json.loads(args.config.read_text())
        config = SimulationConfig(**data.get('config', data))
    else:
        config = SimulationConfig()
    config = replace(config, alloy=args.alloy)
    folder = args.output or Path('results/rendered') / args.alloy
    manifest = render_simulation(config, folder,
                                 progress=lambda f, message: print(f'{f:.0%} {message}', flush=True))
    print(json.dumps({key: manifest[key] for key in ('alloy', 'frame_count', 'render_wall_s', 'files')}, indent=2))

if __name__ == '__main__':
    main()
