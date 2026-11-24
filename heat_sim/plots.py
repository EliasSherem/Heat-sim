"""Labelled static thermal figures shared by the app and exports."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .materials import MATERIALS

plt.rcParams.update({"figure.facecolor": "#10151d", "axes.facecolor": "#17202c",
                     "text.color": "#e8edf5", "axes.labelcolor": "#bcc9db",
                     "xtick.color": "#9aadc6", "ytick.color": "#9aadc6",
                     "axes.edgecolor": "#36465d", "font.size": 9, "savefig.facecolor": "#10151d"})


def thermal_figure(meta, fields):
    cfg = meta["config"]
    material = MATERIALS[cfg["alloy"]]
    substrate = MATERIALS[cfg.get("substrate", cfg["alloy"])]
    y = fields["y_m"] * 1e6
    z = fields["z_m"] * 1e6
    x = fields["x_m"] * 1e6
    section = fields["cross_section_peak_k"]
    top = fields["top_peak_k"]
    # Solver arrays use (x, y, z); 2D fields are (y,z) and (x,y).
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), layout="constrained")
    # Liquidus is local to each material: a 316L powder layer on a 17-4PH
    # substrate must use the substrate threshold below the original interface.
    section_threshold = np.where(fields["z_m"] < cfg["layer_m"],
                                 material.liquidus, substrate.liquidus)[:, None]
    panels = [(axes[0], y, z, section.T, section_threshold, "Transverse thermal envelope", "Transverse y (µm)", "Depth below powder top z (µm)"),
              (axes[1], x, y, top.T, material.liquidus, "Top-cell thermal envelope", "Scan direction x (µm)", "Transverse y (µm)")]
    for ax, a, b, values, threshold, title, xlabel, ylabel in panels:
        mesh = ax.pcolormesh(a, b, values, shading="nearest", cmap="inferno", rasterized=True)
        excess = values - threshold
        if np.min(excess) < 0 < np.max(excess):
            ax.contour(a, b, excess, levels=[0.], colors=["#55e6db"], linewidths=1.3)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(mesh, ax=ax, label="Maximum temperature (K)", shrink=.85)
    axes[0].axhline(cfg["layer_m"] * 1e6, color="#d5e3f4", ls="--", lw=.9, label="Original substrate surface")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"{cfg['alloy']} · {cfg['power_w']:g} W / {cfg['speed_m_s']*1000:g} mm/s · cyan = liquidus")
    return fig


def sweep_figure(frame, metric="geometry_distance"):
    valid = frame[frame["valid"] == True].copy()
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    label = {"geometry_distance": "Distance from 17-4 geometry (lower = closer)",
             "width_um": "Liquidus-envelope width (µm)", "depth_um": "Depth below powder top (µm)"}.get(metric, metric)
    if len(valid) and metric in valid and valid[metric].notna().any():
        plot = ax.scatter(valid["speed_mm_s"], valid["power_w"], c=valid[metric], s=180, cmap="viridis_r" if metric=="geometry_distance" else "inferno", edgecolors="#91a0b5", linewidths=.5)
        fig.colorbar(plot, ax=ax, label=label)
    else:
        ax.text(.5, .5, "No numerically valid, rankable points", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Scan speed (mm/s)", ylabel="Commanded laser power (W)", title="316L power–speed screening map · sampled points only")
    return fig
