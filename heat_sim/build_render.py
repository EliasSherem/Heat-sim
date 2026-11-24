"""Static scientific figures from a completed whole-build solver frame."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .build_viewer import load_build_frame


def frame_figure(folder, manifest, frame_index, vmin_c=20., vmax_c=1800.):
    entry=manifest["frames"][frame_index]
    frame=load_build_frame(folder,entry)
    T=frame["temperature_k"]-273.15
    active=frame.get("active_fraction",frame.get("deposited_fraction"))
    material=frame["part_mask"]&(active>0)&~frame["plate_mask"]
    xyz=[frame[k]*1000 for k in ("x_m","y_m","z_m")]
    points=np.column_stack([a[material] for a in np.meshgrid(*xyz,indexing="ij")])
    if len(xyz[2])>1 and len(points):
        points[:,2]-=(1-active[material])*(xyz[2][1]-xyz[2][0])/2
    values=T[material]
    fig=plt.figure(figsize=(12.4,5.7))
    ax=fig.add_axes([.02,.12,.40,.70],projection="3d")
    section=fig.add_axes([.55,.16,.33,.64])
    norm=Normalize(vmin=vmin_c,vmax=vmax_c)
    if len(points):
        stride=max(1,int(np.ceil(len(points)/15000)))
        ax.scatter(*points[::stride].T,c=values[::stride],cmap="inferno",norm=norm,s=18,marker="s",depthshade=False)
    ax.set(xlabel="Job x (mm)",ylabel="Job y (mm)",zlabel="Height above plate (mm)",title="Actual stored material-cell temperatures")
    indices=np.flatnonzero(np.any(material,axis=(0,1)))
    plane=int(indices[-1]) if len(indices) else int(np.argmin(abs(xyz[2])))
    image=section.pcolormesh(xyz[0],xyz[1],np.where(active[:,:,plane]>0,T[:,:,plane],np.nan).T,cmap="inferno",norm=norm,shading="nearest")
    footprint=material[:,:,plane].T
    if np.any(footprint) and not np.all(footprint):
        section.contour(xyz[0],xyz[1],footprint,levels=[.5],colors=["#65d6cc"],linewidths=.8)
    section.set(xlabel="Job x (mm)",ylabel="Job y (mm)",title=f"Layer cross-section at z={xyz[2][plane]:.3f} mm")
    section.set_aspect("equal",adjustable="datalim")
    fig.colorbar(image,cax=fig.add_axes([.925,.20,.015,.56]),label="Temperature (°C)")
    alloy=manifest.get("config",{}).get("alloy", "material")
    fig.suptitle(f"{alloy} • source layer {entry['layer_index']} • physical time {entry['time_s']:.2f} s\nCalculated voxel fields • coarse thermal-history screen • assumptions apply",fontsize=12)
    return fig
