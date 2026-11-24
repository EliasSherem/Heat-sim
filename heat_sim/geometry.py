"""Inert STL intake and cell-centre occupancy for optional supplemental geometry."""
from pathlib import Path
import struct

import numpy as np
from numba import njit


def read_stl(path, unit_scale=1e-3, translation_m=(0.,0.,0.)):
    """Read binary/ASCII triangles, never execute imported content.

    STL contains no units; unit_scale and registration are explicit user inputs.
    """
    raw = Path(path).read_bytes()
    if len(raw) > 200*1024**2:
        raise ValueError("Supplemental STL exceeds200 MiB limit")
    triangles = None
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if len(raw) == 84 + count*50:
            record = np.dtype([("normal","<f4",(3,)), ("vertices","<f4",(3,3)), ("attribute","<u2")])
            triangles = np.frombuffer(raw, dtype=record, count=count, offset=84)["vertices"].astype(float)
    if triangles is None:
        try:
            vertices = [list(map(float,line.split()[1:])) for line in raw.decode("ascii").splitlines() if line.strip().startswith("vertex ")]
            if not vertices or any(len(v)!=3 for v in vertices) or len(vertices)%3:
                raise ValueError("STL triangle vertex count is invalid")
            triangles = np.asarray(vertices,float).reshape((-1,3,3))
        except (UnicodeDecodeError,ValueError) as exc:
            raise ValueError("Unsupported or truncated binary/ASCII STL") from exc
    shift = np.asarray(translation_m,float)
    if shift.shape!=(3,) or not np.all(np.isfinite(shift)) or not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("STL units and translation must be finite, with positive unit scale")
    if not len(triangles) or len(triangles)>2_000_000 or not np.all(np.isfinite(triangles)):
        raise ValueError("STL needs finite triangles within the two-million triangle limit")
    return triangles*unit_scale + shift


@njit(cache=True)
def _fill_slices(triangles, x, y, z):
    result = np.zeros((len(x),len(y),len(z)),np.bool_)
    intersections = np.zeros((len(triangles),2,2))
    for k in range(len(z)):
        level = z[k]
        segments = 0
        for tri in triangles:
            points = np.zeros((2,2))
            count = 0
            for side in range(3):
                a,b = tri[side],tri[(side+1)%3]
                if (a[2]<=level<b[2]) or (b[2]<=level<a[2]):
                    fraction = (level-a[2])/(b[2]-a[2])
                    if count<2:
                        points[count,0] = a[0]+fraction*(b[0]-a[0])
                        points[count,1] = a[1]+fraction*(b[1]-a[1])
                    count += 1
            if count == 2:
                intersections[segments] = points
                segments += 1
        for j in range(len(y)):
            for i in range(len(x)):
                crossings = 0
                for n in range(segments):
                    a,b = intersections[n,0],intersections[n,1]
                    if (a[1]>y[j]) != (b[1]>y[j]):
                        crossing_x = a[0]+(y[j]-a[1])*(b[0]-a[0])/(b[1]-a[1])
                        if crossing_x > x[i]:
                            crossings += 1
                result[i,j,k] = crossings%2 == 1
    return result


def rasterize_stl(path, x_m, y_m, z_m, unit_scale=1e-3, translation_m=(0.,0.,0.)):
    """Cell-centre even/odd occupancy; not a repair of open or nonmanifold meshes."""
    return _fill_slices(read_stl(path,unit_scale,translation_m),np.asarray(x_m),np.asarray(y_m),np.asarray(z_m))
