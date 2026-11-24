import struct
import numpy as np
import pytest
from heat_sim.geometry import read_stl, rasterize_stl


def test_supplemental_stl_uses_declared_units_and_registration(tmp_path):
    vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    raw=bytes(80)+struct.pack('<I',len(faces))
    for face in faces:
        raw+=struct.pack('<12fH',0,0,0,*vertices[face].ravel(),0)
    path=tmp_path/'box.stl';path.write_bytes(raw)
    points=np.array([-.0005,.0005,.0015])
    mask=rasterize_stl(path,points,points,points)
    assert mask.sum()==1 and mask[1,1,1]
    assert read_stl(path,1e-3,(.025,.035,0)).min(axis=(0,1)) == pytest.approx([.025,.035,0])


def test_truncated_stl_is_reported(tmp_path):
    path=tmp_path/'bad.stl';path.write_bytes(bytes(80)+struct.pack('<I',2))
    with pytest.raises(ValueError):read_stl(path)
