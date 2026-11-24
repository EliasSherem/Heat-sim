"""Offline format decoding tests use explicit tiny fixtures, never hardware code."""
import hashlib
import json
from pathlib import Path
import pickle
import struct
from zipfile import ZipFile

import numpy as np
import pytest

from heat_sim.job_import import JobImportError,load_job,save_imported_job


def legacy(tmp_path,records,**options):
    prop=dict(unit=.01,operation=options.get('operation',dict(translation=[0,0],rotation=0,scaling=[1,1])),
              border=dict(xmin=0,xmax=200,ymin=0,ymax=200),entity=dict(core=dict(enabled=False)))
    header=dict(thickness=6,thickness_real=.06,max_count=0,height=0,part={'0':dict(property=prop,
          section={'0':dict(power=120,speed=1400),'6':dict(power=100,speed=1000),'12':dict(power=380,speed=888)})})
    data=b''
    for pid,points in records:
        data+=struct.pack('<HHI',0,pid,len(points))+b''.join(struct.pack('<ff',*p)for p in points)
    path=tmp_path/'fixture.htmjob'
    with ZipFile(path,'w')as z:
        z.writestr('para.head',pickle.dumps(header,protocol=4));z.writestr('0.epone',data)
    return path


def test_legacy_real_group_order_and_vector_semantics(tmp_path):
    path=legacy(tmp_path,[(0,[[0,0],[1,0]]),(6,[[3,0],[3,1],[4,1]]),
                          (0,[[0,2],[1,2]]),(9,[[9,9],[10,10]]),(12,[[0,0],[5,0]])])
    job=load_job(path);layer=job.layers[0]
    assert layer.process_ids.tolist()==[0,0,6,6]
    np.testing.assert_allclose(layer.vectors[:2,:4],[[0,0,.001,0],[0,.002,.001,.002]])
    np.testing.assert_allclose(layer.vectors[:,4:6],[[120,1.4],[120,1.4],[100,1],[100,1]])
    assert layer.path_ids[0]!=layer.path_ids[1]
    assert layer.path_ids[2]==layer.path_ids[3]
    assert job.metadata['reserved_ignored_points']=={9:2}
    assert job.metadata['disabled_ignored_points']=={12:2}
    assert 'inferred'in job.metadata['provenance']['sequence']
    assert job.layer_thickness_m==60e-6


def test_legacy_imported_transform_order(tmp_path):
    path=legacy(tmp_path,[(0,[[1,0],[2,0]])],operation=dict(translation=[1,0],rotation=90,scaling=[2,1]))
    vector=load_job(path).layers[0].vectors[0]
    np.testing.assert_allclose(vector[:4],[.003,.002,.003,.003],atol=1e-16)


def test_legacy_truncation_and_unknown_semantics_rejected(tmp_path):
    path=legacy(tmp_path,[(0,[[0,0],[1,0]])])
    with ZipFile(path)as z:header=z.read('para.head')
    with ZipFile(path,'w')as z:
        z.writestr('para.head',header);z.writestr('0.epone',struct.pack('<HHI',0,0,10))
    with pytest.raises(JobImportError,match='truncated'):load_job(path)
    path=legacy(tmp_path,[(99,[[0,0],[1,0]])])
    with pytest.raises(JobImportError,match='refusing to invent'):load_job(path)


def test_executable_header_and_unsafe_archive_rejected(tmp_path):
    path=tmp_path/'bad.htmjob'
    with ZipFile(path,'w')as z:z.writestr('para.head',b'cos\nsystem\n(S"echo BAD"\ntR.')
    with pytest.raises(JobImportError,match='Unsafe pickle'):load_job(path)
    with ZipFile(path,'w')as z:z.writestr('../outside',b'x')
    with pytest.raises(JobImportError,match='Unsafe'):load_job(path)


def mjob(tmp_path,binary=False,exchange=False):
    layer=dict(index=0,z_mm=0,slice_z_mm=.03,paths=[dict(process='infill',kind='hatch',points=[[1,2],[3,4]])])
    preset=dict(layer_thickness_mm=.06,infill=dict(power=120,speed_mm_s=1400),power_units='vendor_source_units')
    path=tmp_path/'fixture.mjob'
    with ZipFile(path,'w')as z:
        if exchange:
            data=json.dumps(dict(layers=[layer])).encode();z.writestr('compiled.json',data)
            source=b'opaque source never executed';z.writestr('source.slc',source)
            manifest=dict(format='metal-slc-job',compiled_sha256=hashlib.sha256(data).hexdigest(),
                          source_sha256=hashlib.sha256(source).hexdigest(),import_options=dict(preset=preset))
        else:
            if binary:
                data=struct.pack('<4sIddI',b'MLY1',0,0,.03,1)+struct.pack('<BBI',0,1,2)+struct.pack('<dddd',1,2,3,4)
            else:data=json.dumps(layer).encode()
            z.writestr('layers/00000.bin'if binary else 'layers/00000.json',data)
            manifest=dict(format='metal-compiled-job',preset=preset,sources=[],
                          generator=dict(layer_encoding='paths-f64le-v1'if binary else'json'),
                          layers=[dict(member='layers/00000.bin'if binary else'layers/00000.json',sha256=hashlib.sha256(data).hexdigest())])
        z.writestr('manifest.json',json.dumps(manifest))
    return path


@pytest.mark.parametrize('binary,exchange',[(False,False),(True,False),(False,True)])
def test_mjob_actual_compiled_paths_not_generated(tmp_path,binary,exchange):
    job=load_job(mjob(tmp_path,binary,exchange));vector=job.layers[0].vectors[0]
    np.testing.assert_allclose(vector,[.001,.002,.003,.004,120,1.4])
    assert 'imported'in job.metadata['provenance']['sequence']
    assert job.metadata['verified_layer_hashes']==1


def test_mjob_path_hash_mismatch_rejected(tmp_path):
    path=mjob(tmp_path)
    with ZipFile(path)as z:manifest=json.loads(z.read('manifest.json'));data=z.read('layers/00000.json')
    manifest['layers'][0]['sha256']='0'*64
    with ZipFile(path,'w')as z:z.writestr('manifest.json',json.dumps(manifest));z.writestr('layers/00000.json',data)
    with pytest.raises(JobImportError,match='hash mismatch'):load_job(path)


def test_normalized_explicit_units_and_compact_cache_roundtrip(tmp_path):
    document=dict(format='heat-sim-job-v1',coordinate_units='m',speed_units='m/s',power_units='W_assumed',
                  layer_thickness_m=60e-6,layers=[dict(index=0,z_m=0,vectors=[[0,0,.001,0,120,1.4]])])
    path=tmp_path/'explicit.json';path.write_text(json.dumps(document));job=load_job(path)
    target=save_imported_job(job,tmp_path/'saved');restored=load_job(target)
    np.testing.assert_array_equal(restored.layers[0].vectors,job.layers[0].vectors)
    assert restored.source_sha256==job.source_sha256
    raw=json.loads(target.read_text());raw['normalized_arrays_sha256']='0'*64;target.write_text(json.dumps(raw))
    with pytest.raises(JobImportError,match='array hash'):load_job(target)
    document['coordinate_units']='mm';path.write_text(json.dumps(document))
    with pytest.raises(JobImportError,match='explicit m'):load_job(path)


def test_cached_source_identity_and_opaque_job_fail_clearly(tmp_path):
    job=load_job(legacy(tmp_path,[(0,[[0,0],[1,0]])]))
    target=save_imported_job(job,tmp_path/'cached')
    (target.parent/'source.htmjob').write_bytes(b'changed')
    with pytest.raises(JobImportError,match='source-copy hash'):load_job(target)
    unknown=tmp_path/'unknown.mjob'
    with ZipFile(unknown,'w')as z:z.writestr('opaque.bin',b'not a decoded scan stream')
    with pytest.raises(JobImportError,match='supply normalized JSON'):load_job(unknown)
