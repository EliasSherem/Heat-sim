"""Offline, bounded decoding of real job paths; no printer interfaces.

Vector columns are x0,y0,x1,y1 [m], provisional power [W], speed [m/s].
Source power values remain vendor units until a delivered-power calibration is
provided. See docs/JOB_FORMAT.md for imported/inferred/assumed distinctions.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import struct
from zipfile import ZipFile, is_zipfile

import numpy as np

from .job_reference import _ALLOWED, _DataOnlyUnpickler
import pickletools

MAX_BYTES = 600 * 1024**2
MAX_POINTS = 25_000_000
MAX_LAYERS = 10_000
PROCESS_NAMES = {0:'infill',1:'upper_infill',2:'lower_infill',3:'inner_contour',
                 4:'upper_inner_contour',5:'lower_inner_contour',6:'contour',
                 7:'upper_contour',8:'lower_contour',12:'core',16:'support'}
PROCESS_IDS = {name:key for key,name in PROCESS_NAMES.items()}
FLAT_IDS = {0,1,2,12}
POLYLINE_IDS = {3,4,5,6,7,8,16}
RESERVED_IDS = {9,10,11,13,14,15}
SOURCE_DECODER = '/Users/elias/Downloads/Metal_printer/controller/htmjob.py'
SOURCE_ORDER = '/Users/elias/Downloads/Metal_printer/tools/phase3_rtc4_call_recorder.py'
SOURCE_MJOB = '/Users/elias/Downloads/Metal_printer/controller/slicing/package.py'


class JobImportError(ValueError):
    pass


@dataclass
class ImportedLayer:
    index: int
    z_m: float
    vectors: np.ndarray
    process_ids: np.ndarray
    part_ids: np.ndarray
    path_ids: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def mark_time_s(self):
        return float(np.sum(np.linalg.norm(self.vectors[:,2:4]-self.vectors[:,:2],axis=1)/self.vectors[:,5])) if len(self.vectors) else 0.


@dataclass
class ImportedJob:
    name: str
    source_path: str
    source_sha256: str
    format: str
    layer_thickness_m: float
    layers: list[ImportedLayer]
    metadata: dict = field(default_factory=dict)

    @property
    def bounds_m(self):
        populated = [layer.vectors for layer in self.layers if len(layer.vectors)]
        if not populated:
            return [0.,0.,0.,0.,0.,len(self.layers)*self.layer_thickness_m]
        xmin=min(float(v[:,[0,2]].min()) for v in populated)
        xmax=max(float(v[:,[0,2]].max()) for v in populated)
        ymin=min(float(v[:,[1,3]].min()) for v in populated)
        ymax=max(float(v[:,[1,3]].max()) for v in populated)
        return [xmin,ymin,min(layer.z_m for layer in self.layers),xmax,ymax,
                max(layer.z_m+self.layer_thickness_m for layer in self.layers)]

    def summary(self):
        counts=Counter()
        for layer in self.layers:
            ids,count=np.unique(layer.process_ids,return_counts=True)
            counts.update({PROCESS_NAMES.get(int(i),str(int(i))):int(n) for i,n in zip(ids,count)})
        return dict(name=self.name,source_path=self.source_path,source_sha256=self.source_sha256,
                    format=self.format,layer_count=len(self.layers),layer_thickness_m=self.layer_thickness_m,
                    vector_count=sum(len(layer.vectors) for layer in self.layers),bounds_m=self.bounds_m,
                    emitting_layer_count=sum(bool(len(layer.vectors)) for layer in self.layers),
                    mark_time_s=sum(layer.mark_time_s for layer in self.layers),process_vector_counts=dict(counts),
                    metadata=self.metadata,
                    layers=[dict(index=layer.index,z_m=layer.z_m,vector_count=len(layer.vectors),
                                 mark_time_s=layer.mark_time_s,metadata=layer.metadata) for layer in self.layers])


def _finite(value,name,positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (positive and value<=0):
        raise JobImportError(f'Invalid {name}: expected finite '+('positive number' if positive else 'number'))
    return float(value)


def _json(data):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise JobImportError(f'Duplicate JSON key: {key}')
            result[key]=value
        return result
    return json.loads(data,object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(JobImportError('Non-finite JSON constant')))


def _archive(path):
    archive=ZipFile(path)
    names=set();total=0
    for member in archive.infolist():
        name=PurePosixPath(member.filename)
        if name.is_absolute() or '..' in name.parts or '\\' in member.filename or member.filename in names:
            archive.close();raise JobImportError('Unsafe or duplicate ZIP member')
        if member.flag_bits&1:
            archive.close();raise JobImportError('Encrypted job members are unsupported')
        names.add(member.filename);total+=member.file_size
        if total>MAX_BYTES or member.file_size>MAX_BYTES:
            archive.close();raise JobImportError('Job exceeds uncompressed 600 MiB limit')
    return archive


def _safe_header(data):
    if len(data)>2*1024**2:raise JobImportError('Job header exceeds 2 MiB')
    for opcode,_,_ in pickletools.genops(data):
        if opcode.name not in _ALLOWED:raise JobImportError(f'Unsafe pickle opcode: {opcode.name}')
    value=_DataOnlyUnpickler(io.BytesIO(data)).load()
    if not isinstance(value,dict):raise JobImportError('Job parameter header must be dictionary')
    return value


def _evidence_hashes(paths):
    return {str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths if Path(path).is_file()}


def _source_meta(path,format):
    return dict(name=path.name,source_path=str(path.resolve()),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),format=format)


def _layer(index,z,vectors,process,part,path,metadata=None):
    values=np.asarray(vectors,dtype=float).reshape((-1,6))
    if not np.all(np.isfinite(values)) or (len(values) and (np.any(values[:,4]<0) or np.any(values[:,5]<=0))):
        raise JobImportError(f'Invalid finite power/speed/coordinates at layer {index}')
    if len(values)>MAX_POINTS:raise JobImportError('Layer vector count exceeds limit')
    result=ImportedLayer(int(index),_finite(z,'layer height'),values,
                         np.asarray(process,dtype=np.int16),np.asarray(part,dtype=np.int32),
                         np.asarray(path,dtype=np.int32),metadata or {})
    if any(len(a)!=len(values) for a in (result.process_ids,result.part_ids,result.path_ids)):
        raise JobImportError('Vector metadata arrays do not match vector count')
    return result


def _recipe(section):
    p=_finite(section.get('power'),'section power')
    v=_finite(section.get('speed',section.get('speed_mm_s')),'section speed',positive=True)*1e-3
    if p<0:raise JobImportError('Negative laser power')
    return p,v


def _transform(points,part):
    prop=part.get('property',{})
    op=prop.get('operation',{})
    translation=np.asarray(op.get('translation',[0.,0.]),float)
    scaling=np.asarray(op.get('scaling',[1.,1.]),float)
    rotation=_finite(op.get('rotation',0.),'rotation')
    if translation.shape!=(2,) or scaling.shape!=(2,) or not np.all(np.isfinite(np.r_[translation,scaling])):
        raise JobImportError('Invalid part XY transform')
    # This is the recovered vendor order: translate, rotate around header
    # bounding-box centre, then axis-scale around that same centre.
    border=prop.get('border',{})
    unit=float(prop.get('unit',.01))
    if rotation or np.any(scaling!=1):
        try:center=np.array([(float(border['xmin'])+float(border['xmax']))*unit/2,
                             (float(border['ymin'])+float(border['ymax']))*unit/2])
        except (KeyError,TypeError,ValueError)as exc:raise JobImportError('Transform needs bounding-box centre')from exc
    else:center=np.zeros(2)
    result=np.asarray(points,float)+translation
    if rotation:
        angle=math.radians(rotation);c,s=math.cos(angle),math.sin(angle)
        result=(result-center)@np.array([[c,s],[-s,c]])+center
    result=(result-center)*scaling+center
    return result*1e-3  # Source points are millimetres; header unit scales border only.


def _enabled_property(part,pid):
    prop=part.get('property',{})
    if pid==12:return prop.get('entity',{}).get('core',{}).get('enabled',True)
    if pid in (6,7,8):
        group={6:'entity',7:'up',8:'down'}[pid]
        return prop.get('contour',{}).get(group,{}).get('outer',{}).get('enabled',True)
    return True


def _read_htm(path,archive,progress):
    header=_safe_header(archive.read('para.head'))
    thickness=_finite(header.get('thickness_real'),'layer thickness',positive=True)*1e-3
    step=header.get('thickness');maximum=header.get('max_count')
    if type(step)is not int or step<=0 or type(maximum)is not int or not 0<=maximum<MAX_LAYERS:
        raise JobImportError('Invalid legacy layer index/step')
    expected={f'{i*step}.epone'for i in range(maximum+1)}
    actual={n for n in archive.namelist()if n.endswith('.epone')}
    if actual!=expected:raise JobImportError('Legacy layer members do not match header sequence')
    parts=header.get('part')
    if not isinstance(parts,dict):raise JobImportError('Missing legacy part table')
    layers=[];raw_points=0;zero_length=0;reserved_points=Counter();all_records=0;disabled=Counter()
    for index in range(maximum+1):
        member=f'{index*step}.epone';data=archive.read(member);offset=0;grouped=OrderedDict();record_index=0
        layer_reserved=Counter();layer_disabled=Counter();layer_points=0
        while offset<len(data):
            if offset+8>len(data):raise JobImportError(f'{member}: truncated record header')
            part,pid,count=struct.unpack_from('<HHI',data,offset);offset+=8
            if count>MAX_POINTS or raw_points+count>MAX_POINTS or offset+count*8>len(data):
                raise JobImportError(f'{member}: invalid point count/truncated coordinates')
            points=np.frombuffer(data,dtype='<f4',count=count*2,offset=offset).reshape((-1,2)).astype(float)
            offset+=count*8;raw_points+=count;layer_points+=count;all_records+=1
            if not np.all(np.isfinite(points)):raise JobImportError(f'{member}: non-finite coordinates')
            if pid not in FLAT_IDS|POLYLINE_IDS|RESERVED_IDS:
                raise JobImportError(f'{member}: unsupported process {pid}; refusing to invent laser semantics')
            sections=grouped.setdefault(part,OrderedDict())
            if pid in RESERVED_IDS:
                reserved_points[pid]+=count;layer_reserved[pid]+=count
            else:
                sections.setdefault(pid,[]).append((record_index,points))
            record_index+=1
        arrays=[];pids=[];partids=[];pathids=[];next_path=0
        for partid,sections in grouped.items():
            part=parts.get(str(partid))
            if not isinstance(part,dict):raise JobImportError(f'Unknown part {partid}')
            for pid,records in sections.items():
                if not _enabled_property(part,pid):
                    skipped=sum(len(points)for _,points in records);disabled[pid]+=skipped;layer_disabled[pid]+=skipped;continue
                section=part.get('section',{}).get(str(pid))
                if not isinstance(section,dict):raise JobImportError(f'Missing recipe for part {partid}, process {pid}')
                power,speed=_recipe(section)
                if pid in FLAT_IDS:
                    points=np.concatenate([p for _,p in records]) if records else np.empty((0,2))
                    if len(points)%2:raise JobImportError(f'{member}: odd flat-section points would leave an unmatched jump')
                    groups=[(records[0][0],points[::2],points[1::2])]
                else:
                    groups=[(recordid,points[:-1],points[1:])for recordid,points in records if len(points)>1]
                for pathid,a,b in groups:
                    a=_transform(a,part);b=_transform(b,part)
                    keep=np.any(a!=b,axis=1);zero_length+=int(np.count_nonzero(~keep));a,b=a[keep],b[keep]
                    n=len(a)
                    if not n:continue
                    arrays.append(np.column_stack([a,b,np.full(n,power),np.full(n,speed)]))
                    pids.append(np.full(n,pid,np.int16));partids.append(np.full(n,partid,np.int32))
                    # Flat points are independent jump/mark pairs, not one continuous polyline.
                    pathids.append(np.arange(next_path,next_path+n,dtype=np.int32) if pid in FLAT_IDS else np.full(n,next_path,np.int32))
                    next_path += n if pid in FLAT_IDS else 1
        vectors=np.concatenate(arrays) if arrays else np.empty((0,6))
        layers.append(_layer(index,index*thickness,vectors,
                      np.concatenate(pids)if pids else [],np.concatenate(partids)if partids else [],
                      np.concatenate(pathids)if pathids else [],
                      dict(member=member,source_points=layer_points,reserved_ignored_points=dict(layer_reserved),
                           disabled_ignored_points=dict(layer_disabled),source_member_sha256=hashlib.sha256(data).hexdigest())))
        if index%10==0 or index==maximum:progress((index+1)/(maximum+1),f'Decoded actual legacy paths: layer {index+1}/{maximum+1}')
    recipes={str(partid):part.get('section',{})for partid,part in parts.items()}
    props={str(partid):dict(name=part.get('name'),operation=part.get('property',{}).get('operation'),
                          contour_keep_same=part.get('property',{}).get('contour',{}).get('keep_same'),
                          disabled_core=part.get('property',{}).get('entity',{}).get('core',{}).get('enabled')is False)
           for partid,part in parts.items()}
    metadata=dict(source_points=raw_points,source_records=all_records,zero_length_vectors_omitted=zero_length,
                  reserved_ignored_points=dict(reserved_points),disabled_ignored_points=dict(disabled),
                  recipes=recipes,parts=props,source_header=header,
                  declared_height_m=header.get('height',maximum*thickness*1e3)*1e-3,
                  provenance=dict(geometry='imported binary float32 XY coordinates, transformed by imported part operations',
                                  sequence='inferred from recovered local controller: part/section first-seen grouping; flat pairs or consecutive polyline marks',
                                  power='imported vendor source units; watts provisionally assumed',speed='source speed interpreted as mm/s from retained controller conversion',
                                  delays='unknown: jump speed, list/laser delays and recoating time must be supplied',
                                  source_code=[SOURCE_DECODER,SOURCE_ORDER],source_code_sha256=_evidence_hashes([SOURCE_DECODER,SOURCE_ORDER])),
                  warnings=['File layer index is a base/offset; simulated powder surface is z_m + layer_thickness_m.',
                            'Actual machine scheduling and delays are not recorded; replay timing is conditional on editable assumptions.',
                            'Reserved sections are retained in diagnostics and skipped according to recovered controller semantics.',
                            'No scanner quantization, power calibration, machine communication or printer control is performed.',
                            'raduis is an unknown vendor field and is never interpreted as a beam size.'])
    return ImportedJob(**_source_meta(path,'legacy-htmjob'),layer_thickness_m=thickness,layers=layers,metadata=metadata)


def _decode_binary_layer(data):
    header=struct.Struct('<4sIddI');record=struct.Struct('<BBI')
    if len(data)<header.size:raise JobImportError('Truncated MJOB binary layer')
    magic,index,z,plane,count=header.unpack_from(data)
    if magic!=b'MLY1' or count>MAX_POINTS//2:raise JobImportError('Unsupported MJOB binary layer encoding')
    offset=header.size;paths=[];total=0;kinds=('contour','hatch','contour_segment')
    for _ in range(count):
        if offset+record.size>len(data):raise JobImportError('Truncated MJOB path header')
        pid,kind,n=record.unpack_from(data,offset);offset+=record.size;total+=n
        if pid not in PROCESS_NAMES or kind>=len(kinds) or n<2 or total>MAX_POINTS or offset+n*16>len(data):
            raise JobImportError('Invalid MJOB path count/process')
        points=np.frombuffer(data,dtype='<f8',count=n*2,offset=offset).reshape((-1,2)).copy();offset+=n*16
        paths.append(dict(process=PROCESS_NAMES[pid],kind=kinds[kind],points=points))
    if offset!=len(data):raise JobImportError('Trailing bytes in MJOB binary layer')
    return dict(index=index,z_mm=z,slice_z_mm=plane,paths=paths)


def _paths_to_layer(layer,preset,index,thickness):
    if layer.get('index')!=index:raise JobImportError('MJOB layers must be stored in index order')
    z=_finite(layer.get('z_mm'),'MJOB layer z')*1e-3
    if not math.isclose(z,index*thickness,abs_tol=1e-10):raise JobImportError('Nonuniform MJOB layer z differs from declared thickness')
    arrays=[];pids=[];pathids=[];zero=0
    for pathid,path in enumerate(layer.get('paths',[])):
        process=path.get('process');pid=PROCESS_IDS.get(process)
        if pid is None:raise JobImportError(f'Unsupported MJOB process: {process}')
        kind=path.get('kind')
        points=np.asarray(path.get('points'),dtype=float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.all(np.isfinite(points)):
            raise JobImportError('Invalid MJOB path coordinates')
        if kind=='hatch':
            if len(points)!=2:raise JobImportError('MJOB hatch must contain exactly two endpoints')
        elif kind not in ('contour','contour_segment'):raise JobImportError('Unsupported MJOB path kind')
        section=preset.get(process) if process in ('contour','infill','support') else preset.get('regional',{}).get('recipes',{}).get(process)
        if not isinstance(section,dict):raise JobImportError(f'Missing MJOB recipe for {process}')
        power,speed=_recipe(section)
        a,b=points[:-1]*1e-3,points[1:]*1e-3
        keep=np.any(a!=b,axis=1);zero+=int(np.count_nonzero(~keep));a,b=a[keep],b[keep];n=len(a)
        if not n:continue
        arrays.append(np.column_stack([a,b,np.full(n,power),np.full(n,speed)]))
        pids.append(np.full(n,pid,np.int16));pathids.append(np.full(n,pathid,np.int32))
    vectors=np.concatenate(arrays)if arrays else np.empty((0,6))
    return _layer(index,z,vectors,np.concatenate(pids)if pids else [],np.zeros(len(vectors),np.int32),
                  np.concatenate(pathids)if pathids else [],
                  dict(source_path_count=len(layer.get('paths',[])),slice_z_m=layer.get('slice_z_mm',layer.get('z_mm'))*1e-3,
                       zero_length_vectors_omitted=zero))


def _read_mjob(path,archive,progress):
    manifest=_json(archive.read('manifest.json'))
    format=manifest.get('format')
    if format not in ('metal-compiled-job','metal-slc-job','metal-cli-job'):
        raise JobImportError(f'Unsupported MJOB manifest format {format!r}. Supply heat-sim-job-v1 JSON; paths will not be invented.')
    if format=='metal-compiled-job':
        preset=manifest.get('preset',{});refs=manifest.get('layers',[])
        if not isinstance(refs,list) or not 0<len(refs)<=MAX_LAYERS:raise JobImportError('Invalid MJOB layer references')
        encoding=manifest.get('generator',{}).get('layer_encoding','json')
        if encoding not in ('json','paths-f64le-v1'):raise JobImportError('Unsupported MJOB layer encoding')
        source_layers=[];verified=0
        for ref in refs:
            data=archive.read(ref['member'])
            if hashlib.sha256(data).hexdigest()!=ref.get('sha256'):raise JobImportError('MJOB layer hash mismatch')
            source_layers.append(_decode_binary_layer(data)if encoding=='paths-f64le-v1' else _json(data));verified+=1
        source_verification={}
        for source in manifest.get('sources',[]):
            if 'member'in source and 'sha256'in source:
                digest=hashlib.sha256(archive.read(source['member'])).hexdigest()
                if digest!=source['sha256']:raise JobImportError('MJOB source-geometry hash mismatch')
                source_verification[source['member']]=digest
    else:
        if 'compiled.json'not in archive.namelist():raise JobImportError('Exchange MJOB missing compiled.json; raw source contours are not substituted')
        compiled=archive.read('compiled.json')
        if hashlib.sha256(compiled).hexdigest()!=manifest.get('compiled_sha256'):raise JobImportError('MJOB compiled-path hash mismatch')
        document=_json(compiled);source_layers=document.get('layers',[])
        preset=manifest.get('import_options',{}).get('preset',{})
        source_name='source.slc'if format=='metal-slc-job'else 'source.cli'
        source_verification={}
        if source_name in archive.namelist() and manifest.get('source_sha256'):
            digest=hashlib.sha256(archive.read(source_name)).hexdigest()
            if digest!=manifest['source_sha256']:raise JobImportError('MJOB original-source hash mismatch')
            source_verification[source_name]=digest
        verified=len(source_layers)
    thickness=_finite(preset.get('layer_thickness_mm'),'MJOB layer thickness',positive=True)*1e-3
    if not isinstance(source_layers,list)or not 0<len(source_layers)<=MAX_LAYERS:raise JobImportError('Invalid MJOB layer count')
    layers=[];total=0
    for index,source in enumerate(source_layers):
        layer=_paths_to_layer(source,preset,index,thickness);total+=len(layer.vectors)
        if total>MAX_POINTS:raise JobImportError('MJOB vector count exceeds limit')
        layers.append(layer)
        if index%10==0 or index==len(source_layers)-1:progress((index+1)/len(source_layers),f'Decoded packaged MJOB paths: layer {index+1}/{len(source_layers)}')
    metadata=dict(source_manifest=manifest,preset=preset,verified_layer_hashes=verified,
                  verified_source_hashes=source_verification,
                  provenance=dict(geometry='imported compiled path coordinates in mm; no re-slicing or generated hatch',
                                  sequence='imported stored layer/path/point sequence; actual machine scheduling is not captured',
                                  power='imported vendor source units; watts provisionally assumed',speed='explicit mm/s preset values',
                                  delays='not recorded: editable jump, inter-vector and recoating assumptions required',
                                  source_code=[SOURCE_MJOB],source_code_sha256=_evidence_hashes([SOURCE_MJOB])),
                  warnings=['A packaged contour is exposed only along its imported polyline; no interior hatch is fabricated.',
                            'Stored path order is imported, but it is not a captured printer execution timeline.',
                            'Source power_units may be vendor_source_units; delivered watts require calibration.',
                            'Source geometry and compiled-path hashes are checked where supplied.'])
    return ImportedJob(**_source_meta(path,format),layer_thickness_m=thickness,layers=layers,metadata=metadata)


def _read_normalized(path,document):
    if document.get('format')!='heat-sim-job-v1':raise JobImportError('Unsupported normalized JSON format; expected heat-sim-job-v1')
    if document.get('coordinate_units')!='m' or document.get('speed_units')!='m/s':
        raise JobImportError('Normalized JSON requires explicit m coordinates and m/s speeds')
    thickness=_finite(document.get('layer_thickness_m'),'normalized layer thickness',positive=True)
    source_layers=document.get('layers')
    if not isinstance(source_layers,list)or not 0<len(source_layers)<=MAX_LAYERS:raise JobImportError('Invalid normalized layer count')
    layers=[];total=0
    for i,source in enumerate(source_layers):
        vectors=np.asarray(source.get('vectors'),dtype=float).reshape((-1,6));n=len(vectors);total+=n
        if total>MAX_POINTS:raise JobImportError('Normalized vector limit exceeded')
        if source.get('index',i)!=i:raise JobImportError('Normalized layers must be ordered')
        z=_finite(source.get('z_m'),'normalized z')
        if not math.isclose(z,i*thickness,abs_tol=1e-10):raise JobImportError('Normalized job requires uniform z layers')
        layers.append(_layer(i,z,vectors,source.get('process_ids',np.zeros(n,np.int16)),
                             source.get('part_ids',np.zeros(n,np.int32)),source.get('path_ids',np.arange(n,dtype=np.int32)),source.get('metadata')))
    metadata=dict(document.get('metadata',{}))
    metadata.setdefault('provenance',dict(geometry='imported explicit normalized vectors',sequence='imported JSON row order; physical execution not measured',
                                        power='declared '+str(document.get('power_units','unknown')),
                                        delays='editable assumptions unless independently measured'))
    return ImportedJob(**_source_meta(path,'heat-sim-job-v1'),layer_thickness_m=thickness,layers=layers,metadata=metadata)


def save_imported_job(job: ImportedJob,folder: str|Path,copy_source=True):
    """Save compact normalized arrays, summary and optionally the exact source."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    offsets=np.r_[0,np.cumsum([len(layer.vectors)for layer in job.layers])]
    def packed(name,shape,dtype):
        arrays=[getattr(layer,name)for layer in job.layers]
        return np.concatenate(arrays)if arrays else np.empty(shape,dtype=dtype)
    temporary=folder/'.vectors.tmp.npz'
    np.savez_compressed(temporary,vectors=packed('vectors',(0,6),float),
                        process_ids=packed('process_ids',(0,),np.int16),part_ids=packed('part_ids',(0,),np.int32),
                        path_ids=packed('path_ids',(0,),np.int32),layer_offsets=offsets,
                        layer_z_m=np.array([l.z_m for l in job.layers]),layer_index=np.array([l.index for l in job.layers]))
    temporary.replace(folder/'vectors.npz')
    summary=job.summary();summary['normalized_arrays']='vectors.npz'
    summary['normalized_arrays_sha256']=hashlib.sha256((folder/'vectors.npz').read_bytes()).hexdigest()
    if copy_source and Path(job.source_path).is_file():
        target=folder/('source'+Path(job.source_path).suffix)
        if Path(job.source_path).resolve()!=target.resolve():
            if target.exists()and hashlib.sha256(target.read_bytes()).hexdigest()!=job.source_sha256:
                raise JobImportError('Existing copied source differs; choose a new import folder')
            if not target.exists():shutil.copy2(job.source_path,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=job.source_sha256:raise JobImportError('Copied source hash mismatch')
        summary['source_copy']=target.name
    output=folder/'job-summary.json';output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return output


def _read_cached(path,document):
    arrays_path=path.parent/document['normalized_arrays']
    if arrays_path.parent.resolve()!=path.parent.resolve():raise JobImportError('Unsafe cached array path')
    if document.get('normalized_arrays_sha256') and hashlib.sha256(arrays_path.read_bytes()).hexdigest()!=document['normalized_arrays_sha256']:
        raise JobImportError('Cached vector-array hash mismatch')
    with _archive(arrays_path):pass
    with np.load(arrays_path,allow_pickle=False)as source:
        arrays={key:source[key].copy()for key in source.files}
    layers=[];offsets=arrays['layer_offsets']
    if (len(offsets)!=len(document['layers'])+1 or offsets[0]!=0 or offsets[-1]!=len(arrays['vectors'])
            or np.any(np.diff(offsets)<0)):
        raise JobImportError('Cached layer offset mismatch')
    for i,row in enumerate(document['layers']):
        a,b=int(offsets[i]),int(offsets[i+1])
        layers.append(_layer(row['index'],row['z_m'],arrays['vectors'][a:b],arrays['process_ids'][a:b],
                             arrays['part_ids'][a:b],arrays['path_ids'][a:b],row.get('metadata')))
    source_path=path.parent/document.get('source_copy','')
    if document.get('source_copy'):
        if not source_path.resolve().is_relative_to(path.parent.resolve()):
            raise JobImportError('Unsafe cached source-copy path')
        if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest()!=document['source_sha256']:
            raise JobImportError('Cached job source-copy hash mismatch or missing source')
    return ImportedJob(document['name'],str(source_path)if source_path.is_file()else document['source_path'],
                       document['source_sha256'],document['format'],document['layer_thickness_m'],layers,document['metadata'])


def load_job(path: str|Path,progress=None) -> ImportedJob:
    """Load actual .htmjob/.mjob, explicit normalized JSON, or saved import cache."""
    path=Path(path).expanduser().resolve();progress=progress or(lambda f,message:None)
    if not path.is_file():raise JobImportError(f'Job file does not exist: {path}')
    if path.stat().st_size>MAX_BYTES:raise JobImportError('Job exceeds 600 MiB size limit')
    try:
        if is_zipfile(path):
            with _archive(path)as archive:
                if 'para.head'in archive.namelist():return _read_htm(path,archive,progress)
                if 'manifest.json'in archive.namelist():return _read_mjob(path,archive,progress)
                raise JobImportError('Unrecognized job ZIP: no para.head or supported manifest; supply normalized JSON')
        document=_json(path.read_bytes())
        if 'normalized_arrays'in document:return _read_cached(path,document)
        return _read_normalized(path,document)
    except JobImportError:raise
    except (ValueError,KeyError,TypeError,IndexError,struct.error,OverflowError)as exc:
        raise JobImportError(f'Malformed job data: {exc}')from exc


import_job=load_job


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/imports/job'))
    args=parser.parse_args()
    job=load_job(args.job,progress=lambda f,message:print(f'{f:.0%} {message}',flush=True))
    target=save_imported_job(job,args.output)
    print(target)
    print(json.dumps({key:value for key,value in job.summary().items()if key not in ('layers','metadata')},indent=2))

if __name__=='__main__':main()
