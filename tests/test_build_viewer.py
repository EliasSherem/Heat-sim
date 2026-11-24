"""Viewer data must remain faithful to the saved thermal states and timeline."""
import json
from pathlib import Path
import re

import numpy as np
import pytest

from heat_sim.build_viewer import load_build_frame,playback_html,probe_history


def write_frames(folder,alloy='17-4PH'):
    folder.mkdir(parents=True,exist_ok=True)
    shape=(4,3,4);x=np.arange(4)*.001;y=np.arange(3)*.001;z=np.arange(4)*.00024-.00024
    entries=[]
    for i in range(2):
        T=(293.15+np.arange(np.prod(shape)).reshape(shape)+i*100).astype('f4')
        active=np.ones(shape,'f4')
        if i==0:active[:,:,2:]=0
        consolidated=np.zeros(shape,'f4');consolidated[:,:,0]=1
        target=folder/f'{i}.npz'
        np.savez_compressed(target,temperature_k=T,x_m=x,y_m=y,z_m=z,
                            active_fraction=active,consolidated_fraction=consolidated,
                            plate_mask=np.broadcast_to(z[None,None,:]<0,shape).copy(),part_mask=np.zeros(shape,bool),support_mask=np.zeros(shape,bool))
        entries.append(dict(file=target.name,time_s=float(i),layer_index=i,laser_position_m=[.001,0,.00006],laser_on=i==1))
    return dict(config=dict(alloy=alloy),frames=entries,timeline_file='timeline.json')


def payload(html):
    match=re.search(r'const D=(.*?);const \$=',html,re.S)
    assert match,'Viewer has no embedded stored data'
    return json.loads(match.group(1))


def test_stored_temperatures_are_uninterpolated_and_probe_samples_same_cell(tmp_path):
    manifest=write_frames(tmp_path)
    position=(.0014,.0009,.00048)
    history=probe_history(tmp_path,manifest,position)
    assert np.isnan(history.iloc[0]['temperature_c']) # Material is not present yet.
    with np.load(tmp_path/'1.npz',allow_pickle=False)as raw:
        expected=float(raw['temperature_k'][1,1,3])-273.15
    assert history.iloc[1]['temperature_c']==pytest.approx(expected)
    assert history.iloc[1]['stored_temperature_c']==pytest.approx(expected)
    assert history.iloc[1]['cell_x_mm']==pytest.approx(1.)
    html=playback_html(tmp_path,manifest,max_points=100)
    data=payload(html)
    for frame_index,entry in enumerate(manifest['frames']):
        raw=load_build_frame(tmp_path,entry)
        for point,value in zip(data['points'],data['frames'][frame_index]['temperature']):
            indices=tuple(int(np.argmin(abs(raw[key]-p/1000)))for key,p in zip(('x_m','y_m','z_m'),point))
            assert value==pytest.approx(round(float(raw['temperature_k'][indices])-273.15,2),abs=.011)
    # Cursor playback selects one calculated frame. Linear interpolation is
    # reserved for positions within real beam events, never temperature fields.
    assert 'currentIndex=frameAt(time);let f=D.frames[currentIndex]'in html
    assert 'no thermal interpolation'in html
    assert not re.search(r'lerp\([^)]*temperature',html)


def test_temperature_scale_is_celsius_and_locked_across_alloys(tmp_path):
    first=write_frames(tmp_path/'a','17-4PH');second=write_frames(tmp_path/'b','316L')
    a=payload(playback_html(tmp_path/'a',first,temperature_min_c=20,temperature_max_c=1800))
    b=payload(playback_html(tmp_path/'b',second,temperature_min_c=20,temperature_max_c=1800))
    assert (a['vmin'],a['vmax'])==(b['vmin'],b['vmax'])==(20,1800)
    assert '<input id="lock" type="checkbox" checked>'in playback_html(tmp_path/'a',first)


def test_display_sampling_retains_each_selectable_z_plane(tmp_path):
    manifest=write_frames(tmp_path)
    data=payload(playback_html(tmp_path,manifest,max_points=8))
    assert len(data['points'])<=8
    assert sorted(set(p[2]for p in data['points']))==pytest.approx(data['z'])
    assert data['sample_policy']=='stratified by z plane'
    assert data['cell_mm']==pytest.approx([1.,1.,.24])


def test_npz_event_schedule_decodes_beam_motion_without_thermal_interpolation(tmp_path):
    manifest=write_frames(tmp_path)
    dtype=np.dtype([('t0_s','f8'),('t1_s','f8'),('start_m','f8',(3,)),('end_m','f8',(3,)),
                    ('kind','u1'),('power_w','f8'),('layer_index','i4')])
    events=np.zeros(1,dtype=dtype);events[0]=(0,1,[0,0,.00006],[.001,0,.00006],1,120,0)
    np.savez_compressed(tmp_path/'timeline.npz',events=events)
    manifest.update(timeline_file='timeline.npz',timeline_event_count=1)
    data=payload(playback_html(tmp_path,manifest))
    assert data['trajectory_status']=='exact_stored_events'
    assert data['events'][0]['p']==pytest.approx([0,0,.06])
    assert data['events'][0]['q']==pytest.approx([1,0,.06])
    assert data['events'][0]['on'] is True
    manifest['timeline_event_count']=40_001
    data=payload(playback_html(tmp_path,manifest))
    assert data['events']==[]
    assert data['trajectory_status']=='not_embedded'


def test_untrusted_paths_nonfinite_fields_and_script_names_rejected_or_escaped(tmp_path):
    manifest=write_frames(tmp_path,'</script><script>alert(1)</script>')
    html=playback_html(tmp_path,manifest)
    assert '</script><script>alert(1)'not in html
    assert payload(html)['alloy']=='</script><script>alert(1)</script>'
    with pytest.raises(ValueError,match='inside'):load_build_frame(tmp_path,dict(file='../outside.npz'))
    bad=load_build_frame(tmp_path,manifest['frames'][0]);bad['temperature_k'][0,0,0]=np.nan
    np.savez_compressed(tmp_path/'0.npz',**bad)
    with pytest.raises(ValueError,match='Non-finite'):playback_html(tmp_path,manifest)


def test_mixed_frame_meshes_and_invalid_scale_fail_clearly(tmp_path):
    manifest=write_frames(tmp_path)
    with pytest.raises(ValueError,match='Celsius bounds'):playback_html(tmp_path,manifest,temperature_max_c=np.nan)
    bad=load_build_frame(tmp_path,manifest['frames'][1]);bad['x_m']=bad['x_m']+.0001
    np.savez_compressed(tmp_path/'1.npz',**bad)
    with pytest.raises(ValueError,match='same stored mesh'):playback_html(tmp_path,manifest)


def test_large_schedule_can_show_one_exact_layer_and_hold_all_thermal_frames(tmp_path):
    manifest=write_frames(tmp_path)
    dtype=np.dtype([('t0_s','f8'),('t1_s','f8'),('start_m','f8',(3,)),('end_m','f8',(3,)),
                    ('kind','u1'),('power_w','f8'),('layer_index','i4')])
    events=np.zeros(40001,dtype=dtype)
    events['t0_s']=np.arange(len(events));events['t1_s']=np.arange(len(events))+1
    events['kind']=1;events['power_w']=120;events['layer_index']=1
    events['end_m'][:,0]=.001
    events['layer_index'][100:103]=7
    np.savez_compressed(tmp_path/'timeline.npz',events=events)
    manifest.update(timeline_file='timeline.npz',timeline_event_count=len(events))
    html=playback_html(tmp_path,manifest,trajectory_layer=7)
    data=payload(html)
    assert data['trajectory_status']=='exact_stored_layer_events'
    assert data['trajectory_scope']==dict(layer_index=7,start_s=100.,end_s=103.)
    assert data['trajectory_waiting'] is True # Stored thermal fields stop at1s; future heat is never invented.
    assert len(data['events'])==3 and all(e['layer']==7 for e in data['events'])
    assert [f['t']for f in data['frames']]==[0.,1.] # All calculated fields remain.
    assert data['events'][0]['q']==pytest.approx([1.,0.,0.])
    assert 'source layer 'in html and 'latest calculated frame is held'in html
    assert 'currentIndex=frameAt(time)'in html


def test_partial_vertical_cell_display_height_uses_deposited_fraction_only(tmp_path):
    import shutil
    import subprocess
    node=shutil.which('node')
    if node is None:
        pytest.skip('Optional Node runtime unavailable for executing the viewer geometry helper')
    html=playback_html(tmp_path,write_frames(tmp_path))
    helper=re.search(r'^function displayPoint.*$',html,re.M).group(0)
    script='const D={types:[0,2],dz_mm:1.5};'+helper+';const p=[1,2,.75],f={active:[.04,1]};console.log(JSON.stringify([displayPoint(p,0,f),displayPoint(p,1,f),p]));'
    result=subprocess.run([node,'-e',script],capture_output=True,text=True,check=True)
    partial,plate,unchanged=json.loads(result.stdout)
    assert partial==pytest.approx([1,2,.03]) # 60um of a1.5mm cell has its centre at30um.
    assert plate==unchanged==[1,2,.75]
    assert 'const c=displayPoint(p,i,f)'in html
    assert 'plane choices retain fixed finite-volume centres'in html


def _node_json(script):
    import shutil
    import subprocess
    node=shutil.which('node')
    if node is None:pytest.skip('Optional Node runtime unavailable')
    output=subprocess.run([node,'-e',script],capture_output=True,text=True,check=True)
    return json.loads(output.stdout)


def test_full_large_schedule_uses_all_saved_time_and_exact_compact_layer_clock(tmp_path):
    manifest=write_frames(tmp_path)
    manifest['frames'][-1].update(time_s=2671.4,layer_index=724)
    manifest.update(status='complete',physical_duration_s=2671.4,
                    config=dict(alloy='17-4PH',layer_start=0,layer_end=725))
    dtype=np.dtype([('t0_s','f8'),('t1_s','f8'),('start_m','f8',(3,)),('end_m','f8',(3,)),
                    ('kind','u1'),('power_w','f8'),('layer_index','i4')])
    raw=np.zeros(40001,dtype=dtype)
    boundaries=np.linspace(0.,2671.4,len(raw)+1)
    raw['t0_s']=boundaries[:-1];raw['t1_s']=boundaries[1:]
    raw['layer_index']=np.floor(np.arange(len(raw))*725/len(raw)).astype(int)
    np.savez_compressed(tmp_path/'timeline.npz',events=raw)
    manifest.update(timeline_file='timeline.npz',timeline_event_count=len(raw))
    html=playback_html(tmp_path,manifest)
    data=payload(html)
    assert data['trajectory_scope'] is None
    assert data['events']==[] # Never fake an exact beam using decimated events.
    assert len(data['layer_spans'])==725
    assert data['layer_spans'][0]['a']==0.
    assert data['layer_spans'][-1]['b']==pytest.approx(2671.4)
    assert data['simulated_layer_end']==725
    bounds_helper=re.search(r'^function playbackBounds.*$',html,re.M).group(0)
    clock_helper=re.search(r'^function layerAt.*$',html,re.M).group(0)
    rate_helper=re.search(r'^function playbackRate.*$',html,re.M).group(0)
    minimal={k:data[k] for k in ('frames','trajectory_scope','layer_spans')}
    script='const D='+json.dumps(minimal)+';'+bounds_helper+clock_helper+rate_helper
    script+=';const {begin,end}=playbackBounds(D);console.log(JSON.stringify({begin,end,rate:playbackRate("auto"),layers:[layerAt(0,{layer:0}),layerAt(2671.4,{layer:0})]}));'
    result=_node_json(script)
    assert result['begin']==0. and result['end']==pytest.approx(2671.4)
    assert result['layers']==[0,724]
    assert result['rate']==pytest.approx(2671.4/60)
    assert 'Show finished print'in html and 'id="start"'in html
    live=payload(playback_html(tmp_path,manifest,live=True))
    assert live['live'] is True
    assert 'time=D.live?end:begin'in html


def test_frame_masks_distinguish_exposed_part_support_and_hidden_powder(tmp_path):
    manifest=write_frames(tmp_path)
    raw=load_build_frame(tmp_path,manifest['frames'][-1])
    raw['part_mask'][1,1,1]=True
    raw['support_mask'][2,1,1]=True
    np.savez_compressed(tmp_path/'1.npz',**raw)
    html=playback_html(tmp_path,manifest)
    data=payload(html)
    assert any(data['frames'][-1]['part'])
    assert any(data['frames'][-1]['support'])
    assert set(data['types'])=={0,1,2,3}
    region=re.search(r'^function regionAt.*$',html,re.M).group(0)
    visible=re.search(r'^function visible.*$',html,re.M).group(0)
    follow=re.search(r'^function followPlane.*$',html,re.M).group(0)
    script='const D={types:[2,0,1,3],points:[[0,0,-1],[0,0,1],[0,0,2],[0,0,3]],z:[-1,1,2,3]};const controls={plate:{checked:true},solid:{checked:true},support:{checked:true},powder:{checked:false},plane:{value:"0"}};const $=id=>controls[id];'
    script+=region+visible+follow
    script+=';const f={active:[1,1,1,1],solid:[1,0,0,0],part:[false,true,true,false],support:[false,false,true,false]};followPlane(f);const later=D.points.map((_,i)=>visible(i,f));const early=D.points.map((_,i)=>visible(i,{active:f.active,solid:[1,0,0,0],part:[false,false,false,false],support:[false,false,false,false]}));console.log(JSON.stringify({later,early,plane:controls.plane.value}));'
    result=_node_json(script)
    assert result['later']==[True,True,True,False]
    assert result['early']==[True,False,False,False]
    assert result['plane']=='2' # Follows exposed height, not unexposed powder.
    assert '<input id="powder" type="checkbox">'in html
    assert 'Exposed part'in html and 'Follow build height'in html


def test_complete_embedded_viewer_script_parses(tmp_path):
    import shutil
    import subprocess
    node=shutil.which('node')
    if node is None:pytest.skip('Optional Node runtime unavailable')
    html=playback_html(tmp_path,write_frames(tmp_path))
    script=re.search(r'<script>(.*)</script>',html,re.S).group(1)
    check=subprocess.run([node,'--check'],input=script,capture_output=True,text=True)
    assert check.returncode==0,check.stderr


def test_voxel_corners_use_real_cell_dimensions_and_only_deposited_vertical_fraction(tmp_path):
    html=playback_html(tmp_path,write_frames(tmp_path))
    functions=''.join(re.search(r'^function '+name+r'.*$',html,re.M).group(0)
                      for name in ('displayPoint','voxelCorners'))
    script='const D={types:[0,2],dz_mm:1.5,cell_mm:[2,3,1.5]};'+functions
    script+=';const f={active:[.04,1]};console.log(JSON.stringify({part:voxelCorners([4,6,.75],0,f),plate:voxelCorners([4,6,-.75],1,f)}));'
    actual=_node_json(script)
    part=np.asarray(actual['part']);plate=np.asarray(actual['plate'])
    np.testing.assert_allclose(part.min(axis=0),[3.,4.5,0.],atol=1e-12)
    np.testing.assert_allclose(part.max(axis=0),[5.,7.5,.06],atol=1e-12)
    np.testing.assert_allclose(plate.min(axis=0),[3.,4.5,-1.5],atol=1e-12)
    np.testing.assert_allclose(plate.max(axis=0),[5.,7.5,0.],atol=1e-12)


def test_exposed_zero_consolidation_cells_have_filled_projected_faces_and_readable_colors(tmp_path):
    html=playback_html(tmp_path,write_frames(tmp_path))
    helpers=''.join(re.search(r'^function '+name+r'.*$',html,re.M).group(0)
                    for name in ('displayPoint','voxelCorners','viewPoint','voxelFaces','regionAt','visible'))
    script='const D={types:[0,1,2],dz_mm:1.5,cell_mm:[2,2,1.5]};let yaw=-.7,pitch=.65,zoom=1;const center=[0,0,0],span=20;const $=id=>({checked:true});'+helpers
    script+=';const f={active:[1,1,1],solid:[0,0,1],part:[true,true,false],support:[false,true,false]};const faces=voxelFaces([0,0,1],0,f,640,380);console.log(JSON.stringify({visible:visible(0,f),faces,area:faces.map(face=>Math.abs(face.vertices.reduce((a,p,i)=>{const q=face.vertices[(i+1)%4];return a+p[0]*q[1]-q[0]*p[1]},0))/2)}));'
    result=_node_json(script)
    assert result['visible'] is True
    assert len(result['faces'])==3
    assert all(area>100 for area in result['area'])
    assert all(np.isfinite(face['depth']) for face in result['faces'])
    assert 'ctx.closePath();ctx.fill();ctx.stroke()'in html
    assert 'faces.sort((a,b)=>b.depth-a.depth)'in html
    # CSS legend and temperature function share the same brighter five stops.
    assert '#397fba,#559fc5,#c17594,#ed8d51,#fce65a'in html
    assert 'const stops=[[57,127,186],[85,159,197],[193,117,148],[237,141,81],[252,230,90]]'in html


def test_live_view_avoids_full_schedule_and_hides_playback_controls(tmp_path,monkeypatch):
    manifest=write_frames(tmp_path)
    manifest.update(timeline_file='timeline.npz',timeline_event_count=2_000_000)
    (tmp_path/'timeline.npz').write_bytes(b'not a readable event archive')
    html=playback_html(tmp_path,manifest,live=True)
    assert payload(html)['events']==[] and payload(html)['layer_spans']==[]
    assert '.live .playback-only{display:none}'in html
    assert '<button class="playback-only" id="play">'in html
    assert 'Latest calculated field · source layer'in html
    assert 'time=D.live?end:begin'in html
