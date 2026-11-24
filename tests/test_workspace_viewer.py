"""The main viewport must depict imported paths and saved computed fields."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from heat_sim.job_import import ImportedJob,ImportedLayer
from heat_sim.workspace_viewer import (geometry_preview_data,geometry_preview_html,
                                       thermal_workspace_data,thermal_workspace_html)


def job_fixture():
    layers=[]
    for i in range(4):
        v=np.array([[.001,.002,.003,.004,120.,1.], [.002,.001,.004,.003,80.,.8]])
        layers.append(ImportedLayer(i,i*.00006,v,np.array([0,16]),np.zeros(2,int),np.arange(2)))
    return ImportedJob('actual-path-fixture.mjob','none','abc123','test',.00006,layers,{})


def case_fixture(folder,times=(0.,1.,2.),offset=0.,exposed_from=0,hash_value='source-a',shift=0.):
    folder.mkdir(parents=True,exist_ok=True)
    xyz=[np.array([.00025,.00075])+shift,np.array([.00025,.00075]),np.array([-.00025,.00025])]
    shape=(2,2,2);frames=[]
    for i,t in enumerate(times):
        temperature=(293.151234+offset+np.arange(8).reshape(shape)*.12345+i*10.12345).astype('f4')
        active=np.ones(shape,dtype='f4');active[:,:,1]=float(i>=exposed_from)
        part=np.zeros(shape,bool);part[:,:,1]=True
        plate=np.zeros(shape,bool);plate[:,:,0]=True
        target=folder/f'{i}.npz'
        np.savez_compressed(target,temperature_k=temperature,active_fraction=active,
            consolidated_fraction=np.zeros(shape,'f4'),part_mask=part,support_mask=np.zeros(shape,bool),
            plate_mask=plate,x_m=xyz[0],y_m=xyz[1],z_m=xyz[2],time_s=t)
        frames.append(dict(file=target.name,time_s=float(t),layer_index=i,laser_on=False,laser_position_m=None))
    manifest=dict(status='complete',config=dict(alloy='17-4PH',layer_start=0,layer_end=3,spacing_m=.0005,z_spacing_m=.0005),
                  frames=frames,job=dict(name='part.mjob',source_sha256=hash_value),physical_duration_s=float(times[-1]))
    return dict(folder=folder,manifest=manifest,label=folder.name)


def data_from_html(html):
    return json.loads(re.search(r'const W=(.*?);const \$=',html,re.S).group(1))


def js_helper(name):
    from heat_sim.workspace_viewer import _HTML
    return re.search(r'^function '+name+r'\(.*$',_HTML,re.M).group(0)


def run_node(source):
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable for executing actual viewport clock helpers')
    result=subprocess.run([node,'-e',source],text=True,capture_output=True,check=True)
    return json.loads(result.stdout)


def execute_viewer(html,after,storage=None):
    """Execute the real inline script with only canvas/DOM drawing stubbed."""
    script=html.split('<script>',1)[1].split('</script>',1)[0]
    setup=r'''
const vm=require('vm');const elements={},drawCalls={};let stored=STORAGE;
function element(tag){const e={tag,style:{},classList:{add(){},toggle(){}},children:[],clientWidth:1000,clientHeight:440,checked:false,value:'',appendChild(child){this.children.push(child)},setAttribute(){},setPointerCapture(){},getBoundingClientRect(){return{left:0,top:0}},getContext(){return new Proxy({},{get:(target,key)=>target[key]||(()=>{const calls=drawCalls[e.id]||(drawCalls[e.id]={});calls[key]=(calls[key]||0)+1})})}};Object.defineProperty(e,'id',{get(){return this._id},set(v){this._id=v;elements[v]=this}});return e}
for(const match of HTML.matchAll(/id="([^"]+)"/g)){const e=element('div');e.id=match[1]}
for(const id of ['supports','plate','follow','followcalculation'])elements[id].checked=true;elements.speed.value='auto';elements.representation.value='part';
const document={getElementById:id=>elements[id],createElement:element,body:{classList:{add(){}}},documentElement:{style:{setProperty(){}}}};
const context=vm.createContext({document,drawCalls,window:{devicePixelRatio:1},sessionStorage:{getItem(){return stored},setItem(k,v){stored=v}},requestAnimationFrame(){},console});
vm.runInContext(SCRIPT,context);const result=vm.runInContext(AFTER,context);console.log(JSON.stringify({result,stored:JSON.parse(stored)}));
'''.replace('STORAGE',json.dumps(None if storage is None else json.dumps(storage))).replace('HTML',json.dumps(html)).replace('SCRIPT',json.dumps(script)).replace('AFTER',json.dumps(after))
    return run_node(setup)


def scale_case(folder,offset=0.):
    """Part extrema differ from hot plate/powder and inactive future cells."""
    case=case_fixture(folder)
    for entry,values in zip(case['manifest']['frames'],((80.,116.),(90.,145.),(100.,212.))):
        target=case['folder']/entry['file']
        with np.load(target) as saved:raw={key:saved[key].copy() for key in saved.files}
        temperature=np.full((2,2,2),1800.,dtype=float)
        temperature[0,0,1]=values[0]+offset
        temperature[0,1,1]=values[1]+offset
        temperature[1,0,1]=2400. # Unexposed powder.
        temperature[1,1,1]=3000. # Material not yet deposited.
        raw['temperature_k']=temperature+273.15
        raw['part_mask'][1,:,1]=False
        raw['active_fraction'][1,1,1]=0.
        np.savez_compressed(target,**raw)
    return case


def test_auto_colour_range_fits_current_part_and_excludes_plate_powder_and_future_cells(tmp_path):
    case=scale_case(tmp_path/'part')
    before=data_from_html(thermal_workspace_html([case],initial_time=0,auto_temperature_range=True))
    result=execute_viewer(thermal_workspace_html([case],initial_time=0,auto_temperature_range=True),
        "({min:scaleMinimum,max:scaleMaximum,label:$('scale-label').textContent,minimum:$('minimum').textContent,maximum:$('maximum').textContent,raw:W.cases[0].frames[0].temperature})")['result']
    assert result['min']==80. and result['max']==120.
    assert result['minimum']=='80 °C' and result['maximum']=='120 °C'
    assert 'locked' in result['label']
    assert result['raw']==before['cases'][0]['frames'][0]['temperature']
    assert max(result['raw'])==2400. # These values are unchanged, only excluded from fitting.


def test_auto_colour_scale_remains_locked_when_scrubbing_and_refits_only_on_request(tmp_path):
    case=scale_case(tmp_path/'part')
    script="""
const initial=[scaleMinimum,scaleMaximum];seek(2);const scrub={range:[scaleMinimum,scaleMaximum],note:$('scale-note').textContent,point:pointTemperature(scenes[0],scenes[0].frames[2],1)};
$('speed').value='16';$('fitcolors').onclick();({initial,scrub,fitted:[scaleMinimum,scaleMaximum],note:$('scale-note').textContent,time})
"""
    result=execute_viewer(thermal_workspace_html([case],initial_time=0,auto_temperature_range=True),script)['result']
    assert result['initial']==[80.,120.]
    assert result['scrub']['range']==[80.,120.]
    assert 'Above 120 °C' in result['scrub']['note']
    assert result['fitted']==[100.,220.]
    assert result['note']=='' and result['time']==2.


def test_comparison_auto_colour_range_uses_one_union_for_both_materials(tmp_path):
    a=scale_case(tmp_path/'17-4')
    b=scale_case(tmp_path/'316L',offset=20.)
    html=thermal_workspace_html([a,b],initial_time=0,auto_temperature_range=True)
    result=execute_viewer(html,"({range:[scaleMinimum,scaleMaximum],bounds:currentTemperatureBounds(),colours:scenes.map(s=>color(100)),raw:scenes.map(s=>s.frames[0].temperature[1])})")['result']
    assert result['range']==[80.,140.]
    assert result['bounds']==dict(low=80.,high=136.)
    assert result['colours'][0]==result['colours'][1]
    assert result['raw']==[80.,100.]


def test_custom_colour_range_is_unchanged_by_time_or_fit_action(tmp_path):
    case=scale_case(tmp_path/'part')
    html=thermal_workspace_html([case],temperature_min_c=40.,temperature_max_c=1400.)
    result=execute_viewer(html,"seek(0);$('fitcolors').onclick();({range:[scaleMinimum,scaleMaximum],label:$('scale-label').textContent,fitDisabled:$('fitcolors').disabled})")['result']
    assert result['range']==[40.,1400.]
    assert 'custom' in result['label'] and result['fitDisabled']


def test_live_auto_scale_updates_shared_range_until_user_pauses(tmp_path):
    case=scale_case(tmp_path/'part')
    html=thermal_workspace_html([case],live=True,auto_temperature_range=True)
    initial=execute_viewer(html,"({range:[scaleMinimum,scaleMaximum],following,label:$('scale-label').textContent})")
    assert initial['result']['range']==[100.,220.]
    assert initial['result']['following'] and 'auto live' in initial['result']['label']
    paused=execute_viewer(html,"seek(0);({range:[scaleMinimum,scaleMaximum],following,label:$('scale-label').textContent})",initial['stored'])
    assert paused['result']['range']==[100.,220.]
    assert not paused['result']['following'] and 'locked' in paused['result']['label']
    restored=execute_viewer(html,"({range:[scaleMinimum,scaleMaximum],time,following})",paused['stored'])['result']
    assert restored==dict(range=[100.,220.],time=0.,following=False)


def test_uniform_initial_field_has_finite_span_and_old_wide_scale_is_not_restored(tmp_path):
    case=scale_case(tmp_path/'part')
    first=case['folder']/case['manifest']['frames'][0]['file']
    with np.load(first) as saved:raw={key:saved[key].copy() for key in saved.files}
    raw['temperature_k'][:]=353.15
    raw['part_mask'][:]=False
    np.savez_compressed(first,**raw)
    html=thermal_workspace_html([case],initial_time=0,auto_temperature_range=True)
    old=dict(run=str(case['folder'].resolve()),mode='thermal',temperatureRange=dict(min=20.,max=1800.))
    result=execute_viewer(html,"({range:[scaleMinimum,scaleMaximum],finite:Number.isFinite(scaleMinimum)&&scaleMaximum>scaleMinimum})",old)['result']
    assert result['range']==[75.,85.] and result['finite']


def test_switching_from_custom_to_fitted_colours_recomputes_current_range(tmp_path):
    case=scale_case(tmp_path/'part')
    automatic=thermal_workspace_html([case],initial_time=0,auto_temperature_range=True)
    fitted=execute_viewer(automatic,"({min:scaleMinimum,max:scaleMaximum})")
    custom=execute_viewer(thermal_workspace_html([case],temperature_min_c=20.,temperature_max_c=1800.),
                          "({min:scaleMinimum,max:scaleMaximum})",fitted['stored'])
    assert custom['result']==dict(min=20.,max=1800.)
    assert custom['stored']['temperatureRange'] is None
    restored=execute_viewer(automatic,"({min:scaleMinimum,max:scaleMaximum})",custom['stored'])
    assert restored['result']==dict(min=80.,max=120.)


def test_live_fitted_scale_changes_when_new_calculated_temperatures_arrive(tmp_path):
    case=scale_case(tmp_path/'part')
    old=execute_viewer(thermal_workspace_html([case],live=True,auto_temperature_range=True),
                       "({range:[scaleMinimum,scaleMaximum],following})")
    target=case['folder']/case['manifest']['frames'][-1]['file']
    with np.load(target) as saved:raw={key:saved[key].copy() for key in saved.files}
    raw['temperature_k'][0,0,1]=400.+273.15
    raw['temperature_k'][0,1,1]=780.+273.15
    np.savez_compressed(target,**raw)
    updated=execute_viewer(thermal_workspace_html([case],live=True,auto_temperature_range=True),
                           "({range:[scaleMinimum,scaleMaximum],following,label:$('scale-label').textContent})",old['stored'])
    assert old['result']['range']==[100.,220.]
    assert updated['result']['range']==[400.,800.]
    assert updated['result']['following'] and 'auto live' in updated['result']['label']


def test_preview_shows_supports_by_default_and_preserves_an_explicit_display_choice():
    html=geometry_preview_html(job_fixture())
    default=execute_viewer(html,"({supports:$('supports').checked})")
    assert default['result']['supports']
    hidden=execute_viewer(html,"$('supports').checked=false;$('supports').onchange();({supports:$('supports').checked})",default['stored'])
    assert not hidden['result']['supports']
    restored=execute_viewer(html,"({supports:$('supports').checked})",hidden['stored'])
    assert not restored['result']['supports']


def test_preview_uses_only_actual_vectors_and_neutral_geometry_label():
    job=job_fixture();copy=[layer.vectors.copy() for layer in job.layers]
    data=geometry_preview_data(job,max_segments=4)
    assert data['mode']=='geometry' and data['vmin'] is None
    assert len(data['cases'][0]['segments'])<=4
    assert {s[7] for s in data['cases'][0]['segments']}=={0,1,2,3}
    for s in data['cases'][0]['segments']:
        layer=job.layers[s[7]]
        assert any(np.allclose(s[:2],v[:2]*1000) and np.allclose(s[3:5],v[2:4]*1000) for v in layer.vectors)
        assert s[2]==pytest.approx((layer.z_m+job.layer_thickness_m)*1000)
    for layer,before in zip(job.layers,copy):np.testing.assert_array_equal(layer.vectors,before)
    html=geometry_preview_html(job)
    assert 'Geometry preview · no thermal temperatures' in html
    assert data_from_html(html)['cases'][0]['source_sha256']=='abc123'


def test_point_temperature_is_exact_saved_cell_not_a_rounded_or_fabricated_trail(tmp_path):
    case=case_fixture(tmp_path/'a')
    data=thermal_workspace_data([case])
    scene=data['cases'][0]
    for frame,entry in zip(scene['frames'],case['manifest']['frames']):
        with np.load(case['folder']/entry['file']) as raw:
            for point,temperature in zip(scene['points'],frame['temperature']):
                ijk=tuple(int(np.argmin(abs(raw[key]-p/1000))) for key,p in zip(('x_m','y_m','z_m'),point))
                assert temperature==float(raw['temperature_k'][ijk])-273.15
    assert any(abs(value-round(value,2))>1e-4 for value in scene['frames'][0]['temperature'])
    source=js_helper('pointTemperature')
    actual=run_node(source+";console.log(JSON.stringify([pointTemperature({}, {active:[1,0],temperature:[123.456789,800]},0),pointTemperature({}, {active:[1,0],temperature:[123.456789,800]},1)]));")
    assert actual==[123.456789,None]


def test_comparison_selects_each_case_by_same_physical_time_not_same_frame_index(tmp_path):
    a=case_fixture(tmp_path/'17-4',times=(0.,1.,2.))
    b=case_fixture(tmp_path/'316L',times=(0.,.4,1.4,2.4),offset=20.)
    data=thermal_workspace_data([a,b],temperature_min_c=40,temperature_max_c=1400)
    source=js_helper('physicalBounds')+';'+js_helper('frameIndex')
    result=run_node('const cases='+json.dumps(data['cases'])+';'+source+';console.log(JSON.stringify({bounds:physicalBounds(cases),indices:cases.map(s=>frameIndex(s,1.2))}));')
    assert result==dict(bounds=dict(begin=0,end=2),indices=[1,1])
    # At .8s the two independently saved time grids select different indexes.
    result=run_node('const cases='+json.dumps(data['cases'])+';'+source+';console.log(JSON.stringify(cases.map(s=>frameIndex(s,.8))));')
    assert result==[0,1]
    assert all(scene['vmin']==40 and scene['vmax']==1400 for scene in data['cases'])
    assert data['cases'][0]['frames'][0]['temperature']!=data['cases'][1]['frames'][0]['temperature']


def test_comparison_can_show_pending_case_without_substituting_other_material(tmp_path):
    a=case_fixture(tmp_path/'17-4')
    pending=dict(folder=None,manifest=None,label='316L')
    data=thermal_workspace_data([a,pending],live=True)
    assert data['cases'][1]['pending'] and data['cases'][1]['frames']==[]
    assert data['cases'][1]['label']=='316L'
    result=run_node('const cases='+json.dumps(data['cases'])+';'+js_helper('physicalBounds')+';console.log(JSON.stringify(physicalBounds(cases)));')
    assert result==dict(begin=0,end=2)
    assert 'No results from the other material are substituted.' in thermal_workspace_html([a,pending],live=True)


def test_comparison_grids_are_shared_even_while_one_case_has_only_early_active_layers(tmp_path):
    a=case_fixture(tmp_path/'a')
    b=case_fixture(tmp_path/'b',times=(0.,),exposed_from=5)
    data=thermal_workspace_data([a,b],live=True)
    assert data['cases'][0]['points']==data['cases'][1]['points']
    assert any(value==0 for value in data['cases'][1]['frames'][0]['active'])
    for point,active in zip(data['cases'][1]['points'],data['cases'][1]['frames'][0]['active']):
        if point[2]>0:assert active==0


def test_comparison_rejects_mismatched_source_or_mesh_instead_of_misleading_sync(tmp_path):
    a=case_fixture(tmp_path/'a')
    b=case_fixture(tmp_path/'b',hash_value='another-job')
    with pytest.raises(ValueError,match='source job'):thermal_workspace_data([a,b])
    b=case_fixture(tmp_path/'c',shift=.001)
    with pytest.raises(ValueError,match='matching saved thermal grids'):thermal_workspace_data([a,b])


def test_before_first_available_frame_is_withheld_and_after_last_is_held(tmp_path):
    case=case_fixture(tmp_path/'a',times=(.5,1.,2.))
    data=thermal_workspace_data([case])
    result=run_node('const s='+json.dumps(data['cases'][0])+';'+js_helper('frameIndex')+';console.log(JSON.stringify([frameIndex(s,0),frameIndex(s,.75),frameIndex(s,3)]));')
    assert result==[-1,0,2]


def test_html_escapes_imported_names_and_all_javascript_parses(tmp_path):
    case=case_fixture(tmp_path/'a');case['label']='</script><script>bad()</script>'
    html=thermal_workspace_html([case],live=True,show_probe=True)
    assert '</script><script>bad()' not in html
    assert data_from_html(html)['cases'][0]['label']==case['label']
    script=html.split('<script>',1)[1].split('</script>',1)[0]
    result=run_node('new Function('+json.dumps(script)+');console.log(JSON.stringify(true));')
    assert result is True
    assert html.index('<div class="scenes"')<html.index('<button id="play"')
    assert 'Follow calculation' in html and 'sessionStorage' in html


def test_actual_paths_use_only_completed_events_and_exact_midpoint_cell_temperature(tmp_path):
    case=case_fixture(tmp_path/'a')
    events=[dict(t0_s=0.,t1_s=.5,start_m=[.00025,.00025,.00025],end_m=[.00075,.00025,.00025],
                 laser_on=True,power_w=120.,kind='scan',section=0),
            dict(t0_s=.5,t1_s=1.5,start_m=[.00075,.00025,.00025],end_m=[.00075,.00075,.00025],
                 laser_on=True,power_w=80.,kind='scan',section=16),
            dict(t0_s=1.5,t1_s=2.,start_m=[.00025,.00075,.00025],end_m=[.00075,.00075,.00025],
                 laser_on=False,power_w=0.,kind='scan',section=0)]
    (case['folder']/'timeline.json').write_text(json.dumps(events))
    case['manifest'].update(timeline_file='timeline.json',timeline_event_count=3)
    scene=thermal_workspace_data([case])['cases'][0]
    assert len(scene['paths'])==2 # Disabled/zero-power event is not exposed.
    assert list(scene['paths'][0]['p'])==[.25,.25,.25]
    assert scene['paths'][1]['support'] is True
    for i,entry in enumerate(case['manifest']['frames']):
        with np.load(case['folder']/entry['file']) as raw:
            for path in scene['paths']:
                midpoint=(np.array(path['p'])+path['q'])/2000
                ijk=tuple(int(np.argmin(abs(raw[key]-p))) for key,p in zip(('x_m','y_m','z_m'),midpoint))
                assert scene['frames'][i]['temperature'][path['cell']]==float(raw['temperature_k'][ijk])-273.15
    result=run_node('const s='+json.dumps(scene)+';'+js_helper('completedPath')+';console.log(JSON.stringify(s.paths.map(p=>completedPath(p,s.frames[1]))));')
    assert result==[True,False] # Future scan geometry is not drawn into a held field.


def test_projection_fits_wide_short_browser_without_cropping_the_part():
    source='const lo=[0,0,0],hi=[10,30,20],centre=[5,15,10],span=30;let yaw=-.7,pitch=.65,zoom=1,projectionCache=null;'
    source+=js_helper('rotated')+';'+js_helper('projectionSize')+';'+js_helper('project')
    source+=';console.log(JSON.stringify(Array.from({length:8},(_,k)=>project(lo.map((v,a)=>(k>>a)&1?hi[a]:v),2000,440))));'
    projected=run_node(source)
    assert all(25<p[0]<1975 and 20<p[1]<420 for p in projected)


def test_real_viewer_runtime_scrubs_and_preserves_pause_without_freezing_a_new_live_run(tmp_path):
    case=case_fixture(tmp_path/'a')
    html=thermal_workspace_html([case],live=True)
    old=dict(run='different-run',following=False,time=0,yaw=.3,pitch=.4,zoom=.8)
    current=execute_viewer(html,"({time,following,yaw,pitch,clock:$('clock').textContent})",old)
    assert current['result']['time']==2 and current['result']['following']
    assert current['result']['yaw']==.3
    paused=execute_viewer(html,"seek(.75);({time,following,field:$('frame-0').textContent})",current['stored'])
    assert paused['result']['time']==.75 and not paused['result']['following']
    assert '0.000 s' in paused['result']['field']
    restored=execute_viewer(html,"({time,following})",paused['stored'])
    assert restored['result']==dict(time=.75,following=False)


def test_real_viewer_runtime_geometry_and_pending_comparison_do_not_invent_fields(tmp_path):
    preview=execute_viewer(geometry_preview_html(job_fixture()),"({mode:W.mode,label:$('kind').textContent,supports:$('supports').checked})")
    assert preview['result']['mode']=='geometry' and preview['result']['supports']
    a=case_fixture(tmp_path/'a')
    html=thermal_workspace_html([a,dict(folder=None,manifest=None,label='316L')],live=True)
    result=execute_viewer(html,"({right:$('frame-1').textContent,note:$('note-1').textContent,time})")['result']
    assert result['right']=='No calculated field yet'
    assert result['note']=='No results from the other material are substituted.'
    assert result['time']==2


def test_new_thermal_run_does_not_inherit_hidden_supports_from_geometry_preview(tmp_path):
    preview=execute_viewer(geometry_preview_html(job_fixture()),"yaw=.5;$('supports').checked=false;saveCamera();({supports:$('supports').checked})")
    assert not preview['stored']['supports'] and preview['stored']['mode']=='geometry'
    case=case_fixture(tmp_path/'a')
    html=thermal_workspace_html([case],live=True)
    result=execute_viewer(html,"({supports:$('supports').checked,yaw,following})",preview['stored'])['result']
    assert result==dict(supports=True,yaw=.5,following=True)


def test_intentionally_hidden_thermal_supports_have_an_explanation_instead_of_empty_axes(tmp_path):
    case=case_fixture(tmp_path/'a')
    for entry in case['manifest']['frames']:
        path=case['folder']/entry['file']
        with np.load(path) as saved:raw={key:saved[key].copy() for key in saved.files}
        raw['support_mask']=raw['part_mask'].copy()
        np.savez_compressed(path,**raw)
    html=thermal_workspace_html([case],live=True)
    saved=dict(mode='thermal',supports=False,plate=True,powder=False)
    result=execute_viewer(html,"({supports:$('supports').checked,note:$('note-0').textContent})",saved)['result']
    assert not result['supports']
    assert result['note']=='Only support-like regions are exposed at this time; enable Supports to display them.'


@pytest.mark.parametrize('state,field',[('cancelled','Cancelled · no calculated field'),
    ('failed','Failed · no calculated field'),('not_calculated','No calculated result')])
def test_terminal_missing_case_is_not_shown_as_still_calculating(tmp_path,state,field):
    case=case_fixture(tmp_path/'a')
    missing=dict(folder=None,manifest=None,label='316L',state=state,message='Recorded terminal case outcome')
    data=thermal_workspace_data([case,missing])
    assert data['cases'][1]['state']==state and data['cases'][1]['frames']==[]
    result=execute_viewer(thermal_workspace_html([case,missing]),
        "({right:$('frame-1').textContent,note:$('note-1').textContent})")['result']
    assert result['right']==field
    assert result['note']=='Recorded terminal case outcome'


def test_part_geometry_plate_and_powder_controls_draw_actual_active_context_cells(tmp_path):
    case=case_fixture(tmp_path/'a')
    for entry in case['manifest']['frames']:
        path=case['folder']/entry['file']
        with np.load(path) as saved:raw={key:saved[key].copy() for key in saved.files}
        raw['part_mask'][0,1,1]=False # Existing, unexposed powder.
        raw['part_mask'][1,0,1]=False
        raw['active_fraction'][1,0,1]=0 # Future layer material must not be drawn.
        np.savez_compressed(path,**raw)
    events=[dict(t0_s=0.,t1_s=.5,start_m=[.00025,.00025,.00025],end_m=[.0003,.00025,.00025],
                 laser_on=True,power_w=120.,kind='scan',section=0),
            dict(t0_s=3.,t1_s=5.,start_m=[.00075,.00025,.00025],end_m=[.0008,.00025,.00025],
                 laser_on=True,power_w=120.,kind='scan',section=0)]
    (case['folder']/'timeline.json').write_text(json.dumps(events))
    case['manifest'].update(timeline_file='timeline.json',timeline_event_count=2,status='running',physical_duration_s=5.)
    html=thermal_workspace_html([case])
    script="""
const original={filled:drawCalls['viewport-0'].fill||0,plate:picks[0].filter(p=>!p.line&&regionAt(scenes[0],scenes[0].frames[2],p.index)==='plate').length,paths:picks[0].filter(p=>p.line).length};
$('plate').checked=false;drawCalls['viewport-0']={};$('plate').onchange();const hidden={filled:drawCalls['viewport-0'].fill||0,plate:picks[0].filter(p=>!p.line&&regionAt(scenes[0],scenes[0].frames[2],p.index)==='plate').length,paths:picks[0].filter(p=>p.line).length};
$('powder').checked=true;drawCalls['viewport-0']={};$('powder').onchange();const powder={filled:drawCalls['viewport-0'].fill||0,count:picks[0].filter(p=>!p.line&&regionAt(scenes[0],scenes[0].frames[2],p.index)==='powder').length,allActive:picks[0].every(p=>scenes[0].frames[2].active[p.index]>0)};
view='layer';$('follow').checked=false;$('plate').checked=true;$('powder').checked=false;$('plane').value='-0.25';drawCalls['viewport-0']={};draw();const lower={filled:drawCalls['viewport-0'].fillRect||0,allPlate:picks[0].every(p=>regionAt(scenes[0],scenes[0].frames[2],p.index)==='plate')};
$('plane').value='0.25';drawCalls['viewport-0']={};draw();const upper={filled:drawCalls['viewport-0'].fillRect||0,plate:picks[0].filter(p=>!p.line&&regionAt(scenes[0],scenes[0].frames[2],p.index)==='plate').length};
({original,hidden,powder,lower,upper})
"""
    result=execute_viewer(html,script)['result']
    assert result['original']['filled']>0 and result['original']['plate']>0
    assert result['original']['paths']==1
    assert result['hidden']==dict(filled=0,plate=0,paths=1)
    assert result['powder']['filled']>0 and result['powder']['count']>0
    assert result['powder']['allActive']
    assert result['lower']['filled']>0 and result['lower']['allPlate']
    assert result['upper']==dict(filled=0,plate=0)
