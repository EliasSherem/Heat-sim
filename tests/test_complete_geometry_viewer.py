"""Execute the real complete-geometry renderer against saved thermal fields."""
import json

import numpy as np

from heat_sim.viewer_geometry_transport import pack_path_geometry
from heat_sim.workspace_viewer import thermal_workspace_html
from test_workspace_viewer import case_fixture, execute_viewer


def complete_scene_script():
    """Disconnected, tiny supports must survive; the last scan is in the future."""
    times = np.array([.2, .3, .4, 1.5, 2.7, 3., 4.])
    starts = np.array([[.25, .25, .03], [.75, .25, .06], [.75, .3, .06],
                       [.25, .75, .24], [.75, .75, .42], [.25, .25, .49],
                       [.25, .25, .6]])
    ends = starts + [0.0002, 0., 0.]
    cells = np.array([1, 3, 3, 5, 7, 1, 1])
    support = np.array([False, True, True, False, True, False, False])
    packed = pack_path_geometry(times, starts, ends, cells, support)
    return r'''
this.Path2D=class {constructor(){this.lines=[]}moveTo(x,y){this.start=[x,y]}lineTo(x,y){this.lines.push([...this.start,x,y])}};
const s=scenes[0];s.paths=[];s.path_geometry={count:7,record_bytes:40,format:'tracklab-paths-v1'};
s.geometryData=new DataView(Uint8Array.from(PACKED).buffer);
$('representation').value='part';$('representation').disabled=false;$('plate').checked=false;
const ctx=$('viewport-0').getContext('2d'),strokes=[],composites=[],events=[];
function captureStroke(context,supportLayer){context.stroke=function(shape){if(shape&&shape.lines){const stroke={colour:context.strokeStyle,count:shape.lines.length,alpha:context.globalAlpha,supportLayer};strokes.push(stroke);events.push({type:'stroke',supportLayer})}}}
captureStroke(ctx,false);
ctx.drawImage=function(){composites.push({alpha:ctx.globalAlpha});events.push({type:'composite'})};
const createElement=document.createElement;
document.createElement=function(tag){const element=createElement(tag);if(tag==='canvas'){const context=element.getContext('2d');context.globalAlpha=1;captureStroke(context,true);element.getContext=()=>context}return element};
function renderField(k){picks[0]=[];strokes.length=0;composites.length=0;events.length=0;drawCompleteGeometry(s,0,s.frames[k],ctx,1000,440);return{count:picks[0].reduce((sum,p)=>sum+(p.geometry?p.geometry.count:0),0),groups:picks[0].filter(p=>p.geometry).map(p=>({cell:p.index,support:p.geometry.support,first:p.geometry.firstFrame,count:p.geometry.count})),strokes:strokes.slice(),composites:composites.slice(),events:events.slice()}}
'''.replace("PACKED", json.dumps(list(packed)))


def test_complete_geometry_keeps_isolated_supports_and_first_and_final_print_paths(tmp_path):
    case = case_fixture(tmp_path / 'part', times=(0., 1., 2., 3.))
    html = thermal_workspace_html([case])
    result = execute_viewer(html, complete_scene_script() + r'''
const batches=buildGeometryBatches(s,1000,440);
const firstPoint=project([.25,.25,.03],1000,440),lastPoint=project([.25,.25,.49],1000,440);
({hasPaths:hasPaths(s),available:batches.reduce((n,g)=>n+g.count,0),final:renderField(3),
first:batches.some(g=>g.firstFrame===1&&geometryHit(firstPoint,g)),
last:batches.some(g=>g.firstFrame===3&&!g.support&&geometryHit(lastPoint,g)),
allFinite:batches.every(g=>Array.from(g.lines).every(Number.isFinite))})
''')['result']
    assert result['hasPaths'] and result['first'] and result['last'] and result['allFinite']
    assert result['available'] == 6  # No arbitrary budget removes tiny disconnected lines.
    assert result['final']['count'] == 6
    assert sum(group['count'] for group in result['final']['groups'] if group['support']) == 3


def test_complete_geometry_never_reveals_future_scans_and_holds_previous_saved_frame(tmp_path):
    case = case_fixture(tmp_path / 'part', times=(0., 1., 2., 3.))
    result = execute_viewer(thermal_workspace_html([case]), complete_scene_script() + r'''
({counts:[0,1,2,3].map(k=>renderField(k).count),between:renderField(frameIndex(s,1.9)).count,
firstFrames:buildGeometryBatches(s,1000,440).map(g=>g.firstFrame)})
''')['result']
    assert result['counts'] == [0, 3, 4, 6]
    assert result['between'] == 3
    assert set(result['firstFrames']) == {1, 2, 3}


def test_support_toggle_only_hides_support_geometry_and_exact_cell_temperatures_drive_colours(tmp_path):
    case = case_fixture(tmp_path / 'part', times=(0., 1., 2., 3.))
    result = execute_viewer(thermal_workspace_html([case]), complete_scene_script() + r'''
const before=JSON.stringify(s.frames.map(f=>f.temperature)),on=renderField(3);
$('supports').checked=false;const off=renderField(3);
$('supports').checked=true;const restored=renderField(3);
({on,off,restored,unchanged:before===JSON.stringify(s.frames.map(f=>f.temperature)),
expected:on.groups.map(g=>color(s.frames[3].temperature[g.cell])),
point:s.frames[3].temperature[3],readout:pointTemperature(s,s.frames[3],3)})
''')['result']
    assert result['on']['count'] == result['restored']['count'] == 6
    assert result['off']['count'] == 3
    assert not any(group['support'] for group in result['off']['groups'])
    assert [stroke['colour'] for stroke in result['on']['strokes']] == result['expected']
    assert result['readout'] == result['point'] and result['unchanged']
    assert result['on']['composites'] == [dict(alpha=.35)]
    assert result['off']['composites'] == []
    assert all(stroke['alpha'] == 1 for stroke in result['on']['strokes'])


def test_support_opacity_does_not_accumulate_when_source_paths_are_dense(tmp_path):
    case = case_fixture(tmp_path / 'part', times=(0., 1., 2., 3.))
    result = execute_viewer(thermal_workspace_html([case]), complete_scene_script() + r'''
const sparse=renderField(3),original=new Uint8Array(s.geometryData.buffer),dense=new Uint8Array(original.length+300*40);
dense.set(original);for(let i=0;i<300;i++)dense.set(original.slice(40,80),original.length+i*40);
s.geometryData=new DataView(dense.buffer);s.path_geometry.count+=300;s.geometryCache=null;
const built=renderField(3);
({sparse,built,bodyBefore:sparse.strokes.filter(s=>!s.supportLayer),bodyAfter:built.strokes.filter(s=>!s.supportLayer)})
''')['result']
    assert result['built']['count'] == result['sparse']['count'] + 300
    assert result['sparse']['composites'] == result['built']['composites'] == [dict(alpha=.35)]
    assert result['bodyBefore'] == result['bodyAfter']
    events = result['built']['events']
    composite = next(i for i, event in enumerate(events) if event['type'] == 'composite')
    assert all(event == dict(type='stroke', supportLayer=True) for event in events[:composite])
    assert all(event == dict(type='stroke', supportLayer=False) for event in events[composite + 1:])
    assert events[:composite] and events[composite + 1:]  # Both supports and body were drawn.


def test_geometry_batches_reuse_camera_projection_and_rebuild_after_camera_change(tmp_path):
    case = case_fixture(tmp_path / 'part', times=(0., 1., 2., 3.))
    result = execute_viewer(thermal_workspace_html([case]), complete_scene_script() + r'''
const first=buildGeometryBatches(s,1000,440),same=buildGeometryBatches(s,1000,440);
const before=Array.from(first[0].lines);yaw+=.3;const changed=buildGeometryBatches(s,1000,440);
({same:first===same,changed:first!==changed,coordinatesChanged:JSON.stringify(before)!==JSON.stringify(Array.from(changed[0].lines)),
count:changed.reduce((n,g)=>n+g.count,0)})
''')['result']
    assert result == dict(same=True, changed=True, coordinatesChanged=True, count=6)
