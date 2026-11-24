import io
import json
import pickle
from pathlib import Path
from zipfile import ZipFile

import pytest

from heat_sim.job_reference import load_job_metadata


def test_reference_preserves_unknown_and_disabled():
    data = json.loads((Path(__file__).parents[1] / 'examples/job_reference.json').read_text())
    assert data['layer_thickness_assumed_mm'] == 0.06
    assert data['hatch_spacing_assumed_mm'] == 0.06
    assert data['raduis_raw'] == 0.12
    assert data['contour_keep_same'] is True
    core = next(r for r in data['presets'] if r['preset'] == 'Core preset')
    assert core['active'] is False
    assert core['power_assumed_w'] == 380


def test_data_only_job_header(tmp_path):
    path = tmp_path / 'safe.htmjob'
    with ZipFile(path, 'w') as z:
        z.writestr('para.head', pickle.dumps({'power': 120, 'enabled': False}))
    assert load_job_metadata(path) == {'power': 120, 'enabled': False}


def test_executable_pickle_rejected(tmp_path):
    path = tmp_path / 'unsafe.htmjob'
    with ZipFile(path, 'w') as z:
        z.writestr('para.head', b'cos\nsystem\n(S"touch NEVER_EXECUTE"\ntR.')
    with pytest.raises(ValueError, match='Non-data pickle opcode'):
        load_job_metadata(path)
    assert not (tmp_path / 'NEVER_EXECUTE').exists()
