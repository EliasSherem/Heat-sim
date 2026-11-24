"""Read the supplied printer job as inert reference data; never execute its pickle.

The vendor's ZIP contains a `para.head` primitive Python pickle. We accept only
scalar/list/dict opcodes, reject object construction, and never extract paths.
This module has no printer communication or control capability.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
from zipfile import ZipFile

_ALLOWED = {
    'PROTO', 'FRAME', 'STOP', 'MARK', 'POP', 'POP_MARK', 'DUP',
    'NONE', 'NEWTRUE', 'NEWFALSE', 'INT', 'BININT', 'BININT1', 'BININT2',
    'LONG', 'LONG1', 'LONG4', 'FLOAT', 'BINFLOAT',
    'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8',
    'EMPTY_LIST', 'LIST', 'APPEND', 'APPENDS', 'EMPTY_DICT', 'DICT',
    'SETITEM', 'SETITEMS', 'EMPTY_TUPLE', 'TUPLE', 'TUPLE1', 'TUPLE2', 'TUPLE3',
    'PUT', 'BINPUT', 'LONG_BINPUT', 'MEMOIZE', 'GET', 'BINGET', 'LONG_BINGET',
}


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Object references are not accepted in job metadata.')

    def persistent_load(self, pid):
        raise ValueError('Persistent references are not accepted in job metadata.')


def load_job_metadata(path: str | Path) -> dict:
    """Return data from a bounded header, rejecting executable pickle opcodes."""
    with ZipFile(path) as archive:
        header = archive.getinfo('para.head')
        if header.file_size > 1_000_000:
            raise ValueError('Job parameter header exceeds 1 MB.')
        data = archive.read(header)
    for opcode, _, _ in pickletools.genops(data):
        if opcode.name not in _ALLOWED:
            raise ValueError(f'Non-data pickle opcode rejected: {opcode.name}')
    result = _DataOnlyUnpickler(io.BytesIO(data)).load()
    if not isinstance(result, dict):
        raise ValueError('Expected a dictionary of job parameters.')
    return result


def summarize_job(path: str | Path) -> dict:
    """Extract preset references, retaining unit and vendor-semantics uncertainty."""
    path = Path(path)
    data = load_job_metadata(path)
    part = next(iter(data['part'].values()))
    prop = part['property']
    rows = []
    def add(label, obj, source, active=True, note=''):
        rows.append({'preset': label, 'power_assumed_w': obj.get('power'),
                     'speed_assumed_mm_s': obj.get('speed'), 'active': active,
                     'source_path': source, 'note': note})
    entity = prop['entity']
    add('Main body infill', entity['volume'], 'part.0.property.entity.volume')
    add('Top-surface fill', prop['up']['volume'], 'part.0.property.up.volume')
    add('Down-facing fill', prop['down']['volume'], 'part.0.property.down.volume')
    contour = prop['contour']
    add('Main outer contour', contour['entity']['outer'], 'part.0.property.contour.entity.outer')
    add('Separate top outer contour', contour['up']['outer'], 'part.0.property.contour.up.outer',
        note='keep_same=true may override this separate setting')
    add('Additional scan section 16', part['section']['16'], 'part.0.section.16',
        note='Purpose is uncertain; possibly supports. Section presence alone does not prove exposure.')
    add('Core preset', entity['core'], 'part.0.property.entity.core',
        active=entity['core'].get('enabled'), note='DISABLED in source file')
    add('Special sharp contour', contour['entity']['sharp'], 'part.0.property.contour.entity.sharp',
        active=contour['entity']['sharp'].get('enabled', True))
    return {
        'source_file': path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'interpretation': 'Power provisionally interpreted as watts; speed as mm/s; distances as mm. Verify with vendor.',
        'layer_thickness_assumed_mm': data['thickness_real'],
        'hatch_spacing_assumed_mm': entity['volume']['distance'],
        'raduis_raw': entity['volume']['raduis'],
        'raduis_interpretation': 'Unknown vendor field. Never used as a beam radius or diameter.',
        'contour_keep_same': contour.get('keep_same'),
        'presets': rows,
        'raw_parameter_header': data,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--output', type=Path, default=Path('examples/job_reference.json'))
    args = parser.parse_args()
    summary = summarize_job(args.job)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(args.output)

if __name__ == '__main__':
    main()
