"""Local, content-addressed simulation storage. No printer or network interfaces."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import io
import json
import os
import zipfile
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, default=_json_default))
    os.replace(temp, path)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def model_fingerprint():
    digest = sha256()
    for name in ("solver.py", "materials.py", "storage.py", "workflows.py"):
        path = ROOT / "heat_sim" / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_key(config):
    return sha256((model_fingerprint() + json.dumps(asdict(config), sort_keys=True)).encode()).hexdigest()[:24]


def save_simulation(result, key):
    folder = RESULTS / "cache" / key
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {name: value for name, value in vars(result).items() if isinstance(value, np.ndarray)}
    np.savez_compressed(folder / "fields.npz", **arrays)
    atomic_json(folder / "result.json", {"config": asdict(result.config), "metrics": result.metrics,
                                         "fingerprint": model_fingerprint(), "cache_key": key})
    return folder


def load_simulation(key):
    folder = RESULTS / "cache" / key
    meta = json.loads((folder / "result.json").read_text())
    with np.load(folder / "fields.npz", allow_pickle=False) as fields:
        arrays = {name: fields[name].copy() for name in fields.files}
    return meta, arrays


def snapshot_source(folder):
    """Freeze source, data conventions and dependencies before executing a job."""
    for path in (ROOT / "heat_sim").glob("*.py"):
        target = folder / "source" / "heat_sim" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ("requirements-lock.txt", "requirements.txt", "app.py"):
        path = ROOT / name
        if path.exists():
            shutil.copy2(path, folder / name)
    for path in (ROOT / "docs").glob("*.md"):
        target = folder / "docs" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)


def bundle_run(folder: Path):
    """Include inputs, source provenance, tables, fields and static figures."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("REPRODUCE.txt", "Use Python3.13 on macOS. From this extracted directory:\ncd source\npython3.13 -m venv .venv\n.venv/bin/python -m pip install -r ../run/requirements-lock.txt\n.venv/bin/python -m heat_sim.worker ../run/request.json\n\nThis recomputes the saved workflow using the frozen numerical source. Finished fields are under source/results/cache/. The run remains illustrative, not experimentally validated. Python3.13 interpreter naming can differ by machine.\n")
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix != ".log":
                archive.write(path, str(Path("run") / path.relative_to(folder)))
        report_path = folder / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            for key in sorted({row.get("cache_key") for row in report.get("rows", [])} - {None}):
                for path in (RESULTS / "cache" / key).glob("*"):
                    archive.write(path, f"fields/{key}/{path.name}")
        for name in ("MODEL.md", "MATERIALS.md", "VERIFICATION.md"):
            path = ROOT / "docs" / name
            if path.exists() and not (folder / "docs" / name).exists():
                archive.write(path, f"docs/{name}")
        for name in ("requirements.txt", "requirements-lock.txt"):
            path = ROOT / name
            if path.exists() and not (folder / name).exists():
                archive.write(path, name)
        if not (folder / "source").exists():
            archive.writestr("SOURCE-PROVENANCE.txt", "This earlier run predates source snapshots. Its cache metadata records a source fingerprint; current source may differ. Regenerate the run for an exact source snapshot.")
    return buffer.getvalue()
