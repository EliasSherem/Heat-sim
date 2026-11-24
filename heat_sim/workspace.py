"""Immutable run inputs and local worker lifecycle for the main workspace.

This module adapts the established importer, whole-build solver and preset
comparison worker. It contains no alternative thermal model or UI state.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import psutil

from .build_solver import BuildConfig, estimate_build
from .job_import import load_job
from .materials import MATERIALS, ILLUSTRATIVE_ABSORPTIVITY
from .presets import preset_config, preset_from_config
from .run_catalog import list_runs, read_json
from .run_planning import estimate_run_plan
from .storage import ROOT, RESULTS, atomic_json

ACTIVE_STATES = {"queued", "running"}


@lru_cache(maxsize=8)
def _cached_job(path, modified, size):
    return load_job(path)


def cached_job(summary_path):
    path = Path(summary_path).expanduser().resolve()
    stat = path.stat()
    return _cached_job(str(path), stat.st_mtime_ns, stat.st_size)


def _job_hash(job_or_hash):
    return str(getattr(job_or_hash, "source_sha256", job_or_hash))


def process_summary(job):
    """Compact source settings; powers/speeds remain per-vector imported data."""
    vectors = [layer.vectors for layer in job.layers if len(layer.vectors)]
    bounds = job.bounds_m
    power = [min(float(v[:, 4].min()) for v in vectors), max(float(v[:, 4].max()) for v in vectors)] if vectors else [None, None]
    speed = [min(float(v[:, 5].min()) for v in vectors)*1000, max(float(v[:, 5].max()) for v in vectors)*1000] if vectors else [None, None]
    return dict(name=job.name, filename=job.name, layer_count=len(job.layers),
                dimensions_mm=[max(0., float(bounds[i+3]-bounds[i]))*1000 for i in range(3)],
                bounds_m=list(bounds), layer_thickness_um=job.layer_thickness_m*1e6,
                vector_count=sum(len(layer.vectors) for layer in job.layers),
                power_w=power, speed_mm_s=speed,
                power_min_w=power[0], power_max_w=power[1],
                speed_min_mm_s=speed[0], speed_max_mm_s=speed[1])


def default_config(job, material="17-4PH"):
    """Common coarse full-build numerical defaults; source recipes stay at 100%."""
    if material not in MATERIALS:
        raise ValueError("Choose one of the supported material grades")
    return BuildConfig(alloy=material, substrate=material,
        absorptivity=ILLUSTRATIVE_ABSORPTIVITY[material], power_scale=1., speed_scale=1.,
        layer_end=len(job.layers), spacing_m=.002, z_spacing_m=.0015,
        plate_depth_m=.003, max_event_s=.1, frame_interval_s=30., max_frames=80,
        final_cooldown_s=.5, recoat_time_s=3.,
        support_sections=(16,) if any(np.any(layer.process_ids == 16) for layer in job.layers) else ())


def _config(config):
    value = BuildConfig(**config) if isinstance(config, dict) else config
    # BuildConfig is frozen, but its region dictionary must also be isolated.
    return BuildConfig(**copy.deepcopy(asdict(value)))


def comparison_config(config, material, absorptivity=None):
    base = _config(config)
    if material not in MATERIALS or material == base.alloy:
        raise ValueError("Choose a different supported material for comparison")
    alpha = ILLUSTRATIVE_ABSORPTIVITY[material] if absorptivity is None else float(absorptivity)
    result = replace(base, alloy=material, absorptivity=alpha)
    result.validate()
    return result


def _canonical(value):
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("Configuration inputs must be finite")
        # Unit conversion through controls must not invalidate an unchanged
        # 80µm beam because of a binary floating-point round-trip.
        return float(format(float(value), ".12g"))
    return value


def _canonical_config(config, geometry_root=None):
    data = asdict(_config(config))
    geometry = data.pop("geometry_file", None)
    if geometry:
        path = Path(geometry).expanduser()
        if not path.is_absolute() and geometry_root is not None:
            path = Path(geometry_root)/path
        if not path.is_file():
            raise ValueError("The supplemental geometry file is missing")
        data["geometry_sha256"] = sha256(path.read_bytes()).hexdigest()
    else:
        data["geometry_sha256"] = None
    return _canonical(data)


def configuration_id(job_or_hash, config, comparison_material=None, comparison_absorptivity=None):
    base = _config(config)
    value = dict(schema=1, job_source_sha256=_job_hash(job_or_hash),
                 cases=[_canonical_config(base)])
    if comparison_material:
        value["cases"].append(_canonical_config(comparison_config(base, comparison_material, comparison_absorptivity)))
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _safe_child(root, relative):
    try:
        target = (Path(root).resolve()/relative).resolve()
        return target if target.is_relative_to(Path(root).resolve()) else None
    except (TypeError, ValueError, OSError):
        return None


def _request_cases(folder, request=None, report=None):
    folder = Path(folder).resolve()
    request = request or read_json(folder/"request.json")
    report = report or read_json(folder/"report.json")
    own_manifest = read_json(folder/"manifest.json")
    values = request.get("build_config") or report.get("config") or own_manifest.get("config")
    if not values:
        return []
    base = _config(values)
    options = report.get("options") or request.get("options", {})
    kind = request.get("kind") or report.get("kind") or ("transfer" if report.get("rows") else "build")
    if kind == "presets":
        presets = [options.get("reference") or preset_from_config(base, "Reference"), *options.get("candidates", [])]
        return [(preset.get("name") or preset.get("material", "Material"), preset_config(base, preset)) for preset in presets]
    if kind == "transfer":
        baseline = options.get("baseline_material", base.alloy)
        alphas = options.get("absorptivities", {})
        result = [(baseline, replace(base, alloy=baseline, absorptivity=alphas.get(baseline, base.absorptivity), power_scale=1., speed_scale=1.))]
        for material in options.get("target_materials", report.get("target_materials", [])):
            result.append((material, replace(base, alloy=material,
                absorptivity=alphas.get(material, ILLUSTRATIVE_ABSORPTIVITY[material]),
                substrate=material if options.get("substrate_follows_material") else base.substrate,
                power_scale=1., speed_scale=1.)))
        return result
    return [(base.alloy, base)]


def _source_hash(folder, request=None):
    request = request or read_json(Path(folder)/"request.json")
    summary = read_json(Path(folder)/"job"/"job-summary.json")
    report = read_json(Path(folder)/"report.json")
    manifest = read_json(Path(folder)/"manifest.json")
    return (request.get("job_source_sha256") or summary.get("source_sha256")
            or report.get("job_source_sha256") or manifest.get("job", {}).get("source_sha256"))


def matching_result(folder, job_or_hash, config, comparison_material=None, comparison_absorptivity=None):
    """Match frozen inputs and any actual case manifests, never editor labels."""
    root = Path(folder).resolve()
    request = read_json(root/"request.json")
    if _source_hash(root, request) != _job_hash(job_or_hash):
        return False
    summary_hash = read_json(root/"job"/"job-summary.json").get("source_sha256")
    if summary_hash and summary_hash != _job_hash(job_or_hash):
        return False
    try:
        desired = [_canonical_config(config)]
        if comparison_material:
            desired.append(_canonical_config(comparison_config(config, comparison_material, comparison_absorptivity)))
        saved = _request_cases(root, request)
        if len(saved) != len(desired) or [_canonical_config(cfg, root) for _, cfg in saved] != desired:
            return False
        # A saved request can precede first-frame publication. When artifacts
        # exist, independently ensure they contain the claimed source/settings.
        for path in [root/"manifest.json", *root.glob("cases/*/manifest.json")]:
            manifest = read_json(path)
            if not manifest:
                continue
            source = manifest.get("job", {}).get("source_sha256")
            if source and source != _job_hash(job_or_hash):
                return False
            if _canonical_config(manifest["config"], root) not in desired:
                return False
        return True
    except (TypeError, ValueError, KeyError, OSError):
        return False


def _published_frames(folder, manifest):
    entries = []
    previous = -math.inf
    for entry in manifest.get("frames", []):
        path = _safe_child(folder, entry.get("file"))
        try:
            timestamp = float(entry["time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if path is not None and path.is_file() and math.isfinite(timestamp) and timestamp > previous:
            entries.append(copy.deepcopy(entry)); previous = timestamp
    return entries


def _case_failure(root, folder, label, cfg, key, failures):
    """Associate a failure with its actual frozen case, not another alloy run."""
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        if folder is not None and _safe_child(root, failure.get("case_folder")) == folder:
            return failure
        if failure.get("config"):
            try:
                if _canonical_config(failure["config"], root) == key:
                    return failure
            except (TypeError, ValueError, KeyError, OSError):
                continue
            # A full recorded config is stronger evidence than its label.
            continue
        if failure.get("preset_name", failure.get("name")) == label:
            return failure
        if (failure.get("material") == cfg.alloy
                and all(_canonical(failure[field]) == _canonical(getattr(cfg, field))
                        for field in ("power_scale", "speed_scale", "absorptivity") if field in failure)):
            return failure
    return None


def _case_outcome(manifest, overall, overall_message, failure):
    """A finished worker cannot leave unstarted panels labelled Calculating."""
    recorded = manifest.get("status", "unknown")
    available = bool(manifest.get("frames"))
    message = str(manifest.get("error") or "")
    if failure is not None:
        return "failed", str(failure.get("error") or message or "This case failed and is excluded from comparison.")
    if recorded == "complete" and available:
        return recorded, message
    if recorded in ("failed", "cancelled"):
        return recorded, message or ("This case was cancelled." if recorded == "cancelled" else "This case failed.")
    if overall == "cancelled":
        return "cancelled", ("This case was cancelled; its calculated fields are partial." if available else
                             "This case was not calculated because the simulation was cancelled.")
    if overall == "failed":
        explanation = "This case stopped before completion." if available else "This case was not calculated before the simulation failed."
        return ("failed" if available else "not_calculated"), explanation + (" " + overall_message if overall_message else "")
    if overall in ("complete", "partial"):
        return ("incomplete" if available else "not_calculated"), (
            "This run finished before the case recorded completion; available fields are partial." if available else
            "This run finished without calculated fields for this case.")
    if recorded == "complete" and not available:
        return "not_calculated", "No calculated fields are available for this saved case."
    return recorded, message


def read_view_cases(folder, include_pending=True):
    """Frozen case slots remain stable while a second material is preparing.

    Empty frame lists are explicit placeholders, not reused temperatures from
    an earlier case. The renderer can therefore show an uncalculated second
    panel until its own worker fields become available.
    """
    root = Path(folder).resolve()
    request = read_json(root/"request.json"); report = read_json(root/"report.json")
    status = read_json(root/"status.json")
    overall = status.get("state") or report.get("status")
    overall_message = str(status.get("message") or report.get("error") or "")
    try:
        expected = _request_cases(root, request, report)
    except (TypeError, ValueError, KeyError):
        return []
    actual = []
    for path in [root/"manifest.json", *root.glob("cases/*/manifest.json")]:
        if not path.resolve().is_relative_to(root):
            continue
        manifest = read_json(path)
        if manifest.get("config"):
            try:
                key = _canonical_config(manifest["config"], root)
            except (TypeError, ValueError, KeyError, OSError):
                continue
            actual.append((path.parent, manifest, key, path.stat().st_mtime_ns))
    cases = []
    job = read_json(root/"job"/"job-summary.json")
    for index, (label, cfg) in enumerate(expected):
        try:
            key = _canonical_config(cfg, root)
        except (TypeError, ValueError, OSError):
            continue
        matches = [item for item in actual if item[2] == key]
        if matches:
            case_folder, manifest, _, _ = max(matches, key=lambda item: item[3])
            manifest = copy.deepcopy(manifest)
            manifest["frames"] = _published_frames(case_folder, manifest)
        else:
            case_folder = root/"pending"/str(index)
            manifest = dict(status="queued", config=asdict(cfg), frames=[],
                job=dict(name=job.get("name", request.get("job_name", "Imported print")),
                         source_sha256=_source_hash(root, request)))
        failure = _case_failure(root, case_folder if matches else None, label, cfg, key, report.get("failures", []))
        state, message = _case_outcome(manifest, overall, overall_message, failure)
        if state != manifest.get("status"):
            manifest["recorded_status"] = manifest.get("status", "unknown")
            manifest["status"] = state
        if message and state != "complete":
            manifest["error"] = message
        if include_pending or manifest["frames"]:
            cases.append(dict(folder=case_folder, manifest=manifest, label=label,
                material=cfg.alloy, state=state, message=message, error=message if state != "complete" else "",
                frame_count=len(manifest["frames"]), available=bool(manifest["frames"]), index=index))
    return cases


def _worker_alive(folder, status):
    launch = read_json(Path(folder)/"launch.json")
    pid = status.get("pid") or launch.get("pid")
    expected = status.get("process_started_epoch_s") or launch.get("process_started_epoch_s")
    if not pid:
        return None
    try:
        process = psutil.Process(int(pid))
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
        if expected is not None:
            return abs(process.create_time()-float(expected)) < .01
        # Legacy PID-only records must identify this actual worker invocation.
        command = process.cmdline()
        return "heat_sim.worker" in command and str((Path(folder)/"request.json").resolve()) in command
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
        return False


def read_status(folder, reconcile=True):
    root = Path(folder).resolve()
    status = read_json(root/"status.json")
    if status.get("state") not in ACTIVE_STATES or not reconcile:
        return status
    alive = _worker_alive(root, status)
    if alive is None:
        try:
            # A request has a brief launch window before PID registration.
            alive = time.time()-(root/"request.json").stat().st_mtime < 30.
        except OSError:
            alive = False
    if alive:
        return status
    fresh = read_json(root/"status.json")
    if fresh.get("state") not in ACTIVE_STATES:
        return fresh
    failure = {**fresh, "state": "failed", "phase": "finished",
        "message": "The calculation process exited before reporting its result. See worker.log in Diagnostics.",
        "finished_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(root/"status.json", failure)
    return failure


def active_run(results_root=None, job_hash=None):
    """Locate a live worker, excluding PID reuse and old orphan status files."""
    for info in list_runs(Path(results_root or RESULTS)/"builds", job_hash):
        if info.state not in ACTIVE_STATES:
            continue
        status = read_status(info.folder)
        if status.get("state") in ACTIVE_STATES:
            return info.folder, status
    return None, None


def cancel_run(folder):
    root = Path(folder).resolve()
    status = read_status(root)
    if status.get("state") not in ACTIVE_STATES:
        return False
    atomic_json(root/"CANCEL", dict(run_id=root.name,
        requested_utc=datetime.now(timezone.utc).isoformat(), source="simulation_workspace", action="Cancel simulation"))
    return True


def start_run(summary_path, config, comparison_material=None, comparison_absorptivity=None,
              provenance=None, results_root=None):
    """Freeze one main-workspace request and launch the established CPU worker."""
    summary = Path(summary_path).expanduser().resolve()
    results_root = Path(results_root or RESULTS)
    job = cached_job(summary)
    base = _config(config)
    configs = [base]
    if comparison_material:
        configs.append(comparison_config(base, comparison_material, comparison_absorptivity))
    for cfg in configs:
        cfg.validate(job)
        estimate = estimate_build(job, cfg)
        if estimate["cells"] > 2_000_000 or estimate["max_saved_frame_bytes"] > 8*1024**3:
            raise ValueError("This grid/frame selection exceeds the whole-build memory guard. Choose coarser cells or a shorter range in Advanced.")
        plan = estimate_run_plan(job, cfg)
        if plan["step_limit_exceeded"]:
            raise ValueError(plan["reason"])
    live, _ = active_run(results_root)
    if live:
        raise RuntimeError("A simulation is already running. Wait for it to finish or cancel it before starting another.")
    identity = configuration_id(job, base, comparison_material, comparison_absorptivity)
    name = datetime.now(ZoneInfo("Asia/Hebron")).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]
    folder = Path(results_root).resolve()/"builds"/name
    folder.mkdir(parents=True)
    shutil.copytree(summary.parent, folder/"job")
    if base.geometry_file:
        source = Path(base.geometry_file).expanduser().resolve()
        shutil.copy2(source, folder/"supplement.stl")
        base = replace(base, geometry_file="supplement.stl")
    options = {}
    kind = "build"
    if comparison_material:
        candidate = comparison_config(base, comparison_material, comparison_absorptivity)
        options = dict(study_name=f"{base.alloy} / {candidate.alloy}",
            reference=preset_from_config(base, base.alloy, "workspace-primary"),
            candidates=[preset_from_config(candidate, candidate.alloy, "workspace-comparison")])
        kind = "presets"
    request = dict(kind=kind, job_path="job/job-summary.json", build_config=asdict(base),
        options=options, input_provenance=copy.deepcopy(provenance or {}),
        created_utc=datetime.now(timezone.utc).isoformat(), job_name=job.name,
        job_source_sha256=job.source_sha256, workspace_configuration_id=identity,
        workspace_schema=1)
    atomic_json(folder/"request.json", request)
    atomic_json(folder/"status.json", dict(state="queued", progress=0., message="Starting local thermal calculation"))
    try:
        with (folder/"worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "heat_sim.worker", str(folder/"request.json")],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(folder/"launch.json", dict(pid=process.pid,
            process_started_epoch_s=psutil.Process(process.pid).create_time()))
    except (OSError, psutil.Error) as exc:
        fresh = read_json(folder/"status.json")
        # A tiny worker may finish before launch metadata is written.
        if fresh.get("state") not in ("complete", "partial", "failed", "cancelled"):
            atomic_json(folder/"status.json", dict(state="failed", progress=0., message=f"Could not start calculation: {exc}"))
        if not (folder/"manifest.json").exists() and not (folder/"report.json").exists():
            raise RuntimeError(f"Could not start the local calculation: {exc}") from exc
    return folder
