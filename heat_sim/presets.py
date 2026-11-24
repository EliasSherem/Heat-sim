"""Named print presets and real, reproducible thermal preset comparisons.

Presets alter physical process inputs on unchanged imported paths. Numerical
resolution and geometry remain common to all candidates in a study. Rankings
are closeness to a selected reference, never a qualified process optimum.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import uuid

from .build_solver import BuildConfig, run_build
from .materials import ILLUSTRATIVE_ABSORPTIVITY
from .run_planning import estimate_run_plan
from .solver import SimulationCancelled, SimulationError
from .storage import atomic_json
from .transfer import (SCORE_FEATURES, _case_features, _code_fingerprint,
                       _external_geometry_fingerprint, _job_fingerprint,
                       _write_csv, thermal_similarity_score)


PRESET_FIELDS = ("material", "substrate", "power_scale", "speed_scale", "initial_k",
                 "recoat_time_s", "beam_diameter_m", "absorptivity",
                 "powder_packing", "powder_k_w_mk", "jump_speed_m_s", "jump_delay_s",
                 "section_overrides")
NUMERIC_PRESET_FIELDS = tuple(k for k in PRESET_FIELDS if k not in ("material", "substrate", "section_overrides"))


def preset_from_config(config, name="Reference", preset_id=None):
    """Return an editable, complete physical-input preset from a build config."""
    base = BuildConfig(**config) if isinstance(config, dict) else config
    value = {k: getattr(base, "alloy" if k == "material" else k, {} if k == "section_overrides" else None) for k in PRESET_FIELDS}
    value["section_overrides"] = json.loads(json.dumps(value["section_overrides"]))
    if value["absorptivity"] is None:
        value["absorptivity"] = ILLUSTRATIVE_ABSORPTIVITY[value["material"]]
    return {"id": preset_id or uuid.uuid4().hex[:16], "name": name, **value}


def normalize_preset(preset, base_config=None):
    """Validate a preset; geometry/numerical overrides are deliberately rejected."""
    if not isinstance(preset, dict):
        raise ValueError("A preset must be an object")
    allowed = {*PRESET_FIELDS, "id", "name", "saved_utc"}
    unknown = set(preset) - allowed
    if unknown:
        raise ValueError("Unsupported preset fields: " + ", ".join(sorted(unknown)))
    base = base_config or BuildConfig()
    if isinstance(base, dict):
        base = BuildConfig(**base)
    name = str(preset.get("name", "")).strip()
    if not name or len(name) > 80:
        raise ValueError("Give each preset a name of 1–80 characters")
    result = preset_from_config(base, name, str(preset.get("id") or uuid.uuid4().hex[:16]))
    result.update({k: preset[k] for k in PRESET_FIELDS if k in preset})
    # Saved/reference/editing snapshots must not share a mutable region table.
    try:
        result["section_overrides"] = json.loads(json.dumps(result["section_overrides"], allow_nan=False))
    except (ValueError, TypeError):
        raise ValueError("Region overrides must contain finite JSON-compatible settings") from None
    # An alloy change needs its own independent optical assumption, never the
    # reference alloy's value by accidental inheritance.
    if result["material"] != base.alloy and "absorptivity" not in preset:
        result["absorptivity"] = ILLUSTRATIVE_ABSORPTIVITY.get(result["material"])
    for key in NUMERIC_PRESET_FIELDS:
        try:
            result[key] = float(result[key])
        except (ValueError, TypeError):
            raise ValueError(f"{key} must be a finite number") from None
        if not math.isfinite(result[key]):
            raise ValueError(f"{key} must be a finite number")
    cfg = replace(base, alloy=result["material"], **{k: result[k] for k in PRESET_FIELDS if k != "material"})
    cfg.validate()
    if preset.get("saved_utc"):
        result["saved_utc"] = str(preset["saved_utc"])
    return result


def preset_config(base_config, preset):
    base = BuildConfig(**base_config) if isinstance(base_config, dict) else base_config
    value = normalize_preset(preset, base)
    return replace(base, alloy=value["material"], **{k: value[k] for k in PRESET_FIELDS if k != "material"})


def _library_path(root, job_sha256):
    # Hash even caller-supplied nonhex job ids, rather than interpolate a path.
    digest = sha256(str(job_sha256).encode()).hexdigest()
    return Path(root) / (digest + ".json")


def load_presets(root, job_sha256):
    path = _library_path(root, job_sha256)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("presets", []), list):
            raise ValueError("Saved preset library has an invalid structure")
        if data.get("job_source_sha256") != str(job_sha256):
            raise ValueError("Preset library belongs to a different imported job")
        return [normalize_preset(item) for item in data.get("presets", [])]
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read saved preset library: {exc}") from exc


def save_preset(root, job_sha256, preset, base_config=None):
    """Create/update one named preset. Return the saved complete snapshot."""
    value = normalize_preset(preset, base_config)
    value["saved_utc"] = datetime.now(timezone.utc).isoformat()
    rows = load_presets(root, job_sha256)
    if any(row["id"] != value["id"] and row["name"].casefold() == value["name"].casefold() for row in rows):
        raise ValueError("A saved preset already has this name. Choose it to update or use a new name.")
    rows = [row for row in rows if row["id"] != value["id"]]
    rows.append(value)
    atomic_json(_library_path(root, job_sha256), dict(schema=1, job_source_sha256=str(job_sha256), presets=rows))
    return value


def delete_preset(root, job_sha256, preset_id):
    rows = load_presets(root, job_sha256)
    atomic_json(_library_path(root, job_sha256), dict(schema=1, job_source_sha256=str(job_sha256),
        presets=[row for row in rows if row["id"] != preset_id]))


def plan_presets(job, base_config, options):
    """Validate every physical preset and return its independent step preflight."""
    reference = options.get("reference") or preset_from_config(base_config, "Reference")
    candidates = options.get("candidates", [])
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
        raise ValueError("Choose between 1 and 12 candidate presets")
    values = [normalize_preset(reference, base_config), *[normalize_preset(p, base_config) for p in candidates]]
    if len({v["id"] for v in values}) != len(values):
        raise ValueError("Each reference/candidate preset needs a distinct id; duplicate a preset before editing it")
    if len({v["name"].casefold() for v in values}) != len(values):
        raise ValueError("Use a different name for each reference and candidate preset")
    return [dict(preset=value, config=asdict(preset_config(base_config, value)),
                 plan=estimate_run_plan(job, preset_config(base_config, value))) for value in values]


def run_presets(job, base_config, options, folder, progress=None, cancel=None):
    """Compute a fixed-path reference plus named physical-setting candidates.

    options: ``reference`` preset, ``candidates`` list. The returned report has
    the same case/row shape as a transfer report for playback and diagnostics.
    Each score uses its own preheat-to-liquidus enthalpy normalization, so a
    changed preheat compares normalized thermal response rather than absolute
    temperature or a melt fraction.
    """
    started = time.perf_counter()
    base = BuildConfig(**base_config) if isinstance(base_config, dict) else base_config
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda f, message: None)
    cancel = cancel or (lambda: False)
    plans = plan_presets(job, base, options)
    reference = plans[0]["preset"]
    fingerprint = sha256((_code_fingerprint() + Path(__file__).read_text()).encode()).hexdigest()
    job_hash = _job_fingerprint(job)
    geometry_hash = _external_geometry_fingerprint(base)
    study_name = str(options.get("study_name") or "Preset comparison").strip()[:120]
    report = dict(schema=1, kind="presets", study_name=study_name, status="running", baseline_material=reference["material"],
        reference_name=reference["name"], reference_preset=reference,
        target_materials=list(dict.fromkeys(p["preset"]["material"] for p in plans[1:])),
        rows=[], failures=[], rankings=[], uncertainty={}, cache_hits=0,
        config=asdict(base), options={"study_name": study_name, "reference": reference, "candidates": [p["preset"] for p in plans[1:]]},
        job_source_sha256=job.source_sha256, job_fingerprint=job_hash, model_fingerprint=fingerprint,
        supplemental_geometry_sha256=geometry_hash, numerical_resolution_assessed=False,
        experimental_validation=False, microscopic_geometry_resolved=False,
        score_definition="RMS relative differences of mean,95th-percentile,max peak-field h-fractions; denominator floor0.02; lower is closer to the selected reference",
        normalization="Each case uses its own material and initial temperature to normalize excess enthalpy by its preheat-to-liquidus interval. This is not absolute temperature, liquid fraction or a defect metric.",
        warnings=["The closest sampled preset is a thermal screening candidate, not an optimal or qualified printing recipe.",
                  "Imported geometry, layer/vector bounds and numerical resolution are held fixed; enabled region overrides replace imported settings before global multipliers apply.",
                  "Small score differences have no established experimental or numerical significance without refinement and measurements.",
                  "Changing preheat changes both thermal physics and enthalpy normalization. Review absolute temperatures and settings alongside the score.",
                  "Hatch spacing, layer thickness and scan paths require a newly sliced/imported job to compare faithfully."])
    atomic_json(folder / "preset-request.json", dict(config=asdict(base), options=report["options"]))
    baseline_features = None

    def persist():
        atomic_json(folder / "report.json", report)
        _write_csv(folder / "comparison.csv", [{k: v for k, v in row.items() if k != "metrics"} for row in report["rows"]])
        atomic_json(folder / "failures.json", report["failures"])

    persist()
    try:
        for slot, plan in enumerate(plans):
            if cancel():
                raise SimulationCancelled("Preset comparison cancelled between cases; completed cases retained")
            preset = plan["preset"]
            cfg = BuildConfig(**plan["config"])
            key = sha256((fingerprint + job_hash + geometry_hash + json.dumps(asdict(cfg), sort_keys=True)).encode()).hexdigest()[:24]
            case = folder / "cases" / key
            identity = dict(name=preset["name"], preset_name=preset["name"], preset_id=preset["id"],
                material=cfg.alloy, kind="baseline" if slot == 0 else "candidate", scenario="nominal",
                power_scale=cfg.power_scale, speed_scale=cfg.speed_scale, absorptivity=cfg.absorptivity,
                config=asdict(cfg), preset=preset, case_folder=case.relative_to(folder).as_posix())
            try:
                if plan["plan"]["step_limit_exceeded"]:
                    raise SimulationError(plan["plan"]["reason"])
                path = case / "manifest.json"
                manifest = json.loads(path.read_text()) if path.exists() else None
                label = f"{preset['name']} · {cfg.alloy}"
                if manifest and manifest.get("status") == "complete":
                    report["cache_hits"] += 1
                    progress(slot / len(plans), "Cached · " + label)
                else:
                    manifest = run_build(job, cfg, case,
                        progress=lambda f, message: progress(min(.999, (slot + f) / len(plans)), label + " · " + message), cancel=cancel)
                if manifest.get("status") != "complete":
                    raise SimulationError("Case did not complete numerical integration")
                metrics = dict(manifest["metrics"])
                error = float(metrics.get("energy_error_fraction", math.inf))
                if not math.isfinite(error) or not 0 <= error <= 2e-6:
                    raise SimulationError("Case energy check is missing or failed")
                features = _case_features(case, manifest, probe_m=baseline_features["probe_position_m"] if baseline_features else None)
                if slot == 0:
                    baseline_features = features
                comparable = bool(baseline_features and all(features[k] == baseline_features[k]
                    for k in ("exposure_signature", "mask_signature", "grid_signature")))
                row = {**identity, "valid": True, "comparable": comparable,
                    "score": thermal_similarity_score(features, baseline_features) if comparable else None,
                    "comparison_exclusion": None if comparable else "Reference unavailable or actual exposed path/grid/selected-voxel footprints differ",
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str))},
                    **{k: features[k] for k in (*SCORE_FEATURES, "peak_temperature_rise_k", "mean_peak_temperature_rise_k", "exposed_vector_count", "selected_part_voxels")},
                    "probe_csv": (case / "transfer-probe.csv").relative_to(folder).as_posix(),
                    "metrics": {**metrics, **features}}
                report["rows"].append(row)
            except SimulationCancelled:
                raise
            except Exception as exc:
                report["failures"].append({**identity, "valid": False, "error": f"{type(exc).__name__}: {exc}"})
            persist()
        ordered = sorted((r for r in report["rows"] if r["kind"] != "baseline" and r["score"] is not None),
                         key=lambda r: (r["score"], r["name"]))
        for material in report["target_materials"]:
            rows = [r for r in ordered if r["material"] == material]
            closest = rows[0] if rows else {}
            report["rankings"].append(dict(material=material, ranked_candidates_count=len(rows),
                closest_case_folder=closest.get("case_folder"), closest_preset_name=closest.get("name"),
                closest_power_scale=closest.get("power_scale"), closest_speed_scale=closest.get("speed_scale"),
                score=closest.get("score"), candidates=[{k: r[k] for k in ("case_folder", "name", "power_scale", "speed_scale", "score")} for r in rows],
                interpretation="Closest sampled normalized thermal response to the reference; not a qualified recipe or defect prediction"))
        good, bad = len(report["rows"]), len(report["failures"])
        report.update(status="partial" if good and bad else "failed" if bad else "complete",
            processing_complete=True, runtime_s=time.perf_counter() - started,
            completed_case_count=good, failed_case_count=bad, comparison_available=baseline_features is not None,
            closest_preset_name=ordered[0]["name"] if ordered else None)
        persist()
        progress(1., f"Preset comparison finished · {good} computed cases · {bad} failures")
        return report
    except SimulationCancelled:
        report.update(status="cancelled", runtime_s=time.perf_counter() - started)
        persist()
        raise
