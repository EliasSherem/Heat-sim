"""Explain saved thermal evidence without turning a screen into a print recipe.

Reads frozen run inputs and converged results only. No current editor values,
synthetic temperatures, or defect/quality predictions enter this view.
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np

from .run_catalog import read_json, read_run


_SETTING_LABELS = {
    "alloy": "Material", "substrate": "Substrate", "power_scale": "Imported power multiplier",
    "speed_scale": "Imported speed multiplier", "initial_k": "Preheat (°C)",
    "recoat_time_s": "Inter-layer pause (s)", "beam_diameter_m": "Beam diameter (µm)",
    "absorptivity": "Effective absorptivity", "power_factor": "Delivered power factor",
    "powder_packing": "Powder packing fraction", "powder_k_w_mk": "Powder conductivity (W/m·K)",
    "jump_speed_m_s": "Jump speed (m/s)", "jump_delay_s": "Jump settling delay (s)",
}


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _inside(folder, relative):
    if not isinstance(relative, (str, Path)):
        return None
    root = Path(folder).resolve()
    target = (root / relative).resolve()
    return target if target.is_relative_to(root) else None


def _display_setting(key, value):
    number = _finite(value)
    if number is None:
        return str(value)
    if key == "initial_k":
        number -= 273.15
    elif key == "beam_diameter_m":
        number *= 1e6
    return f"{number:g}"


def config_changes(candidate, reference):
    """Human readable changes between saved physical configurations."""
    rows = []
    for key, label in _SETTING_LABELS.items():
        if key not in candidate or key not in reference:
            continue
        left, right = reference[key], candidate[key]
        a, b = _finite(left), _finite(right)
        equal = math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) if a is not None and b is not None else left == right
        if not equal:
            rows.append({"Setting": label, "Reference": _display_setting(key, left),
                         "Selected preset": _display_setting(key, right)})
    prior = {str(key): value for key, value in (reference.get("section_overrides") or {}).items()}
    current = {str(key): value for key, value in (candidate.get("section_overrides") or {}).items()}
    for section in sorted(set(prior) | set(current)):
        for field, label, factor in (("power_w", "power (W)", 1.), ("speed_m_s", "speed (mm/s)", 1000.)):
            left = prior.get(section, {}).get(field)
            right = current.get(section, {}).get(field)
            if left != right:
                rows.append({"Setting": f"Region {section} {label} before global multipliers",
                             "Reference": "Imported setting" if left is None else f"{float(left)*factor:g}",
                             "Selected preset": "Imported setting" if right is None else f"{float(right)*factor:g}"})
    return rows


def _case_summary(folder, row, manifest, report_config, case_id):
    config = dict(manifest.get("config") or row.get("config") or report_config or {})
    metrics = dict(manifest.get("metrics") or {})
    metrics.update(row.get("metrics") or {})
    metrics.update(row.get("features") or {})
    for key in ("mean_peak_enthalpy_fraction", "p95_peak_enthalpy_fraction", "peak_enthalpy_fraction",
                "peak_temperature_rise_k", "mean_peak_temperature_rise_k", "physical_duration_s",
                "stored_energy_j", "absorbed_energy_j", "energy_error_fraction", "peak_temperature_k"):
        if key in row:
            metrics[key] = row[key]
    status = manifest.get("status", "unknown")
    completed = status == "complete" and row.get("valid", True) is not False
    comparable = completed and row.get("comparable") is True
    score = _finite(row.get("score")) if comparable else None
    if score is not None and score < 0:
        score = None
    peak = _finite(metrics.get("peak_temperature_k"))
    mean_index = _finite(metrics.get("mean_peak_enthalpy_fraction"))
    weighted = mean_index is None
    if weighted:
        mean_index = _finite(metrics.get("mean_material_peak_enthalpy_ratio"))
    error = _finite(metrics.get("energy_error_fraction"))
    name = row.get("preset_name") or row.get("name")
    material = row.get("material") or config.get("alloy") or "Material not recorded"
    if not name:
        if row.get("kind") in ("reference", "baseline"):
            name = f"Reference · {material}"
        elif "power_scale" in row or "speed_scale" in row:
            name = f"{material} · power ×{row.get('power_scale', 1):g} / speed ×{row.get('speed_scale', 1):g}"
            if row.get("scenario") not in (None, "nominal"):
                name += " · " + str(row["scenario"]).replace("_", " ")
        else:
            name = str(material)
    return dict(id=case_id, folder=str(folder), name=str(name), material=str(material),
                kind=row.get("kind", "single"), state=status, completed=completed,
                comparable=comparable, score=score, config=config, metrics=metrics,
                peak_temperature_c=peak-273.15 if peak is not None else None,
                mean_peak_enthalpy_index=mean_index, mean_index_weighted=weighted,
                part_peak_temperature_rise_k=_finite(metrics.get("peak_temperature_rise_k")),
                duration_s=_finite(metrics.get("physical_duration_s", manifest.get("physical_duration_s"))),
                retained_energy_j=_finite(metrics.get("stored_energy_j")),
                absorbed_energy_j=_finite(metrics.get("absorbed_energy_j")),
                energy_error_fraction=error, energy_check_passed=error is not None and 0 <= error <= 2e-6,
                high_temperature_caution=bool(metrics.get("high_temperature_caution", False)),
                exclusion=row.get("comparison_exclusion") or row.get("exclusion_reason"),
                frames=len(manifest.get("frames", [])))


def case_interpretation(case, reference=None):
    """Plain-language observations are conditional on stored comparable data."""
    statements = []
    if not case["completed"]:
        return ["This case did not complete. Its partial frames cannot establish a final thermal comparison."]
    if reference and case["id"] != reference["id"]:
        if case["score"] is None:
            statements.append("This case has no comparable thermal score. " +
                              (case.get("exclusion") or "A complete matching reference and exposure footprint are required."))
        else:
            statements.append(f"Thermal difference from the reference: {case['score']:.4g}. Lower means closer among the simulated settings; zero means these summary features match.")
            a = _finite(case["metrics"].get("mean_peak_enthalpy_fraction"))
            b = _finite(reference["metrics"].get("mean_peak_enthalpy_fraction"))
            if a is not None and b is not None:
                if math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10):
                    statements.append("The mean resolved peak enthalpy index matches the reference within rounding.")
                else:
                    direction = "higher" if a > b else "lower"
                    statements.append(f"The mean resolved peak enthalpy index is {direction}: {a:.4g} versus {b:.4g}. Each index uses that case's own alloy and preheat; it is not liquid fraction or a melt-pool measurement.")
                    if case.get("kind") != "uncertainty":
                        bracket = "lower power or faster scans" if a > b else "higher power or slower scans"
                        statements.append(f"If your aim is to move this mean index toward the reference, test a bracket of {bracket}, changing one setting at a time. This is a direction to check in simulation; the mean, upper-tail and maximum response can move differently.")
    else:
        statements.append("This is the saved reference response. Simulate alternative presets over the same layer range to compare changes.")
    retained = case.get("retained_energy_j")
    absorbed = case.get("absorbed_energy_j")
    if retained is not None and absorbed is not None and absorbed > 0:
        statements.append(f"At the end of the selected print and cooldown, the part/powder/plate domain retains {retained:.4g} J of excess heat from {absorbed:.4g} J absorbed. This domain measure includes the plate and depends on the saved cooldown and boundaries.")
    if case.get("high_temperature_caution"):
        statements.append("The solver flagged temperatures beyond its stated high-temperature range; examine this case before using it to narrow a printer trial.")
    if not case.get("energy_check_passed"):
        statements.append("A passing saved energy-balance check is unavailable. Review run diagnostics before interpreting this result.")
    return statements


def load_result_insights(folder):
    """Load single-build, material-transfer and named-preset study evidence."""
    folder = Path(folder).resolve()
    info = read_run(folder)
    if info is None:
        raise ValueError("No saved simulation was found in this folder")
    report = read_json(folder / "report.json")
    request = read_json(folder / "request.json")
    cases, failures = [], list(report.get("failures") or [])
    if report:
        for index, row in enumerate(report.get("rows", [])):
            if not isinstance(row, dict):
                continue
            case_folder = _inside(folder, row.get("case_folder"))
            if case_folder is None:
                failures.append(dict(preset_name=row.get("preset_name", "Unnamed case"), error="Saved case path is invalid"))
                continue
            manifest = read_json(case_folder / "manifest.json")
            cases.append(_case_summary(case_folder, row, manifest,
                                       report.get("config") or {}, f"{index}:{row.get('case_folder', '')}"))
    else:
        manifest = read_json(folder / "manifest.json")
        if manifest:
            cases.append(_case_summary(folder, {"kind": "single"}, manifest,
                                       request.get("build_config") or {}, "single"))
    reference = next((case for case in cases if case["kind"] in ("baseline", "reference")), None)
    if reference is None and len(cases) == 1 and not report:
        reference = cases[0]
    for case in cases:
        case["changes"] = config_changes(case["config"], reference["config"]) if reference else []
        case["interpretation"] = case_interpretation(case, reference)
    ranked = sorted([case for case in cases if case["score"] is not None and
                     case["kind"] not in ("baseline", "reference", "uncertainty")], key=lambda item: item["score"])
    return dict(folder=str(folder), run_id=info.run_id, job_name=info.job_name,
                study_name=report.get("study_name") or request.get("options", {}).get("study_name"),
                material_label=info.material_label, layer_label=info.layer_label,
                state=info.state, status_label=info.status_label, message=info.message,
                cases=cases, reference=reference, ranked=ranked, failures=failures,
                warnings=list(report.get("warnings") or []), uncertainty=report.get("uncertainty") or {},
                numerical_resolution_assessed=report.get("numerical_resolution_assessed", False),
                reference_name=reference["name"] if reference else "No completed reference",
                scope_config=request.get("build_config") or report.get("config") or
                             (cases[0]["config"] if cases else {}))


def load_temperature_history(case_folder, max_points=24):
    """Read actual saved samples at one fixed exposed cell plus spatial maxima.

    A centre-near exposed cell in the final frame is held fixed through time.
    Before that cell is exposed the probe is absent, not a made-up temperature.
    No interpolation is performed. Frame thinning retains first and last frames.
    """
    folder = Path(case_folder).resolve()
    manifest = read_json(folder / "manifest.json")
    entries = manifest.get("frames") or []
    max_points = max(2, min(int(max_points), 200))
    if not entries:
        return dict(rows=[], position_m=None, available_count=0, sampled_count=0)
    last = _inside(folder, entries[-1].get("file"))
    if last is None or not last.is_file():
        raise ValueError("The final saved frame is unavailable")
    with np.load(last, allow_pickle=False) as saved:
        active = saved["active_fraction"] if "active_fraction" in saved else saved["deposited_fraction"]
        mask = saved["part_mask"].astype(bool) & (active > 0) & ~saved["plate_mask"].astype(bool)
        if not np.any(mask):
            return dict(rows=[], position_m=None, available_count=len(entries), sampled_count=0)
        indices = np.argwhere(mask)
        centre = indices.mean(axis=0)
        ijk = tuple(indices[np.argmin(np.sum((indices-centre)**2, axis=1))])
        coords = [saved[key].copy() for key in ("x_m", "y_m", "z_m")]
        position = [float(axis[index]) for axis, index in zip(coords, ijk)]
    selected = np.unique(np.linspace(0, len(entries)-1, min(max_points, len(entries)), dtype=int))
    rows = []
    for index in selected:
        entry = entries[int(index)]
        path = _inside(folder, entry.get("file"))
        if path is None or not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as saved:
            if any(not np.array_equal(saved[key], axis) for key, axis in zip(("x_m", "y_m", "z_m"), coords)):
                raise ValueError("Saved temperature history uses inconsistent grid coordinates")
            active = saved["active_fraction"] if "active_fraction" in saved else saved["deposited_fraction"]
            mask = saved["part_mask"].astype(bool) & (active > 0) & ~saved["plate_mask"].astype(bool)
            temperature = np.asarray(saved["temperature_k"], float)
            rows.append(dict(time_s=float(entry["time_s"]),
                             probe_temperature_c=float(temperature[ijk]-273.15) if mask[ijk] and np.isfinite(temperature[ijk]) else None,
                             part_max_temperature_c=float(np.max(temperature[mask])-273.15) if np.any(mask) else None))
    return dict(rows=rows, position_m=position, available_count=len(entries), sampled_count=len(rows))


def insights_text(insights):
    lines = ["Tracklab simulation interpretation", insights["job_name"],
             f"Run: {insights['run_id']} · {insights['status_label']}", insights["layer_label"],
             f"Reference: {insights['reference_name']}", ""]
    if insights.get("study_name"):
        lines.insert(1, insights["study_name"])
    for case in insights["cases"]:
        lines += [case["name"], f"Material: {case['material']} · outcome: {case['state']}"]
        for change in case["changes"]:
            lines.append(f"{change['Setting']}: {change['Reference']} → {change['Selected preset']}")
        lines.extend(case["interpretation"])
        lines.append("")
    if insights["failures"]:
        lines.append("Failed cases excluded from ranking:")
        for failure in insights["failures"]:
            lines.append(f"{failure.get('preset_name') or failure.get('material', 'Case')}: {failure.get('error', 'Failed')}")
    lines += ["Lower thermal difference identifies a closer sampled coarse response, not an optimal printer recipe.",
              "Scores do not establish melt-pool dimensions, density, porosity or successful printing.",
              "Refine the numerical range and verify promising presets with measured track/part evidence."]
    return "\n".join(lines)
