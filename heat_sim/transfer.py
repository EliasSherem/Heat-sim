"""Compare one imported geometry across alloys using actual coarse build solves.

Scores are thermal similarity screens, not melt-pool/defect/recipe predictions.
Every alloy has an independent optical input; no settings are derived from a
melting-point ratio. See docs/MATERIAL_TRANSFER.md.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
import csv
import json
import math
import time

import numpy as np

from .build_solver import BuildConfig, run_build
from .materials import MATERIALS, ILLUSTRATIVE_ABSORPTIVITY
from .solver import SimulationCancelled, SimulationError
from .storage import atomic_json


SCORE_FEATURES = ("mean_peak_enthalpy_fraction", "p95_peak_enthalpy_fraction",
                  "peak_enthalpy_fraction")


def enthalpy_fraction(temperature_k, alloy, initial_k):
    """Excess h divided by excess h needed to reach local liquidus.

    This is an enthalpy index, NOT liquid fraction. A coarse voxel index below
    one does not rule out transient melting below the resolved cell scale.
    """
    material = MATERIALS[alloy]
    h0 = float(material.enthalpy(initial_k))
    denominator = float(material.enthalpy(material.liquidus))-h0
    if denominator <= 0:
        raise ValueError("Initial temperature must be below the material liquidus")
    return np.maximum(0., (material.enthalpy(temperature_k)-h0)/denominator)


def thermal_similarity_score(candidate, baseline):
    """Dimensionless relative RMS of three resolved peak-field summaries.

    An additive floor of 0.02 prevents almost unheated reference regions from
    producing an arbitrary huge relative score. It is a disclosed scaling
    choice, not a material property or acceptance threshold.
    """
    a = np.asarray([candidate[k] for k in SCORE_FEATURES], dtype=float)
    b = np.asarray([baseline[k] for k in SCORE_FEATURES], dtype=float)
    if not np.all(np.isfinite(np.r_[a,b])):
        raise ValueError("Non-finite thermal features cannot be scored")
    return float(np.sqrt(np.mean(((a-b)/np.maximum(abs(b), .02))**2)))


def _code_fingerprint():
    h=sha256()
    for name in ("build_solver.py", "solver.py", "materials.py", "transfer.py", "presets.py", "geometry.py", "job_import.py"):
        path=Path(__file__).with_name(name)
        if path.exists():h.update(path.read_bytes())
    return h.hexdigest()


def _external_geometry_fingerprint(config):
    source=getattr(config,"geometry_file",None)
    if not source:return "none"
    h=sha256()
    with Path(source).expanduser().open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def _job_fingerprint(job):
    h=sha256(str(job.source_sha256).encode())
    h.update(repr(job.layer_thickness_m).encode())
    for layer in job.layers:
        h.update(repr((layer.index,layer.z_m)).encode())
        for name in ("vectors", "process_ids", "part_ids", "path_ids"):
            h.update(np.asarray(getattr(layer,name)).tobytes())
    return h.hexdigest()


def _exposure_signature(folder):
    """Compare the actual scan prefix, including any clipped final vector."""
    with np.load(Path(folder)/"timeline.npz",allow_pickle=False) as data:
        events=data["events"]
        events=events[events["kind"]==1]
        h=sha256()
        for name in ("start_m","end_m","layer_index","section","vector_index"):
            h.update(np.ascontiguousarray(events[name]).tobytes())
        return h.hexdigest(), int(len(events))


def _safe_child(folder, relative):
    target=(Path(folder)/relative).resolve()
    if not target.is_relative_to(Path(folder).resolve()):
        raise ValueError("Unsafe case artifact path")
    return target


def _case_features(folder,manifest,probe_m=None):
    """Read saved computed fields; no synthetic histories or geometry metrics."""
    cfg=manifest["config"]
    frames=manifest.get("frames",[])
    if not frames:
        raise SimulationError("Build has no converged saved frames")
    with np.load(_safe_child(folder,frames[-1]["file"]),allow_pickle=False) as saved:
        active=saved["active_fraction"] if "active_fraction" in saved else saved["deposited_fraction"]
        plate=saved["plate_mask"].astype(bool)
        mask=saved["part_mask"].astype(bool)&(active>0)&~plate
        coords=[saved[k].copy() for k in ("x_m","y_m","z_m")]
        shape=active.shape
    if not np.any(mask):
        raise SimulationError("No exposed part voxels in selected range; thermal comparison unavailable")
    with np.load(Path(folder)/"peak-fields.npz",allow_pickle=False) as saved:
        peak=np.asarray(saved["temperature_k"],float)
        if peak.shape!=shape:
            raise SimulationError("Saved peak field/grid mismatch")
        for key,axis in zip(("x_m","y_m","z_m"),coords):
            if not np.array_equal(saved[key],axis):
                raise SimulationError("Saved peak-field coordinates differ from converged frames")
    if not np.all(np.isfinite(peak[mask])):
        raise SimulationError("Non-finite peak field cannot be compared")
    indices=enthalpy_fraction(peak[mask],cfg["alloy"],cfg["initial_k"])
    if probe_m is None:
        selected=np.argwhere(mask)
        centre=selected.mean(axis=0)
        ijk=selected[np.argmin(np.sum((selected-centre)**2,axis=1))]
        probe_m=[float(axis[i]) for axis,i in zip(coords,ijk)]
    probe_indices=tuple(int(np.argmin(abs(axis-value))) for axis,value in zip(coords,probe_m))
    history=[]
    for entry in frames:
        with np.load(_safe_child(folder,entry["file"]),allow_pickle=False) as saved:
            temperature=float(saved["temperature_k"][probe_indices])
            a=saved["active_fraction"] if "active_fraction" in saved else saved["deposited_fraction"]
            history.append(dict(time_s=float(entry["time_s"]),temperature_k=temperature,
                temperature_rise_k=temperature-float(cfg["initial_k"]),
                enthalpy_fraction=float(enthalpy_fraction(temperature,cfg["alloy"],cfg["initial_k"])),
                active_fraction=float(a[probe_indices])))
    _write_csv(Path(folder)/"transfer-probe.csv",history)
    signature,count=_exposure_signature(folder)
    mask_signature=sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
    grid_signature=sha256(b"".join(np.ascontiguousarray(axis).tobytes() for axis in coords)).hexdigest()
    result=dict(mean_peak_enthalpy_fraction=float(np.mean(indices)),
                p95_peak_enthalpy_fraction=float(np.quantile(indices,.95)),
                peak_enthalpy_fraction=float(np.max(indices)),
                peak_temperature_rise_k=float(np.max(peak[mask])-cfg["initial_k"]),
                mean_peak_temperature_rise_k=float(np.mean(peak[mask])-cfg["initial_k"]),
                selected_part_voxels=int(mask.sum()),exposure_signature=signature,
                exposed_vector_count=count,mask_signature=mask_signature,grid_signature=grid_signature,
                probe_position_m=[float(v) for v in probe_m],probe_history=history,
                probe_csv="transfer-probe.csv")
    atomic_json(Path(folder)/"transfer-features.json",result)
    return result


def _write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value) if isinstance(value,(dict,list,tuple)) else value
                             for key,value in row.items()})
    temporary.replace(path)


def _multipliers(values,name,lower,upper):
    values=sorted(set(float(value) for value in values))
    if not values or any(not math.isfinite(v) or not lower<=v<=upper for v in values):
        raise ValueError(f"{name} must contain finite multipliers in [{lower},{upper}]")
    return values


def run_transfer(job,base_config,request_options,folder,progress=None,cancel=None):
    """Replay unchanged imported paths for current and selected target materials.

    Options: baseline_material, target_materials, absorptivities, power_scales,
    speed_scales, evaluate_candidates, substrate_follows_material,
    evaluate_uncertainty, uncertainty_absorptivity_ranges.
    Completed cases are cached under folder/cases/<content-hash>/.
    """
    start=time.perf_counter();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    progress=progress or (lambda fraction,message:None)
    cancel=cancel or (lambda:False)
    base=BuildConfig(**base_config) if isinstance(base_config,dict) else base_config
    options=dict(request_options)
    baseline=options.get("baseline_material",base.alloy)
    targets=list(dict.fromkeys(options.get("target_materials",["316L"])))
    if baseline not in MATERIALS or not targets or any(m not in MATERIALS for m in targets):
        raise ValueError("Choose a supported baseline and at least one target alloy")
    if baseline in targets:
        raise ValueError("Target materials must differ from the current baseline material")
    absorptivities=dict(ILLUSTRATIVE_ABSORPTIVITY)
    if base.absorptivity is not None:absorptivities[baseline]=float(base.absorptivity)
    absorptivities.update({m:float(a) for m,a in options.get("absorptivities",{}).items()})
    if any(not math.isfinite(absorptivities[m]) or not 0<absorptivities[m]<=1 for m in [baseline,*targets]):
        raise ValueError("Each alloy needs an independent absorptivity in (0,1]")
    powers=_multipliers(options.get("power_scales",[.9,1.,1.1]),"Power scales",.05,3.)
    speeds=_multipliers(options.get("speed_scales",[.9,1.,1.1]),"Speed scales",.1,5.)
    points=[(1.,1.)]
    if options.get("evaluate_candidates",False):
        points+= [(p,v) for p in powers for v in speeds if (p,v)!=(1.,1.)]
    plans=[dict(material=baseline,power_scale=1.,speed_scale=1.,kind="baseline",
                absorptivity=absorptivities[baseline],scenario="nominal")]
    for material in targets:
        for p,v in points:
            plans.append(dict(material=material,power_scale=p,speed_scale=v,
                kind="identical" if (p,v)==(1.,1.) else "candidate",
                absorptivity=absorptivities[material],scenario="nominal"))
    ranges=options.get("uncertainty_absorptivity_ranges",{})
    if options.get("evaluate_uncertainty",False):
        for material in targets:
            if material not in ranges:
                raise ValueError(f"Supply independent absorptivity bounds for {material}")
            bounds=list(ranges[material])
            if len(bounds)!=2 or not all(math.isfinite(float(x)) for x in bounds) or not 0<float(bounds[0])<=float(bounds[1])<=1:
                raise ValueError(f"Invalid absorptivity bounds for {material}")
            for label,a in (("absorptivity_low",float(bounds[0])),("absorptivity_high",float(bounds[1]))):
                for p,v in points:
                    plans.append(dict(material=material,power_scale=p,speed_scale=v,kind="uncertainty",
                                      absorptivity=a,scenario=label))
    if len(plans)>120:
        raise ValueError("Transfer request exceeds120cases; reduce target/grid/scenario counts")
    fingerprint=_code_fingerprint();job_hash=_job_fingerprint(job)
    geometry_hash=_external_geometry_fingerprint(base)
    report=dict(schema=1,status="running",baseline_material=baseline,target_materials=targets,
                rows=[],failures=[],rankings=[],uncertainty={},cache_hits=0,
                config=asdict(base),options=options,absorptivities=absorptivities,
                job_source_sha256=job.source_sha256,job_fingerprint=job_hash,model_fingerprint=fingerprint,
                supplemental_geometry_sha256=geometry_hash,numerical_resolution_assessed=False,
                score_definition="RMS relative differences of mean,95th-percentile,max peak-field h-fractions; denominator floor0.02; lower means closer thermal screen only",
                experimental_validation=False,microscopic_geometry_resolved=False,
                warnings=["Thermal similarity is a comparative screen, not an optimal recipe or validated alloy conversion.",
                          "Whole-build voxels do not resolve melt-pool width/depth, flow, recoil, vaporisation, keyhole instability or defect probability.",
                          "All optical inputs remain independently assumed unless measured; a baseline fit does not identify target absorptivity.",
                          "Small score differences have no established numerical/experimental significance until this selected geometry/range is refined and measured.",
                          "Power/speed multipliers preserve each imported scan section's relative settings and unchanged geometric paths.",
                          "Scores compare the selected path prefix only; no conclusion about full-part density or fatigue follows."])
    if options.get("substrate_follows_material",False):
        report["warnings"].append("Substrate follows each selected alloy; plate-property changes contribute to the comparison.")
    if base.time_limit_s is not None:
        report["warnings"].append("Physical-time limit may truncate different scan prefixes at different speeds; mismatched exposures are excluded from rankings. Prefer layer/vector bounds for same-geometry comparisons.")
    # Keep the worker's typed, reproducible request intact. Standalone callers
    # receive a separate numerical-workflow configuration.
    atomic_json(folder/"transfer-request.json",dict(config=asdict(base),options=options))
    baseline_features=None

    def persist():
        atomic_json(folder/"report.json",report)
        _write_csv(folder/"comparison.csv",[{k:v for k,v in row.items() if k!="metrics"} for row in report["rows"]])
        atomic_json(folder/"failures.json",report["failures"])

    try:
        for slot,plan in enumerate(plans):
            if cancel():raise SimulationCancelled("Transfer cancelled between cases; completed cases retained")
            material=plan["material"]
            cfg=replace(base,alloy=material,substrate=material if options.get("substrate_follows_material",False) else base.substrate,
                        absorptivity=plan["absorptivity"],power_scale=plan["power_scale"],speed_scale=plan["speed_scale"])
            key=sha256((fingerprint+job_hash+geometry_hash+json.dumps(asdict(cfg),sort_keys=True)).encode()).hexdigest()[:24]
            case=folder/"cases"/key
            label=f"{material} · power×{cfg.power_scale:g} / speed×{cfg.speed_scale:g} · {plan['scenario']}"
            try:
                cfg.validate(job)
                cached=case/"manifest.json"
                manifest=json.loads(cached.read_text()) if cached.exists() else None
                if manifest and manifest.get("status")=="complete":
                    report["cache_hits"]+=1
                    progress(slot/len(plans),f"Cached · {label}")
                else:
                    manifest=run_build(job,cfg,case,
                        progress=lambda f,message:progress(min(.999,(slot+f)/len(plans)),f"{label} · {message}"),
                        cancel=cancel)
                if manifest.get("status")!="complete":
                    raise SimulationError("Case did not complete numerical integration")
                metrics=dict(manifest["metrics"])
                if not math.isfinite(float(metrics.get("energy_error_fraction",math.inf))) or float(metrics["energy_error_fraction"])>2e-6:
                    raise SimulationError("Case energy check is missing or failed")
                features=_case_features(case,manifest,
                    probe_m=baseline_features["probe_position_m"] if baseline_features else None)
                if plan["kind"]=="baseline":baseline_features=features
                comparable=bool(baseline_features and all(features[k]==baseline_features[k]
                    for k in ("exposure_signature","mask_signature","grid_signature")))
                reason=None if comparable else "Baseline failed or actual exposure/grid/selected-voxel footprints differ"
                score=thermal_similarity_score(features,baseline_features) if comparable else None
                row={**plan,"case_folder":case.relative_to(folder).as_posix(),"score":score,"comparable":comparable,
                     "comparison_exclusion":reason,"valid":True,
                     **{k:v for k,v in metrics.items() if isinstance(v,(int,float,bool,str))},
                     **{k:features[k] for k in (*SCORE_FEATURES,"peak_temperature_rise_k","mean_peak_temperature_rise_k","exposed_vector_count","selected_part_voxels")},
                     "probe_csv":(case/"transfer-probe.csv").relative_to(folder).as_posix(),"metrics":{**metrics,**features}}
                report["rows"].append(row)
            except SimulationCancelled:
                raise
            except Exception as exc:
                report["failures"].append({**plan,"case_folder":case.relative_to(folder).as_posix(),"valid":False,
                    "error":f"{type(exc).__name__}: {exc}"})
            persist()
        for material in targets:
            nominal=[row for row in report["rows"] if row["material"]==material and row["scenario"]=="nominal" and row["score"] is not None]
            ordered=sorted(nominal,key=lambda row:(row["score"],row["power_scale"],row["speed_scale"]))
            report["rankings"].append(dict(material=material,ranked_candidates_count=len(ordered),
                closest_case_folder=ordered[0]["case_folder"] if ordered else None,
                closest_power_scale=ordered[0]["power_scale"] if ordered else None,
                closest_speed_scale=ordered[0]["speed_scale"] if ordered else None,
                score=ordered[0]["score"] if ordered else None,
                candidates=[dict(case_folder=row["case_folder"],power_scale=row["power_scale"],speed_scale=row["speed_scale"],score=row["score"]) for row in ordered],
                interpretation="Closest sampled coarse thermal response; requires independent track measurements, no optimal recipe claim"))
            if options.get("evaluate_uncertainty",False):
                scenarios=[]
                for scenario in ("absorptivity_low","nominal","absorptivity_high"):
                    rows=sorted([row for row in report["rows"] if row["material"]==material and row["scenario"]==scenario and row["score"] is not None],key=lambda row:(row["score"],row["power_scale"],row["speed_scale"]))
                    scenarios.append(dict(scenario=scenario,absorptivity=rows[0]["absorptivity"] if rows else None,
                        closest=[rows[0]["power_scale"],rows[0]["speed_scale"]] if rows else None,
                        ranking=[[row["power_scale"],row["speed_scale"]] for row in rows],
                        scores=[row["score"] for row in rows],complete=len(rows)==len(points)))
                complete=all(s["complete"] for s in scenarios)
                enough=len(points)>1
                stable=bool(complete and enough and all(s["closest"]==scenarios[0]["closest"] for s in scenarios))
                report["uncertainty"][material]=dict(scenarios=scenarios,ranking_evaluable=complete and enough,
                    closest_consistent=stable,
                    full_order_consistent=complete and enough and all(s["ranking"]==scenarios[0]["ranking"] for s in scenarios),
                    baseline_absorptivity_fixed=absorptivities[baseline],
                    interpretation=("Closest sampled thermal candidate stays the same across these absorptivity assumptions; this is not experimental validation."
                        if stable else "Ranking is assumption-sensitive or incomplete. Measure target effective absorptivity/delivered power and beam profile, then obtain baseline and target track cross-sections before narrowing printer trials."),
                    omitted_uncertainty="Only independent target absorptivity endpoints varied; beam size, powder k, geometry, timing and numerical/model errors remain.")
        report.update(status="complete",runtime_s=time.perf_counter()-start,
                      completed_case_count=len(report["rows"]),failed_case_count=len(report["failures"]),
                      comparison_available=baseline_features is not None)
        if report["failures"]:report["warnings"].append("Failed cases are recorded separately and excluded from scores/rankings.")
        persist();progress(1.,f"Transfer complete · {len(report['rows'])} computed cases · {len(report['failures'])} failures")
        return report
    except SimulationCancelled:
        report.update(status="cancelled",runtime_s=time.perf_counter()-start)
        persist();raise

