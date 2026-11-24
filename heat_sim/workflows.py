"""Reproducible comparison, screening, uncertainty and constrained calibration."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .solver import SimulationConfig, SimulationCancelled, simulate
from .storage import RESULTS, atomic_json, cache_key, load_simulation, save_simulation, model_fingerprint


def geometry_distance(width, depth, target_width, target_depth):
    """Dimensionless relative distance, undefined if either case has no liquidus pool."""
    values = (width, depth, target_width, target_depth)
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in values):
        return None
    return math.hypot(width / target_width - 1, depth / target_depth - 1)


def uncertainty_scenarios(bounds):
    """One-factor endpoints: a diagnostic envelope, not a probability distribution."""
    center = {k: (float(v[0]) + float(v[1])) / 2 for k, v in bounds.items()}
    scenarios = [("range midpoint", center)]
    for name, (low, high) in bounds.items():
        for label, value in (("low", low), ("high", high)):
            scenarios.append((f"{name} {label}", {**center, name: float(value)}))
    return scenarios


def run_workflow(request, folder: Path, progress=None, cancel=None):
    started = time.perf_counter()
    base = SimulationConfig(**request["config"])
    options = request.get("options", {})
    kind = request["kind"]
    rows = []
    failures = []
    cache_hits = 0
    progress = progress or (lambda fraction, message: None)
    cancel = cancel or (lambda: False)
    units = request.get("assumptions", {})

    def run_one(config, label, scenario="nominal", slot=0, total=1):
        nonlocal cache_hits
        if cancel():
            raise SimulationCancelled("Cancelled between simulations")
        key = cache_key(config)
        try:
            if (RESULTS / "cache" / key / "result.json").exists():
                meta, _ = load_simulation(key)
                metrics = meta["metrics"]
                cache_hits += 1
                progress(slot / total, f"Cached · {label}")
            else:
                result = simulate(config,
                    progress=lambda f, msg: progress(min(.99, (slot + f) / total), f"{label} · {msg}"),
                    cancel=cancel)
                save_simulation(result, key)
                metrics = result.metrics
            row = {"label": label, "scenario": scenario, "alloy": config.alloy,
                   "power_w": config.power_w, "speed_mm_s": config.speed_m_s * 1000,
                   "absorptivity": config.absorptivity, "beam_um": config.beam_diameter_m * 1e6,
                   "powder_k_w_mk": config.powder_k_w_mk, "tracks": config.tracks,
                   "cache_key": key, **metrics}
            row["width_to_hatch"] = row["width_um"] / (config.hatch_m * 1e6) if config.tracks == 1 else None
            row["geometry_distance"] = None
            row["penetrates_substrate"] = row["penetration_um"] > 0
            rows.append(row)
            return row
        except SimulationCancelled:
            raise
        except Exception as exc:
            error = {"label": label, "scenario": scenario, "config": asdict(config),
                     "error": f"{type(exc).__name__}: {exc}", "valid": False}
            failures.append(error)
            atomic_json(folder / "failures.json", failures)
            progress(min(.99, (slot + 1) / total), f"Failed · {label} · recorded")
            return None

    baseline_abs = float(options.get("absorptivity_17", base.absorptivity))
    steel_abs = float(options.get("absorptivity_316", .35))
    baseline = replace(base, alloy="17-4PH", absorptivity=baseline_abs)
    steel = replace(base, alloy="316L", absorptivity=steel_abs)
    extra = {}

    if kind == "comparison":
        candidates = [("17-4PH baseline", baseline), ("316L identical settings", steel)]
        for speed in (1260, 1400, 1540):
            # The requested bracket is anchored at 120W even when the editable baseline changes.
            candidates.append((f"316L bracket {speed} mm/s", replace(steel, power_w=120, speed_m_s=speed / 1000)))
        target = None
        for i, (label, cfg) in enumerate(candidates):
            row = run_one(cfg, label, slot=i, total=len(candidates))
            if i == 0:
                target = row
            elif row and target:
                row["geometry_distance"] = geometry_distance(row["width_um"], row["depth_um"], target["width_um"], target["depth_um"])
    elif kind == "sweep":
        powers = options.get("powers_w", [100, 120, 140])
        speeds = options.get("speeds_mm_s", [1260, 1400, 1540])
        if len(powers) * len(speeds) > 81:
            raise ValueError("Limit sweep to 81 points")
        total = 1 + len(powers) * len(speeds)
        target = run_one(baseline, "17-4PH baseline", total=total)
        for i, (power, speed) in enumerate((p, v) for p in powers for v in speeds):
            row = run_one(replace(steel, power_w=float(power), speed_m_s=float(speed) / 1000), f"316L {power:g} W / {speed:g} mm/s", slot=i+1, total=total)
            if row and target:
                row["geometry_distance"] = geometry_distance(row["width_um"], row["depth_um"], target["width_um"], target["depth_um"])
    elif kind == "uncertainty":
        bounds = options["bounds"]
        scenarios = uncertainty_scenarios(bounds)
        for i, (name, params) in enumerate(scenarios):
            shared = dict(beam_diameter_m=params["beam_um"] * 1e-6, powder_k_w_mk=params["powder_k_w_mk"])
            total = len(scenarios) * 4
            target = run_one(replace(baseline, **shared, absorptivity=params["absorptivity_17"]), "17-4PH baseline", name, i*4, total)
            for j, speed in enumerate((1260, 1400, 1540)):
                row = run_one(replace(steel, **shared, absorptivity=params["absorptivity_316"], power_w=120, speed_m_s=speed/1000), f"316L 120 W / {speed} mm/s", name, i*4+j+1, total)
                if row and target:
                    row["geometry_distance"] = geometry_distance(row["width_um"], row["depth_um"], target["width_um"], target["depth_um"])
        valid_scenarios = []
        for name, _ in scenarios:
            candidates = [r for r in rows if r["scenario"] == name and r["alloy"] == "316L" and r["geometry_distance"] is not None]
            if len(candidates) != 3:
                continue
            ordered = sorted(candidates, key=lambda r: r["geometry_distance"])
            # Quantized geometry can give ties; retain all winners within 1% relative distance.
            best = ordered[0]["geometry_distance"]
            winners = [r["speed_mm_s"] for r in ordered if abs(r["geometry_distance"]-best) <= max(.01*best, 1e-8)]
            valid_scenarios.append({"scenario": name, "closest_speeds_mm_s": winners})
        winner_sets = [set(item["closest_speeds_mm_s"]) for item in valid_scenarios]
        common = sorted(set.intersection(*winner_sets)) if winner_sets else []
        extra = {"rankings": valid_scenarios, "common_closest_speeds_mm_s": common,
                 "rankable_scenarios": len(valid_scenarios), "scenario_count": len(scenarios),
                 "interpretation": "One-factor endpoints; interactions and statistical confidence are not established. Shared beam/powder conductivity are varied together for the alloys; absorptivities are varied independently."}
    elif kind == "calibration":
        measured_w = float(options["measured_width_um"])
        measured_d = float(options["measured_depth_um"])
        bounds = tuple(options.get("absorption_bounds", [.1, .8]))
        if not (0 < measured_w and 0 < measured_d and .05 <= bounds[0] < bounds[1] <= .9):
            raise ValueError("Positive measurements and absorptivity bounds within 0.05–0.9 required")
        trials = []
        def objective(alpha):
            row = run_one(replace(baseline, absorptivity=float(alpha)), f"17-4PH fit A={alpha:.4f}", slot=len(trials), total=13)
            score = geometry_distance(row["width_um"], row["depth_um"], measured_w, measured_d) if row else None
            trials.append((float(alpha), score))
            if row:
                row["measurement_distance"] = score
            return float(score) if score is not None else 1e6
        # Evaluate endpoints as well as the interior; report the best actual evaluation.
        objective(bounds[0]); objective(bounds[1])
        minimize_scalar(objective, bounds=bounds, method="bounded", options={"maxiter": 9, "xatol": .005})
        usable = [item for item in trials if item[1] is not None]
        if not usable:
            raise ValueError("No fit evaluation produced a resolved liquidus pool")
        best_alpha, score = min(usable, key=lambda t: t[1])
        best_row = min((r for r in rows if r.get("measurement_distance") is not None), key=lambda r: r["measurement_distance"])
        extra = {"fitted_absorptivity_17": best_alpha, "relative_geometry_residual": score,
                 "measured_width_um": measured_w, "measured_depth_um": measured_d,
                 "predicted_width_um": best_row["width_um"], "predicted_depth_um": best_row["depth_um"],
                 "absorption_bounds": bounds, "evaluations": len(trials),
                 "note": "Only 17-4PH absorptivity is fitted. Beam, power, powder and geometry remain fixed. Mesh quantization, measurement error and omitted flow can confound the fit.316L absorptivity is unchanged."}
    else:
        raise ValueError(f"Unknown workflow: {kind}")

    report = {"kind": kind, "fingerprint": model_fingerprint(), "rows": rows, "failures": failures,
              "extra": extra, "assumptions": units, "cache_hits": cache_hits,
              "wall_runtime_s": time.perf_counter()-started, "experimental_validation": False}
    atomic_json(folder / "report.json", report)
    pd.DataFrame(rows).to_csv(folder / "comparison.csv", index=False)
    if rows:
        from .plots import thermal_figure, sweep_figure
        import matplotlib.pyplot as plt
        figure_row = next((r for r in rows if r["alloy"] == "316L"), rows[0])
        meta, fields = load_simulation(figure_row["cache_key"])
        fig = thermal_figure(meta, fields)
        fig.savefig(folder / "thermal.png", dpi=170)
        plt.close(fig)
        if kind == "sweep":
            fig = sweep_figure(pd.DataFrame([r for r in rows if r["alloy"] == "316L"]))
            fig.savefig(folder / "power-speed.png", dpi=170)
            plt.close(fig)
    progress(1., "Finished; results and configuration saved locally")
    return report
