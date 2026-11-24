"""Bounded-memory timing and provable thermal-step preflight for build replay.

Durations follow build_timeline's imported-vector order without allocating its
global event table. Only one layer's vectorized durations are materialized.
Step counts are LOWER BOUNDS: frames, nonlinear retries and convergence may add
work. Passing this check promises neither runtime nor numerical convergence.
"""
from __future__ import annotations

import math
import numpy as np

from .build_solver import vector_settings


def _step_bound(exposure_blocks, off_blocks, grouping):
    """Activations/recoats flush exposure blocks; cooling groups are <=max(.25,g)."""
    def count(blocks,maximum):
        if not len(blocks):return 0
        # The solver ignores <=1e-12 s event remainders. Allow a conservative
        # small floating tolerance, so an exact grouping boundary cannot add a
        # spurious guaranteed step in this lower-bound calculation.
        duration=np.maximum(np.asarray(blocks,dtype=float)-1e-8,0.)
        return int(np.ceil(duration/maximum).sum())
    return count(exposure_blocks,grouping)+count(off_blocks,max(.25,grouping))


def estimate_run_plan(job,config):
    """Return selected physical timing and a conservative solver-step bound.

    ``truncated_estimate`` means a vector/time cap clips the selected layer
    range; totals for that selected prefix are still evaluated, including a
    fractional final jump/scan. XY jump distance is independent of platform Z.
    ``feasible``/``step_limit_feasible`` only mean that the minimum step count
    does not already exceed max_steps; they do not imply a converged run.

    max_vectors mirrors the timeline's current exact-boundary convention: if
    its cap ends at the final vector of one layer, a following selected layer
    can incur recoating before its first vector discovers the reached cap.
    """
    config.validate(job)
    if not isinstance(config.max_steps,(int,np.integer)) or config.max_steps<=0:
        raise ValueError("max_steps must be a positive integer")
    selected=job.layers[config.layer_start:config.layer_end]
    selected_count=sum(len(layer.vectors) for layer in selected)
    vector_limit=config.max_vectors
    time_limit=config.time_limit_s
    scan_time=jump_time=interlayer_time=cooldown_time=elapsed=0.
    seen=0;layers_activated=0;last_xy=None;stopped=False;has_events=False
    exposure_blocks=[];off_blocks=[]
    clipped_time=False;clipped_vectors=bool(vector_limit is not None and vector_limit<selected_count)

    def remaining_time():
        return math.inf if time_limit is None else max(0.,time_limit-elapsed)

    for ordinal,layer in enumerate(selected):
        if stopped:break
        if ordinal and config.recoat_time_s:
            duration=min(config.recoat_time_s,remaining_time())
            interlayer_time+=duration;elapsed+=duration
            if duration>0:off_blocks.append(duration);has_events=True
            if duration<config.recoat_time_s or (time_limit is not None and elapsed>=time_limit):
                clipped_time=True;stopped=True;break
        if time_limit is not None and elapsed>=time_limit:
            clipped_time=True;stopped=True;break
        # Zero-duration activation is itself an event, including empty layers.
        has_events=True;layers_activated+=1
        available=len(layer.vectors)
        if not available:continue
        if vector_limit is not None and seen>=vector_limit:
            stopped=True;break
        take=available if vector_limit is None else min(available,vector_limit-seen)
        vectors=np.asarray(layer.vectors[:take])
        _,speeds=vector_settings(vectors,layer.process_ids[:take],config)
        scans=np.linalg.norm(vectors[:,2:4]-vectors[:,:2],axis=1)/speeds
        jumps=np.zeros(take,dtype=float)
        if last_xy is not None:
            jumps[0]=float(np.linalg.norm(vectors[0,:2]-last_xy))/config.jump_speed_m_s+config.jump_delay_s
        if take>1:
            gaps=np.linalg.norm(vectors[1:,:2]-vectors[:-1,2:4],axis=1)
            continuous=(np.asarray(layer.path_ids[1:take])==np.asarray(layer.path_ids[:take-1]))&(gaps<1e-12)
            jumps[1:]=gaps/config.jump_speed_m_s+np.where(continuous,0.,config.jump_delay_s)
        pair_duration=scans+jumps
        total=float(pair_duration.sum())
        budget=remaining_time()
        if total<=budget:
            scan_part=float(scans.sum());jump_part=float(jumps.sum())
            seen+=take;last_xy=vectors[-1,2:4]
            if take<available:stopped=True
        else:
            # A clipped prefix may end during either the jump or scan portion
            # of its final pair. No per-vector Python event list is needed.
            prefix=np.cumsum(pair_duration)
            completed=int(np.searchsorted(prefix,budget,side="right"))
            scan_part=float(scans[:completed].sum());jump_part=float(jumps[:completed].sum())
            before=float(prefix[completed-1]) if completed else 0.
            residual=max(0.,budget-before)
            if completed<take:
                partial_jump=min(residual,float(jumps[completed]))
                partial_scan=max(0.,residual-partial_jump)
                jump_part+=partial_jump;scan_part+=partial_scan
                seen+=completed+int(partial_scan>0)
            else:seen+=completed
            clipped_time=True;stopped=True
        scan_time+=scan_part;jump_time+=jump_part
        block=scan_part+jump_part
        if block>0:exposure_blocks.append(block)
        elapsed+=block
        if time_limit is not None and elapsed>=time_limit:
            clipped_time=True;stopped=True

    # The timeline appends cooldown after a vector cap as well as a full job;
    # the global time cap can consume some or all of that cooldown.
    if has_events and config.final_cooldown_s:
        duration=min(config.final_cooldown_s,remaining_time())
        cooldown_time=duration;elapsed+=duration
        if duration>0:off_blocks.append(duration)
        if duration<config.final_cooldown_s:clipped_time=True

    minimum=_step_bound(exposure_blocks,off_blocks,config.max_event_s)
    exceeded=minimum>config.max_steps
    target=.75*config.max_steps
    suggested=float(config.max_event_s)
    if minimum>target:
        if _step_bound(exposure_blocks,off_blocks,1.)>target:
            suggested=None
        else:
            lo=float(config.max_event_s);hi=1.
            for _ in range(48):
                mid=(lo+hi)/2
                if _step_bound(exposure_blocks,off_blocks,mid)<=target:hi=mid
                else:lo=mid
            suggested=min(1.,math.ceil((hi-1e-12)*1000)/1000)
            # Keep the displayed 1 ms-rounded suggestion on the safe side.
            while suggested<1. and _step_bound(exposure_blocks,off_blocks,suggested)>target:
                suggested=round(suggested+.001,3)
    reason=(f"The selected scan/jump/recoat/cooldown schedule requires at least {minimum:,} thermal steps, exceeding the {config.max_steps:,}-step limit. "
            + (f"A grouping interval of {suggested:g} s reduces this lower bound below 75% of the limit; it still needs numerical sensitivity/convergence checks."
               if suggested is not None else "No grouping interval within the solver's 1 s bound reaches the planning margin; narrow the range or explicitly revise the step budget.")) if exceeded else None
    return dict(scan_time_s=scan_time,jump_time_s=jump_time,interlayer_time_s=interlayer_time,
                cooldown_time_s=cooldown_time,estimated_physical_time_s=elapsed,
                minimum_thermal_steps=minimum,step_limit_exceeded=exceeded,
                suggested_max_event_s=suggested,truncated_estimate=clipped_time or clipped_vectors,
                vector_count=seen,selected_vector_count=selected_count,activated_layer_count=layers_activated,
                time_limit_reached=clipped_time,vector_limit_reached=clipped_vectors,
                min_steps=minimum,feasible=not exceeded,step_limit_feasible=not exceeded,
                reason=reason,failure_reason=reason,
                estimate_kind="Vector-derived selected-prefix timing under configured jump/recoat assumptions; conservative thermal-step bound, not a runtime or convergence estimate",
                numerical_convergence_guaranteed=False,
                planning_note="Frames, nonlinear retries and adaptive integration can require more steps. Grouping changes thermal accuracy and requires sensitivity checks.")


# Shared alias for worker preflight; both names return identical public fields.
workload_plan=estimate_run_plan
