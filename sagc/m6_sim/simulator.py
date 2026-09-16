"""M6: trace-driven campaign simulator.

A discrete-event simulator over a virtual clock. It exists for two reasons:

  1. Campaign scale. A 200-pipeline campaign on two T4s would consume far more
     GPU-hours than the project's budget allows. The simulator replays measured
     solo durations and measured (or predicted) slowdowns to reach that scale,
     and is validated against real hardware runs so the extrapolation is
     defensible rather than decorative.
  2. Policy development. Five policies across five slowdown bounds is
     twenty-five campaigns. Iterating on that against hardware would be
     intolerable; iterating in simulation is instant and costs nothing.

FLUID MODEL
-----------
Naively, one could fix a stage's slowdown at the moment it is placed. That is
wrong: residency changes while a stage runs, so a stage that starts alone and
acquires a co-tenant halfway through is not slowed uniformly. The simulator
tracks each running stage's REMAINING SOLO WORK and consumes it at a rate of
1/slowdown, recomputing slowdown whenever residency on that device changes. A
stage's reported `actual_slowdown` is total wall time divided by solo time,
which is what a real measurement would show.

Nothing produced here is a hardware measurement; traces are stamped SIMULATED.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common import env
from ..m1_executor import dag
from ..m1_executor.dag import (STATE_DONE, STATE_PENDING, STATE_RUNNING,
                               CampaignTrace, Pipeline, StageInstance)
from ..m5_scheduler import policies as pol
from ..workloads import interference, registry

EPS = 1e-9
MAX_EVENTS = 2_000_000


@dataclass
class _Running:
    stage: StageInstance
    remaining_solo_s: float
    wall_elapsed_s: float = 0.0


class GroundTruth:
    """Supplies the slowdown a stage ACTUALLY experiences.

    In simulation this is the analytic interference model. When validating
    against hardware it is replaced by a lookup into measured pair data, so the
    simulator replays reality rather than re-deriving it.
    """

    def __init__(self, device_vram_mb: int = 15360,
                 measured: Optional[Dict[Tuple[str, str, int], float]] = None,
                 noise: bool = False):
        self.device_vram_mb = device_vram_mb
        self.measured = measured or {}
        self.noise = noise

    def slowdown(self, target: str, co_tenants: Sequence[str],
                 thread_pct: int = 100) -> float:
        if not co_tenants:
            return 1.0
        if self.measured:
            vals = [self.measured.get((target, c, thread_pct),
                                      self.measured.get((target, c, 100)))
                    for c in co_tenants]
            vals = [v for v in vals if v is not None]
            if vals:
                return float(max(vals))
        return interference.predict_multi(
            registry.get(target), [registry.get(c) for c in co_tenants],
            thread_pct=thread_pct, device_vram_mb=self.device_vram_mb,
            noise=self.noise)


def simulate(policy: pol.Policy, n_pipelines: int = 20,
             n_devices: Optional[int] = None,
             device_vram_mb: Optional[int] = None,
             ground_truth: Optional[GroundTruth] = None,
             slowdown_bound: float = 1.25,
             solo_seconds: Optional[Dict[str, float]] = None,
             specs: Optional[List] = None,
             max_clock_s: float = 1e7) -> CampaignTrace:
    """Run one campaign under `policy` and return its trace.

    `solo_seconds` lets measured per-workload solo timings replace the modelled
    ones, which is what makes this trace-driven rather than purely synthetic.
    """
    n_devices = n_devices if n_devices is not None else env.n_devices()
    device_vram_mb = (device_vram_mb if device_vram_mb is not None
                      else env.device_memory_mb())
    gt = ground_truth or GroundTruth(device_vram_mb)
    solo_seconds = solo_seconds or {}

    pipelines: List[Pipeline] = dag.build_campaign(n_pipelines, specs)
    for p in pipelines:
        for s in p.stages:
            if s.workload in solo_seconds:
                s.solo_seconds = float(solo_seconds[s.workload])

    devices = [pol.DeviceState(i, device_vram_mb) for i in range(n_devices)]
    running: List[_Running] = []
    cpu_running: List[_Running] = []      # CPU stages that hold no device slot
    clock = 0.0
    events = 0

    def ready_stages() -> List[StageInstance]:
        out = []
        for p in pipelines:
            s = p.next_ready()
            if s is not None and s.state == STATE_PENDING:
                out.append(s)
        return out

    def current_slowdown(r: _Running) -> float:
        st = r.stage
        if not st.gpu or st.device is None or st.device < 0:
            return 1.0
        dev = devices[st.device]
        others = [x.workload for x in dev.resident if x.key != st.key]
        return max(1.0, gt.slowdown(st.workload, others, policy.thread_pct))

    while events < MAX_EVENTS and clock < max_clock_s:
        events += 1

        # ---- 1. offer ready stages to the policy ---------------------------
        ready = ready_stages()
        if ready:
            for stage, dev_idx, predicted in policy.place(ready, devices):
                stage.state = STATE_RUNNING
                stage.device = dev_idx
                stage.start_s = clock
                stage.predicted_slowdown = (
                    None if predicted is None or np.isnan(predicted)
                    else float(predicted))
                r = _Running(stage, remaining_solo_s=float(stage.solo_seconds or 0.0))
                if stage.gpu and dev_idx >= 0:
                    stage.co_tenants = [x.workload for x in devices[dev_idx].resident
                                        if x.key != stage.key]
                    running.append(r)
                else:
                    cpu_running.append(r)

        # ---- 2. termination -------------------------------------------------
        if not running and not cpu_running:
            if all(p.done for p in pipelines):
                break
            if not ready:
                break                       # nothing runnable: deadlock guard
            continue

        # ---- 3. how long until the next completion --------------------------
        def finish_in(r: _Running) -> float:
            if not r.stage.gpu:
                return max(r.remaining_solo_s, EPS)
            return max(r.remaining_solo_s * current_slowdown(r), EPS)

        dt = max(min(finish_in(r) for r in running + cpu_running), EPS)

        # ---- 4. advance the clock, consuming work at the current rate --------
        for r in running:
            r.remaining_solo_s -= dt / current_slowdown(r)
            r.wall_elapsed_s += dt
        for r in cpu_running:
            r.remaining_solo_s -= dt
            r.wall_elapsed_s += dt
        clock += dt

        # ---- 5. retire finished stages ---------------------------------------
        for r in [x for x in running + cpu_running if x.remaining_solo_s <= EPS]:
            st = r.stage
            st.state = STATE_DONE
            st.end_s = clock
            st.actual_seconds = r.wall_elapsed_s
            solo = float(st.solo_seconds or 0.0)
            st.actual_slowdown = (r.wall_elapsed_s / solo) if solo > 0 else 1.0
            policy.observe(st, st.actual_slowdown)
            # Release ANY stage that occupied a device slot, CPU-only stages
            # under exclusive allocation included: those hold the device without
            # using it, which is the waste being measured, but they must still be
            # released when they finish or the campaign deadlocks.
            if st.device is not None and st.device >= 0:
                d = devices[st.device]
                d.resident = [x for x in d.resident if x.key != st.key]
        running = [r for r in running if r.remaining_solo_s > EPS]
        cpu_running = [r for r in cpu_running if r.remaining_solo_s > EPS]

        # ---- 6. release exclusive reservations for completed pipelines --------
        for dev in devices:
            pid = dev.reserved_by_pipeline
            if pid is not None and pipelines[pid].done and not dev.resident:
                dev.reserved_by_pipeline = None

    return dag.trace_from_stages(
        [s for p in pipelines for s in p.stages],
        policy=policy.name, n_devices=n_devices, device_vram_mb=device_vram_mb,
        slowdown_bound=slowdown_bound, backend=env.BACKEND_SIM,
        meta={"events": events,
              "ground_truth": "measured" if gt.measured else "analytic"})


def sweep_policies(predictor: Optional[pol.SlowdownPredictor] = None,
                   util_predictor: Optional[pol.SlowdownPredictor] = None,
                   n_pipelines: int = 20,
                   bounds: Sequence[float] = (1.05, 1.10, 1.25, 1.50, 999.0),
                   kinds: Sequence[str] = ("exclusive", "blind", "utilisation",
                                           "greedy", "oracle"),
                   n_devices: Optional[int] = None,
                   device_vram_mb: Optional[int] = None,
                   ground_truth: Optional[GroundTruth] = None,
                   solo_seconds: Optional[Dict[str, float]] = None,
                   verbose: bool = False):
    """Every policy against every slowdown bound. The evaluation matrix."""
    import pandas as pd

    device_vram_mb = (device_vram_mb if device_vram_mb is not None
                      else env.device_memory_mb())
    traces: List[CampaignTrace] = []
    rows = []

    for kind in kinds:
        # exclusive and blind ignore the bound, so run them once
        these = [bounds[0]] if kind in ("exclusive", "blind") else list(bounds)
        for bound in these:
            if kind == "greedy":
                p = pol.build_policy("greedy", predictor, bound,
                                     device_vram_mb=device_vram_mb)
            elif kind == "utilisation":
                if util_predictor is None:
                    continue
                p = pol.build_policy("utilisation", util_predictor, bound,
                                     device_vram_mb=device_vram_mb)
            elif kind == "oracle":
                p = pol.build_policy("oracle", None, bound,
                                     device_vram_mb=device_vram_mb)
            else:
                p = pol.build_policy(kind, device_vram_mb=device_vram_mb)

            tr = simulate(p, n_pipelines=n_pipelines, n_devices=n_devices,
                          device_vram_mb=device_vram_mb, ground_truth=ground_truth,
                          slowdown_bound=bound, solo_seconds=solo_seconds)
            traces.append(tr)
            row = tr.summary()
            row["bound_label"] = "none" if bound >= 100 else f"{bound:.2f}"
            rows.append(row)
            if verbose:
                print(f"  {kind:12s} bound={row['bound_label']:>5s}  "
                      f"makespan={row['makespan_s']:9.1f}s  "
                      f"tput={row['throughput_per_hour']:7.2f}/h  "
                      f"occ={row['mean_achieved_occupancy']:.3f}  "
                      f"viol={row['violation_rate']:.3f}")

    return traces, pd.DataFrame(rows)
