"""M5: integer linear programming bound on the greedy heuristic.

The greedy policy is what actually runs: it is O(ready x devices), needs no
solver, and decides in microseconds. The question a reviewer will ask is how
much throughput that simplicity costs.

This module answers it by solving the SINGLE-EPOCH placement problem exactly on
small instances and comparing the objective against greedy's choice. It is a
diagnostic, not a scheduler: the ILP is solved one epoch at a time, so it bounds
the per-epoch optimality gap rather than proving anything about the whole
campaign. Saying more than that would be overclaiming.

    maximise   sum over (device, resident-set) of chosen_set_value
    subject to each device takes exactly one resident set
               each ready stage is placed at most once
               VRAM capacity and tenant count per device
               predicted slowdown of every placed stage within the bound

Assignments are enumerated over feasible *sets* per device rather than
individual stages, because a stage's slowdown depends on which other stages
share the device, and a linear objective over individual assignment variables
cannot express that coupling. With two tenants per device and a modest ready
set, the enumeration stays small.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..m1_executor.dag import StageInstance
from . import policies as pol


@dataclass
class EpochComparison:
    n_ready: int
    n_devices: int
    greedy_value: float
    optimal_value: float
    greedy_placed: int
    optimal_placed: int
    solver_status: str

    @property
    def gap(self) -> float:
        """Fraction of the optimal objective greedy leaves on the table."""
        if self.optimal_value <= 0:
            return 0.0
        return float(max(0.0, (self.optimal_value - self.greedy_value)
                         / self.optimal_value))


def _value(stage: StageInstance, slowdown: float) -> float:
    solo = float(stage.solo_seconds or 1.0)
    return 1.0 / max(solo * max(slowdown, 1.0), 1e-9)


def solve_epoch(ready: Sequence[StageInstance], devices: Sequence[pol.DeviceState],
                predictor: pol.SlowdownPredictor, slowdown_bound: float = 1.25,
                max_tenants: int = 2, thread_pct: int = 100,
                time_limit_s: int = 20) -> Tuple[float, int, str]:
    """Exact single-epoch placement. Returns (objective, n_placed, status)."""
    import pulp

    gpu_ready = [s for s in ready if s.gpu]
    if not gpu_ready or not devices:
        return 0.0, 0, "trivial"

    candidates: Dict[int, List[Tuple[Tuple[int, ...], float]]] = {}
    for d in devices:
        opts: List[Tuple[Tuple[int, ...], float]] = [((), 0.0)]
        existing = d.tenant_names
        room = max_tenants - len(existing)
        for size in range(1, room + 1):
            for combo in combinations(range(len(gpu_ready)), size):
                stages = [gpu_ready[i] for i in combo]
                if sum(s.vram_mb for s in stages) + d.vram_used_mb > d.vram_total_mb:
                    continue
                names = [s.workload for s in stages] + existing
                ok, value = True, 0.0
                for idx, s in enumerate(stages):
                    others = [n for j, n in enumerate(names)
                              if not (j == idx)]           # everything but this stage
                    sd = predictor.predict(s.workload, others, thread_pct)
                    if sd > slowdown_bound:
                        ok = False
                        break
                    value += _value(s, sd)
                if ok:
                    opts.append((combo, value))
        candidates[d.index] = opts

    prob = pulp.LpProblem("epoch_placement", pulp.LpMaximize)
    y = {(d_idx, k): pulp.LpVariable(f"y_{d_idx}_{k}", cat="Binary")
         for d_idx, opts in candidates.items() for k in range(len(opts))}

    prob += pulp.lpSum(y[(d, k)] * opts[k][1]
                       for d, opts in candidates.items() for k in range(len(opts)))
    for d, opts in candidates.items():
        prob += pulp.lpSum(y[(d, k)] for k in range(len(opts))) == 1
    for i in range(len(gpu_ready)):
        prob += pulp.lpSum(y[(d, k)] for d, opts in candidates.items()
                           for k in range(len(opts)) if i in opts[k][0]) <= 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s))
    obj = float(pulp.value(prob.objective) or 0.0)
    placed = sum(len(opts[k][0]) for d, opts in candidates.items()
                 for k in range(len(opts))
                 if y[(d, k)].value() and y[(d, k)].value() > 0.5)
    return obj, placed, pulp.LpStatus[status]


def compare_epoch(ready: Sequence[StageInstance], devices: Sequence[pol.DeviceState],
                  predictor: pol.SlowdownPredictor, slowdown_bound: float = 1.25,
                  max_tenants: int = 2, thread_pct: int = 100) -> EpochComparison:
    """Greedy versus exact on the same epoch."""
    g_devices = copy.deepcopy(list(devices))
    greedy = pol.GreedyPredictivePolicy(predictor, slowdown_bound,
                                        max_tenants, thread_pct)
    placements = greedy.place(list(ready), g_devices)
    g_value = sum(_value(s, sd if sd and not np.isnan(sd) else 1.0)
                  for s, _, sd in placements if s.gpu)
    g_placed = sum(1 for s, _, _ in placements if s.gpu)

    try:
        o_value, o_placed, status = solve_epoch(
            ready, copy.deepcopy(list(devices)), predictor,
            slowdown_bound, max_tenants, thread_pct)
    except Exception as exc:  # noqa: BLE001
        o_value, o_placed, status = g_value, g_placed, f"solver-failed:{type(exc).__name__}"

    return EpochComparison(
        n_ready=len([s for s in ready if s.gpu]), n_devices=len(devices),
        greedy_value=g_value, optimal_value=max(o_value, g_value),
        greedy_placed=g_placed, optimal_placed=o_placed, solver_status=status)


def optimality_study(predictor: pol.SlowdownPredictor, device_vram_mb: int = 15360,
                     n_devices: int = 2, ready_sizes: Sequence[int] = (2, 3, 4, 5, 6),
                     slowdown_bound: float = 1.25, trials: int = 8, seed: int = 0):
    """Sample random ready sets and report the greedy optimality gap."""
    import pandas as pd

    from ..workloads import registry

    rng = np.random.default_rng(seed)
    names = registry.zoo_names()
    rows = []
    for size in ready_sizes:
        for t in range(trials):
            ready = []
            for i, w in enumerate(rng.choice(names, size=size, replace=True)):
                spec = registry.get(w)
                ready.append(StageInstance(pipeline_id=i, stage_idx=0, workload=w,
                                           gpu=True, solo_seconds=spec.sim_solo_seconds,
                                           vram_mb=spec.sim_vram_mb))
            devices = [pol.DeviceState(i, device_vram_mb) for i in range(n_devices)]
            c = compare_epoch(ready, devices, predictor, slowdown_bound)
            rows.append({"n_ready": size, "trial": t, "greedy_value": c.greedy_value,
                         "optimal_value": c.optimal_value, "gap": c.gap,
                         "greedy_placed": c.greedy_placed,
                         "optimal_placed": c.optimal_placed, "status": c.solver_status})
    return pd.DataFrame(rows)
