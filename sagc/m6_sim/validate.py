"""M6: validating the simulator against hardware.

The simulator is only worth anything if its answers match reality on the cases
where reality is available. This module runs the same campaign both ways and
reports the disagreement.

ACCEPTANCE CRITERION
--------------------
Simulated makespan within 10 per cent of measured makespan, on the same policy,
pipeline count, device count and solo baselines. That threshold is the module's
completion criterion; outside it, the simulator must not be used for
campaign-scale extrapolation until the discrepancy is explained.

The comparison is driven by MEASURED solo durations, so the simulator is not
being asked to predict how long a stage takes. It is being asked whether its
model of co-location and its scheduling loop reproduce the observed campaign.
That is the only claim being made for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..m1_executor.dag import CampaignTrace
from ..m5_scheduler import policies as pol
from . import simulator as sim

ACCEPTANCE_THRESHOLD = 0.10


@dataclass
class ValidationResult:
    metric_table: pd.DataFrame
    makespan_rel_error: float
    throughput_rel_error: float
    passed: bool
    threshold: float = ACCEPTANCE_THRESHOLD

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (f"simulator validation: {verdict} "
                f"(makespan error {self.makespan_rel_error*100:.1f}%, "
                f"threshold {self.threshold*100:.0f}%)\n"
                + self.metric_table.to_string(index=False))


def _rel(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return abs(a - b) / abs(b)


def solo_baselines_from_trace(trace: CampaignTrace) -> Dict[str, float]:
    """Recover per-workload solo timing from a hardware trace.

    Stages that ran with no co-tenant are solo observations by definition, so a
    campaign trace carries its own baselines wherever the policy left a stage
    alone. Falls back to dividing observed duration by observed slowdown.
    """
    df = trace.stages.copy()
    df["co_tenants"] = df["co_tenants"].fillna("")
    solo_rows = df[(df["co_tenants"] == "") & df["actual_seconds"].notna()]
    out: Dict[str, float] = {}
    for w, grp in solo_rows.groupby("workload"):
        out[w] = float(grp["actual_seconds"].median())
    for w, grp in df.groupby("workload"):
        if w in out:
            continue
        sd = pd.to_numeric(grp["actual_slowdown"], errors="coerce")
        dur = pd.to_numeric(grp["actual_seconds"], errors="coerce")
        est = (dur / sd.replace(0, np.nan)).dropna()
        if not est.empty:
            out[w] = float(est.median())
    return out


def measured_pair_table(pairs: pd.DataFrame) -> Dict[Tuple[str, str, int], float]:
    """Build a ground-truth lookup from the measured pair dataset.

    Feeding this into the simulator makes it replay measured interference rather
    than re-deriving it from the analytic model, which is what makes it
    trace-driven.
    """
    clean = pairs[~pairs["oom"].astype(bool)].copy()
    clean = clean[clean["slowdown_a"].notna()]
    if clean.empty:
        return {}
    grp = clean.groupby(["workload_a", "workload_b",
                         "thread_pct_a"])["slowdown_a"].median()
    return {(a, b, int(p)): float(v) for (a, b, p), v in grp.items()}


def validate(hardware_trace: CampaignTrace, pairs: Optional[pd.DataFrame] = None,
             threshold: float = ACCEPTANCE_THRESHOLD,
             predictor: Optional[pol.SlowdownPredictor] = None) -> ValidationResult:
    """Replay a hardware campaign in the simulator and compare."""
    solo = solo_baselines_from_trace(hardware_trace)
    gt = sim.GroundTruth(hardware_trace.device_vram_mb,
                         measured=measured_pair_table(pairs) if pairs is not None else None)

    kind = hardware_trace.policy
    if kind in ("greedy", "utilisation", "oracle"):
        p = pol.build_policy(kind if kind != "utilisation" else "greedy",
                             predictor or pol.OraclePredictor(hardware_trace.device_vram_mb),
                             hardware_trace.slowdown_bound,
                             device_vram_mb=hardware_trace.device_vram_mb)
        p.name = kind
    else:
        p = pol.build_policy(kind, device_vram_mb=hardware_trace.device_vram_mb)

    sim_trace = sim.simulate(p, n_pipelines=hardware_trace.n_pipelines,
                             n_devices=hardware_trace.n_devices,
                             device_vram_mb=hardware_trace.device_vram_mb,
                             ground_truth=gt,
                             slowdown_bound=hardware_trace.slowdown_bound,
                             solo_seconds=solo)

    hw, sm = hardware_trace.summary(), sim_trace.summary()
    metrics = ["makespan_s", "throughput_per_hour", "mean_achieved_occupancy",
               "held_but_idle_fraction", "violation_rate", "mean_slowdown"]
    tab = pd.DataFrame([{"metric": m, "hardware": hw.get(m), "simulated": sm.get(m),
                         "rel_error": _rel(float(sm.get(m) or 0.0),
                                           float(hw.get(m) or 0.0))}
                        for m in metrics])

    mk = _rel(sim_trace.makespan_s, hardware_trace.makespan_s)
    tp = _rel(sim_trace.throughput_per_hour, hardware_trace.throughput_per_hour)
    return ValidationResult(tab, mk, tp, passed=bool(mk <= threshold),
                            threshold=threshold)


def self_consistency_check(n_pipelines: int = 12,
                           device_vram_mb: int = 15360) -> pd.DataFrame:
    """Sanity checks that must hold for any correct scheduler.

    Run whenever the simulator or a policy changes. These catch the class of bug
    that produces plausible-looking but wrong numbers, which is the dangerous
    kind.
    """
    from ..m5_scheduler import policies as P

    oracle = P.OraclePredictor(device_vram_mb)
    ex = sim.simulate(P.build_policy("exclusive"), n_pipelines=n_pipelines,
                      device_vram_mb=device_vram_mb)
    bl = sim.simulate(P.build_policy("blind"), n_pipelines=n_pipelines,
                      device_vram_mb=device_vram_mb)
    tight = sim.simulate(P.build_policy("greedy", oracle, 1.05),
                         n_pipelines=n_pipelines, device_vram_mb=device_vram_mb,
                         slowdown_bound=1.05)
    loose = sim.simulate(P.build_policy("greedy", oracle, 1.50),
                         n_pipelines=n_pipelines, device_vram_mb=device_vram_mb,
                         slowdown_bound=1.50)

    checks = [
        ("every stage completes",
         all(t.stages["state"].eq("done").all() for t in (ex, bl, tight, loose))),
        ("exclusive holds but under-uses the device",
         ex.held_but_under_occupied_fraction > 0.0),
        ("sharing beats exclusive on throughput",
         bl.throughput_per_hour > ex.throughput_per_hour),
        ("exclusive never violates a bound",
         ex.violation_rate(1.0 + 1e-9) == 0.0),
        ("a tight bound yields no more violations than a loose one",
         tight.violation_rate(1.05) <= loose.violation_rate(1.05) + 1e-9),
        ("a tight bound never beats a loose one on occupancy",
         tight.mean_achieved_occupancy <= loose.mean_achieved_occupancy + 1e-6),
        ("blind sharing violates more than bounded greedy",
         bl.violation_rate(1.25) >= tight.violation_rate(1.25)),
    ]
    return pd.DataFrame([{"check": n, "passed": bool(ok)} for n, ok in checks])
