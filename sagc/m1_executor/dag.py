"""M1: the pipeline DAG, campaign traces, and the metrics computed from them.

A pipeline instance is a chain of four stages executed in order. A campaign is
many such instances processed over a bounded set of GPUs. Everything the
evaluation reports derives from a CampaignTrace, so the trace format is the
contract between the hardware executor, the simulator, and the figures.

THE MOTIVATING METRIC
---------------------
`held_but_idle_fraction` is Objective 1. Under whole-device allocation a
pipeline holds a GPU from the moment its first stage starts until its last stage
finishes. Some of that held time is spent on stages that issue no GPU work at
all, and much of the rest at low occupancy. Two figures are reported, because
they answer different questions:

  held_but_idle             fraction of held GPU-seconds with no GPU work
                            resident. Unambiguous, hard to argue with.
  held_but_under_occupied   fraction of held GPU-seconds below an occupancy
                            threshold. Depends on the threshold, so the
                            threshold is reported alongside it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..workloads import registry

STATE_PENDING = "pending"
STATE_READY = "ready"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"

#: Occupancy below this counts as "under-occupied" in the characterisation.
UNDER_OCCUPANCY_THRESHOLD = 0.30


@dataclass
class StageInstance:
    pipeline_id: int
    stage_idx: int
    workload: str
    gpu: bool
    state: str = STATE_PENDING
    device: Optional[int] = None
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    solo_seconds: Optional[float] = None
    actual_seconds: Optional[float] = None
    predicted_slowdown: Optional[float] = None
    actual_slowdown: Optional[float] = None
    co_tenants: List[str] = field(default_factory=list)
    vram_mb: int = 0

    @property
    def key(self) -> str:
        return f"p{self.pipeline_id}s{self.stage_idx}"

    def to_row(self) -> dict:
        d = asdict(self)
        d["co_tenants"] = ",".join(self.co_tenants)
        d["key"] = self.key
        return d


@dataclass
class Pipeline:
    pipeline_id: int
    stages: List[StageInstance]

    @property
    def done(self) -> bool:
        return all(s.state in (STATE_DONE, STATE_FAILED) for s in self.stages)

    def next_ready(self) -> Optional[StageInstance]:
        """The single next stage, since stages are strictly sequential."""
        for s in self.stages:
            if s.state == STATE_DONE:
                continue
            if s.state == STATE_PENDING:
                return s
            return None          # something is already running for this pipeline
        return None


def build_pipeline(pipeline_id: int, specs: Optional[List] = None) -> Pipeline:
    specs = specs or registry.PIPELINE
    stages = [
        StageInstance(pipeline_id=pipeline_id, stage_idx=i, workload=spec.name,
                      gpu=spec.gpu, solo_seconds=spec.sim_solo_seconds,
                      vram_mb=spec.sim_vram_mb)
        for i, spec in enumerate(specs)
    ]
    return Pipeline(pipeline_id, stages)


def build_campaign(n_pipelines: int, specs: Optional[List] = None) -> List[Pipeline]:
    return [build_pipeline(i, specs) for i in range(n_pipelines)]


@dataclass
class CampaignTrace:
    """Everything the evaluation needs from one campaign run."""

    stages: pd.DataFrame
    policy: str
    n_pipelines: int
    n_devices: int
    device_vram_mb: int
    makespan_s: float
    slowdown_bound: float
    backend: str
    wall_seconds: Optional[float] = None
    meta: Dict = field(default_factory=dict)

    # -- primary metrics ----------------------------------------------------
    @property
    def throughput_per_hour(self) -> float:
        if self.makespan_s <= 0:
            return 0.0
        return self.n_pipelines / (self.makespan_s / 3600.0)

    @property
    def gpu_seconds_available(self) -> float:
        return self.makespan_s * self.n_devices

    def held_gpu_seconds(self) -> float:
        """GPU-seconds reserved by pipelines, however used.

        Under exclusive allocation a pipeline holds its device for its whole
        span, including its CPU-only stage. Under co-location a stage holds only
        its own execution window.
        """
        if self.policy == "exclusive":
            spans = self.stages.groupby("pipeline_id").agg(
                s=("start_s", "min"), e=("end_s", "max"))
            return float((spans["e"] - spans["s"]).sum())
        gpu = self.stages[self.stages["gpu"]]
        return float((gpu["end_s"] - gpu["start_s"]).sum())

    def busy_gpu_seconds(self) -> float:
        """GPU-seconds during which GPU work was actually resident."""
        gpu = self.stages[self.stages["gpu"]]
        return float((gpu["end_s"] - gpu["start_s"]).sum())

    def occupancy_weighted_gpu_seconds(self) -> float:
        gpu = self.stages[self.stages["gpu"]].copy()
        occ = gpu["workload"].map(lambda w: registry.get(w).sim_occupancy)
        return float(((gpu["end_s"] - gpu["start_s"]) * occ).sum())

    @property
    def held_but_idle_fraction(self) -> float:
        held = self.held_gpu_seconds()
        if held <= 0:
            return 0.0
        return float(max(0.0, 1.0 - self.busy_gpu_seconds() / held))

    @property
    def held_but_under_occupied_fraction(self) -> float:
        """Held GPU-seconds spent idle or below the occupancy threshold."""
        held = self.held_gpu_seconds()
        if held <= 0:
            return 0.0
        gpu = self.stages[self.stages["gpu"]].copy()
        occ = gpu["workload"].map(lambda w: registry.get(w).sim_occupancy)
        well_used = float((
            (gpu["end_s"] - gpu["start_s"]) * (occ >= UNDER_OCCUPANCY_THRESHOLD)).sum())
        return float(max(0.0, 1.0 - well_used / held))

    @property
    def mean_achieved_occupancy(self) -> float:
        """Time-weighted occupancy across all available GPU-seconds."""
        avail = self.gpu_seconds_available
        if avail <= 0:
            return 0.0
        return float(self.occupancy_weighted_gpu_seconds() / avail)

    def violation_rate(self, bound: Optional[float] = None) -> float:
        bound = bound if bound is not None else self.slowdown_bound
        gpu = self.stages[self.stages["gpu"]]
        sd = pd.to_numeric(gpu["actual_slowdown"], errors="coerce").dropna()
        if sd.empty:
            return 0.0
        return float((sd > bound + 1e-9).mean())

    def p95_stage_latency(self) -> float:
        gpu = self.stages[self.stages["gpu"]]
        d = pd.to_numeric(gpu["actual_seconds"], errors="coerce").dropna()
        return float(d.quantile(0.95)) if not d.empty else float("nan")

    def mean_slowdown(self) -> float:
        gpu = self.stages[self.stages["gpu"]]
        sd = pd.to_numeric(gpu["actual_slowdown"], errors="coerce").dropna()
        return float(sd.mean()) if not sd.empty else 1.0

    def summary(self) -> Dict:
        return {
            "policy": self.policy,
            "backend": self.backend,
            "n_pipelines": self.n_pipelines,
            "n_devices": self.n_devices,
            "slowdown_bound": self.slowdown_bound,
            "makespan_s": round(self.makespan_s, 2),
            "throughput_per_hour": round(self.throughput_per_hour, 3),
            "mean_achieved_occupancy": round(self.mean_achieved_occupancy, 4),
            "held_but_idle_fraction": round(self.held_but_idle_fraction, 4),
            "held_but_under_occupied_fraction": round(
                self.held_but_under_occupied_fraction, 4),
            "violation_rate": round(self.violation_rate(), 4),
            "mean_slowdown": round(self.mean_slowdown(), 4),
            "p95_stage_latency_s": round(self.p95_stage_latency(), 2),
        }

    def save(self, directory: str | Path, tag: str = "") -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        tag = tag or f"{self.policy}_{self.n_pipelines}p"
        self.stages.to_csv(d / f"trace_{tag}.csv", index=False)
        (d / f"summary_{tag}.json").write_text(json.dumps(self.summary(), indent=2))
        return d / f"trace_{tag}.csv"


def trace_from_stages(stages: List[StageInstance], policy: str, n_devices: int,
                      device_vram_mb: int, slowdown_bound: float, backend: str,
                      wall_seconds: Optional[float] = None,
                      meta: Optional[Dict] = None) -> CampaignTrace:
    df = pd.DataFrame([s.to_row() for s in stages])
    ends = pd.to_numeric(df["end_s"], errors="coerce").dropna()
    starts = pd.to_numeric(df["start_s"], errors="coerce").dropna()
    makespan = float(ends.max() - starts.min()) if not ends.empty else 0.0
    return CampaignTrace(
        stages=df, policy=policy, n_pipelines=int(df["pipeline_id"].nunique()),
        n_devices=n_devices, device_vram_mb=device_vram_mb, makespan_s=makespan,
        slowdown_bound=slowdown_bound, backend=backend, wall_seconds=wall_seconds,
        meta=meta or {},
    )
