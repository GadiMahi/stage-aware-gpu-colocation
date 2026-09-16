"""M5: placement policies.

A policy answers one question at each scheduling epoch: given the stages whose
predecessors have completed, and the current residency of each device, which
stages should start now and where?

    maximise   sum over placed stages of 1 / predicted_duration
    subject to predicted_slowdown(s) <= S_max  for every placed stage
               sum of VRAM on any device <= device capacity
               pipeline precedence (enforced by construction: only ready stages
               are ever offered to a policy)

Five policies implement the evaluation's baselines plus the contribution:

    exclusive     one pipeline owns a whole device for its entire lifetime.
                  The status quo, and the thing to beat.
    blind         pack to VRAM capacity, ignore interference entirely.
                  The naive alternative.
    utilisation   the greedy policy, but its predictor sees only coarse
                  utilisation features. Isolates the value of counter features.
    greedy        the contribution: greedy placement under a learned slowdown
                  predictor with a per-stage bound.
    oracle        greedy placement using TRUE slowdown rather than predicted.
                  Upper bound; measures how much of the achievable gain the
                  predictor captures.

A contextual-bandit policy lives in `bandit.py`; it is a stretch item and never
on the critical path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..m1_executor.dag import StageInstance
from ..workloads import interference, registry


class SlowdownPredictor(Protocol):
    """Anything that can estimate how much a stage will be slowed."""

    name: str

    def predict(self, target: str, co_tenants: Sequence[str],
                thread_pct: int = 100) -> float:
        ...


class OraclePredictor:
    """Ground truth. On hardware this is a lookup into measured data."""

    name = "oracle"

    def __init__(self, device_vram_mb: int = 15360):
        self.device_vram_mb = device_vram_mb

    def predict(self, target: str, co_tenants: Sequence[str],
                thread_pct: int = 100) -> float:
        if not co_tenants:
            return 1.0
        return interference.predict_multi(
            registry.get(target), [registry.get(c) for c in co_tenants],
            thread_pct=thread_pct, device_vram_mb=self.device_vram_mb, noise=False)


class TablePredictor:
    """Predictor backed by a precomputed pairwise table.

    This is how the learned model is actually consumed at runtime. Because stage
    identity is known in advance, every pairing is evaluated offline once and
    looked up in constant time, so model inference never sits in the scheduling
    path. Multi-tenant residency falls back to the maximum pairwise prediction,
    which is conservative.
    """

    name = "table"

    def __init__(self, table: Dict[Tuple[str, str, int], float],
                 default: float = 1.0, aggregate: str = "max"):
        self.table = table
        self.default = default
        self.aggregate = aggregate
        self.lookups = 0

    def predict(self, target: str, co_tenants: Sequence[str],
                thread_pct: int = 100) -> float:
        if not co_tenants:
            return 1.0
        self.lookups += 1
        vals = [self.table.get((target, c, thread_pct),
                               self.table.get((target, c, 100), self.default))
                for c in co_tenants]
        if not vals:
            return self.default
        if self.aggregate == "sum":
            return float(1.0 + sum(v - 1.0 for v in vals))
        return float(max(vals))


@dataclass
class DeviceState:
    index: int
    vram_total_mb: int
    resident: List[StageInstance] = field(default_factory=list)
    reserved_by_pipeline: Optional[int] = None   # exclusive allocation only

    @property
    def vram_used_mb(self) -> int:
        return sum(s.vram_mb for s in self.resident)

    @property
    def vram_free_mb(self) -> int:
        return self.vram_total_mb - self.vram_used_mb

    def fits(self, stage: StageInstance) -> bool:
        return stage.vram_mb <= self.vram_free_mb

    @property
    def tenant_names(self) -> List[str]:
        return [s.workload for s in self.resident]


Placement = Tuple[StageInstance, int, float]   # stage, device index, predicted slowdown


class Policy:
    name = "base"
    thread_pct = 100

    def place(self, ready: List[StageInstance],
              devices: List[DeviceState]) -> List[Placement]:
        raise NotImplementedError

    def observe(self, stage: StageInstance, actual_slowdown: float) -> None:
        """Optional feedback hook; only the bandit policy uses it."""
        return None


class ExclusivePolicy(Policy):
    """Whole-device allocation. One pipeline owns a GPU until it finishes.

    Note that the CPU-only stage still occupies the device slot. That is not a
    modelling artefact: under whole-device allocation the GPU genuinely is held
    while featurisation runs on the host, and that held-but-idle time is exactly
    what Objective 1 measures.
    """

    name = "exclusive"

    def place(self, ready, devices):
        out: List[Placement] = []
        for stage in ready:
            dev = next((d for d in devices
                        if d.reserved_by_pipeline == stage.pipeline_id), None)
            if dev is None:
                dev = next((d for d in devices
                            if d.reserved_by_pipeline is None and not d.resident), None)
                if dev is None:
                    continue
                dev.reserved_by_pipeline = stage.pipeline_id
            if dev.resident:
                continue                      # one stage at a time per pipeline
            out.append((stage, dev.index, 1.0))
            dev.resident.append(stage)
        return out


class BlindSharingPolicy(Policy):
    """Pack to capacity, ignore interference. Efficient until it is not."""

    name = "blind"

    def __init__(self, max_tenants: int = 2):
        self.max_tenants = max_tenants

    def place(self, ready, devices):
        out: List[Placement] = []
        for stage in ready:
            if not stage.gpu:
                out.append((stage, -1, 1.0))   # CPU stage, no device needed
                continue
            cands = [d for d in devices
                     if d.fits(stage) and len(d.resident) < self.max_tenants]
            if not cands:
                continue
            dev = min(cands, key=lambda d: (len(d.resident), -d.vram_free_mb))
            out.append((stage, dev.index, float("nan")))
            dev.resident.append(stage)
        return out


class GreedyPredictivePolicy(Policy):
    """The contribution: greedy placement under a predicted slowdown bound.

    Ready stages are considered longest-first, because a long stage placed badly
    costs more than a short one, and greedy algorithms of this shape behave
    better when the large items are placed while choice remains.

    A placement is accepted only if the predicted slowdown of BOTH the incoming
    stage and every stage already resident stays within the bound. Checking the
    incumbent matters: admitting a new tenant that wrecks a running stage is
    exactly the failure mode blind sharing exhibits.
    """

    name = "greedy"

    def __init__(self, predictor: SlowdownPredictor, slowdown_bound: float = 1.25,
                 max_tenants: int = 2, thread_pct: int = 100):
        self.predictor = predictor
        self.slowdown_bound = slowdown_bound
        self.max_tenants = max_tenants
        self.thread_pct = thread_pct

    def _score(self, stage: StageInstance, dev: DeviceState) -> Optional[float]:
        """Predicted slowdown of `stage` on `dev`, or None if inadmissible."""
        tenants = dev.tenant_names
        s_new = self.predictor.predict(stage.workload, tenants, self.thread_pct)
        if s_new > self.slowdown_bound:
            return None
        for incumbent in dev.resident:
            others = [t for t in tenants if t != incumbent.workload] + [stage.workload]
            if self.predictor.predict(incumbent.workload, others,
                                      self.thread_pct) > self.slowdown_bound:
                return None
        return s_new

    def place(self, ready, devices):
        out: List[Placement] = []
        for stage in sorted(ready, key=lambda s: -(s.solo_seconds or 0.0)):
            if not stage.gpu:
                out.append((stage, -1, 1.0))
                continue
            best: Optional[Tuple[float, DeviceState]] = None
            for dev in devices:
                if not dev.fits(stage) or len(dev.resident) >= self.max_tenants:
                    continue
                s = self._score(stage, dev)
                if s is None:
                    continue
                # prefer the device that damages this stage least; break ties
                # toward the emptier device so capacity stays available
                if best is None or (s, len(dev.resident)) < (best[0], len(best[1].resident)):
                    best = (s, dev)
            if best is None:
                continue
            pred, dev = best
            out.append((stage, dev.index, pred))
            dev.resident.append(stage)
        return out


class UtilisationPolicy(GreedyPredictivePolicy):
    """Identical policy, cruder predictor. Ablation A1 made executable."""

    name = "utilisation"


def build_policy(kind: str, predictor: Optional[SlowdownPredictor] = None,
                 slowdown_bound: float = 1.25, max_tenants: int = 2,
                 device_vram_mb: int = 15360, thread_pct: int = 100) -> Policy:
    kind = kind.lower()
    if kind == "exclusive":
        return ExclusivePolicy()
    if kind == "blind":
        return BlindSharingPolicy(max_tenants=max_tenants)
    if kind == "oracle":
        p = GreedyPredictivePolicy(predictor or OraclePredictor(device_vram_mb),
                                   slowdown_bound, max_tenants, thread_pct)
        p.name = "oracle"
        return p
    if kind == "utilisation":
        if predictor is None:
            raise ValueError("utilisation policy needs a utilisation-fed predictor")
        p = UtilisationPolicy(predictor, slowdown_bound, max_tenants, thread_pct)
        p.name = "utilisation"
        return p
    if kind == "greedy":
        if predictor is None:
            raise ValueError("greedy policy needs a predictor")
        return GreedyPredictivePolicy(predictor, slowdown_bound, max_tenants, thread_pct)
    raise ValueError(f"unknown policy {kind!r}")


POLICY_KINDS = ["exclusive", "blind", "utilisation", "greedy", "oracle"]
