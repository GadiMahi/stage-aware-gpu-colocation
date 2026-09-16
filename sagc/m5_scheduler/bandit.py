"""M5 stretch: a contextual-bandit placement policy.

STATUS: OPTIONAL. Never on the critical path.

The greedy policy is myopic. It accepts any placement predicted to stay within
the slowdown bound, and among admissible options takes the one that damages the
incoming stage least. That is a reasonable heuristic but it ignores two things:
the predictor has error, and some pairings are better than the bound alone
suggests.

A contextual bandit is the right amount of extra machinery here. Full
reinforcement learning would need long-horizon credit assignment, is
sample-inefficient, and is notoriously unstable to tune; a bandit learns the
thing this problem actually needs, which is a per-context correction to the
predicted cost, with a closed-form update and no divergence risk.

    context  features of (incoming stage, resident co-tenants, partition)
    action   which device to place on
    reward   -(observed slowdown - 1), revealed when the stage completes

The model is LinUCB: ridge regression with an optimism bonus. Training happens
inside the simulator, so exploration costs no GPU-hours and an unstable run can
simply be discarded.

If it beats greedy, that is a result: a learned policy captures packing gains a
myopic heuristic misses. If it does not, that is equally reportable: the gains
come from accurate interference prediction rather than cleverer placement.
Either way the paper's spine is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..m1_executor.dag import StageInstance
from ..workloads import registry
from . import policies as pol

D_CONTEXT = 11


def _context(stage: StageInstance, dev: pol.DeviceState,
             predicted: float, thread_pct: int) -> np.ndarray:
    """Feature vector for one candidate placement."""
    spec = registry.get(stage.workload)
    tenants = dev.tenant_names
    if tenants:
        t_occ = float(np.mean([registry.get(t).sim_occupancy for t in tenants]))
        t_bw = float(np.mean([registry.get(t).sim_dram_bw for t in tenants]))
    else:
        t_occ = t_bw = 0.0
    return np.array([
        1.0,
        spec.sim_occupancy,
        spec.sim_dram_bw,
        t_occ,
        t_bw,
        spec.sim_occupancy * t_occ,
        spec.sim_dram_bw * t_bw,
        float(len(tenants)),
        dev.vram_free_mb / max(dev.vram_total_mb, 1),
        predicted - 1.0,
        thread_pct / 100.0,
    ], dtype=float)


@dataclass
class LinUCB:
    """Ridge regression with an upper-confidence bonus."""

    d: int = D_CONTEXT
    alpha: float = 0.35
    lam: float = 1.0
    A: np.ndarray = field(default=None)      # type: ignore[assignment]
    b: np.ndarray = field(default=None)      # type: ignore[assignment]
    n_updates: int = 0

    def __post_init__(self):
        if self.A is None:
            self.A = self.lam * np.eye(self.d)
        if self.b is None:
            self.b = np.zeros(self.d)

    @property
    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def score(self, x: np.ndarray, explore: bool = True) -> float:
        """Estimated reward plus optimism. Higher is better."""
        mean = float(self.theta @ x)
        if not explore:
            return mean
        bonus = self.alpha * float(np.sqrt(max(x @ np.linalg.solve(self.A, x), 0.0)))
        return mean + bonus

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n_updates += 1


class BanditPolicy(pol.Policy):
    """Greedy admission, bandit-guided choice among admissible devices.

    The slowdown bound is still enforced by the predictor: the bandit is never
    allowed to choose a placement the predictor rejects. It only reorders
    preference among options already considered safe, which keeps the safety
    property of the greedy policy intact while letting experience override the
    predictor's ranking.
    """

    name = "bandit"

    def __init__(self, predictor: pol.SlowdownPredictor,
                 slowdown_bound: float = 1.25, max_tenants: int = 2,
                 thread_pct: int = 100, alpha: float = 0.35, explore: bool = True):
        self.predictor = predictor
        self.slowdown_bound = slowdown_bound
        self.max_tenants = max_tenants
        self.thread_pct = thread_pct
        self.explore = explore
        self.model = LinUCB(alpha=alpha)
        self._pending: Dict[str, np.ndarray] = {}
        self.decisions = 0

    def _admissible(self, stage: StageInstance,
                    dev: pol.DeviceState) -> Optional[float]:
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
        out: List[pol.Placement] = []
        for stage in sorted(ready, key=lambda s: -(s.solo_seconds or 0.0)):
            if not stage.gpu:
                out.append((stage, -1, 1.0))
                continue
            options: List[Tuple[float, pol.DeviceState, float, np.ndarray]] = []
            for dev in devices:
                if not dev.fits(stage) or len(dev.resident) >= self.max_tenants:
                    continue
                pred = self._admissible(stage, dev)
                if pred is None:
                    continue
                x = _context(stage, dev, pred, self.thread_pct)
                options.append((self.model.score(x, self.explore), dev, pred, x))
            if not options:
                continue
            options.sort(key=lambda o: -o[0])
            _, dev, pred, x = options[0]
            self._pending[stage.key] = x
            self.decisions += 1
            out.append((stage, dev.index, pred))
            dev.resident.append(stage)
        return out

    def observe(self, stage: StageInstance, actual_slowdown: float) -> None:
        x = self._pending.pop(stage.key, None)
        if x is None:
            return
        # reward is higher when the stage was slowed less
        self.model.update(x, reward=-(float(actual_slowdown) - 1.0))


def train_bandit(predictor: pol.SlowdownPredictor, episodes: int = 12,
                 n_pipelines: int = 20, slowdown_bound: float = 1.25,
                 n_devices: Optional[int] = None,
                 device_vram_mb: Optional[int] = None,
                 ground_truth=None, alpha: float = 0.35, verbose: bool = False):
    """Train inside the simulator across repeated campaigns.

    Costs no GPU-hours, so an unstable run is simply discarded and retried.
    """
    import pandas as pd

    from ..m6_sim import simulator as sim

    policy = BanditPolicy(predictor, slowdown_bound, alpha=alpha, explore=True)
    rows = []
    for ep in range(episodes):
        tr = sim.simulate(policy, n_pipelines=n_pipelines, n_devices=n_devices,
                          device_vram_mb=device_vram_mb, ground_truth=ground_truth,
                          slowdown_bound=slowdown_bound)
        s = tr.summary()
        s["episode"] = ep
        s["updates"] = policy.model.n_updates
        rows.append(s)
        if verbose:
            print(f"  episode {ep:2d}  makespan={s['makespan_s']:9.1f}s  "
                  f"tput={s['throughput_per_hour']:7.2f}/h  "
                  f"viol={s['violation_rate']:.3f}")
    policy.explore = False          # greedy exploitation at evaluation time
    return policy, pd.DataFrame(rows)
