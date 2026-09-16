"""Analytic co-location interference model.

WHAT THIS IS FOR
----------------
On real hardware, slowdown is measured: run A alone, run A beside B, divide.
This module is *not* used on that path. It exists for two legitimate purposes:

  1. Pipeline validation. Before spending GPU-hours, the entire analysis chain
     (profiler -> dataset -> model -> scheduler -> figures) can be exercised end
     to end against a ground truth known in closed form. If the learned model
     cannot recover a relationship that is genuinely there, the bug is in the
     pipeline, not the hardware.
  2. Campaign-scale extrapolation in M6, where replaying measured stage
     durations at a scale larger than the GPU budget allows requires some model
     of what co-location would have done.

WHAT IT IS NOT
--------------
It is not a measurement and is never presented as one. Everything derived from
it is stamped SIMULATED by sagc.common.provenance.

THE FORM OF THE MODEL
---------------------
The decomposition follows the contention sources named by Prophet (ASPLOS'17),
expressed in the pressure/sensitivity framing of Bubble-Up (MICRO'11). Each
tenant contributes *pressure* on a shared resource and possesses a *sensitivity*
to pressure from the other. Slowdown is therefore asymmetric: the same pair can
hurt one tenant far more than the other.

    slowdown_A = 1
               + sens_bw_A      * bandwidth_shortfall(A, B)
               + sens_compute_A * sm_shortfall(A, B)
               + sens_cache_A   * cache_pressure(A, B)
               + mps_throttle(A, thread_fraction_A)
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .registry import WorkloadSpec

# --- model constants --------------------------------------------------------
# Chosen so the worst pairings land near 2x and benign pairings near 1.05x,
# which is the range reported across the MPS co-location literature.
K_BANDWIDTH = 1.45
K_COMPUTE = 1.25
K_CACHE = 0.55
K_MPS_THROTTLE = 0.85
MAX_SLOWDOWN = 6.0

#: Multiplicative run-to-run noise (sigma), matching the few-per-cent variance
#: seen between repeat runs on a shared cloud GPU.
NOISE_SIGMA = 0.025


@dataclass(frozen=True)
class Outcome:
    """Result of co-locating two workloads."""

    slowdown_a: float
    slowdown_b: float
    oom: bool
    vram_required_mb: int
    vram_available_mb: int

    @property
    def feasible(self) -> bool:
        return not self.oom


def _pressure(spec: WorkloadSpec) -> Tuple[float, float]:
    """(bandwidth pressure, SM pressure) exerted by a resident workload."""
    return spec.sim_dram_bw, spec.sim_occupancy


def _bandwidth_shortfall(bw_self: float, bw_other: float) -> float:
    """Relative bandwidth lost when combined demand exceeds the memory pipe.

    Below saturation there is a small but non-zero cost from request
    interleaving; above saturation the shortfall grows with the excess.
    """
    total = bw_self + bw_other
    if total <= 1.0:
        return 0.12 * bw_self * bw_other          # sub-saturation interleaving
    return (total - 1.0) + 0.12 * bw_self * bw_other


def _sm_shortfall(occ_self: float, occ_other: float) -> float:
    """Relative scheduling slots lost when combined occupancy exceeds the device."""
    total = occ_self + occ_other
    if total <= 1.0:
        return 0.05 * occ_self * occ_other
    return (total - 1.0) + 0.05 * occ_self * occ_other


def _cache_pressure(bw_self: float, bw_other: float) -> float:
    """Last-level cache thrash. Worst when both tenants stream heavily."""
    return bw_self * bw_other


def _mps_throttle(occ_self: float, thread_fraction: float) -> float:
    """Cost of MPS capping a tenant below the occupancy it wants.

    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE limits how many SMs a client may use. A
    workload wanting more than its allowance is throttled proportionally.
    """
    if thread_fraction >= 1.0 or occ_self <= 0:
        return 0.0
    if occ_self <= thread_fraction:
        return 0.0
    return (occ_self / max(thread_fraction, 1e-6)) - 1.0


def _deterministic_noise(seed_key: str, sigma: float) -> float:
    """Reproducible pseudo-noise keyed on the pair and configuration.

    Deterministic rather than random so repeated calls with the same inputs
    agree, which keeps the simulator reproducible across processes.
    """
    if sigma <= 0:
        return 1.0
    h = hashlib.sha256(seed_key.encode()).digest()
    u1 = (int.from_bytes(h[0:4], "big") + 1) / (2**32 + 1)
    u2 = (int.from_bytes(h[4:8], "big") + 1) / (2**32 + 1)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return max(0.5, 1.0 + sigma * z)


def predict_pair(
    a: WorkloadSpec,
    b: WorkloadSpec,
    thread_pct_a: int = 100,
    thread_pct_b: int = 100,
    device_vram_mb: int = 15360,
    noise: bool = True,
    repetition: int = 0,
) -> Outcome:
    """Ground-truth slowdown for co-locating `a` and `b`.

    Returns slowdowns >= 1.0, or an OOM outcome if the pair cannot fit.
    """
    vram_required = a.sim_vram_mb + b.sim_vram_mb
    if vram_required > device_vram_mb:
        return Outcome(
            slowdown_a=float("nan"), slowdown_b=float("nan"), oom=True,
            vram_required_mb=vram_required, vram_available_mb=device_vram_mb,
        )

    bw_a, occ_a = _pressure(a)
    bw_b, occ_b = _pressure(b)
    ta, tb = thread_pct_a / 100.0, thread_pct_b / 100.0

    def one(spec, bw_self, occ_self, bw_other, occ_other, tfrac, tag):
        s = 1.0
        s += spec.sens_bandwidth * K_BANDWIDTH * _bandwidth_shortfall(bw_self, bw_other)
        s += spec.sens_compute * K_COMPUTE * _sm_shortfall(occ_self, occ_other)
        s += spec.sens_cache * K_CACHE * _cache_pressure(bw_self, bw_other)
        s += K_MPS_THROTTLE * _mps_throttle(occ_self, tfrac)
        if noise:
            key = f"{a.name}|{b.name}|{thread_pct_a}|{thread_pct_b}|{repetition}|{tag}"
            s *= _deterministic_noise(key, NOISE_SIGMA)
        return float(min(max(s, 1.0), MAX_SLOWDOWN))

    return Outcome(
        slowdown_a=one(a, bw_a, occ_a, bw_b, occ_b, ta, "a"),
        slowdown_b=one(b, bw_b, occ_b, bw_a, occ_a, tb, "b"),
        oom=False,
        vram_required_mb=vram_required,
        vram_available_mb=device_vram_mb,
    )


def predict_multi(
    target: WorkloadSpec,
    co_tenants: Sequence[WorkloadSpec],
    thread_pct: int = 100,
    device_vram_mb: int = 15360,
    noise: bool = False,
) -> float:
    """Slowdown of `target` against an arbitrary number of co-tenants.

    The dataset only ever measures pairs, so three-way and higher residency is
    estimated by aggregating co-tenant pressure. This is an approximation; the
    scheduler is configured to prefer pairs, and it exists so the simulator does
    not fail when a policy packs three stages onto one device.
    """
    if not co_tenants:
        return 1.0
    bw_other = min(1.0, sum(c.sim_dram_bw for c in co_tenants))
    occ_other = min(1.0, sum(c.sim_occupancy for c in co_tenants))
    bw_self, occ_self = _pressure(target)
    tfrac = thread_pct / 100.0

    s = 1.0
    s += target.sens_bandwidth * K_BANDWIDTH * _bandwidth_shortfall(bw_self, bw_other)
    s += target.sens_compute * K_COMPUTE * _sm_shortfall(occ_self, occ_other)
    s += target.sens_cache * K_CACHE * _cache_pressure(bw_self, bw_other)
    s += K_MPS_THROTTLE * _mps_throttle(occ_self, tfrac)
    if noise:
        key = "|".join([target.name] + sorted(c.name for c in co_tenants))
        s *= _deterministic_noise(key, NOISE_SIGMA)
    return float(min(max(s, 1.0), MAX_SLOWDOWN))


def fits(specs: List[WorkloadSpec], device_vram_mb: int) -> bool:
    return sum(s.sim_vram_mb for s in specs) <= device_vram_mb
