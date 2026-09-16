"""M2: the workload signature vector.

Eighteen features reduce a run's counter time series plus its kernel statistics
to a fixed-length description. This vector is the only thing the slowdown
predictor ever sees about a workload, which is deliberate: the predictor must
infer a workload's *sensitivity* to contention from observable hardware
behaviour, never from privileged knowledge of what the workload is.

    12  counter statistics : {mean, p95, frac_above_half} x
                             {sm_active, sm_occupancy, dram_active, gr_engine_active}
     2  memory footprint   : peak VRAM, mean VRAM
     3  kernel statistics  : count, mean duration, kernel-time / wall-time
     1  transfer volume    : host-to-device + device-to-host bytes

`frac_above_half` is included because mean and p95 alone cannot distinguish a
workload that is steadily half-occupied from one alternating between idle and
saturated. Those two behave very differently under co-location.

The utilisation-only subset used for ablation A1 is defined below. It contains
exactly the information a scheduler would have if it trusted `nvidia-smi`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

COUNTERS = ["sm_active", "sm_occupancy", "dram_active", "gr_engine_active"]
STATS = ["mean", "p95", "frac_above_half"]

#: Full 18-feature signature, in fixed order.
FEATURE_NAMES: List[str] = (
    [f"{c}_{s}" for c in COUNTERS for s in STATS]
    + ["vram_peak_mb", "vram_mean_mb"]
    + ["kernel_count", "kernel_mean_us", "kernel_time_ratio"]
    + ["transfer_mb"]
)

#: Ablation A1: what a utilisation-driven scheduler would have instead.
UTILISATION_ONLY_FEATURES: List[str] = ["util_gpu_mean", "util_gpu_p95", "vram_peak_mb"]

KERNEL_SOURCE_PROFILER = "torch_profiler"
KERNEL_SOURCE_SPEC = "spec_estimate"


@dataclass
class Signature:
    workload: str
    features: Dict[str, float]
    tier: str
    kernel_source: str
    n_samples: int
    wall_seconds: float
    occupancy_available: bool

    def vector(self, names: Optional[List[str]] = None) -> np.ndarray:
        names = names or FEATURE_NAMES
        return np.array([self.features.get(n, np.nan) for n in names], dtype=float)

    def to_row(self) -> dict:
        row = {
            "workload": self.workload,
            "tier": self.tier,
            "kernel_source": self.kernel_source,
            "n_samples": self.n_samples,
            "wall_seconds": self.wall_seconds,
            "occupancy_available": self.occupancy_available,
        }
        row.update(self.features)
        return row


def _stats_for(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": np.nan, "p95": np.nan, "frac_above_half": np.nan}
    peak = float(s.max())
    half = peak / 2.0 if peak > 0 else 0.0
    return {
        "mean": float(s.mean()),
        "p95": float(s.quantile(0.95)),
        "frac_above_half": float((s > half).mean()) if peak > 0 else 0.0,
    }


def extract(
    workload: str,
    trace,
    wall_seconds: float,
    kernel_count: Optional[float] = None,
    kernel_mean_us: Optional[float] = None,
    kernel_total_s: Optional[float] = None,
    transfer_mb: Optional[float] = None,
    kernel_source: str = KERNEL_SOURCE_SPEC,
) -> Signature:
    """Reduce a CounterTrace plus kernel statistics to a Signature."""
    df = trace.samples
    feats: Dict[str, float] = {}

    for counter in COUNTERS:
        st = (_stats_for(df[counter]) if counter in df.columns
              else {"mean": np.nan, "p95": np.nan, "frac_above_half": np.nan})
        for stat in STATS:
            feats[f"{counter}_{stat}"] = st[stat]

    if "vram_used_mb" in df.columns and df["vram_used_mb"].notna().any():
        v = pd.to_numeric(df["vram_used_mb"], errors="coerce").dropna()
        feats["vram_peak_mb"] = float(v.max())
        feats["vram_mean_mb"] = float(v.mean())
    else:
        feats["vram_peak_mb"] = np.nan
        feats["vram_mean_mb"] = np.nan

    feats["kernel_count"] = float(kernel_count) if kernel_count is not None else np.nan
    feats["kernel_mean_us"] = float(kernel_mean_us) if kernel_mean_us is not None else np.nan
    feats["kernel_time_ratio"] = (
        float(min(1.0, kernel_total_s / wall_seconds))
        if (kernel_total_s is not None and wall_seconds > 0) else np.nan)
    feats["transfer_mb"] = float(transfer_mb) if transfer_mb is not None else np.nan

    # utilisation-only features, kept alongside for ablation A1
    if "util_gpu" in df.columns and df["util_gpu"].notna().any():
        u = _stats_for(df["util_gpu"])
    elif "gr_engine_active" in df.columns:
        u = _stats_for(df["gr_engine_active"])   # NVML tier records it here
    else:
        u = {"mean": np.nan, "p95": np.nan}
    feats["util_gpu_mean"] = u["mean"]
    feats["util_gpu_p95"] = u["p95"]

    return Signature(
        workload=workload, features=feats, tier=trace.tier,
        kernel_source=kernel_source, n_samples=len(df), wall_seconds=wall_seconds,
        occupancy_available=bool(trace.has_occupancy),
    )


def from_spec(spec, wall_seconds: Optional[float] = None) -> Signature:
    """Signature computed directly from a workload's roofline parameters.

    Used only where no trace exists (documentation, unit tests, the simulator's
    cold start). Always reports tier="spec".
    """
    from ..workloads.registry import reported_utilisation, sm_active_level

    wall = wall_seconds if wall_seconds is not None else spec.sim_solo_seconds
    feats: Dict[str, float] = {}
    levels = {
        "sm_active": sm_active_level(spec),
        "sm_occupancy": spec.sim_occupancy,
        "dram_active": spec.sim_dram_bw,
        "gr_engine_active": 0.97 if spec.gpu else 0.0,
    }
    for counter, lvl in levels.items():
        feats[f"{counter}_mean"] = lvl
        feats[f"{counter}_p95"] = min(1.0, lvl * 1.12)
        feats[f"{counter}_frac_above_half"] = 1.0 if lvl > 0 else 0.0

    feats["vram_peak_mb"] = float(spec.sim_vram_mb)
    feats["vram_mean_mb"] = float(spec.sim_vram_mb) * 0.94
    iters = spec.iterations
    feats["kernel_count"] = float(spec.sim_kernel_count * iters)
    feats["kernel_mean_us"] = float(spec.sim_kernel_us)
    total_kernel_s = spec.sim_kernel_count * iters * spec.sim_kernel_us / 1e6
    feats["kernel_time_ratio"] = float(min(1.0, total_kernel_s / max(wall, 1e-6)))
    feats["transfer_mb"] = float(spec.sim_h2d_mb)

    u = reported_utilisation(spec)
    feats["util_gpu_mean"] = u
    feats["util_gpu_p95"] = min(1.0, u * 1.01)

    return Signature(
        workload=spec.name, features=feats, tier="spec",
        kernel_source=KERNEL_SOURCE_SPEC, n_samples=0, wall_seconds=wall,
        occupancy_available=True,
    )


def signatures_to_frame(sigs: List[Signature]) -> pd.DataFrame:
    return pd.DataFrame([s.to_row() for s in sigs])
