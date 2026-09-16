"""M2: hardware counter sampling.

A background sampler records a device time series at a fixed interval while a
workload runs in the foreground. Three sources are supported, in descending
order of fidelity:

  dcgm       `dcgmi dmon` streaming DCGM_FI_PROF fields. The only source that
             reports true achieved SM OCCUPANCY (field 1003) as well as SM
             activity (1002) and DRAM activity (1005). This is what the project
             actually wants.

  pynvml     NVML utilisation rates. Reports `utilization.gpu`, the coarse
             metric this project argues against: it indicates only that *some*
             kernel was resident during the sampling window, not how much of the
             device was filled. Occupancy is NOT available here and is recorded
             as missing rather than guessed.

  simulated  No GPU. The time series is generated from the workload's roofline
             parameters so the rest of the pipeline can be exercised. Stamped
             SIMULATED.

DCGM FIELD IDS
--------------
  1001  GR_ENGINE_ACTIVE   fraction of time any graphics/compute engine active
  1002  SM_ACTIVE          fraction of SMs with at least one warp resident
  1003  SM_OCCUPANCY       resident warps as a fraction of the maximum
  1005  DRAM_ACTIVE        fraction of cycles the memory interface was active

The distinction between 1002 and 1003 is the heart of the argument: SM_ACTIVE
can read high while SM_OCCUPANCY is low, which is exactly the illusion that
`nvidia-smi` utilisation creates.
"""
from __future__ import annotations

import math
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..common import env

DEFAULT_INTERVAL_MS = 100

DCGM_FIELDS = {
    1001: "gr_engine_active",
    1002: "sm_active",
    1003: "sm_occupancy",
    1005: "dram_active",
}

TIER_DCGM = "dcgm"
TIER_PYNVML = "pynvml"
TIER_SIM = "simulated"


@dataclass
class CounterTrace:
    """A sampled device time series for one run."""

    samples: pd.DataFrame
    tier: str
    device_index: int
    interval_ms: int
    notes: Dict[str, str] = field(default_factory=dict)

    @property
    def has_occupancy(self) -> bool:
        return ("sm_occupancy" in self.samples.columns
                and self.samples["sm_occupancy"].notna().any())

    @property
    def duration_s(self) -> float:
        if self.samples.empty:
            return 0.0
        return float(self.samples["t"].iloc[-1] - self.samples["t"].iloc[0])

    def __len__(self) -> int:
        return len(self.samples)


class _BaseSampler(threading.Thread):
    tier = "base"

    def __init__(self, device_index: int = 0, interval_ms: int = DEFAULT_INTERVAL_MS):
        super().__init__(daemon=True)
        self.device_index = device_index
        self.interval_ms = interval_ms
        self._stop = threading.Event()
        self.rows: List[dict] = []
        self.notes: Dict[str, str] = {}

    def stop(self) -> CounterTrace:
        self._stop.set()
        self.join(timeout=10)
        df = pd.DataFrame(self.rows)
        if not df.empty and "t" in df.columns:
            df["t"] = df["t"] - df["t"].iloc[0]
        return CounterTrace(df, self.tier, self.device_index, self.interval_ms, self.notes)


class DCGMSampler(_BaseSampler):
    """Stream `dcgmi dmon` and parse its output."""

    tier = TIER_DCGM

    def run(self) -> None:
        cmd = ["dcgmi", "dmon",
               "-e", ",".join(str(f) for f in DCGM_FIELDS),
               "-d", str(self.interval_ms),
               "-i", str(self.device_index)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True)
        except (OSError, subprocess.SubprocessError) as exc:
            self.notes["error"] = f"dcgmi launch failed: {exc}"
            return

        order = list(DCGM_FIELDS.values())
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if self._stop.is_set():
                    break
                line = line.strip()
                if not line or line.startswith("#") or line.lower().startswith("entity"):
                    continue
                parts = line.split()
                if len(parts) < 2 + len(order):     # "GPU 0 v1 v2 v3 v4"
                    continue
                row = {"t": time.time()}
                ok = True
                for name, raw in zip(order, parts[-len(order):]):
                    try:
                        row[name] = float(raw)
                    except ValueError:
                        ok = False
                        break
                if ok:
                    self.rows.append(row)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


class PyNVMLSampler(_BaseSampler):
    """NVML utilisation sampling. Deliberately does NOT synthesise occupancy."""

    tier = TIER_PYNVML

    def run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:  # noqa: BLE001
            self.notes["error"] = f"pynvml init failed: {exc}"
            return

        self.notes["limitation"] = (
            "NVML reports utilisation only; achieved SM occupancy is unavailable "
            "at this tier and is recorded as missing.")
        period = self.interval_ms / 1000.0
        try:
            while not self._stop.is_set():
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    m = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.rows.append({
                        "t": time.time(),
                        "gr_engine_active": u.gpu / 100.0,
                        "sm_active": u.gpu / 100.0,   # NVML cannot separate these
                        "sm_occupancy": math.nan,     # genuinely unavailable
                        "dram_active": u.memory / 100.0,
                        "vram_used_mb": m.used / (1024 ** 2),
                        "util_gpu": u.gpu / 100.0,
                    })
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(period)
        finally:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


class SimulatedSampler(_BaseSampler):
    """Generate a plausible time series from a workload's roofline parameters."""

    tier = TIER_SIM

    def __init__(self, spec, device_index: int = 0,
                 interval_ms: int = DEFAULT_INTERVAL_MS,
                 slowdown: float = 1.0, seed: int = 0):
        super().__init__(device_index, interval_ms)
        self.spec = spec
        self.slowdown = slowdown
        self.rng = np.random.default_rng(
            abs(hash((spec.name, seed, round(slowdown, 4)))) % (2**32))
        self.notes["model"] = "analytic; not a hardware measurement"

    def run(self) -> None:
        from ..workloads.registry import reported_utilisation, sm_active_level

        period = self.interval_ms / 1000.0
        spec = self.spec
        util = reported_utilisation(spec)
        smact = sm_active_level(spec)
        t0 = time.time()

        while not self._stop.is_set():
            elapsed = time.time() - t0
            ramp = min(1.0, elapsed / 0.4) if spec.gpu else 0.0
            j = lambda s: float(self.rng.normal(0.0, s))  # noqa: E731
            self.rows.append({
                "t": time.time(),
                "gr_engine_active": max(0.0, min(1.0, (0.97 if spec.gpu else 0.0) * ramp + j(0.02))),
                "sm_active": max(0.0, min(1.0, smact * ramp + j(0.03))),
                "sm_occupancy": max(0.0, min(1.0, spec.sim_occupancy * ramp + j(0.025))),
                "dram_active": max(0.0, min(1.0, spec.sim_dram_bw * ramp + j(0.03))),
                "vram_used_mb": spec.sim_vram_mb * (0.9 + 0.1 * ramp) + j(20.0),
                "util_gpu": max(0.0, min(1.0, util * ramp + j(0.01))),
            })
            time.sleep(period)


def synth_trace(spec, duration_s: float, interval_ms: int = DEFAULT_INTERVAL_MS,
                seed: int = 0, n_cap: int = 4000) -> CounterTrace:
    """Build a simulated trace directly, without sleeping in real time.

    The threaded SimulatedSampler exists for fidelity when someone wants to
    watch a run happen; this function is what the batch pipeline uses, because a
    78-pair sweep of 25-second runs would otherwise take hours of wall clock to
    produce data that is analytic anyway. The *reported* durations are the full
    modelled ones; only the waiting is skipped.
    """
    from ..workloads.registry import reported_utilisation, sm_active_level

    rng = np.random.default_rng(
        abs(hash((spec.name, seed, round(duration_s, 4)))) % (2**32))
    n = int(max(3, min(n_cap, round(duration_s * 1000.0 / interval_ms))))
    t = np.arange(n) * (interval_ms / 1000.0)
    ramp = np.clip(t / 0.4, 0.0, 1.0)
    if not spec.gpu:
        ramp = np.zeros_like(ramp)

    def series(level: float, sigma: float) -> np.ndarray:
        return np.clip(level * ramp + rng.normal(0.0, sigma, n), 0.0, 1.0)

    df = pd.DataFrame({
        "t": t,
        "gr_engine_active": series(0.97 if spec.gpu else 0.0, 0.02),
        "sm_active": series(sm_active_level(spec), 0.03),
        "sm_occupancy": series(spec.sim_occupancy, 0.025),
        "dram_active": series(spec.sim_dram_bw, 0.03),
        "vram_used_mb": np.clip(
            spec.sim_vram_mb * (0.9 + 0.1 * ramp) + rng.normal(0.0, 20.0, n), 0.0, None),
        "util_gpu": series(reported_utilisation(spec), 0.01),
    })
    return CounterTrace(df, TIER_SIM, 0, interval_ms,
                        {"model": "analytic; not a hardware measurement"})


def make_sampler(spec=None, device_index: int = 0,
                 interval_ms: int = DEFAULT_INTERVAL_MS,
                 slowdown: float = 1.0, seed: int = 0) -> _BaseSampler:
    """Pick the best available sampler for this machine."""
    tier = env.detect().profiler_tier()
    if tier == TIER_DCGM:
        return DCGMSampler(device_index, interval_ms)
    if tier == TIER_PYNVML:
        return PyNVMLSampler(device_index, interval_ms)
    if spec is None:
        raise ValueError("simulated sampling requires a WorkloadSpec")
    return SimulatedSampler(spec, device_index, interval_ms, slowdown, seed)


def probe_report() -> str:
    """Human-readable statement of what profiling this machine can actually do."""
    caps = env.detect()
    tier = caps.profiler_tier()
    lines = [f"profiler tier: {tier}"]
    if tier == TIER_DCGM:
        lines.append("  achieved occupancy available (DCGM field 1003). Full signature.")
    elif tier == TIER_PYNVML:
        lines.append("  WARNING: occupancy NOT available. Only coarse utilisation.")
        lines.append("  The counters-vs-utilisation ablation cannot be run at this tier.")
        lines.append("  Install datacenter-gpu-manager or enable profiling counters.")
    elif tier == TIER_SIM:
        lines.append("  no GPU detected. Traces are generated analytically.")
        lines.append("  Nothing produced in this mode is a measurement.")
    else:
        lines.append("  no counters at all; only wall-clock timing is available.")
    if caps.has_nvidia_smi and not caps.has_dcgm:
        lines.append("  hint: `dcgmi` not found; `ncu` present."
                     if caps.has_ncu else "  hint: neither dcgmi nor ncu found.")
    return "\n".join(lines)
