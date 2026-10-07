"""M2: execute a workload under profiling.

`run()` is the single entry point used by every other module. It executes the
fixed work described by a WorkloadSpec, samples counters throughout, collects
kernel statistics where possible, and returns a RunResult carrying the wall time
and the extracted signature.

Two execution paths share this interface:

  cuda   The torch workload is built and `step()` is called `iterations` times
         after `warmup` untimed iterations, wrapped in CUDA synchronisation so
         the timing is honest. Kernel statistics come from torch.profiler.

  sim    No GPU. Wall time is the workload's modelled solo duration multiplied
         by any co-location slowdown supplied by the caller, and the trace is
         generated analytically. Nothing here is a measurement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..common import env
from . import calibrate
from ..workloads import registry
from . import sampler as sampler_mod
from . import signature as sig_mod


@dataclass
class RunResult:
    workload: str
    wall_seconds: float
    signature: sig_mod.Signature
    backend: str
    impl: str = ""
    detail: str = ""
    device_index: int = 0
    oom: bool = False
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.oom and not self.error


# ---------------------------------------------------------------------------
# CUDA path
# ---------------------------------------------------------------------------
def _kernel_stats_cuda(runnable, n: int = 3) -> Dict[str, float]:
    """Count kernels and total device time with torch.profiler.

    Deliberately profiles only a few iterations: the profiler adds noticeable
    overhead, and kernel statistics are a per-iteration property that does not
    need the full fixed-work run to estimate.
    """
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            for _ in range(n):
                runnable.step()
            torch.cuda.synchronize()

        count = 0
        total_us = 0.0
        for evt in prof.key_averages():
            dev_us = getattr(evt, "device_time_total", None)
            if dev_us is None:
                dev_us = getattr(evt, "cuda_time_total", 0.0)
            if dev_us and dev_us > 0:
                count += int(evt.count)
                total_us += float(dev_us)
        if count == 0:
            return {}
        return {
            "kernel_count_per_iter": count / n,
            "kernel_mean_us": total_us / max(count, 1),
            "kernel_time_per_iter_s": (total_us / n) / 1e6,
        }
    except Exception:  # noqa: BLE001
        return {}


def _peak_vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def _run_cuda(spec, device_index: int, interval_ms: int,
              iterations: Optional[int]) -> RunResult:
    import torch

    from ..workloads import torch_workloads

    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats()

    try:
        runnable = torch_workloads.build(spec)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return RunResult(spec.name, float("nan"), sig_mod.from_spec(spec),
                             env.BACKEND_CUDA, oom=True, error=str(exc)[:200])
        raise

    n_iter = calibrate.iterations_for(spec, iterations)

    try:
        for _ in range(spec.warmup):
            runnable.step()
        torch.cuda.synchronize()

        kstats = _kernel_stats_cuda(runnable)

        sampler = sampler_mod.make_sampler(spec, device_index, interval_ms)
        sampler.start()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            runnable.step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        trace = sampler.stop()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return RunResult(spec.name, float("nan"), sig_mod.from_spec(spec),
                             env.BACKEND_CUDA, impl=runnable.impl, oom=True,
                             error=str(exc)[:200])
        raise
    finally:
        runnable.teardown()

    peak = _peak_vram_mb()
    if not np.isnan(peak) and "vram_used_mb" not in trace.samples.columns:
        trace.samples["vram_used_mb"] = peak

    signature = sig_mod.extract(
        spec.name, trace, wall,
        kernel_count=(kstats.get("kernel_count_per_iter", np.nan) * n_iter
                      if kstats else None),
        kernel_mean_us=kstats.get("kernel_mean_us"),
        kernel_total_s=(kstats.get("kernel_time_per_iter_s", 0.0) * n_iter
                        if kstats else None),
        transfer_mb=spec.sim_h2d_mb,
        kernel_source=(sig_mod.KERNEL_SOURCE_PROFILER if kstats
                       else sig_mod.KERNEL_SOURCE_SPEC),
    )
    if not np.isnan(peak):
        signature.features["vram_peak_mb"] = peak

    return RunResult(spec.name, wall, signature, env.BACKEND_CUDA,
                     impl=runnable.impl, detail=runnable.detail,
                     device_index=device_index,
                     extra={"iterations": n_iter, **kstats})


# ---------------------------------------------------------------------------
# Simulated path
# ---------------------------------------------------------------------------
def _run_sim(spec, device_index: int, interval_ms: int,
             slowdown: float, seed: int, oom: bool) -> RunResult:
    if oom:
        return RunResult(spec.name, float("nan"), sig_mod.from_spec(spec),
                         env.BACKEND_SIM, impl="sim", oom=True,
                         error="modelled VRAM exhaustion")

    rng = np.random.default_rng(abs(hash((spec.name, seed))) % (2**32))
    wall = max(spec.sim_solo_seconds * slowdown * float(rng.normal(1.0, 0.01)), 0.05)
    trace = sampler_mod.synth_trace(spec, wall, interval_ms, seed)

    iters = spec.iterations
    signature = sig_mod.extract(
        spec.name, trace, wall,
        kernel_count=spec.sim_kernel_count * iters,
        kernel_mean_us=spec.sim_kernel_us,
        kernel_total_s=spec.sim_kernel_count * iters * spec.sim_kernel_us / 1e6,
        transfer_mb=spec.sim_h2d_mb,
        kernel_source=sig_mod.KERNEL_SOURCE_SPEC,
    )
    return RunResult(spec.name, wall, signature, env.BACKEND_SIM,
                     impl="sim", detail="analytic model", device_index=device_index,
                     extra={"iterations": iters, "applied_slowdown": slowdown})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run(spec_or_name, device_index: int = 0,
        interval_ms: int = sampler_mod.DEFAULT_INTERVAL_MS,
        iterations: Optional[int] = None, slowdown: float = 1.0,
        seed: int = 0, oom: bool = False) -> RunResult:
    """Execute one fixed-work run under profiling.

    `slowdown` and `oom` are only consulted on the simulated path, where the
    caller supplies what the interference model predicted. On real hardware,
    slowdown is an *outcome*, never an input.
    """
    spec = spec_or_name if hasattr(spec_or_name, "name") else registry.get(spec_or_name)
    caps = env.detect()

    if caps.backend == env.BACKEND_CUDA and spec.gpu:
        return _run_cuda(spec, device_index, interval_ms, iterations)

    if caps.backend == env.BACKEND_CUDA and not spec.gpu:
        # CPU-only stage on a GPU machine: run it for real and sample anyway,
        # so the trace shows the device sitting idle. That idle trace is the
        # evidence for Objective 1.
        from ..workloads import torch_workloads

        runnable = torch_workloads.build(spec)
        n_iter = calibrate.iterations_for(spec, iterations)
        s = sampler_mod.make_sampler(spec, device_index, interval_ms)
        s.start()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            runnable.step()
        wall = time.perf_counter() - t0
        trace = s.stop()
        signature = sig_mod.extract(spec.name, trace, wall, kernel_count=0.0,
                                    kernel_mean_us=0.0, kernel_total_s=0.0,
                                    transfer_mb=0.0)
        return RunResult(spec.name, wall, signature, env.BACKEND_CUDA,
                         impl=runnable.impl, detail=runnable.detail,
                         device_index=device_index, extra={"iterations": n_iter})

    return _run_sim(spec, device_index, interval_ms, slowdown, seed, oom)


def profile_zoo(reps: int = 3, device_index: int = 0,
                interval_ms: int = sampler_mod.DEFAULT_INTERVAL_MS,
                names: Optional[List[str]] = None,
                verbose: bool = True) -> List[RunResult]:
    """Profile every workload solo, `reps` times each. M2's deliverable."""
    names = names or registry.zoo_names()
    out: List[RunResult] = []
    for name in names:
        spec = registry.get(name)
        for r in range(reps):
            res = run(spec, device_index=device_index, interval_ms=interval_ms, seed=r)
            out.append(res)
            if verbose:
                status = "OOM" if res.oom else f"{res.wall_seconds:7.2f}s"
                print(f"  {name:26s} rep {r}  {status}")
    return out
