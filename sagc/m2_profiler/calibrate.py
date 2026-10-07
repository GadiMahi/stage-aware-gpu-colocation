"""Calibrate fixed work against the device actually present.

WHY THIS EXISTS
---------------
Every workload in the zoo declares an `iterations` count chosen so that a solo
run takes roughly 20 to 30 seconds. Those counts were derived from an ESTIMATE
of T4 throughput rather than a measurement, and the estimate was wrong by up to
a factor of twenty: `resnet50_train_b32` is declared at 28 seconds and takes 575
on a real T4. At that rate the full campaign needs about seventy GPU-hours,
which does not fit in a twelve-hour session or a thirty-hour weekly quota.

Guessing better numbers would repeat the mistake on the next GPU. Instead this
module measures the per-iteration cost on the device in front of it and solves
for the iteration count that hits a target duration.

FIXED WORK IS PRESERVED
-----------------------
Slowdown is a ratio, so the denominator and numerator must do identical work.
Calibration therefore happens ONCE, before any measurement, and the resulting
counts are written to `data/calibration.json` and reused unchanged by every
solo run and every co-located run in the campaign. What varies between machines
is the amount of work; what never varies is the work within one session's
comparisons. A time-boxed run would break this, which is why the runner still
counts iterations rather than watching the clock.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..common import env
from ..workloads import registry

CALIBRATION_PATH = Path("data/calibration.json")

#: Target solo wall time per workload. Long enough that sampling has something
#: to see and start-up cost is amortised, short enough that 78 pairings fit a
#: session.
DEFAULT_TARGET_SECONDS = 20.0

#: Minimum iterations, so a very slow workload still runs a sane number of
#: steps rather than one enormous one.
MIN_ITERATIONS = 8

#: How long the probe itself may take per workload.
PROBE_SECONDS = 4.0
PROBE_MAX_ITERS = 20000


def _probe_one(spec, device_index: int, probe_seconds: float) -> Optional[float]:
    """Seconds per timed iteration for one workload, or None if it failed."""
    import torch

    from ..workloads import torch_workloads

    torch.cuda.set_device(device_index)
    try:
        runnable = torch_workloads.build(spec)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return None
        raise

    try:
        for _ in range(max(1, min(spec.warmup, 3))):
            runnable.step()
        torch.cuda.synchronize()

        # Grow the batch until it has run long enough to time reliably.
        n = 1
        while True:
            t0 = time.perf_counter()
            for _ in range(n):
                runnable.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if elapsed >= probe_seconds or n >= PROBE_MAX_ITERS:
                return elapsed / n
            # aim straight at the probe budget, with a cap so one step cannot
            # explode the next round
            grow = min(8.0, max(2.0, probe_seconds / max(elapsed, 1e-6)))
            n = min(PROBE_MAX_ITERS, int(n * grow) + 1)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return None
        raise
    finally:
        runnable.teardown()


def calibrate(names: Optional[List[str]] = None,
              target_seconds: float = DEFAULT_TARGET_SECONDS,
              device_index: int = 0,
              probe_seconds: float = PROBE_SECONDS,
              verbose: bool = True) -> Dict[str, dict]:
    """Measure per-iteration cost and solve for the iteration count."""
    caps = env.detect()
    names = list(names or registry.ZOO)

    if caps.backend != env.BACKEND_CUDA:
        if verbose:
            print("backend is not cuda; calibration is a no-op and the "
                  "declared iteration counts are kept")
        return {}

    out: Dict[str, dict] = {}
    if verbose:
        print("=" * 72)
        print(f"CALIBRATE  target {target_seconds:.0f}s per solo run")
        print("=" * 72)
        print(f"{'workload':26s} {'declared':>9s} {'s/iter':>10s} "
              f"{'calibrated':>11s} {'would have been':>16s}")

    for name in names:
        spec = registry.get(name)
        per_iter = _probe_one(spec, device_index, probe_seconds)
        if per_iter is None or per_iter <= 0:
            if verbose:
                print(f"{name:26s} {spec.iterations:9d} {'OOM/failed':>10s} "
                      f"{'kept':>11s}")
            continue
        iters = max(MIN_ITERATIONS, int(round(target_seconds / per_iter)))
        would_have = spec.iterations * per_iter
        out[name] = {
            "iterations": iters,
            "seconds_per_iteration": per_iter,
            "declared_iterations": spec.iterations,
            "declared_would_take_s": would_have,
            "target_seconds": target_seconds,
        }
        if verbose:
            print(f"{name:26s} {spec.iterations:9d} {per_iter:10.5f} "
                  f"{iters:11d} {would_have:14.0f}s")

    if verbose and out:
        saved = sum(v["declared_would_take_s"] for v in out.values())
        now = target_seconds * len(out)
        print(f"\none solo pass over these workloads: {now:.0f}s calibrated, "
              f"{saved:.0f}s as declared ({saved / max(now, 1e-9):.1f}x)")
    return out


def save(table: Dict[str, dict], path: Path = CALIBRATION_PATH) -> Path:
    caps = env.detect()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": caps.gpu_names[0] if caps.gpu_names else "unknown",
        "backend": caps.backend,
        "workloads": table,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load(path: Path = CALIBRATION_PATH) -> Dict[str, int]:
    """Calibrated iteration counts, or an empty mapping if none exist.

    Returns an empty mapping rather than raising, so every call site degrades to
    the declared counts instead of failing.
    """
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return {}
    table = payload.get("workloads") or {}
    out: Dict[str, int] = {}
    for name, rec in table.items():
        try:
            out[name] = int(rec["iterations"])
        except Exception:  # noqa: BLE001
            continue
    return out


def iterations_for(spec, override: Optional[int] = None,
                   table: Optional[Dict[str, int]] = None) -> int:
    """The iteration count to actually run: explicit, then calibrated, then declared."""
    if override is not None:
        return int(override)
    table = load() if table is None else table
    return int(table.get(spec.name, spec.iterations))
