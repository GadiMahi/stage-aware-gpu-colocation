"""One co-location client.

Run as a subprocess, one per tenant. The orchestrator starts two of these and
they overlap in time, which is what produces the contention being measured.

Overlap is achieved with a *scheduled start*: both workers build their model,
complete warm-up, then wait until a shared wall-clock instant passed in as
`--start-at`. This avoids inter-process synchronisation machinery entirely and
guarantees that neither tenant's timed region includes the other's model
construction, which would otherwise contaminate the slowdown ratio.

Usage:
    python -m sagc.m3_dataset.worker --workload bert_base_s128_b32 \
        --device 0 --thread-pct 100 --start-at 1726200000.0 --out /tmp/a.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Co-location worker (one tenant)")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--thread-pct", type=int, default=100)
    ap.add_argument("--start-at", type=float, default=0.0,
                    help="unix timestamp to begin the timed region")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from ..common import env
    from ..m2_profiler import runner, sampler as sampler_mod, signature as sig_mod
    from ..workloads import registry

    spec = registry.get(args.workload)
    caps = env.detect()
    result = {"workload": args.workload, "device": args.device,
              "thread_pct": args.thread_pct, "backend": caps.backend,
              "pid": os.getpid()}

    try:
        if caps.backend == env.BACKEND_CUDA and spec.gpu:
            import torch

            from ..workloads import torch_workloads

            torch.cuda.set_device(0)   # CUDA_VISIBLE_DEVICES already narrows this
            torch.cuda.reset_peak_memory_stats()
            runnable = torch_workloads.build(spec)

            for _ in range(spec.warmup):
                runnable.step()
            torch.cuda.synchronize()

            # hold until the shared start instant so both tenants overlap
            if args.start_at > 0:
                while time.time() < args.start_at:
                    time.sleep(0.001)

            n_iter = args.iterations or spec.iterations
            s = sampler_mod.make_sampler(spec, 0, args.interval_ms)
            s.start()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                runnable.step()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            trace = s.stop()

            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if "vram_used_mb" not in trace.samples.columns:
                trace.samples["vram_used_mb"] = peak

            sig = sig_mod.extract(spec.name, trace, wall,
                                  kernel_count=spec.sim_kernel_count * n_iter,
                                  kernel_mean_us=spec.sim_kernel_us,
                                  kernel_total_s=None,
                                  transfer_mb=spec.sim_h2d_mb)
            sig.features["vram_peak_mb"] = peak
            runnable.teardown()

            result.update({"ok": True, "oom": False, "wall_seconds": wall,
                           "iterations": n_iter, "impl": runnable.impl,
                           "detail": runnable.detail, "peak_vram_mb": peak,
                           "signature": sig.to_row(), "tier": trace.tier})
        else:
            res = runner.run(spec, device_index=args.device,
                             interval_ms=args.interval_ms, seed=args.seed)
            result.update({"ok": res.ok, "oom": res.oom,
                           "wall_seconds": res.wall_seconds,
                           "iterations": res.extra.get("iterations"),
                           "impl": res.impl, "detail": res.detail,
                           "peak_vram_mb": res.signature.features.get("vram_peak_mb"),
                           "signature": res.signature.to_row(),
                           "tier": res.signature.tier})
    except RuntimeError as exc:
        result.update({"ok": False, "oom": "out of memory" in str(exc).lower(),
                       "error": str(exc)[:400], "wall_seconds": None})
    except Exception as exc:  # noqa: BLE001
        result.update({"ok": False, "oom": False,
                       "error": f"{type(exc).__name__}: {exc}"[:400],
                       "wall_seconds": None})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
