"""Phase 1 go/no-go, as one command with a verdict at the end.

Run this before spending GPU-hours on the full campaign. It exercises every
stage of the pipeline on a three-workload subset, then prints an explicit
GO or NO-GO with a reason, so the decision is not left to eyeballing scroll-back.

    python scripts/smoke_test.py

Exit status is 0 on GO and 1 on NO-GO, so a notebook cell fails loudly.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBSET = "esm2_35m_infer,mlp_admet_infer,resnet50_train_b32"

#: Each check is (label, callable -> (ok, detail)). A FATAL check failing means
#: the full campaign cannot produce what the review needs; a WARN check failing
#: costs a specific claim and is reported rather than hidden.
FATAL = "fatal"
WARN = "warn"


def _run(args: list[str], timeout: int = 2400,
         env_extra: dict[str, str] | None = None) -> tuple[int, str]:
    t0 = time.time()
    print(f"\n$ {' '.join(args)}", flush=True)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(args, cwd=REPO, capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print(f"[TIMED OUT after {timeout}s]", flush=True)
        return 124, f"timed out after {timeout}s"
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-4000:], flush=True)
    print(f"[{time.time() - t0:.0f}s, exit {p.returncode}]", flush=True)
    return p.returncode, out


def main() -> int:
    results: list[tuple[str, str, bool, str]] = []

    def record(label: str, severity: str, ok: bool, detail: str) -> None:
        results.append((label, severity, ok, detail))

    # ---- environment -------------------------------------------------------
    rc, out = _run([sys.executable, "-m", "sagc", "probe"], timeout=300)
    record("probe ran", FATAL, rc == 0, f"exit {rc}")

    backend = "unknown"
    tier = "unknown"
    mode = "unknown"
    for line in out.splitlines():
        if line.strip().startswith("backend"):
            backend = line.split(":", 1)[-1].strip()
        elif line.strip().startswith("profiler tier") and tier == "unknown":
            tier = line.split(":", 1)[-1].strip()
        elif line.strip().startswith("co-location mode") and mode == "unknown":
            mode = line.split(":", 1)[-1].strip()

    record("GPU backend is cuda", FATAL, backend == "cuda",
           f"backend={backend}; 'sim' means the accelerator is not attached and "
           f"every number would be stamped SIMULATED")
    record("achieved occupancy available", WARN, tier == "dcgm",
           f"tier={tier}; without dcgm, ablation A1 runs REDUCED (no occupancy)")
    record("MPS partition control", WARN, mode == "mps",
           f"mode={mode}; without mps, use --thread-pcts 100 and A6 is unavailable")

    # ---- tests -------------------------------------------------------------
    # The suite is logic-only and is pinned to the simulated backend by
    # tests/conftest.py. Pinned again here so an older checkout of that file
    # cannot quietly turn pytest into a billed GPU campaign.
    rc, out = _run([sys.executable, "-m", "pytest", "tests/", "-q"],
                   timeout=600, env_extra={"SAGC_BACKEND": "sim"})
    record("test suite passes", FATAL, rc == 0,
           f"exit {rc}" + (" (timed out; the suite must not touch the GPU)"
                           if rc == 124 else ""))

    # ---- the pipeline, on a subset -----------------------------------------
    rc, out = _run([sys.executable, "-m", "sagc", "profile", "--reps", "2",
                    "--workloads", SUBSET])
    record("profile stage completes", FATAL, rc == 0, f"exit {rc}")
    fallbacks = out.count("impl=fallback")
    native = out.count("impl=native")
    record("real models built", WARN, fallbacks == 0,
           f"{native} native, {fallbacks} fallback; fallbacks are recorded and "
           f"acceptable but must be disclosed")

    rc, _ = _run([sys.executable, "-m", "sagc", "sweep", "--reps", "2",
                  "--thread-pcts", "100", "--workloads", SUBSET])
    record("co-location sweep completes", FATAL, rc == 0, f"exit {rc}")

    rc, out = _run([sys.executable, "-m", "sagc", "train", "--skip-ablations"])
    record("training completes", FATAL, rc == 0, f"exit {rc}")

    rc, _ = _run([sys.executable, "-m", "sagc", "schedule", "--pipelines", "4"])
    record("scheduler consumes the model", FATAL, rc == 0, f"exit {rc}")

    # ---- provenance --------------------------------------------------------
    stamps = []
    for p in sorted(glob.glob(str(REPO / "data" / "*.provenance.json"))):
        try:
            d = json.loads(Path(p).read_text())
        except Exception as exc:  # noqa: BLE001
            stamps.append((Path(p).name, f"unreadable: {exc}"))
            continue
        stamps.append((Path(p).name, d.get("kind", "?")))
    measured = [n for n, k in stamps if k == "MEASURED"]
    record("provenance says MEASURED", FATAL, bool(stamps) and len(measured) == len(stamps),
           "; ".join(f"{n}={k}" for n, k in stamps) or "no provenance files written")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("PHASE 1 GO / NO-GO")
    print("=" * 72)
    for label, severity, ok, detail in results:
        flag = "ok  " if ok else ("FAIL" if severity == FATAL else "warn")
        print(f"[{flag}] {label}")
        if not ok:
            print(f"        {detail}")

    fatal_failures = [r for r in results if r[1] == FATAL and not r[2]]
    warnings = [r for r in results if r[1] == WARN and not r[2]]

    print("-" * 72)
    print(f"environment: backend={backend}  tier={tier}  co-location={mode}")
    if fatal_failures:
        print(f"\nNO-GO. {len(fatal_failures)} blocking failure(s). Do not start the "
              f"full campaign until these are fixed.")
        return 1

    print("\nGO. Launch the full campaign with Save Version -> Save & Run All (Commit).")
    print("    python -m sagc profile --reps 3")
    print("    python -m sagc sweep   --reps 3 --thread-pcts "
          + ("100,50" if mode == "mps" else "100"))
    print("    python -m sagc train")
    print("    python -m sagc schedule --pipelines 20")
    print("    python -m sagc figures")
    print("    python -m sagc report")
    if warnings:
        print(f"\n{len(warnings)} caveat(s) to disclose in the review:")
        for label, _s, _ok, detail in warnings:
            print(f"  - {label}: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
