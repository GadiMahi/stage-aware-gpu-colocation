"""M3: the pairwise co-location dataset.

Produces `pairs.parquet`, the central empirical artefact of the project and the
thing of most value to other researchers.

MEASUREMENT PROTOCOL
--------------------
For each pairing, each MPS thread-percentage setting, and each repetition:

  1. Re-measure the SOLO baseline for both tenants *in the current session*.
     Not optional. Clock behaviour on a shared cloud GPU varies between
     sessions, so a slowdown computed against yesterday's baseline is noise
     dressed up as a result. Solo baselines are cached per session and re-taken
     when the session identifier changes.
  2. Launch both tenants as concurrent processes with a shared scheduled start
     so their timed regions overlap.
  3. Record slowdown = co-located wall time / solo wall time, for each tenant
     independently. The two are different numbers; that asymmetry is the signal
     the predictor is trained on.
  4. Treat VRAM exhaustion as a LABELLED OUTCOME (`oom=True`), never as a crash
     to be retried. Which pairs cannot co-reside is information the scheduler
     needs.

Each measured pair yields TWO training rows by exchanging tenant roles.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common import env, provenance
from ..m2_profiler import runner as runner_mod
from ..m2_profiler import signature as sig_mod
from ..workloads import interference, registry
from . import mps as mps_mod

DEFAULT_THREAD_PCTS = [100, 50]
DEFAULT_REPS = 3
LAUNCH_LEAD_SECONDS = 6.0      # time allowed for both workers to warm up


@dataclass
class SweepConfig:
    thread_pcts: List[int] = field(default_factory=lambda: list(DEFAULT_THREAD_PCTS))
    reps: int = DEFAULT_REPS
    device_index: int = 0
    interval_ms: int = 100
    include_self_pairs: bool = True
    workloads: Optional[List[str]] = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def pairs(self) -> List[Tuple[str, str]]:
        return registry.unordered_pairs(self.workloads, self.include_self_pairs)

    def n_runs(self) -> int:
        return len(self.pairs()) * len(self.thread_pcts) * self.reps

    def estimate_gpu_hours(self) -> float:
        """Measurement budget: the number that decides whether this is feasible."""
        names = self.workloads or registry.zoo_names()
        mean_solo = float(np.mean([registry.get(n).sim_solo_seconds for n in names]))
        setup = 10.0
        colocated = self.n_runs() * (mean_solo + setup)
        baselines = len(names) * self.reps * 6 * (mean_solo + setup)  # ~6 sessions
        return (colocated + baselines) / 3600.0


class BaselineCache:
    """Per-session solo timings.

    Keyed on (session_id, workload, rep) so a new session forces fresh
    measurement. This is the guard against the most common way a cloud-based
    measurement study produces meaningless slowdown ratios.
    """

    def __init__(self, session_id: str, cfg=None, status=None, backend: str = ""):
        self.session_id = session_id
        self.cfg = cfg
        self.status = status
        self.backend = backend
        self._solo: Dict[Tuple[str, int], float] = {}
        self._sig: Dict[str, sig_mod.Signature] = {}

    def solo(self, name: str, rep: int, device_index: int, interval_ms: int) -> float:
        """Solo wall time, measured the same way the co-located run will be.

        On hardware the baseline is a worker subprocess running alone at a 100%
        thread partition, not an in-process call. Anything else makes the ratio
        measure the difference between two execution paths on top of the
        interference it is supposed to isolate.
        """
        key = (name, rep)
        if key in self._solo:
            return self._solo[key]

        if self.backend == env.BACKEND_CUDA and self.cfg is not None:
            rep_out = _measure_solo_cuda(name, 100, self.cfg, self.status, rep)
            wall = rep_out.get("wall_seconds")
            if wall:
                self._solo[key] = float(wall)
                return self._solo[key]
            # worker failed; fall through so the sweep still produces a row

        res = runner_mod.run(registry.get(name), device_index=device_index,
                             interval_ms=interval_ms, seed=rep)
        self._solo[key] = res.wall_seconds
        self._sig.setdefault(name, res.signature)
        return self._solo[key]

    def signature(self, name: str) -> sig_mod.Signature:
        if name not in self._sig:
            self._sig[name] = runner_mod.run(registry.get(name), seed=0).signature
        return self._sig[name]

    def median_solo(self, name: str) -> float:
        vals = [v for (n, _), v in self._solo.items() if n == name]
        return float(np.median(vals)) if vals else float("nan")


def _launch_worker(name: str, pct: int, out: Path, start_at: float,
                   cfg: SweepConfig, status: mps_mod.MPSStatus, rep: int):
    """Spawn one tenant. The ONLY way a timing is ever produced.

    Solo baselines and co-located runs must come from an identical execution
    path, or the ratio between them measures the path difference as well as the
    interference. Running the baseline in the sweep's own process did exactly
    that: the parent is not an MPS client, so it pays full kernel-launch
    overhead, while a worker goes through the MPS daemon and pays less. On a
    launch-bound workload such as gin_small_infer, 59k kernels of ~0.34 ms, that
    alone made the co-located run FASTER than its own baseline and produced
    slowdowns below 1.0, which is physically impossible under contention.
    """
    cmd = [sys.executable, "-m", "sagc.m3_dataset.worker",
           "--workload", name, "--device", "0", "--thread-pct", str(pct),
           "--start-at", f"{start_at:.3f}", "--interval-ms", str(cfg.interval_ms),
           "--seed", str(rep), "--out", str(out)]
    e = {**os.environ, **mps_mod.client_env(pct, cfg.device_index, status)}
    return subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def _read_report(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {"ok": False, "oom": False, "error": "worker produced no report",
            "wall_seconds": None}


def _measure_solo_cuda(name: str, pct: int, cfg: SweepConfig,
                       status: mps_mod.MPSStatus, rep: int) -> Dict:
    """One tenant alone, through the same worker path a co-located run uses."""
    tmp = Path(tempfile.mkdtemp(prefix="sagc-solo-"))
    out = tmp / "solo.json"
    start_at = time.time() + LAUNCH_LEAD_SECONDS
    p = _launch_worker(name, pct, out, start_at, cfg, status, rep)
    try:
        p.wait(timeout=LAUNCH_LEAD_SECONDS + 600)
    except subprocess.TimeoutExpired:
        p.kill()
    return _read_report(out)


def _measure_pair_cuda(a: str, b: str, thread_a: int, thread_b: int,
                       cfg: SweepConfig, status: mps_mod.MPSStatus,
                       rep: int) -> Dict:
    """Launch two worker processes that overlap, and collect their reports."""
    tmp = Path(tempfile.mkdtemp(prefix="sagc-pair-"))
    out_a, out_b = tmp / "a.json", tmp / "b.json"
    start_at = time.time() + LAUNCH_LEAD_SECONDS

    def launch(name: str, pct: int, out: Path):
        return _launch_worker(name, pct, out, start_at, cfg, status, rep)

    pa, pb = launch(a, thread_a, out_a), launch(b, thread_b, out_b)
    timeout = LAUNCH_LEAD_SECONDS + 600
    try:
        pa.wait(timeout=timeout)
        pb.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        for p in (pa, pb):
            p.kill()

    return {"a": _read_report(out_a), "b": _read_report(out_b)}


def _measure_pair_sim(a: str, b: str, thread_a: int, thread_b: int,
                      cfg: SweepConfig, rep: int) -> Dict:
    sa, sb = registry.get(a), registry.get(b)
    outcome = interference.predict_pair(sa, sb, thread_a, thread_b,
                                        device_vram_mb=env.device_memory_mb(),
                                        repetition=rep)
    if outcome.oom:
        return {"a": {"ok": False, "oom": True, "wall_seconds": None},
                "b": {"ok": False, "oom": True, "wall_seconds": None},
                "vram_required_mb": outcome.vram_required_mb}
    ra = runner_mod.run(sa, interval_ms=cfg.interval_ms,
                        slowdown=outcome.slowdown_a, seed=rep * 31 + 1)
    rb = runner_mod.run(sb, interval_ms=cfg.interval_ms,
                        slowdown=outcome.slowdown_b, seed=rep * 31 + 2)
    return {"a": {"ok": True, "oom": False, "wall_seconds": ra.wall_seconds},
            "b": {"ok": True, "oom": False, "wall_seconds": rb.wall_seconds},
            "vram_required_mb": outcome.vram_required_mb}


CHECKPOINT_PATH = Path("data/pairs.checkpoint.jsonl")


def _load_checkpoint(path: Path, session_id: str) -> List[dict]:
    """Rows from a previous attempt at THIS session, or an empty list.

    Rows from a different session are discarded rather than reused: their solo
    baselines were measured on a different process and machine state, so mixing
    them would silently corrupt the slowdown ratios this whole project rests on.
    """
    try:
        text = Path(path).read_text()
    except Exception:  # noqa: BLE001
        return []
    out: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue          # a torn final line from a killed process
        if rec.get("session_id") == session_id:
            out.append(rec)
    return out


def _append_checkpoint(path: Path, new_rows: List[dict]) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            for r in new_rows:
                fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:  # noqa: BLE001
        pass                  # checkpointing must never break the measurement


def run_sweep(cfg: Optional[SweepConfig] = None, verbose: bool = True,
              progress_every: int = 20,
              checkpoint: Path = CHECKPOINT_PATH
              ) -> Tuple[pd.DataFrame, provenance.Provenance]:
    """Execute the full pairwise sweep and return the labelled dataset."""
    cfg = cfg or SweepConfig()
    caps = env.detect()
    status = mps_mod.start()
    device_vram = env.device_memory_mb()

    if verbose:
        print(mps_mod.report(status))
        print(f"pairs={len(cfg.pairs())} thread_pcts={cfg.thread_pcts} reps={cfg.reps}")
        print(f"total co-located runs = {cfg.n_runs()}")
        print(f"estimated budget      = {cfg.estimate_gpu_hours():.1f} GPU-hours")
        print()

    if status.mode == mps_mod.MODE_CONCURRENT and len(cfg.thread_pcts) > 1:
        if verbose:
            print("MPS partition control unavailable; collapsing thread-pct sweep to [100]")
        cfg.thread_pcts = [100]

    cache = BaselineCache(cfg.session_id, cfg=cfg, status=status,
                          backend=caps.backend)
    # Resume support. The sweep is hours long and a session that dies at hour
    # four with everything still in memory loses every measurement, so each
    # measurement is appended to a JSONL checkpoint as soon as it is taken and
    # already-measured cells are skipped on a restart. The checkpoint is keyed
    # on the session id as well as the cell, because a slowdown ratio is only
    # comparable against a baseline measured in the same session.
    rows: List[dict] = _load_checkpoint(checkpoint, cfg.session_id)
    n_recovered = len(rows)
    measured = {(r["workload_a"], r["workload_b"], r["thread_pct_a"], r["rep"])
                for r in rows}
    if rows and verbose:
        print(f"RESUMING: {len(measured)} cells already measured, "
              f"{len(rows)} rows recovered from {checkpoint}")
    if verbose:
        print(f"session id: {cfg.session_id}")
        print(f"  to resume this sweep after a crash, re-run with "
              f"--session-id {cfg.session_id}")
    t_start = time.time()
    done = 0

    try:
        for (a, b) in cfg.pairs():
            for pct in cfg.thread_pcts:
                for rep in range(cfg.reps):
                    if (a, b, pct, rep) in measured:
                        done += 1
                        continue
                    solo_a = cache.solo(a, rep, cfg.device_index, cfg.interval_ms)
                    solo_b = cache.solo(b, rep, cfg.device_index, cfg.interval_ms)

                    meas = (_measure_pair_cuda(a, b, pct, pct, cfg, status, rep)
                            if caps.backend == env.BACKEND_CUDA
                            else _measure_pair_sim(a, b, pct, pct, cfg, rep))

                    ra, rb = meas["a"], meas["b"]
                    oom = bool(ra.get("oom") or rb.get("oom"))
                    vram_sum = registry.get(a).sim_vram_mb + registry.get(b).sim_vram_mb

                    if oom:
                        slow_a = slow_b = np.nan
                    else:
                        wa, wb = ra.get("wall_seconds"), rb.get("wall_seconds")
                        slow_a = (wa / solo_a) if (wa and solo_a) else np.nan
                        slow_b = (wb / solo_b) if (wb and solo_b) else np.nan

                    base = {"session_id": cfg.session_id, "rep": rep,
                            "thread_pct_a": pct, "thread_pct_b": pct, "oom": oom,
                            "vram_sum_mb": vram_sum,
                            "vram_headroom_mb": device_vram - vram_sum,
                            "device_vram_mb": device_vram,
                            "colocation_mode": status.mode}
                    # two rows per measurement: (target=a, other=b) and the swap
                    rows.append({**base, "workload_a": a, "workload_b": b,
                                 "solo_seconds_a": solo_a, "solo_seconds_b": solo_b,
                                 "colocated_seconds_a": ra.get("wall_seconds"),
                                 "colocated_seconds_b": rb.get("wall_seconds"),
                                 "slowdown_a": slow_a, "slowdown_b": slow_b,
                                 "role": "ab"})
                    rows.append({**base, "workload_a": b, "workload_b": a,
                                 "solo_seconds_a": solo_b, "solo_seconds_b": solo_a,
                                 "colocated_seconds_a": rb.get("wall_seconds"),
                                 "colocated_seconds_b": ra.get("wall_seconds"),
                                 "slowdown_a": slow_b, "slowdown_b": slow_a,
                                 "role": "ba"})
                    _append_checkpoint(checkpoint, rows[-2:])
                    measured.add((a, b, pct, rep))
                    done += 1
                    if verbose and done % progress_every == 0:
                        el = time.time() - t_start
                        rate = done / max(el, 1e-6)
                        eta = (cfg.n_runs() - done) / max(rate, 1e-9)
                        state = "OOM" if oom else f"{slow_a:.2f}/{slow_b:.2f}"
                        print(f"  {done:5d}/{cfg.n_runs()}  elapsed {el/60:5.1f}m  "
                              f"eta {eta/60:5.1f}m  {a[:18]}+{b[:18]} {state}")
    finally:
        mps_mod.stop(status)

    df = pd.DataFrame(rows)
    # A resumed sweep was measured by more than one process. The solo baselines
    # are re-measured in each process, so the dataset is not strictly
    # single-session, and that must be recorded rather than quietly absorbed.
    prov = provenance.capture(module="M3", colocation_mode=status.mode,
                              thread_pcts=cfg.thread_pcts, reps=cfg.reps,
                              n_pairs=len(cfg.pairs()), session_id=cfg.session_id,
                              resumed=bool(n_recovered),
                              rows_recovered_from_checkpoint=n_recovered)
    return provenance.stamp_dataframe(df, prov), prov


def attach_signatures(df: pd.DataFrame, signatures: pd.DataFrame) -> pd.DataFrame:
    """Join each row's tenant signatures on, producing the model's feature table."""
    feats = sig_mod.FEATURE_NAMES + ["util_gpu_mean", "util_gpu_p95"]
    sig = signatures.set_index("workload")
    keep = [c for c in feats if c in sig.columns]
    out = df.merge(sig[keep].add_prefix("a_"), left_on="workload_a",
                   right_index=True, how="left")
    out = out.merge(sig[keep].add_prefix("b_"), left_on="workload_b",
                    right_index=True, how="left")
    return out


def save(df: pd.DataFrame, prov: provenance.Provenance,
         path: str = "data/pairs.parquet") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False)
    except Exception:  # noqa: BLE001
        p = p.with_suffix(".csv")
        df.to_csv(p, index=False)
    prov.write(p.with_suffix(".provenance.json"))
    return p
