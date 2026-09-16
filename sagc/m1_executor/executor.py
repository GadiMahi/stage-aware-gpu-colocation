"""M1: the hardware campaign executor.

Runs N pipeline instances over the available GPUs under a chosen placement
policy, on real hardware, and records what actually happened. This module
produces the characterisation in Objective 1 and the hardware results that
validate the simulator.

CHECKPOINTING
-------------
A Kaggle session is capped at twelve hours and can be interrupted at any point.
A campaign that loses everything on interruption is unusable, so the executor
writes a manifest after every state change and can resume from it. Completed
stages are never re-run; stages in flight at the moment of interruption are
re-executed, since their timing is contaminated.

DIFFERENCE FROM THE SIMULATOR
-----------------------------
The simulator advances a virtual clock and computes slowdown from a model. This
executor advances real time and OBSERVES slowdown by dividing measured
co-located duration by a measured solo baseline. The two produce the same trace
format on purpose: every figure and metric works identically on either, and M6's
validation compares them directly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..common import env
from ..m2_profiler import runner as runner_mod
from ..m3_dataset import mps as mps_mod
from ..m5_scheduler import policies as pol
from ..workloads import registry
from . import dag
from .dag import STATE_DONE, STATE_FAILED, STATE_PENDING, STATE_RUNNING, CampaignTrace

POLL_INTERVAL_S = 0.25
STAGE_TIMEOUT_S = 900


@dataclass
class _Launched:
    stage: dag.StageInstance
    proc: subprocess.Popen
    out_path: Path
    started_at: float


class HardwareExecutor:
    """Execute a campaign on real GPUs."""

    def __init__(self, policy: pol.Policy, n_pipelines: int = 20,
                 n_devices: Optional[int] = None,
                 device_vram_mb: Optional[int] = None,
                 slowdown_bound: float = 1.25,
                 solo_baselines: Optional[Dict[str, float]] = None,
                 checkpoint_path: Optional[str] = None,
                 specs: Optional[List] = None, verbose: bool = True):
        self.policy = policy
        self.n_pipelines = n_pipelines
        self.n_devices = n_devices if n_devices is not None else env.n_devices()
        self.device_vram_mb = (device_vram_mb if device_vram_mb is not None
                               else env.device_memory_mb())
        self.slowdown_bound = slowdown_bound
        self.solo_baselines = dict(solo_baselines or {})
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.specs = specs
        self.verbose = verbose

        self.pipelines = dag.build_campaign(n_pipelines, specs)
        self.devices = [pol.DeviceState(i, self.device_vram_mb)
                        for i in range(self.n_devices)]
        self.run_id = uuid.uuid4().hex[:8]
        self._tmp = Path(tempfile.mkdtemp(prefix=f"sagc-campaign-{self.run_id}-"))

    # -- baselines ----------------------------------------------------------
    def measure_baselines(self, reps: int = 3) -> Dict[str, float]:
        """Solo timing for every stage, measured in THIS session.

        Slowdown is meaningless without it, and a baseline carried over from a
        previous session on a shared cloud GPU is not a baseline.
        """
        out: Dict[str, float] = {}
        for spec in (self.specs or registry.PIPELINE):
            times = [r.wall_seconds for r in
                     (runner_mod.run(spec, device_index=0, seed=i) for i in range(reps))
                     if r.ok]
            if times:
                out[spec.name] = float(np.median(times))
                if self.verbose:
                    print(f"  baseline {spec.name:26s} {out[spec.name]:7.2f}s "
                          f"(median of {len(times)})")
        self.solo_baselines.update(out)
        return out

    # -- checkpointing -------------------------------------------------------
    def _checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        state = {"run_id": self.run_id, "policy": self.policy.name,
                 "n_pipelines": self.n_pipelines, "n_devices": self.n_devices,
                 "slowdown_bound": self.slowdown_bound,
                 "solo_baselines": self.solo_baselines,
                 "stages": [s.to_row() for p in self.pipelines for s in p.stages]}
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(self.checkpoint_path)

    def resume(self) -> bool:
        """Restore completed stages from a checkpoint. True if anything resumed."""
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return False
        try:
            state = json.loads(self.checkpoint_path.read_text())
        except json.JSONDecodeError:
            return False
        by_key = {f"p{s['pipeline_id']}s{s['stage_idx']}": s
                  for s in state.get("stages", [])}
        restored = 0
        for p in self.pipelines:
            for s in p.stages:
                prior = by_key.get(s.key)
                if prior and prior.get("state") == STATE_DONE:
                    s.state = STATE_DONE
                    for k in ("device", "start_s", "end_s", "actual_seconds",
                              "actual_slowdown", "predicted_slowdown"):
                        setattr(s, k, prior.get(k))
                    restored += 1
        self.solo_baselines.update(state.get("solo_baselines", {}))
        if self.verbose and restored:
            print(f"  resumed {restored} completed stages from checkpoint")
        return restored > 0

    # -- launching -----------------------------------------------------------
    def _launch(self, stage: dag.StageInstance, device_index: int,
                status: mps_mod.MPSStatus) -> _Launched:
        out = self._tmp / f"{stage.key}.json"
        cmd = [sys.executable, "-m", "sagc.m3_dataset.worker",
               "--workload", stage.workload, "--device", "0",
               "--thread-pct", str(self.policy.thread_pct),
               "--start-at", "0", "--out", str(out)]
        e = {**os.environ}
        if device_index >= 0:
            e.update(mps_mod.client_env(self.policy.thread_pct, device_index, status))
        else:
            e["CUDA_VISIBLE_DEVICES"] = ""       # CPU stage: hide the GPU entirely
        proc = subprocess.Popen(cmd, env=e, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)
        return _Launched(stage, proc, out, time.time())

    # -- main loop -----------------------------------------------------------
    def run(self) -> CampaignTrace:
        caps = env.detect()
        status = mps_mod.start()
        if self.verbose:
            print(mps_mod.report(status))
            print(f"campaign {self.run_id}: {self.n_pipelines} pipelines, "
                  f"{self.n_devices} devices, policy={self.policy.name}, "
                  f"bound={self.slowdown_bound}")

        if not self.solo_baselines:
            if self.verbose:
                print("measuring solo baselines in this session")
            self.measure_baselines()

        for p in self.pipelines:
            for s in p.stages:
                if s.workload in self.solo_baselines:
                    s.solo_seconds = self.solo_baselines[s.workload]

        t0 = time.time()
        inflight: List[_Launched] = []

        try:
            while True:
                ready = [s for s in (p.next_ready() for p in self.pipelines)
                         if s is not None and s.state == STATE_PENDING]
                if ready:
                    for stage, dev_idx, predicted in self.policy.place(ready, self.devices):
                        stage.state = STATE_RUNNING
                        stage.device = dev_idx
                        stage.start_s = time.time() - t0
                        stage.predicted_slowdown = (
                            None if predicted is None or np.isnan(predicted)
                            else float(predicted))
                        if dev_idx >= 0:
                            stage.co_tenants = [x.workload
                                                for x in self.devices[dev_idx].resident
                                                if x.key != stage.key]
                        inflight.append(self._launch(stage, dev_idx, status))
                        if self.verbose:
                            co = "+".join(stage.co_tenants) or "alone"
                            print(f"  [{stage.start_s:7.1f}s] start {stage.key} "
                                  f"{stage.workload[:22]:22s} dev={dev_idx} with {co}")
                    self._checkpoint()

                if not inflight:
                    if all(p.done for p in self.pipelines):
                        break
                    if not ready:
                        if self.verbose:
                            print("  no runnable stages and nothing in flight; stopping")
                        break
                    time.sleep(POLL_INTERVAL_S)
                    continue

                time.sleep(POLL_INTERVAL_S)

                for L in [x for x in inflight
                          if x.proc.poll() is not None
                          or (time.time() - x.started_at) > STAGE_TIMEOUT_S]:
                    inflight.remove(L)
                    stage = L.stage
                    if L.proc.poll() is None:
                        L.proc.kill()
                    report = {}
                    if L.out_path.exists():
                        try:
                            report = json.loads(L.out_path.read_text())
                        except json.JSONDecodeError:
                            report = {}
                    stage.end_s = time.time() - t0
                    wall = report.get("wall_seconds")
                    stage.actual_seconds = (float(wall) if wall
                                            else stage.end_s - (stage.start_s or stage.end_s))
                    solo = self.solo_baselines.get(stage.workload) or stage.solo_seconds
                    stage.actual_slowdown = (float(stage.actual_seconds) / float(solo)
                                             if solo and stage.actual_seconds else None)
                    stage.state = STATE_DONE if report.get("ok") else STATE_FAILED
                    self.policy.observe(stage, stage.actual_slowdown or 1.0)
                    if stage.device is not None and stage.device >= 0:
                        d = self.devices[stage.device]
                        d.resident = [x for x in d.resident if x.key != stage.key]
                    if self.verbose:
                        print(f"  [{stage.end_s:7.1f}s] done  {stage.key} "
                              f"{stage.actual_seconds:6.2f}s "
                              f"slowdown={stage.actual_slowdown or float('nan'):.3f}")

                for d in self.devices:
                    pid = d.reserved_by_pipeline
                    if pid is not None and self.pipelines[pid].done and not d.resident:
                        d.reserved_by_pipeline = None

                self._checkpoint()
        finally:
            for L in inflight:
                L.proc.kill()
            mps_mod.stop(status)

        trace = dag.trace_from_stages(
            [s for p in self.pipelines for s in p.stages],
            policy=self.policy.name, n_devices=self.n_devices,
            device_vram_mb=self.device_vram_mb, slowdown_bound=self.slowdown_bound,
            backend=caps.backend, wall_seconds=time.time() - t0,
            meta={"run_id": self.run_id, "colocation_mode": status.mode})
        self._checkpoint()
        return trace


def characterise(n_pipelines: int = 4, verbose: bool = True) -> CampaignTrace:
    """Objective 1: run a campaign under exclusive allocation and measure waste.

    The motivating measurement. It deliberately uses the exclusive policy,
    because the number being reported is how much capacity the STATUS QUO leaves
    on the floor.
    """
    ex = HardwareExecutor(policy=pol.ExclusivePolicy(), n_pipelines=n_pipelines,
                          slowdown_bound=999.0, verbose=verbose)
    trace = ex.run()
    if verbose:
        s = trace.summary()
        print()
        print("CHARACTERISATION")
        print(f"  held but idle            : {s['held_but_idle_fraction']*100:.1f}%")
        print(f"  held but under-occupied  : {s['held_but_under_occupied_fraction']*100:.1f}%")
        print(f"  mean achieved occupancy  : {s['mean_achieved_occupancy']:.3f}")
        print(f"  throughput               : {s['throughput_per_hour']:.2f} pipelines/hour")
    return trace
