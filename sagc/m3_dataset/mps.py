"""MPS control plane.

NVIDIA's Multi-Process Service lets several processes submit work to one GPU
concurrently. It is the *mechanism* this project builds a policy on top of; it
is not itself a contribution.

Two things matter here:

  1. The control daemon must be running before any client starts, and clients
     inherit it through the environment.
  2. `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` caps how many SMs an individual client
     may use. It is set per-client, which is what lets the scheduler hand
     different partitions to different co-tenants.

If MPS cannot be started (common in a sandboxed or shared session), co-location
falls back to plain concurrent processes on the same device. That still produces
real contention and real slowdown; it simply removes partition control, so the
thread-percentage sweep degrades to a single setting. The fallback is recorded
so the dataset never conflates the two.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PIPE_DIR = "/tmp/sagc-mps-pipe"
DEFAULT_LOG_DIR = "/tmp/sagc-mps-log"

MODE_MPS = "mps"
MODE_CONCURRENT = "concurrent"   # no MPS; plain concurrent processes
MODE_SIM = "simulated"


@dataclass
class MPSStatus:
    mode: str
    available: bool
    started_by_us: bool = False
    detail: str = ""

    @property
    def can_set_thread_pct(self) -> bool:
        return self.mode == MODE_MPS


def _control(cmd: str, env: Optional[Dict[str, str]] = None,
             timeout: int = 10) -> Optional[str]:
    if not shutil.which("nvidia-cuda-mps-control"):
        return None
    try:
        p = subprocess.run(["nvidia-cuda-mps-control"], input=cmd + "\n",
                           capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **(env or {})})
        return p.stdout if p.returncode == 0 else None
    except (subprocess.SubprocessError, OSError):
        return None


def is_running() -> bool:
    return _control("get_default_active_thread_percentage") is not None


def start(pipe_dir: str = DEFAULT_PIPE_DIR,
          log_dir: str = DEFAULT_LOG_DIR) -> MPSStatus:
    """Start the MPS control daemon if it is not already up."""
    from ..common import env as envmod

    caps = envmod.detect()
    if caps.backend != envmod.BACKEND_CUDA:
        return MPSStatus(MODE_SIM, available=False,
                         detail="no GPU; co-location is modelled analytically")

    if not caps.has_mps_control:
        return MPSStatus(MODE_CONCURRENT, available=True,
                         detail="nvidia-cuda-mps-control not found; using plain "
                                "concurrent processes (no partition control)")

    if is_running():
        return MPSStatus(MODE_MPS, available=True, started_by_us=False,
                         detail="MPS daemon already running")

    Path(pipe_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    child_env = {"CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
                 "CUDA_MPS_LOG_DIRECTORY": log_dir}
    try:
        subprocess.run(["nvidia-cuda-mps-control", "-d"],
                       env={**os.environ, **child_env}, capture_output=True,
                       text=True, timeout=20, check=False)
    except (subprocess.SubprocessError, OSError) as exc:
        return MPSStatus(MODE_CONCURRENT, available=True,
                         detail=f"MPS start failed ({exc}); using concurrent processes")

    for _ in range(10):
        time.sleep(0.3)
        if _control("get_default_active_thread_percentage", child_env) is not None:
            os.environ.update(child_env)
            return MPSStatus(MODE_MPS, available=True, started_by_us=True,
                             detail=f"MPS daemon started, pipe={pipe_dir}")

    return MPSStatus(MODE_CONCURRENT, available=True,
                     detail="MPS daemon did not come up; using concurrent processes")


def stop(status: MPSStatus) -> None:
    """Shut down the daemon, but only if we started it."""
    if status.mode == MODE_MPS and status.started_by_us:
        _control("quit")


def client_env(thread_pct: int = 100, device_index: int = 0,
               status: Optional[MPSStatus] = None) -> Dict[str, str]:
    """Environment for one co-located client process."""
    e = {"CUDA_VISIBLE_DEVICES": str(device_index), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if status is None or status.can_set_thread_pct:
        e["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(thread_pct))
    for k in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
        if k in os.environ:
            e[k] = os.environ[k]
    return e


def report(status: MPSStatus) -> str:
    lines = [f"co-location mode: {status.mode}"]
    if status.detail:
        lines.append(f"  {status.detail}")
    if status.mode == MODE_MPS:
        pct = _control("get_default_active_thread_percentage")
        if pct:
            lines.append(f"  default active thread percentage: {pct.strip()}")
    elif status.mode == MODE_CONCURRENT:
        lines.append("  thread-percentage sweep unavailable; only the 100% "
                     "setting will be collected")
    return "\n".join(lines)
