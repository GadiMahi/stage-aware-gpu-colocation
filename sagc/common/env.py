"""Environment and capability detection.

The project targets NVIDIA GPUs with MPS and hardware performance counters. Not
every environment provides all of that, so every capability is probed once and
the rest of the codebase branches on the result rather than assuming. This is
what lets the identical code path run on a Kaggle 2xT4 session and on a CPU-only
machine.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from typing import List, Optional

BACKEND_CUDA = "cuda"          # real GPU, real counters, real co-location
BACKEND_SIM = "sim"            # analytic model, no GPU required

#: Environment variable that forces a backend regardless of what is detected.
BACKEND_ENV = "SAGC_BACKEND"


@dataclass
class GPUInfo:
    index: int
    name: str
    uuid: str
    memory_total_mb: int
    compute_capability: Optional[str] = None


@dataclass
class Capabilities:
    """Everything the pipeline needs to know about where it is running."""

    backend: str
    has_nvidia_smi: bool = False
    has_torch: bool = False
    has_torch_cuda: bool = False
    gpus: List[GPUInfo] = field(default_factory=list)

    # profiling ladder, in descending order of preference
    has_dcgm: bool = False          # dcgmi -- best: true occupancy counters
    has_ncu: bool = False           # Nsight Compute -- per-kernel counters
    has_nsys: bool = False          # Nsight Systems -- kernel timeline
    has_pynvml: bool = False        # coarse utilisation, always a fallback

    # co-location mechanism
    has_mps_control: bool = False
    mps_running: bool = False

    driver_version: str = ""
    cuda_version: str = ""

    def profiler_tier(self) -> str:
        """Which profiling source will actually be used."""
        if self.backend == BACKEND_SIM:
            return "simulated"
        if self.has_dcgm:
            return "dcgm"
        if self.has_pynvml:
            return "pynvml"
        return "wallclock"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["profiler_tier"] = self.profiler_tier()
        return d

    def summary(self) -> str:
        lines = [
            f"backend          : {self.backend}",
            f"profiler tier    : {self.profiler_tier()}",
            f"GPUs             : {len(self.gpus)}",
        ]
        for g in self.gpus:
            lines.append(f"  [{g.index}] {g.name}  {g.memory_total_mb} MB  {g.uuid[:20]}")
        lines += [
            f"torch / cuda     : {self.has_torch} / {self.has_torch_cuda}",
            f"dcgm ncu nsys    : {self.has_dcgm} {self.has_ncu} {self.has_nsys}",
            f"pynvml           : {self.has_pynvml}",
            f"MPS ctrl/running : {self.has_mps_control} / {self.mps_running}",
            f"driver / cuda    : {self.driver_version or '-'} / {self.cuda_version or '-'}",
        ]
        return "\n".join(lines)


def _run(cmd: List[str], timeout: int = 10) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        if out.returncode == 0:
            return out.stdout
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def _probe_gpus() -> tuple[List[GPUInfo], str, str]:
    gpus: List[GPUInfo] = []
    driver = cuda = ""
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if out:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(GPUInfo(
                    index=int(parts[0]), name=parts[1], uuid=parts[2],
                    memory_total_mb=int(float(parts[3])),
                ))
                driver = parts[4]
    ver = _run(["nvcc", "--version"])
    if ver:
        for tok in ver.split():
            if tok.startswith("V") and tok[1:2].isdigit():
                cuda = tok[1:]
                break
    return gpus, driver, cuda


def _probe_dcgm() -> bool:
    """dcgmi must exist AND be able to enumerate devices."""
    if not shutil.which("dcgmi"):
        return False
    return _run(["dcgmi", "discovery", "-l"], timeout=20) is not None


def _probe_mps() -> tuple[bool, bool]:
    has_ctl = shutil.which("nvidia-cuda-mps-control") is not None
    if not has_ctl:
        return False, False
    try:
        p = subprocess.run(["nvidia-cuda-mps-control"],
                           input="get_default_active_thread_percentage\n",
                           capture_output=True, text=True, timeout=8)
        running = p.returncode == 0 and bool(p.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        running = False
    return True, running


@lru_cache(maxsize=1)
def detect() -> Capabilities:
    """Probe the machine once and cache the answer."""
    forced = os.environ.get(BACKEND_ENV, "").strip().lower()

    has_smi = shutil.which("nvidia-smi") is not None
    gpus, driver, cuda = _probe_gpus() if has_smi else ([], "", "")

    has_torch = has_cuda = False
    try:
        import torch  # noqa: F401

        has_torch = True
        has_cuda = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass

    has_pynvml = False
    try:
        import pynvml  # noqa: F401

        has_pynvml = True
    except Exception:  # noqa: BLE001
        pass

    has_ctl, mps_running = _probe_mps() if has_smi else (False, False)

    if forced in (BACKEND_CUDA, BACKEND_SIM):
        backend = forced
    elif gpus and has_cuda:
        backend = BACKEND_CUDA
    else:
        backend = BACKEND_SIM

    return Capabilities(
        backend=backend,
        has_nvidia_smi=has_smi,
        has_torch=has_torch,
        has_torch_cuda=has_cuda,
        gpus=gpus,
        has_dcgm=_probe_dcgm() if has_smi else False,
        has_ncu=shutil.which("ncu") is not None,
        has_nsys=shutil.which("nsys") is not None,
        has_pynvml=has_pynvml,
        has_mps_control=has_ctl,
        mps_running=mps_running,
        driver_version=driver,
        cuda_version=cuda,
    )


def is_simulated() -> bool:
    return detect().backend == BACKEND_SIM


def n_devices(default_sim: int = 2) -> int:
    """Number of schedulable devices. Simulated runs model a 2-GPU node."""
    caps = detect()
    if caps.backend == BACKEND_CUDA and caps.gpus:
        return len(caps.gpus)
    return int(os.environ.get("SAGC_SIM_GPUS", default_sim))


def device_memory_mb(default_sim: int = 15360) -> int:
    """Usable VRAM per device. A T4 reports 15360 MB usable of its 16 GB."""
    caps = detect()
    if caps.backend == BACKEND_CUDA and caps.gpus:
        return min(g.memory_total_mb for g in caps.gpus)
    return int(os.environ.get("SAGC_SIM_VRAM_MB", default_sim))


if __name__ == "__main__":  # pragma: no cover
    caps = detect()
    print(caps.summary())
    print()
    print(json.dumps(caps.to_dict(), indent=2))
