"""Provenance stamping.

Every artefact this project produces carries a record of how it was obtained.
The distinction that matters is:

    MEASURED   produced by executing real workloads on real GPUs and reading
               real hardware counters. Citable in the paper.
    SIMULATED  produced by the analytic interference model with no GPU
               involved. Useful for validating the analysis pipeline and for
               campaign-scale extrapolation, NEVER citable as a measurement.

Nothing in this codebase may emit a figure, table or dataset without a
provenance stamp, and figures render the stamp visibly. The point of the project
is honest measurement; mislabelling simulated output as measured would destroy
its value entirely.
"""
from __future__ import annotations

import getpass
import json
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional

from . import env

MEASURED = "MEASURED"
SIMULATED = "SIMULATED"
MIXED = "MIXED"

_BANNER = {
    MEASURED: "MEASURED on real hardware",
    SIMULATED: "SIMULATED -- analytic model, not a hardware measurement",
    MIXED: "MIXED -- contains both measured and simulated records",
}


def kind_for_backend(backend: Optional[str] = None) -> str:
    b = backend or env.detect().backend
    return MEASURED if b == env.BACKEND_CUDA else SIMULATED


@dataclass
class Provenance:
    kind: str
    backend: str
    profiler_tier: str
    created_utc: str
    git_commit: str = ""
    host: str = ""
    user: str = ""
    python: str = ""
    platform: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    gpus: list = field(default_factory=list)
    seed: Optional[int] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def banner(self) -> str:
        return _BANNER[self.kind]

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=False)
        if out.returncode == 0:
            return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return ""


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "unknown"


def capture(seed: Optional[int] = None, **notes: Any) -> Provenance:
    """Snapshot the current environment as a provenance record."""
    caps = env.detect()
    return Provenance(
        kind=kind_for_backend(caps.backend),
        backend=caps.backend,
        profiler_tier=caps.profiler_tier(),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=_git_commit(),
        host=socket.gethostname(),
        user=_safe_user(),
        python=platform.python_version(),
        platform=platform.platform(),
        driver_version=caps.driver_version,
        cuda_version=caps.cuda_version,
        gpus=[{"index": g.index, "name": g.name, "uuid": g.uuid,
               "memory_total_mb": g.memory_total_mb} for g in caps.gpus],
        seed=seed,
        notes=dict(notes),
    )


def stamp_dataframe(df, prov: Provenance):
    """Attach provenance columns so a stray CSV stays honest."""
    df = df.copy()
    df["provenance"] = prov.kind
    df["backend"] = prov.backend
    df["profiler_tier"] = prov.profiler_tier
    df["created_utc"] = prov.created_utc
    return df


def dataframe_kind(df) -> str:
    """Read back the provenance of a stamped dataframe."""
    if "provenance" not in df.columns:
        return SIMULATED  # fail closed: unknown provenance is never MEASURED
    vals = set(df["provenance"].dropna().unique())
    if vals == {MEASURED}:
        return MEASURED
    if vals == {SIMULATED}:
        return SIMULATED
    return MIXED


def figure_footer(kind: str, extra: str = "") -> str:
    """Text rendered onto every figure."""
    base = _BANNER.get(kind, _BANNER[SIMULATED])
    return f"{base}{('  |  ' + extra) if extra else ''}"
