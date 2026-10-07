"""Test configuration.

THE TEST SUITE MUST NEVER TOUCH THE GPU.

Every check in this suite is about logic: that the scheduler does not drop
stages, that a bounded policy respects its bound, that VRAM capacity is never
exceeded, that a slowdown ratio is never below one, that an unstamped frame is
never read as measured. None of that needs hardware, and all of it must give
the same answer on a laptop, in CI, and on a GPU node.

Left unpinned, `small_dataset` detects CUDA and runs a genuine five-workload,
fifteen-pairing, two-partition, two-repetition campaign on the device. On a
2x T4 Kaggle session that is tens of minutes of billed GPU time spent to
re-establish facts about control flow, and it is the reason `scripts/smoke_test.py`
timed out before reaching the stages that actually needed measuring.

Forcing the simulated backend here also makes the suite deterministic, which is
what `test_simulator_is_deterministic` and `test_profiler_is_reproducible_within_tolerance`
depend on. Hardware measurement is exercised by the `profile` and `sweep`
commands, not by pytest.
"""
from __future__ import annotations

import os

# Must be set before anything imports sagc.common.env, whose detect() is
# lru_cached and reads this variable once.
os.environ["SAGC_BACKEND"] = "sim"

from sagc.common import env  # noqa: E402

env.detect.cache_clear()


def pytest_report_header(config):  # noqa: ARG001
    caps = env.detect()
    return (f"sagc: backend pinned to {caps.backend!r} for the test suite "
            f"(tests never use the GPU)")
