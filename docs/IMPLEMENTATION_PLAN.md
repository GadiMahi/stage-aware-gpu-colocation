# Implementation plan and build notes

---

## The constraint that shaped everything

The project needs NVIDIA GPUs, MPS, and hardware performance counters. The container this
code was written in has none of those: two CPUs, no CUDA. So "implement the project" had
to mean something precise rather than something vague.

It means: **write the code that runs for real on Kaggle's 2x T4 with no changes, and
validate the entire analysis chain before a single GPU-hour is spent.**

That is not a compromise. It is the order a measurement study should be built in anyway.
The expensive, irreversible part of this project is the pairwise sweep; discovering a bug
in the feature extractor *after* burning five GPU-hours is the failure mode to design
against. Every stage therefore has two backends behind one interface.

---

## Architecture: one interface, two backends

```
             detect() probes the machine once
                         |
        +----------------+----------------+
        |                                 |
   backend=cuda                      backend=sim
   real torch workloads              analytic interference model
   real DCGM counters                generated counter traces
   real MPS co-location              closed-form slowdown
        |                                 |
        +----------------+----------------+
                         |
            identical downstream pipeline
      signatures -> dataset -> model -> scheduler -> figures
                         |
              provenance stamp: MEASURED | SIMULATED
```

The downstream half never branches on backend. A figure generated from simulated data and
one generated from measured data differ only in the stamp rendered in the corner, so the
hardware run on Kaggle produces publishable figures with no code changes at all.

---

## Build order and why

| Step | Module | Rationale for the position |
|---|---|---|
| 1 | `common/env` | Everything branches on capability detection, so it comes first. |
| 2 | `common/provenance` | Built before any artefact exists, so nothing can be produced unstamped. |
| 3 | `workloads/registry` | The zoo definition is the contract the rest of the code reads. |
| 4 | `workloads/interference` | The analytic ground truth the simulated backend needs. |
| 5 | `workloads/torch_workloads` | Real GPU workloads with offline fallbacks. |
| 6 | `m2_profiler` | Signatures are the input to everything downstream. |
| 7 | `m3_dataset` | Needs the profiler; produces the dataset. |
| 8 | `m4_model` | Needs the dataset. |
| 9 | `m5_scheduler` | Needs the model as a cost function. |
| 10 | `m6_sim` | Needs policies to simulate. |
| 11 | `m1_executor` | Needs a policy to execute; its simulated counterpart is M6. |
| 12 | `figures`, `tests` | Need everything. |

---

## Decisions worth recording

**Fixed work, not fixed time.** Every workload runs a predetermined iteration count. This
makes slowdown a clean ratio and bounds the sweep to ~7 GPU-hours instead of ~45. The
single most consequential decision in the project.

**Two rows per measurement.** Each measured pair is emitted twice with tenant roles
exchanged. Interference is structurally symmetric but numerically asymmetric, and that
asymmetry is exactly what the model must learn.

**Leave-one-workload-out, not k-fold.** A random split puts a row's role-swapped twin and
its repetitions into training, so reported error would describe interpolation. Holding out
an entire workload is the only evaluation that answers the deployment question.

**VRAM exhaustion is a label, not an exception.** Four of the 78 pairings cannot
co-reside. Which ones is information the scheduler needs, so `oom=True` is a first-class
outcome in the dataset.

**The predictor is consumed as a precomputed table.** Because stage identity is known
before a campaign starts, every pairing is evaluated offline once and looked up in
constant time. Model inference never enters the scheduling path. This is the concrete
answer to "isn't ML too slow for scheduling".

**A fluid model in the simulator.** Fixing a stage's slowdown at placement time would be
wrong, because residency changes while a stage runs. The simulator tracks remaining solo
work and consumes it at 1/slowdown, recomputing whenever residency changes.

**CPU stages hold the device under exclusive allocation.** Not a modelling artefact: the
GPU genuinely is reserved while featurisation runs on the host, and that held-but-idle
time is precisely what Objective 1 measures. An early version failed to *release* that
slot on completion, which deadlocked the exclusive policy and made it look like a
0.2-second campaign. Guarded now by `test_every_stage_completes_under_every_policy`.

---

## A modelling error that was caught and fixed

The first version modelled `nvidia-smi` utilisation as `0.86 + 0.14 * occupancy`. Ablation
A1 then reported that **utilisation features beat counter features**, the exact opposite
of the project's third contribution.

That was not a result; it was a bug. NVML defines `utilization.gpu` as the fraction of the
sampling window during which at least one kernel was resident, so any workload submitting
work back to back reads high *regardless of how much of the device it fills*. Making it
proportional to occupancy quietly turned it into a perfect proxy.

The corrected model compresses utilisation into roughly 0.95 to 0.99 across a zoo whose
occupancy ranges 0.07 to 0.78. A1 then reports counters reducing error by 22 per cent,
which follows from the semantics rather than from a coincidence.

`tests/test_sagc.py::test_utilisation_does_not_track_occupancy` guards this, because it is
an easy mistake to reintroduce and it silently inverts a headline claim.

The same pass also introduced a modelled `SM_ACTIVE` that sits *between* utilisation and
occupancy, since DCGM field 1002 measures the fraction of time at least one warp is
resident averaged over SMs: a tiny-kernel workload reads low there while utilisation still
reads 99 per cent. That intermediate signal is what makes the profiler's three-way
distinction meaningful.

**This still needs confirming on hardware.** The simulated A1 result demonstrates that the
analysis recovers the relationship when it is present; it is not evidence that the
relationship is present on a T4. Only the Kaggle run settles that.

---

## Results from the simulated backend

20 pipelines, two modelled T4s, measured-pair ground truth:

| policy | throughput | vs exclusive | violations |
|---|---|---|---|
| exclusive | 77.4 /h | 1.00x | 0.0% |
| blind sharing | 143.8 /h | 1.86x | 66.7% |
| stage-aware, bound 1.25 | 149.4 /h | 1.93x | 0.0% |

Leave-one-workload-out MAE 0.093, R2 0.73. A1: counters 0.093 vs utilisation 0.120.
Greedy within 6.8% of a single-epoch ILP optimum. Seven self-consistency checks and 31
tests pass.

**Two null results worth reporting honestly:**

- The utilisation-driven scheduler matches greedy on throughput. Its predictions are
  measurably worse, but not in a way that changes admission decisions at this scale. The
  value of counter features shows up in prediction error, not yet in placement.
- The contextual bandit matches greedy exactly. With two devices and two tenants per
  device there is very little room for a cleverer choice among admissible options.

Neither weakens the paper. Both are the kind of finding a reviewer respects more than a
suspiciously clean sweep.

---

## What is genuinely unfinished

| Item | Status | Needed for |
|---|---|---|
| Hardware measurement of everything | **not started** — needs the Kaggle run | every citable number |
| A5 cross-architecture ablation | reports "not available" by design | needs a second GPU generation (Colab L4) |
| Multi-tenant beyond pairs | approximated by aggregating co-tenant pressure | 3-way residency is out of scope |
| ILP bound | single-epoch only | stated as such; not a campaign-level claim |
| Contextual bandit | implemented, trains in simulation | stretch goal, explicitly off the critical path |

The ILP limitation is worth being careful about in the paper. It bounds the per-epoch
optimality gap of the greedy heuristic. It proves nothing about the campaign-level
schedule, and saying otherwise would be overclaiming.

---

## Running order on Kaggle

```
probe        ->  confirms backend=cuda, profiler tier, MPS mode      (seconds)
pytest       ->  31 tests before spending anything                   (seconds)
profile      ->  M2, twelve workloads x 3 reps                       (~25 min)
sweep        ->  M3, 468 co-located runs                             (~5 h)
train        ->  M4, predictor + ablations                           (minutes)
schedule     ->  M5/M6, policy sweep + ILP gap                       (minutes)
figures      ->  all eight                                           (seconds)
characterise ->  M1, hardware campaign under exclusive allocation    (~30 min)
validate     ->  simulator vs hardware, 10 per cent criterion        (seconds)
```

Stages read the previous stage's output from `data/`, so an interrupted session resumes
from the last completed stage rather than from the beginning.
