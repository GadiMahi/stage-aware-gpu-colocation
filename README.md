# Stage-Aware GPU Co-location

**Learning Interference for Multi-Stage AI-for-Science Pipelines**

BCSE303P Operating Systems Laboratory · target venue IEEE CCGrid 2027 (paper deadline 1 Dec 2026)

---

## What this is

Workflow schedulers hand out GPUs whole. A four-stage AI-for-science pipeline holds an
entire GPU from its first stage to its last, even though one stage issues no GPU work at
all and another barely fills the device. That capacity could be recovered by co-locating
stages from independent pipelines, which NVIDIA's Multi-Process Service already permits,
but co-location sometimes wrecks both tenants and the damage cannot be derived
analytically: it depends on the joint behaviour of the pair, not on either alone.

This repository measures the interference instead of deriving it, learns a model that
predicts it for pairings never run, and uses that model as the cost function of a
scheduler that shares a device only when it predicts it is safe.

**The contribution is not "learn interference from measurement".** Bubble-Up did that in
2011 and Prophet did it for GPUs in 2017. It is that prediction happens at
*pipeline-stage* granularity, and that because a scientific campaign runs the same stages
thousands of times, a stage can be profiled once and reused forever. Every prior system
has to predict for a workload it has never seen.

---

## MEASURED versus SIMULATED

The most important thing to understand before using any output.

The code runs in two backends, detected automatically:

| Backend | When | What it produces |
|---|---|---|
| `cuda` | NVIDIA GPU present | Real workloads, real counters, real co-location. **MEASURED.** |
| `sim` | No GPU | Analytic interference model, no GPU involved. **SIMULATED.** |

Every dataset, figure and results file carries a provenance stamp, and figures render it
visibly in the corner. **Nothing produced by the `sim` backend is a measurement and it
must never be reported as one.** The simulated path exists so the analysis chain can be
validated before GPU-hours are spent, and so M6 can extrapolate to campaign sizes the
hardware budget cannot reach.

Force a backend with `SAGC_BACKEND=cuda` or `SAGC_BACKEND=sim`.

---

## Quick start

```bash
pip install -r requirements.txt

python -m sagc probe          # what can this machine actually measure?
python -m sagc all            # the whole pipeline end to end
```

Or stage by stage, which is what you want on Kaggle where a session is capped at
twelve hours:

```bash
python -m sagc profile   --reps 3              # M2: profile the zoo solo
python -m sagc sweep     --reps 3              # M3: pairwise co-location dataset
python -m sagc train                           # M4: predictor + ablations A1-A6
python -m sagc schedule  --pipelines 20        # M5/M6: policy sweep, ILP gap
python -m sagc figures                         # all eight paper figures
```

Add `--bandit` to `schedule` to train the contextual-bandit policy (stretch goal, never on
the critical path).

### On Kaggle

Open `notebooks/kaggle_run.ipynb`, set the accelerator to **GPU T4 x2**, and run it. It
finds the code from a Kaggle Dataset or a git clone, probes the environment, and runs the
full pipeline with checkpointing. Expect roughly seven GPU-hours for the complete sweep.

---

## Modules

| ID | Module | What it does | Completion criterion |
|---|---|---|---|
| **M1** | `m1_executor` | Runs the four-stage DAG, N pipelines concurrently, with checkpoint and resume | 20 pipelines unattended for two hours |
| **M2** | `m2_profiler` | Samples DCGM counters at 100 ms, reduces a run to an 18-feature signature | run-to-run variance below 5% |
| **M3** | `m3_dataset` | Executes pairs under MPS, records mutual slowdown, labels VRAM exhaustion | 78 pairings x 2 partitions x 3 reps |
| **M4** | `m4_model` | Gradient-boosted predictor, leave-one-workload-out CV, ablations | beats mean and utilisation baselines |
| **M5** | `m5_scheduler` | Five placement policies, ILP optimality bound, contextual bandit | beats exclusive on makespan |
| **M6** | `m6_sim` | Trace-driven simulator, validated against hardware | within 10% of measured makespan |

### The profiling ladder

Counter availability varies by machine, so M2 degrades explicitly rather than silently:

1. **DCGM** (`dcgmi dmon`), fields 1001/1002/1003/1005. The only source that reports true
   achieved occupancy. This is what you want.
2. **pynvml**, utilisation only. Occupancy is recorded as *missing*, never guessed. The
   counters-versus-utilisation ablation cannot be run at this tier.
3. **simulated**, no GPU.

`python -m sagc probe` tells you which tier you are on and what it costs you.

---

## Why fixed-work runners matter

Every workload performs a predetermined number of iterations, not a fixed duration. This
is the single most consequential design decision in the project:

```
78 combinations x 2 partitions x 3 reps        = 468 co-located runs
x (25 s work + 10 s setup)                     ~ 4.6 GPU-hours
+ solo baselines re-measured per session       ~ 2.1 GPU-hours
                                         TOTAL ~ 6.7 GPU-hours
```

Kaggle grants 30 GPU-hours per week, so this fits comfortably. Let runs drift to three
minutes each and the same experiment costs 45 hours and the deadline is gone.

---

## Measurement hygiene

Three rules the code enforces rather than merely documents:

1. **Solo baselines are re-measured in every session.** Clock behaviour on a shared cloud
   GPU varies between sessions; a slowdown computed against yesterday's baseline is noise
   with a decimal point. `BaselineCache` is keyed on session id.
2. **VRAM exhaustion is a labelled outcome, not a crash.** Which pairs cannot co-reside is
   information the scheduler needs.
3. **Every measured pair yields two training rows** by exchanging tenant roles.
   Interference is structurally symmetric but numerically not, and that asymmetry is the
   signal the model learns.

Cross-validation is **leave-one-workload-out**, not random k-fold. A random split puts
near-duplicates of a test row into training and reports interpolation error dressed up as
generalisation.

---

## Repository layout

```
sagc/
  common/        capability detection, provenance stamping
  workloads/     the 12-workload zoo, torch builders, interference model
  m1_executor/   pipeline DAG, campaign traces, hardware executor
  m2_profiler/   counter sampling, signature extraction
  m3_dataset/    MPS control, pairwise sweep, co-location worker
  m4_model/      predictor, cross-validation, ablations A1-A6
  m5_scheduler/  placement policies, ILP bound, contextual bandit
  m6_sim/        discrete-event simulator, hardware validation
  figures/       the eight paper figures, one function each
data/            signatures.parquet, pairs.parquet (+ provenance json)
results/         model, cv predictions, policy summary, ablation tables
figures/         png + pdf, every one script-generated
tests/           31 tests, including the ones that catch wrong-but-plausible output
```

---

## Results at a glance

Regenerate with `python -m sagc all`. On the simulated backend, 20 pipelines, two modelled
T4s:

| | throughput | vs exclusive | bound violations |
|---|---|---|---|
| Exclusive (status quo) | 77.4 /h | 1.00x | 0.0% |
| Blind sharing | 143.8 /h | 1.86x | **66.7%** |
| **Stage-aware (bound 1.25)** | **149.4 /h** | **1.93x** | **0.0%** |

Leave-one-workload-out MAE 0.093, R2 0.73. Counter features reduce error 22% against
utilisation-only features. Greedy sits within 7% of a single-epoch ILP optimum. All seven
self-consistency checks and 31 tests pass.

Two honest observations from the simulated run, both worth reporting as-is:

- The **utilisation-driven scheduler matches greedy on throughput**. Its predictions are
  measurably worse, but not in a way that changes admission decisions at this scale.
- The **contextual bandit matches greedy exactly**. At two devices and two tenants there
  is little room for a cleverer choice among admissible options.

**These are simulated numbers.** They demonstrate the pipeline works end to end and that
the analysis recovers a relationship genuinely present in the data. The hardware run on
2xT4 is what produces citable results.

---

## Reproducing

```bash
./reproduce.sh          # probe, tests, full pipeline, figures
python -m pytest tests/ -q
```

## Requirements

Python 3.10+. `numpy pandas scikit-learn lightgbm matplotlib pyarrow pulp`. On a GPU
machine additionally `torch`, and optionally `transformers timm torch_geometric rdkit
pynvml`. Every one of those has a pure-PyTorch fallback, so a hub outage or a disabled
internet toggle cannot destroy a measurement campaign. Fallback use is recorded in the run
manifest as `impl="fallback"` and never silently conflated with the real model.
