# Kaggle runbook

How to take this from a pushed repository to measured results you can put in front of
your guide. Follow it in order. The whole thing is about six GPU-hours of compute spread
over two sessions, plus maybe an hour of your attention.

---

## Before you start

**Budget.** Kaggle gives roughly 30 GPU-hours a week (check the number in the right-hand
sidebar of any notebook; it resets weekly). The full campaign needs about six. Do not
burn the quota on a broken run, which is what Phase 1 below exists to prevent.

**Session limits.** A GPU session is capped at 12 hours. An *interactive* session also
dies if you stop interacting with it, so the five-hour sweep must be run as a **committed
batch job**, not babysat in a browser tab. That is Phase 2.

---

## Phase 0: set the notebook up (5 minutes)

1. kaggle.com → **Create → New Notebook**
2. Right-hand panel → **Settings**:
   - **Accelerator: GPU T4 x2** (not P100, not a single T4, because the project schedules across
     two devices)
   - **Internet: On** (only needed to clone and pip install; the workloads themselves all
     have offline fallbacks)
   - **Persistence: Files only** if offered
3. **File → Import Notebook** and upload `notebooks/kaggle_run.ipynb` from your repo.
4. In cell 1, set your repository URL:

```python
REPO_URL = "https://github.com/gadimahi/stage-aware-gpu-colocation.git"
```

If the repo is private, use a token URL instead and delete it before sharing the notebook:
`https://<token>@github.com/gadimahi/stage-aware-gpu-colocation.git`

---

## Phase 1: the go/no-go run (30 minutes, ~0.4 GPU-hours)

**Run this before the full sweep. Every time.** It costs almost nothing and tells you
whether the expensive run will produce anything usable.

Run cells 1, 2 and 3. **Then calibrate**, which is not optional on a machine
this code has not run on before:

```python
!python -m sagc calibrate
```

The iteration counts declared in the registry are estimates. On a real T4 they
were wrong by up to a factor of twenty, which turns a five-hour campaign into a
seventy-hour one. `calibrate` measures the per-iteration cost on the device in
front of it and solves for the count that hits a 20 second solo run. It writes
`data/calibration.json`, which every later stage reads. Run it once per machine,
before any measurement, so that every solo run and every co-located run in the
campaign does identical work.

Then run a deliberately tiny sweep:

```python
!python -m sagc profile --reps 2 --workloads esm2_35m_infer,mlp_admet_infer,resnet50_train_b32
!python -m sagc sweep   --reps 2 --thread-pcts 100 --workloads esm2_35m_infer,mlp_admet_infer,resnet50_train_b32
!python -m sagc train   --skip-ablations
```

### Read cell 2's output carefully. This is the decision point.

| What it says | What it means | What to do |
|---|---|---|
| `backend: cuda` | Real GPU. Results will be **MEASURED**. | Good. Continue. |
| `backend: sim` | No GPU visible. Everything will be SIMULATED. | Accelerator is not set to GPU. Fix Settings and restart. |
| `profiler tier: dcgm` | True achieved occupancy available. | Best case. Everything works, including ablation A1. |
| `profiler tier: pynvml` | Occupancy **not** available, only coarse utilisation. | See "If DCGM is missing" below. Do not just plough ahead. |
| `co-location mode: mps` | Partition control available. | Run the full `--thread-pcts 100,50`. |
| `co-location mode: concurrent` | No MPS; real contention but no partition control. | Fine. Use `--thread-pcts 100`; ablation A6 will report unavailable. |

### If DCGM is missing

This is the most likely thing to go wrong, and it matters because ablation A1 (counters
versus utilisation) is one of your three contributions. Try, in order:

```bash
!pip install -q nvidia-ml-py            # gets you the pynvml tier at minimum
!apt-get install -y datacenter-gpu-manager 2>/dev/null; which dcgmi
!nv-hostengine 2>/dev/null; dcgmi discovery -l
```

If none of that works, you have two honest options, and you should pick one deliberately
rather than drift:

1. **Run at the pynvml tier anyway.** You get real slowdown measurements (which is the
   dataset, the model and the scheduler, which is most of the project). You lose ablation A1. Say
   so plainly in the review: *"achieved occupancy counters were not available in this
   environment, so the counters-versus-utilisation ablation is reported from simulation
   and flagged as such."* The generated `RESULTS.md` prints this caveat for you.
2. **Get an hour on a machine where DCGM works.** Your department's server, or Colab with
   a runtime that permits it. Run the profile stage there just for the signatures.

Either is defensible. Pretending you had occupancy data when you did not is not.

### Also check

- No stage crashed with a traceback.
- The `impl=` field in the run output. `native` means the real model built; `fallback`
  means a pure-PyTorch stand-in was used because a library or download failed. Fallbacks
  are fine and recorded, but you should know which you got.
- All 31 tests passed in cell 3.

---

## Phase 2: the full campaign (~5 GPU-hours, unattended)

Do **not** run this interactively. Use Kaggle's batch mode so it survives you closing the
laptop.

1. Delete or comment out the Phase-1 subset lines.
2. Make sure cells 4 through 12 have the full arguments (no `--workloads`).
3. Top right → **Save Version → Save & Run All (Commit)** → Save.
4. Close the tab. Kaggle runs the whole notebook top to bottom in the background.
5. Come back in five to six hours. The finished version appears under the notebook's
   **Version history**, with all outputs and files.

If MPS was unavailable in Phase 1, change cell 5 to `--thread-pcts 100`. Pair breadth
matters more than partition depth, because the generalisation claim lives on breadth.

### If the session dies partway

Nothing is lost that was already written. Each stage reads the previous stage's output
from `data/`, so restart from the first stage that did not complete:

```bash
!ls -la data/ results/          # see what exists
!python -m sagc train           # e.g. if the sweep finished but training did not
```

---

## Phase 3: the hardware campaign (~30 minutes)

Cell 9 runs a real campaign under exclusive allocation. **This is Objective 1**: the
measured held-but-idle fraction that motivates the entire project.

Start with `n_pipelines=4`. If it completes cleanly, re-run with 12 or 20 for a better
number. Cell 10 then validates the simulator against it, and saves the result so the
report can pick it up.

---

## Phase 4: collect what you need (10 minutes)

Cell 12 generates `results/RESULTS.md`, a single page with the provenance banner, the
environment, module-by-module completion against each stated criterion, the headline
numbers, every ablation table, and an explicit list of what is still missing. It is
generated, never typed, so it cannot drift from the data.

Download from the **Output** pane:

| File | What it is | Where it goes |
|---|---|---|
| `results/RESULTS.md` | the summary page | hand this to your guide |
| `figures/*.png`, `*.pdf` | all eight paper figures | the deck and the paper |
| `data/pairs.parquet` | the measured dataset | commit it; it is a contribution in itself |
| `data/*.provenance.json` | environment records | proof the run was real |
| `results/policy_summary.csv` | the evaluation matrix | the results table |
| `results/ablation_*.csv` | A1 to A6 | the ablation section |

Then commit the artefacts back to your repo so they are version-controlled with the code
that produced them:

```bash
git add -f data/pairs.parquet data/signatures.parquet data/*.provenance.json \
           results/ figures/
git commit -m "Add measured results from Kaggle 2xT4 run"
git push
```

(`-f` because `.gitignore` excludes generated artefacts by default; you want these
particular ones kept.)

---

## Phase 5: record the demo (20 minutes)

Module Completion is worth 4 marks at Review 2 and the word is *completion*: they want to
see it run, not hear that it ran.

**Record a 90-second screen capture, in advance.** Never demo live off a Kaggle session
in front of a panel, because the session will pick that moment to time out.

Show, in this order:

1. `python -m sagc probe`. 10 seconds. Establishes this is real hardware.
2. `python -m pytest tests/ -q`. 10 seconds. 33 passing tests.
3. A few seconds of the sweep running, with the progress line ticking.
4. `python -m sagc train`. The leave-one-workload-out MAE appearing.
5. `python -m sagc schedule`. The policy sweep table and the headline line.
6. Scroll through `figures/` showing F1, F5 and F7.

Speed up the boring parts in any video editor, or just cut between them. OBS Studio, the
Windows Game Bar (Win+G), or QuickTime all work.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `backend: sim` on Kaggle | accelerator not set to GPU | Settings → Accelerator → GPU T4 x2, restart session |
| `nvidia-cuda-mps-control not found` | MPS not installed in the image | Expected. Falls back to concurrent processes automatically; use `--thread-pcts 100` |
| `ERR_NVGPUCTRPERM` | profiling counters restricted | Use the pynvml tier; see "If DCGM is missing" |
| `impl=fallback` everywhere | internet off, or hub download failed | Turn Internet on, or accept the fallbacks and say so |
| CUDA out of memory during profiling | `vit_large_batch_infer` is deliberately large | It is a capacity stressor and is *meant* to sometimes fail. Only a problem if it happens for every workload |
| Sweep is slower than the estimate | runs drifting past 30 seconds each | Check the per-run times in the progress output. If they have grown, drop `--reps` to 2 before dropping pairings |
| Session died at hour 4 | interactive timeout | Use Save & Run All (Commit), not an interactive session |
| Quota exhausted | ran the full sweep more than once | Wait for the weekly reset. Phase 1 exists to stop this happening |

---

## One-line version

```bash
# Phase 1, go/no-go, ~30 min
python -m sagc probe && python -m pytest tests/ -q
python -m sagc profile --reps 2 --workloads esm2_35m_infer,mlp_admet_infer,resnet50_train_b32
python -m sagc sweep   --reps 2 --thread-pcts 100 --workloads esm2_35m_infer,mlp_admet_infer,resnet50_train_b32
python -m sagc train   --skip-ablations

# Phase 2, the real run, ~5 h, via Save & Run All
python -m sagc profile --reps 3
python -m sagc sweep   --reps 3 --thread-pcts 100,50
python -m sagc train
python -m sagc schedule --pipelines 20
python -m sagc figures
python -m sagc report
```
