"""Command-line entry point.

    python -m sagc probe          what this machine can actually measure
    python -m sagc profile        M2: profile the workload zoo solo
    python -m sagc sweep          M3: pairwise co-location dataset
    python -m sagc train          M4: fit the predictor, run ablations
    python -m sagc schedule       M5/M6: policy sweep and simulator validation
    python -m sagc figures        regenerate all eight figures
    python -m sagc report         one-page results summary for the review
    python -m sagc all            the whole pipeline, in order

Every stage writes its artefacts under data/ and reads the previous stage's
output from there, so stages can be run separately across sessions. That matters
on Kaggle, where a session is capped at twelve hours.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA = Path("data")
FIGS = Path("figures")
RESULTS = Path("results")


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"{path} not found; run the earlier stage first")


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001
        p = path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p


def cmd_probe(args) -> int:
    from .common import env
    from .m2_profiler import sampler
    from .m3_dataset import mps

    _banner("ENVIRONMENT")
    caps = env.detect()
    print(caps.summary())
    print()
    print(sampler.probe_report())
    print()
    status = mps.start()
    print(mps.report(status))
    mps.stop(status)
    if caps.backend == env.BACKEND_SIM:
        print()
        print("NOTE: no GPU detected. Everything produced in this mode is")
        print("      SIMULATED and must never be reported as a measurement.")
    return 0


def cmd_calibrate(args) -> int:
    """Solve for the iteration count that hits the target solo duration.

    Must run before `profile` and `sweep` on any new machine. The declared
    counts in the registry are estimates and were wrong by up to 20x on a T4.
    """
    from .m2_profiler import calibrate as cal

    names = ([w.strip() for w in args.workloads.split(",") if w.strip()]
             if args.workloads else None)
    table = cal.calibrate(names, target_seconds=args.target_seconds,
                          device_index=args.device)
    if table:
        path = cal.save(table)
        print(f"\nwrote {path}")
        print("profile and sweep will now use these counts; delete the file to "
              "fall back to the declared ones")
    return 0


def cmd_profile(args) -> int:
    from .common import provenance
    from .m2_profiler import runner, signature as sig_mod

    _banner(f"M2  PROFILE THE ZOO  (reps={args.reps})")
    t0 = time.time()
    names = ([w.strip() for w in args.workloads.split(",") if w.strip()]
             if args.workloads else None)
    if names:
        print(f"SUBSET RUN: {len(names)} workloads only. Smoke test, not the full zoo.")
    results = runner.profile_zoo(reps=args.reps, device_index=args.device,
                                 interval_ms=args.interval_ms, names=names,
                                 verbose=True)
    sigs = sig_mod.signatures_to_frame([r.signature for r in results])
    num = list(sigs.select_dtypes("number").columns)
    agg = sigs.groupby("workload")[num].median().reset_index()

    prov = provenance.capture(module="M2", reps=args.reps)
    _write(provenance.stamp_dataframe(sigs, prov), DATA / "signature_runs.parquet")
    out = _write(provenance.stamp_dataframe(agg, prov), DATA / "signatures.parquet")
    prov.write(DATA / "signatures.provenance.json")

    # Run-to-run variance is the completion criterion for M2. Occupancy is the
    # preferred basis, but it is unavailable below the dcgm tier, where it is
    # recorded as NaN; reporting "max nan" would leave the criterion silently
    # unevaluated, so fall back to a counter this machine actually measured and
    # say which one was used.
    criterion_col = next(
        (c for c in ("sm_occupancy_mean", "sm_active_mean", "util_gpu_mean")
         if c in sigs.columns and sigs[c].notna().any()), None)
    print()
    print(f"wrote {out}  ({len(agg)} workloads)  in {time.time()-t0:.1f}s")
    if criterion_col is None:
        print("run-to-run variance: NOT EVALUABLE, no counter was measured")
    else:
        var = (sigs.groupby("workload")[criterion_col]
               .agg(["mean", "std"]).assign(cv=lambda d: d["std"] / d["mean"]))
        basis = criterion_col.replace("_mean", "")
        note = "" if criterion_col == "sm_occupancy_mean" else \
            "  (occupancy unavailable at this tier, substituted)"
        print(f"{basis} coefficient of variation: max {var['cv'].max():.3f} "
              f"(criterion: below 0.05){note}")
    return 0


def cmd_sweep(args) -> int:
    from .m3_dataset import pairgen

    _banner(f"M3  PAIRWISE CO-LOCATION SWEEP  (reps={args.reps})")
    workloads = ([w.strip() for w in args.workloads.split(",") if w.strip()]
                 if args.workloads else None)
    cfg = pairgen.SweepConfig(reps=args.reps,
                              thread_pcts=[int(x) for x in args.thread_pcts.split(",")],
                              device_index=args.device, interval_ms=args.interval_ms,
                              workloads=workloads)
    if workloads:
        print(f"SUBSET RUN: {len(workloads)} workloads only. This is a smoke test, "
              f"not the full sweep.")
    t0 = time.time()
    df, prov = pairgen.run_sweep(cfg, verbose=True)
    joined = pairgen.attach_signatures(df, _read(DATA / "signatures.parquet"))
    out = _write(joined, DATA / "pairs.parquet")
    prov.write(DATA / "pairs.provenance.json")

    clean = joined[~joined["oom"].astype(bool)]
    print()
    print(f"wrote {out}  ({len(joined)} rows, {int(joined['oom'].sum())} OOM) "
          f"in {time.time()-t0:.1f}s")
    print(f"slowdown: median {clean['slowdown_a'].median():.3f}  "
          f"p95 {clean['slowdown_a'].quantile(0.95):.3f}  "
          f"max {clean['slowdown_a'].max():.3f}")
    return 0


def cmd_train(args) -> int:
    from .m4_model import ablations, train

    _banner("M4  SLOWDOWN PREDICTOR")
    pairs = _read(DATA / "pairs.parquet")
    sigs = _read(DATA / "signatures.parquet")

    t0 = time.time()
    model = train.fit(pairs, sigs, kind=train.FEATURESET_COUNTERS,
                      model_kind=args.model)
    print(f"trained on {model.train_rows} rows in {time.time()-t0:.1f}s")
    print(f"leave-one-workload-out: MAE={model.cv.mae:.4f} "
          f"RMSE={model.cv.rmse:.4f} R2={model.cv.r2:.4f}")
    gk = train.grouped_kfold(pairs)
    print(f"grouped k-fold        : MAE={gk.mae:.4f} R2={gk.r2:.4f}")
    print()
    print("top features")
    print(model.feature_importance(12).to_string(index=False))

    RESULTS.mkdir(parents=True, exist_ok=True)
    model.save(RESULTS / "model")
    model.cv.predictions.to_csv(RESULTS / "cv_predictions.csv", index=False)

    # the deployable artefact: a precomputed table, not a live model call
    table = model.build_table(thread_pcts=sorted(pairs["thread_pct_a"].unique()))
    pd.DataFrame([{"workload_a": a, "workload_b": b, "thread_pct": p, "slowdown": v}
                  for (a, b, p), v in table.items()]
                 ).to_csv(RESULTS / "slowdown_table.csv", index=False)
    print(f"\nprecomputed {len(table)} pairings into results/slowdown_table.csv")

    if not args.skip_ablations:
        _banner("M4  ABLATIONS")
        for key, r in ablations.run_all(pairs, verbose=True).items():
            r.table.to_csv(RESULTS / f"ablation_{key}.csv", index=False)
    return 0


def cmd_schedule(args) -> int:
    from .common import env
    from .m4_model import train
    from .m5_scheduler import bandit, ilp
    from .m6_sim import simulator as sim, validate as val

    _banner("M5/M6  SCHEDULER AND SIMULATOR")
    pairs = _read(DATA / "pairs.parquet")
    sigs = _read(DATA / "signatures.parquet")
    device_vram = env.device_memory_mb()
    pcts = sorted(pairs["thread_pct_a"].unique())

    model = train.fit(pairs, sigs, kind=train.FEATURESET_COUNTERS, run_cv=False)
    learned = train.ModelPredictor(model.build_table(thread_pcts=pcts), "learned")

    util_model = train.fit(pairs, sigs, kind=train.FEATURESET_UTILISATION, run_cv=False)
    util_pred = train.ModelPredictor(util_model.build_table(thread_pcts=pcts),
                                     "utilisation")

    gt = sim.GroundTruth(device_vram, measured=val.measured_pair_table(pairs))

    print("policy sweep")
    traces, summary = sim.sweep_policies(
        predictor=learned, util_predictor=util_pred, n_pipelines=args.pipelines,
        ground_truth=gt, bounds=(1.05, 1.10, 1.25, 1.50, 999.0), verbose=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "policy_summary.csv", index=False)

    ex = summary[summary.policy == "exclusive"].iloc[0]
    bl = summary[summary.policy == "blind"].iloc[0]
    gr_rows = summary[(summary.policy == "greedy") & (summary.slowdown_bound == 1.25)]
    if not gr_rows.empty:
        gr = gr_rows.iloc[0]
        print()
        print(f"headline: stage-aware at bound 1.25 gives "
              f"{gr.throughput_per_hour/ex.throughput_per_hour:.2f}x the throughput "
              f"of exclusive allocation, with {gr.violation_rate*100:.1f}% violations")
        print(f"          blind sharing reaches "
              f"{bl.throughput_per_hour/ex.throughput_per_hour:.2f}x but violates "
              f"{bl.violation_rate*100:.1f}% of the time")

    print()
    print("self-consistency checks")
    checks = val.self_consistency_check(n_pipelines=args.pipelines,
                                        device_vram_mb=device_vram)
    print(checks.to_string(index=False))
    checks.to_csv(RESULTS / "self_consistency.csv", index=False)
    if not checks["passed"].all():
        print("WARNING: a self-consistency check failed")

    if not args.skip_ilp:
        print()
        print("greedy optimality gap (single-epoch ILP)")
        gap = ilp.optimality_study(learned, device_vram_mb=device_vram,
                                   trials=args.ilp_trials)
        print(gap.groupby("n_ready")["gap"].agg(["mean", "max"]).to_string())
        gap.to_csv(RESULTS / "ilp_optimality.csv", index=False)
        print(f"mean gap across all instances: {gap['gap'].mean()*100:.2f}%")

    if args.bandit:
        print()
        print("contextual bandit (stretch, trained in simulation)")
        bp, hist = bandit.train_bandit(learned, episodes=args.bandit_episodes,
                                       n_pipelines=args.pipelines,
                                       device_vram_mb=device_vram,
                                       ground_truth=gt, verbose=True)
        tr = sim.simulate(bp, n_pipelines=args.pipelines,
                          device_vram_mb=device_vram, ground_truth=gt,
                          slowdown_bound=1.25)
        hist.to_csv(RESULTS / "bandit_history.csv", index=False)
        row = tr.summary()
        base = gr_rows.iloc[0].throughput_per_hour if not gr_rows.empty else float("nan")
        print(f"  final: tput={row['throughput_per_hour']:.2f}/h  "
              f"viol={row['violation_rate']:.3f}   greedy was {base:.2f}/h")

    # keep the exclusive trace for figure F1
    next(t for t in traces if t.policy == "exclusive").save(RESULTS, tag="exclusive")
    return 0


def cmd_figures(args) -> int:
    from .common import provenance
    from .figures import make_all as figs
    from .m1_executor import dag
    from .m4_model import ablations

    _banner("FIGURES")
    pairs = _read(DATA / "pairs.parquet")
    sigs = _read(DATA / "signatures.parquet")
    summary = pd.read_csv(RESULTS / "policy_summary.csv")
    cvp = pd.read_csv(RESULTS / "cv_predictions.csv")

    a1_path = RESULTS / "ablation_A1.csv"
    a1 = (pd.read_csv(a1_path) if a1_path.exists()
          else ablations.a1_feature_set(pairs).table)

    stages = pd.read_csv(RESULTS / "trace_exclusive.csv")
    trace = dag.CampaignTrace(
        stages=stages, policy="exclusive",
        n_pipelines=int(stages["pipeline_id"].nunique()),
        n_devices=2, device_vram_mb=15360,
        makespan_s=float(stages["end_s"].max() - stages["start_s"].min()),
        slowdown_bound=999.0, backend="sim")

    kind = provenance.dataframe_kind(pairs)
    made = figs.make_all(trace, sigs, pairs, cvp, a1, summary, kind, FIGS)
    print(f"\n{len(made)} figures written to {FIGS}/  (stamped {kind})")
    return 0


def cmd_report(args) -> int:
    from . import report

    _banner("RESULTS SUMMARY")
    out = report.build()
    print(f"wrote {out}")
    print()
    print(out.read_text()[:1200])
    print("...")
    return 0


def cmd_all(args) -> int:
    for fn in (cmd_probe, cmd_calibrate, cmd_profile, cmd_sweep, cmd_train, cmd_schedule,
               cmd_figures, cmd_report):
        rc = fn(args)
        if rc != 0:
            return rc
    _banner("DONE")
    print("artefacts: data/  results/  figures/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they are accepted both before
    # and after the subcommand. Students will type them in either order.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--reps", type=int, default=3)
    common.add_argument("--device", type=int, default=0)
    common.add_argument("--interval-ms", type=int, default=100)
    common.add_argument("--thread-pcts", default="100,50")
    common.add_argument("--workloads", default=None,
                        help="comma-separated subset for a smoke test, e.g. "
                             "esm2_35m_infer,mlp_admet_infer,resnet50_train_b32")
    common.add_argument("--model", default="lgbm")
    common.add_argument("--pipelines", type=int, default=20)
    common.add_argument("--skip-ablations", action="store_true")
    common.add_argument("--target-seconds", type=float, default=20.0,
                        help="target solo wall time per workload for calibrate")
    common.add_argument("--skip-ilp", action="store_true")
    common.add_argument("--ilp-trials", type=int, default=6)
    common.add_argument("--bandit", action="store_true",
                        help="train the contextual-bandit policy (stretch goal)")
    common.add_argument("--bandit-episodes", type=int, default=10)

    ap = argparse.ArgumentParser(prog="sagc", parents=[common],
                                 description="Stage-aware GPU co-location")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, fn in [("probe", cmd_probe), ("calibrate", cmd_calibrate),
                     ("profile", cmd_profile),
                     ("sweep", cmd_sweep), ("train", cmd_train),
                     ("schedule", cmd_schedule), ("figures", cmd_figures),
                     ("report", cmd_report), ("all", cmd_all)]:
        sub.add_parser(name, parents=[common]).set_defaults(func=fn)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
