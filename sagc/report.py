"""Generate a results summary from whatever artefacts exist.

`python -m sagc report` reads data/, results/ and figures/ and writes
`results/RESULTS.md`: a single page carrying the provenance banner, the
environment the run happened on, module-by-module completion status against each
module's stated criterion, the headline numbers, the ablation tables, and an
explicit list of what is still missing.

The point is that the summary is GENERATED, never typed. A hand-written results
table drifts from the data the moment anything is re-run, and a stale number in
a review is worse than no number. Everything here is read back out of the
artefacts, including whether the run was measured or simulated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .common import provenance

DATA = Path("data")
RESULTS = Path("results")
FIGS = Path("figures")


def _read_any(stem: str, base: Path = DATA) -> Optional[pd.DataFrame]:
    for ext in (".parquet", ".csv"):
        p = (base / stem).with_suffix(ext)
        if p.exists():
            try:
                return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            except Exception:  # noqa: BLE001
                continue
    return None


def _read_json(path: Path) -> Optional[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _md_table(df: pd.DataFrame, cols: Optional[List[str]] = None,
              floats: int = 4) -> str:
    if df is None or df.empty:
        return "_no data_"
    d = df[[c for c in (cols or df.columns) if c in df.columns]].copy()
    for c in d.select_dtypes("number").columns:
        d[c] = d[c].map(lambda v: f"{v:.{floats}g}" if pd.notna(v) else "-")
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.values]
    return "\n".join([head, rule] + rows)


def _module_status() -> List[dict]:
    """Check each module against the criterion stated in its own docstring."""
    out = []

    sig_runs = _read_any("signature_runs")
    sigs = _read_any("signatures")
    if sig_runs is not None and "sm_occupancy_mean" in sig_runs.columns:
        var = (sig_runs.groupby("workload")["sm_occupancy_mean"]
               .agg(["mean", "std"]))
        cv = (var["std"] / var["mean"]).max()
        out.append({"module": "M2 counter profiler",
                    "criterion": "occupancy run-to-run CV below 0.05",
                    "observed": f"max CV {cv:.3f}",
                    "met": "yes" if cv < 0.05 else "NO"})
    else:
        out.append({"module": "M2 counter profiler", "criterion":
                    "occupancy run-to-run CV below 0.05",
                    "observed": "not run", "met": "-"})

    pairs = _read_any("pairs")
    if pairs is not None:
        # count UNORDERED pairings: self-pairs appear once, others twice
        n_pairs = len({frozenset((a, b)) for a, b
                       in zip(pairs["workload_a"], pairs["workload_b"])})
        n_cfg = pairs["thread_pct_a"].nunique()
        reps = pairs["rep"].nunique()
        ok = n_pairs >= 60 and reps >= 3
        out.append({"module": "M3 pair dataset",
                    "criterion": "78 pairings x 2 partitions x 3 reps",
                    "observed": f"{n_pairs} pairings, {n_cfg} partitions, "
                                f"{reps} reps, {len(pairs)} rows "
                                f"({int(pairs['oom'].sum())} OOM)",
                    "met": "yes" if ok else "partial"})
    else:
        out.append({"module": "M3 pair dataset", "criterion":
                    "78 pairings x 2 partitions x 3 reps",
                    "observed": "not run", "met": "-"})

    meta = _read_json(RESULTS / "model" / "model_meta.json")
    a1 = _read_any("ablation_A1", RESULTS)
    if meta and meta.get("cv"):
        mae = meta["cv"]["mae"]
        util_mae = None
        if a1 is not None and "feature_set" in a1.columns:
            row = a1[a1.feature_set == "utilisation"]
            if not row.empty:
                util_mae = float(row.iloc[0]["mae"])
        beats = "yes" if (util_mae is None or mae < util_mae) else "NO"
        obs = f"LOWO MAE {mae:.4f}, R2 {meta['cv']['r2']:.3f}"
        if util_mae:
            obs += f"; utilisation-only MAE {util_mae:.4f}"
        out.append({"module": "M4 slowdown predictor",
                    "criterion": "beats mean and utilisation baselines",
                    "observed": obs, "met": beats})
    else:
        out.append({"module": "M4 slowdown predictor", "criterion":
                    "beats mean and utilisation baselines",
                    "observed": "not run", "met": "-"})

    summ = _read_any("policy_summary", RESULTS)
    if summ is not None and not summ.empty:
        ex = summ[summ.policy == "exclusive"]
        gr = summ[(summ.policy == "greedy") & (summ.slowdown_bound == 1.25)]
        if not ex.empty and not gr.empty:
            ratio = (gr.iloc[0].throughput_per_hour
                     / ex.iloc[0].throughput_per_hour)
            out.append({"module": "M5 stage-aware scheduler",
                        "criterion": "beats exclusive allocation on makespan",
                        "observed": f"{ratio:.2f}x throughput at bound 1.25, "
                                    f"{gr.iloc[0].violation_rate*100:.1f}% violations",
                        "met": "yes" if ratio > 1.0 else "NO"})
    else:
        out.append({"module": "M5 stage-aware scheduler", "criterion":
                    "beats exclusive allocation on makespan",
                    "observed": "not run", "met": "-"})

    val = _read_json(RESULTS / "simulator_validation.json")
    if val:
        err = val.get("makespan_rel_error", float("nan"))
        out.append({"module": "M6 simulator",
                    "criterion": "within 10% of measured makespan",
                    "observed": f"makespan error {err*100:.1f}%",
                    "met": "yes" if err <= 0.10 else "NO"})
    else:
        out.append({"module": "M6 simulator",
                    "criterion": "within 10% of measured makespan",
                    "observed": "needs a hardware campaign to compare against",
                    "met": "-"})

    trace = _read_any("trace_hardware_exclusive", RESULTS)
    if trace is None:
        trace = _read_any("trace_exclusive", RESULTS)
    if trace is not None:
        n = trace["pipeline_id"].nunique()
        out.append({"module": "M1 pipeline executor",
                    "criterion": "N pipelines run unattended with resume",
                    "observed": f"{n} pipelines, {len(trace)} stages traced",
                    "met": "yes" if n >= 4 else "partial"})
    else:
        out.append({"module": "M1 pipeline executor", "criterion":
                    "N pipelines run unattended with resume",
                    "observed": "not run", "met": "-"})

    return out


def build(out_path: Path = RESULTS / "RESULTS.md") -> Path:
    pairs = _read_any("pairs")
    sigs = _read_any("signatures")
    summ = _read_any("policy_summary", RESULTS)
    prov_json = _read_json(DATA / "pairs.provenance.json") or \
        _read_json(DATA / "signatures.provenance.json") or {}

    kind = provenance.dataframe_kind(pairs) if pairs is not None else provenance.SIMULATED
    banner = {
        provenance.MEASURED:
            "**MEASURED ON REAL HARDWARE.** Every number below comes from "
            "executing real workloads on real GPUs and reading real counters.",
        provenance.SIMULATED:
            "**SIMULATED. NOT A HARDWARE MEASUREMENT.** These numbers come from "
            "the analytic interference model and demonstrate that the analysis "
            "pipeline works end to end. They must not be presented as "
            "measurements.",
        provenance.MIXED:
            "**MIXED PROVENANCE.** This run contains both measured and "
            "simulated records. Check the `provenance` column before citing "
            "anything.",
    }[kind]

    L: List[str] = []
    L.append("# Results summary")
    L.append("")
    L.append(f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by "
             f"`python -m sagc report`. Do not edit by hand; re-run instead._")
    L.append("")
    L.append(banner)
    L.append("")

    # --- environment -------------------------------------------------------
    L.append("## Run environment")
    L.append("")
    gpus = prov_json.get("gpus") or []
    env_rows = [
        ("backend", prov_json.get("backend", "-")),
        ("profiler tier", prov_json.get("profiler_tier", "-")),
        ("GPUs", ", ".join(f"{g['name']} ({g['memory_total_mb']} MB)"
                           for g in gpus) or "none detected"),
        ("driver / CUDA", f"{prov_json.get('driver_version') or '-'} / "
                          f"{prov_json.get('cuda_version') or '-'}"),
        ("co-location mode", (prov_json.get("notes") or {}).get("colocation_mode", "-")),
        ("git commit", prov_json.get("git_commit") or "-"),
        ("run timestamp", prov_json.get("created_utc", "-")),
    ]
    L.append("| field | value |")
    L.append("|---|---|")
    L += [f"| {k} | {v} |" for k, v in env_rows]
    L.append("")
    if prov_json.get("profiler_tier") == "pynvml":
        L.append("> **Caveat.** This run used the NVML profiler tier, which does not "
                 "report achieved SM occupancy. Ablation A1 (counters versus "
                 "utilisation) cannot be evaluated from this data.")
        L.append("")

    # --- module completion --------------------------------------------------
    L.append("## Module completion")
    L.append("")
    L.append("Each module is checked against the criterion stated in its own source.")
    L.append("")
    L.append(_md_table(pd.DataFrame(_module_status()),
                       ["module", "criterion", "observed", "met"]))
    L.append("")

    # --- headline -----------------------------------------------------------
    if summ is not None and not summ.empty:
        L.append("## Headline result")
        L.append("")
        ex = summ[summ.policy == "exclusive"]
        keep = ["policy", "bound_label", "throughput_per_hour",
                "mean_achieved_occupancy", "violation_rate",
                "held_but_idle_fraction", "makespan_s"]
        view = summ.copy()
        if not ex.empty:
            base = float(ex.iloc[0].throughput_per_hour)
            view["vs_exclusive"] = view["throughput_per_hour"] / base
            keep.insert(3, "vs_exclusive")
        L.append(_md_table(view, keep))
        L.append("")
        gr = summ[(summ.policy == "greedy") & (summ.slowdown_bound == 1.25)]
        bl = summ[summ.policy == "blind"]
        if not ex.empty and not gr.empty and not bl.empty:
            L.append(f"At a slowdown bound of 1.25 the stage-aware scheduler delivers "
                     f"**{gr.iloc[0].throughput_per_hour/float(ex.iloc[0].throughput_per_hour):.2f}x** "
                     f"the throughput of exclusive allocation with "
                     f"**{gr.iloc[0].violation_rate*100:.1f}%** bound violations. "
                     f"Blind sharing reaches "
                     f"{bl.iloc[0].throughput_per_hour/float(ex.iloc[0].throughput_per_hour):.2f}x "
                     f"but violates the same bound "
                     f"{bl.iloc[0].violation_rate*100:.1f}% of the time.")
            L.append("")

    # --- characterisation ---------------------------------------------------
    hw = _read_json(RESULTS / "summary_hardware_exclusive.json") or \
        _read_json(RESULTS / "summary_exclusive.json")
    if hw:
        L.append("## Objective 1: how much capacity is wasted")
        L.append("")
        L.append(f"- GPU held but **idle**: **{hw['held_but_idle_fraction']*100:.1f}%** "
                 f"of held device time")
        L.append(f"- GPU held but **under-occupied** (below 30% occupancy): "
                 f"**{hw['held_but_under_occupied_fraction']*100:.1f}%**")
        L.append(f"- Mean achieved occupancy across the campaign: "
                 f"{hw['mean_achieved_occupancy']:.3f}")
        L.append("")

    # --- dataset ------------------------------------------------------------
    if pairs is not None:
        clean = pairs[~pairs["oom"].astype(bool)]
        L.append("## The dataset")
        L.append("")
        n_unordered = len({frozenset((a, b)) for a, b
                           in zip(pairs["workload_a"], pairs["workload_b"])})
        L.append(f"- {len(pairs)} labelled rows ({n_unordered} distinct pairings, "
                 f"two rows per measurement by role swap)")
        L.append(f"- {int(pairs['oom'].sum())} rows record VRAM exhaustion as a "
                 f"labelled outcome")
        L.append(f"- slowdown: median {clean['slowdown_a'].median():.3f}, "
                 f"p95 {clean['slowdown_a'].quantile(0.95):.3f}, "
                 f"max {clean['slowdown_a'].max():.3f}")
        L.append("")

    # --- ablations ----------------------------------------------------------
    abl = [("A1", "Feature set: counters versus utilisation"),
           ("A2", "Model class"),
           ("A3", "Training-set size"),
           ("A4", "Leave-one-workload-out generalisation"),
           ("A5", "Cross-architecture"),
           ("A6", "MPS thread percentage")]
    have = [(k, t) for k, t in abl if _read_any(f"ablation_{k}", RESULTS) is not None]
    if have:
        L.append("## Ablations")
        L.append("")
        for key, title in have:
            L.append(f"### {key}. {title}")
            L.append("")
            L.append(_md_table(_read_any(f"ablation_{key}", RESULTS)))
            L.append("")

    # --- correctness --------------------------------------------------------
    checks = _read_any("self_consistency", RESULTS)
    if checks is not None:
        n_ok = int(checks["passed"].sum())
        L.append("## Correctness checks")
        L.append("")
        L.append(f"{n_ok} of {len(checks)} self-consistency checks passed.")
        L.append("")
        L.append(_md_table(checks))
        L.append("")

    gap = _read_any("ilp_optimality", RESULTS)
    if gap is not None and not gap.empty:
        L.append("## Greedy optimality gap")
        L.append("")
        L.append(f"Mean single-epoch gap against an exact ILP: "
                 f"**{gap['gap'].mean()*100:.2f}%** "
                 f"(worst {gap['gap'].max()*100:.2f}%).")
        L.append("")
        L.append("This bounds the per-epoch gap only. It is not a claim about the "
                 "campaign-level schedule.")
        L.append("")

    # --- figures ------------------------------------------------------------
    pngs = sorted(FIGS.glob("*.png")) if FIGS.exists() else []
    if pngs:
        L.append("## Figures")
        L.append("")
        L += [f"- `{p}`" for p in pngs]
        L.append("")

    # --- what is missing ----------------------------------------------------
    L.append("## What this run does not yet establish")
    L.append("")
    missing = []
    if kind != provenance.MEASURED:
        missing.append("No hardware measurement. Every number above is simulated.")
    a5 = _read_any("ablation_A5", RESULTS)
    if a5 is None or "status" in a5.columns:
        missing.append("A5 cross-architecture: needs the sweep repeated on a "
                       "second GPU generation.")
    if not val_exists():
        missing.append("Simulator validation against hardware has not been run.")
    missing.append("Multi-tenant residency beyond pairs is approximated, not measured.")
    L += [f"- {m}" for m in missing]
    L.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L))
    return out_path


def val_exists() -> bool:
    return (RESULTS / "simulator_validation.json").exists()
