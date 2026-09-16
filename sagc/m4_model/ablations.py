"""M4: ablation studies A1 to A6.

A1  Feature set          counters vs utilisation vs both.        MANDATORY
A2  Model class          gbm vs ridge vs mlp vs mean baseline.
A3  Training-set size    how many pairings before error plateaus.
A4  Generalisation       leave-one-workload-out.                 MANDATORY
A5  Cross-architecture   train on one GPU generation, test another.
A6  Partition setting    sensitivity to MPS thread percentage.

A1 and A4 carry the two contributions quantifiable from the dataset alone: that
counter-derived features beat coarse utilisation, and that the model generalises
to workloads it has never observed. A2 and A3 cost no new measurement and run by
default. A5 needs a second GPU architecture and A6 needs the full
thread-percentage sweep, so both degrade to a recorded "not available" rather
than silently producing something misleading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import train


@dataclass
class AblationResult:
    name: str
    table: pd.DataFrame
    available: bool = True
    note: str = ""

    def __str__(self) -> str:
        head = f"[{self.name}] {'' if self.available else '(unavailable) '}{self.note}"
        return head + "\n" + self.table.to_string(index=False)


def a1_feature_set(df: pd.DataFrame, model_kind: str = "lgbm",
                   seed: int = 0) -> AblationResult:
    """Does the counter signature actually beat the utilisation number?"""
    rows = []
    for kind in (train.FEATURESET_UTILISATION, train.FEATURESET_COUNTERS,
                 train.FEATURESET_BOTH):
        cv = train.leave_one_workload_out(df, kind, model_kind, seed)
        rows.append({"feature_set": kind, "mae": cv.mae, "rmse": cv.rmse,
                     "r2": cv.r2, "n_test": cv.n_test,
                     "n_features": len(train.feature_columns(kind))})
    tab = pd.DataFrame(rows)
    base = tab.loc[tab.feature_set == train.FEATURESET_UTILISATION, "mae"]
    if not base.empty and base.iloc[0] > 0:
        tab["mae_vs_utilisation"] = tab["mae"] / base.iloc[0]
    return AblationResult("A1 feature set", tab,
                          note="leave-one-workload-out error by feature set")


def a2_model_class(df: pd.DataFrame, kind: str = train.FEATURESET_COUNTERS,
                   seed: int = 0,
                   kinds: Sequence[str] = tuple(train.MODEL_KINDS)) -> AblationResult:
    rows = []
    for mk in kinds:
        try:
            cv = train.leave_one_workload_out(df, kind, mk, seed)
            rows.append({"model": mk, "mae": cv.mae, "rmse": cv.rmse, "r2": cv.r2})
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": mk, "mae": np.nan, "rmse": np.nan,
                         "r2": np.nan, "error": type(exc).__name__})
    return AblationResult("A2 model class", pd.DataFrame(rows),
                          note="leave-one-workload-out error by model family")


def a3_training_size(df: pd.DataFrame, kind: str = train.FEATURESET_COUNTERS,
                     model_kind: str = "lgbm",
                     fractions: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
                     seeds: Sequence[int] = (0, 1, 2)) -> AblationResult:
    """How many measured pairings are needed before error stops improving?

    Sampling is by PAIRING, not by row: dropping a random row leaves its
    role-swapped twin in training and would understate how much data is really
    required.
    """
    work = df[~df["oom"].astype(bool)].copy()
    work["pair_key"] = work.apply(
        lambda r: "|".join(sorted([r["workload_a"], r["workload_b"]])), axis=1)
    pairs = sorted(work["pair_key"].unique())
    rows = []
    for frac in fractions:
        maes = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            k = max(2, int(round(frac * len(pairs))))
            keep = set(rng.choice(pairs, size=k, replace=False))
            sub = work[work["pair_key"].isin(keep)]
            if sub["workload_a"].nunique() < 3:
                continue
            cv = train.leave_one_workload_out(sub, kind, model_kind, seed)
            if not np.isnan(cv.mae):
                maes.append(cv.mae)
        rows.append({"fraction": frac, "n_pairs": int(round(frac * len(pairs))),
                     "mae_mean": float(np.mean(maes)) if maes else np.nan,
                     "mae_std": float(np.std(maes)) if maes else np.nan,
                     "n_seeds": len(maes)})
    return AblationResult("A3 training-set size", pd.DataFrame(rows),
                          note="error against number of measured pairings")


def a4_generalisation(df: pd.DataFrame, kind: str = train.FEATURESET_COUNTERS,
                      model_kind: str = "lgbm", seed: int = 0) -> AblationResult:
    """Per-workload held-out error, related to that workload's extremity."""
    from ..workloads import registry

    cv = train.leave_one_workload_out(df, kind, model_kind, seed)
    rows = []
    for w, mae in sorted(cv.per_workload.items(), key=lambda kv: -kv[1]):
        spec = registry.get(w)
        rows.append({"workload": w, "family": spec.family, "held_out_mae": mae,
                     "occupancy": spec.sim_occupancy, "dram_bw": spec.sim_dram_bw,
                     "vram_mb": spec.sim_vram_mb})
    return AblationResult(
        "A4 leave-one-workload-out", pd.DataFrame(rows),
        note=f"overall MAE={cv.mae:.4f} RMSE={cv.rmse:.4f} R2={cv.r2:.4f}")


def a5_cross_architecture(df: pd.DataFrame,
                          kind: str = train.FEATURESET_COUNTERS,
                          model_kind: str = "lgbm") -> AblationResult:
    """Train on one GPU architecture, test on another.

    Requires the dataset to contain more than one device model, which means
    running the sweep on a second machine. Reports unavailability rather than
    inventing a number when only one architecture is present.
    """
    col = next((c for c in ("gpu_name", "device_name", "arch") if c in df.columns), None)
    if col is None or df[col].nunique() < 2:
        return AblationResult(
            "A5 cross-architecture",
            pd.DataFrame([{"status": "not available",
                           "reason": "dataset contains a single GPU architecture"}]),
            available=False,
            note="collect the sweep on a second device to enable this")
    rows = []
    for arch in sorted(df[col].unique()):
        Xtr, ytr, cols = train.prepare(df[df[col] != arch], kind)
        Xte, yte, _ = train.prepare(df[df[col] == arch], kind)
        if Xtr.empty or Xte.empty:
            continue
        model = train._make_model(model_kind)
        model.fit(Xtr, ytr)
        yhat = np.clip(
            model.predict(Xte.reindex(columns=cols, fill_value=np.nan)), 1.0, None)
        mae, rmse, r2 = train._metrics(yte.values, yhat)
        rows.append({"tested_on": arch, "mae": mae, "rmse": rmse,
                     "r2": r2, "n": len(yte)})
    return AblationResult("A5 cross-architecture", pd.DataFrame(rows),
                          note="trained on all other architectures")


def a6_thread_percentage(df: pd.DataFrame,
                         kind: str = train.FEATURESET_COUNTERS,
                         model_kind: str = "lgbm") -> AblationResult:
    """Does the MPS partition setting change the picture?"""
    if "thread_pct_a" not in df.columns or df["thread_pct_a"].nunique() < 2:
        return AblationResult(
            "A6 MPS thread percentage",
            pd.DataFrame([{"status": "not available",
                           "reason": "only one thread-percentage setting in dataset"}]),
            available=False,
            note="MPS partition control was unavailable, or the sweep was collapsed")
    rows = []
    for pct in sorted(df["thread_pct_a"].unique()):
        sub = df[df["thread_pct_a"] == pct]
        cv = train.leave_one_workload_out(sub, kind, model_kind)
        clean = sub[~sub["oom"].astype(bool)]
        rows.append({"thread_pct": int(pct), "n_rows": len(sub),
                     "mean_slowdown": float(clean[train.TARGET].mean()),
                     "p95_slowdown": float(clean[train.TARGET].quantile(0.95)),
                     "mae": cv.mae})
    return AblationResult("A6 MPS thread percentage", pd.DataFrame(rows),
                          note="observed slowdown and model error by partition setting")


def run_all(df: pd.DataFrame, verbose: bool = True,
            include: Optional[Sequence[str]] = None) -> Dict[str, AblationResult]:
    """Run every ablation this dataset supports."""
    include = set(include or ["A1", "A2", "A3", "A4", "A5", "A6"])
    runners = {
        "A1": lambda: a1_feature_set(df),
        "A2": lambda: a2_model_class(df),
        "A3": lambda: a3_training_size(df),
        "A4": lambda: a4_generalisation(df),
        "A5": lambda: a5_cross_architecture(df),
        "A6": lambda: a6_thread_percentage(df),
    }
    out: Dict[str, AblationResult] = {}
    for key in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        if key not in include:
            continue
        res = runners[key]()
        out[key] = res
        if verbose:
            print(res)
            print()
    return out
