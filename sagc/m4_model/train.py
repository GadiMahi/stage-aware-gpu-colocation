"""M4: the slowdown predictor.

Learns f(signature_A, signature_B, mps_config) -> slowdown_A from the measured
pair dataset.

WHY GRADIENT BOOSTING
---------------------
The target is a bounded continuous quantity, the feature count is small (40),
the dataset is in the hundreds of rows, and the underlying relationship has
sharp regime changes: below memory-bandwidth saturation almost nothing happens;
above it, slowdown climbs steeply. Tree ensembles handle that thresholding
natively, need no feature scaling, and train in under a second at this size.
Ablation A2 compares against ridge, an MLP, and a mean baseline to substantiate
the choice rather than assert it.

THE EVALUATION THAT MATTERS
---------------------------
Random k-fold cross-validation would be misleading. Each measured pairing
contributes two rows and several repetitions, so a random split puts near
duplicates of a test row into training, and the reported error would describe
interpolation rather than generalisation.

The honest evaluation is LEAVE-ONE-WORKLOAD-OUT: remove every row in which a
given workload appears on either side, train on the rest, and test only on that
workload. It answers the question a deployment actually asks, which is whether a
workload never profiled alongside anything can still have its interference
predicted.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..m2_profiler import signature as sig_mod
from ..workloads import registry

TARGET = "slowdown_a"

FEATURESET_COUNTERS = "counters"
FEATURESET_UTILISATION = "utilisation"
FEATURESET_BOTH = "both"

_CONFIG_FEATURES = ["thread_pct_a", "thread_pct_b", "vram_sum_mb", "vram_headroom_mb"]


def feature_columns(kind: str = FEATURESET_COUNTERS) -> List[str]:
    """Column names for one feature set, in fixed order."""
    if kind == FEATURESET_UTILISATION:
        base = sig_mod.UTILISATION_ONLY_FEATURES
    elif kind == FEATURESET_BOTH:
        base = sig_mod.FEATURE_NAMES + [
            f for f in sig_mod.UTILISATION_ONLY_FEATURES
            if f not in sig_mod.FEATURE_NAMES]
    else:
        base = sig_mod.FEATURE_NAMES
    return [f"a_{f}" for f in base] + [f"b_{f}" for f in base] + _CONFIG_FEATURES


def prepare(df: pd.DataFrame, kind: str = FEATURESET_COUNTERS,
            drop_oom: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Feature matrix, target, and column list from the joined pair dataset."""
    work = df.copy()
    if drop_oom:
        work = work[~work["oom"].astype(bool)]
    work = work[work[TARGET].notna()]
    cols = [c for c in feature_columns(kind) if c in work.columns]
    missing = [c for c in feature_columns(kind) if c not in work.columns]
    if missing:
        warnings.warn(f"{len(missing)} feature columns absent and skipped: {missing[:4]}")
    return work[cols].astype(float), work[TARGET].astype(float), cols


def _make_model(kind: str, seed: int = 0):
    if kind == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1)
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                          random_state=seed, early_stopping=False))
    if kind == "mean":
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(strategy="mean")
    raise ValueError(f"unknown model kind {kind!r}")


MODEL_KINDS = ["lgbm", "ridge", "mlp", "mean"]


@dataclass
class CVResult:
    mae: float
    rmse: float
    r2: float
    n_test: int
    per_workload: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict:
        return {"mae": round(self.mae, 5), "rmse": round(self.rmse, 5),
                "r2": round(self.r2, 5), "n_test": self.n_test,
                "per_workload": {k: round(v, 5) for k, v in self.per_workload.items()}}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return mae, rmse, r2


def leave_one_workload_out(df: pd.DataFrame, kind: str = FEATURESET_COUNTERS,
                           model_kind: str = "lgbm", seed: int = 0,
                           workloads: Optional[Sequence[str]] = None) -> CVResult:
    """Hold out every row mentioning a workload, train, predict only that workload.

    The headline generalisation experiment. A held-out workload has never been
    seen on either side of any training pair, so the model must infer its
    sensitivity purely from its counter signature.
    """
    workloads = list(workloads
                     or sorted(set(df["workload_a"]) | set(df["workload_b"])))
    preds = []

    for held in workloads:
        te = df[df["workload_a"] == held]
        tr = df[(df["workload_a"] != held) & (df["workload_b"] != held)]
        if tr.empty or te.empty:
            continue
        Xtr, ytr, cols = prepare(tr, kind)
        Xte, yte, _ = prepare(te, kind)
        if Xtr.empty or Xte.empty:
            continue
        Xte = Xte.reindex(columns=cols, fill_value=np.nan)

        model = _make_model(model_kind, seed)
        model.fit(Xtr, ytr)
        yhat = np.clip(model.predict(Xte), 1.0, None)
        preds.append(pd.DataFrame({
            "workload_held_out": held,
            "workload_a": te.loc[Xte.index, "workload_a"].values,
            "workload_b": te.loc[Xte.index, "workload_b"].values,
            "y_true": yte.values, "y_pred": yhat}))

    if not preds:
        return CVResult(float("nan"), float("nan"), float("nan"), 0)

    allp = pd.concat(preds, ignore_index=True)
    mae, rmse, r2 = _metrics(allp["y_true"].values, allp["y_pred"].values)
    per = (allp.assign(err=(allp["y_true"] - allp["y_pred"]).abs())
              .groupby("workload_held_out")["err"].mean().to_dict())
    return CVResult(mae, rmse, r2, len(allp), per, allp)


def grouped_kfold(df: pd.DataFrame, kind: str = FEATURESET_COUNTERS,
                  model_kind: str = "lgbm", n_splits: int = 5,
                  seed: int = 0) -> CVResult:
    """Cross-validation grouped by unordered pairing.

    Weaker than leave-one-workload-out but still honest: the two role-swapped
    rows and all repetitions of a pairing stay on the same side of the split, so
    no near-duplicate leaks across.
    """
    from sklearn.model_selection import GroupKFold

    work = df[~df["oom"].astype(bool)].copy()
    work = work[work[TARGET].notna()]
    groups = work.apply(
        lambda r: "|".join(sorted([r["workload_a"], r["workload_b"]])), axis=1)
    X, y, cols = prepare(work, kind)
    groups = groups.loc[X.index]

    gkf = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    rows = []
    for tr_idx, te_idx in gkf.split(X, y, groups):
        model = _make_model(model_kind, seed)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        yhat = np.clip(model.predict(X.iloc[te_idx]), 1.0, None)
        rows.append(pd.DataFrame({"y_true": y.iloc[te_idx].values, "y_pred": yhat}))
    allp = pd.concat(rows, ignore_index=True)
    mae, rmse, r2 = _metrics(allp["y_true"].values, allp["y_pred"].values)
    return CVResult(mae, rmse, r2, len(allp), {}, allp)


@dataclass
class SlowdownModel:
    """A trained model plus everything needed to consume it at scheduling time."""

    model: object
    columns: List[str]
    featureset: str
    model_kind: str
    signatures: pd.DataFrame
    train_rows: int
    cv: Optional[CVResult] = None
    name: str = "learned"

    def _row_for(self, a: str, b: str, thread_pct_a: int, thread_pct_b: int,
                 device_vram_mb: int) -> pd.DataFrame:
        sig = self.signatures.set_index("workload")
        row: Dict[str, float] = {}
        for prefix, w in (("a", a), ("b", b)):
            if w not in sig.index:
                raise KeyError(f"no signature for workload {w!r}")
            for col, val in sig.loc[w].items():
                row[f"{prefix}_{col}"] = val
        va, vb = registry.get(a).sim_vram_mb, registry.get(b).sim_vram_mb
        row["thread_pct_a"] = thread_pct_a
        row["thread_pct_b"] = thread_pct_b
        row["vram_sum_mb"] = va + vb
        row["vram_headroom_mb"] = device_vram_mb - (va + vb)
        return (pd.DataFrame([row])
                .reindex(columns=self.columns, fill_value=np.nan).astype(float))

    def predict_pair(self, a: str, b: str, thread_pct: int = 100,
                     device_vram_mb: int = 15360) -> float:
        X = self._row_for(a, b, thread_pct, thread_pct, device_vram_mb)
        return float(np.clip(self.model.predict(X)[0], 1.0, None))

    def build_table(self, workloads: Optional[Sequence[str]] = None,
                    thread_pcts: Sequence[int] = (100, 50),
                    device_vram_mb: int = 15360
                    ) -> Dict[Tuple[str, str, int], float]:
        """Precompute every pairing offline.

        Because stage identity is known before a campaign starts, the scheduler
        never needs model inference in its decision path: it consults this table
        in constant time. That is what keeps the policy cheap as the campaign
        grows, and it is the concrete answer to the objection that machine
        learning is too slow for scheduling.
        """
        workloads = list(workloads or registry.zoo_names())
        sig = self.signatures.set_index("workload")
        rows, index = [], []
        for pct in thread_pcts:
            for a in workloads:
                for b in workloads:
                    if a not in sig.index or b not in sig.index:
                        continue
                    rows.append(self._row_for(a, b, pct, pct, device_vram_mb).iloc[0])
                    index.append((a, b, pct))
        if not rows:
            return {}
        X = (pd.DataFrame(rows)
             .reindex(columns=self.columns, fill_value=np.nan).astype(float))
        yhat = np.clip(self.model.predict(X), 1.0, None)
        return {k: float(v) for k, v in zip(index, yhat)}

    def feature_importance(self, top: int = 20) -> pd.DataFrame:
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return pd.DataFrame(columns=["feature", "importance"])
        return (pd.DataFrame({"feature": self.columns, "importance": imp})
                .sort_values("importance", ascending=False)
                .head(top).reset_index(drop=True))

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "model_meta.json").write_text(json.dumps({
            "featureset": self.featureset, "model_kind": self.model_kind,
            "columns": self.columns, "train_rows": self.train_rows,
            "cv": self.cv.to_dict() if self.cv else None}, indent=2))
        try:
            import joblib

            joblib.dump(self.model, d / "model.joblib")
        except Exception:  # noqa: BLE001
            pass
        return d / "model_meta.json"


def fit(df: pd.DataFrame, signatures: pd.DataFrame,
        kind: str = FEATURESET_COUNTERS, model_kind: str = "lgbm",
        seed: int = 0, run_cv: bool = True) -> SlowdownModel:
    """Train on the whole dataset, and report leave-one-workload-out error."""
    X, y, cols = prepare(df, kind)
    model = _make_model(model_kind, seed)
    model.fit(X, y)
    cv = leave_one_workload_out(df, kind, model_kind, seed) if run_cv else None

    sig_cols = [c for c in signatures.columns
                if c in sig_mod.FEATURE_NAMES + sig_mod.UTILISATION_ONLY_FEATURES]
    return SlowdownModel(
        model=model, columns=cols, featureset=kind, model_kind=model_kind,
        signatures=signatures[["workload"] + sig_cols].drop_duplicates("workload"),
        train_rows=len(X), cv=cv)


class ModelPredictor:
    """Adapts a SlowdownModel (via its lookup table) to the scheduler interface."""

    def __init__(self, table: Dict[Tuple[str, str, int], float],
                 name: str = "learned"):
        self.table = table
        self.name = name
        self.lookups = 0

    def predict(self, target: str, co_tenants: Sequence[str],
                thread_pct: int = 100) -> float:
        if not co_tenants:
            return 1.0
        self.lookups += 1
        vals = [self.table.get((target, c, thread_pct),
                               self.table.get((target, c, 100), 1.0))
                for c in co_tenants]
        return float(max(vals)) if vals else 1.0
