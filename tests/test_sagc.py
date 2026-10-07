"""Test suite.

These are the checks that catch the dangerous class of bug: the kind that
produces plausible-looking but wrong numbers. A scheduler that silently drops
stages, a slowdown ratio computed against a stale baseline, or a simulator whose
policies all collapse to the same behaviour would all still "run".

Run with:  python -m pytest tests/ -q
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sagc.common import env, provenance
from sagc.m2_profiler import runner, signature as sig_mod
from sagc.m3_dataset import pairgen
from sagc.m4_model import train
from sagc.m5_scheduler import policies as pol
from sagc.m6_sim import simulator as sim, validate as val
from sagc.workloads import interference, registry


# ---------------------------------------------------------------------------
# registry / workload zoo
# ---------------------------------------------------------------------------
def test_zoo_size_and_pairs():
    assert len(registry.zoo_names()) == 12
    assert len(registry.unordered_pairs()) == 78          # C(12,2) + 12 self
    assert len(registry.unordered_pairs(include_self=False)) == 66


def test_zoo_spans_the_plane():
    """The workload selection must actually be diverse, not just claimed to be."""
    occ = [registry.get(n).sim_occupancy for n in registry.zoo_names()]
    bw = [registry.get(n).sim_dram_bw for n in registry.zoo_names()]
    assert max(occ) - min(occ) > 0.5, "occupancy range too narrow to learn from"
    assert max(bw) - min(bw) > 0.4, "bandwidth range too narrow to learn from"


def test_pipeline_has_a_cpu_only_stage():
    """The held-but-idle argument depends on a stage that touches no GPU."""
    assert any(not s.gpu for s in registry.PIPELINE)
    assert registry.PIPELINE[0].sim_vram_mb == 0


def test_kernel_statistics_are_self_consistent():
    """count x duration x iterations must match the intended duty cycle."""
    for n in registry.zoo_names():
        d = registry.duty_cycle(registry.get(n))
        assert 0.3 < d <= 1.0, f"{n} has an implausible duty cycle {d:.3f}"


def test_capacity_stressor_exists():
    """Without one, VRAM exhaustion never appears as a labelled outcome."""
    oom = [p for p in registry.unordered_pairs()
           if interference.predict_pair(registry.get(p[0]), registry.get(p[1])).oom]
    assert len(oom) >= 3


def test_utilisation_does_not_track_occupancy():
    """The central claim: nvidia-smi utilisation is a poor proxy for occupancy.

    Modelling utilisation as a monotone function of occupancy silently inverts
    ablation A1, so this test guards the semantics. A draft of registry.py had
    exactly that bug.
    """
    specs = [registry.get(n) for n in registry.zoo_names()]
    occ = np.array([s.sim_occupancy for s in specs])
    util = np.array([registry.reported_utilisation(s) for s in specs])
    assert occ.max() - occ.min() > 0.5
    assert util.max() - util.min() < 0.10, "utilisation should be compressed"
    assert np.all(util > occ), "utilisation should overstate occupancy everywhere"


def test_sm_active_sits_between_utilisation_and_occupancy():
    """SM_ACTIVE is the intermediate signal; it must not duplicate either."""
    specs = [registry.get(n) for n in registry.zoo_names()]
    smact = np.array([registry.sm_active_level(s) for s in specs])
    util = np.array([registry.reported_utilisation(s) for s in specs])
    assert smact.max() - smact.min() > 0.5, "SM_ACTIVE should discriminate"
    assert np.all(smact <= util + 1e-9)


# ---------------------------------------------------------------------------
# interference model
# ---------------------------------------------------------------------------
def test_slowdown_is_never_below_one():
    for a, b in registry.unordered_pairs():
        o = interference.predict_pair(registry.get(a), registry.get(b))
        if o.oom:
            continue
        assert o.slowdown_a >= 1.0 and o.slowdown_b >= 1.0


def test_interference_is_asymmetric():
    """A pair hurts its two members differently. That asymmetry is the signal."""
    o = interference.predict_pair(registry.get("bert_base_s512_b8"),
                                  registry.get("mlp_admet_infer"))
    assert abs(o.slowdown_a - o.slowdown_b) > 1e-3


def test_complementary_pair_beats_conflicting_pair():
    """Bandwidth+tiny should be far cheaper than bandwidth+bandwidth."""
    bw1 = registry.get("bert_base_s512_b8")
    conflict = interference.predict_pair(bw1, registry.get("esm2_35m_infer")).slowdown_a
    complement = interference.predict_pair(bw1, registry.get("mlp_admet_infer")).slowdown_a
    assert complement < conflict


def test_oom_when_capacity_exceeded():
    big = registry.get("vit_large_batch_infer")
    o = interference.predict_pair(big, big, device_vram_mb=15360)
    assert o.oom and o.vram_required_mb > o.vram_available_mb


# ---------------------------------------------------------------------------
# profiler / signatures
# ---------------------------------------------------------------------------
def test_signature_has_expected_width():
    assert len(sig_mod.FEATURE_NAMES) == 18


def test_signature_extraction_runs():
    res = runner.run("gcn_molecule_infer", seed=0)
    assert res.ok and res.wall_seconds > 0
    vec = res.signature.vector()
    assert vec.shape == (18,)
    assert np.isfinite(vec).sum() >= 15


def test_profiler_is_reproducible_within_tolerance():
    a = runner.run("esm2_35m_infer", seed=7).wall_seconds
    b = runner.run("esm2_35m_infer", seed=7).wall_seconds
    assert math.isclose(a, b, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_dataset():
    cfg = pairgen.SweepConfig(
        reps=2, thread_pcts=[100, 50],
        workloads=["esm2_35m_infer", "mlp_admet_infer", "resnet50_train_b32",
                   "gcn_molecule_infer", "bert_base_s512_b8"])
    df, prov = pairgen.run_sweep(cfg, verbose=False)
    res = runner.profile_zoo(reps=2, names=cfg.workloads, verbose=False)
    sigs = sig_mod.signatures_to_frame([r.signature for r in res])
    num = list(sigs.select_dtypes("number").columns)
    agg = sigs.groupby("workload")[num].median().reset_index()
    return pairgen.attach_signatures(df, agg), agg, prov


def test_dataset_emits_two_rows_per_measurement(small_dataset):
    df, _, _ = small_dataset
    assert (set(zip(df["workload_a"], df["workload_b"]))
            == set(zip(df["workload_b"], df["workload_a"])))


def test_dataset_is_stamped(small_dataset):
    df, _, prov = small_dataset
    assert "provenance" in df.columns
    assert provenance.dataframe_kind(df) in (provenance.MEASURED, provenance.SIMULATED)
    if env.is_simulated():
        assert prov.kind == provenance.SIMULATED


def test_oom_rows_have_no_slowdown(small_dataset):
    df, _, _ = small_dataset
    oom = df[df["oom"].astype(bool)]
    if not oom.empty:
        assert oom["slowdown_a"].isna().all()


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def test_model_beats_the_mean_baseline(small_dataset):
    df, _, _ = small_dataset
    assert (train.leave_one_workload_out(df, model_kind="lgbm").mae
            < train.leave_one_workload_out(df, model_kind="mean").mae)


def test_predictions_are_never_below_one(small_dataset):
    df, sigs, _ = small_dataset
    m = train.fit(df, sigs, run_cv=False)
    table = m.build_table(workloads=list(sigs["workload"]), thread_pcts=(100,))
    assert table and all(v >= 1.0 for v in table.values())


def test_lookup_table_covers_every_pairing(small_dataset):
    df, sigs, _ = small_dataset
    m = train.fit(df, sigs, run_cv=False)
    names = list(sigs["workload"])
    assert len(m.build_table(workloads=names, thread_pcts=(100,))) == len(names) ** 2


# ---------------------------------------------------------------------------
# scheduler and simulator
# ---------------------------------------------------------------------------
def test_every_stage_completes_under_every_policy():
    oracle = pol.OraclePredictor()
    for kind in ("exclusive", "blind", "greedy", "oracle"):
        p = (pol.build_policy(kind, oracle, 1.25) if kind in ("greedy", "oracle")
             else pol.build_policy(kind))
        tr = sim.simulate(p, n_pipelines=6, n_devices=2)
        assert (tr.stages["state"] == "done").all(), f"{kind} stranded a stage"


def test_sharing_beats_exclusive_on_throughput():
    ex = sim.simulate(pol.build_policy("exclusive"), n_pipelines=8)
    gr = sim.simulate(pol.build_policy("greedy", pol.OraclePredictor(), 1.25),
                      n_pipelines=8, slowdown_bound=1.25)
    assert gr.throughput_per_hour > ex.throughput_per_hour


def test_bounded_policy_respects_its_bound():
    """The whole safety claim. A bound that is not enforced is not a bound."""
    for bound in (1.05, 1.25):
        tr = sim.simulate(pol.build_policy("greedy", pol.OraclePredictor(), bound),
                          n_pipelines=10, slowdown_bound=bound)
        assert tr.violation_rate(bound) <= 0.02, f"bound {bound} breached"


def test_blind_sharing_violates_more_than_bounded():
    bl = sim.simulate(pol.build_policy("blind"), n_pipelines=10, slowdown_bound=1.25)
    gr = sim.simulate(pol.build_policy("greedy", pol.OraclePredictor(), 1.25),
                      n_pipelines=10, slowdown_bound=1.25)
    assert bl.violation_rate(1.25) > gr.violation_rate(1.25)


def test_exclusive_leaves_the_device_under_used():
    tr = sim.simulate(pol.build_policy("exclusive"), n_pipelines=8)
    assert tr.held_but_idle_fraction > 0.0
    assert tr.held_but_under_occupied_fraction > tr.held_but_idle_fraction


def test_vram_capacity_is_never_exceeded():
    tr = sim.simulate(pol.build_policy("greedy", pol.OraclePredictor(), 1.5),
                      n_pipelines=10, device_vram_mb=15360)
    gpu = tr.stages[tr.stages["gpu"] & tr.stages["start_s"].notna()]
    for dev, grp in gpu.groupby("device"):
        if dev < 0:
            continue
        events = []
        for _, r in grp.iterrows():
            events.append((r["start_s"], r["vram_mb"]))
            events.append((r["end_s"], -r["vram_mb"]))
        used = 0
        for _, delta in sorted(events):
            used += delta
            assert used <= 15360 + 1e-6


def test_bandit_never_breaks_the_bound():
    """The bandit reorders admissible options; it must not widen admission."""
    from sagc.m5_scheduler import bandit

    p = bandit.BanditPolicy(pol.OraclePredictor(), slowdown_bound=1.25)
    tr = sim.simulate(p, n_pipelines=8, slowdown_bound=1.25)
    assert tr.violation_rate(1.25) <= 0.02


def test_self_consistency_suite_passes():
    checks = val.self_consistency_check(n_pipelines=8)
    failed = checks[~checks["passed"]]
    assert failed.empty, f"failed: {list(failed['check'])}"


def test_simulator_is_deterministic():
    def once():
        return sim.simulate(pol.build_policy("greedy", pol.OraclePredictor(), 1.25),
                            n_pipelines=6, slowdown_bound=1.25).makespan_s

    assert math.isclose(once(), once(), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_unstamped_frame_is_never_treated_as_measured():
    assert provenance.dataframe_kind(pd.DataFrame({"x": [1, 2, 3]})) == provenance.SIMULATED


def test_capture_matches_backend():
    prov = provenance.capture()
    expected = (provenance.SIMULATED if env.detect().backend == env.BACKEND_SIM
                else provenance.MEASURED)
    assert prov.kind == expected


def test_all_nan_feature_columns_are_dropped_not_silently_trained_on(small_dataset):
    """At the pynvml tier sm_occupancy is NaN. It must not reach the model.

    An all-NaN feature is accepted by LightGBM without error and never split
    on, so without this the counters-vs-utilisation ablation would quietly be
    run on a signature missing achieved occupancy while still being reported as
    though occupancy had been measured.
    """
    import numpy as np
    from sagc.m4_model import train as t

    df = small_dataset[0].copy()
    occ = [c for c in t.feature_columns(t.FEATURESET_COUNTERS)
           if "sm_occupancy" in c]
    assert occ, "feature set does not expose occupancy features"
    for c in occ:
        df[c] = np.nan

    assert set(t.unpopulated_columns(df, t.FEATURESET_COUNTERS)) == set(occ)
    X, _y, cols = t.prepare(df, t.FEATURESET_COUNTERS)
    assert not set(cols) & set(occ)
    assert not set(X.columns) & set(occ)
    assert cols, "dropping unmeasured columns must not empty the feature set"


def test_a1_flags_itself_as_reduced_when_occupancy_is_unmeasurable(small_dataset):
    import numpy as np
    from sagc.m4_model import ablations, train as t

    df = small_dataset[0].copy()
    for c in t.feature_columns(t.FEATURESET_COUNTERS):
        if "sm_occupancy" in c:
            df[c] = np.nan

    res = ablations.a1_feature_set(df)
    assert "REDUCED" in res.note
    assert "sm_occupancy" in res.note
    row = res.table[res.table.feature_set == t.FEATURESET_COUNTERS].iloc[0]
    assert row["n_features_unmeasured"] == 6
    assert row["n_features"] < row["n_features_declared"]
