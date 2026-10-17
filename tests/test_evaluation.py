"""Auswertung: Einstellungen, eine Stichprobe mit allen Kennzahlen und die vier Experimente."""

import numpy as np
import pytest

import sns_constants as C
import sns_evaluation as ev
import sns_sensitivity as S


def test_number_formats():
    assert ev.de(-1.234) == "−1,23" and ev.de(0.5, 1) == "0,5" and ev.signed(0.3) == "+0,30" and ev.signed(-0.3) == "−0,30" and ev.signed(1e-12) == "+0,00" and ev.pct(0.2) == "20 %" and ev.pct(0.0912, 1) == "9,1 %"


def test_settings_defaults_and_world():
    s = ev.Settings()
    assert (s.gu, s.bu, s.tau, s.n, s.seed, s.q, s.k, s.direction) == (0.5, 1.5, 3.5, 400, 35, 1.0, 1, "Abschwächung") and s.world.gu == 0.5 and s.has_u
    assert not ev.Settings(gu=0.0).has_u and not ev.Settings(bu=0.0).has_u and not ev.Settings(gu=0.0, bu=0.0).has_u


def test_analyse_invariants():
    a = ev.analyse(ev.Settings())
    assert a.truth == 3.5 and a.fit.df == 397 and a.full.df == 396 and a.rv_one == pytest.approx(S.robustness_value(a.fit.t, a.fit.df)) and a.rv == a.rv_one
    assert a.bias_exact == pytest.approx(a.bias_formula, abs=1e-9) and a.se_formula == pytest.approx(a.full.se, abs=1e-12) and a.f_cohen == pytest.approx(abs(a.fit.t) / np.sqrt(397))
    assert a.rv_alpha < a.rv and a.covers == (a.fit.lo <= 3.5 <= a.fit.hi) and a.full_covers == (a.full.lo <= 3.5 <= a.full.hi)
    assert 0.0 < a.r2dz_u < 1.0 and 0.0 < a.r2yz_u < 1.0 and 0.0 < a.r2dz_z < 1.0 and 0.0 < a.r2yz_z < 1.0


def test_q_changes_the_robustness_value_only():
    a, b = ev.analyse(ev.Settings()), ev.analyse(ev.Settings(q=0.5))
    assert b.rv == pytest.approx(S.robustness_value(a.fit.t, a.fit.df, 0.5)) and b.rv < a.rv and b.fit == a.fit and b.rv_one == a.rv_one


def test_benchmark_follows_k_and_direction():
    a1, a2 = ev.analyse(ev.Settings(k=1)), ev.analyse(ev.Settings(k=2))
    up = ev.analyse(ev.Settings(k=1, direction="Verstärkung"))
    assert a2.benchmark["bias"] > a1.benchmark["bias"] and a1.benchmark["estimate"] == pytest.approx(a1.fit.estimate - a1.benchmark["bias"]) and up.benchmark["estimate"] == pytest.approx(up.fit.estimate + up.benchmark["bias"])
    assert a1.benchmark["bias"] == pytest.approx(S.bias(a1.r2dz_z, a1.r2yz_z, a1.fit.se, a1.fit.df))


def test_without_u_the_estimate_hits_the_truth_and_the_orakel_equals_the_estimate():
    a = ev.analyse(ev.Settings(gu=0.0, bu=0.0, n=3000))
    assert a.fit.estimate == pytest.approx(3.5, abs=0.6) and abs(a.fit.estimate - a.full.estimate) < 0.1 and a.r2dz_u < 0.01 and a.r2yz_u < 0.01 and a.covers


def test_analyse_is_deterministic_and_seed_dependent():
    assert ev.analyse(ev.Settings()).fit == ev.analyse(ev.Settings()).fit and ev.analyse(ev.Settings(seed=1)).fit != ev.analyse(ev.Settings(seed=2)).fit


def test_identity_experiment(identity):
    r = identity
    assert r["reps"] == 300 and r["n"] == 400 and r["dev_bias"] < 1e-9 and r["dev_se"] < 1e-10 and r["dev_rv"] < 1e-7
    assert r["dists"]["ohne U"]["mean"] > r["dists"]["mit U"]["mean"] and len(r["dists"]["mit U"]["rv"]) == C.EXP_RV_REPS and r["violations"] == 0 and r["tested"] == sum(p["tested"] for p in r["protect"]) > 100
    assert [p["gu"] for p in r["protect"]] == [g for g, _ in C.EXP_PROTECT] and all(p["violations"] == 0 for p in r["protect"])


def test_evalue_experiment(evalue_res):
    r = evalue_res
    assert r["scenarios"] == 378 and r["violations"] == 0 and 0.0 <= r["min_gap"] < 1e-9
    assert [t["rr"] for t in r["table"]] == list(C.EXP_RR) and all(t["bound_at_e"] == pytest.approx(t["rr"]) for t in r["table"]) and r["table"][1]["evalue"] == pytest.approx(2.3660254)


def test_falsification_experiment_structure_and_direction(falsify):
    r = falsify
    assert r["n"] == 500 and r["reps"] == 500 and [x["gu"] for x in r["rows"]] == list(C.EXP_FALSIFY_GU)
    nc = [x["negative_control"] for x in r["rows"]]
    assert nc == sorted(nc) and nc[0] < 0.1 and nc[-1] > 0.95 and all(x["negative_control_weak"] <= x["negative_control"] for x in r["rows"][1:])
    assert all(x["random_cause"] <= 0.02 and x["placebo"] < 0.12 and x["subset"] < 0.12 for x in r["rows"]) and 0.3 < r["invalid_alarm"] < 0.7
    assert [x["error"] for x in r["rows"]] == sorted((x["error"] for x in r["rows"]), reverse=True) and abs(r["rows"][0]["error"]) < 0.3


def test_benchmark_experiment(bench):
    r = bench
    assert [t["k"] for t in r["table"]] == [1, 2, 3] and r["covered"] == r["reps"] == 300 and r["table"][0]["bias"] < r["table"][1]["bias"] < r["table"][2]["bias"]
    assert r["table"][0]["down"] == pytest.approx(r["estimate"] - r["table"][0]["bias"]) and r["table"][0]["up"] == pytest.approx(r["estimate"] + r["table"][0]["bias"])
    assert r["k_true_mean"] < 1.0                                                  # in der Standardwelt ist U schwächer als Z
