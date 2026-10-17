"""Sensitivität: Regression, partielle R², Robustheitswert, Verzerrung, angepasste Schätzung, Konturkarte, Benchmark, E-Wert und Schranke; an den Handrechnungen aus conftest.py und
an unabhängigen Wegen (Residualisierung, Nullstellensuche, die Regression mit dem Confounder)."""

import numpy as np
import pytest

import sns_constants as C
import sns_sensitivity as S
import sns_world as W
from kausal_generator import SplitMix64


def _world(gu=0.5, bu=1.5, n=400, reps=1, seed=3, tau=3.5):
    return W.simulate(W.World(gu, bu, tau), n, SplitMix64(seed), reps)


def test_ols_against_numpy_lstsq():
    rng = SplitMix64(1)
    x1, x2 = rng.normal(80), rng.normal(80)
    y = 1.0 + 2.0 * x1 - x2 + 0.5 * rng.normal(80)
    beta, se, df, res = S.ols(y, [x1, x2])
    d = np.column_stack([np.ones(80), x1, x2])
    ref, rss, *_ = np.linalg.lstsq(d, y, rcond=None)
    assert beta == pytest.approx(ref) and df == 77 and res @ res == pytest.approx(rss[0]) and se == pytest.approx(np.sqrt(np.diag(np.linalg.inv(d.T @ d)) * rss[0] / 77))


def test_ols_on_exact_data(small_data):
    c, d, y = small_data
    beta, se, df, res = S.ols(y, [d, c])
    assert beta == pytest.approx([1.0, 2.0, 3.0]) and df == 5 and np.abs(res).max() < 1e-12 and se == pytest.approx(0.0, abs=1e-12)
    with np.errstate(divide="ignore"):
        assert S.treatment_fit(y, d, [c]).estimate == pytest.approx(2.0)


def test_fit_interval():
    f = S.Fit(2.0, 0.5, 4.0, 50)
    assert f.lo == pytest.approx(2.0 - C.Z_CRIT * 0.5) and f.hi == pytest.approx(2.0 + C.Z_CRIT * 0.5)


def test_partial_r2_against_residualisation():
    s = _world(n=300)
    z, x, y = s["z"][0], s["x"][0], s["y"][0]
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    assert S.partial_r2(y, x, [z]) == pytest.approx(np.corrcoef(ry, rx)[0, 1] ** 2)
    assert S.partial_r2(y, x, []) == pytest.approx(np.corrcoef(y, x)[0, 1] ** 2)


def test_partial_r2_of_a_perfect_fit_is_one(small_data):
    c, d, y = small_data
    noisy = y + np.array([0.0, 1e-3, -1e-3, 0.0, 1e-3, -1e-3, 0.0, 0.0])
    assert S.partial_r2(noisy, d, [c]) > 0.999999 and 0.0 < S.partial_r2(noisy, c, [d]) <= 1.0


def test_robustness_value_by_hand():
    assert S.cohens_f(2.0, 4) == pytest.approx(1.0) and S.cohens_f(-2.0, 4) == pytest.approx(1.0)
    assert S.robustness_value(2.0, 4) == pytest.approx((np.sqrt(5.0) - 1.0) / 2.0) and S.robustness_value(-2.0, 4) == pytest.approx(0.6180339887)
    assert S.robustness_value(0.0, 100) == 0.0 and S.robustness_value(2.0, 4, q=0.5) == pytest.approx(0.5 * (np.sqrt(0.25 ** 2 + 4 * 0.25) - 0.25))


def test_robustness_value_is_the_strength_that_drives_the_estimate_to_zero(hand_fit):
    """Bei gleicher Stärke s für beide Wege ist die angepasste Schätzung bei s = RV_1 genau null."""
    rv = S.robustness_value(hand_fit.t, hand_fit.df)
    assert S.adjusted_estimate(hand_fit, rv, rv)["estimate"] == pytest.approx(0.0, abs=1e-9)
    half = S.robustness_value(hand_fit.t, hand_fit.df, 0.5)
    assert S.adjusted_estimate(hand_fit, half, half)["estimate"] == pytest.approx(0.5 * hand_fit.estimate, abs=1e-9)


@pytest.mark.parametrize("q", [1.0, 0.75, 0.5, 0.25])
@pytest.mark.parametrize("t,df", [(1.0, 100), (3.28, 397), (2.5, 50), (6.0, 400), (50.0, 100), (12.0, 30)])
def test_alpha_robustness_value_against_the_numerical_solution(t, df, q):
    """Die geschlossene Form gilt auch für große t (dort weicht eine Fallunterscheidung aus der Erinnerung an die Software um 5e-4 ab): Maßstab ist die Nullstellensuche der Definition."""
    fit = S.Fit(t * 0.3, 0.3, t, df)
    assert S.robustness_value_alpha(t, df, q) == pytest.approx(S.robustness_value_numeric(fit, q), abs=1e-7)


def test_alpha_robustness_value_branches():
    assert S.robustness_value_alpha(1.0, 100) == 0.0                                     # nicht signifikant: es gibt nichts zu kippen
    mid, big = S.robustness_value_alpha(3.28, 397), S.robustness_value_alpha(50.0, 100)
    assert 0.0 < mid < S.robustness_value(3.28, 397) and 0.9 < big < 1.0
    assert S.robustness_value_alpha(3.28, 397, q=0.5) < mid and S.robustness_value_alpha(3.28, 397, alpha=0.2) > mid


def test_bias_and_adjusted_se_by_hand(hand_fit):
    assert S.bias(0.5, 0.5, 1.0, 100) == pytest.approx(7.0710678, abs=1e-6) and S.bias(0.0, 0.5, 1.0, 100) == 0.0 and S.bias(0.5, 0.0, 1.0, 100) == 0.0
    assert S.adjusted_se(0.5, 0.5, 1.0, 100) == pytest.approx(np.sqrt(100 / 99)) and S.adjusted_se(0.0, 0.0, 1.0, 100) == pytest.approx(np.sqrt(100 / 99))


def test_adjusted_estimate_by_hand(hand_fit):
    down = S.adjusted_estimate(hand_fit, 0.5, 0.5, C.DIRECTIONS[0])
    up = S.adjusted_estimate(hand_fit, 0.5, 0.5, C.DIRECTIONS[1])
    assert down["estimate"] == pytest.approx(10 - 7.0710678, abs=1e-6) and up["estimate"] == pytest.approx(17.0710678, abs=1e-6) and down["bias"] == pytest.approx(7.0710678, abs=1e-6)
    assert down["lo"] == pytest.approx(down["estimate"] - C.Z_CRIT * down["se"]) and down["se"] == pytest.approx(np.sqrt(100 / 99))
    neg = S.adjusted_estimate(S.Fit(-10.0, 1.0, -10.0, 100), 0.5, 0.5, C.DIRECTIONS[0])
    assert neg["estimate"] == pytest.approx(-2.9289322, abs=1e-6) and S.adjusted_estimate(S.Fit(-10.0, 1.0, -10.0, 100), 0.5, 0.5, C.DIRECTIONS[1])["estimate"] == pytest.approx(-17.0710678, abs=1e-6)


def test_the_bias_identity_is_exact_with_a_known_confounder():
    """Mit bekanntem U: Differenz der Schätzungen ohne und mit U und der Standardfehler mit U folgen den Formeln bis auf Rundung."""
    s = _world(n=400, reps=20)
    for r in range(20):
        z, u, x, y = (s[k][r] for k in ("z", "u", "x", "y"))
        fit, full = S.treatment_fit(y, x, [z]), S.treatment_fit(y, x, [z, u])
        rd, ry = S.partial_r2(x, u, [z]), S.partial_r2(y, u, [x, z])
        assert abs(fit.estimate - full.estimate) == pytest.approx(S.bias(rd, ry, fit.se, fit.df), abs=1e-9)
        assert full.se == pytest.approx(S.adjusted_se(rd, ry, fit.se, fit.df), abs=1e-12)


def test_the_direction_of_the_true_bias_matches_the_signed_formula():
    """U, das x erhöht und y senkt, drückt die Schätzung nach unten: Die Schätzung mit U liegt höher, die Abschwächung der Schätzung ohne U trifft sie nicht, die Verstärkung schon."""
    s = _world(n=400, reps=10)
    for r in range(10):
        z, u, x, y = (s[k][r] for k in ("z", "u", "x", "y"))
        fit, full = S.treatment_fit(y, x, [z]), S.treatment_fit(y, x, [z, u])
        rd, ry = S.partial_r2(x, u, [z]), S.partial_r2(y, u, [x, z])
        assert full.estimate > fit.estimate and S.adjusted_estimate(fit, rd, ry, C.DIRECTIONS[1])["estimate"] == pytest.approx(full.estimate, abs=1e-9)
        assert S.adjusted_estimate(fit, rd, ry, C.DIRECTIONS[0])["estimate"] < fit.estimate


def test_the_protection_statement():
    """Liegt die wahre Stärke von U (beide Wege) unter dem Robustheitswert, bleibt die Schätzung mit U über null (Rechenaussage, an Stichproben bestätigt)."""
    tested = 0
    for gu, bu in ((0.25, 1.0), (0.5, 1.5)):
        s = _world(gu, bu, n=400, reps=30, seed=7)
        for r in range(30):
            z, u, x, y = (s[k][r] for k in ("z", "u", "x", "y"))
            fit = S.treatment_fit(y, x, [z])
            if max(S.partial_r2(x, u, [z]), S.partial_r2(y, u, [x, z])) < S.robustness_value(fit.t, fit.df):
                tested += 1
                assert S.treatment_fit(y, x, [z, u]).estimate > 0
    assert tested > 30


def test_contour_grid(hand_fit):
    axis, z = S.contour_grid(hand_fit, points=7, top=0.6)
    assert len(axis) == 7 and axis[0] == 0.0 and axis[-1] == pytest.approx(0.6) and z.shape == (7, 7)
    assert z[0, 0] == pytest.approx(10.0) and z[:, 0] == pytest.approx(10.0) and z[0, :] == pytest.approx(10.0)          # ohne einen der beiden Wege keine Verzerrung
    assert z[3, 3] == pytest.approx(S.adjusted_estimate(hand_fit, axis[3], axis[3])["estimate"]) and z[2, 5] == pytest.approx(S.adjusted_estimate(hand_fit, axis[5], axis[2])["estimate"])
    assert (np.diff(z, axis=0) <= 1e-12).all() and (np.diff(z, axis=1) <= 1e-12).all()                                    # je stärker, desto tiefer
    _, up = S.contour_grid(hand_fit, C.DIRECTIONS[1], points=7)
    assert (up >= 10.0 - 1e-12).all()


def test_benchmark_scales_and_clamps():
    assert S.benchmark(0.2, 0.1, 1) == (0.2, 0.1) and S.benchmark(0.2, 0.1, 3) == pytest.approx((0.6, 0.3)) and S.benchmark(0.5, 0.4, 3) == (0.99, 0.99)


def test_evalue_by_hand():
    assert S.evalue(2.0) == pytest.approx(2 + np.sqrt(2.0)) and S.evalue(3.0) == pytest.approx(5.4494897) and S.evalue(4.0) == pytest.approx(4 + np.sqrt(12.0)) and S.evalue(1.0) == 1.0
    assert S.evalue(0.5) == pytest.approx(S.evalue(2.0)) and S.evalue(0.25) == pytest.approx(S.evalue(4.0))


def test_evalue_for_the_interval():
    assert S.evalue_ci(2.0, 1.5, 2.7) == pytest.approx(S.evalue(1.5)) and S.evalue_ci(2.0, 0.8, 3.0) == 1.0 and S.evalue_ci(0.5, 0.3, 0.8) == pytest.approx(S.evalue(0.8))
    assert S.evalue_ci(1.0, 0.9, 1.1) == 1.0


def test_ding_bound_by_hand_and_at_the_evalue():
    assert S.ding_bound(4.0, 3.0) == pytest.approx(2.0) and S.ding_bound(1.0, 5.0) == pytest.approx(1.0)
    for rr in (1.2, 1.5, 2.0, 3.0, 5.0):
        e = S.evalue(rr)
        assert S.ding_bound(e, e) == pytest.approx(rr)                       # bei gleicher Stärke in beiden Wegen ist die Schranke genau das RR


def test_binary_scenario_by_hand():
    sc = S.binary_scenario(0.5, 0.8, 0.2, 0.6, 0.2)
    assert sc["rr_obs"] == pytest.approx(0.52 / 0.28) and sc["rr_ud"] == pytest.approx(4.0) and sc["rr_uy"] == pytest.approx(3.0) and sc["bound"] == pytest.approx(2.0) and sc["rr_obs"] <= sc["bound"]
    assert S.binary_scenario(0.5, 0.2, 0.8, 0.6, 0.2) is None                # D wird von U gesenkt: beobachtetes RR unter 1


def test_binary_scenario_is_asymmetric_in_the_constants():
    """Mit P(U) = 0,3 statt 0,5 ändern sich alle Größen; die Rechnung benutzt die Parameter."""
    a, b = S.binary_scenario(0.3, 0.8, 0.2, 0.6, 0.2), S.binary_scenario(0.5, 0.8, 0.2, 0.6, 0.2)
    p_d = 0.3 * 0.8 + 0.7 * 0.2
    assert a["rr_obs"] == pytest.approx(((0.3 * 0.8 * 0.6 + 0.7 * 0.2 * 0.2) / p_d) / ((0.3 * 0.2 * 0.6 + 0.7 * 0.8 * 0.2) / (1 - p_d))) and a["rr_obs"] != pytest.approx(b["rr_obs"])
    assert a["rr_ud"] == pytest.approx(max((0.3 * 0.8 / p_d) / (0.3 * 0.2 / (1 - p_d)), (1 - 0.3 * 0.2 / (1 - p_d)) / (1 - 0.3 * 0.8 / p_d)))


def test_adjusted_estimate_direction_for_small_and_negative_estimates():
    """Das Vorzeichen der Verzerrung folgt dem Vorzeichen der Schätzung, auch zwischen −1 und 1."""
    pos = S.adjusted_estimate(S.Fit(0.5, 0.1, 5.0, 100), 0.2, 0.2, C.DIRECTIONS[0])
    neg = S.adjusted_estimate(S.Fit(-0.5, 0.1, -5.0, 100), 0.2, 0.2, C.DIRECTIONS[0])
    assert pos["estimate"] == pytest.approx(0.5 - pos["bias"]) and neg["estimate"] == pytest.approx(-0.5 + neg["bias"]) and pos["bias"] == pytest.approx(neg["bias"])
    assert S.adjusted_estimate(S.Fit(0.5, 0.1, 5.0, 100), 0.2, 0.2, C.DIRECTIONS[1])["estimate"] == pytest.approx(0.5 + pos["bias"])
    assert S.adjusted_estimate(S.Fit(-0.5, 0.1, -5.0, 100), 0.2, 0.2, C.DIRECTIONS[1])["estimate"] == pytest.approx(-0.5 - neg["bias"])
    assert S.adjusted_estimate(S.Fit(0.0, 0.1, 0.0, 100), 0.2, 0.2)["estimate"] == pytest.approx(-S.bias(0.2, 0.2, 0.1, 100))          # Schätzung 0 zählt als nicht negativ


def test_adjusted_interval_is_symmetric_around_the_adjusted_estimate(hand_fit):
    r = S.adjusted_estimate(hand_fit, 0.3, 0.2)
    assert r["hi"] - r["estimate"] == pytest.approx(C.Z_CRIT * r["se"]) and r["estimate"] - r["lo"] == pytest.approx(C.Z_CRIT * r["se"])


def test_evalue_ci_at_the_boundaries():
    assert S.evalue_ci(2.0, 1.0, 3.0) == 1.0 and S.evalue_ci(0.5, 0.3, 1.0) == 1.0 and S.evalue_ci(2.0, 1.01, 3.0) == pytest.approx(S.evalue(1.01)) and S.evalue_ci(0.5, 0.3, 0.99) == pytest.approx(S.evalue(0.99))


def test_binary_scenario_is_invariant_to_relabelling_u():
    """Tauscht man die Namen von U = 0 und U = 1, ändert sich nichts: Das Szenario (pu, pd1, pd0, py1, py0) ist dasselbe wie (1 − pu, pd0, pd1, py0, py1)."""
    for args in ((0.5, 0.8, 0.2, 0.6, 0.2), (0.3, 0.5, 0.05, 0.5, 0.1), (0.7, 0.2, 0.8, 0.2, 0.6)):
        a = S.binary_scenario(*args)
        b = S.binary_scenario(1 - args[0], args[2], args[1], args[4], args[3])
        assert a is not None and b is not None and a == pytest.approx(b)


def test_binary_scenario_second_branch_by_hand():
    """U senkt D und Y zugleich (pd1 < pd0, py1 < py0): beobachtetes RR = 0,52 / 0,28, die Verhältnisse sind die gespiegelten (4 und 3), Schranke 2."""
    sc = S.binary_scenario(0.5, 0.2, 0.8, 0.2, 0.6)
    assert sc["rr_obs"] == pytest.approx(0.52 / 0.28) and sc["rr_ud"] == pytest.approx(4.0) and sc["rr_uy"] == pytest.approx(3.0) and sc["bound"] == pytest.approx(2.0)


def test_binary_scenario_ud_ratio_uses_the_larger_direction():
    """pu = 0,1: P(U | D = 1) = 0,5, P(U | D = 0) = 0,0556...: die Verhältnisse 9 (der Anteile) und 0,5 / 0,944 (der Gegenanteile); das größere zählt."""
    sc = S.binary_scenario(0.1, 0.5, 0.0556, 0.5, 0.1)
    p_d = 0.1 * 0.5 + 0.9 * 0.0556
    a, b = 0.1 * 0.5 / p_d, 0.1 * 0.5 / (1 - p_d)
    assert sc["rr_ud"] == pytest.approx(max(a / b, (1 - b) / (1 - a)))
