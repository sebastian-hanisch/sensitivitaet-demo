"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil) oder sind exakte Rechnungen; Fließkomma-Reihenfolgen können je Bibliotheksversion die letzte Stelle ändern, deshalb
Toleranzen von einer halben Einheit der gedruckten Stelle. Die Aussage „Placebo, zufällige Ursache und Teilstichprobe schlagen nie an“ ist Konstruktion (sie verändern x, den Schätzer oder
die Daten, nicht die Annahme über U), kein Befund; der Befund ist die Gegenseite, die negative Kontrolle.
"""

import numpy as np
import pytest

import sns_constants as C
import sns_evaluation as ev


def test_default_sample():
    a = ev.analyse(ev.Settings())
    assert a.truth == 3.5 and a.fit.estimate == pytest.approx(2.60, abs=0.005) and a.fit.se == pytest.approx(0.582, abs=0.0006) and a.fit.t == pytest.approx(4.46, abs=0.005) and a.fit.df == 397
    assert a.fit.lo == pytest.approx(1.45, abs=0.005) and a.fit.hi == pytest.approx(3.74, abs=0.005) and a.covers is True
    assert a.full.estimate == pytest.approx(3.62, abs=0.005) and a.full.se == pytest.approx(0.587, abs=0.0006)
    assert a.rv == pytest.approx(0.200, abs=0.0006) and a.rv_alpha == pytest.approx(0.117, abs=0.0006) and a.f_cohen == pytest.approx(0.224, abs=0.0006)
    assert a.r2dz_u == pytest.approx(0.091, abs=0.0006) and a.r2yz_u == pytest.approx(0.079, abs=0.0006) and a.r2dz_z == pytest.approx(0.267, abs=0.0006) and a.r2yz_z == pytest.approx(0.278, abs=0.0006)
    assert a.bias_exact == pytest.approx(1.03, abs=0.005) and a.benchmark["estimate"] == pytest.approx(-1.10, abs=0.005) and a.benchmark["bias"] == pytest.approx(3.70, abs=0.005)
    assert ev.verdict_instance(a)[0] == "info"


def test_hand_values_of_the_text():
    from sns_sensitivity import adjusted_estimate, bias, evalue, robustness_value, Fit

    assert robustness_value(2.0, 4) == pytest.approx(0.618, abs=0.0005) and evalue(2.0) == pytest.approx(3.414, abs=0.0005) and evalue(3.0) == pytest.approx(5.449, abs=0.0005) and evalue(4.0) == pytest.approx(7.464, abs=0.0005)
    assert bias(0.5, 0.5, 1.0, 100) == pytest.approx(7.07, abs=0.005) and adjusted_estimate(Fit(10.0, 1.0, 10.0, 100), 0.5, 0.5)["estimate"] == pytest.approx(2.93, abs=0.005)


def test_experiment_1_identity_and_protection(identity):
    r = identity
    assert r["dev_bias"] < 1e-12 and r["dev_se"] < 1e-14 and r["dev_rv"] < 1e-12
    no, w = r["dists"]["ohne U"], r["dists"]["mit U"]
    assert no["mean"] == pytest.approx(0.248, abs=0.0006) and no["sd"] == pytest.approx(0.038, abs=0.0006) and w["mean"] == pytest.approx(0.178, abs=0.0006) and w["sd"] == pytest.approx(0.043, abs=0.0006)
    assert no["mean_alpha"] == pytest.approx(0.171, abs=0.0006) and w["mean_alpha"] == pytest.approx(0.093, abs=0.0006)
    assert [p["tested"] for p in r["protect"]] == [100, 91, 0, 0, 0] and r["tested"] == 191 and r["violations"] == 0


def test_experiment_2_evalue(evalue_res):
    r = evalue_res
    assert r["scenarios"] == 378 and r["violations"] == 0 and r["min_gap"] == pytest.approx(0.0, abs=1e-9)
    assert [round(t["evalue"], 2) for t in r["table"]] == [1.69, 2.37, 3.41, 5.45, 9.47] and [t["rr"] for t in r["table"]] == [1.2, 1.5, 2.0, 3.0, 5.0]
    assert all(t["bound_at_e"] == pytest.approx(t["rr"]) for t in r["table"])


def test_experiment_3_falsification(falsify):
    rows = {r["gu"]: r for r in falsify["rows"]}
    for gu, err in ((0.0, 0.02), (0.25, -0.30), (0.5, -1.12), (1.0, -3.52), (1.5, -6.14)):
        assert rows[gu]["error"] == pytest.approx(err, abs=0.005), gu
    for gu, strong, weak in ((0.0, 0.054, 0.044), (0.25, 0.184, 0.082), (0.5, 0.522, 0.100), (1.0, 0.948, 0.208), (1.5, 0.980, 0.312)):
        assert rows[gu]["negative_control"] == pytest.approx(strong, abs=0.0006) and rows[gu]["negative_control_weak"] == pytest.approx(weak, abs=0.0006), gu
    assert [r["placebo"] for r in falsify["rows"]] == pytest.approx([0.070, 0.054, 0.056, 0.056, 0.056], abs=0.0006)
    assert [r["subset"] for r in falsify["rows"]] == pytest.approx([0.046, 0.046, 0.062, 0.064, 0.064], abs=0.0006) and [r["random_cause"] for r in falsify["rows"]] == [0.0] * 5
    assert falsify["invalid_alarm"] == pytest.approx(0.448, abs=0.0006) and max(max(r["placebo"], r["random_cause"], r["subset"]) for r in falsify["rows"]) == pytest.approx(0.07, abs=0.005)


def test_experiment_4_benchmark(bench):
    r = bench
    assert r["estimate"] == pytest.approx(2.60, abs=0.005) and r["rv"] == pytest.approx(0.200, abs=0.0006) and r["rz_d"] == pytest.approx(0.267, abs=0.0006) and r["rz_y"] == pytest.approx(0.278, abs=0.0006)
    t = {x["k"]: x for x in r["table"]}
    assert (round(t[1]["bias"], 2), round(t[1]["down"], 2), round(t[1]["up"], 2)) == (3.70, -1.10, 6.29) and (round(t[2]["bias"], 2), round(t[2]["down"], 2), round(t[2]["up"], 2)) == (9.28, -6.69, 11.88)
    assert (round(t[3]["down"], 2), round(t[3]["up"], 2)) == (-18.76, 23.95) and r["covered"] == 300 and r["k_true_mean"] == pytest.approx(0.40, abs=0.005)


def test_the_text_numbers_of_the_app_follow_the_findings(identity, evalue_res, falsify, bench):
    assert "4e-13 bzw. 1e-15" in ev.verdict_identity(identity)[1] and "0,248" in ev.verdict_identity(identity)[1] and "0,178" in ev.verdict_identity(identity)[1] and "In 191 Läufen" in ev.verdict_identity(identity)[1]
    assert "E-Wert 2,37" in ev.verdict_evalue(evalue_res)[1] and "5 %, bei 0,5 in 52 % und bei 1,5 in 98 %" in ev.verdict_falsify(falsify)[1] and "in 45 % an" in ev.verdict_falsify(falsify)[1]
    assert "26,7 % von x und 27,8 % von y" in ev.verdict_benchmark(bench)[1] and "−1,10 und 6,29" in ev.verdict_benchmark(bench)[1] and "−18,76 und 23,95" in ev.verdict_benchmark(bench)[1]
