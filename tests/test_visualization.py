"""Figuren und Tabellen: bauen ohne Fehler, und die gezeichneten Zahlen sind die der Auswertung."""

import pytest

import sns_constants as C
import sns_evaluation as ev
import sns_visualization as V


def test_contour_has_the_zero_level_line_the_robustness_point_and_the_benchmarks():
    a = ev.analyse(ev.Settings())
    fig = V.build_contour(a)
    names = [t.name for t in fig.data]
    assert any(t.type == "contour" for t in fig.data) and "Robustheitswert" in names and "tatsächliches U" in names and "1-mal so stark wie Z" in names
    rv = next(t for t in fig.data if t.name == "Robustheitswert")
    assert rv.x[0] == pytest.approx(a.rv) and rv.y[0] == pytest.approx(a.rv)
    star = next(t for t in fig.data if t.name == "tatsächliches U")
    assert star.x[0] == pytest.approx(a.r2dz_u) and star.y[0] == pytest.approx(a.r2yz_u)
    line = next(t for t in fig.data if t.type == "contour" and t.contours.type == "constraint")
    assert line.contours.value == pytest.approx(0.0, abs=1e-12)                          # q = 1: Schätzung bis auf null


def test_contour_level_follows_q_and_benchmarks_leave_the_plot_when_too_big():
    a = ev.analyse(ev.Settings(q=0.5))
    line = next(t for t in V.build_contour(a).data if t.type == "contour" and t.contours.type == "constraint")
    assert line.contours.value == pytest.approx(0.5 * a.fit.estimate)
    names = [t.name for t in V.build_contour(a).data]
    assert "3-mal so stark wie Z" not in names                                           # 3 × 0,27 liegt außerhalb der Karte bis 0,6


def test_figures_and_tables_build_for_every_preset():
    for p in C.PRESETS.values():
        a = ev.analyse(ev.Settings(**p))
        assert V.build_estimates_chart(a) is not None and V.build_contour(a) is not None and len(V.build_summary_table(a)) == 8


def test_estimates_chart_has_three_points_and_the_truth_line():
    a = ev.analyse(ev.Settings())
    fig = V.build_estimates_chart(a)
    assert len([t for t in fig.data if t.mode == "markers"]) == 3 and len(fig.layout.shapes) == 2 and fig.layout.shapes[0].x0 == pytest.approx(a.truth)


def test_summary_table_quotes_the_analysis():
    a = ev.analyse(ev.Settings())
    t = V.build_summary_table(a)
    assert "Robustheitswert RV für q = 1,00" in t.iloc[2]["Größe"] and t.iloc[2]["Wert"] == f"{ev.pct(a.rv, 1)} (mit Intervall: {ev.pct(a.rv_alpha, 1)})"
    assert "1-mal so stark wie Z (abschwächung)" in t.iloc[4]["Größe"] and t.iloc[7]["Wert"].startswith("1,0276 gegen 1,0276")


def test_experiment_figures_and_tables_build(identity, evalue_res, falsify, bench):
    assert V.build_rv_chart(identity) is not None and len(V.build_identity_table(identity)) == 6 and len(V.build_protect_table(identity)) == 5
    assert V.build_evalue_chart(evalue_res) is not None and len(V.build_evalue_table(evalue_res)) == 5
    assert V.build_falsify_chart(falsify) is not None and len(V.build_falsify_table(falsify)) == 5 and len(V.build_falsify_chart(falsify).data) == 5
    assert len(V.build_benchmark_table(bench)) == 3


def test_falsify_table_by_hand(falsify):
    t = V.build_falsify_table(falsify)
    assert list(t["U auf x / y"]) == ["0,00 / 0,00", "0,25 / 0,75", "0,50 / 1,50", "1,00 / 3,00", "1,50 / 4,50"]
    assert list(t["negative Kontrolle (stark)"])[0] == ev.pct(falsify["rows"][0]["negative_control"]) and list(t["zufällige Ursache"]) == ["0 %"] * 5
