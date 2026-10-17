"""Verdikte: Art und Text je Lage; jede Zahl im Text stammt aus dem Befund, nicht aus einer Konstanten."""

import sns_evaluation as ev


def test_instance_verdict_without_u_is_a_success():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(gu=0.0, bu=0.0, n=2000)))
    assert kind == "success" and "Hier gibt es kein U" in text and "Der Robustheitswert RV ist" in text and "Tatsächlich erklärt U" not in text


def test_instance_verdict_with_u_names_its_true_strength():
    a = ev.analyse(ev.Settings())
    kind, text = ev.verdict_instance(a)
    assert kind == "info" and f"Tatsächlich erklärt U {ev.pct(a.r2dz_u, 1)} von x und {ev.pct(a.r2yz_u, 1)} von y." in text and "20 % der verbleibenden Streuung" in text and "um 100 % zu senken" in text
    assert "mit Intervall: 12 %" in text


def test_instance_verdict_warns_when_the_interval_misses_because_of_u():
    for seed in range(200):
        a = ev.analyse(ev.Settings(gu=1.0, bu=3.0, seed=seed))
        if not a.covers:
            kind, text = ev.verdict_instance(a)
            assert kind == "warning" and "verfehlt die Wahrheit, weil U fehlt" in text
            return
    raise AssertionError("bei starkem U sollte das Intervall in 200 Seeds mindestens einmal verfehlen")


def test_q_appears_in_the_text_when_it_is_not_one():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(q=0.5)))
    assert "RV_0,50" in text and "um 50 % zu senken" in text


def test_identity_verdict(identity):
    kind, text = ev.verdict_identity(identity)
    assert kind == "info" and "Mit bekanntem U stimmen Verzerrung und Standardfehler auf" in text and "Der Wert zeigt nicht, ob U da ist" in text and f"In {identity['tested']} Läufen" in text
    assert f"in {identity['tested'] - identity['violations']} über null" in text


def test_evalue_verdict(evalue_res):
    kind, text = ev.verdict_evalue(evalue_res)
    assert kind == "info" and "In 378 binären Szenarien" in text and "in 378 unter der Schranke" in text and "E-Wert 2,37" in text and "E-Wert 3,41" in text and "kleinste Lücke 0,0000" in text


def test_falsify_verdict(falsify):
    kind, text = ev.verdict_falsify(falsify)
    rows = {r["gu"]: r for r in falsify["rows"]}
    assert kind == "info" and f"bei U-Stärke 0 in {ev.pct(rows[0.0]['negative_control'])}" in text and f"bei 1,5 in {ev.pct(rows[1.5]['negative_control'])}" in text and "Placebo, zufällige Ursache und Teilstichprobe schlagen" in text
    assert f"in {ev.pct(falsify['invalid_alarm'])} an" in text


def test_benchmark_verdict(bench):
    kind, text = ev.verdict_benchmark(bench)
    assert kind == "info" and "Wäre U so stark wie Z" in text and "bei dreifacher Stärke zwischen" in text and "in 300 von 300 Läufen" in text and f"{ev.pct(bench['rz_d'], 1)} von x" in text
