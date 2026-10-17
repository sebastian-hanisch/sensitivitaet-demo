"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen und prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import sns_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SLIDERS = ("gu_slider", "bu_slider", "tau_slider", "q_slider", "k_slider", "n_slider")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_numbers():
    at = _fresh_app()
    m = _metrics(at)
    assert m["Wahre Wirkung"] == "3,50" and m["Schätzung ohne U"] == "+2,595" and m["Robustheitswert RV"] == "20,0 %" and m["Schätzung mit U (Orakel)"] == "+3,623"
    assert at.slider(key="gu_slider").value == C.DEFAULT_GU and at.slider(key="q_slider").value == 1.0 and at.radio(key="direction_radio").value == C.DEFAULT_DIRECTION and at.number_input(key="seed_input").value == C.DEFAULT_SEED
    assert any("Die Schätzung ohne U ist 2,60" in i.value for i in at.info)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_sliders_at_min_and_max_do_not_raise():
    for key in SLIDERS:
        for edge in ("min", "max"):
            at = _fresh_app()
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        for key, state_key in (("gu", "gu_slider"), ("bu", "bu_slider"), ("tau", "tau_slider"), ("n", "n_slider"), ("q", "q_slider"), ("k", "k_slider")):
            assert at.slider(key=state_key).value == preset[key], (name, key)
        assert at.radio(key="direction_radio").value == preset["direction"]


def test_no_confounder_preset_is_a_success_and_the_direction_radio_works():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Kein verborgener Confounder")].click().run(timeout=TIMEOUT)
    assert any("Hier gibt es kein U" in s.value for s in at.success)
    at.radio(key="direction_radio").set_value(C.DIRECTIONS[1]).run(timeout=TIMEOUT)
    assert not at.exception and at.radio(key="direction_radio").value == C.DIRECTIONS[1]


def test_q_changes_the_robustness_value():
    at = _fresh_app()
    at.slider(key="q_slider").set_value(0.5).run(timeout=TIMEOUT)
    assert not at.exception and float(_metrics(at)["Robustheitswert RV"].replace(" %", "").replace(",", ".")) < 20.0


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["gu"] = "9"
    at.query_params["tau"] = "0.74"
    at.query_params["n"] = "50"
    at.query_params["k"] = "99"
    at.query_params["dir"] = "1"
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="gu_slider").value == C.GU_MAX and at.slider(key="tau_slider").value == 0.5 and at.slider(key="n_slider").value == C.N_MIN and at.slider(key="k_slider").value == C.K_MAX
    assert at.radio(key="direction_radio").value == C.DIRECTIONS[1] and at.number_input(key="seed_input").value == C.DEFAULT_SEED


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.slider(key="bu_slider").set_value(3.0).run(timeout=TIMEOUT)
    assert at.query_params["bu"] in ("3.0", ["3.0"])
    at.radio(key="direction_radio").set_value(C.DIRECTIONS[1]).run(timeout=TIMEOUT)
    assert at.query_params["dir"] in ("1", ["1"])
