"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import sns_constants as C
import sns_evaluation as ev
import sns_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = p[key]
            if spec.choices:
                assert v in spec.choices, (name, key)
                continue
            assert spec.lo <= v <= spec.hi, (name, key)
            assert P.snap(v, spec) == pytest.approx(v), (name, key)


def test_first_preset_is_the_default_setting():
    p = C.PRESETS["Standard (Voreinstellung)"]
    assert ev.Settings() == ev.Settings(**p) and next(iter(C.PRESETS)) == "Standard (Voreinstellung)"
    assert all(P.SETTING_SPECS[sk].default == p[k] for k, sk in P.PRESET_KEYS.items())


def test_presets_differ_and_each_changes_at_most_one_setting():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)
    base = C.PRESETS["Standard (Voreinstellung)"]
    for name, p in C.PRESETS.items():
        assert len([k for k in p if p[k] != base[k]]) <= 2, name                  # "kein U" und "starkes U" verändern beide Wege
    assert C.PRESETS["Kein verborgener Confounder"]["gu"] == 0.0 and C.PRESETS["Wirkung null"]["tau"] == 0.0 and C.PRESETS["Große Stichprobe"]["n"] > C.DEFAULT_N


def test_snap_clamps_rounds_and_maps_the_direction():
    gu = P.SETTING_SPECS["gu_slider"]
    assert P.snap(5.0, gu) == C.GU_MAX and P.snap(-1.0, gu) == C.GU_MIN and P.snap(0.6, gu) == pytest.approx(0.5) and P.snap(0.7, gu) == pytest.approx(0.75)
    n = P.SETTING_SPECS["n_slider"]
    assert P.snap(50, n) == C.N_MIN and P.snap(10 ** 6, n) == C.N_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100
    d = P.SETTING_SPECS["direction_radio"]
    assert P.snap(0, d) == C.DIRECTIONS[0] and P.snap(1, d) == C.DIRECTIONS[1] and P.snap(7, d) == C.DIRECTIONS[1] and P.snap(-3, d) == C.DIRECTIONS[0]
    assert P.url_value("direction_radio", C.DIRECTIONS[1]) == 1 and P.url_value("gu_slider", 0.5) == 0.5 and P.snap(3, P.SETTING_SPECS["k_slider"]) == 3 and P.snap(99, P.SETTING_SPECS["k_slider"]) == C.K_MAX


def test_bounds_defaults_and_params():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("q_slider") == (C.Q_MIN, C.Q_MAX)
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"gu", "bu", "tau", "n", "seed", "q", "k", "dir"} and C.DEFAULT_N in range(C.N_MIN, C.N_MAX + 1, C.N_STEP)
    assert C.DEFAULT_DIRECTION == C.DIRECTIONS[0] and C.Q_MAX == 1.0 and C.DEFAULT_K in range(C.K_MIN, C.K_MAX + 1)


def test_experiment_constants_are_consistent():
    assert C.EXP_ID_REPS >= 200 and C.EXP_RV_REPS >= 200 and C.EXP_FALSIFY_REPS >= 400 and C.EXP_FALSIFY_GU[0] == 0.0 and C.EXP_FALSIFY_GU == tuple(sorted(C.EXP_FALSIFY_GU))
    assert C.NC_WEAK < C.NC_STRONG and C.NC_DIRECT > 0 and len(C.EXP_PROTECT) == 5 and C.EXP_RR == tuple(sorted(C.EXP_RR)) and C.GRID_MAX < 1.0
    assert abs(C.DEFAULT_GU - 0.5) < 1e-12 and C.DEFAULT_BU == 1.5 and C.DEFAULT_TAU == 3.5


def test_default_world_is_not_a_degenerate_case():
    """Zweig-Test: Die Voreinstellung hat ein tatsächliches U (Fehler ≠ 0) und einen Robustheitswert, der nicht an einer Grenze liegt."""
    a = ev.analyse(ev.Settings())
    assert abs(a.fit.estimate - a.truth) > 0.5 and 0.05 < a.rv < 0.9 and a.r2dz_u > 0.01
