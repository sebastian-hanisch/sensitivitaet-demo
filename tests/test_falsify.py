"""Falsifikation: negative Kontrolle und die drei Widerlegungstests; Verhalten an exakten und an simulierten Daten."""

import numpy as np
import pytest

import sns_constants as C
import sns_falsify as F
import sns_sensitivity as S
import sns_world as W
from kausal_generator import SplitMix64


def test_negative_control_alarms_on_a_clear_effect_and_stays_quiet_on_pure_noise():
    rng = SplitMix64(1)
    z, x = rng.normal(300), (rng.uniform(300) > 0.5).astype(float)
    assert F.negative_control(x, z, 2.0 * x + rng.normal(300), alpha=0.05)["alarm"] is True
    quiet = F.negative_control(x, z, rng.normal(300))
    assert quiet["alarm"] == (quiet["p"] < 0.05) and 0.0 <= quiet["p"] <= 1.0


def test_negative_control_estimate_is_the_treatment_coefficient():
    rng = SplitMix64(2)
    z, x = rng.normal(100), (rng.uniform(100) > 0.5).astype(float)
    nco = 0.7 * x + 0.3 * z + rng.normal(100)
    assert F.negative_control(x, z, nco)["estimate"] == pytest.approx(S.treatment_fit(nco, x, [z]).estimate)


def test_placebo_uses_the_order_of_the_uniform_numbers():
    rng = SplitMix64(3)
    z, x, y = rng.normal(80), (rng.uniform(80) > 0.5).astype(float), rng.normal(80)
    u = rng.uniform(80)
    out = F.placebo_treatment(y, x, z, u)
    assert out["estimate"] == pytest.approx(S.treatment_fit(y, x[np.argsort(u)], [z]).estimate) and out["alarm"] == (out["p"] < 0.05)
    assert F.placebo_treatment(y, x, z, np.arange(80.0))["estimate"] == pytest.approx(S.treatment_fit(y, x, [z]).estimate)         # Reihenfolge unverändert: das echte x


def test_random_common_cause_shift_by_hand():
    rng = SplitMix64(4)
    z, x = rng.normal(100), (rng.uniform(100) > 0.5).astype(float)
    y = 1.0 * x + z + rng.normal(100)
    base = S.treatment_fit(y, x, [z])
    out = F.random_common_cause(y, x, z, rng.normal(100), base)
    assert out["shift"] == pytest.approx(abs(out["estimate"] - base.estimate) / base.se) and out["shift"] < 1.0 and out["alarm"] is False
    assert F.random_common_cause(y, x, z, y.copy(), base, threshold=0.0)["alarm"] is True                    # eine echte Ursache (y selbst) verändert die Schätzung


def test_subset_refuter_uses_the_smaller_half():
    rng = SplitMix64(5)
    z, x = rng.normal(100), (rng.uniform(100) > 0.5).astype(float)
    y = x + z + rng.normal(100)
    base = S.treatment_fit(y, x, [z])
    u = rng.uniform(100)
    idx = np.argsort(u)[:50]
    out = F.subset_refuter(y, x, z, u, base)
    assert out["estimate"] == pytest.approx(S.treatment_fit(y[idx], x[idx], [z[idx]]).estimate) and out["shift"] == pytest.approx(abs(out["estimate"] - base.estimate) / base.se)


def test_all_tests_keys_and_determinism():
    sim = W.simulate(W.World(), 200, SplitMix64(6), reps=3)
    nco = W.negative_control_outcome(sim, C.NC_STRONG)
    a, b = F.all_tests(sim, 1, nco, SplitMix64(7)), F.all_tests(sim, 1, nco, SplitMix64(7))
    assert set(a) == {"base", "negative_control", "placebo", "random_cause", "subset"} and a["base"] == S.treatment_fit(sim["y"][1], sim["x"][1], [sim["z"][1]])
    assert a["placebo"] == b["placebo"] and a["subset"] == b["subset"] and a["random_cause"] == b["random_cause"]


def test_refuters_stay_quiet_where_the_negative_control_alarms():
    """Bei starkem U: negative Kontrolle alarmiert fast immer, die drei Widerlegungstests nie über der Zufallsquote."""
    sim = W.simulate(W.World(1.0, 3.0), 500, SplitMix64(8), reps=120)
    nco = W.negative_control_outcome(sim, C.NC_STRONG)
    rng = SplitMix64(9)
    res = [F.all_tests(sim, r, nco, rng) for r in range(120)]

    def rate(key):
        return np.mean([x[key]["alarm"] for x in res])

    assert rate("negative_control") > 0.8 and rate("placebo") < 0.15 and rate("random_cause") < 0.05 and rate("subset") < 0.15


def test_an_invalid_negative_control_alarms_without_a_confounder():
    sim = W.simulate(W.World(0.0, 0.0), 500, SplitMix64(10), reps=120)
    nco = W.negative_control_outcome(sim, 0.0, C.NC_DZ, C.NC_DIRECT)
    assert np.mean([F.negative_control(sim["x"][r], sim["z"][r], nco[r])["alarm"] for r in range(120)]) > 0.3
