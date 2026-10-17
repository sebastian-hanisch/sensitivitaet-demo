"""Vehikel: Wahrheit, Stichprobe, die Verzerrung durch U und die negative Kontrolle."""

import numpy as np
import pytest

import sns_constants as C
import sns_sensitivity as S
import sns_world as W
from kausal_generator import SplitMix64


def test_truth_is_tau_whatever_u_does():
    assert W.truth(W.World()) == C.DEFAULT_TAU and W.truth(W.World(gu=2.0, bu=6.0, tau=1.5)) == 1.5 and W.truth(W.World(tau=0.0)) == 0.0


def test_simulate_shapes_streams_and_binary_treatment():
    sim = W.simulate(W.World(), 50, SplitMix64(3), reps=4)
    assert set(sim) == {"z", "u", "x", "y", "nu"} and all(v.shape == (4, 50) for v in sim.values()) and set(np.unique(sim["x"])) <= {0.0, 1.0}
    again = W.simulate(W.World(), 50, SplitMix64(3), reps=4)
    assert all((again[k] == sim[k]).all() for k in sim) and not (W.simulate(W.World(), 50, SplitMix64(4), reps=4)["y"] == sim["y"]).all()


def test_outcome_equation_by_hand():
    """Y = 80 − 4 Z − bu U + τ X + 5 ε: Der Rest nach Abzug der Terme hat Standardabweichung 5 und ist unabhängig von Z, U und X."""
    sim = W.simulate(W.World(0.5, 1.5, 3.5), 200000, SplitMix64(5))
    resid = sim["y"][0] - (C.BASE - C.BZ * sim["z"][0] - 1.5 * sim["u"][0] + 3.5 * sim["x"][0])
    assert resid.std() == pytest.approx(5.0, abs=0.05) and abs(resid.mean()) < 0.05
    for k in ("z", "u", "x"):
        assert abs(np.corrcoef(resid, sim[k][0])[0, 1]) < 0.01


def test_treatment_depends_on_z_and_u_through_the_probit_rule():
    sim = W.simulate(W.World(1.0, 0.0, 3.5), 300000, SplitMix64(6))
    z, u, x = sim["z"][0], sim["u"][0], sim["x"][0]
    assert x.mean() == pytest.approx(0.5, abs=0.005)
    assert x[(z + u) > 2.0].mean() > 0.9 and x[(z + u) < -2.0].mean() < 0.1
    assert np.corrcoef(x, z)[0, 1] > 0.3 and np.corrcoef(x, u)[0, 1] > 0.3


def test_u_is_independent_of_z():
    sim = W.simulate(W.World(), 100000, SplitMix64(7))
    assert abs(np.corrcoef(sim["z"][0], sim["u"][0])[0, 1]) < 0.01


def test_leaving_out_u_biases_the_estimate_downwards_and_the_full_model_is_right():
    sim = W.simulate(W.World(1.0, 3.0, 3.5), 200000, SplitMix64(8))
    z, u, x, y = (sim[k][0] for k in ("z", "u", "x", "y"))
    assert S.treatment_fit(y, x, [z, u]).estimate == pytest.approx(3.5, abs=0.15)
    assert S.treatment_fit(y, x, [z]).estimate < 3.5 - 2.5                                    # Phase 0: Fehler etwa −3,5 bei gu = 1, bu = 3


def test_without_u_the_adjusted_estimate_is_unbiased():
    sim = W.simulate(W.World(0.0, 0.0, 3.5), 200000, SplitMix64(9))
    assert S.treatment_fit(sim["y"][0], sim["x"][0], [sim["z"][0]]).estimate == pytest.approx(3.5, abs=0.15)


def test_negative_control_outcome_by_hand():
    sim = {"u": np.array([1.0, 2.0]), "z": np.array([10.0, 20.0]), "x": np.array([1.0, 0.0]), "nu": np.array([0.1, 0.2])}
    assert W.negative_control_outcome(sim, 1.5, 0.5, 0.0) == pytest.approx([1.5 + 5.0 + 0.5, 3.0 + 10.0 + 1.0])
    assert W.negative_control_outcome(sim, 0.0, 0.0, 1.0) == pytest.approx([1.0 + 0.5, 0.0 + 1.0])
    assert W.negative_control_outcome(sim, 1.0)[0] == pytest.approx(1.0 + C.NC_DZ * 10.0 + C.SIGMA * 0.1)


def test_negative_control_shares_u_with_the_outcome():
    sim = W.simulate(W.World(0.5, 1.5, 3.5), 100000, SplitMix64(10))
    n = W.negative_control_outcome(sim, 1.5)[0]
    assert np.corrcoef(n, sim["u"][0])[0, 1] > 0.2 and abs(S.treatment_fit(n, sim["x"][0], [sim["z"][0], sim["u"][0]]).estimate) < 0.2
