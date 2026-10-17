"""Gemeinsame Fixtures: die Hand-Instanzen (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand_fit():
    """Mini-Instanz 1: Schätzung 10, Standardfehler 1, t = 10, df = 100; beide partiellen R² 0,5.
    BF = √(0,5 · 0,5 / (1 − 0,5)) = 0,7071; Verzerrung = 0,7071 · 1 · √100 = 7,0711; angepasster Standardfehler = 1 · √(0,5 / 0,5) · √(100 / 99) = 1,00504.
    Abschwächung: 10 − 7,0711 = 2,9289; Verstärkung: 17,0711. Bei der Schätzung −10 (Abschwächung gegen null): −10 + 7,0711 = −2,9289."""
    from sns_sensitivity import Fit

    return Fit(10.0, 1.0, 10.0, 100)


@pytest.fixture
def small_data():
    """Mini-Instanz 2 (acht Einheiten, exakt): c = (0, 0, 0, 0, 1, 1, 1, 1) als Kontrolle, d = (0, 1, 0, 1, 0, 1, 0, 1) als Behandlung, y = 2 d + 3 c + 1 (kein Rest).
    Die Regression von y auf d und c trifft die Koeffizienten (2 und 3) genau; das partielle R² von d ist 1, das von c ebenfalls."""
    c = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    d = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    return c, d, 2.0 * d + 3.0 * c + 1.0


@pytest.fixture(scope="session")
def identity():
    import sns_evaluation as ev

    return ev.identity_experiment()


@pytest.fixture(scope="session")
def evalue_res():
    import sns_evaluation as ev

    return ev.evalue_experiment()


@pytest.fixture(scope="session")
def falsify():
    import sns_evaluation as ev

    return ev.falsification_experiment()


@pytest.fixture(scope="session")
def bench():
    import sns_evaluation as ev

    return ev.benchmark_experiment()
