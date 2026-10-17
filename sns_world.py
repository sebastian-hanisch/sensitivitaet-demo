"""Vehikel: Die Welt aus Stück 1 mit einem zusätzlichen verborgenen Confounder U. Ein Disponent wählt den hohen Faktor X (Probit-Schwelle) nach der beobachteten Schwankung Z und der verborgenen
Standortqualität U; der Servicegrad Y hängt von Z, U und X ab. Die negative Kontrolle N hängt von U und Z ab, nicht von X. Nur numpy und der Generator der Linie, kein Streamlit.

X = 1[GZ·Z + gu·U + η > 0], Y = BASE − BZ·Z − bu·U + τ·X + SIGMA·ε, N = c·U + dz·Z + direct·X + SIGMA·ν; Z, U, η, ε, ν ~ N(0, 1) unabhängig."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sns_constants as C
from kausal_generator import SplitMix64


@dataclass(frozen=True)
class World:
    gu: float = C.DEFAULT_GU
    bu: float = C.DEFAULT_BU
    tau: float = C.DEFAULT_TAU


def truth(w: World) -> float:
    """Wirkung des Faktors auf den Servicegrad: τ (eine Wirkung beim Festsetzen von X)."""
    return w.tau


def simulate(w: World, n: int, rng: SplitMix64, reps: int = 1) -> dict:
    """Alle Größen mit Form (reps, n): z, u, x, y und das Rauschen nu der negativen Kontrolle (siehe `negative_control_outcome`)."""
    draw = rng.normal(reps * n * 5).reshape(reps, n, 5)
    z, u, eta, eps, nu = (draw[..., k] for k in range(5))
    x = (C.GZ * z + w.gu * u + eta > 0).astype(float)
    y = C.BASE - C.BZ * z - w.bu * u + w.tau * x + C.SIGMA * eps
    return {"z": z, "u": u, "x": x, "y": y, "nu": nu}


def negative_control_outcome(sim: dict, c: float, dz: float = C.NC_DZ, direct: float = 0.0) -> np.ndarray:
    """Ein Ergebnis, auf das x keine Wirkung haben soll (direct = 0): N = c·U + dz·Z + direct·X + SIGMA·ν. Mit direct ≠ 0 ist die Kontrolle ungültig."""
    return c * sim["u"] + dz * sim["z"] + direct * sim["x"] + C.SIGMA * sim["nu"]
