"""Falsifikation: die negative Kontrolle (ein Ergebnis, auf das x keine Wirkung haben darf) und die drei Standard-Widerlegungstests (Placebo-Behandlung, zufällige gemeinsame Ursache,
Teilstichprobe). Jeder liefert ein Alarmsignal je Datensatz. Nur numpy und scipy, kein Streamlit."""

from __future__ import annotations

import numpy as np
from scipy import stats

import sns_constants as C
import sns_sensitivity as S
from kausal_generator import SplitMix64


def negative_control(x: np.ndarray, z: np.ndarray, nco: np.ndarray, alpha: float = C.ALPHA) -> dict:
    """Wirkt x auf das Ergebnis, auf das es nicht wirken darf (Regression nco ~ x + z)? Alarm bei p < alpha."""
    f = S.treatment_fit(nco, x, [z])
    p = 2 * stats.t.sf(abs(f.t), f.df)
    return {"estimate": f.estimate, "p": float(p), "alarm": bool(p < alpha)}


def placebo_treatment(y: np.ndarray, x: np.ndarray, z: np.ndarray, uniform: np.ndarray, alpha: float = C.ALPHA) -> dict:
    """Placebo: x wird zufällig umgeordnet (Reihenfolge nach `uniform`); die Wirkung der umgeordneten Behandlung muss null sein. Alarm bei p < alpha."""
    f = S.treatment_fit(y, x[np.argsort(uniform)], [z])
    p = 2 * stats.t.sf(abs(f.t), f.df)
    return {"estimate": f.estimate, "p": float(p), "alarm": bool(p < alpha)}


def random_common_cause(y: np.ndarray, x: np.ndarray, z: np.ndarray, noise: np.ndarray, base: S.Fit, threshold: float = 2.0) -> dict:
    """Zufällige gemeinsame Ursache: eine Zufallsvariable als zusätzlicher Regressor; die Schätzung darf sich nicht verändern. Alarm bei einer Veränderung über `threshold` Standardfehlern."""
    f = S.treatment_fit(y, x, [z, noise])
    shift = abs(f.estimate - base.estimate) / base.se
    return {"estimate": f.estimate, "shift": float(shift), "alarm": bool(shift > threshold)}


def subset_refuter(y: np.ndarray, x: np.ndarray, z: np.ndarray, uniform: np.ndarray, base: S.Fit, threshold: float = 2.0) -> dict:
    """Teilstichprobe: die Hälfte der Daten (die kleineren Werte von `uniform`); die Schätzung darf sich nicht verändern. Alarm bei einer Veränderung über `threshold` Standardfehlern."""
    idx = np.argsort(uniform)[: len(y) // 2]
    f = S.treatment_fit(y[idx], x[idx], [z[idx]])
    shift = abs(f.estimate - base.estimate) / base.se
    return {"estimate": f.estimate, "shift": float(shift), "alarm": bool(shift > threshold)}


def all_tests(sim: dict, rep: int, nco: np.ndarray, rng: SplitMix64) -> dict:
    """Alle vier Tests für einen Lauf (Zeile `rep` der Simulation); `nco` ist das Ergebnis der negativen Kontrolle, `rng` liefert die Zufallsgrößen der Widerlegungstests."""
    x, y, z = sim["x"][rep], sim["y"][rep], sim["z"][rep]
    n = len(y)
    base = S.treatment_fit(y, x, [z])
    u = rng.uniform(2 * n)
    return {"base": base, "negative_control": negative_control(x, z, nco[rep]), "placebo": placebo_treatment(y, x, z, u[:n]), "random_cause": random_common_cause(y, x, z, rng.normal(n), base),
            "subset": subset_refuter(y, x, z, u[n:], base)}
