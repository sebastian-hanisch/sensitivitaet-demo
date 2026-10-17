"""Sensitivität gegen unbeobachtete Confounder (Cinelli/Hazlett 2020, aus dem Gedächtnis; hier an exakten Identitäten und einer numerischen Nullstellensuche geprüft) und der E-Wert
(VanderWeele/Ding 2017) mit der Schranke von Ding/VanderWeele (2016). Regression, partielle R², Robustheitswert, Verzerrung, angepasster Schätzer, Konturkarte, Benchmark, E-Wert.
Nur numpy und scipy (t-Quantile), kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

import sns_constants as C


def ols(y: np.ndarray, columns) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Kleinste Quadrate mit Achsenabschnitt: (Koeffizienten, Standardfehler, Freiheitsgrade, Reste); `columns` sind die Regressoren ohne Achsenabschnitt."""
    d = np.column_stack([np.ones(len(y))] + list(columns))
    beta = np.linalg.solve(d.T @ d, d.T @ y)
    res = y - d @ beta
    df = len(y) - d.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(d.T @ d)) * (res @ res) / df)
    return beta, se, df, res


@dataclass(frozen=True)
class Fit:
    estimate: float
    se: float
    t: float
    df: int

    @property
    def lo(self) -> float:
        return self.estimate - C.Z_CRIT * self.se

    @property
    def hi(self) -> float:
        return self.estimate + C.Z_CRIT * self.se


def treatment_fit(y: np.ndarray, x: np.ndarray, controls) -> Fit:
    """Koeffizient von x in y ~ x + controls."""
    beta, se, df, _ = ols(y, [x, *controls])
    return Fit(float(beta[1]), float(se[1]), float(beta[1] / se[1]), df)


def partial_r2(y: np.ndarray, d: np.ndarray, controls) -> float:
    """Partielles R² von d in der Regression y ~ d + controls: t² / (t² + df)."""
    beta, se, df, _ = ols(y, [d, *controls])
    t2 = float(beta[1] / se[1]) ** 2
    return t2 / (t2 + df)


# ----------------------------------------------------------------------------- Robustheitswert

def cohens_f(t: float, df: int) -> float:
    return abs(t) / np.sqrt(df)


def robustness_value(t: float, df: int, q: float = 1.0) -> float:
    """RV_q: Stärke (gleiches partielles R² von U mit x und mit y), bei der die Schätzung um den Anteil q fällt: ½(√(f_q⁴ + 4 f_q²) − f_q²), f_q = q |t| / √df."""
    f = q * cohens_f(t, df)
    return float(0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2))


def robustness_value_alpha(t: float, df: int, q: float = 1.0, alpha: float = C.ALPHA) -> float:
    """RV_q,α: wie RV_q, aber bis die Untergrenze des (1 − α)-Intervalls der angepassten Schätzung den Wert (1 − q) · Schätzung erreicht (q = 1: die Null). Bei gleicher Stärke s beider Wege ist der angepasste
    Standardfehler konstant, die Gleichung ist s / √(1 − s) = f_q − f_krit und hat dieselbe Lösung wie RV_q mit f_q,α = f_q − f_krit, f_krit = t*_{α, df−1} / √(df − 1); 0, wenn f_q,α ≤ 0."""
    f = q * cohens_f(t, df)
    f_crit = abs(stats.t.ppf(alpha / 2, df - 1)) / np.sqrt(df - 1)
    fqa = f - f_crit
    if fqa <= 0:
        return 0.0
    return float(0.5 * (np.sqrt(fqa ** 4 + 4 * fqa ** 2) - fqa ** 2))


def robustness_value_numeric(fit: Fit, q: float = 1.0, alpha: float = C.ALPHA) -> float:
    """Zweiter Weg zu RV_q,α (Orakel für die geschlossene Form): die kleinste gleiche Stärke s beider Wege, bei der die Untergrenze des (1 − α)-Intervalls der angepassten Schätzung den Wert
    (1 − q) · Schätzung erreicht, aus Verzerrung und angepasstem Standardfehler gerechnet (Nullstellensuche)."""
    tcrit = stats.t.ppf(1 - alpha / 2, fit.df - 1)

    def lower(s: float) -> float:
        return q * abs(fit.estimate) - bias(s, s, fit.se, fit.df) - tcrit * adjusted_se(s, s, fit.se, fit.df)

    return float(optimize.brentq(lower, 0.0, 0.999)) if lower(0.0) > 0 else 0.0


def bias(r2dz: float, r2yz: float, se: float, df: int) -> float:
    """Betrag der Verzerrung durch einen weggelassenen Confounder mit den partiellen R² r2dz (mit der Behandlung) und r2yz (mit dem Ergebnis, gegeben die Behandlung): √(r2yz r2dz / (1 − r2dz)) · se · √df."""
    return float(np.sqrt(r2yz * r2dz / (1 - r2dz)) * se * np.sqrt(df))


def adjusted_se(r2dz: float, r2yz: float, se: float, df: int) -> float:
    return float(se * np.sqrt((1 - r2yz) / (1 - r2dz)) * np.sqrt(df / (df - 1)))


def adjusted_estimate(fit: Fit, r2dz: float, r2yz: float, direction: str = C.DEFAULT_DIRECTION) -> dict:
    """Schätzung, Standardfehler und Intervall, wenn ein Confounder der Stärke (r2dz, r2yz) fehlt. Abschwächung: Die Verzerrung geht in Richtung null (gegen das Vorzeichen der Schätzung), Verstärkung: davon weg."""
    b = bias(r2dz, r2yz, fit.se, fit.df)
    sign = 1.0 if fit.estimate >= 0 else -1.0
    est = fit.estimate - sign * b if direction == C.DIRECTIONS[0] else fit.estimate + sign * b
    se = adjusted_se(r2dz, r2yz, fit.se, fit.df)
    return {"estimate": float(est), "se": se, "lo": float(est - C.Z_CRIT * se), "hi": float(est + C.Z_CRIT * se), "bias": b}


def contour_grid(fit: Fit, direction: str = C.DEFAULT_DIRECTION, points: int = 61, top: float = C.GRID_MAX):
    """Angepasste Schätzung über dem Gitter der partiellen R² (Spalten: r2dz, Zeilen: r2yz). Rückgabe: (Achse, Matrix)."""
    axis = np.linspace(0.0, top, points)
    z = np.array([[adjusted_estimate(fit, rd, ry, direction)["estimate"] for rd in axis] for ry in axis])
    return axis, z


def benchmark(r2dz_obs: float, r2yz_obs: float, k: float) -> tuple[float, float]:
    """U ist k-mal so stark wie die beobachtete Kovariate (beide Wege), gedeckelt unterhalb von 1."""
    return min(k * r2dz_obs, 0.99), min(k * r2yz_obs, 0.99)


# ----------------------------------------------------------------------------- E-Wert

def evalue(rr: float) -> float:
    """E-Wert: die Stärke (Risikoverhältnis von U mit x und mit y), die ein Confounder haben muss, um ein beobachtetes Risikoverhältnis rr ganz zu erklären: RR + √(RR (RR − 1)); RR < 1 wird gespiegelt."""
    rr = max(rr, 1.0 / rr)
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def evalue_ci(rr: float, lo: float, hi: float) -> float:
    """E-Wert für die dem Wert 1 nähere Intervallgrenze; enthält das Intervall die 1, ist er 1."""
    if lo <= 1.0 <= hi:
        return 1.0
    return evalue(lo if rr > 1.0 else hi)


def ding_bound(rr_ud: float, rr_uy: float) -> float:
    """Obere Schranke des Verzerrungsfaktors durch einen Confounder mit den Risikoverhältnissen rr_ud (mit der Behandlung) und rr_uy (mit dem Ergebnis): rr_ud rr_uy / (rr_ud + rr_uy − 1)."""
    return rr_ud * rr_uy / (rr_ud + rr_uy - 1.0)


def binary_scenario(pu: float, pd1: float, pd0: float, py1: float, py0: float) -> dict | None:
    """Binärer Confounder U, binäre Behandlung D, binäres Ergebnis Y, **wahre Wirkung null** (Y hängt nur von U ab): beobachtetes Risikoverhältnis und die Risikoverhältnisse von U mit D und Y.
    P(U=1) = pu, P(D=1 | U=1) = pd1, P(D=1 | U=0) = pd0, P(Y=1 | U=1) = py1, P(Y=1 | U=0) = py0. None, wenn das beobachtete RR unter 1 liegt (dann wird gespiegelt, hier nicht gebraucht)."""
    p_d = pu * pd1 + (1 - pu) * pd0
    p_y_d1 = (pu * pd1 * py1 + (1 - pu) * pd0 * py0) / p_d
    p_y_d0 = (pu * (1 - pd1) * py1 + (1 - pu) * (1 - pd0) * py0) / (1 - p_d)
    rr_obs = p_y_d1 / p_y_d0
    if rr_obs < 1.0:
        return None
    pu_d1, pu_d0 = pu * pd1 / p_d, pu * (1 - pd1) / (1 - p_d)
    big, small = max(pu_d1, pu_d0), min(pu_d1, pu_d0)
    rr_ud = max(big / small, (1 - small) / (1 - big))                     # Verhältnis der Anteile von U bei D = 1 und D = 0, in der größeren Richtung
    rr_uy = max(py1 / py0, py0 / py1)
    return {"rr_obs": float(rr_obs), "rr_ud": float(rr_ud), "rr_uy": float(rr_uy), "bound": float(ding_bound(rr_ud, rr_uy))}
