"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import sns_constants as C
import sns_sensitivity as S
from sns_evaluation import Analysis, de, pct, signed


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_estimates_chart(a: Analysis) -> go.Figure:
    """Schätzung ohne U (mit Intervall), die Schätzung mit U (Orakel) und die angepasste Schätzung bei Benchmark k; senkrecht die Wahrheit."""
    s = a.settings
    b = a.benchmark
    rows = [("ohne U (Y ~ X + Z)", a.fit.estimate, a.fit.lo, a.fit.hi, C.COLOR_CONTROL), ("mit U (Orakel, nur weil simuliert)", a.full.estimate, a.full.lo, a.full.hi, C.COLOR_GOOD),
            (f"angepasst: U {s.k}-mal so stark wie Z", b["estimate"], b["lo"], b["hi"], C.COLOR_NEUTRAL)]
    fig = go.Figure()
    for label, est, lo, hi, color in rows:
        fig.add_trace(go.Scatter(x=[est], y=[label], mode="markers", marker=dict(color=color, size=11), showlegend=False,
                                 error_x=dict(type="data", array=[hi - est], arrayminus=[est - lo], color=color, thickness=2), hovertemplate=f"{label}: {de(est, 2)} [{de(lo, 2)}; {de(hi, 2)}]<extra></extra>"))
    fig.add_vline(x=a.truth, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"Wahrheit {de(a.truth, 2)}", annotation_position="top")
    fig.add_vline(x=0.0, line=dict(color="#888888", width=1, dash="dot"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Wirkung des hohen Faktors auf den Servicegrad (Punkte)")
    return _fixed(fig, 300)


def build_contour(a: Analysis) -> go.Figure:
    """Konturkarte: angepasste Schätzung über den partiellen R² des Confounders mit x (waagerecht) und mit y (senkrecht); Nulllinie; Diagonale mit RV; Benchmarks 1× bis 3× Z; tatsächliches U."""
    s = a.settings
    axis, z = S.contour_grid(a.fit, s.direction)
    fig = go.Figure()
    fig.add_trace(go.Contour(x=axis, y=axis, z=z, colorscale="RdBu", zmid=0.0, contours=dict(showlabels=True, labelfont=dict(size=10)), colorbar=dict(title="Schätzung"), hovertemplate="R² mit x %{x:.2f}, R² mit y %{y:.2f}: %{z:.2f}<extra></extra>"))
    fig.add_trace(go.Contour(x=axis, y=axis, z=z, contours=dict(type="constraint", operation="=", value=(1 - s.q) * a.fit.estimate), line=dict(color="black", width=3), showscale=False, hoverinfo="skip",
                             name=f"Schätzung um {pct(s.q)} gesenkt"))
    fig.add_trace(go.Scatter(x=[a.rv], y=[a.rv], mode="markers+text", marker=dict(color="black", size=11, symbol="diamond"), text=[f"RV {pct(a.rv)}"], textposition="top right", name="Robustheitswert"))
    for k in (1, 2, 3):
        rd, ry = S.benchmark(a.r2dz_z, a.r2yz_z, k)
        if rd <= C.GRID_MAX and ry <= C.GRID_MAX:
            fig.add_trace(go.Scatter(x=[rd], y=[ry], mode="markers+text", marker=dict(color=C.COLOR_TRUTH, size=9, symbol="square"), text=[f"{k}× Z"], textposition="bottom right", name=f"{k}-mal so stark wie Z"))
    fig.add_trace(go.Scatter(x=[a.r2dz_u], y=[a.r2yz_u], mode="markers+text", marker=dict(color="white", size=11, symbol="star", line=dict(color="black", width=1.5)), text=["U (tatsächlich)"], textposition="top center",
                             name="tatsächliches U"))
    fig.update_xaxes(title="partielles R² des Confounders mit x", range=[0, C.GRID_MAX])
    fig.update_yaxes(title="partielles R² des Confounders mit y", range=[0, C.GRID_MAX])
    return _fixed(fig, 560)


def build_summary_table(a: Analysis) -> pd.DataFrame:
    s = a.settings
    rows = [
        ("Schätzung ohne U (Y ~ X + Z)", f"{signed(a.fit.estimate, 3)}, Standardfehler {de(a.fit.se, 3)}, t = {de(a.fit.t, 2)}, Intervall {de(a.fit.lo, 2)} bis {de(a.fit.hi, 2)}"),
        ("Cohens f der Behandlung", de(a.f_cohen, 3)),
        (f"Robustheitswert RV für q = {de(s.q, 2)}", f"{pct(a.rv, 1)} (mit Intervall: {pct(a.rv_alpha, 1)})"),
        ("Beobachtete Kovariate Z: partielles R² mit x und mit y", f"{pct(a.r2dz_z, 1)} und {pct(a.r2yz_z, 1)}"),
        (f"Benchmark: U {s.k}-mal so stark wie Z ({s.direction.lower()})", f"{signed(a.benchmark['estimate'], 2)}, Intervall {de(a.benchmark['lo'], 2)} bis {de(a.benchmark['hi'], 2)}"),
        ("Tatsächliches U (Orakel): partielles R² mit x und mit y", f"{pct(a.r2dz_u, 1)} und {pct(a.r2yz_u, 1)}"),
        ("Schätzung mit U (Orakel)", f"{signed(a.full.estimate, 3)}, Intervall {de(a.full.lo, 2)} bis {de(a.full.hi, 2)}"),
        ("Verzerrung durch U: gemessen gegen Formel", f"{de(a.bias_exact, 4)} gegen {de(a.bias_formula, 4)}"),
    ]
    return pd.DataFrame([{"Größe": k, "Wert": v} for k, v in rows])


# ----------------------------------------------------------------------------- Experiment 1

def build_rv_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    for label, color in (("ohne U", C.COLOR_GOOD), ("mit U", C.COLOR_BAD)):
        fig.add_trace(go.Histogram(x=res["dists"][label]["rv"], name=f"{label}: RV₁", marker_color=color, opacity=0.65, nbinsx=30))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Robustheitswert RV₁ (400 Läufe je Fall)")
    fig.update_yaxes(title="Läufe")
    return _fixed(fig, 340)


def build_identity_table(res: dict) -> pd.DataFrame:
    d = res["dists"]
    return pd.DataFrame([
        {"Prüfung": "Verzerrung durch U gegen die Formel (größte Abweichung)", "Ergebnis": f"{res['dev_bias']:.1e}"},
        {"Prüfung": "Standardfehler mit U gegen die Formel (größte Abweichung)", "Ergebnis": f"{res['dev_se']:.1e}"},
        {"Prüfung": "RV₁,α geschlossen gegen Nullstellensuche (größte Abweichung)", "Ergebnis": f"{res['dev_rv']:.1e}"},
        {"Prüfung": "RV₁ ohne U: Mittel (Streuung)", "Ergebnis": f"{de(d['ohne U']['mean'], 3)} ({de(d['ohne U']['sd'], 3)})"},
        {"Prüfung": "RV₁ mit U: Mittel (Streuung)", "Ergebnis": f"{de(d['mit U']['mean'], 3)} ({de(d['mit U']['sd'], 3)})"},
        {"Prüfung": "RV₁,α ohne U / mit U: Mittel", "Ergebnis": f"{de(d['ohne U']['mean_alpha'], 3)} / {de(d['mit U']['mean_alpha'], 3)}"},
    ])


def build_protect_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"U wirkt auf x / auf y": f"{de(r['gu'], 2)} / {de(r['bu'], 1)}", "Läufe mit wahrer Stärke unter RV₁": r["tested"], "davon Schätzung mit U ≤ 0": r["violations"]} for r in res["protect"]])


# ----------------------------------------------------------------------------- Experiment 2

def build_evalue_chart(res: dict) -> go.Figure:
    import numpy as np

    rr = np.linspace(1.0, 5.5, 60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr, y=[S.evalue(v) for v in rr], mode="lines", line=dict(color=C.COLOR_GOOD, width=3), name="E-Wert = RR + √(RR (RR − 1))"))
    fig.add_trace(go.Scatter(x=[r["rr"] for r in res["table"]], y=[r["evalue"] for r in res["table"]], mode="markers+text", marker=dict(color=C.COLOR_TRUTH, size=10), text=[de(r["evalue"], 2) for r in res["table"]],
                             textposition="top left", name="Tabelle"))
    fig.update_xaxes(title="beobachtetes Risikoverhältnis RR")
    fig.update_yaxes(title="E-Wert")
    return _fixed(fig, 340)


def build_evalue_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"beobachtetes RR": de(r["rr"], 1), "E-Wert": de(r["evalue"], 2), "Schranke bei RR_UD = RR_UY = E-Wert": de(r["bound_at_e"], 2)} for r in res["table"]])


# ----------------------------------------------------------------------------- Experiment 3

def build_falsify_chart(res: dict) -> go.Figure:
    gu = [r["gu"] for r in res["rows"]]
    fig = go.Figure()
    for key, label, color, dash in (("negative_control", "negative Kontrolle (geteilte Ursache stark)", C.COLOR_GOOD, "solid"), ("negative_control_weak", "negative Kontrolle (geteilte Ursache schwach)", "#6aa49d", "dash"),
                                     ("placebo", "Placebo-Behandlung", C.COLOR_NEUTRAL, "solid"), ("random_cause", "zufällige gemeinsame Ursache", C.COLOR_BAD, "solid"), ("subset", "Teilstichprobe", C.COLOR_CONTROL, "solid")):
        fig.add_trace(go.Scatter(x=gu, y=[100 * r[key] for r in res["rows"]], mode="lines+markers", name=label, line=dict(color=color, dash=dash, width=3)))
    fig.add_hline(y=5.0, line=dict(color="#888888", width=1, dash="dot"), annotation_text="5 %", annotation_position="right")
    fig.update_xaxes(title="Stärke von U auf die Wahl des Faktors (auf den Servicegrad: das Dreifache)")
    fig.update_yaxes(title="Alarmquote (%)", range=[-2, 104])
    return _fixed(fig, 400)


def build_falsify_table(res: dict) -> pd.DataFrame:
    out = []
    for r in res["rows"]:
        out.append({"U auf x / y": f"{de(r['gu'], 2)} / {de(r['bu'], 2)}", "mittlerer Fehler der Schätzung": signed(r["error"], 2), "negative Kontrolle (stark)": pct(r["negative_control"]),
                    "negative Kontrolle (schwach)": pct(r["negative_control_weak"]), "Placebo": pct(r["placebo"]), "zufällige Ursache": pct(r["random_cause"]), "Teilstichprobe": pct(r["subset"])})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- Experiment 4

def build_benchmark_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"U ist … mal so stark wie Z": r["k"], "partielles R² mit x / mit y": f"{pct(r['r2dz'], 1)} / {pct(r['r2yz'], 1)}", "Verzerrung bis": de(r["bias"], 2),
                          "Schätzung bei Abschwächung": signed(r["down"], 2), "Schätzung bei Verstärkung": signed(r["up"], 2)} for r in res["table"]])
