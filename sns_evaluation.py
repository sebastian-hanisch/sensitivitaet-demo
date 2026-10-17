"""Auswertung: eine Stichprobe mit allen Kennzahlen, die vier Experimente über feste Ströme und die Verdikte. Nur numpy und die Module der Linie, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sns_constants as C
import sns_falsify as F
import sns_sensitivity as S
import sns_world as W
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -digits * 1e-3 else x
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits) + " %"


@dataclass(frozen=True)
class Settings:
    gu: float = C.DEFAULT_GU
    bu: float = C.DEFAULT_BU
    tau: float = C.DEFAULT_TAU
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    q: float = C.DEFAULT_Q
    k: int = C.DEFAULT_K
    direction: str = C.DEFAULT_DIRECTION

    @property
    def world(self) -> W.World:
        return W.World(self.gu, self.bu, self.tau)

    @property
    def has_u(self) -> bool:
        return self.gu > 0 and self.bu > 0


@dataclass
class Analysis:
    settings: Settings
    truth: float
    fit: S.Fit              # Y ~ X + Z (U fehlt)
    full: S.Fit             # Y ~ X + Z + U (Orakel, nur weil simuliert)
    covers: bool
    full_covers: bool
    rv: float               # RV_q
    rv_alpha: float         # RV_q,alpha
    rv_one: float           # RV_1
    f_cohen: float
    r2dz_u: float           # wahre Stärke von U (Orakel)
    r2yz_u: float
    r2dz_z: float           # beobachtete Kovariate Z (Benchmark)
    r2yz_z: float
    bias_exact: float       # |Schätzung ohne U − Schätzung mit U|
    bias_formula: float     # dieselbe Größe aus den partiellen R² von U
    se_formula: float       # angepasster Standardfehler aus den partiellen R² von U
    benchmark: dict         # angepasste Schätzung, wenn U k-mal so stark wie Z ist


def analyse(s: Settings) -> Analysis:
    sim = W.simulate(s.world, s.n, SplitMix64(s.seed))
    z, u, x, y = (sim[k][0] for k in ("z", "u", "x", "y"))
    fit = S.treatment_fit(y, x, [z])
    full = S.treatment_fit(y, x, [z, u])
    r2dz_u, r2yz_u = S.partial_r2(x, u, [z]), S.partial_r2(y, u, [x, z])
    r2dz_z, r2yz_z = S.partial_r2(x, z, []), S.partial_r2(y, z, [x])
    bench = S.adjusted_estimate(fit, *S.benchmark(r2dz_z, r2yz_z, s.k), s.direction)
    return Analysis(s, W.truth(s.world), fit, full, bool(fit.lo <= s.tau <= fit.hi), bool(full.lo <= s.tau <= full.hi), S.robustness_value(fit.t, fit.df, s.q),
                    S.robustness_value_alpha(fit.t, fit.df, s.q), S.robustness_value(fit.t, fit.df, 1.0), S.cohens_f(fit.t, fit.df), r2dz_u, r2yz_u, r2dz_z, r2yz_z,
                    abs(fit.estimate - full.estimate), S.bias(r2dz_u, r2yz_u, fit.se, fit.df), S.adjusted_se(r2dz_u, r2yz_u, fit.se, fit.df), bench)


# ----------------------------------------------------------------------------- Experiment 1: exakte Identität, Schutz, RV ohne und mit U

def identity_experiment() -> dict:
    """(a) Mit bekanntem U gilt die Verzerrungsformel und die des Standardfehlers exakt (größte Abweichung über die Läufe); (b) der Robustheitswert in geschlossener Form gegen die Nullstellensuche;
    (c) Verteilung von RV_1 und RV_1,α ohne und mit U; (d) Schutzaussage: Ist die wahre Stärke beider Wege kleiner als RV_1, bleibt die Schätzung mit U über null."""
    w = W.World(0.5, 1.5, C.DEFAULT_TAU)
    sim = W.simulate(w, C.EXP_ID_N, SplitMix64(C.SEED_BASE), C.EXP_ID_REPS)
    dev_bias = dev_se = 0.0
    for r in range(C.EXP_ID_REPS):
        z, u, x, y = (sim[k][r] for k in ("z", "u", "x", "y"))
        fit, full = S.treatment_fit(y, x, [z]), S.treatment_fit(y, x, [z, u])
        rd, ry = S.partial_r2(x, u, [z]), S.partial_r2(y, u, [x, z])
        dev_bias = max(dev_bias, abs(abs(fit.estimate - full.estimate) - S.bias(rd, ry, fit.se, fit.df)))
        dev_se = max(dev_se, abs(S.adjusted_se(rd, ry, fit.se, fit.df) - full.se))
    dists = {}
    dev_rv = 0.0
    for i, (label, gu, bu) in enumerate((("ohne U", 0.0, 0.0), ("mit U", 0.5, 1.5))):
        sim = W.simulate(W.World(gu, bu, C.DEFAULT_TAU), C.EXP_ID_N, SplitMix64(C.SEED_BASE + 1 + i), C.EXP_RV_REPS)
        rv1, rva = [], []
        for r in range(C.EXP_RV_REPS):
            fit = S.treatment_fit(sim["y"][r], sim["x"][r], [sim["z"][r]])
            rv1.append(S.robustness_value(fit.t, fit.df))
            rva.append(S.robustness_value_alpha(fit.t, fit.df))
            dev_rv = max(dev_rv, abs(rva[-1] - S.robustness_value_numeric(fit)))
        dists[label] = {"rv": rv1, "rv_alpha": rva, "mean": float(np.mean(rv1)), "sd": float(np.std(rv1)), "mean_alpha": float(np.mean(rva)), "sd_alpha": float(np.std(rva))}
    tested = violations = 0
    protect = []
    for i, (gu, bu) in enumerate(C.EXP_PROTECT):
        sim = W.simulate(W.World(gu, bu, C.DEFAULT_TAU), C.EXP_ID_N, SplitMix64(C.SEED_BASE + 10 + i), C.EXP_PROTECT_REPS)
        t_i = v_i = 0
        for r in range(C.EXP_PROTECT_REPS):
            z, u, x, y = (sim[k][r] for k in ("z", "u", "x", "y"))
            fit = S.treatment_fit(y, x, [z])
            rv = S.robustness_value(fit.t, fit.df)
            if max(S.partial_r2(x, u, [z]), S.partial_r2(y, u, [x, z])) < rv:
                t_i += 1
                v_i += S.treatment_fit(y, x, [z, u]).estimate <= 0
        protect.append({"gu": gu, "bu": bu, "tested": t_i, "violations": int(v_i)})
        tested += t_i
        violations += int(v_i)
    return {"reps": C.EXP_ID_REPS, "n": C.EXP_ID_N, "dev_bias": dev_bias, "dev_se": dev_se, "dev_rv": dev_rv, "dists": dists, "protect": protect, "tested": tested, "violations": violations}


# ----------------------------------------------------------------------------- Experiment 2: E-Wert, exakt

def evalue_experiment() -> dict:
    """Binäre Szenarien mit wahrer Wirkung null: Das beobachtete Risikoverhältnis liegt nie über der Schranke rr_ud rr_uy / (rr_ud + rr_uy − 1) und erreicht sie. Dazu die Tabelle E-Wert zu RR."""
    scen = viol = 0
    min_gap = float("inf")
    for pu in C.BIN_PU:
        for pd1 in C.BIN_PD1:
            for pd0 in C.BIN_PD0:
                for py1 in C.BIN_PY1:
                    for py0 in C.BIN_PY0:
                        if py0 >= py1:
                            continue
                        sc = S.binary_scenario(pu, pd1, pd0, py1, py0)
                        if sc is None:
                            continue
                        scen += 1
                        viol += sc["rr_obs"] > sc["bound"] + 1e-12
                        min_gap = min(min_gap, sc["bound"] - sc["rr_obs"])
    table = [{"rr": rr, "evalue": S.evalue(rr), "bound_at_e": S.ding_bound(S.evalue(rr), S.evalue(rr))} for rr in C.EXP_RR]
    return {"scenarios": scen, "violations": int(viol), "min_gap": float(max(min_gap, 0.0)), "table": table}          # ein Rundungsrest unter null ist keine Verletzung


# ----------------------------------------------------------------------------- Experiment 3: Falsifikation

def falsification_experiment() -> dict:
    rows = []
    for i, gu in enumerate(C.EXP_FALSIFY_GU):
        w = W.World(gu, 3 * gu, C.DEFAULT_TAU)
        sim = W.simulate(w, C.EXP_FALSIFY_N, SplitMix64(C.SEED_BASE + 20 + i), C.EXP_FALSIFY_REPS)
        strong = W.negative_control_outcome(sim, C.NC_STRONG)
        weak = W.negative_control_outcome(sim, C.NC_WEAK)
        rng = SplitMix64(C.SEED_BASE + 40 + i)
        err, nc_s, nc_w, pl, rc, sub = [], [], [], [], [], []
        for r in range(C.EXP_FALSIFY_REPS):
            res = F.all_tests(sim, r, strong, rng)
            err.append(res["base"].estimate - C.DEFAULT_TAU)
            nc_s.append(res["negative_control"]["alarm"])
            nc_w.append(F.negative_control(sim["x"][r], sim["z"][r], weak[r])["alarm"])
            pl.append(res["placebo"]["alarm"])
            rc.append(res["random_cause"]["alarm"])
            sub.append(res["subset"]["alarm"])
        rows.append({"gu": gu, "bu": 3 * gu, "error": float(np.mean(err)), "negative_control": float(np.mean(nc_s)), "negative_control_weak": float(np.mean(nc_w)), "placebo": float(np.mean(pl)),
                     "random_cause": float(np.mean(rc)), "subset": float(np.mean(sub))})
    sim = W.simulate(W.World(0.0, 0.0, C.DEFAULT_TAU), C.EXP_FALSIFY_N, SplitMix64(C.SEED_BASE + 60), C.EXP_FALSIFY_REPS)
    invalid = W.negative_control_outcome(sim, 0.0, C.NC_DZ, C.NC_DIRECT)
    invalid_alarm = float(np.mean([F.negative_control(sim["x"][r], sim["z"][r], invalid[r])["alarm"] for r in range(C.EXP_FALSIFY_REPS)]))
    return {"n": C.EXP_FALSIFY_N, "reps": C.EXP_FALSIFY_REPS, "rows": rows, "invalid_alarm": invalid_alarm}


# ----------------------------------------------------------------------------- Experiment 4: Benchmark

def benchmark_experiment(ks=(1, 2, 3)) -> dict:
    """Standard-Stichprobe: angepasste Schätzung, wenn U k-mal so stark wie Z ist; dazu über viele Läufe, wie oft die Schranke mit dem wahren Vielfachen k die Schätzung mit U enthält (Identität, kein Zufall)."""
    a = analyse(Settings())
    table = []
    for k in ks:
        rd, ry = S.benchmark(a.r2dz_z, a.r2yz_z, k)
        adj = S.adjusted_estimate(a.fit, rd, ry)
        up = S.adjusted_estimate(a.fit, rd, ry, C.DIRECTIONS[1])
        table.append({"k": k, "r2dz": rd, "r2yz": ry, "bias": adj["bias"], "down": adj["estimate"], "up": up["estimate"]})
    sim = W.simulate(W.World(), C.EXP_ID_N, SplitMix64(C.SEED_BASE + 70), C.EXP_BENCH_REPS)
    cover = 0
    ratios = []
    for r in range(C.EXP_BENCH_REPS):
        z, u, x, y = (sim[k][r] for k in ("z", "u", "x", "y"))
        fit, full = S.treatment_fit(y, x, [z]), S.treatment_fit(y, x, [z, u])
        rz_d, rz_y = S.partial_r2(x, z, []), S.partial_r2(y, z, [x])
        k_true = max(S.partial_r2(x, u, [z]) / rz_d, S.partial_r2(y, u, [x, z]) / rz_y)
        ratios.append(k_true)
        rd, ry = S.benchmark(rz_d, rz_y, k_true)
        cover += abs(full.estimate - fit.estimate) <= S.bias(rd, ry, fit.se, fit.df) + 1e-9
    return {"table": table, "rz_d": a.r2dz_z, "rz_y": a.r2yz_z, "ru_d": a.r2dz_u, "ru_y": a.r2yz_u, "estimate": a.fit.estimate, "rv": a.rv_one, "covered": int(cover), "reps": C.EXP_BENCH_REPS,
            "k_true_mean": float(np.mean(ratios))}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    s = a.settings
    f = a.fit
    text = (f"Die Schätzung ohne U ist {de(f.estimate)} (Wahrheit {de(a.truth)}, Fehler {signed(f.estimate - a.truth)}), das Intervall {'deckt' if a.covers else 'verfehlt'} die Wahrheit. "
            f"Der Robustheitswert RV{'' if s.q == 1.0 else '_' + de(s.q, 2)} ist {pct(a.rv)}: Ein verborgener Confounder müsste mit x und mit y je mindestens {pct(a.rv)} der verbleibenden Streuung erklären, "
            f"um die Schätzung um {pct(s.q)} zu senken (mit Intervall: {pct(a.rv_alpha)}).")
    if not s.has_u:
        return "success", text + " Hier gibt es kein U; der Robustheitswert fragt trotzdem nach einem möglichen."
    tail = f" Tatsächlich erklärt U {pct(a.r2dz_u, 1)} von x und {pct(a.r2yz_u, 1)} von y."
    if not a.covers:
        return "warning", text + tail + " Die Schätzung verfehlt die Wahrheit, weil U fehlt."
    return "info", text + tail


def verdict_identity(res: dict) -> tuple[str, str]:
    no, with_u = res["dists"]["ohne U"], res["dists"]["mit U"]
    return "info", (f"Mit bekanntem U stimmen Verzerrung und Standardfehler auf {res['dev_bias']:.0e} bzw. {res['dev_se']:.0e} mit den Formeln überein; der Robustheitswert in geschlossener Form weicht von der Nullstellensuche um höchstens "
                    f"{res['dev_rv']:.0e} ab. Ohne U ist RV_1 im Mittel {de(no['mean'], 3)}, mit U {de(with_u['mean'], 3)} (Streuung je etwa {de(with_u['sd'], 2)}): Der Wert zeigt nicht, ob U da ist. "
                    f"In {res['tested']} Läufen mit wahrer Stärke unter dem Robustheitswert bleibt die Schätzung mit U in {res['tested'] - res['violations']} über null.")


def verdict_evalue(res: dict) -> tuple[str, str]:
    t = {r["rr"]: r for r in res["table"]}
    return "info", (f"In {res['scenarios']} binären Szenarien mit wahrer Wirkung null liegt das beobachtete Risikoverhältnis in {res['scenarios'] - res['violations']} unter der Schranke und erreicht sie (kleinste Lücke "
                    f"{de(res['min_gap'], 4)}). Ein Risikoverhältnis von {de(1.5, 1)} hat den E-Wert {de(t[1.5]['evalue'], 2)}, eines von {de(2.0, 1)} den E-Wert {de(t[2.0]['evalue'], 2)}.")


def verdict_falsify(res: dict) -> tuple[str, str]:
    rows = {r["gu"]: r for r in res["rows"]}
    mid, top = rows[0.5], rows[1.5]
    return "info", (f"Die negative Kontrolle schlägt bei U-Stärke 0 in {pct(rows[0.0]['negative_control'])}, bei {de(mid['gu'], 1)} in {pct(mid['negative_control'])} und bei {de(top['gu'], 1)} in {pct(top['negative_control'])} an "
                    f"(mit nur schwach geteilter Ursache {pct(rows[0.0]['negative_control_weak'])}, {pct(mid['negative_control_weak'])} und {pct(top['negative_control_weak'])}). Placebo, zufällige Ursache und Teilstichprobe schlagen bei jeder Stärke "
                    f"kaum an (höchstens {pct(max(max(r['placebo'], r['random_cause'], r['subset']) for r in res['rows']))}). Eine ungültige negative Kontrolle schlägt ohne Confounder in {pct(res['invalid_alarm'])} an.")


def verdict_benchmark(res: dict) -> tuple[str, str]:
    t = {r["k"]: r for r in res["table"]}
    return "info", (f"Die beobachtete Schwankung Z erklärt {pct(res['rz_d'], 1)} von x und {pct(res['rz_y'], 1)} von y. Wäre U so stark wie Z, läge die Schätzung ({de(res['estimate'])}) zwischen {de(t[1]['down'])} und {de(t[1]['up'])}; "
                    f"bei dreifacher Stärke zwischen {de(t[3]['down'])} und {de(t[3]['up'])}. Mit dem wahren Vielfachen (im Mittel {de(res['k_true_mean'], 2)}) trägt die Schranke in {res['covered']} von {res['reps']} Läufen.")
