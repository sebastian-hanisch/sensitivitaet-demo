"""Konstanten: Regler-Grenzen, feste Experimente, Farben und Presets."""

N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 5000, 100, 400
SEED_MAX, DEFAULT_SEED = 999999, 35
Z_CRIT = 1.959964
ALPHA = 0.05

# Welt (wie Stück 1, mit zusätzlichem verborgenem Confounder U): X = 1[GZ·Z + gu·U + η > 0], Y = 80 − BZ·Z − bu·U + τ·X + SIGMA·ε
GZ, BZ, SIGMA, BASE = 1.0, 4.0, 5.0, 80.0
GU_MIN, GU_MAX, GU_STEP, DEFAULT_GU = 0.0, 2.0, 0.25, 0.5          # Stärke von U auf die Wahl des Faktors
BU_MIN, BU_MAX, BU_STEP, DEFAULT_BU = 0.0, 6.0, 0.5, 1.5            # Stärke von U auf den Servicegrad
TAU_MIN, TAU_MAX, TAU_STEP, DEFAULT_TAU = 0.0, 6.0, 0.5, 3.5        # wahre Wirkung
Q_MIN, Q_MAX, Q_STEP, DEFAULT_Q = 0.25, 1.0, 0.25, 1.0              # um welchen Anteil der Effekt fallen soll
K_MIN, K_MAX, K_STEP, DEFAULT_K = 1, 5, 1, 1                        # Benchmark: U ist k-mal so stark wie die beobachtete Schwankung Z
DIRECTIONS = ("Abschwächung", "Verstärkung")
DEFAULT_DIRECTION = DIRECTIONS[0]
GRID_MAX = 0.6                                                     # Konturkarte: partielle R² von 0 bis GRID_MAX

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 1010000
EXP_ID_N, EXP_ID_REPS = 400, 300                                    # Experiment 1: exakte Identität (U bekannt)
EXP_RV_REPS = 400                                                   # Experiment 1: Verteilung des Robustheitswerts ohne und mit U
EXP_PROTECT = ((0.25, 1.0), (0.5, 1.5), (1.0, 3.0), (1.5, 4.0), (2.0, 6.0))   # (gu, bu) der Schutzprüfung
EXP_PROTECT_REPS = 100
EXP_FALSIFY_N, EXP_FALSIFY_REPS = 500, 500                          # Experiment 3
EXP_FALSIFY_GU = (0.0, 0.25, 0.5, 1.0, 1.5)                          # U wirkt auf x mit gu und auf y mit 3 gu
NC_STRONG, NC_WEAK, NC_DZ, NC_DIRECT = 1.5, 0.5, 0.5, 1.0            # negative Kontrolle: Stärke von U auf N (stark, schwach), Z auf N, direkte Wirkung von x (ungültig)
EXP_BENCH_REPS = 300                                                # Experiment 4
EXP_RR = (1.2, 1.5, 2.0, 3.0, 5.0)                                  # Experiment 2: Tabelle E-Wert
BIN_PU, BIN_PD1, BIN_PD0, BIN_PY1, BIN_PY0 = (0.1, 0.3, 0.5, 0.7), (0.05, 0.2, 0.5, 0.8), (0.05, 0.2, 0.5, 0.8), (0.05, 0.2, 0.5, 0.8), (0.02, 0.1, 0.3, 0.5)

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_CONTROL = "#4c78a8"


def preset(**kw) -> dict:
    base = dict(gu=DEFAULT_GU, bu=DEFAULT_BU, tau=DEFAULT_TAU, n=DEFAULT_N, seed=DEFAULT_SEED, q=DEFAULT_Q, k=DEFAULT_K, direction=DEFAULT_DIRECTION)
    base.update(kw)
    return base


PRESETS = {
    "Standard (Voreinstellung)": preset(),
    "Kein verborgener Confounder": preset(gu=0.0, bu=0.0),
    "Starker verborgener Confounder": preset(gu=1.5, bu=4.5),
    "Schwacher Effekt": preset(tau=1.0),
    "Große Stichprobe": preset(n=2000),
    "Wirkung null": preset(tau=0.0),
}
PRESET_HELP = {
    "Standard (Voreinstellung)": "Ein verborgener Confounder von mittlerer Stärke: Die Schätzung mit der beobachteten Schwankung Z liegt unter der Wahrheit. Der Robustheitswert sagt, wie stark U sein müsste, um die Schätzung auf null zu drücken.",
    "Kein verborgener Confounder": "Es gibt kein U: Die Schätzung trifft die Wahrheit. Der Robustheitswert ist trotzdem nicht null: Er fragt nach einer möglichen, nicht nach einer tatsächlichen Ursache.",
    "Starker verborgener Confounder": "U ist stark: Die Schätzung liegt weit unter der Wahrheit, der Robustheitswert ist nicht kleiner als im Standardfall, liegt aber unter der wahren Stärke von U, die negative Kontrolle schlägt fast immer an (Experiment 3).",
    "Schwacher Effekt": "Die wahre Wirkung ist klein: Der Robustheitswert ist klein, schon ein schwacher Confounder kippt die Schlussfolgerung.",
    "Große Stichprobe": "Mehr Daten, gleiche Struktur: Der Robustheitswert bleibt etwa gleich (er misst die Effektstärke), nur die Variante mit Intervall steigt; die Verzerrung durch U bleibt.",
    "Wirkung null": "Es gibt keine Wirkung: Der Robustheitswert ist klein, mit Intervall nahe null (es gibt nichts zu kippen), das Intervall enthält null.",
}
