"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Alle Regler sind in jeder Einstellung wirksam und werden immer gezeichnet;
es gibt weder KEPT-Regler noch berechnete Grenzen. Die Richtung der Verzerrung ist eine Auswahl (Adresszeile: Index 0 oder 1)."""

import random
from dataclasses import dataclass

import streamlit as st

import sns_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float
    hi: float
    step: float
    choices: tuple | None = None


SETTING_SPECS = {
    "gu_slider": SettingSpec("gu", float, C.DEFAULT_GU, C.GU_MIN, C.GU_MAX, C.GU_STEP),
    "bu_slider": SettingSpec("bu", float, C.DEFAULT_BU, C.BU_MIN, C.BU_MAX, C.BU_STEP),
    "tau_slider": SettingSpec("tau", float, C.DEFAULT_TAU, C.TAU_MIN, C.TAU_MAX, C.TAU_STEP),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX, C.N_STEP),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, 1),
    "q_slider": SettingSpec("q", float, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX, C.Q_STEP),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX, C.K_STEP),
    "direction_radio": SettingSpec("dir", int, C.DEFAULT_DIRECTION, 0, len(C.DIRECTIONS) - 1, 1, C.DIRECTIONS),
}
PRESET_KEYS = {"gu": "gu_slider", "bu": "bu_slider", "tau": "tau_slider", "n": "n_slider", "seed": "seed_input", "q": "q_slider", "k": "k_slider", "direction": "direction_radio"}


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen (bei einer Auswahl: auf einen gültigen Index)."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step > 0 and (spec.caster is float or spec.step > 1 or spec.choices):
        value = round((value - spec.lo) / spec.step) * spec.step + spec.lo
        value = round(value, 6) if spec.caster is float else int(value)
    return spec.choices[int(value)] if spec.choices else value


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = snap(spec.caster(float(qp[spec.url_param])), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def url_value(state_key: str, value):
    spec = SETTING_SPECS[state_key]
    return spec.choices.index(value) if spec.choices else value


def sync_query_params(values: dict):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(url_value(state_key, value))
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
