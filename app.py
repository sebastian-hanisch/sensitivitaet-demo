"""Sensitivität und Falsifikation: Wie stark müsste die verborgene Ursache sein? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes Stück der Linie Kausale Diagramme und Adjustierung. Wenn eine Wirkung nicht bestimmbar ist, lässt sich messen, wie viel Annahme die Schlussfolgerung trägt: Robustheitswert und
Konturkarte (Cinelli/Hazlett), E-Wert (VanderWeele/Ding), und die Annahmen lassen sich auf Spuren abklopfen: negative Kontrolle gegen die Standard-Widerlegungstests.
Kern: sns_sensitivity.py, sns_falsify.py, sns_world.py.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import sns_constants as C
from sns_evaluation import (
    Settings,
    analyse,
    benchmark_experiment,
    de,
    evalue_experiment,
    falsification_experiment,
    identity_experiment,
    pct,
    signed,
    verdict_benchmark,
    verdict_evalue,
    verdict_falsify,
    verdict_identity,
    verdict_instance,
)
from sns_pdf_export import build_pdf
from sns_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from sns_visualization import (
    build_benchmark_table,
    build_contour,
    build_estimates_chart,
    build_evalue_chart,
    build_evalue_table,
    build_falsify_chart,
    build_falsify_table,
    build_identity_table,
    build_protect_table,
    build_rv_chart,
    build_summary_table,
)

st.set_page_config(page_title="Sensitivität und Falsifikation – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _identity():
    return identity_experiment()


@st.cache_data(show_spinner=False)
def _evalue():
    return evalue_experiment()


@st.cache_data(show_spinner=SPINNER)
def _falsify():
    return falsification_experiment()


@st.cache_data(show_spinner=False)
def _benchmark():
    return benchmark_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🎚️ Sensitivität und Falsifikation")
st.markdown(
    """
**Nicht bestimmbar heißt nicht wertlos.** Dies ist das elfte Stück der Linie **Kausale Diagramme und Adjustierung**. Das zehnte hat gezeigt, wann keine Tür offen ist; hier, was man dann noch sagen kann: **wie stark**
eine verborgene Ursache sein müsste, um die Schlussfolgerung zu kippen (Robustheitswert, Konturkarte, E-Wert), und was sich an den Annahmen **abklopfen** lässt (negative Kontrolle). Beides beantwortet nicht,
ob die Ursache existiert.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/sensitivitaet-verborgene-ursache-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Wer aus Beobachtungsdaten auf eine Wirkung schließt, übersieht vielleicht eine Ursache, die niemand gemessen hat. Ob es sie gibt, lässt sich aus den Daten nicht klären. Die Demo fragt deshalb anders: Wie stark müsste so eine verborgene Ursache sein, damit die Schlussfolgerung kippt? Der **Robustheitswert** und der **E-Wert** antworten darauf mit einer Zahl. Die **negative Kontrolle** prüft zusätzlich die Annahme, dass nichts Verborgenes mitspielt, die üblichen Widerlegungstests prüfen dagegen nur die Technik. Alles ist simuliert, deshalb kennt die Demo die Wahrheit und kann zeigen, was davon den verborgenen Einfluss wirklich sieht und was nur beruhigt.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/sensitivitaet-verborgene-ursache-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So liest man die Konturkarte", expanded=True):
    st.markdown(
        """
1. **Die Schätzung** ist die Regression des Servicegrads auf den hohen Faktor und die beobachtete Schwankung Z. Ein verborgener Confounder U fehlt darin.
2. **Zwei Zahlen beschreiben einen Confounder:** wie viel von der Streuung des Faktors (waagerecht) und wie viel von der Streuung des Servicegrads (senkrecht) er erklärt, jeweils nach Abzug dessen, was die Regression schon erklärt (partielles R²).
3. **Die Karte** zeigt, auf welchen Wert die Schätzung fiele, wenn ein Confounder dieser Stärke fehlte. **Die schwarze Linie** ist der Wert, bei dem die Schlussfolgerung kippt (die Schätzung um den Anteil q gesunken).
4. **Der Robustheitswert RV** ist der Punkt auf der Diagonalen, wo beide Wege gleich stark sind und die Linie erreicht wird: je größer, desto robuster. **Die Quadrate** zeigen, wie stark ein Confounder wäre, der so stark wie die beobachtete Schwankung Z ist.
5. **Nicht gezeigt wird, ob U existiert.** Das bleibt eine Annahme; nur das simulierte U (Stern) ist bekannt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Verborgener Confounder U**")
    gu = st.slider("U wirkt auf die Wahl des Faktors", *bounds("gu_slider"), step=C.GU_STEP, key="gu_slider", help="Stärke, mit der die verborgene Standortqualität den hohen Faktor beeinflusst (Probit-Schwelle). 0: kein U.")
    bu = st.slider("U wirkt auf den Servicegrad", *bounds("bu_slider"), step=C.BU_STEP, key="bu_slider", help="Punkte Servicegrad je Standardabweichung von U (bei negativem Vorzeichen im Modell). 0: kein U.")
    tau = st.slider("Wahre Wirkung des Faktors", *bounds("tau_slider"), step=C.TAU_STEP, key="tau_slider", help="Punkte Servicegrad, die der hohe Faktor wirklich bringt.")
    st.markdown("**Sensitivität**")
    q = st.slider("Um welchen Anteil soll die Schätzung fallen (q)?", *bounds("q_slider"), step=C.Q_STEP, key="q_slider", help="q = 1: bis auf null. q = 0,5: auf die Hälfte.")
    k = st.slider("Benchmark: U ist … mal so stark wie Z", *bounds("k_slider"), step=C.K_STEP, key="k_slider", help="Eine Annahme über die Stärke von U, gemessen an der beobachteten Schwankung (Konvention der Demo: k-faches partielles R² von Z; nach Cinelli/Hazlett wären die Intervalle breiter).")
    direction = st.radio("Richtung der Verzerrung", C.DIRECTIONS, key="direction_radio", help="Abschwächung: Die Verzerrung geht gegen die Schätzung (Worst Case für die Schlussfolgerung). Verstärkung: in dieselbe Richtung.")
    st.markdown("**Daten**")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Stichprobengröße der simulierten Daten.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"gu_slider": round(float(gu), 2), "bu_slider": round(float(bu), 1), "tau_slider": round(float(tau), 1), "n_slider": int(n), "seed_input": int(seed), "q_slider": round(float(q), 2),
                   "k_slider": int(k), "direction_radio": direction})

settings = Settings(round(float(gu), 2), round(float(bu), 1), round(float(tau), 1), int(n), int(seed), round(float(q), 2), int(k), direction)
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
c1.metric("Wahre Wirkung", de(a.truth, 2), delta="aus dem Modell, nicht aus den Daten", delta_color="off")
c2.metric("Schätzung ohne U", signed(a.fit.estimate, 3), delta=f"Fehler {signed(a.fit.estimate - a.truth, 2)}, Intervall {de(a.fit.lo, 2)} bis {de(a.fit.hi, 2)}", delta_color="off",
          help="Koeffizient des Faktors in der Regression des Servicegrads auf Faktor und Schwankung.")
c3.metric("Robustheitswert RV", pct(a.rv, 1), delta=f"mit Intervall {pct(a.rv_alpha, 1)}", delta_color="off",
          help="Gleiche Stärke von U mit Faktor und Servicegrad (partielles R²), bei der die Schätzung um den Anteil q fällt.")
c4.metric("Schätzung mit U (Orakel)", signed(a.full.estimate, 3), delta=f"Fehler {signed(a.full.estimate - a.truth, 2)}", delta_color="off", help="Nur weil simuliert: Die Regression mit U. In der Praxis unbekannt.")

left, right = st.columns([1.15, 1])
with left:
    st.plotly_chart(build_contour(a), width="stretch", key="contour_chart")
    st.caption("Farbe: Schätzung, wenn ein Confounder der Stärke (waagerecht, senkrecht) fehlte. Schwarze Linie: die Schätzung ist um den Anteil q gesunken. Raute: Robustheitswert. Quadrate: U so stark wie Z (1× bis 3×). Stern: das tatsächliche U.")
with right:
    st.plotly_chart(build_estimates_chart(a), width="stretch", key="estimates_chart")
    st.caption("Punkt mit Intervall; senkrecht die Wahrheit und die Null. Das Orakel kennt U und ist in der Praxis nicht möglich.")
verdict(verdict_instance(a))
st.dataframe(build_summary_table(a), hide_index=True, width="stretch")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="sensitivitaet.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Die Formeln sind exakt, der Robustheitswert zeigt U nicht an")
res1 = _identity()
st.caption(f"Standardwelt (U wirkt mit 0,5 und 1,5), n = {res1['n']}, {res1['reps']} feste Wiederholungen für die Identität; je {C.EXP_RV_REPS} Läufe für die Verteilung des Robustheitswerts ohne und mit U.")
verdict(verdict_identity(res1))
st.dataframe(build_identity_table(res1), hide_index=True, width="stretch")
st.plotly_chart(build_rv_chart(res1), width="stretch", key="rv_chart")
st.caption("Schutzprüfung: Läufe, in denen die wahre Stärke von U auf beiden Wegen unter RV₁ liegt; die Schätzung mit U bleibt dort über null (das folgt aus der Verzerrungsformel und ist hier bestätigt). Bei starkem U gibt es keine solchen Läufe (RV₁ ist dann kleiner als die Stärke).")
st.dataframe(build_protect_table(res1), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Der E-Wert (binäres Ergebnis)")
res2 = _evalue()
st.caption("Binärer Confounder U, binäre Behandlung, binäres Ergebnis, **wahre Wirkung null** (das Ergebnis hängt nur von U ab): alle Kombinationen aus je vier Wahrscheinlichkeiten. Der E-Wert ist die Stärke "
           "(Risikoverhältnis von U mit Behandlung und mit Ergebnis), die ein Confounder haben muss, um ein beobachtetes Risikoverhältnis ganz zu erklären.")
st.plotly_chart(build_evalue_chart(res2), width="stretch", key="evalue_chart")
verdict(verdict_evalue(res2))
st.dataframe(build_evalue_table(res2), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Falsifikation – wer sieht den Confounder?")
res3 = _falsify()
st.caption(f"n = {res3['n']}, {res3['reps']} feste Wiederholungen je Stärke. U wirkt auf den Faktor mit der angegebenen Stärke und auf den Servicegrad mit dem Dreifachen. Die negative Kontrolle ist ein Ergebnis, auf das der Faktor "
           f"keine Wirkung hat und das U teilt (stark: {de(C.NC_STRONG, 1)}, schwach: {de(C.NC_WEAK, 1)}). Die drei Widerlegungstests ändern Behandlung, Schätzer oder Daten, nicht die Annahme über U.")
st.plotly_chart(build_falsify_chart(res3), width="stretch", key="falsify_chart")
verdict(verdict_falsify(res3))
st.dataframe(build_falsify_table(res3), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Der Benchmark gegen die beobachtete Kovariate")
res4 = _benchmark()
st.caption("Standard-Stichprobe. „U so stark wie Z“ ist eine **Annahme**: Sie lässt sich aus den Daten nicht prüfen. Die Tabelle zeigt, was sie bedeutet.")
verdict(verdict_benchmark(res4))
st.dataframe(build_benchmark_table(res4), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Robustheitswert und E-Wert beschreiben Stärke, nicht Existenz** | Sie sagen, wie stark ein Confounder sein müsste, nicht ob einer da ist; ohne Kenntnis der Grundlinie lässt sich ein beobachteter Wert nicht dem Fall mit oder ohne U zuordnen (die Verteilungen überlappen, Experiment 1). | Wie falsch darf das Diagramm sein? |
| **Ein verborgener Confounder, lineare Regression** | Mehrere Confounder, nichtlineare Wirkungen und andere Schätzer (Instrument, Frontdoor, doppelt robust) sind nicht gebaut. | nicht gebaut |
| **Der Benchmark ist plausibel** | „U so stark wie Z“ ist eine Annahme über etwas Unbeobachtetes; die Daten prüfen sie nicht. | nicht gebaut |
| **Die negative Kontrolle teilt die Ursache** | Teilt sie U kaum, schweigt sie; wirkt der Faktor selbst auf sie, schlägt sie ohne Confounder an (Experiment 3). | nicht gebaut |
| **Widerlegungstests prüfen die Annahme** | Placebo, zufällige Ursache und Teilstichprobe prüfen die Technik; den verborgenen Confounder sehen sie nie (Experiment 3). | nicht gebaut |
| **Das Modell ist richtig** | Die partiellen R² gelten für die gerechnete Regression, bei binärer Behandlung für ein lineares Wahrscheinlichkeitsmodell. | Schätzen (Stück 6) |
| **Die Wirkung ist für alle gleich** | Bei unterschiedlichen Wirkungen gilt die Rechnung für die durchschnittliche. | Übertragbarkeit |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/sensitivitaet-verborgene-ursache-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind eine Schätzung der Wirkung aus Beobachtungsdaten und die Annahme, dass eine Ursache fehlt. Gesucht ist, wie stark diese Ursache auf Faktor und Ergebnis wirken müsste, damit die Schätzung um einen Anteil $q$ fällt.

| Symbol | Bedeutung |
|---|---|
| $D$, $Y$, $X$ | Behandlung (der hohe Faktor), Ergebnis (Servicegrad), Kontrollvariablen (in der Welt der Demo die beobachtete Schwankung) |
| $Z$ | in den Formeln der weggelassene Confounder (in der Welt der Demo die verborgene Standortqualität $U$) |
| $\hat\tau_{\mathrm{res}}$, $\hat\tau_{\mathrm{voll}}$ | Schätzung ohne und mit dem Confounder |
| $R^2$ | partielles Bestimmtheitsmaß: Anteil der Reststreuung, den eine Variable erklärt |
| $t$, $\mathrm{df}$, $\mathrm{se}$ | $t$-Wert, Freiheitsgrade und Standardfehler der Schätzung |
| $f$, $f_q$ | Cohens $f = |t|/\sqrt{\mathrm{df}}$ und $f_q = q\,f$ |
| $q$ | Anteil, um den die Schätzung fallen soll |
| $\mathrm{RV}_q$ | Robustheitswert: nötige Stärke des Confounders auf beiden Wegen |
| $\mathrm{RR}$, $E$ | Risikoverhältnis und E-Wert |
| $\mathrm{RR}_{UD}$, $\mathrm{RR}_{UY}$, $B$ | Risikoverhältnisse des Confounders zum Faktor und zum Ergebnis, Schranke für den Verzerrungsfaktor |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/sensitivitaet-verborgene-ursache-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Partielles R².** Für die Behandlung $D$ in der Regression von $Y$ auf $D$ und Kontrollen mit $t$-Wert $t$ und $\mathrm{df}$ Freiheitsgraden: $R^2_{Y\sim D\mid X} = t^2/(t^2+\mathrm{df})$; Cohens $f = |t|/\sqrt{\mathrm{df}}$.

**Verzerrung durch einen weggelassenen Confounder $Z$** mit partiellem $R^2_{D\sim Z\mid X}$ und $R^2_{Y\sim Z\mid D,X}$: $|\hat\tau_{\mathrm{res}} - \hat\tau_{\mathrm{voll}}| = \sqrt{R^2_{Y\sim Z\mid D,X}\,R^2_{D\sim Z\mid X}/(1-R^2_{D\sim Z\mid X})}\cdot \mathrm{se}(\hat\tau_{\mathrm{res}})\sqrt{\mathrm{df}}$;
angepasster Standardfehler $\mathrm{se}\sqrt{(1-R^2_{Y\sim Z})/(1-R^2_{D\sim Z})}\sqrt{\mathrm{df}/(\mathrm{df}-1)}$. Beide gelten als Rechenidentität mit bekanntem $Z$.

**Robustheitswert.** $\mathrm{RV}_q = \tfrac12\big(\sqrt{f_q^4 + 4 f_q^2} - f_q^2\big)$ mit $f_q = q\,f$: die Stärke (gleiches partielles $R^2$ mit $D$ und $Y$), bei der die Schätzung um den Anteil $q$ fällt. Mit Intervall
($\mathrm{RV}_{q,\alpha}$): $f_{q,\alpha} = f_q - f_{\mathrm{krit}}$, $f_{\mathrm{krit}} = t^*_{\alpha,\mathrm{df}-1}/\sqrt{\mathrm{df}-1}$, mit einer Fallunterscheidung bei großem $f_{q,\alpha}$.

**E-Wert.** Für ein Risikoverhältnis $\mathrm{RR}\ge 1$: $E = \mathrm{RR} + \sqrt{\mathrm{RR}\,(\mathrm{RR}-1)}$; für $\mathrm{RR}<1$ wird gespiegelt. Zugrunde liegt die Schranke $B = \mathrm{RR}_{UD}\,\mathrm{RR}_{UY}/(\mathrm{RR}_{UD}+\mathrm{RR}_{UY}-1)$ für den
Verzerrungsfaktor eines Confounders mit den Risikoverhältnissen $\mathrm{RR}_{UD}$ und $\mathrm{RR}_{UY}$; bei $\mathrm{RR}_{UD}=\mathrm{RR}_{UY}=E$ ist $B=\mathrm{RR}$. Für stetige Ergebnisse gibt es eine Umrechnung (etwa $\mathrm{RR}\approx e^{0{,}91\,d}$), die hier nicht gebaut ist.

**Negative Kontrolle.** Ein Ergebnis $N$, auf das $D$ keine Wirkung haben darf; schlägt die Regression von $N$ auf $D$ und Kontrollen an, deutet das auf einen Confounder, der $D$ und $N$ gemeinsam bewegt. Gültig nur, wenn $N$
von demselben $U$ abhängt wie $Y$ und $D$ nicht auf $N$ wirkt.

**Literatur.** C. Cinelli, C. Hazlett, *Making sense of sensitivity: extending omitted variable bias*, JRSS-B 82, 2020; T. VanderWeele, P. Ding, *Sensitivity analysis in observational research: introducing the E-value*,
Annals of Internal Medicine 167, 2017; P. Ding, T. VanderWeele, *Sensitivity analysis without assumptions*, Epidemiology 27, 2016; M. Lipsitch, E. Tchetgen Tchetgen, T. Cohen, *Negative controls*, Epidemiology 21, 2010
(alle aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html)."
)
