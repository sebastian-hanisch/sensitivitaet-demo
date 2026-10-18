# Sensitivität und Falsifikation: Wie stark müsste die verborgene Ursache sein? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-sensitivitaet-demo.streamlit.app/)**

Elftes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [identifizierbarkeit-demo](https://github.com/sebastian-hanisch/identifizierbarkeit-demo): Dort stand, wann eine Wirkung aus Beobachtungsdaten nicht bestimmbar ist. Hier, was man dann noch sagen kann: **wie stark** eine verborgene Ursache sein müsste, um die Schlussfolgerung zu kippen, und was sich an den Annahmen **abklopfen** lässt.

## Kernfrage

Ein Disponent wählt den hohen Faktor nach der beobachteten Schwankung Z und nach einer verborgenen Standortqualität U. Die Regression mit Z allein liegt daneben. Der Robustheitswert und der E-Wert sagen, wie stark U sein müsste, um die Schätzung zu kippen; die negative Kontrolle prüft die Annahme fehlender Konfundierung, die Standard-Widerlegungstests prüfen nur die Technik. Was davon sieht den Confounder, und was beruhigt nur?

## Modell

- **Vehikel:** die Welt aus Stück 1 mit einem zusätzlichen verborgenen Confounder: X = 1, wenn Z + gu·U + η > 0 (hoher Faktor), Y = 80 − 4 Z − bu·U + τ X + 5 ε, wahre Wirkung **τ = 3,5**. Die negative Kontrolle N = c·U + 0,5 Z + 5 ν hängt von U ab, nicht von X.
- **Regler:** Stärke von U auf den Faktor (gu) und auf den Servicegrad (bu), wahre Wirkung, n, Seed; dazu die Annahmen der Sensitivität: q (um welchen Anteil die Schätzung fällt), Benchmark k („U ist k-mal so stark wie Z“) und die Richtung der Verzerrung.
- **Daten:** `SplitMix64`-Ströme (Generator `kausal_generator.py`, wortgleich in allen Stücken der Linie).

## Methodik

- **Schätzer:** Regression des Servicegrads auf den Faktor und Z (U fehlt); als Orakel dieselbe Regression mit U (nur weil simuliert).
- **Robustheitswert** (`sns_sensitivity.py`): RV_q = ½(√(f_q⁴ + 4 f_q²) − f_q²) mit f_q = q |t| / √df und die Variante mit Intervall RV_q,α; **Verzerrung** √(R²_Y R²_X / (1 − R²_X)) · se · √df und der angepasste Standardfehler; **Konturkarte** der angepassten Schätzung über den partiellen R² des Confounders; **Benchmark** gegen Z.
- **E-Wert** RR + √(RR (RR − 1)) mit der Schranke von Ding/VanderWeele für binäre Szenarien (exakte Rechnung, keine Stichprobe).
- **Falsifikation** (`sns_falsify.py`): negative Kontrolle, Placebo-Behandlung, zufällige gemeinsame Ursache, Teilstichprobe; je ein Alarm je Datensatz.
- **Experimente** laufen über feste Zufallsströme; die E-Wert-Tabelle ist exakt.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Wahrheit τ = 3,5; U wirkt in der Standardwelt mit gu = 0,5 und bu = 1,5.

| Frage | Befund | Test |
|---|---|---|
| Standard-Stichprobe (n = 400, Seed 35) | Ohne U **2,60** (Standardfehler 0,582, t = 4,46, Intervall 1,45 bis 3,74: deckt), mit U (Orakel) **3,62**. Robustheitswert **RV₁ = 20,0 %** (mit Intervall 11,7 %). U erklärt tatsächlich 9,1 % von x und 7,9 % von y; Z erklärt 26,7 % und 27,8 %. | `test_default_sample` |
| Gelten die Formeln? (U bekannt, 300 Läufe) | Die Verzerrung durch U und der Standardfehler mit U folgen den Formeln auf **4·10⁻¹³** bzw. **1·10⁻¹⁵**: Rechenidentitäten, keine Näherungen. Der Robustheitswert RV₁,α in geschlossener Form weicht von der Nullstellensuche um höchstens 5·10⁻¹³ ab. | `test_experiment_1_identity_and_protection` |
| Zeigt der Robustheitswert, ob U da ist? (400 Läufe je Fall) | **Nein**, jedenfalls nicht ohne die Grundlinie zu kennen. Ohne U ist RV₁ im Mittel **0,248** (Streuung 0,038), mit U **0,178** (0,043); mit Intervall 0,171 gegen 0,093. Die Verteilungen überlappen und sind gegeneinander verschoben; die Grundlinie ohne U ist in der Praxis unbekannt. | `test_experiment_1_identity_and_protection` |
| Schützt der Robustheitswert? | In **191** Läufen mit wahrer Stärke beider Wege unter RV₁ (100 bei U 0,25 / 1,0; 91 bei 0,5 / 1,5; bei stärkerem U keiner) bleibt die Schätzung mit U in **191** über null. Das ist eine Identität, kein Zufallsbefund: Ist U auf beiden Wegen schwächer als RV₁, kann es die Schätzung nicht auf null bringen. | `test_experiment_1_identity_and_protection` |
| E-Wert (binär, wahre Wirkung null, exakt) | In **378** Szenarien liegt das beobachtete Risikoverhältnis nie über der Schranke RR_UD · RR_UY / (RR_UD + RR_UY − 1) und erreicht sie. E-Wert zu RR 1,2 / 1,5 / 2 / 3 / 5: **1,69 / 2,37 / 3,41 / 5,45 / 9,47**; bei Stärke gleich dem E-Wert ist die Schranke genau das RR. | `test_experiment_2_evalue` |
| Wer sieht den Confounder? (n = 500, 500 Läufe je Stärke gu = 0 / 0,25 / 0,5 / 1 / 1,5; bu = 3 gu) | Fehler der Schätzung 0,02 / −0,30 / −1,12 / −3,52 / −6,14. **Negative Kontrolle** (U mit 1,5 geteilt) schlägt in **5 / 18 / 52 / 95 / 98 %** an, **schwach geteilt** (0,5) in 4 / 8 / 10 / 21 / 31 %. **Placebo-Behandlung** 5 bis 7 %, **zufällige gemeinsame Ursache** 0 %, **Teilstichprobe** 5 bis 6 %: Sie sehen den Confounder nie. | `test_experiment_3_falsification` |
| Eine ungültige negative Kontrolle? | Wirkt der Faktor selbst auf N (1,0), schlägt sie **ohne** Confounder in **45 %** der Läufe an. | `test_experiment_3_falsification` |
| Benchmark gegen Z (Standard-Stichprobe) | U **so stark wie Z** bringt die Schätzung (2,60) auf **−1,10** (Abschwächung) bis **+6,29** (Verstärkung), 2-mal so stark auf −6,69 bis +11,88, 3-mal auf −18,76 bis +23,95. Das tatsächliche U ist im Mittel nur 0,40-mal so stark wie Z; mit dem wahren Vielfachen trägt die Schranke in 300 von 300 Läufen (Identität). Konvention der Demo: U erklärt das k-fache partielle R² von Z; nach Cinelli/Hazlett wären die Intervalle breiter. | `test_experiment_4_benchmark` |

Handwerte des Textes (`test_hand_values_of_the_text`): f = 1 gibt RV = 0,618; RR 2 / 3 / 4 geben den E-Wert 3,414 / 5,449 / 7,464; bei beiden partiellen R² gleich 0,5, se = 1, df = 100 ist die Verzerrung 7,07.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Robustheitswert und E-Wert beschreiben Stärke, nicht Existenz** | Sie sagen, wie stark ein Confounder sein müsste, nicht ob einer da ist; ohne Kenntnis der Grundlinie lässt sich ein beobachteter Wert nicht dem Fall mit oder ohne U zuordnen (die Verteilungen überlappen, Experiment 1). | Wie falsch darf das Diagramm sein? |
| **Ein verborgener Confounder, lineare Regression** | Mehrere Confounder, nichtlineare Wirkungen und andere Schätzer (Instrument, Frontdoor, doppelt robust) sind nicht gebaut. | nicht gebaut |
| **Der Benchmark ist plausibel** | „U so stark wie Z“ ist eine Annahme über etwas Unbeobachtetes; die Daten prüfen sie nicht. | nicht gebaut |
| **Die negative Kontrolle teilt die Ursache** | Teilt sie U kaum, schweigt sie; wirkt der Faktor selbst auf sie, schlägt sie ohne Confounder an (Experiment 3). | nicht gebaut |
| **Widerlegungstests prüfen die Annahme** | Placebo, zufällige Ursache und Teilstichprobe prüfen die Technik; den verborgenen Confounder sehen sie nie (Experiment 3). | nicht gebaut |
| **Das Modell ist richtig** | Die partiellen R² gelten für die gerechnete Regression, bei binärer Behandlung für ein lineares Wahrscheinlichkeitsmodell. | Schätzen (Stück 6) |
| **Die Wirkung ist für alle gleich** | Bei unterschiedlichen Wirkungen gilt die Rechnung für die durchschnittliche. | Übertragbarkeit |

Die Aussage „Placebo, zufällige Ursache und Teilstichprobe schlagen nie an“ ist **Konstruktion**: Diese Tests ändern x, den Schätzer oder die Daten, aber sie fragen U nirgends. Als Befund steht nur die Gegenseite, die negative Kontrolle, und sie hängt an der Stärke, mit der N die Ursache teilt (die Zeile „schwach geteilt“ zeigt die Richtung, nicht die Zahl). Die Formeln für Robustheitswert, Verzerrung und angepassten Standardfehler sind aus dem Gedächtnis geschrieben und an zwei unabhängigen Wegen geprüft (die Rechenidentität mit bekanntem U und die Nullstellensuche der Definition); **eine Fallunterscheidung für große t-Werte, an die ich mich aus der Software erinnerte, wich um bis zu 5·10⁻⁴ von der Nullstellensuche ab und ist nicht übernommen**: bei gleicher Stärke beider Wege ist der angepasste Standardfehler konstant, die geschlossene Form gilt für jedes t. Die Quellen im 📐-Expander sind aus dem Gedächtnis, nicht geprüft.

## Bezug zu den Nachbarn im Portfolio

- [identifizierbarkeit-demo](https://github.com/sebastian-hanisch/identifizierbarkeit-demo): „Nicht bestimmbar heißt nicht wertlos“; dort nur die Schranken ohne Annahme (Breite 1). Hier macht eine Annahme über die Stärke aus der Breite eine Zahl.
- [konfundierung-demo](https://github.com/sebastian-hanisch/konfundierung-demo): dieselbe Welt mit der beobachteten Schwankung Z; hier kommt ein zweiter, verborgener Confounder hinzu.
- [schaetzen-demo](https://github.com/sebastian-hanisch/schaetzen-demo), [doppelt-robust-demo](https://github.com/sebastian-hanisch/doppelt-robust-demo), [frontdoor-demo](https://github.com/sebastian-hanisch/frontdoor-demo), [instrumente-demo](https://github.com/sebastian-hanisch/instrumente-demo): alle verweisen für ihre unbeobachtete Ursache auf dieses Stück; gebaut ist hier nur die lineare Regression.
- [d-separation-demo](https://github.com/sebastian-hanisch/d-separation-demo): „Alle Größen sind beobachtet“ ist dort die Grenze; hier das Maß für ihre Verletzung.

## Tests

Hand-Instanzen (f = 1: RV 0,618; E-Wert 3,414 / 5,449 / 7,464; Verzerrung 7,07 bei R² 0,5), die Rechenidentität mit bekanntem U auf 10⁻⁹, der Robustheitswert in geschlossener Form gegen die Nullstellensuche für sechs Paare (t, df) und vier q, die Schutzaussage, die exakte Schranke der binären Szenarien, die Konturkarte (Ränder, Monotonie), die vier Alarme an exakten und simulierten Daten, Auswertung, Verdikte, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Richtung, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `sns_world.py` | Vehikel: Welt mit verborgenem U, negative Kontrolle |
| `sns_sensitivity.py` | partielle R², Robustheitswert, Verzerrung, Konturkarte, Benchmark, E-Wert, Schranke |
| `sns_falsify.py` | negative Kontrolle, Placebo, zufällige Ursache, Teilstichprobe |
| `sns_evaluation.py` | Einstellungen, Analyse, vier Experimente, Verdikte |
| `sns_visualization.py`, `sns_pdf_export.py` | Figuren, Tabellen, PDF |
| `sns_constants.py`, `sns_presets.py` | Regler, Experimente, Presets, Permalink |
| `tests/` | Hand-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **E-Wert für stetige Ergebnisse** (Umrechnung über ein Risikoverhältnis aus der standardisierten Differenz): eine Näherung; hier nur binäre Szenarien exakt, stetige über den Robustheitswert.
- **Sensitivität für Instrument, Frontdoor und doppelt robuste Schätzer** sowie mehrere Confounder: eigenes Thema, hier nur die lineare Regression.
- **Bounds mit Annahmen** (Balke und Pearl) und eine Wahrscheinlichkeit für Konfundierung: Der Robustheitswert beschreibt Stärke, nicht Existenz.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, scipy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html).
