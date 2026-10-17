"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from sns_evaluation import Analysis, de, pct, signed, verdict_instance

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "„": '"', "“": '"', "×": "x", "·": "*", "**": "", "₁": "1", "²": "2", "≤": "<=", "≥": ">=", "α": "alpha", "τ": "tau", "√": "sqrt"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Sensitivität und Falsifikation"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"U wirkt auf x mit {de(s.gu, 2)} und auf y mit {de(s.bu, 1)}; wahre Wirkung {de(s.tau, 1)}; n = {s.n}, Seed {s.seed}; q = {de(s.q, 2)}, Benchmark k = {s.k}, Richtung {s.direction.lower()}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    for line in (f"Schätzung ohne U: {signed(a.fit.estimate, 3)} [{de(a.fit.lo, 2)}; {de(a.fit.hi, 2)}], t = {de(a.fit.t, 2)}, df = {a.fit.df}.",
                 f"Schätzung mit U (Orakel): {signed(a.full.estimate, 3)} [{de(a.full.lo, 2)}; {de(a.full.hi, 2)}].",
                 f"Robustheitswert RV = {pct(a.rv, 1)}, mit Intervall {pct(a.rv_alpha, 1)}.",
                 f"Beobachtete Kovariate Z: partielles R2 mit x {pct(a.r2dz_z, 1)}, mit y {pct(a.r2yz_z, 1)}.",
                 f"Benchmark ({s.k}-mal so stark wie Z): {signed(a.benchmark['estimate'], 2)} [{de(a.benchmark['lo'], 2)}; {de(a.benchmark['hi'], 2)}].",
                 f"Tatsächliches U: partielles R2 mit x {pct(a.r2dz_u, 1)}, mit y {pct(a.r2yz_u, 1)}."):
        pdf.multi_cell(0, 4.6, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_instance(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: sensitivitaet-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
