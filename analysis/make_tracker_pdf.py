"""Render the reviewer-response tracker as a shareable PDF, with the
supplement figures appended.

    python -m analysis.make_tracker_pdf \
        --tracker supplement/reviewer_response_tracker.md \
        --figures-dir supplement/figures \
        --output supplement/reviewer_response_tracker.pdf

Status emoji in the source markdown are replaced with plain text labels
(DONE / PARTLY / NOT DONE / TEXT ONLY) so the PDF prints cleanly and stays
readable if a font lacks emoji glyphs.

Rendering goes markdown -> styled HTML -> Chrome headless -> PDF. Chrome is
used rather than LaTeX because the tracker's main content is a wide
three-column table with long prose cells, which CSS handles far better than
LaTeX's longtable; landscape orientation is set via an @page rule.
"""

from __future__ import annotations

import argparse
import base64
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

EMOJI_REPLACEMENTS = {
    "✅": "DONE",           # white heavy check mark
    "⚠️": "PARTLY",   # warning sign (with variation selector)
    "⚠": "PARTLY",         # warning sign (bare)
    "❌": "NOT DONE",       # cross mark
    "\U0001f4dd": "TEXT ONLY",  # memo
}

FIGURE_CAPTIONS = {
    "state_metric_distributions.png": (
        "Per-state distributions of fractional occupancy, mean lifetime and mean interval, "
        "with individual subject points overlaid (Reviewer 3, concern 2b). Interval uses a "
        "log axis because state 1's range would otherwise compress the other states."
    ),
    "cognition_fo_scatters.png": (
        "Fractional occupancy against WASI-II matrix reasoning T-score for the four states "
        "of interest. Note the direction: state 3 is negative (higher cognitive ability, less "
        "occupancy) while states 4, 6 and 7 are positive."
    ),
    "transition_cognition_heatmap.png": (
        "Cell-wise correlations between each transition probability and cognitive ability. "
        "Descriptive only: NONE of these 49 cells survives multiplicity correction (smallest "
        "adjusted p = 0.063). The significant transition result is at the level of entering a "
        "state - states 3, 4 and 6, p_adj = 0.041 - and is reported in the results table."
    ),
    "transition_matrix_heatmap.png": (
        "Mean state transition matrix across subjects, self-transitions masked. Self-transition "
        "probability is roughly 0.99 for every state at this sampling rate, so leaving the "
        "diagonal in place renders all off-diagonal structure invisible."
    ),
    "cognitive_behavioural_correlations.png": (
        "Correlations among age, cognitive and behavioural measures."
    ),
    "residual_qq_comparison.png": (
        "Residual QQ plots before and after the rank-based inverse-normal transform, for "
        "fractional occupancy and entropy rate. Raw fractional occupancy residuals are "
        "severely non-normal, which is what motivated using permutation inference for all "
        "headline tests."
    ),
}

FIGURE_ORDER = [
    "state_metric_distributions.png",
    "cognition_fo_scatters.png",
    "transition_cognition_heatmap.png",
    "transition_matrix_heatmap.png",
    "cognitive_behavioural_correlations.png",
    "residual_qq_comparison.png",
]

CSS = """
@page { size: A4 landscape; margin: 14mm 12mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
       font-size: 8.6pt; line-height: 1.45; color: #111; }
h1 { font-size: 16pt; margin: 0 0 4pt 0; }
h2 { font-size: 12pt; margin: 16pt 0 6pt 0; padding-top: 6pt;
     border-top: 1.5px solid #444; page-break-after: avoid; }
h3 { font-size: 10pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }
p  { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt 0;
        table-layout: fixed; page-break-inside: auto; }
th { background: #ececea; text-align: left; font-weight: 600;
     border: 0.6px solid #999; padding: 4pt 5pt; font-size: 8.4pt; }
td { border: 0.6px solid #bbb; padding: 4pt 5pt; vertical-align: top;
     word-wrap: break-word; overflow-wrap: break-word; }
tr { page-break-inside: avoid; }
table th:nth-child(1), table td:nth-child(1) { width: 26%; }
table th:nth-child(2), table td:nth-child(2) { width: 37%; }
table th:nth-child(3), table td:nth-child(3) { width: 37%; }
table.two-col th:nth-child(1), table.two-col td:nth-child(1) { width: 34%; }
table.two-col th:nth-child(2), table.two-col td:nth-child(2) { width: 66%; }
/* 4+ columns (the results tables): let the browser size them evenly rather
   than inheriting the 26/37/37 split meant for the three-column tracker */
table.multi-col { table-layout: auto; }
table.multi-col th, table.multi-col td { width: auto; white-space: nowrap; }
table.multi-col td:nth-child(2) { white-space: normal; }
code { background: #f2f2f0; padding: 0 2px; font-size: 8pt; }
hr { border: none; border-top: 1px solid #ccc; margin: 10pt 0; }
.figure { page-break-inside: avoid; margin: 0 0 16pt 0; text-align: center; }
.figure img { max-width: 100%; max-height: 155mm; }
.caption { font-size: 8.2pt; color: #444; margin-top: 4pt;
           text-align: left; }
.figures-page { page-break-before: always; }
"""


MOBILE_CSS = """
@page { size: A4 portrait; margin: 10mm 8mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.5; color: #111; }
h1 { font-size: 17pt; margin: 0 0 6pt 0; }
h2 { font-size: 13pt; margin: 16pt 0 6pt 0; padding-top: 6pt;
     border-top: 1.5px solid #444; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }
p, li { margin: 5pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt 0;
        table-layout: auto; font-size: 9pt; }
th { background: #ececea; text-align: left; font-weight: 600;
     border: 0.6px solid #999; padding: 3pt 4pt; }
td { border: 0.6px solid #bbb; padding: 3pt 4pt; vertical-align: top;
     word-wrap: break-word; overflow-wrap: break-word; }
tr { page-break-inside: avoid; }
code { background: #f2f2f0; padding: 0 2px; font-size: 9pt; }
hr { border: none; border-top: 1px solid #ccc; margin: 10pt 0; }
.figure { page-break-inside: avoid; margin: 0 0 14pt 0; text-align: center; }
.figure img { max-width: 100%; max-height: 110mm; }
.caption { font-size: 9pt; color: #444; margin-top: 4pt; text-align: left; }
.figures-page { page-break-before: always; }
"""


def strip_emoji(text: str) -> str:
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, replacement)
    # any stray pictographs the table above didn't cover
    return re.sub(r"[\U0001F300-\U0001FAFF☀-➿️]", "", text)


def _embed(path: Path, max_width: int | None = None) -> str:
    """Inline an image as a data URI so the HTML is self-contained.

    `max_width` downsamples first, which is what keeps the mobile build small
    enough to open comfortably on a phone - the full-resolution figures are
    several hundred KB each and only matter in print.
    """
    data = path.read_bytes()
    if max_width:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(data))
        if img.width > max_width:
            img = img.resize((max_width, round(img.height * max_width / img.width)),
                             Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def build_html(tracker_md: str, figures_dir: Path, mobile: bool = False) -> str:
    body = markdown.markdown(strip_emoji(tracker_md), extensions=["tables", "sane_lists"])

    # Column widths are set per table by column count: the fixed 26/37/37
    # split suits the three-column tracker but would push a six-column
    # results table off the page.
    parts = body.split("<table>")
    rebuilt = parts[0]
    for chunk in parts[1:]:
        header_count = chunk.split("</thead>")[0].count("<th>")
        if header_count == 2:
            cls = ' class="two-col"'
        elif header_count >= 4:
            cls = ' class="multi-col"'
        else:
            cls = ""
        rebuilt += f"<table{cls}>" + chunk
    body = rebuilt

    figure_html = ['<div class="figures-page"><h2>Figures</h2>']
    for name in FIGURE_ORDER:
        path = figures_dir / name
        if not path.exists():
            continue
        figure_html.append(
            f'<div class="figure"><img src="{_embed(path, 1100 if mobile else None)}" />'
            f'<div class="caption"><b>{name}</b> &mdash; {FIGURE_CAPTIONS.get(name, "")}</div></div>'
        )
    figure_html.append("</div>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{MOBILE_CSS if mobile else CSS}</style></head><body>{body}{''.join(figure_html)}</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", default="supplement/reviewer_response_tracker.md")
    parser.add_argument("--figures-dir", default="supplement/figures")
    parser.add_argument("--output", default="supplement/reviewer_response_tracker.pdf")
    parser.add_argument("--mobile", action="store_true",
                        help="Portrait A4, larger type, downsampled figures - for reading on a phone")
    args = parser.parse_args()

    html = build_html(Path(args.tracker).read_text(), Path(args.figures_dir), mobile=args.mobile)
    output = Path(args.output).resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "tracker.html"
        html_path.write_text(html)
        subprocess.run(
            [
                "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
                f"--user-data-dir={tmpdir}/chrome",
                "--no-pdf-header-footer",
                f"--print-to-pdf={output}",
                f"file://{html_path}",
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
