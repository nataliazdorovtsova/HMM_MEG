"""Render a Markdown supplement to .docx for journal submission.

    python -m analysis.make_supplement_docx supplement/supplementary_materials.md

Writes the .docx alongside the source file. Pandoc is not available on the CBU
cluster, so the route is Markdown -> HTML -> LibreOffice -> .docx.

Figures are inlined into the HTML as base64 data URIs rather than referenced by
path. LibreOffice's HTML filter drops both relative and absolute file paths
without reporting an error, so a path-based version produces a .docx that looks
complete but contains no images at all; data URIs are embedded reliably.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import markdown

# Word renders an unstyled HTML table as borderless text, which makes the
# p-value tables unreadable. These rules survive the LibreOffice import.
CSS = """
body { font-family: 'Times New Roman', serif; font-size: 11pt; line-height: 1.4; }
h1 { font-size: 16pt; } h2 { font-size: 14pt; } h3 { font-size: 12pt; }
table { border-collapse: collapse; margin: 10pt 0; }
th, td { border: 1px solid #000000; padding: 3pt 6pt; font-size: 10pt; }
th { background-color: #EEEEEE; }
img { max-width: 620px; }
code, pre { font-family: 'Courier New', monospace; font-size: 10pt; }
pre { border: 1px solid #999999; padding: 6pt; }
"""


SUFFIX_TO_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".gif": "image/gif", ".svg": "image/svg+xml"}


def _inline_images(text: str, source_dir: Path) -> tuple[str, int]:
    """Rewrite every Markdown image reference as a base64 data URI."""
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        alt, target = match.group(1), match.group(2).strip()
        if target.startswith(("http://", "https://", "data:")):
            return match.group(0)
        resolved = (source_dir / target).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"figure referenced but not found: {target} -> {resolved}")
        mime = SUFFIX_TO_MIME.get(resolved.suffix.lower())
        if mime is None:
            raise ValueError(f"unsupported image type for {resolved}")
        encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
        count += 1
        return f"![{alt}](data:{mime};base64,{encoded})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace, text), count


def convert(source: Path) -> Path:
    source = source.resolve()
    text, image_count = _inline_images(source.read_text(), source.parent)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / f"{source.stem}.html").write_text(html, encoding="utf-8")
        # LibreOffice needs a writable HOME for its user profile; the shared
        # cluster home may be read-only or already locked by another instance.
        environment = {**os.environ, "HOME": tmp}
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "docx:MS Word 2007 XML",
             str(tmp_path / f"{source.stem}.html"), "--outdir", tmp],
            check=True, capture_output=True, env=environment, timeout=300,
        )
        produced = tmp_path / f"{source.stem}.docx"
        if not produced.exists():
            raise RuntimeError("LibreOffice produced no .docx")
        # Verify the figures actually survived the conversion. LibreOffice
        # reports success whether or not it embedded them.
        with zipfile.ZipFile(produced) as archive:
            embedded = sum(1 for name in archive.namelist() if name.startswith("word/media/"))
        if embedded != image_count:
            raise RuntimeError(
                f"{image_count} figures in the source but {embedded} embedded in the .docx"
            )
        target = source.with_suffix(".docx")
        shutil.copy(produced, target)
    print(f"{image_count} figures embedded")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Markdown file to convert")
    args = parser.parse_args()
    print(f"wrote {convert(args.source)}")


if __name__ == "__main__":
    main()
