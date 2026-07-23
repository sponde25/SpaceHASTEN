#!/usr/bin/env python3
# ruff: noqa: E501

import argparse
import base64
import getpass
import logging
import re
import socket
import subprocess
import tempfile
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

LOGGER = logging.getLogger(__name__)

IMAGE_PATTERN = re.compile(r"!\[[^]]*\]\(([^)]+)\)")
LOCAL_LINK_PATTERN = re.compile(
    r"(?<!!)\[([^]]+)\]\((?!https?://|mailto:|#)([^)]+)\)"
)
LOCAL_PATH_PATTERN = re.compile(r"`/(?:data|wrk|fastwrk)/[^`\n]+`")

CSS = r"""
:root {
  --ink: #18232d;
  --muted: #5f6d78;
  --navy: #102a3d;
  --blue: #176b87;
  --teal: #2a8f82;
  --gold: #d6a84b;
  --line: #dbe3e8;
  --paper: #ffffff;
  --canvas: #f2f5f7;
  --soft-blue: #edf6f8;
  --soft-gold: #fbf6ea;
}

* { box-sizing: border-box; }

html {
  scroll-behavior: smooth;
  scroll-padding-top: 1.5rem;
}

body {
  margin: 0;
  color: var(--ink);
  background: var(--canvas);
  font-family: Charter, "Bitstream Charter", Georgia, serif;
  font-size: 17px;
  line-height: 1.7;
}

.top-rule {
  height: 5px;
  background: linear-gradient(90deg, var(--gold), var(--teal), #4c91bd);
}

.hero {
  position: relative;
  overflow: hidden;
  color: #fff;
  background:
    radial-gradient(circle at 83% 22%, rgba(58, 164, 155, 0.28), transparent 31%),
    radial-gradient(circle at 70% 130%, rgba(214, 168, 75, 0.17), transparent 35%),
    var(--navy);
}

.hero::after {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
  background-size: 34px 34px;
  content: "";
  pointer-events: none;
}

.hero-inner {
  position: relative;
  z-index: 1;
  max-width: 1160px;
  margin: 0 auto;
  padding: 4.5rem 2rem 4rem;
}

.kicker,
.rail-label {
  margin: 0 0 .65rem;
  color: #75c8bf;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: .76rem;
  font-weight: 750;
  letter-spacing: .14em;
  text-transform: uppercase;
}

.hero h1 {
  max-width: 850px;
  margin: 0;
  color: #fff;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: clamp(2.5rem, 6vw, 4.8rem);
  font-weight: 760;
  letter-spacing: -.045em;
  line-height: 1.02;
}

.dek {
  max-width: 790px;
  margin: 1.35rem 0 0;
  color: #d5e2e8;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: clamp(1rem, 2.1vw, 1.25rem);
  line-height: 1.55;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: .7rem 1.4rem;
  margin-top: 1.7rem;
  color: #a9c0cc;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: .79rem;
  letter-spacing: .025em;
}

.meta span + span::before {
  margin-right: 1.4rem;
  color: var(--gold);
  content: "|";
}

.shell {
  display: grid;
  grid-template-columns: minmax(215px, 275px) minmax(0, 960px);
  gap: clamp(1.5rem, 4vw, 3.5rem);
  justify-content: center;
  max-width: 1390px;
  margin: 0 auto;
  padding: 3rem 2rem 5rem;
}

.rail-inner {
  position: sticky;
  top: 1.5rem;
  max-height: calc(100vh - 3rem);
  overflow-y: auto;
  padding: 1.2rem 1.15rem 1.35rem;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255,255,255,.8);
  box-shadow: 0 8px 24px rgba(31, 51, 63, .05);
  backdrop-filter: blur(8px);
}

.rail-label {
  color: var(--blue);
}

#TOC ul {
  margin: 0;
  padding: 0;
  list-style: none;
}

#TOC ul ul {
  margin: .35rem 0 .45rem .6rem;
  padding-left: .75rem;
  border-left: 1px solid var(--line);
}

#TOC li {
  margin: .23rem 0;
  line-height: 1.28;
}

#TOC a {
  display: block;
  padding: .2rem .25rem;
  border-radius: 5px;
  color: #50616d;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: .77rem;
  text-decoration: none;
}

#TOC a:hover,
#TOC a:focus {
  color: var(--blue);
  background: var(--soft-blue);
}

main {
  min-width: 0;
  padding: clamp(2rem, 5vw, 4.3rem);
  border: 1px solid #e1e7eb;
  border-radius: 16px;
  background: var(--paper);
  box-shadow: 0 22px 55px rgba(31, 51, 63, .09);
}

.export-note {
  margin: 0 0 3rem;
  padding: 1rem 1.15rem;
  border-left: 4px solid var(--teal);
  border-radius: 0 8px 8px 0;
  color: #40515d;
  background: var(--soft-blue);
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: .88rem;
  line-height: 1.55;
}

h2, h3, h4 {
  color: var(--navy);
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  line-height: 1.22;
  text-wrap: balance;
}

h2 {
  margin: 4.5rem 0 1.25rem;
  padding-top: .85rem;
  border-top: 3px solid var(--navy);
  font-size: clamp(1.65rem, 3.2vw, 2.25rem);
  letter-spacing: -.025em;
}

main section:first-of-type h2 { margin-top: 0; }

h3 {
  margin: 3rem 0 1rem;
  color: var(--blue);
  font-size: clamp(1.3rem, 2.6vw, 1.65rem);
  letter-spacing: -.015em;
}

h4 {
  margin: 2.25rem 0 .75rem;
  font-size: 1.12rem;
}

p { margin: 0 0 1.15rem; }

strong { color: #102f40; }

a {
  color: #096c87;
  text-decoration-thickness: .08em;
  text-underline-offset: .15em;
}

ul, ol {
  margin: .6rem 0 1.35rem;
  padding-left: 1.45rem;
}

li { margin: .35rem 0; }

code {
  padding: .1em .33em;
  border: 1px solid #dae3e8;
  border-radius: 4px;
  color: #174b60;
  background: #f3f7f9;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: .86em;
}

.table-wrap {
  max-width: 100%;
  margin: 1.4rem 0 2rem;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 9px;
}

table {
  width: 100%;
  min-width: 590px;
  border-spacing: 0;
  border-collapse: separate;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: .82rem;
  line-height: 1.45;
}

th, td {
  padding: .68rem .8rem;
  border-right: 1px solid #e5ebee;
  border-bottom: 1px solid #e5ebee;
  text-align: left;
  vertical-align: top;
}

th:last-child, td:last-child { border-right: 0; }
tr:last-child td { border-bottom: 0; }

thead th {
  color: #fff;
  background: var(--navy);
  font-size: .76rem;
  font-weight: 700;
  letter-spacing: .015em;
}

tbody tr:nth-child(even) { background: #f7f9fa; }
tbody tr:hover { background: var(--soft-gold); }

figure {
  margin: 2rem 0 2.7rem;
  padding: .8rem;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fbfcfd;
  box-shadow: 0 10px 28px rgba(31, 51, 63, .07);
}

figure img,
p > img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 7px;
  background: #fff;
}

figcaption {
  padding: .72rem .45rem .2rem;
  color: var(--muted);
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: .8rem;
  line-height: 1.45;
  text-align: center;
}

footer {
  margin-top: 4.5rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--line);
  color: #70808a;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: .76rem;
}

.back-to-top {
  position: fixed;
  right: 1.2rem;
  bottom: 1.2rem;
  z-index: 5;
  padding: .55rem .8rem;
  border: 1px solid rgba(255,255,255,.25);
  border-radius: 999px;
  color: #fff;
  background: rgba(16, 42, 61, .92);
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  font-size: .72rem;
  text-decoration: none;
  box-shadow: 0 6px 18px rgba(16, 42, 61, .2);
}

@media (max-width: 940px) {
  .shell {
    grid-template-columns: 1fr;
    padding: 1.5rem 1rem 3rem;
  }

  .rail-inner {
    position: static;
    max-height: none;
  }

  #TOC > ul {
    columns: 2;
    column-gap: 2rem;
  }

  #TOC li { break-inside: avoid; }
  main { padding: clamp(1.35rem, 5vw, 3rem); }
}

@media (max-width: 620px) {
  body { font-size: 16px; }
  .hero-inner { padding: 3.2rem 1.25rem 3rem; }
  .meta span + span::before { display: none; }
  #TOC > ul { columns: 1; }
  main { border-radius: 10px; }
  h2 { margin-top: 3.5rem; }
  .back-to-top { display: none; }
}

@media print {
  @page { size: A4; margin: 16mm; }
  body { background: #fff; font-size: 10.5pt; }
  .top-rule, .rail, .back-to-top { display: none; }
  .hero { color: #000; background: #fff; }
  .hero::after { display: none; }
  .hero-inner { max-width: none; padding: 0 0 12mm; }
  .hero h1 { color: #102a3d; font-size: 28pt; }
  .dek, .meta { color: #40515d; }
  .shell { display: block; max-width: none; padding: 0; }
  main { padding: 0; border: 0; border-radius: 0; box-shadow: none; }
  h2, h3, h4 { break-after: avoid; }
  figure, .table-wrap { break-inside: avoid; box-shadow: none; }
  a { color: inherit; text-decoration: none; }
}
"""

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>$title$</title>
  <style>
$css$
  </style>
</head>
<body id="top">
  <div class="top-rule"></div>
  <header class="hero">
    <div class="hero-inner">
      <p class="kicker">Virtual screening study</p>
      <h1>$title$</h1>
      <p class="dek">Hit quality, chemical-space diversity, acquisition-mechanism attribution, and persistent-atlas runtime impact.</p>
      <div class="meta">
        <span>Standalone collaborator edition</span>
        <span>Generated $date$</span>
      </div>
    </div>
  </header>
  <div class="shell">
    <aside class="rail" aria-label="Report navigation">
      <div class="rail-inner">
        <p class="rail-label">Contents</p>
        <nav id="TOC" role="doc-toc">
$toc$
        </nav>
      </div>
    </aside>
    <main id="report">
      <div class="export-note"><strong>About this edition:</strong> every report figure is embedded in this HTML file. References to supplemental machine-readable artifacts are retained as non-clickable path labels and are not included in the standalone export.</div>
$body$
      <footer>Self-contained HTML export. No external stylesheets, scripts, fonts, or image files are required.</footer>
    </main>
  </div>
  <a class="back-to-top" href="#top" aria-label="Back to top">Back to top</a>
</body>
</html>
"""


class DocumentAssets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.image_sources: list[str] = []
        self.links: list[str] = []
        self.external_assets: list[str] = []
        self.figure_count = 0
        self.element_ids: set[str] = set()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.element_ids.add(attributes["id"])
        if tag == "figure":
            self.figure_count += 1
        if tag == "img" and attributes.get("src"):
            self.image_sources.append(attributes["src"])
        elif tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])
        elif tag in {"script", "link"}:
            source = attributes.get("src") or attributes.get("href")
            if source:
                self.external_assets.append(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Markdown as a validated standalone HTML report."
    )
    parser.add_argument("input", type=Path, help="Source Markdown report")
    parser.add_argument("output", type=Path, help="Destination HTML file")
    return parser.parse_args()


def prepare_markdown(source: str, fallback_title: str) -> tuple[str, str]:
    title_match = re.search(r"^#\s+(.+)$", source, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else fallback_title
    if title_match:
        source = source[: title_match.start()] + source[title_match.end() :]

    source = LOCAL_PATH_PATTERN.sub(
        "`local working directory (not included in this export)`", source
    )
    source = LOCAL_LINK_PATTERN.sub(r"\1", source)
    return source.lstrip(), title


def wrap_tables(html: str) -> str:
    html = html.replace("<table", '<div class="table-wrap"><table')
    return html.replace("</table>", "</table></div>")


def wrap_figures(html: str) -> str:
    image_paragraph = re.compile(
        r'<p><img src="([^"]+)" alt="([^"]*)" /></p>'
    )

    def replace(match: re.Match[str]) -> str:
        source, caption = match.groups()
        return (
            f'<figure><img src="{source}" alt="{caption}" loading="lazy" '
            f'decoding="async"><figcaption>{caption}</figcaption></figure>'
        )

    return image_paragraph.sub(replace, html)


def render_report(source_path: Path, output_path: Path) -> list[Path]:
    source = source_path.read_text(encoding="utf-8")
    image_paths = [source_path.parent / path for path in IMAGE_PATTERN.findall(source)]
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        missing = "\n".join(str(path) for path in missing_images)
        raise FileNotFoundError(f"Missing report images:\n{missing}")

    prepared, title = prepare_markdown(source, source_path.stem)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="standalone-report-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        prepared_path = temp_dir / "report.md"
        template_path = temp_dir / "template.html"
        rendered_path = temp_dir / "report.html"
        prepared_path.write_text(prepared, encoding="utf-8")
        template_path.write_text(TEMPLATE, encoding="utf-8")

        command = [
            "pandoc",
            str(prepared_path),
            "--from=gfm+implicit_figures",
            "--to=html5",
            "--standalone",
            "--embed-resources",
            "--toc",
            "--toc-depth=3",
            "--section-divs",
            f"--template={template_path}",
            f"--resource-path={source_path.parent}",
            "--metadata",
            f"title={title}",
            "--metadata",
            f"date={date.today().strftime('%-d %B %Y')}",
            "--variable",
            f"css={CSS}",
            "--output",
            str(rendered_path),
        ]
        subprocess.run(command, check=True)
        html = wrap_tables(rendered_path.read_text(encoding="utf-8"))
        html = wrap_figures(html)
        output_path.write_text(html, encoding="utf-8")

    return image_paths


def validate_report(output_path: Path, source_images: list[Path]) -> None:
    html = output_path.read_text(encoding="utf-8")
    assets = DocumentAssets()
    assets.feed(html)

    if len(assets.image_sources) != len(source_images):
        raise ValueError(
            f"Expected {len(source_images)} embedded images, "
            f"found {len(assets.image_sources)}"
        )
    if assets.figure_count != len(source_images):
        raise ValueError(
            f"Expected {len(source_images)} semantic figures, "
            f"found {assets.figure_count}"
        )

    for index, (source_path, source) in enumerate(
        zip(source_images, assets.image_sources, strict=True), start=1
    ):
        prefix = "data:image/png;base64,"
        if not source.startswith(prefix):
            raise ValueError(f"Image {index} is not an embedded PNG")
        decoded = base64.b64decode(source.removeprefix(prefix), validate=True)
        if decoded != source_path.read_bytes():
            raise ValueError(f"Embedded image {index} differs from {source_path.name}")

    local_links = [
        link
        for link in assets.links
        if not link.startswith(("#", "http://", "https://", "mailto:"))
    ]
    if local_links:
        raise ValueError(f"Standalone report contains local links: {local_links}")
    missing_targets = [
        link
        for link in assets.links
        if link.startswith("#") and link[1:] not in assets.element_ids
    ]
    if missing_targets:
        raise ValueError(f"Internal links have no target: {missing_targets}")
    if assets.external_assets:
        raise ValueError(
            f"Standalone report contains external assets: {assets.external_assets}"
        )

    text_without_images = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "", html
    )
    sensitive_tokens = [
        token
        for token in (
            "/data/",
            "/wrk/",
            "/fastwrk/",
            getpass.getuser(),
            socket.gethostname(),
        )
        if token in text_without_images
    ]
    if sensitive_tokens:
        raise ValueError(f"Sensitive local identifiers remain: {sensitive_tokens}")

    LOGGER.info(
        "Validated %d embedded figures and %d internal navigation links",
        len(assets.image_sources),
        len(assets.links),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    source_images = render_report(source_path, output_path)
    validate_report(output_path, source_images)
    LOGGER.info(
        "Wrote %s (%.1f MiB)",
        output_path,
        output_path.stat().st_size / (1024 * 1024),
    )


if __name__ == "__main__":
    main()
