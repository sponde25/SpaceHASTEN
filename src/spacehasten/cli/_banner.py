"""ASCII banner for the SpaceHASTEN CLI.

The artwork combines a figlet-style "SpaceHASTEN" wordmark with a
funnel motif, evoking the program's purpose: hastening through
*nonenumerated* chemical spaces by funnelling them via similarity
search and docking down to a tractable set of hits.
"""

from __future__ import annotations

import sys

from spacehasten import __version__

# Wordmark in figlet "small" font; raw string so the backslashes survive.
_WORDMARK = r"""
 ___                  _  _   _   ___ _____ ___ _  _
/ __|_ __  __ _ __ ___| || | /_\ / __|_   _| __| \| |
\__ \ '_ \/ _` / _/ -_) __ |/ _ \\__ \ | |  | _|| .` |
|___/ .__/\__,_\__\___|_||_/_/ \_\___/ |_| |___|_|\_|
    |_|
"""

# Funnel: a billion-compound space narrows through search + docking
# to a handful of hits.
_FUNNEL = r"""
        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         ░░░░░░░░░░░░  10⁹+ molecules  ░░░░░░░░░░░░
          ▒▒▒▒▒▒▒▒▒  similarity search  ▒▒▒▒▒▒▒▒▒
            ▓▓▓▓▓▓▓  property filter  ▓▓▓▓▓▓▓
              ▓▓▓▓▓  chemprop predict  ▓▓▓▓▓
                ███  Glide docking  ███
                   █  hits  █
                    ▀▀
"""

_TAGLINE = (
    "Iterative docking-based exploration of nonenumerated chemical libraries"
)
_CREDIT = (
    "Originally written by Tuomo Kalliokoski (Orion Pharma) · "
    "rewrite continues in his footsteps."
)


def banner() -> str:
    """Return the full multi-line banner string (no trailing newline)."""
    lines: list[str] = []
    lines.extend(_WORDMARK.splitlines()[1:])  # drop the leading blank
    lines.extend(_FUNNEL.splitlines()[1:])
    lines.append(f"  {_TAGLINE}")
    lines.append(f"  {_CREDIT}")
    lines.append(f"  v{__version__}")
    return "\n".join(lines)


def print_banner(stream=sys.stderr) -> None:  # type: ignore[no-untyped-def]
    """Print the banner to ``stream`` (default ``sys.stderr``)."""
    stream.write(banner())
    stream.write("\n\n")
    stream.flush()


__all__ = ["banner", "print_banner"]
