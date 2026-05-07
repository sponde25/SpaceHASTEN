"""FTrees similarity-search adapter.

Builds the command-line invocation for the FTrees binary. The shape
matches the legacy call in ``scheduler_functions.write_search_scheduler``
(CODEBASE_REFERENCE.md §A.4 row 1, §A.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FTreesAdapter:
    """Adapter that builds FTrees command lines."""

    exe: str

    def command_for(
        self,
        query: str,
        space: str | Path,
        output: str | Path,
        *,
        max_results: int,
        similarity: float,
        threads: int = 2,
    ) -> list[str]:
        """Return the argv list for one FTrees invocation."""
        return [
            self.exe,
            "-i", query,
            "-s", str(space),
            "-o", str(output),
            "--max-nof-results", str(max_results),
            "--min-similarity-threshold", str(similarity),
            "--thread-count", str(threads),
        ]


__all__ = ["FTreesAdapter"]
