"""SpaceLight similarity-search adapter.

Builds the command-line invocation for the SpaceLight binary. The shape
matches the legacy call in ``scheduler_functions.write_search_scheduler``
(CODEBASE_REFERENCE.md §A.4 row 1, §A.5):

.. code-block:: text

    <exe> -i <query> -s <space> -o <output>
          --max-nof-results <N> --min-similarity-threshold <S>
          --thread-count <T>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpacelightAdapter:
    """Adapter that builds SpaceLight command lines."""

    exe: str

    def command_for(
        self,
        query: str,
        space: str | Path,
        output: str | Path,
        *,
        max_results: int,
        similarity: float,
        threads: int = 1,
    ) -> list[str]:
        """Return the argv list for one SpaceLight invocation.

        :param query: SMILES string to search with (passed to ``-i``).
        :param space: path to a ``.space`` file (passed to ``-s``).
        :param output: path for the result CSV (passed to ``-o``).
        :param max_results: ``--max-nof-results`` value.
        :param similarity: ``--min-similarity-threshold`` value.
        :param threads: ``--thread-count`` value (legacy default 1; 0
            means use all cores).
        """
        return [
            self.exe,
            "-i", query,
            "-s", str(space),
            "-o", str(output),
            "--max-nof-results", str(max_results),
            "--min-similarity-threshold", str(similarity),
            "--thread-count", str(threads),
        ]


__all__ = ["SpacelightAdapter"]
