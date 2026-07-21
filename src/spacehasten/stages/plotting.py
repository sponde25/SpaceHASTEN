"""Plotting stage — diagnostic score-distribution and accuracy plots.

Three independent plots, all driven directly by SQL aggregates over the
``data`` table (no intermediate CSVs, unlike the AlphaFold-oriented KDE
plotting this module took inspiration from):

1. :func:`plot_dock_score_distribution` — KDE of ``dock_score`` for the
   seed batch (``dock_iteration == 0``, grey) overlaid with each later
   screening-cycle round (``dock_iteration >= 1``, Blues gradient). A
   vertical line/legend entry marks the worst ``dock_score`` among
   simsearch-cycle-1 queries (:meth:`Database.query_cycle1_score_cutoff`)
   — the acquisition threshold that defined "hit" for the very first
   round of similarity searching against the initial random seed batch.
   Docking scores are *lower is better*, so the region left of the
   cutoff is shaded as the "good" side.

2. :func:`plot_pred_score_distribution` — KDE of chemprop ``pred_score``
   for each simsearch cycle's freshly-acquired candidates (Blues
   gradient), with the seed batch's *actual* ``dock_score`` distribution
   drawn in the background (grey) as a reference for how far predicted
   scores have drifted from the ground truth the model was trained on.

3. :func:`plot_pred_vs_dock_accuracy` — hexbin density of predicted vs.
   actual ``dock_score`` for one or more dock iterations (default: 1 and
   2), since a plain scatter of ~1e6 points would be dominated by
   overplotting. Each panel is annotated with Pearson r, Spearman rho,
   and RMSE, plus a y=x reference line.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no DISPLAY on compute/head nodes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from spacehasten.core.db import Database
from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)


def _blues_palette(n: int) -> list:
    """``n`` colors from the Blues palette, skipping the near-white first
    entry (mirrors the AlphaFold reference module's ``[1:]`` trick so
    the lightest curve stays visible against a white background)."""
    n = max(n, 1)
    return sns.color_palette("Blues", n_colors=n + 1)[1:]


def _xlim_with_buffer(values: list[float], buffer_frac: float = 0.2) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        span = abs(lo) if lo else 1.0
    return lo - span * buffer_frac, hi + span * buffer_frac


def plot_dock_score_distribution(
    db: Database,
    workdir: WorkDir,
    *,
    bw_adjust: float = 2.0,
    max_dock_score: float | None = 0.0,
    output: Path | None = None,
) -> Path:
    """Plot seed vs. iteration ``dock_score`` KDEs with the cycle-1 query
    cutoff annotated.

    :param max_dock_score: cap the x-axis maximum at this value (default
        ``0.0``). A handful of compounds can dock with a positive
        (unfavourable) score; by default that long right-hand tail is
        clipped from view since it is not of interest. Pass ``None`` to
        show the full score range instead.
    :raises ValueError: if the seed batch (``dock_iteration == 0``) has
        no docked compounds yet (i.e. ``seed-training`` has not run).
    """
    by_iteration = db.dock_scores_by_iteration()
    if 0 not in by_iteration:
        raise ValueError(
            "no seed docking scores (dock_iteration == 0) found;"
            " run `spacehasten seed-training` first"
        )
    seed_scores = by_iteration[0]
    iterations = sorted(k for k in by_iteration if k > 0)
    cutoff = db.query_cycle1_score_cutoff()

    fig, ax = plt.subplots(figsize=(9, 6))
    all_scores = list(seed_scores)
    sns.kdeplot(seed_scores, color="grey", label="Seed (iteration 0)",
                linewidth=2, ax=ax, bw_adjust=bw_adjust)

    palette = _blues_palette(len(iterations))
    for i, iteration in enumerate(iterations):
        scores = by_iteration[iteration]
        all_scores.extend(scores)
        sns.kdeplot(scores, color=palette[i % len(palette)],
                    label=f"Iteration {iteration}", linewidth=2, ax=ax,
                    bw_adjust=bw_adjust)

    xlim_min, xlim_max = _xlim_with_buffer(all_scores)
    if max_dock_score is not None:
        xlim_max = min(xlim_max, max_dock_score)
    ax.set_xlim(xlim_min, xlim_max)

    if cutoff is not None:
        # Lower dock_score is better: shade the acquired ("good") side.
        ax.axvspan(xlim_min, cutoff, color="lightblue", alpha=0.3)
        ax.axvspan(cutoff, xlim_max, color="lightgrey", alpha=0.3)
        ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.5,
                   label=f"Cycle-1 query cutoff: {cutoff:.2f}")
    else:
        logger.warning(
            "no simsearch cycle 1 found; skipping query-cutoff annotation"
        )

    ax.set_xlabel("Docking score (kcal/mol)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Docking score distribution: seed vs. screening iterations")
    ax.legend(fontsize=10)

    output = Path(output) if output is not None else workdir.plots_dir() / "dock_score_distribution.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    logger.info("Dock score distribution plot saved to %s", output)
    return output


def plot_pred_score_distribution(
    db: Database,
    workdir: WorkDir,
    *,
    bw_adjust: float = 2.0,
    max_dock_score: float | None = 0.0,
    output: Path | None = None,
) -> Path:
    """Plot per-simsearch-cycle ``pred_score`` KDEs against the seed
    batch's actual ``dock_score`` distribution (background reference).

    :param max_dock_score: cap the x-axis maximum at this value (default
        ``0.0``), hiding the long positive-score tail. Pass ``None`` to
        show the full score range instead.
    :raises ValueError: if no simsearch cycle has produced ``pred_score``
        values yet (i.e. ``search``/``screening-cycle`` has not run).
    """
    by_cycle = db.pred_scores_by_simsearch_cycle()
    if not by_cycle:
        raise ValueError(
            "no predicted scores found; run `spacehasten search` or"
            " `spacehasten screening-cycle` first"
        )
    by_iteration = db.dock_scores_by_iteration()
    seed_scores = by_iteration.get(0, [])
    cutoff = db.query_cycle1_score_cutoff()

    fig, ax = plt.subplots(figsize=(9, 6))
    all_scores: list[float] = list(seed_scores)
    if seed_scores:
        sns.kdeplot(seed_scores, color="grey", linestyle="--",
                    label="Seed (actual dock_score)", linewidth=2, ax=ax,
                    bw_adjust=bw_adjust)

    cycles = sorted(by_cycle)
    palette = _blues_palette(len(cycles))
    for i, cycle in enumerate(cycles):
        scores = by_cycle[cycle]
        all_scores.extend(scores)
        sns.kdeplot(scores, color=palette[i % len(palette)],
                    label=f"Cycle {cycle} (pred_score)", linewidth=2,
                    ax=ax, bw_adjust=bw_adjust)

    if all_scores:
        xlim_min, xlim_max = _xlim_with_buffer(all_scores)
        if max_dock_score is not None:
            xlim_max = min(xlim_max, max_dock_score)
        ax.set_xlim(xlim_min, xlim_max)

    if cutoff is not None:
        ax.axvline(cutoff, color="black", linestyle=":", linewidth=1.5,
                   label=f"Seed cycle-1 query cutoff: {cutoff:.2f}")

    ax.set_xlabel("Docking score (kcal/mol)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Predicted score distribution per screening cycle vs. seed baseline")
    ax.legend(fontsize=10)

    output = Path(output) if output is not None else workdir.plots_dir() / "pred_score_distribution.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    logger.info("Predicted score distribution plot saved to %s", output)
    return output


def plot_pred_vs_dock_accuracy(
    db: Database,
    workdir: WorkDir,
    *,
    dock_iterations: tuple[int, ...] = (1, 2),
    gridsize: int = 60,
    output: Path | None = None,
) -> Path:
    """Hexbin density plots of predicted vs. actual ``dock_score`` for
    each requested ``dock_iteration``.

    A plain scatter of the ~1e6 compounds docked per screening round
    would be dominated by overplotting and would be slow/huge to
    render; hexbin gives an exact (non-subsampled), log-scaled density
    view instead. Each panel is annotated with Pearson r, Spearman rho,
    and RMSE, and a y=x reference line marks perfect prediction.

    :raises ValueError: if none of ``dock_iterations`` has any rows with
        both ``pred_score`` and ``dock_score`` populated.
    """
    per_iteration: dict[int, list[tuple[float, float]]] = {}
    for it in dock_iterations:
        pairs = db.pred_vs_dock_pairs(it)
        if pairs:
            per_iteration[it] = pairs
        else:
            logger.warning(
                "no pred_score/dock_score pairs for dock_iteration=%d; skipping", it
            )
    if not per_iteration:
        raise ValueError(
            f"no pred_score/dock_score pairs found for dock_iterations={dock_iterations}"
        )

    iterations = sorted(per_iteration)
    fig, axes = plt.subplots(1, len(iterations), figsize=(7 * len(iterations), 6.5),
                              squeeze=False)
    axes = axes[0]

    for ax, iteration in zip(axes, iterations):
        pairs = per_iteration[iteration]
        pred = pd.Series([p for p, _ in pairs])
        dock = pd.Series([d for _, d in pairs])

        hb = ax.hexbin(pred, dock, gridsize=gridsize, cmap="viridis",
                        bins="log", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Compound count (log scale)")

        lo = min(pred.min(), dock.min())
        hi = max(pred.max(), dock.max())
        ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.5,
                label="y = x (perfect prediction)")

        pearson_r = pred.corr(dock)
        spearman_r = pred.corr(dock, method="spearman")
        rmse = math.sqrt(((pred - dock) ** 2).mean())
        ax.text(
            0.03, 0.97,
            f"n = {len(pairs):,}\n"
            f"Pearson r = {pearson_r:.3f}\n"
            f"Spearman \u03c1 = {spearman_r:.3f}\n"
            f"RMSE = {rmse:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Predicted docking score", fontsize=13)
        ax.set_ylabel("Actual docking score", fontsize=13)
        ax.set_title(f"Iteration {iteration}")
        ax.legend(fontsize=9, loc="lower right")

    output = Path(output) if output is not None else workdir.plots_dir() / "pred_vs_dock_accuracy.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    logger.info("Predicted-vs-actual accuracy plot saved to %s", output)
    return output


__all__ = [
    "plot_dock_score_distribution",
    "plot_pred_score_distribution",
    "plot_pred_vs_dock_accuracy",
]
