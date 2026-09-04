"""The four headline figures of section 10.6.

Figure 2 is described as "the project in a single image": attack F1 against
mean added latency, one point per defence. D0 sits top-left (perfect attack,
zero cost), D1 bottom-right (no attack, maximum cost), and D4's claim is the
favourable corner - most of D1's protection at a fraction of its latency.

Every function takes plain data and returns a Matplotlib figure, so the same
call works from a script or a notebook and nothing here needs a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # file output only; no interactive backend required
import matplotlib.pyplot as plt  # noqa: E402

from .experiments import RunResult  # noqa: E402


def hit_rate_vs_capacity(
    results: Sequence[RunResult],
    tau: float | None = None,
    title: str = "Hit rate vs cache size",
) -> plt.Figure:
    """Figure 1 - one line per eviction policy (Contribution A)."""
    rows = [r for r in results if tau is None or r.tau == tau]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in sorted({r.policy for r in rows}):
        series = sorted((r.capacity, r.hit_rate) for r in rows if r.policy == policy)
        if series:
            xs, ys = zip(*series)
            ax.plot(xs, [100 * y for y in ys], marker="o", label=policy)

    ax.set_xscale("log")
    ax.set_xlabel("Cache capacity (entries)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(title if tau is None else f"{title} (tau={tau})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def security_latency_frontier(
    points: dict[str, tuple[float, float]],
    title: str = "Attack F1 vs mean added latency",
) -> plt.Figure:
    """Figure 2 - the project in one image.

    ``points`` maps a defence label to ``(mean_added_latency_ms, attack_f1)``.
    The favourable corner is bottom-left: no leak, no cost.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, (latency_ms, f1) in points.items():
        ax.scatter(latency_ms, f1, s=110, zorder=3)
        ax.annotate(
            label,
            (latency_ms, f1),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10,
        )

    ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(
        0.01,
        0.52,
        "chance",
        transform=ax.get_yaxis_transform(),
        fontsize=8,
        alpha=0.7,
    )

    ax.set_xlabel("Mean added latency (ms)")
    ax.set_ylabel("Attack F1")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def three_way_tradeoff(
    taus: Sequence[float],
    hit_rates: Sequence[float],
    attack_f1: Sequence[float],
    false_hit_rates: Sequence[float] | None = None,
    title: str = "The tau trade-off",
) -> plt.Figure:
    """Figure 3 - hit rate and attack success against tau, on twin axes.

    The point of the figure: prior work reads tau as a two-way trade-off
    between hit rate and false hits. Lowering it also widens the adversary's
    matching radius, so privacy is the third axis and it moves the same way.
    """
    fig, ax_left = plt.subplots(figsize=(7, 4.5))

    ax_left.plot(taus, [100 * h for h in hit_rates], marker="o", label="hit rate")
    if false_hit_rates is not None:
        ax_left.plot(
            taus,
            [100 * f for f in false_hit_rates],
            marker="s",
            linestyle=":",
            label="false-hit rate",
        )
    ax_left.set_xlabel("Similarity threshold tau")
    ax_left.set_ylabel("Rate (%)")
    ax_left.grid(alpha=0.3)

    ax_right = ax_left.twinx()
    ax_right.plot(taus, attack_f1, marker="^", linestyle="--", color="crimson", label="attack F1")
    ax_right.set_ylabel("Attack F1")
    ax_right.set_ylim(-0.05, 1.05)

    handles = ax_left.get_legend_handles_labels()[0] + ax_right.get_legend_handles_labels()[0]
    labels = ax_left.get_legend_handles_labels()[1] + ax_right.get_legend_handles_labels()[1]
    ax_left.legend(handles, labels, loc="center left")

    ax_left.set_title(title)
    fig.tight_layout()
    return fig


def jitter_repetitions(
    jitter_stds_ms: Sequence[float],
    measured: Sequence[int],
    predicted: Sequence[int],
    title: str = "Repetitions needed to defeat jitter",
) -> plt.Figure:
    """Figure 4 - D2's failure, analytical prediction against measurement.

    The prediction is ``2 * (z*sigma/separation)^2``. Measured points landing
    on that curve is the satisfying part: the defence fails by exactly the
    amount the theory says it should.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(jitter_stds_ms, predicted, linestyle="--", label="predicted  2(z*sigma/sep)^2")
    ax.scatter(jitter_stds_ms, measured, s=70, zorder=3, color="crimson", label="measured")

    ax.set_xlabel("Jitter standard deviation (ms)")
    ax.set_ylabel("Repetitions to distinguish")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    return fig


def leakage_curve(
    curve: Sequence[tuple[int, float]],
    bound: Sequence[tuple[int, float]] | None = None,
    title: str = "Reachable leakage vs probe budget",
) -> plt.Figure:
    """Graph-model leakage: fraction of concept space revealed per probe budget.

    Attack-independent, so it holds against a better adversary than the one
    implemented here.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    budgets, fractions = zip(*curve)
    ax.plot(budgets, [100 * f for f in fractions], marker="o", label="greedy coverage")

    if bound:
        b_budgets, b_fractions = zip(*bound)
        ax.plot(
            b_budgets,
            [100 * f for f in b_fractions],
            linestyle="--",
            alpha=0.7,
            label="overlap-free bound",
        )

    ax.set_xlabel("Probe budget")
    ax.set_ylabel("Concept space revealed (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def eviction_overhead(
    results: Sequence[RunResult],
    title: str = "Eviction cost vs cache size",
) -> plt.Figure:
    """Supports the central empirical claim of section 8.

    Insert and evict cost should stay flat as n grows, because rehashing holds
    average bucket occupancy near-constant. Flat lines here are the amortised
    analysis argument demonstrated rather than asserted.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for policy in sorted({r.policy for r in results}):
        series = sorted((r.capacity, r.mean_evict_us) for r in results if r.policy == policy)
        if series:
            xs, ys = zip(*series)
            ax.plot(xs, ys, marker="o", label=policy)

    ax.set_xscale("log")
    ax.set_xlabel("Cache capacity (entries)")
    ax.set_ylabel("Mean eviction time (microseconds)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, path: Path | str, dpi: int = 160) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
