#!/usr/bin/env python3
"""Render the corrected static and driven-stability figures for the blog post.

The script consumes the committed CSV outputs in ``rework/code/results`` and
writes publication-sized PNGs into ``assets/post4``.  All categorical colors
come from the Nord palette; the canvas is intentionally pure white.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "blog/a-ghost-in-the-machine/rework/code/results"
OUTPUT = REPO_ROOT / "assets/post4"

# Nord, plus pure white as required by the site background.
WHITE = "#ffffff"
POLAR_1 = "#2e3440"
POLAR_2 = "#3b4252"
POLAR_4 = "#4c566a"
SNOW_1 = "#d8dee9"
SNOW_2 = "#e5e9f0"
SNOW_3 = "#eceff4"
FROST_TEAL = "#8fbcbb"
FROST_LIGHT = "#88c0d0"
FROST_BLUE = "#5e81ac"
AURORA_RED = "#bf616a"
AURORA_ORANGE = "#d08770"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.edgecolor": POLAR_2,
            "axes.labelcolor": POLAR_1,
            "axes.titlecolor": POLAR_1,
            "xtick.color": POLAR_4,
            "ytick.color": POLAR_4,
            "text.color": POLAR_1,
            "font.family": "serif",
            "font.serif": ["EB Garamond", "Georgia", "DejaVu Serif"],
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "mathtext.fontset": "dejavuserif",
        }
    )


def mean_ci95(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = frame.groupby("rho", sort=True)[column]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    tcrit = stats.t.ppf(0.975, out["count"] - 1)
    out["ci"] = tcrit * out["std"] / np.sqrt(out["count"])
    return out


def finish_axis(ax: plt.Axes) -> None:
    ax.axhline(0.0, color=POLAR_4, lw=0.8, alpha=0.65, zorder=0)
    ax.tick_params(length=3.5)


def render_static_outlier(*, mobile: bool = False) -> Path:
    data = pd.read_csv(RESULTS / "static_outlier_full.csv")
    data["detached_corrected"] = np.abs(data["predicted"]) > data["g"] + 1e-12
    g = float(data["g"].iloc[0])

    fig_size = (4.8, 5.15) if mobile else (7.2, 3.65)
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    ax.axvspan(-g, g, color=SNOW_3, alpha=0.95, zorder=0)

    attached = data.loc[~data["detached_corrected"]]
    detached = data.loc[data["detached_corrected"]]
    ax.scatter(
        attached["predicted"],
        attached["empirical_real"],
        s=28 if mobile else 19,
        facecolors=WHITE,
        edgecolors=SNOW_1,
        linewidths=0.7,
        alpha=0.62,
        zorder=2,
    )
    ax.scatter(
        detached["predicted"],
        detached["empirical_real"],
        s=29 if mobile else 20,
        color=FROST_BLUE,
        edgecolors=WHITE,
        linewidths=0.35,
        alpha=0.58,
        zorder=3,
    )
    lim = 4.35
    ax.plot([-lim, lim], [-lim, lim], color=POLAR_2, lw=1.15, ls="--")
    ax.axvline(-g, color=POLAR_4, lw=0.75, ls=":")
    ax.axvline(g, color=POLAR_4, lw=0.75, ls=":")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
    if mobile:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\rho\alpha$")
    ax.set_ylabel(r"$\operatorname{Re}\lambda$")
    ax.set_xticks([-4, -2, -1, 0, 1, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    if mobile:
        ax.tick_params(labelsize=11.5)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)

    filename = "static-outlier-nord-mobile.png" if mobile else "static-outlier-nord.png"
    path = OUTPUT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return path


def render_dynamic_diagnostics(*, mobile: bool = False) -> Path:
    data = pd.read_csv(RESULTS / "low_rank_corrected.csv")
    series = [
        ("scalar_proxy", "scalar shortcut", AURORA_ORANGE, ":", "D"),
        ("mean_gain_proxy", "mean-gain baseline", FROST_TEAL, "-.", "^"),
        ("avg_spectral_abscissa", r"averaged edge $s(A_{\rm avg})$", FROST_BLUE, "-", "s"),
        ("lambda_1", r"time-ordered $\lambda_1$", POLAR_1, "--", "o"),
    ]

    if mobile:
        fig, axes = plt.subplots(2, 1, figsize=(5.0, 8.6), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.25, 3.65), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    ax = axes[0]
    summaries: dict[str, pd.DataFrame] = {}
    for column, label, color, linestyle, marker in series:
        summary = mean_ci95(data, column)
        summaries[column] = summary
        ax.errorbar(
            summary["rho"],
            summary["mean"],
            yerr=summary["ci"],
            color=color,
            lw=1.9 if mobile else 1.65,
            ls=linestyle,
            marker=marker,
            ms=5.0 if mobile else 4.0,
            mfc=WHITE if column == "lambda_1" else color,
            mec=color,
            mew=0.8,
            capsize=2.5 if mobile else 2.0,
            elinewidth=0.8,
            label=label,
            zorder=3,
        )

    lambda_summary = summaries["lambda_1"].set_index("rho")
    disagreement_labeled = False
    for column, _, _, _, _ in series[:-1]:
        summary = summaries[column]
        mismatch = [
            mean > 0 and lambda_summary.loc[rho, "mean"] < 0
            for rho, mean in zip(summary["rho"], summary["mean"])
        ]
        if any(mismatch):
            marked = summary.loc[mismatch]
            ax.scatter(
                marked["rho"],
                marked["mean"],
                s=82 if mobile else 64,
                facecolors="none",
                edgecolors=AURORA_RED,
                linewidths=1.25,
                label="mean sign disagreement" if not disagreement_labeled else None,
                zorder=5,
            )
            disagreement_labeled = True

    finish_axis(ax)
    ax.set_xlim(-0.08, 4.08)
    ax.set_ylim(-1.08, 0.48)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\lambda$")
    ax.text(
        0.02,
        0.97,
        "a",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=12 if mobile else 10,
    )

    ax = axes[1]
    bias_series = [
        ("scalar_proxy", "scalar shortcut", AURORA_ORANGE, "D"),
        ("mean_gain_proxy", "mean gain", FROST_TEAL, "^"),
        ("avg_spectral_abscissa", r"$s(A_{\rm avg})$", FROST_BLUE, "s"),
    ]
    offsets = [-0.035, 0.0, 0.035]
    for (column, label, color, marker), offset in zip(bias_series, offsets):
        paired = data[["seed", "rho", column, "lambda_1"]].copy()
        paired["bias"] = paired[column] - paired["lambda_1"]
        grouped = paired.groupby("rho", sort=True)["bias"]
        summary = grouped.agg(["mean", "std", "count"]).reset_index()
        summary["ci"] = stats.t.ppf(0.975, summary["count"] - 1) * summary["std"] / np.sqrt(summary["count"])

        jitter = ((paired["seed"].to_numpy() % 8) - 3.5) * 0.006
        ax.scatter(
            paired["rho"] + offset + jitter,
            paired["bias"],
            s=14 if mobile else 10,
            color=color,
            alpha=0.15,
            linewidths=0,
            zorder=1,
        )
        ax.errorbar(
            summary["rho"] + offset,
            summary["mean"],
            yerr=summary["ci"],
            color=color,
            lw=1.7 if mobile else 1.45,
            marker=marker,
            ms=4.8 if mobile else 3.9,
            capsize=2.5 if mobile else 2.0,
            elinewidth=0.85,
            label=label,
            zorder=3,
        )

    finish_axis(ax)
    ax.set_xlim(-0.08, 4.08)
    ax.set_ylim(-0.08, 0.68)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\widehat{\lambda}-\lambda_1$")
    ax.text(
        0.02,
        0.97,
        "b",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=12 if mobile else 10,
    )

    if mobile:
        for axis in axes:
            axis.tick_params(labelsize=11.5)
            axis.xaxis.label.set_size(13)
            axis.yaxis.label.set_size(13)

    filename = "dynamic-diagnostics-nord-mobile.png" if mobile else "dynamic-diagnostics-nord.png"
    path = OUTPUT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return path


def main() -> None:
    set_style()
    paths = [
        render_static_outlier(),
        render_static_outlier(mobile=True),
        render_dynamic_diagnostics(),
        render_dynamic_diagnostics(mobile=True),
    ]
    for path in paths:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
