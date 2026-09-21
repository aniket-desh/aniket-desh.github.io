#!/usr/bin/env python3
"""Render Post VII figures from the saved synthetic experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "assets"

NORD = {
    "ink": "#2E3440",
    "muted": "#4C566A",
    "snow": "#ECEFF4",
    "snow_dark": "#D8DEE9",
    "frost_dark": "#5E81AC",
    "frost": "#81A1C1",
    "cyan": "#88C0D0",
    "red": "#BF616A",
    "orange": "#D08770",
    "yellow": "#EBCB8B",
    "green": "#A3BE8C",
    "purple": "#B48EAD",
}

Q_COLORS = (NORD["frost_dark"], NORD["cyan"], NORD["orange"], NORD["purple"])
Q_MARKERS = ("o", "s", "^", "D")
SIGMA_COLORS = {0.0: NORD["frost_dark"], 0.1: NORD["cyan"], 0.25: NORD["orange"]}
STEP_MARKERS = {1: "o", 4: "s", 16: "^"}


plt.rcParams.update(
    {
        # Post VI falls back to Matplotlib's bundled serif on this machine.
        # Pinning it here keeps exports reproducible and avoids host font scans.
        "font.family": "DejaVu Serif",
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 9.0,
        "axes.titleweight": "regular",
        "axes.labelcolor": NORD["ink"],
        "axes.edgecolor": NORD["muted"],
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "xtick.color": NORD["muted"],
        "ytick.color": NORD["muted"],
        "legend.fontsize": 8.0,
        "text.color": NORD["ink"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": True,
    }
)


def style_axis(axis: plt.Axes) -> None:
    axis.patch.set_alpha(0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(NORD["muted"])
    axis.tick_params(length=3, width=0.7)
    axis.grid(False)


def panel_title(axis: plt.Axes, tag: str, title: str) -> None:
    axis.set_title(f"({tag}) {title}", loc="left", pad=4, color=NORD["ink"])


def mean_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def save_figure(figure: plt.Figure, name: str, *, mobile: bool) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    suffix = "-mobile" if mobile else ""
    figure.savefig(
        ASSETS / f"{name}{suffix}.png",
        dpi=240,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)


def render_overlap(*, mobile: bool) -> None:
    data = np.load(DATA / "overlap_results.npz")
    q_values = data["q_values"]
    d_values = data["d_values"]
    ratios = data["ratios"]
    empirical = data["empirical"]
    theory = data["theory"]
    relative_mean = data["relative_mean"]
    relative_se = data["relative_se"]

    if mobile:
        figure, axes = plt.subplots(3, 1, figsize=(3.9, 9.2), constrained_layout=True)
    else:
        figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), constrained_layout=True)

    q_index = int(np.argmin(abs(q_values - 0.9)))
    d_index = len(d_values) - 1
    empirical_corr = empirical[q_index, d_index]
    empirical_corr = empirical_corr / np.sqrt(np.outer(np.diag(empirical_corr), np.diag(empirical_corr)))
    theory_corr = theory[q_index, d_index]
    theory_corr = theory_corr / np.sqrt(np.outer(np.diag(theory_corr), np.diag(theory_corr)))
    combined = np.triu(empirical_corr) + np.tril(theory_corr, -1)
    cmap = LinearSegmentedColormap.from_list(
        "snow-frost", ("#FFFFFF", NORD["snow_dark"], NORD["cyan"], NORD["frost_dark"], NORD["ink"])
    )
    image = axes[0].imshow(combined, vmin=0.5, vmax=1.0, cmap=cmap, origin="lower")
    tick_positions = np.arange(len(ratios))
    tick_labels = ("1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16")
    axes[0].set_xticks(tick_positions[::2], tick_labels[::2])
    axes[0].set_yticks(tick_positions[::2], tick_labels[::2])
    axes[0].set_xlabel(r"$\lambda_j/\lambda_\star$")
    axes[0].set_ylabel(r"$\lambda_i/\lambda_\star$")
    axes[0].text(0.03, 0.97, "theory below\nsimulation above", transform=axes[0].transAxes, va="top", fontsize=7.5)
    panel_title(axes[0], "a", "error correlation")
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    colorbar.ax.tick_params(length=2, labelsize=7.5, colors=NORD["muted"])
    colorbar.outline.set_visible(False)

    last = len(d_values) - 1
    global_min = np.inf
    global_max = 0.0
    for qi, q in enumerate(q_values):
        predicted = theory[qi, last].ravel()
        observed = empirical[qi, last].ravel()
        global_min = min(global_min, predicted.min(), observed.min())
        global_max = max(global_max, predicted.max(), observed.max())
        diag = np.eye(len(ratios), dtype=bool).ravel()
        axes[1].scatter(
            predicted[~diag],
            observed[~diag],
            s=11,
            alpha=0.48,
            color=Q_COLORS[qi],
            marker=Q_MARKERS[qi],
            linewidths=0,
            label=fr"$q={q:g}$",
        )
        axes[1].scatter(
            predicted[diag],
            observed[diag],
            s=21,
            facecolors="none",
            edgecolors=Q_COLORS[qi],
            marker=Q_MARKERS[qi],
            linewidths=0.8,
        )
    identity = np.geomspace(global_min * 0.9, global_max * 1.1, 100)
    axes[1].plot(identity, identity, color=NORD["muted"], linewidth=0.9, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(identity[0], identity[-1])
    axes[1].set_ylim(identity[0], identity[-1])
    axes[1].set_xlabel("predicted overlap")
    axes[1].set_ylabel("simulated overlap")
    axes[1].legend(frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.7)
    panel_title(axes[1], "b", "all snapshot pairs")
    style_axis(axes[1])

    for qi, q in enumerate(q_values):
        axes[2].errorbar(
            d_values,
            relative_mean[qi],
            yerr=1.96 * relative_se[qi],
            color=Q_COLORS[qi],
            marker=Q_MARKERS[qi],
            markersize=4.0,
            linewidth=1.45,
            capsize=2.2,
            label=fr"$q={q:g}$",
        )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(d_values, [str(int(d)) for d in d_values])
    axes[2].set_xlabel("dimension $d$")
    axes[2].set_ylabel("relative matrix error")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    panel_title(axes[2], "c", "finite-size convergence")
    style_axis(axes[2])
    if mobile:
        axes[2].legend(frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8)

    save_figure(figure, "overlap-theory", mobile=mobile)


def render_weighting(*, mobile: bool) -> None:
    data = np.load(DATA / "weighting_results.npz")
    k_values = data["k_values"]
    fit_sizes = data["fit_sizes"]
    fit_index = int(np.where(fit_sizes == 512)[0][0])
    oracle = float(data["oracle_risk"])

    if mobile:
        figure, axes = plt.subplots(3, 1, figsize=(3.9, 9.3), constrained_layout=True)
    else:
        figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), constrained_layout=True)

    axes[0].plot(
        k_values,
        data["quality_coarse"],
        color=NORD["purple"],
        marker="^",
        linewidth=1.5,
        markersize=4.5,
        label="quality-only",
    )
    axes[0].plot(
        k_values,
        data["fixed_coarse"],
        color=NORD["green"],
        marker="s",
        linewidth=1.7,
        markersize=4.5,
        label="overlap weights",
    )
    axes[0].plot(
        k_values,
        data["fixed_augmented"],
        color=NORD["frost_dark"],
        marker="o",
        linewidth=1.5,
        markersize=4.0,
        label=r"pool includes $\lambda_\star$",
    )
    axes[0].axhline(oracle, color=NORD["ink"], linewidth=0.9, linestyle="--", label="oracle ridge")
    axes[0].set_xticks(k_values)
    axes[0].set_xlabel("retained snapshots $K$")
    axes[0].set_ylabel("population test MSE")
    axes[0].legend(frameon=False, handletextpad=0.4)
    panel_title(axes[0], "a", "fixed population weights")
    style_axis(axes[0])

    adaptive = data["adaptive_risk"][:, fit_index]
    greedy = data["greedy_risk"][:, fit_index]
    fixed = data["fixed_risk"][:, fit_index]
    for values, color, marker, label in (
        (fixed, NORD["frost_dark"], "o", "fixed MP weights"),
        (greedy, NORD["purple"], "^", "fitness greedy"),
        (adaptive, NORD["green"], "s", "fitness simplex"),
    ):
        mean, se = mean_se(values)
        if label == "fitness simplex":
            for row in values:
                axes[1].plot(k_values, row, color=color, linewidth=0.6, alpha=0.08)
        axes[1].errorbar(
            k_values,
            mean,
            yerr=1.96 * se,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=4.5,
            capsize=2.2,
            label=label,
        )
    tuned_mean = float(data["tuned_ridge_risk"][:, fit_index].mean())
    tuned_se = float(data["tuned_ridge_risk"][:, fit_index].std(ddof=1) / np.sqrt(len(adaptive)))
    axes[1].axhspan(
        tuned_mean - 1.96 * tuned_se,
        tuned_mean + 1.96 * tuned_se,
        color=NORD["orange"],
        alpha=0.12,
        linewidth=0,
    )
    axes[1].axhline(tuned_mean, color=NORD["orange"], linewidth=1.2, linestyle="-.", label="same-fitness ridge")
    axes[1].set_xticks(k_values)
    axes[1].set_xlabel("retained snapshots $K$")
    axes[1].set_ylabel("realized test MSE")
    axes[1].legend(frameon=False, handletextpad=0.4)
    panel_title(axes[1], "b", "512 fitness examples")
    style_axis(axes[1])

    mean_risk = data["individual_risk"].mean(axis=0)
    mean_weight = data["adaptive_weights"][:, fit_index].mean(axis=0)
    ratios = data["coarse_ratios"]
    axes[2].scatter(
        mean_risk,
        mean_weight,
        s=38,
        color=NORD["green"],
        edgecolor=NORD["ink"],
        linewidth=0.45,
        zorder=3,
    )
    label_offsets = {
        1 / 8: (-14, -8),
        16.0: (5, 12),
    }
    for x, y, ratio in zip(mean_risk, mean_weight, ratios):
        if ratio < 1:
            label = f"1/{int(round(1 / ratio))}"
        else:
            label = str(int(ratio))
        axes[2].annotate(
            label,
            (x, y),
            xytext=label_offsets.get(float(ratio), (3, 4)),
            textcoords="offset points",
            fontsize=7.2,
            color=NORD["muted"],
        )
    axes[2].set_xlabel("standalone test MSE")
    axes[2].set_ylabel("mean learned weight")
    axes[2].text(0.98, 0.96, r"labels: $\lambda/\lambda_\star$", transform=axes[2].transAxes, ha="right", va="top", fontsize=7.5)
    panel_title(axes[2], "c", "worse alone, useful together")
    style_axis(axes[2])

    save_figure(figure, "weighting-anatomy", mobile=mobile)


def render_chain(*, mobile: bool) -> None:
    data = np.load(DATA / "chain_results.npz")
    cycles = data["exact_cycles"]
    distance = data["exact_distance"]
    configs = data["configs"]

    if mobile:
        figure, axes = plt.subplots(3, 1, figsize=(3.9, 9.25), constrained_layout=True)
    else:
        figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), constrained_layout=True)

    numerical_floor = 1e-14
    for row in distance:
        axes[0].plot(
            cycles,
            np.maximum(row, numerical_floor),
            color=NORD["frost_dark"],
            linewidth=0.65,
            alpha=0.10,
        )
    mean, se = mean_se(distance)
    lower = np.maximum(mean - 1.96 * se, numerical_floor)
    upper = np.maximum(mean + 1.96 * se, numerical_floor)
    axes[0].fill_between(cycles, lower, upper, color=NORD["frost_dark"], alpha=0.14, linewidth=0)
    axes[0].plot(cycles, np.maximum(mean, numerical_floor), color=NORD["frost_dark"], linewidth=1.9)
    axes[0].axhline(numerical_floor, color=NORD["muted"], linewidth=0.7, linestyle=":")
    axes[0].set_yscale("log")
    axes[0].set_ylim(5e-15, 1.8)
    axes[0].set_xlabel("distillation cycle")
    axes[0].set_ylabel("relative distance to effective ridge")
    panel_title(axes[0], "a", "exact chain collapse")
    style_axis(axes[0])

    individual = data["individual_risk"]
    cross = data["cross_correlation"]
    learned = data["learned_risk"]
    best = data["best_risk"]
    for panel, alpha in zip((axes[1], axes[2]), (0.0, 0.5)):
        for sigma in data["sigmas"]:
            for step in data["steps"]:
                index = np.where(
                    np.isclose(configs[:, 0], alpha)
                    & np.isclose(configs[:, 1], sigma)
                    & np.isclose(configs[:, 2], step)
                )[0][0]
                x = cross[:, index]
                y = individual[:, index]
                panel.errorbar(
                    x.mean(),
                    y.mean(),
                    xerr=1.96 * x.std(ddof=1) / np.sqrt(len(x)),
                    yerr=1.96 * y.std(ddof=1) / np.sqrt(len(y)),
                    color=SIGMA_COLORS[float(sigma)],
                    marker=STEP_MARKERS[int(step)],
                    markeredgecolor=NORD["ink"],
                    markeredgewidth=0.35,
                    markersize=5.2,
                    capsize=1.8,
                    linewidth=0.8,
                )
        gap = learned[:, np.isclose(configs[:, 0], alpha)] - best[:, np.isclose(configs[:, 0], alpha)]
        panel.text(
            0.98,
            0.96,
            fr"top-8 $-$ best: {1e3 * gap.mean():+.1f} milli-MSE",
            transform=panel.transAxes,
            ha="right",
            va="top",
            fontsize=7.4,
            color=NORD["muted"],
        )
        panel.set_xlabel("cross-trajectory error correlation")
        panel.set_ylabel("mean snapshot test MSE")
        panel.xaxis.set_major_locator(MaxNLocator(nbins=4))
        title = "finite time, no distillation" if alpha == 0.0 else "finite time with distillation"
        panel_title(panel, "b" if alpha == 0.0 else "c", title)
        style_axis(panel)

    color_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=fr"$\sigma_0={sigma:g}$")
        for sigma, color in SIGMA_COLORS.items()
    ]
    step_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor="white", markeredgecolor=NORD["ink"], color=NORD["ink"], label=fr"$L={step}$")
        for step, marker in STEP_MARKERS.items()
    ]
    axes[2].legend(
        handles=color_handles + step_handles,
        frameon=False,
        ncol=2,
        loc="lower left",
        handletextpad=0.25,
        columnspacing=0.7,
    )

    save_figure(figure, "chain-dynamics", mobile=mobile)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobile-only", action="store_true")
    parser.add_argument("--desktop-only", action="store_true")
    args = parser.parse_args()
    if args.mobile_only and args.desktop_only:
        parser.error("choose at most one output mode")
    modes = (True,) if args.mobile_only else (False,) if args.desktop_only else (False, True)
    for mobile in modes:
        render_overlap(mobile=mobile)
        render_weighting(mobile=mobile)
        render_chain(mobile=mobile)


if __name__ == "__main__":
    main()
