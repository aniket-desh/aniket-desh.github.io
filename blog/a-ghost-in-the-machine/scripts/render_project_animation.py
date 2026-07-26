#!/usr/bin/env python3
"""Animate the controlled low-rank experiment used in the blog post.

The animation follows the corrected rho grid.  Its left and center panels use
one fixed representative realization reconstructed with the experiment's seed
streams; its right panel reads the eight paired-seed means from
``low_rank_corrected.csv``.  Binning in the connectivity panel is display-only:
the spectrum is always computed from the full 500 by 500 matrix.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "blog/a-ghost-in-the-machine/rework/code"
RESULTS = EXPERIMENT_ROOT / "results/low_rank_corrected.csv"
OUTPUT = REPO_ROOT / "assets/post4"

sys.path.insert(0, str(EXPERIMENT_ROOT))
from experiments.config import rng_from_seed_parts  # noqa: E402
from experiments.connectivity import (  # noqa: E402
    balanced_connectivity,
    controlled_rank_one_vectors,
    sample_ginibre,
)


# Pure white canvas plus Nord colors only.
WHITE = "#ffffff"
POLAR_1 = "#2e3440"
POLAR_2 = "#3b4252"
POLAR_4 = "#4c566a"
SNOW_1 = "#d8dee9"
SNOW_2 = "#e5e9f0"
SNOW_3 = "#eceff4"
FROST_TEAL = "#8fbcbb"
FROST_BLUE = "#5e81ac"
AURORA_RED = "#bf616a"
AURORA_ORANGE = "#d08770"
NORD_DIVERGING = LinearSegmentedColormap.from_list(
    "nord-diverging",
    [FROST_BLUE, WHITE, AURORA_RED],
    N=256,
)


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
            "font.size": 11.0,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "mathtext.fontset": "dejavuserif",
        }
    )


def load_corrected_results(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    data = pd.read_csv(path)
    required = {
        "seed",
        "n",
        "g",
        "b",
        "rho",
        "alpha",
        "scalar_proxy",
        "avg_spectral_abscissa",
        "lambda_1",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    parameters: dict[str, float] = {}
    for column in ("n", "g", "b", "alpha"):
        values = np.sort(data[column].unique())
        if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
            raise ValueError(f"expected one {column}, found {values.tolist()}")
        parameters[column] = float(np.mean(values))

    counts = data.groupby("rho", sort=True)["seed"].nunique()
    if not np.all(counts.to_numpy() == counts.iloc[0]):
        raise ValueError("each rho must contain the same number of paired seeds")
    parameters["n_seeds"] = float(counts.iloc[0])
    return data, parameters


def representative_matrices(
    parameters: dict[str, float],
    rho_values: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    n = int(parameters["n"])
    g = parameters["g"]
    b = parameters["b"]
    alpha = parameters["alpha"]

    w = sample_ginibre(n, rng_from_seed_parts(seed, 11))
    base = balanced_connectivity(w, g, b)
    u, v = controlled_rank_one_vectors(
        n,
        rng_from_seed_parts(seed, 23),
        alpha=alpha,
        orthogonal_to_mean=True,
    )
    rank_one = np.outer(u, v)

    matrices: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    tracked: list[complex] = []
    for rho in rho_values:
        matrix = base + rho * rank_one
        eigvals, eigvecs = np.linalg.eig(matrix)
        norms = np.linalg.norm(eigvecs, axis=0) + 1e-12
        overlap = np.abs(np.conjugate(u) @ eigvecs) / norms
        tracked.append(complex(eigvals[int(np.argmax(overlap))]))
        matrices.append(matrix)
        spectra.append(eigvals)
    return u, v, matrices, spectra, np.asarray(tracked)


def block_average_sorted(
    matrix: np.ndarray,
    order: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    n = matrix.shape[0]
    if n % n_bins:
        raise ValueError(f"n={n} must be divisible by n_bins={n_bins}")
    block = n // n_bins
    sorted_matrix = matrix[np.ix_(order, order)]
    return sorted_matrix.reshape(n_bins, block, n_bins, block).mean(axis=(1, 3))


def diagnostics_by_rho(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["scalar_proxy", "avg_spectral_abscissa", "lambda_1"]
    return data.groupby("rho", sort=True)[columns].mean().reset_index()


def render_frame(
    *,
    index: int,
    path: Path,
    parameters: dict[str, float],
    rho_values: np.ndarray,
    u: np.ndarray,
    matrices: list[np.ndarray],
    spectra: list[np.ndarray],
    tracked: np.ndarray,
    diagnostics: pd.DataFrame,
    binned: list[np.ndarray],
    color_limit: float,
    dpi: int,
) -> None:
    rho = float(rho_values[index])
    n = int(parameters["n"])
    g = parameters["g"]
    alpha = parameters["alpha"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.5, 5.65625),
        dpi=dpi,
        gridspec_kw={"width_ratios": [1.02, 1.16, 1.34]},
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.14, top=0.925, wspace=0.28)

    # Panel a: display-only block averaging exposes the delocalized rank-one mode.
    ax = axes[0]
    image = ax.imshow(
        n * binned[index],
        cmap=NORD_DIVERGING,
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
        origin="lower",
    )
    sorted_u = np.sort(u)
    zero_boundary = np.searchsorted(sorted_u, 0.0) / n * binned[index].shape[0] - 0.5
    ax.axhline(zero_boundary, color=POLAR_2, lw=0.85)
    ax.axvline(zero_boundary, color=POLAR_2, lw=0.85)
    ax.text(
        0.0,
        1.01,
        "a",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_ticks([-color_limit, 0.0, color_limit])
    cbar.ax.tick_params(labelsize=8.5, length=2.5)
    cbar.outline.set_edgecolor(POLAR_2)
    cbar.outline.set_linewidth(0.6)

    # Panel b: the full static J spectrum; only the image in panel a is binned.
    ax = axes[1]
    eigvals = spectra[index]
    ax.scatter(
        eigvals.real,
        eigvals.imag,
        s=9,
        color=SNOW_1,
        edgecolors=WHITE,
        linewidths=0.2,
        alpha=0.9,
        zorder=2,
    )
    circle = plt.Circle(
        (0.0, 0.0),
        g,
        fill=False,
        color=POLAR_4,
        lw=0.9,
        ls=":",
        zorder=1,
    )
    ax.add_patch(circle)
    detached = abs(rho * alpha) > g + 1e-12
    ax.scatter(
        [tracked[index].real],
        [tracked[index].imag],
        s=76,
        facecolors=FROST_BLUE if detached else WHITE,
        edgecolors=FROST_BLUE,
        linewidths=1.5,
        marker="o",
        zorder=5,
    )
    ax.scatter(
        [rho * alpha],
        [0.0],
        s=66,
        color=AURORA_ORANGE,
        marker="x",
        linewidths=1.8,
        zorder=6,
    )
    ax.axhline(0.0, color=SNOW_2, lw=0.75, zorder=0)
    ax.axvline(0.0, color=SNOW_2, lw=0.75, zorder=0)
    ax.set_xlim(-2.05, 4.35)
    ax.set_ylim(-2.05, 2.05)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.0,
        1.01,
        "b",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(r"$\operatorname{Re}$")
    ax.set_ylabel(r"$\operatorname{Im}$")

    # Panel c: corrected paired-seed means from the nonlinear driven network.
    ax = axes[2]
    series = [
        ("scalar_proxy", "scalar shortcut", AURORA_ORANGE, ":", "D"),
        (
            "avg_spectral_abscissa",
            r"averaged edge $s(A_{\rm avg})$",
            FROST_BLUE,
            "-",
            "s",
        ),
        ("lambda_1", r"time-ordered $\lambda_1$", POLAR_1, "--", "o"),
    ]
    for column, _label, color, linestyle, marker in series:
        ax.plot(
            diagnostics["rho"],
            diagnostics[column],
            color=SNOW_1,
            lw=0.8,
            ls=linestyle,
            zorder=1,
        )
        upto = diagnostics["rho"] <= rho + 1e-12
        ax.plot(
            diagnostics.loc[upto, "rho"],
            diagnostics.loc[upto, column],
            color=color,
            lw=2.0,
            ls=linestyle,
            marker=marker,
            ms=5.0,
            mfc=WHITE if column == "lambda_1" else color,
            mec=color,
            mew=0.9,
            zorder=3,
        )
    ax.axvline(rho, color=POLAR_4, lw=0.9, ls=":", zorder=0)
    ax.axhline(0.0, color=POLAR_4, lw=0.8, alpha=0.65, zorder=0)
    ax.grid(axis="y", color=SNOW_2, lw=0.65, alpha=0.8, zorder=0)
    ax.set_xlim(-0.08, 4.08)
    ax.set_ylim(-1.08, 0.48)
    ax.text(
        0.02,
        0.97,
        "c",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.97,
        rf"$\rho={rho:.1f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=POLAR_2,
    )
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("rate")

    fig.savefig(path, dpi=dpi, facecolor=WHITE)
    plt.close(fig)


def encode_movie(
    *,
    data_path: Path,
    mp4_path: Path,
    poster_path: Path,
    representative_seed: int,
    n_bins: int,
    fps: float,
    hold_frames: int,
    dpi: int,
) -> tuple[int, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the MP4")

    data, parameters = load_corrected_results(data_path)
    rho_values = np.sort(data["rho"].unique())
    diagnostics = diagnostics_by_rho(data)
    u, _, matrices, spectra, tracked = representative_matrices(
        parameters,
        rho_values,
        representative_seed,
    )
    order = np.argsort(u, kind="stable")
    binned = [block_average_sorted(matrix, order, n_bins) for matrix in matrices]
    scaled = np.asarray(binned) * parameters["n"]
    color_limit = float(np.quantile(np.abs(scaled), 0.995))
    color_limit = max(5.0, np.ceil(color_limit / 5.0) * 5.0)

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ghost-project-animation-") as tmp:
        directory = Path(tmp)
        state_paths: list[Path] = []
        for index in range(rho_values.size):
            frame_path = directory / f"state-{index:02d}.png"
            render_frame(
                index=index,
                path=frame_path,
                parameters=parameters,
                rho_values=rho_values,
                u=u,
                matrices=matrices,
                spectra=spectra,
                tracked=tracked,
                diagnostics=diagnostics,
                binned=binned,
                color_limit=color_limit,
                dpi=dpi,
            )
            state_paths.append(frame_path)

        shutil.copy2(state_paths[-1], poster_path)
        sequence = list(range(rho_values.size)) + list(range(rho_values.size - 2, 0, -1))
        frame_number = 0
        for state in sequence:
            for _ in range(hold_frames):
                shutil.copy2(
                    state_paths[state],
                    directory / f"frame-{frame_number:04d}.png",
                )
                frame_number += 1

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(directory / "frame-%04d.png"),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-metadata",
                "title=Controlled low-rank RNN sweep",
                "-metadata",
                (
                    "comment=Representative seed 0; corrected paired-seed means; "
                    "display-only connectivity binning"
                ),
                str(mp4_path),
            ],
            check=True,
        )
    return rho_values.size, frame_number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RESULTS)
    parser.add_argument(
        "--mp4",
        type=Path,
        default=OUTPUT / "training-modes-nord.mp4",
    )
    parser.add_argument(
        "--poster",
        type=Path,
        default=OUTPUT / "training-modes-nord-poster.png",
    )
    parser.add_argument("--representative-seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--hold-frames", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=128)
    args = parser.parse_args()

    set_style()
    states, frames = encode_movie(
        data_path=args.data,
        mp4_path=args.mp4,
        poster_path=args.poster,
        representative_seed=args.representative_seed,
        n_bins=args.bins,
        fps=args.fps,
        hold_frames=args.hold_frames,
        dpi=args.dpi,
    )
    print(f"rendered {states} corrected rho states into {frames} frames")
    print(args.mp4)
    print(args.poster)


if __name__ == "__main__":
    main()
