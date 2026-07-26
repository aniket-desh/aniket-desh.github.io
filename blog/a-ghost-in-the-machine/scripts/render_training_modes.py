#!/usr/bin/env python3
"""Python renderer for the recurrent-weight training movie supplied in Julia.

Input is an exported NPZ with ``J`` shaped ``[frame, neuron, neuron]``, ``ws``
shaped ``[frame, neuron, input_feature]``, and optional ``epochs``.  Unlike the
original plotting loop, this renderer clusters the final input weights once and
keeps that neuron ordering fixed, so motion in the heatmap reflects learning
rather than a changing permutation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans


WHITE = "#ffffff"
POLAR_1 = "#2e3440"
POLAR_2 = "#3b4252"
POLAR_4 = "#4c566a"
SNOW_2 = "#e5e9f0"
FROST_BLUE = "#5e81ac"
AURORA_RED = "#bf616a"
NORD_DIVERGING = LinearSegmentedColormap.from_list(
    "nord-diverging", [FROST_BLUE, WHITE, AURORA_RED], N=256
)


def normalize_ws_shape(ws: np.ndarray, n_frames: int, n_neurons: int) -> np.ndarray:
    if ws.ndim != 3 or ws.shape[0] != n_frames:
        raise ValueError("ws must have shape [frame, neuron, input_feature]")
    if ws.shape[1] == n_neurons:
        return ws
    if ws.shape[2] == n_neurons:
        return np.swapaxes(ws, 1, 2)
    raise ValueError("neuron dimension of ws does not match J")


def fixed_final_clustering(ws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = KMeans(n_clusters=2, random_state=2, n_init=20, max_iter=200).fit_predict(ws[-1])
    centers = np.vstack([ws[-1][raw == label].mean(axis=0) for label in (0, 1)])
    # Give the two arbitrary k-means labels a deterministic order.
    direction = centers[1] - centers[0]
    scores = centers @ direction
    first, second = np.argsort(scores)
    labels = np.where(raw == first, 0, 1)
    order = np.argsort(labels, kind="stable")
    return labels, order


def block_means(j: np.ndarray, labels: np.ndarray, include_diagonal: bool) -> tuple[float, float, float]:
    same = labels[:, None] == labels[None, :]
    different = ~same
    valid = np.ones_like(same, dtype=bool)
    if not include_diagonal:
        np.fill_diagonal(valid, False)
    return (
        float(j[same & valid].mean()),
        float(j[different & valid].mean()),
        float(j[valid].mean()),
    )


def load_snapshots(path: Path, include_diagonal: bool) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        if "J" not in source or "ws" not in source:
            raise ValueError("NPZ must contain J and ws arrays")
        j = np.asarray(source["J"], dtype=float)
        if j.ndim != 3 or j.shape[1] != j.shape[2]:
            raise ValueError("J must have shape [frame, neuron, neuron]")
        ws = normalize_ws_shape(np.asarray(source["ws"], dtype=float), j.shape[0], j.shape[1])
        epochs = np.asarray(source["epochs"] if "epochs" in source else np.arange(j.shape[0]), dtype=float)
    if epochs.shape != (j.shape[0],):
        raise ValueError("epochs must have one value per frame")

    labels, order = fixed_final_clustering(ws)
    shift = np.eye(j.shape[1])
    eigs = np.array([np.linalg.eigvals(frame - shift) for frame in j])
    means = np.array([block_means(frame, labels, include_diagonal) for frame in j])
    return {"J": j, "ws": ws, "epochs": epochs, "labels": labels, "order": order, "eigs": eigs, "means": means}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.edgecolor": POLAR_2,
            "axes.labelcolor": POLAR_1,
            "xtick.color": POLAR_4,
            "ytick.color": POLAR_4,
            "text.color": POLAR_1,
            "font.family": "serif",
            "font.serif": ["EB Garamond", "Georgia", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "mathtext.fontset": "dejavuserif",
        }
    )


def render_frame(data: dict[str, np.ndarray], index: int, path: Path, dpi: int = 160) -> None:
    j = data["J"]
    order = data["order"]
    eigs = data["eigs"]
    means = data["means"]
    epochs = data["epochs"]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.525), dpi=dpi)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.15, top=0.91, wspace=0.34)

    ax = axes[0]
    vmax = max(float(np.quantile(np.abs(j), 0.995)), 1e-9)
    ax.imshow(j[index][np.ix_(order, order)], cmap=NORD_DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
    split = int(np.sum(data["labels"] == 0)) - 0.5
    ax.axhline(split, color=POLAR_2, lw=0.65, alpha=0.8)
    ax.axvline(split, color=POLAR_2, lw=0.65, alpha=0.8)
    ax.set_title("reordered recurrent weights", fontsize=10)
    ax.set_xlabel("presynaptic neuron")
    ax.set_ylabel("postsynaptic neuron")

    ax = axes[1]
    initial = eigs[0]
    current = eigs[index]
    ax.scatter(initial.real, initial.imag, s=10, color=AURORA_RED, alpha=0.72, linewidths=0, label="before training")
    ax.scatter(current.real, current.imag, s=10, color=POLAR_1, alpha=0.82, linewidths=0, label="current")
    all_real = eigs.real
    all_imag = eigs.imag
    xlo, xhi = float(all_real.min()), float(all_real.max())
    yspan = max(abs(float(all_imag.min())), abs(float(all_imag.max())), 0.1)
    xpad = max(0.08 * (xhi - xlo), 0.2)
    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(-1.08 * yspan, 1.08 * yspan)
    ax.axvline(0.0, color=SNOW_2, lw=0.8, zorder=0)
    ax.axhline(0.0, color=SNOW_2, lw=0.8, zorder=0)
    ax.set_title(r"spectrum of $J-I$", fontsize=10)
    ax.set_xlabel(r"$\operatorname{Re}\lambda$")
    ax.set_ylabel(r"$\operatorname{Im}\lambda$")
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    ax = axes[2]
    upto = slice(0, index + 1)
    ax.plot(epochs[upto], means[upto, 0], color=AURORA_RED, marker="o", ms=2.7, lw=1.4, label="same cluster")
    ax.plot(epochs[upto], means[upto, 1], color=FROST_BLUE, marker="s", ms=2.5, lw=1.4, label="different clusters")
    ax.plot(epochs[upto], means[upto, 2], color=POLAR_1, marker="o", ms=2.3, lw=1.35, label="all couplings")
    xmin, xmax = float(epochs.min()), float(epochs.max())
    xpad = max(0.025 * (xmax - xmin), 1.0)
    mlo, mhi = float(means.min()), float(means.max())
    mpad = max(0.1 * (mhi - mlo), 0.002)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(mlo - mpad, mhi + mpad)
    ax.set_title(f"block means · epoch {epochs[index]:g}", fontsize=10)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("mean recurrent weight")
    ax.grid(axis="y", color=SNOW_2, lw=0.6, alpha=0.8)
    ax.legend(frameon=False, loc="best", fontsize=8)

    fig.savefig(path, dpi=dpi, facecolor=WHITE)
    plt.close(fig)


def render_movie(data: dict[str, np.ndarray], mp4: Path, poster: Path, fps: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the MP4")
    mp4.parent.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ghost-python-render-") as tmp:
        directory = Path(tmp)
        for index in range(data["J"].shape[0]):
            render_frame(data, index, directory / f"frame-{index:04d}.png")
        shutil.copy2(directory / f"frame-{data['J'].shape[0] - 1:04d}.png", poster)
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
                str(mp4),
            ],
            check=True,
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path)
    parser.add_argument("--mp4", type=Path, default=repo_root / "assets/post4/training-modes-python.mp4")
    parser.add_argument("--poster", type=Path, default=repo_root / "assets/post4/training-modes-python-poster.png")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--include-diagonal",
        action="store_true",
        help="include self-couplings in block means, matching the Julia loop literally",
    )
    args = parser.parse_args()

    style()
    data = load_snapshots(args.snapshots, args.include_diagonal)
    render_movie(data, args.mp4, args.poster, args.fps)
    print(f"rendered {data['J'].shape[0]} frames with one fixed final-weight clustering")
    print(args.mp4)
    print(args.poster)


if __name__ == "__main__":
    main()
