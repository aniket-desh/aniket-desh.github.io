#!/usr/bin/env python3
"""Restyle Rainer Engelken's training movie with a white/Nord palette.

This is deliberately a color-and-size remaster of the supplied source GIF,
not a numerical reproduction.  The original frames remain the provenance for
all matrices, eigenvalues, cluster assignments, and block means.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


WHITE = np.array([255.0, 255.0, 255.0])
POLAR_1 = np.array([46.0, 52.0, 64.0])       # #2e3440
FROST_BLUE = np.array([94.0, 129.0, 172.0])  # #5e81ac
AURORA_RED = np.array([191.0, 97.0, 106.0])  # #bf616a


def blend(anchor: np.ndarray, strength: np.ndarray, levels: int = 14) -> np.ndarray:
    """Blend white toward a Nord anchor, quantizing to stable palette steps."""
    strength = np.clip(strength, 0.0, 1.0)
    strength = np.rint(strength * (levels - 1)) / (levels - 1)
    return WHITE + strength[..., None] * (anchor - WHITE)


def nord_remap(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    hi = rgb.max(axis=2)
    lo = rgb.min(axis=2)
    chroma = hi - lo
    lightness = rgb.mean(axis=2)

    red = (rgb[..., 0] > rgb[..., 2] + 7) & (rgb[..., 0] > rgb[..., 1] + 4) & (chroma > 8)
    blue = (rgb[..., 2] > rgb[..., 0] + 7) & (chroma > 8)
    neutral = ~(red | blue)

    # Colored antialiasing is controlled chiefly by chroma; neutral text and
    # axes are controlled by darkness.  Every result lies on a white-to-Nord
    # interpolation, so no unrelated plotting palette is introduced.
    colored_strength = np.clip(chroma / 175.0 + (255.0 - lightness) / 620.0, 0.0, 1.0)
    neutral_strength = np.clip(((255.0 - lightness) / 225.0) ** 0.82, 0.0, 1.0)

    out = np.empty_like(rgb)
    out[red] = blend(AURORA_RED, colored_strength)[red]
    out[blue] = blend(FROST_BLUE, colored_strength)[blue]
    out[neutral] = blend(POLAR_1, neutral_strength)[neutral]
    # The source GIF uses a lightly dithered near-white canvas.  Flatten that
    # residue so the remaster really sits on the blog's white background.
    out[lightness > 238] = WHITE
    return Image.fromarray(np.rint(out).astype(np.uint8), mode="RGB")


def remaster(
    source: Path,
    mp4_path: Path,
    poster_path: Path,
    width: int = 1600,
    fps: float = 6.0,
) -> tuple[int, tuple[int, int]]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the MP4")

    with Image.open(source) as gif:
        source_size = gif.size
        height = int(round(source_size[1] * width / source_size[0]))
        height += height % 2
        target_size = (width, height)

        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ghost-training-modes-") as tmp:
            frame_dir = Path(tmp)
            count = 0
            final_frame: Image.Image | None = None
            for count, frame in enumerate(ImageSequence.Iterator(gif), start=1):
                resized = frame.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
                converted = nord_remap(resized)
                converted.save(frame_dir / f"frame-{count - 1:04d}.png", optimize=True)
                final_frame = converted

            if final_frame is None:
                raise ValueError(f"{source} contains no frames")
            final_frame.save(poster_path, optimize=True)

            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame-%04d.png"),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "24",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ]
            subprocess.run(command, check=True)

    return count, target_size


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source GIF supplied by the original experiment author")
    parser.add_argument(
        "--mp4",
        type=Path,
        default=repo_root / "assets/post4/training-modes-nord.mp4",
    )
    parser.add_argument(
        "--poster",
        type=Path,
        default=repo_root / "assets/post4/training-modes-nord-poster.png",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()

    count, size = remaster(args.source, args.mp4, args.poster, args.width, args.fps)
    print(f"remastered {count} frames at {size[0]}x{size[1]}")
    print(args.mp4)
    print(args.poster)


if __name__ == "__main__":
    main()
