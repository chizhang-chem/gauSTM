"""Read STM MATLAB-style output and render a grayscale STM image."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MATRIX_PATTERN = r"{name}\s*=\s*\[\.\.\.\s*(.*?)\];"
STATE_PATTERN = re.compile(r"\b((?:Ex\d+|MOA\d+|MOB\d+|TDP\d+|TDH\d+)LMN\d+)\s*=\s*\[\.\.\.", re.S)


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk")


def parse_matlab_matrix(text: str, name: str) -> np.ndarray:
    pattern = re.compile(MATRIX_PATTERN.format(name=re.escape(name)), re.S)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Matrix '{name}' not found")

    rows = [row.strip() for row in match.group(1).split(";...") if row.strip()]
    return np.array([[float(value) for value in row.split()] for row in rows], dtype=float)


def detect_first_state_name(text: str) -> str:
    match = STATE_PATTERN.search(text)
    if not match:
        raise ValueError("No STM state array was found in the .m file")
    return match.group(1)


def render_stm_image(z_value: np.ndarray, output_path: Path, dpi: int) -> None:
    intensity = np.square(np.abs(z_value))
    vmax = float(intensity.max())
    if vmax > 0.0:
        intensity = intensity / vmax

    fig = plt.figure(figsize=(7.68, 7.68), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="black")
    ax.imshow(intensity, cmap="gray", origin="lower", interpolation="bicubic")
    ax.set_axis_off()
    fig.savefig(output_path, dpi=dpi, facecolor="black", edgecolor="black")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot |M_ts|^2 from STM MATLAB-style output."
    )
    parser.add_argument("input_m", type=Path, help="Path to the STM .m output file")
    parser.add_argument("output_png", type=Path, help="Path to the output PNG image")
    parser.add_argument(
        "--matrix",
        default=None,
        help="Matrix name to plot, e.g. Ex000LMN1 or TDP001LMN1. Defaults to the first STM state found.",
    )
    parser.add_argument("--dpi", type=int, default=100, help="PNG DPI, default: 100")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    text = read_text_auto(args.input_m)
    matrix_name = args.matrix or detect_first_state_name(text)

    parse_matlab_matrix(text, "X")
    parse_matlab_matrix(text, "Y")
    z_value = parse_matlab_matrix(text, matrix_name)

    render_stm_image(z_value, args.output_png, args.dpi)
    print(f"Saved: {args.output_png}")
    print(f"Matrix: {matrix_name}")


if __name__ == "__main__":
    main()
