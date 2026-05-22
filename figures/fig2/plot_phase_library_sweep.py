#!/usr/bin/env python3
"""Plot phase-library sweep results (transmission phase vs pillar diameter)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "fig2" / "phase_library_sweep.npy"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "figures" / "fig2" / "phase_library_sweep.png"

COLUMN_NAMES = (
    "diameter_nm",
    "transmission",
    "transmission_amplitude_real",
    "transmission_amplitude_imag",
    "transmission_phase_rad",
    "transmission_phase_unwrapped_rad",
)


def load_phase_library(path: Path) -> np.ndarray:
    """Load results saved by scripts/run_phase_library_sweep.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Phase-library results not found: {path}. "
            "Run scripts/run_phase_library_sweep.py first, or pass --input."
        )

    if path.suffix == ".npy":
        results = np.load(path)
    elif path.suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            results = np.asarray(
                [[float(row[column]) for column in COLUMN_NAMES] for row in reader],
                dtype=float,
            )
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .npy or .csv.")

    if results.ndim != 2 or results.shape[1] < 6:
        raise ValueError("Phase-library results must be a 2D array with six columns.")

    return results


def plot_phase_library(
    results: np.ndarray,
    output_path: Path,
    wavelength_nm: float | None = None,
    show_transmission: bool = True,
    show: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install matplotlib and rerun.") from exc

    diameter_nm = results[:, 0]
    transmission = results[:, 1]
    phase_unwrapped_rad = results[:, 5]

    if show_transmission:
        fig, (ax_phase, ax_trans) = plt.subplots(
            2,
            1,
            figsize=(7.0, 6.5),
            sharex=True,
            constrained_layout=True,
        )
    else:
        fig, ax_phase = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        ax_trans = None

    ax_phase.plot(
        diameter_nm,
        phase_unwrapped_rad,
        color="tab:purple",
        linewidth=2.0,
        label="Unwrapped transmission phase",
    )
    title_suffix = f" at {wavelength_nm:.0f} nm" if wavelength_nm is not None else ""
    ax_phase.set_ylabel("Unwrapped transmission phase (rad)")
    ax_phase.set_title(f"Figure 2 phase library{title_suffix}")
    ax_phase.grid(True, which="major", color="0.85", linewidth=0.8)
    ax_phase.legend(loc="best", frameon=True)

    if ax_trans is not None:
        ax_trans.plot(
            diameter_nm,
            transmission,
            color="tab:blue",
            linewidth=2.0,
            label="Transmission intensity",
        )
        ax_trans.set_ylabel("Transmission")
        ax_trans.set_ylim(0.0, 1.05)
        ax_trans.grid(True, which="major", color="0.85", linewidth=0.8)
        ax_trans.legend(loc="best", frameon=True)
        ax_trans.set_xlabel("Pillar diameter (nm)")
    else:
        ax_phase.set_xlabel("Pillar diameter (nm)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        default=None,
        help="Optional wavelength label for the plot title.",
    )
    parser.add_argument(
        "--no-transmission-panel",
        action="store_true",
        help="Plot only unwrapped phase vs diameter (single panel).",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_phase_library(args.input)
    plot_phase_library(
        results,
        args.output,
        wavelength_nm=args.wavelength_nm,
        show_transmission=not args.no_transmission_panel,
        show=args.show,
    )
    print(f"Saved phase-library plot: {args.output}")


if __name__ == "__main__":
    main()
