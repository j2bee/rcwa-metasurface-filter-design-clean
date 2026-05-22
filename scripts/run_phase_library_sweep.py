#!/usr/bin/env python3
"""Run a fixed-wavelength, diameter-swept RCWA phase-library validation sweep.

Metasurface phase libraries are usually characterized at one design wavelength
while geometry (pillar diameter) is varied to span a target phase range. Each
unit cell imparts a local transmission phase; arranging cells with different
diameters builds a discretized wavefront for beam steering, metalenses, and
other wavefront-engineering applications.

This script is separate from scripts/run_fig2.py, which sweeps wavelength at a
fixed pillar diameter for spectral Figure 2 reproduction.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "fig2"
OUTPUT_STEM = "phase_library_sweep"

CSV_COLUMNS = (
    "diameter_nm",
    "transmission",
    "transmission_amplitude_real",
    "transmission_amplitude_imag",
    "transmission_phase_rad",
    "transmission_phase_unwrapped_rad",
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.fig2.geometry import build_unit_cell_geometry  # noqa: E402
from scripts.run_fig2 import (  # noqa: E402
    _assemble_rcwa_object,
    _complex_transmission_amplitude,
    _load_rcwa_module,
)


def run_phase_library_sweep(
    wavelength_nm: float,
    diameter_start_nm: float,
    diameter_stop_nm: float,
    num_diameters: int,
    grid_shape: tuple[int, int],
    nG: int,
    q_ref: float,
) -> np.ndarray:
    """Sweep pillar diameter at fixed wavelength and return phase-library rows."""
    if num_diameters < 1:
        raise ValueError("num_diameters must be at least 1")

    rcwa_module = _load_rcwa_module()
    diameters_nm = np.linspace(diameter_start_nm, diameter_stop_nm, num_diameters)
    transmission_amplitudes: list[complex] = []
    intensity_rows: list[tuple[float, float]] = []

    for diameter_nm in diameters_nm:
        geometry_payload = build_unit_cell_geometry(diameter_nm=float(diameter_nm), grid_shape=grid_shape)
        obj, _frequency = _assemble_rcwa_object(
            rcwa_module,
            geometry_payload,
            wavelength_nm,
            nG=nG,
            q_ref=q_ref,
        )
        _reflection, transmission = obj.RT_Solve(normalize=1)
        # Complex zero-order coefficient before squaring; intensity from Poynting flux.
        transmission_amplitude = _complex_transmission_amplitude(rcwa_module, obj)
        transmission_amplitudes.append(transmission_amplitude)
        intensity_rows.append((float(diameter_nm), float(np.real(transmission))))

    amplitudes = np.asarray(transmission_amplitudes, dtype=complex)
    phase_rad = np.angle(amplitudes)
    # Unwrap along diameter so the phase library is a continuous lookup curve.
    phase_unwrapped_rad = np.unwrap(phase_rad)

    return np.column_stack(
        (
            diameters_nm,
            [row[1] for row in intensity_rows],
            np.real(amplitudes),
            np.imag(amplitudes),
            phase_rad,
            phase_unwrapped_rad,
        )
    )


def save_phase_library(results: np.ndarray, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{stem}.npy"
    csv_path = output_dir / f"{stem}.csv"

    np.save(npy_path, results)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(results.tolist())

    return npy_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength-nm", type=float, default=580.0, help="Fixed operating wavelength.")
    parser.add_argument("--diameter-start-nm", type=float, default=60.0)
    parser.add_argument("--diameter-stop-nm", type=float, default=220.0)
    parser.add_argument("--num-diameters", type=int, default=50)
    parser.add_argument("--grid-nx", type=int, default=128)
    parser.add_argument("--grid-ny", type=int, default=128)
    parser.add_argument("--nG", type=int, default=101)
    parser.add_argument("--q-ref", type=float, default=1e10)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_shape = (args.grid_nx, args.grid_ny)

    results = run_phase_library_sweep(
        wavelength_nm=args.wavelength_nm,
        diameter_start_nm=args.diameter_start_nm,
        diameter_stop_nm=args.diameter_stop_nm,
        num_diameters=args.num_diameters,
        grid_shape=grid_shape,
        nG=args.nG,
        q_ref=args.q_ref,
    )
    npy_path, csv_path = save_phase_library(results, args.output_dir, args.output_stem)

    phase_span_rad = float(np.max(results[:, 5]) - np.min(results[:, 5]))
    print("Phase-library sweep completed")
    print(f"  wavelength_nm: {args.wavelength_nm}")
    print(f"  diameter_nm: {args.diameter_start_nm} to {args.diameter_stop_nm} ({args.num_diameters} points)")
    print(f"  phase_span_rad (unwrapped): {phase_span_rad:.4f}")
    print(f"  results_npy: {npy_path}")
    print(f"  results_csv: {csv_path}")


if __name__ == "__main__":
    main()
