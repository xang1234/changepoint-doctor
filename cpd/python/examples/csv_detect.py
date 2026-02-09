#!/usr/bin/env python3
"""Detect change points from a CSV column using low-level detect_offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

import cpd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--column",
        type=int,
        default=0,
        help="Zero-based column index to detect over (default: 0)",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,)",
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip the first row",
    )
    parser.add_argument(
        "--detector",
        choices=("pelt", "binseg"),
        default="pelt",
        help="Offline detector implementation to use (default: pelt)",
    )
    parser.add_argument(
        "--cost",
        choices=("l2", "normal"),
        default="l2",
        help="Cost model (default: l2)",
    )
    parser.add_argument(
        "--n-bkps",
        type=int,
        default=1,
        help="Requested number of breakpoints for KnownK stopping (default: 1)",
    )
    return parser.parse_args()


def load_column(path: Path, column: int, delimiter: str, skip_header: bool) -> np.ndarray:
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=1 if skip_header else 0,
        dtype=np.float64,
    )
    if data.size == 0:
        raise ValueError("input CSV produced no numeric rows")

    if data.ndim == 1:
        values = np.asarray(data, dtype=np.float64)
    else:
        if column < 0 or column >= data.shape[1]:
            raise ValueError(
                f"column index {column} out of range for CSV with {data.shape[1]} columns"
            )
        values = np.asarray(data[:, column], dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("CSV column contains NaN or inf values; clean data before detection")

    return values


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    values = load_column(csv_path, args.column, args.delimiter, args.skip_header)

    result = cpd.detect_offline(
        values,
        detector=args.detector,
        cost=args.cost,
        constraints={"min_segment_len": 2},
        stopping={"n_bkps": args.n_bkps},
        repro_mode="balanced",
    )

    print(f"input_rows={values.shape[0]}")
    print("breakpoints:", result.breakpoints)
    print("change_points:", result.change_points)
    print("algorithm:", result.diagnostics.algorithm)
    print("cost_model:", result.diagnostics.cost_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
