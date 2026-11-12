"""Call gradient-ring regions from a CSV of typed cells and export a reviewable fit."""

import torch  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from lamination_station.regions import GradientSettings, fit_regions, prepare_gradient_rings


def main() -> None:
    """Run an explicit-resolution analysis without machine-specific input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV containing one row per typed cell")
    parser.add_argument(
        "output", type=Path, help="New output directory; existing directories are rejected"
    )
    parser.add_argument("--cell-id", default="cell_id")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--type-col", default="cell_type")
    parser.add_argument(
        "--sample-col", help="Section column; omit only for a single coordinate frame"
    )
    parser.add_argument("--piece-col", help="Optional tissue-piece column inside each section")
    parser.add_argument("--n-regions", type=int, default=8)
    parser.add_argument("--ring-ends", nargs="+", type=int, default=[15, 30, 60])
    parser.add_argument("--far-k", type=int, default=90)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to preserve earlier calls")
    text_columns = [args.cell_id, args.type_col]
    text_columns.extend(c for c in (args.sample_col, args.piece_col) if c is not None)
    cells = pd.read_csv(args.input, dtype={c: "string" for c in text_columns})
    if args.cell_id not in cells:
        parser.error(f"Missing cell ID column: {args.cell_id}")
    cells = cells.set_index(args.cell_id)
    prepared = prepare_gradient_rings(
        cells=cells,
        x_col=args.x_col,
        y_col=args.y_col,
        celltype_col=args.type_col,
        sample_col=args.sample_col,
        piece_col=args.piece_col,
        settings=GradientSettings(ring_ends=tuple(args.ring_ends), far_k=args.far_k),
    )
    result = fit_regions(
        features=prepared,
        n_regions=args.n_regions,
        seed=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
    )
    result.save(args.output)
    print(result.region_summary.to_string())
    print(
        f"Saved {len(cells)} cells to {args.output}; selected fit converged={result.model.converged}"
    )


if __name__ == "__main__":
    main()
