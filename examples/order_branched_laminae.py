"""Export a synthetic V1/V2 bifurcation example with a vector graph figure."""

import torch  # noqa: F401

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lamination_station.ordering import order_regions


def main() -> None:
    """Write a new directory containing the graph, partial order, and example PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="New output directory; existing directories are rejected"
    )
    args = parser.parse_args()
    names = ["L1", "L2/3", "V1 L4", "V2 L4", "V1 L5", "V2 L5", "L6", "WM"]
    graph = pd.DataFrame(0.0, index=names, columns=names)
    for area in ["V1", "V2"]:
        path = ["L1", "L2/3", f"{area} L4", f"{area} L5", "L6", "WM"]
        for a, b in zip(path[:-1], path[1:], strict=True):
            graph.loc[a, b] = graph.loc[b, a] = 1
    result = order_regions(adjacency=graph)
    result.save(args.output)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    x = result.regions.coordinate.to_numpy()
    y = np.array([0, 0, 1, -1, 1, -1, 0, 0])
    for a, b in zip(*np.where(np.triu(graph.to_numpy() > 0, 1)), strict=True):
        ax.plot(x[[a, b]], y[[a, b]], color="#66717f", zorder=1)
    colors = [
        "#475569"
        if name in ["L1", "L2/3", "L6", "WM"]
        else "#0072B2"
        if name.startswith("V1")
        else "#D55E00"
        for name in names
    ]
    ax.scatter(x, y, c=colors, s=220, edgecolors="white", zorder=2)
    for i, name in enumerate(names):
        number = result.regions.loc[name, "lamina_number"]
        ax.annotate(
            f"{name}\ndisplay #{number}",
            (x[i], y[i]),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    ax.set(
        xlabel="Automatically inferred graph coordinate (not physical depth)",
        yticks=[],
        ylim=(-1.5, 1.7),
        xlim=(-0.08, 1.08),
        title="Synthetic V1/V2: shared layers, parallel L4/L5 paths, shared L6\n"
        "Display numbers are unique; parallel branches need not be ordered relative to each other",
    )
    for side in ["top", "left", "right"]:
        ax.spines[side].set_visible(False)
    fig.savefig(args.output / "branch_demo.pdf")
    fig.savefig(args.output / "branch_demo.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
