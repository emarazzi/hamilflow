from __future__ import annotations

import argparse
from pathlib import Path

from hamilflow.candidate_selection import CandidateSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select representative structures from a trajectory via "
        "AMD pairwise distances (FPS or k-means), and plot a 2D MDS map."
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("./dataset.extxyz"),
        help="Path to a trajectory file readable by ase.io.read (e.g. .extxyz).",
    )
    parser.add_argument("-n", "--n-select", type=int, default=20, help="Number of structures to select.")
    parser.add_argument(
        "--method",
        choices=["fps", "kmeans", "pattern"],
        default="kmeans",
        help="Selection algorithm (default: kmeans).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Regex on atoms.info['index']. Restricts the candidate pool for "
        "fps/kmeans, or selects every match directly for method='pattern'.",
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=None,
        help="Text file, one atoms.info['index'] label per line, to exclude "
        "from the candidate pool (structures stay in the MDS map).",
    )
    parser.add_argument("-k", "--k-neighbors", type=int, default=100, help="AMD k parameter.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for FPS/k-means/MDS.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./selected_indices.txt"),
        help="Output file for the selected labels.",
    )
    parser.add_argument(
        "--plot-html",
        type=Path,
        default=Path("./mds_map.html"),
        help="Output path for the interactive 2D MDS map.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trajectory.exists():
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory}")

    selector = CandidateSelector(k_neighbors=args.k_neighbors, seed=args.seed)

    # 1. AMD + AMD pairwise distances (done inside select) and 2. selection
    #    (FPS / k-means centroids / regex pattern), optionally restricted to
    #    a candidate pool by --pattern and/or --exclude-file.
    result = selector.select(
        args.trajectory,
        n=args.n_select,
        method=args.method,
        pattern=args.pattern,
        exclude=args.exclude_file,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_labels(args.output)
    print(f"Selected {len(result.selected_idx)} / {len(result.frame_indices)} structures via {args.method}")
    print(f"Labels written to {args.output.resolve()}")

    # 3. MDS embedding + plot, saved as an interactive HTML file. `customize`
    #    is optional -- here it just changes marker colors/sizes; you could
    #    also add extra traces, change the title, etc.
    def customize(fig):
        fig.update_traces(marker=dict(size=5, color="lightgray"), selector=dict(name="not selected"))
        fig.update_traces(marker=dict(size=11, color="darkorange", symbol="diamond"), selector=dict(name="selected"))
        return fig

    args.plot_html.parent.mkdir(parents=True, exist_ok=True)
    fig, coords = selector.plot(result, save_path=args.plot_html, customize=customize)
    print(f"MDS map saved to {args.plot_html.resolve()}")

    # Alternative: skip the built-in plot entirely and build your own from
    # the raw coords + selected mask (e.g. with matplotlib):
    #
    # coords, is_selected = selector.mds_coords(result)
    # import matplotlib.pyplot as plt
    # plt.scatter(coords[~is_selected, 0], coords[~is_selected, 1], c="lightgray")
    # plt.scatter(coords[is_selected, 0], coords[is_selected, 1], c="crimson")
    # plt.savefig("mds_map.png")


if __name__ == "__main__":
    main()
