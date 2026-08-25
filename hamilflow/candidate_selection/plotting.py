from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def fit_mds(dist_matrix: np.ndarray, seed: int = 42) -> np.ndarray:
    """Embed a precomputed distance matrix in 2D via MDS."""
    from sklearn.manifold import MDS

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
    )
    return mds.fit_transform(dist_matrix)


def plot_mds(
    dist_matrix: np.ndarray,
    frame_indices: Sequence[str],
    selected_idx: Sequence[int],
    *,
    coords: np.ndarray | None = None,
    seed: int = 42,
    show_fig: bool = False,
    save_path: str | Path | None = None,
    customize: Callable[["go.Figure"], "go.Figure | None"] | None = None,
):
    """Plot a 2D MDS embedding of an AMD distance matrix, coloring selected
    vs. non-selected structures.

    ``coords`` can be passed in to reuse a previously fitted embedding
    (e.g. from :func:`fit_mds`); otherwise it is fitted here.

    ``customize`` is an optional hook ``fig -> fig | None`` applied right
    before the figure is shown/saved, letting callers tweak layout, colors,
    titles, etc. without having to reimplement the base plot. If it returns
    a value, that value replaces the figure; returning ``None`` is treated
    as an in-place edit of the passed figure.
    """
    import plotly.graph_objects as go

    if coords is None:
        coords = fit_mds(dist_matrix, seed=seed)

    is_selected = np.zeros(len(frame_indices), dtype=bool)
    is_selected[np.asarray(selected_idx)] = True

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=coords[~is_selected, 0],
            y=coords[~is_selected, 1],
            mode="markers",
            name="not selected",
            marker=dict(color="lightgray", size=6),
            text=[frame_indices[i] for i in np.where(~is_selected)[0]],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=coords[is_selected, 0],
            y=coords[is_selected, 1],
            mode="markers",
            name="selected",
            marker=dict(color="crimson", size=9, symbol="diamond"),
            text=[frame_indices[i] for i in np.where(is_selected)[0]],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="MDS embedding of AMD distance matrix",
        xaxis_title="MDS 1",
        yaxis_title="MDS 2",
        template="plotly_white",
    )

    if customize is not None:
        fig = customize(fig) or fig

    if save_path is not None:
        fig.write_html(str(save_path))
    if show_fig:
        fig.show()

    return fig, coords


__all__ = ["fit_mds", "plot_mds"]
