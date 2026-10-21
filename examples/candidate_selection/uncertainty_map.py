"""Color the 2D map of a dataset by the ensemble band uncertainty.

Reads the MDS map saved by ``select_candidates.py`` (``selection_state.npz``)
and the per-structure uncertainty written by
``uncertainty/ensemble_band_uncertainty.py``, and plots every structure
colored by its uncertainty. Structures without an uncertainty value are drawn
in grey; the structures selected by ``select_candidates.py`` can be outlined.
Useful to see which regions of structure space the models are unsure about.

The uncertainty JSON is keyed by structure folder name: these must match the
``atoms.info["index"]`` labels of the dataset.

Output: an interactive HTML map, OUTPUT_HTML.

Run with:  python uncertainty_map.py
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from hamilflow.uncertainty.band_postprocess import BandUncertaintyPostProcessor

# --- User settings -----------------------------------------------------------
SELECTION_DIR = Path("/path/to/your/selection/folder")  # OUTPUT_DIR of select_candidates.py
UNCERTAINTY_JSON = Path("/path/to/your/band_uncertainty.json")
SHOW_SELECTED = True  # outline the structures in selected_labels.txt
OUTPUT_HTML = SELECTION_DIR / "uncertainty_map.html"
# Per-structure scalar: band-edge-weighted mean of the ensemble sigma, counting
# states within ~E_CUT eV of the VBM/CBM (smoothed over TAU eV). Other choices:
# "weighted_max_eV", "mean_eV", "max_eV" (see BandUncertaintyPostProcessor).
STAT = "weighted_mean_eV"
E_CUT, TAU = 1.0, 0.1
# -----------------------------------------------------------------------------

state = np.load(SELECTION_DIR / "selection_state.npz", allow_pickle=True)
labels = np.array(state["frame_indices"], dtype=str)
coords = state["coords"]

pp = BandUncertaintyPostProcessor.from_json(UNCERTAINTY_JSON)
stats = pp.weighted_summary(e_cut=E_CUT, tau=TAU)
uncertainty = np.array([stats[lbl][STAT] if lbl in stats else np.nan for lbl in labels])
has_unc = ~np.isnan(uncertainty)
print(f"{has_unc.sum()} / {len(labels)} structures have an uncertainty value")

fig = go.Figure()
# Structures without uncertainty first, so they stay underneath.
fig.add_trace(go.Scatter(
    x=coords[~has_unc, 0], y=coords[~has_unc, 1],
    mode="markers",
    marker=dict(color="lightgrey", size=6, opacity=0.5),
    text=labels[~has_unc],
    hovertemplate="%{text}<extra>no uncertainty</extra>",
    name="not evaluated",
))
fig.add_trace(go.Scatter(
    x=coords[has_unc, 0], y=coords[has_unc, 1],
    mode="markers",
    marker=dict(
        color=uncertainty[has_unc],
        colorscale="Viridis",
        size=9,
        line=dict(width=0.5, color="black"),
        colorbar=dict(title=f"{STAT}"),
        showscale=True,
    ),
    text=labels[has_unc],
    customdata=uncertainty[has_unc],
    hovertemplate="%{text}<br>uncertainty = %{customdata:.4g} eV<extra></extra>",
    name="evaluated",
))

selected_file = SELECTION_DIR / "selected_labels.txt"
if SHOW_SELECTED and selected_file.exists():
    selected = set(selected_file.read_text().split())
    is_selected = np.array([lbl in selected for lbl in labels])
    fig.add_trace(go.Scatter(
        x=coords[is_selected, 0], y=coords[is_selected, 1],
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=2, color="crimson")),
        text=labels[is_selected],
        hovertemplate="%{text}<extra>selected</extra>",
        name="selected",
    ))

fig.update_layout(
    title="Ensemble band uncertainty on the AMD/MDS map",
    xaxis_title="MDS 1",
    yaxis_title="MDS 2",
    template="plotly_white",
)
fig.write_html(OUTPUT_HTML)
print(f"Map written to {OUTPUT_HTML}")
