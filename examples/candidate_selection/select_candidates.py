"""Select representative structures from a dataset via AMD distances.

Each structure is described by its Average Minimum Distance (AMD) vector;
from the pairwise AMD distances, N structures are picked with:

  * "fps":     farthest-point sampling (maximally diverse set);
  * "kmeans":  the structure closest to each of N k-means centroids
               (representative of the density of the dataset);
  * "pattern": every structure whose label matches a regex (N is ignored).

The candidate pool can be restricted with PATTERN (for fps/kmeans) and/or an
EXCLUDE_FILE listing labels to skip, e.g. structures already computed with DFT.
Distances and the 2D map always cover the full dataset.

Input: an extended-XYZ file where every frame has a unique label in
``atoms.info["index"]``.

Outputs, in OUTPUT_DIR:
  * selected_labels.txt    one selected label per line
  * selected.extxyz        the selected structures, ready for
                           ``dft_workflows/aims_to_projected_deeph.py``
  * selection_state.npz    AMDs, distance matrix, labels and 2D MDS
                           coordinates, reused by ``uncertainty_map.py``
  * selection_map.html     interactive 2D MDS map of the selection
  * select_candidates.log  hamilflow's progress messages and warnings

The MDS embedding is slow for large datasets; it is computed once and read
back from ``selection_state.npz`` on later runs over the same dataset.

Run with:  python select_candidates.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
from ase.io import write

from hamilflow.candidate_selection import CandidateSelector

# --- User settings -----------------------------------------------------------
INPUT_FILE = Path("/path/to/your/dataset.extxyz")
OUTPUT_DIR = Path("/path/to/your/selection/folder")
N_SELECT = 50
METHOD = "kmeans"  # "fps", "kmeans" or "pattern"
PATTERN = None  # regex on the labels, e.g. r"^twisted_"; None = all structures
EXCLUDE_FILE = None  # e.g. Path("/path/to/your/already_computed.txt"); None = no exclusion
FIRST_POINT = None  # fps only: label to start from; None = random (seeded)
K_NEIGHBORS = 100  # length of the AMD vectors
# -----------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Show hamilflow's progress messages on stdout and save them, with any
# warnings, to OUTPUT_DIR/select_candidates.log.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUTPUT_DIR / "select_candidates.log")],
)
logging.captureWarnings(True)
state_path = OUTPUT_DIR / "selection_state.npz"

selector = CandidateSelector(k_neighbors=K_NEIGHBORS)
result = selector.select(
    input_file=INPUT_FILE,
    n=N_SELECT,
    method=METHOD,
    pattern=PATTERN,
    exclude=EXCLUDE_FILE,
    first_point=FIRST_POINT,
)
print(f"Selected {len(result.selected_labels)} structures: {result.selected_labels}")

result.write_labels(OUTPUT_DIR / "selected_labels.txt")
traj, _ = selector.load_trajectory(INPUT_FILE)
write(OUTPUT_DIR / "selected.extxyz", [traj[i] for i in result.selected_idx])

# Reuse the cached MDS coordinates if they were computed for the same dataset.
coords = None
if state_path.exists():
    state = np.load(state_path, allow_pickle=True)
    if list(state["frame_indices"]) == result.frame_indices:
        coords = state["coords"]
if coords is None:
    coords = selector.fit_mds(result.dist_matrix)
selector.save_state(state_path, result, coords)

selector.plot(result, coords=coords, save_path=OUTPUT_DIR / "selection_map.html")
