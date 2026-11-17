"""Select N structures split over groups by quota, diverse within each group.

Useful when the candidate pool is made of several families (e.g. MD runs of
different parent structures, compositions or temperatures) and one family
would otherwise take most of the selection, or some families matter more.

1. Groups: every frame gets a group label from its ``atoms.info["index"]``
   label with GROUP_REGEX (the first capture group), e.g.
   ``"cellA-300K-0042" -> "cellA"``.
2. Quotas: N_SELECT is split over the groups in proportion to WEIGHTS
   (default: equal), at least MIN_PER_GROUP each, FIXED overrides single
   groups; rounding by largest remainder.
3. Score filter (optional): with UNCERTAINTY_FILE, keep within each group the
   TOP_FRACTION of frames with the highest ensemble band uncertainty
   (weighted quantile from ``uncertainty/ensemble_band_uncertainty.py``).
4. Diversity: k-means (k = quota) on the AMDs of each group's candidates,
   one frame per cluster (closest to the centroid, or the highest uncertainty
   with PICK = "max_score").

Outputs, in OUTPUT_DIR:
  * quota_selected_labels.txt   one selected label per line
  * quota_selected.extxyz       the selected structures
  * printed per-group summary (weight, quota, available, candidates, selected)

Run with:  python select_by_group_quota.py
"""

import logging
import re
from pathlib import Path

import numpy as np
from ase.io import read, write

from hamilflow.candidate_selection import CandidateSelector

# --- User settings -----------------------------------------------------------
INPUT_FILE = Path("/path/to/your/pool.extxyz")
OUTPUT_DIR = Path("/path/to/your/selection/folder")
N_SELECT = 20
GROUP_REGEX = r"^(.+)-\d+K?-\d+$"  # first capture group of the label = group
WEIGHTS = None  # e.g. {"cellA": 3, "cellB": 1}; None = equal weights
FIXED = None  # e.g. {"cellC": 5}: exact quota for some groups; None = none
MIN_PER_GROUP = 1
UNCERTAINTY_FILE = None  # e.g. Path("/path/to/uncertainty.json"); None = no score filter
TOP_FRACTION = 0.3  # with UNCERTAINTY_FILE: fraction of each group kept by score
E_CUT, TAU, QUANTILE = 2.0, 0.1, 0.9  # uncertainty statistic (weighted quantile)
PICK = "centroid"  # or "max_score" (needs UNCERTAINTY_FILE)
K_NEIGHBORS = 100  # length of the AMD vectors
SEED = 42
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

traj = read(str(INPUT_FILE), index=":")
labels = [a.info["index"] for a in traj]
groups = []
for lab in labels:
    m = re.match(GROUP_REGEX, lab)
    if m is None:
        raise SystemExit(f"label {lab!r} does not match GROUP_REGEX {GROUP_REGEX!r}")
    groups.append(m.group(1))

scores = None
if UNCERTAINTY_FILE is not None:
    from hamilflow.uncertainty.band_postprocess import BandUncertaintyPostProcessor

    pp = BandUncertaintyPostProcessor.from_json(str(UNCERTAINTY_FILE))
    summary = pp.weighted_summary(e_cut=E_CUT, tau=TAU, quantile=QUANTILE)
    # frames without an uncertainty are left out of the pool
    keep = [i for i, lab in enumerate(labels) if lab in summary]
    if len(keep) < len(labels):
        print(f"{len(labels) - len(keep)} frames have no uncertainty and are skipped")
    traj = [traj[i] for i in keep]
    labels = [labels[i] for i in keep]
    groups = [groups[i] for i in keep]
    scores = np.array([summary[lab]["weighted_quantile_eV"] for lab in labels])

selector = CandidateSelector(k_neighbors=K_NEIGHBORS, seed=SEED)
amds = np.asarray(selector.compute_amds(traj))
result = selector.quota_selection(
    amds,
    groups,
    N_SELECT,
    weights=WEIGHTS,
    fixed=FIXED,
    min_per_group=MIN_PER_GROUP,
    scores=scores,
    top_fraction=TOP_FRACTION if scores is not None else None,
    pick=PICK,
)

print(f"{'group':30s} {'weight':>6s} {'quota':>5s} {'avail':>5s} {'cand':>5s} {'sel':>4s}")
for row in result.summary():
    print(
        f"{row['group']!s:30s} {row['weight']:6g} {row['quota']:5d} "
        f"{row['available']:5d} {row['candidates']:5d} {row['selected']:4d}"
    )

sel = [int(i) for i in result.selected_idx]
(OUTPUT_DIR / "quota_selected_labels.txt").write_text("\n".join(labels[i] for i in sel) + "\n")
write(str(OUTPUT_DIR / "quota_selected.extxyz"), [traj[i] for i in sel], format="extxyz")
print(f"wrote {len(sel)} structures to {OUTPUT_DIR}")
