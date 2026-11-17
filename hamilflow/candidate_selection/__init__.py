"""Structure candidate selection from AMD pairwise distances.

Provides:
- ``CandidateSelector``: compute AMDs + the AMD distance matrix for a
  trajectory, then select representative structures via farthest-point
  sampling, k-means centroids, or a regex pattern on
  ``atoms.info["index"]`` -- with optional pattern/exclude restriction of
  the candidate pool; ``CandidateSelector.quota_selection`` for per-group
  quotas.
- ``SelectionResult``: the output of ``CandidateSelector.select``.
- ``allocate_quotas``, ``select_by_group_quota``, ``GroupQuotaSelection``:
  split a selection budget over user-defined groups (weights, minimum,
  fixed quotas, capacities; largest-remainder rounding), optionally keep the
  top-scoring candidates of each group (e.g. by uncertainty), then pick one
  structure per k-means cluster within each group.
- ``fit_mds``, ``plot_mds``: 2D MDS embedding and plotting helpers, also
  exposed as ``CandidateSelector.fit_mds``/``CandidateSelector.plot``.
  ``CandidateSelector.mds_coords`` returns the raw coords + selected-mask
  for callers who want to build their own plot instead.
"""

from .plotting import fit_mds, plot_mds
from .quota import GroupQuotaSelection, allocate_quotas, select_by_group_quota
from .selector import CandidateSelector, SelectionResult

__all__ = [
    "CandidateSelector",
    "GroupQuotaSelection",
    "SelectionResult",
    "allocate_quotas",
    "fit_mds",
    "plot_mds",
    "select_by_group_quota",
]
