"""Structure candidate selection from AMD pairwise distances.

Provides:
- ``CandidateSelector``: compute AMDs + the AMD distance matrix for a
  trajectory, then select representative structures via farthest-point
  sampling, k-means centroids, or a regex pattern on
  ``atoms.info["index"]`` -- with optional pattern/exclude restriction of
  the candidate pool.
- ``SelectionResult``: the output of ``CandidateSelector.select``.
- ``fit_mds``, ``plot_mds``: 2D MDS embedding and plotting helpers, also
  exposed as ``CandidateSelector.fit_mds``/``CandidateSelector.plot``.
  ``CandidateSelector.mds_coords`` returns the raw coords + selected-mask
  for callers who want to build their own plot instead.
"""

from .selector import CandidateSelector, SelectionResult
from .plotting import fit_mds, plot_mds

__all__ = ["CandidateSelector", "SelectionResult", "fit_mds", "plot_mds"]
