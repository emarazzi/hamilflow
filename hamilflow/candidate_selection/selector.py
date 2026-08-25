from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .plotting import fit_mds, plot_mds


@dataclass
class SelectionResult:
    """Output of :meth:`CandidateSelector.select`.

    ``candidate_idx`` is the pool the selection algorithm actually ran over
    (post pattern/exclude filtering); ``selected_idx``/``selected_labels``
    index into the full ``frame_indices``/``dist_matrix``/``amds``.
    """

    method: str
    frame_indices: list[str]
    dist_matrix: np.ndarray
    amds: list
    candidate_idx: np.ndarray
    selected_idx: np.ndarray

    @property
    def selected_labels(self) -> list[str]:
        return [self.frame_indices[i] for i in self.selected_idx]

    def write_labels(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(self.selected_labels))


@dataclass
class CandidateSelector:
    """Select representative structures from a trajectory via AMD pairwise
    distances, using farthest-point sampling (FPS), k-means centroids, or a
    regex pattern on ``atoms.info["index"]``.

    The candidate pool fed to FPS/k-means can be restricted with a pattern
    and/or an exclude list, while the AMD distance matrix and MDS map always
    cover the full dataset.

    Example usage:
      selector = CandidateSelector(k_neighbors=100)
      result = selector.select("dataset.extxyz", n=20, method="kmeans")
      result.write_labels("selected.txt")
      selector.plot(result, save_path="map.html")
    """

    k_neighbors: int = 100
    seed: int = 42

    # -- loading / AMD computation --------------------------------------

    def load_trajectory(self, input_file: str | Path):
        """Read a trajectory and return ``(traj, frame_indices)``, where
        ``frame_indices`` are the ``atoms.info["index"]`` labels."""
        from ase.io import read

        traj = read(str(input_file), index=":")
        frame_indices = [atoms.info["index"] for atoms in traj]
        return traj, frame_indices

    def compute_amds(self, traj: Iterable) -> list:
        """Compute an AMD descriptor per frame."""
        import amd

        return [
            amd.AMD(amd.PeriodicSet(a.get_positions(), a.get_cell()[:]), k=self.k_neighbors)
            for a in traj
        ]

    def compute_distance_matrix(self, amds: Sequence) -> np.ndarray:
        """Compute the pairwise AMD distance matrix (AMD_pdist), as a dense
        square matrix."""
        import amd
        from scipy.spatial.distance import squareform

        dist_condensed = amd.AMD_pdist(amds)
        return squareform(dist_condensed)

    # -- candidate-pool filtering ----------------------------------------

    def pattern_selection(self, frame_indices: Sequence[str], pattern: str) -> np.ndarray:
        """Return indices of all frames whose label matches a regex."""
        regex = re.compile(pattern)
        sel_idx = np.array([i for i, lbl in enumerate(frame_indices) if regex.search(lbl)])
        if len(sel_idx) == 0:
            raise ValueError(f"No frame_indices matched pattern '{pattern}'")
        return sel_idx

    def exclude_selection(
        self,
        candidate_idx: np.ndarray,
        frame_indices: Sequence[str],
        exclude: Iterable[str],
    ) -> np.ndarray:
        """Drop labels in ``exclude`` from a candidate index pool. Structures
        excluded here stay in the full dataset used for the distance matrix
        and MDS map -- only the FPS/k-means candidate pool shrinks."""
        exclude_labels = set(exclude)
        missing = exclude_labels - set(frame_indices)
        if missing:
            print(
                f"Warning: {len(missing)} excluded label(s) not found in dataset: "
                f"{sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}"
            )
        keep_mask = np.array([frame_indices[i] not in exclude_labels for i in candidate_idx])
        n_before = len(candidate_idx)
        candidate_idx = candidate_idx[keep_mask]
        print(
            f"Excluded {n_before - len(candidate_idx)} structure(s) from candidate pool "
            f"({len(candidate_idx)} remain)"
        )
        return candidate_idx

    @staticmethod
    def load_exclude_list(path: str | Path) -> set[str]:
        """One label per line -> set of labels to exclude."""
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}

    # -- selection algorithms --------------------------------------------

    def farthest_point_sampling(
        self,
        dist_matrix: np.ndarray,
        n_samples: int,
        first_point: int | None = None,
    ) -> np.ndarray:
        n = dist_matrix.shape[0]
        if n_samples > n:
            raise ValueError(f"n_samples ({n_samples}) exceeds number of frames ({n})")
        if first_point is None:
            rng = np.random.default_rng(self.seed)
            first_point = int(rng.integers(n))
        selected = [first_point]
        dist = dist_matrix[first_point].copy()
        for _ in range(n_samples - 1):
            next_idx = int(np.argmax(dist))
            selected.append(next_idx)
            dist = np.minimum(dist, dist_matrix[next_idx])
        return np.array(selected)

    def kmeans_selection(self, amd_vectors: Sequence, n_samples: int) -> np.ndarray:
        """Cluster AMD feature vectors with k-means, pick the frame closest
        to each cluster centroid as the representative point."""
        from sklearn.cluster import KMeans

        n = len(amd_vectors)
        if n_samples > n:
            raise ValueError(f"n_samples ({n_samples}) exceeds number of frames ({n})")

        X = np.asarray(amd_vectors)
        km = KMeans(n_clusters=n_samples, random_state=self.seed, n_init=10).fit(X)

        selected = []
        for c in range(n_samples):
            cluster_idx = np.where(km.labels_ == c)[0]
            if len(cluster_idx) == 0:
                continue  # empty cluster, can happen with n_init/degenerate data
            d = np.linalg.norm(X[cluster_idx] - km.cluster_centers_[c], axis=1)
            selected.append(int(cluster_idx[np.argmin(d)]))

        return np.array(selected)

    # -- high-level orchestration -----------------------------------------

    def select(
        self,
        input_file: str | Path,
        n: int | None = None,
        method: str = "fps",
        pattern: str | None = None,
        exclude: Iterable[str] | str | Path | None = None,
        first_point: str | None = None,
    ) -> SelectionResult:
        """Compute AMDs + the AMD distance matrix for the full trajectory,
        then select representative structures.

        - ``method``: "fps", "kmeans", or "pattern" (select every match of
          ``pattern`` directly, ignoring ``n``/``exclude``).
        - ``pattern``: for fps/kmeans, restricts the *candidate pool* to
          labels matching this regex before running the algorithm; the
          returned distance matrix/AMDs still cover the full dataset.
        - ``exclude``: an iterable of labels, or a path to a file with one
          label per line (see :meth:`load_exclude_list`); ignored for
          "pattern".
        - ``first_point``: label to start FPS from; ignored for kmeans and
          pattern. Must itself match ``pattern`` if both are given.
        """
        if method not in ("fps", "kmeans", "pattern"):
            raise ValueError(f"Unknown method '{method}' (expected fps/kmeans/pattern)")
        if method == "pattern" and pattern is None:
            raise ValueError("method='pattern' requires a pattern")
        if method != "pattern" and n is None:
            raise ValueError("n is required for method='fps'/'kmeans'")
        if method == "kmeans" and first_point is not None:
            print("Warning: first_point is ignored when method='kmeans'")
        if method == "pattern" and exclude is not None:
            print("Warning: exclude is ignored when method='pattern'")

        traj, frame_indices = self.load_trajectory(input_file)
        amds = self.compute_amds(traj)
        dist_matrix = self.compute_distance_matrix(amds)
        n_frames = len(frame_indices)

        if method == "pattern":
            sel_idx = self.pattern_selection(frame_indices, pattern)
            candidate_idx = sel_idx
        else:
            if pattern is not None:
                candidate_idx = self.pattern_selection(frame_indices, pattern)
                print(
                    f"Pattern '{pattern}' restricts candidates to "
                    f"{len(candidate_idx)} / {n_frames} structures"
                )
            else:
                candidate_idx = np.arange(n_frames)

            if exclude is not None:
                exclude_labels = (
                    self.load_exclude_list(exclude) if isinstance(exclude, (str, Path)) else set(exclude)
                )
                candidate_idx = self.exclude_selection(candidate_idx, frame_indices, exclude_labels)

            if method == "fps":
                sub_dist = dist_matrix[np.ix_(candidate_idx, candidate_idx)]
                first_point_idx = None
                if first_point is not None:
                    if first_point not in frame_indices:
                        raise ValueError(f"'{first_point}' not found in frame_indices")
                    global_fp = frame_indices.index(first_point)
                    local_matches = np.where(candidate_idx == global_fp)[0]
                    if len(local_matches) == 0:
                        raise ValueError(
                            f"first_point '{first_point}' does not match pattern '{pattern}'"
                        )
                    first_point_idx = int(local_matches[0])
                local_sel = self.farthest_point_sampling(sub_dist, n, first_point=first_point_idx)
            else:  # kmeans
                sub_amds = [amds[i] for i in candidate_idx]
                local_sel = self.kmeans_selection(sub_amds, n)

            sel_idx = candidate_idx[local_sel]  # map back to global indices

        return SelectionResult(
            method=method,
            frame_indices=frame_indices,
            dist_matrix=dist_matrix,
            amds=amds,
            candidate_idx=candidate_idx,
            selected_idx=sel_idx,
        )

    # -- plotting ----------------------------------------------------------

    def fit_mds(self, dist_matrix: np.ndarray) -> np.ndarray:
        return fit_mds(dist_matrix, seed=self.seed)

    def mds_coords(
        self,
        result: SelectionResult,
        *,
        coords: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(coords, is_selected)`` for ``result`` without building a
        figure, for callers who'd rather plot the MDS embedding themselves
        (custom marker colors/sizes, extra traces, a different library...).

        ``coords`` is an ``(n_frames, 2)`` array (fitted here via
        :meth:`fit_mds` if not passed in); ``is_selected`` is a boolean mask
        over ``result.frame_indices``, aligned row-for-row with ``coords``.
        """
        if coords is None:
            coords = self.fit_mds(result.dist_matrix)
        is_selected = np.zeros(len(result.frame_indices), dtype=bool)
        is_selected[result.selected_idx] = True
        return coords, is_selected

    def plot(
        self,
        result: SelectionResult,
        *,
        coords: np.ndarray | None = None,
        show_fig: bool = False,
        save_path: str | Path | None = None,
        customize: Callable[["go.Figure"], "go.Figure | None"] | None = None,
    ):
        """Plot the 2D MDS embedding of ``result``'s distance matrix,
        coloring selected vs. non-selected structures. Pass ``customize`` to
        tweak the figure (colors, titles, extra traces, ...) before it is
        shown/saved; see :func:`hamilflow.candidate_selection.plotting.plot_mds`.

        Returns ``(fig, coords)``.
        """
        return plot_mds(
            result.dist_matrix,
            result.frame_indices,
            result.selected_idx,
            coords=coords,
            seed=self.seed,
            show_fig=show_fig,
            save_path=save_path,
            customize=customize,
        )

    def save_state(
        self,
        path: str | Path,
        result: SelectionResult,
        coords: np.ndarray,
    ) -> None:
        """Save AMDs, distance matrix, frame labels and MDS coords to an
        .npz file, so later structures can be added to the same 2D map."""
        np.savez(
            str(path),
            amds=np.array(result.amds),
            dist_matrix=result.dist_matrix,
            frame_indices=np.array(result.frame_indices, dtype=object),
            coords=coords,
            k_neighbors=self.k_neighbors,
        )


__all__ = ["CandidateSelector", "SelectionResult"]
