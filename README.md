# hamilflow

Community-maintained utilities around [DeepH](https://github.com/kYangLi/DeepH-dock) workflows: from picking training structures, through FHI-aims data generation and basis projection, to uncertainty estimates on model predictions. Not an official DeepH package.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.12. `deepx-dock` is installed from its GitHub repository.

The dense eigensolvers (band structures, band uncertainty, orbital removal) run on a GPU when
CuPy is installed and a CUDA device is visible:

```bash
pip install -e .[gpu]
```

**This installs `cupy-cuda12x`, built for CUDA 12.** If the CUDA of your machine or cluster
module is another major version, replace it with the matching wheel, e.g. for CUDA 11:

```bash
pip uninstall cupy-cuda12x && pip install cupy-cuda11x
```

Check with `python -c "import cupy; cupy.show_config()"`; hamilflow also warns at the first
diagonalization if CuPy is installed but cannot use the GPU, and then runs on the CPU. Set
`HAMILFLOW_GPU=0` to always use the CPU, or `HAMILFLOW_GPU=1` to fail instead of falling back
(see `hamilflow/sparse_hamiltonian/dense_backend.py`).

## Components

### Candidate selection — `hamilflow.candidate_selection`

`CandidateSelector` picks representative structures from a trajectory using Average Minimum Distance (AMD) fingerprints. It selects by farthest-point sampling, k-means centroids, or a regex on frame labels, and can draw a 2D MDS map of the dataset. See [examples/candidate_selection/](examples/candidate_selection).

### DFT workflows — `hamilflow.dft_workflows`

jobflow/atomate2 flows that produce DeepH-ready data from FHI-aims, one job per structure:

- `GenerateAimsDFTData`: runs AIMS calculations (or collects existing ones), with optional AIMS → DeepH conversion.
- `GenerateProjectedDeephInputs`, `GenerateAimsToProjectedDeephData`: chain DFT, conversion and projection end to end. `GenerateOvlOnlyAimsToProjectedDeephData` does the same for overlap only.
- Helpers: `get_ksampling` for k-point settings.

### Projection — `hamilflow.projection`

`run_projection(ProjectionConfig)` removes orbitals from a DeepH Hamiltonian and overlap with a k-space Schur-complement reduction, following a `RemovalPlan` (JSON, dict or object). The reduction runs on the sparse `hamilflow.sparse_hamiltonian` objects, so it never builds the dense real-space arrays and works on large structures. Overlap-only and dummy-Hamiltonian modes are supported. See [examples/projection/](examples/projection).

### Uncertainty quantification — `hamilflow.uncertainty`

Tools that estimate how uncertain an ensemble of DeepH predictions is, both on the Hamiltonian matrices and on the band structures, measured against DFT references. This module is still changing quickly, so expect its API to move. See [examples/uncertainty/](examples/uncertainty).

### Other utilities

- `hamilflow.band_structures`: k-path setup, band calculation and plotting, gap and shift analysis.
- `hamilflow.training`: writes DeepH training and inference TOML configs from dataclasses; analyzes training logs (LR reductions, convergence, `patience` and `cooldown` suggestions).

## Testing

```bash
pip install -e .[test]
pytest
```

The suite takes about 15 seconds and little memory: it builds small synthetic structures on the fly (see
`tests/conftest.py`), so it needs no DFT data, no FHI-aims and no GPU. `tests/` mirrors the package.

```bash
pytest tests/projection                  # one part of the package
pytest -m "not slow"                     # skip the slowest tests
pytest --cov=hamilflow --cov-report=term # with coverage
```

Tests that need a GPU are marked `gpu` and skipped when none is available. GitHub Actions runs `ruff check .` and
`pytest` on every push; its runners have no GPU, so the GPU code paths have to be checked by hand on a machine with one.
`dft_workflows` is only lightly tested so far.
