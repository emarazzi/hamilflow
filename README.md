# hamilflow

Community-maintained utilities around [DeepH](https://github.com/kYangLi/DeepH-dock) workflows: from picking training structures, through FHI-aims data generation and basis projection, to uncertainty estimates on model predictions. Not an official DeepH package.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.12. `deepx-dock` is installed from its GitHub repository.

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
- `hamilflow.train_infer_config`: writes DeepH training and inference TOML configs from dataclasses.
