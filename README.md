# hamilflow

Community-maintained utilities for DeepH workflows (not an official DeepH package).

## Core API

- `hamilflow.run_projection`
- `hamilflow.ProjectionConfig`
- `hamilflow.RemovalPlan` / `hamilflow.RemovalRule`

The projection accepts removal plans as:

- JSON file path
- Python dict/list payload
- Prebuilt `RemovalPlan` instance

`ProjectionConfig` also controls the k→R reduction itself: `reduction_mode` ("schur"), `overlap_only` to skip the Hamiltonian entirely, `write_dummy_hamiltonian`, and `n_workers` for the thread pool used during k-point elimination. Outputs always include written Hamiltonian and overlap files and return a serializable `ProjectionResult` with paths and metadata.

## Sparse Hamiltonian (`sparse_hamiltonian`)

`hamilflow.sparse_hamiltonian` provides a memory-efficient, drop-in counterpart to `deepx_dock`'s dense `HamiltonianObj`/`AOMatrixObj`:

- `hamilflow.sparse_hamiltonian.SparseAOMatrixObj`: keeps a matrix in its on-disk sparse block form and only ever densifies a single k-point at a time, avoiding the `R_quantity * Nb^2` dense allocation that OOMs on large structures.
- `hamilflow.sparse_hamiltonian.SparseHamiltonianObj`: the Hamiltonian-side counterpart, pairing sparse Hamiltonian and overlap blocks; exposes `Sk_and_Hk(ks)` / `diag(ks, ...)` but never materializes a dense real-space `HR`/`SR` array. Supports `load_hamiltonian=False` for overlap-only workflows that never touch Hamiltonian values.

The projection core streams k→R reduction directly against sparse objects, so large datasets that would OOM the dense pipeline can be projected in place. See [examples/projection/compare_sparse_vs_dense.py](examples/projection/compare_sparse_vs_dense.py) for a numerically-verified peak-memory/time comparison against the original dense pipeline.

## DFT Workflows (`dft_workflows`)

The `hamilflow.dft_workflows` subpackage provides workflow helpers for FHI-aims data generation and collection.

- Main entry point: `hamilflow.dft_workflows.GenerateAimsDFTData`
- Supports two modes:
	- Run-and-collect: generate new AIMS runs from structure folders and collect outputs.
	- Collect-only: collect existing AIMS run directories into a single organized root.
- K-point handling can be provided either as a plain `kgrid` or via `hamilflow.dft_workflows.get_ksampling`-style sampling settings that are resolved into `k_grid` for AIMS inputs.
- Optional conversion: provide `aims_to_deeph_config` to append an AIMS-to-DeepH conversion step after collection.
- Projection-only flow: `hamilflow.dft_workflows.GenerateProjectedDeephInputs` runs one projection job per DeepH subdirectory.
- End-to-end wrapper: `hamilflow.dft_workflows.GenerateAimsToProjectedDeephData` chains DFT/collection/conversion with projection, and can optionally add a second truncation projection stage.
- Overlap-only end-to-end wrapper: `hamilflow.dft_workflows.GenerateOvlOnlyAimsToProjectedDeephData`.
- Structure generation: `hamilflow.dft_workflows.generate_perturbed_population` creates `structure_0..structure_{N-1}` folders from a base `pymatgen` structure (`structure_0` unperturbed, the rest perturbed) for ensemble-style training/uncertainty datasets.

Collected run folders preserve structure-oriented naming to keep downstream mapping explicit.

## Band Structure Utilities (`band_structures`)

Band-related utilities are grouped under `hamilflow.band_structures`:

- `hamilflow.band_structures.band_calculation`: build k-path configurations, load Hamiltonians, and plot bands.
- `hamilflow.band_structures.band_analysis`: analyze computed bands (gaps, shifts, comparisons, and k-point corrections).

## Uncertainty (`uncertainty`)

`hamilflow.uncertainty` estimates band-energy uncertainty across an ensemble of model predictions:

- `hamilflow.uncertainty.BandUncertaintyCalculator`: given a set of per-model prediction directories and a DFT reference directory, builds an irreducible k-point mesh (via `spglib`), aligns bands to mid-gap, and computes per-band/per-k uncertainty across the ensemble.
- `hamilflow.uncertainty.discover_structures` / `hamilflow.uncertainty.link_ensemble_files`: glob-discover structure folders and assemble an ensemble working directory (symlinking shared inputs, copying predicted Hamiltonians where needed).
- `hamilflow.uncertainty.read_deeph_hamiltonian` / `write_deeph_hamiltonian` / `average_predicted_hamiltonians`: read/write DeepH-style Hamiltonian HDF5 files and average predictions across an ensemble.

## Examples

The [examples/](examples) folder contains runnable scripts for common usage patterns:

- [examples/projection/](examples/projection): projection workflows, including single-task, batch, overlap-only, dense-vs-sparse comparison, and SLURM submission examples.
- [examples/band_structure/](examples/band_structure): band-structure workflows, including local runs and SLURM-oriented scripts.
- [examples/uncertainty/](examples/uncertainty): ensemble discovery, linking, and band-uncertainty computation.

These examples are the fastest way to see expected inputs, output layout, and end-to-end execution patterns.
