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

Outputs always include written Hamiltonian and overlap files and return a serializable result object with paths and metadata.

## DFT Workflows (`dft_workflows`)

The `hamilflow.dft_workflows` subpackage provides workflow helpers for FHI-aims data generation and collection.

- Main entry point: `hamilflow.dft_workflows.GenerateAimsDFTData`
- Supports two modes:
	- Run-and-collect: generate new AIMS runs from structure folders and collect outputs.
	- Collect-only: collect existing AIMS run directories into a single organized root.
- Optional conversion: provide `aims_to_deeph_config` to append an AIMS-to-DeepH conversion step after collection.
- Projection-only flow: `hamilflow.dft_workflows.GenerateProjectedDeephInputs` runs one projection job per DeepH subdirectory.
- End-to-end wrapper: `hamilflow.dft_workflows.GenerateAimsToProjectedDeephData` chains DFT/collection/conversion with projection.

Collected run folders preserve structure-oriented naming to keep downstream mapping explicit.

## Band Structure Utilities (`band_structures`)

Band-related utilities are grouped under `hamilflow.band_structures`:

- `hamilflow.band_structures.band_calculation`: build k-path configurations, load Hamiltonians, and plot bands.
- `hamilflow.band_structures.band_analysis`: analyze computed bands (gaps, shifts, comparisons, and k-point corrections).

## Examples

The [examples/](examples) folder contains runnable scripts for common usage patterns:

- [examples/projection/](examples/projection): projection workflows, including single-task, batch, and SLURM submission examples.
- [examples/band_structure/](examples/band_structure): band-structure workflows, including local runs and SLURM-oriented scripts.

These examples are the fastest way to see expected inputs, output layout, and end-to-end execution patterns.
