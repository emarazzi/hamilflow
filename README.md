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

## Band Structure Calculation (`band_structure`)

The `hamilflow.band_structure` module provides utilities to build k-paths and compute band structures from Hamiltonian/overlap data. It is intended for scripted workflows where you want reproducible, file-driven post-processing.

## Band Structure Analysis (`band_analysis`)

The `hamilflow.band_analysis` module provides analysis helpers for computed bands (for example, extracting summary quantities and comparing trends across runs). It is designed to work on top of the band-structure outputs and support downstream interpretation/automation.

## Examples

The [examples/](examples) folder contains runnable scripts for common usage patterns:

- [examples/projection/](examples/projection): projection workflows, including single-task, batch, and SLURM submission examples.
- [examples/band_structure/](examples/band_structure): band-structure workflows, including local runs and SLURM-oriented scripts.

These examples are the fastest way to see expected inputs, output layout, and end-to-end execution patterns.
