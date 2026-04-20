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
