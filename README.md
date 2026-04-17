# deepx-postdock

Initial package scaffold for DeepX post-processing utilities and k->R reduction pipeline.

## Core API

- `deepx_postdock.run_pipeline`
- `deepx_postdock.PipelineConfig`
- `deepx_postdock.RemovalPlan` / `deepx_postdock.RemovalRule`

The pipeline accepts removal plans as:

- JSON file path
- Python dict/list payload
- Prebuilt `RemovalPlan` instance

Outputs always include written Hamiltonian and overlap files and return a serializable result object with paths and metadata.
