# Examples

User-facing scripts for the hamilflow components. Each script starts with a
docstring describing what it does, and most have a "User settings" section
(paths, SLURM resources, numerical options) to edit before running. The
folders follow the usual order of a DeepH active-learning loop:

    candidate_selection -> dft_workflows -> (projection) -> training -> uncertainty -> candidate_selection
                                                                    \-> band_structure

## `candidate_selection/`

Pick which structures to compute with DFT.

- `select_candidates.py`: selects N representative structures from an
  extended-XYZ dataset using Average Minimum Distance (AMD) fingerprints, by
  farthest-point sampling, k-means, or a regex on the frame labels. Writes the
  selected structures (`selected.extxyz`, ready for `dft_workflows/`) and an
  interactive 2D MDS map of the dataset.
- `uncertainty_map.py`: colors that 2D map by the ensemble band uncertainty
  from `uncertainty/ensemble_band_uncertainty.py`, to see which regions of
  structure space the models are unsure about.

## `dft_workflows/`

jobflow-remote flows that go from structures to projected DeepH training or
inference inputs with FHI-aims.

- `aims_to_projected_deeph.py`: full chain, per structure: AIMS SCF ->
  AIMS-to-DeepH conversion -> Schur projection -> truncation projection.
- `aims_overlap_only_to_projected_deeph.py`: same chain for the overlap matrix
  only (short, loosely converged AIMS runs), e.g. to prepare inputs for DeepH
  inference on new structures.
- `removal_plan_schur.json`, `removal_plan_truncate.json`: example removal
  plans for MoS2 (core shells folded out with the Schur complement, then
  high-l polarisation shells truncated). Their `_comment` keys explain the
  plan format and how to adapt it to other elements and basis sets.

## `projection/`

The same two-step projection, run on DeepH-format data that is already on
disk (e.g. the `deeph_full/` folder written by `dft_workflows/`).

- `two_step_projection.py`: Schur projection followed by truncation, for H and S.
- `two_step_projection_overlap_only.py`: the same for S only.
- `removal_plan_schur.json`, `removal_plan_truncate.json`: copies of the
  example plans in `dft_workflows/`.

The scripts in `dft_workflows/` and `projection/` read the removal plans from
their `WORKDIR`, so copy the plans there and edit them for your system.

## `uncertainty/`

Uncertainty of an ensemble of DeepH models (e.g. from k-fold training) that
ran inference on the same structures.

- `ensemble_band_uncertainty.py`: spread of the band energies across the
  ensemble, per structure and k-point. Needs no DFT reference, so it can drive
  the selection of new structures. The "projected" method diagonalizes only the
  ensemble-averaged Hamiltonian and works in a band window around the gap.
- `ensemble_error_vs_dft.py`: error of the ensemble-averaged Hamiltonian's
  bands against DFT, on structures with a reference, to check how well the
  ensemble spread follows the true error.

## `band_structure/`

Band structures along a k-path from DeepH-format Hamiltonians (DFT or
predicted).

- `band_structure_example.py`: one structure, computes and plots the bands.
- `band_structure_batch_example.py`: every `structure_*` subfolder of a parent
  directory, in parallel processes.
- `band_structure_single_task.py` + `band_structure_slurm.sh`: one structure
  per SLURM array task, for large batches on a cluster.
