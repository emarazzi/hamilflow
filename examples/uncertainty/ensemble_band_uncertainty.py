"""Band-energy uncertainty of an ensemble of DeepH models.

Given N independently trained DeepH models (e.g. from k-fold training) that
all ran inference on the same structures, this script obtains every
member's band energies on a k-grid (see METHOD below) and writes, per structure and
k-point, the spread (sigma, eV) of the band energies across the ensemble.
No DFT reference Hamiltonian is needed, so it can be used to pick new
structures for active learning (see ``candidate_selection/uncertainty_map.py``).

Expected layout (edit the "User settings" section to match yours):

    /path/to/your/ensemble/folder/
        train_1/infer/outputs/<run>/dft/<structure>/hamiltonian_pred.h5
        train_2/infer/outputs/<run>/dft/<structure>/hamiltonian_pred.h5
        ...
    /path/to/your/inference/inputs/dft/<structure>/{POSCAR, info.json, overlap.h5}

``link_ensemble_files`` first prepares each model's output folder: it copies
``hamiltonian_pred.h5`` to ``hamiltonian.h5`` and links ``POSCAR``,
``info.json`` and ``overlap.h5`` from the inference inputs.

METHOD picks how the members' band energies are obtained:

- "exact": every member's Hamiltonian is diagonalized at every k-point, all
  bands are written.
- "projected": only the averaged Hamiltonian is diagonalized. Each member is
  projected onto the averaged Hamiltonian's eigenstates in a window of
  N_BANDS_BELOW valence and N_BANDS_ABOVE conduction bands around the gap,
  and its band energies are obtained by diagonalizing that small window
  block. That is one large diagonalization per structure instead of one per
  member, and the result matches "exact" closely as long as the members'
  errors are small compared to the distance to the bands outside the window.
  Only the window bands are written (same JSON layout; ``vbm_index`` counts
  from the window's first band, ``first_band_index`` gives its position).
  A member state that enters the window from outside it, such as a spurious
  in-gap state, is not seen, so keep ILL_METHOD = "orbital_removal". For
  band-edge-weighted statistics, the window should cover e_cut plus a few tau
  on both sides of the gap: WINDOW_E sets it by energy (the same distance from
  the VBM and from the CBM, like e_cut), so the number of bands follows each
  structure's band density (otherwise N_BANDS_BELOW / N_BANDS_ABOVE count bands). With DOWNFOLD the coupling to
  all the bands outside the window (up to tens of eV above the gap, which no
  window reaches) is added to the window block at second order; without it
  the window eigenvalues of a member that differs from the average by more
  than a rigid shift can come out too high by several meV.

The averaged Hamiltonian of each structure is written in AVERAGE_HAM_DIR (or
read from it, if already there) and is reused by ``ensemble_error_vs_dft.py``.
The output JSON is updated after every structure; rerunning skips structures
already in it. Summarize it with
``hamilflow.uncertainty.band_postprocess.BandUncertaintyPostProcessor``.

One worker process runs per allocated CPU. With CuPy installed (``pip install -e .[gpu]``,
or the cupy-cudaXXx wheel matching your CUDA) the orbital removal and the diagonalizations run
on the GPU, shared by all workers: request one GPU and a few CPUs (e.g. 8) per job, which then
do the rest (loading, projection). Set HAMILFLOW_GPU=1 to stop with an error rather than
silently falling back to the CPU if the GPU cannot be used, HAMILFLOW_GPU=0 to stay on the CPU.
``job_gpu.sh`` is a SLURM job script for this setup.

Run with:  python ensemble_band_uncertainty.py   (e.g. inside a SLURM job)
"""

import os
from pathlib import Path

from hamilflow.uncertainty.band_uncertainty import BandUncertaintyCalculator
from hamilflow.uncertainty.ensemble_io import link_ensemble_files

# --- User settings -----------------------------------------------------------
ENSEMBLE_ROOT = Path("/path/to/your/ensemble/folder")
N_MODELS = 5
# DeepH inference output of model i (the glob matches the date-stamped run folder).
MODEL_OUTPUT_GLOB = "train_{i}/infer/outputs/*/dft"
# DeepH inputs the models ran inference on (one subfolder per structure).
INFERENCE_INPUTS_DIR = Path("/path/to/your/inference/inputs/dft")
AVERAGE_HAM_DIR = ENSEMBLE_ROOT / "avg_ham"
STRUCTURE_PATTERN = "*"  # glob on the structure folder names, e.g. "twisted_*"
# k-point mesh, chosen per structure from its cell (see
# dft_workflows.kpoints.get_ksampling). Must resolve to a k_grid, e.g.
# {"reciprocal_density": 100}, {"grid_density": 1000} or {"k_grid": [3, 3, 1]}.
KPOINTS_SETTINGS = {"reciprocal_density": 100}
FORCE_2D = True  # set k_z = 1 (slabs / 2D materials)
# Treatment of an ill-conditioned overlap: "orbital_removal" removes orbitals
# until S(k) is well conditioned on the k-grid; None = no treatment.
ILL_METHOD = "orbital_removal"
# "exact" (diagonalize every member) or "projected" (see the docstring above).
METHOD = "exact"
# With METHOD = "projected": the window by energy (eV below the VBM and above the
# CBM; e_cut plus a few tau), or, with WINDOW_E = None, by number of valence bands
# (counting the VBM band) and conduction bands (counting the CBM band); None = all
# bands on that side.
WINDOW_E = 2.5
N_BANDS_BELOW = None
N_BANDS_ABOVE = None
# With METHOD = "projected": second-order coupling to the bands outside the window.
DOWNFOLD = True
# Use a new file when changing METHOD or the window: rerunning on an existing
# file keeps the structures already in it.
OUTPUT_JSON = "band_uncertainty.json"
# -----------------------------------------------------------------------------

# One worker process per allocated CPU.
cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

structures = [p for p in INFERENCE_INPUTS_DIR.iterdir() if p.is_dir()]

model_dirs = []
for i in range(1, N_MODELS + 1):
    matches = sorted(ENSEMBLE_ROOT.glob(MODEL_OUTPUT_GLOB.format(i=i)))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one inference output for model {i}, found {matches}")
    model_dir = matches[0]
    link_ensemble_files(structures=structures, ensemble_dir=model_dir, dst_parent=ENSEMBLE_ROOT)
    model_dirs.append(model_dir)

calc = BandUncertaintyCalculator(
    kpoints_settings=KPOINTS_SETTINGS,
    kpoints_force_2d=FORCE_2D,
    ill_method=ILL_METHOD,
    method=METHOD,
    n_bands_below=N_BANDS_BELOW,
    n_bands_above=N_BANDS_ABOVE,
    window_e=WINDOW_E,
    downfold=DOWNFOLD,
)
calc.compute_parallel(
    model_dirs,
    average_hamiltonian_dir=AVERAGE_HAM_DIR,
    structure_pattern=STRUCTURE_PATTERN,
    max_workers=cpus,
    output_path=OUTPUT_JSON,
)
