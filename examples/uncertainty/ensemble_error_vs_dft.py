"""Error of the ensemble-averaged Hamiltonian against DFT.

For structures with a DFT reference (e.g. a test set), this compares the
band energies of the ensemble-averaged Hamiltonian to those of the DFT
Hamiltonian on a k-grid, and writes per-structure, per-k absolute errors (eV).
Together with the uncertainty from ``ensemble_band_uncertainty.py`` on the
same structures, it tells how well the ensemble spread tracks the true error
(see ``UncertaintyCalibrator`` in ``hamilflow.uncertainty.band_postprocess``).

Run ``ensemble_band_uncertainty.py`` first with the same settings: it prepares
the model folders and writes the averaged Hamiltonians reused here.

Run with:  python ensemble_error_vs_dft.py   (e.g. inside a SLURM job)
"""

import os
from pathlib import Path

from hamilflow.uncertainty.band_uncertainty import BandUncertaintyCalculator

# --- User settings -----------------------------------------------------------
ENSEMBLE_ROOT = Path("/path/to/your/ensemble/folder")
N_MODELS = 5
MODEL_OUTPUT_GLOB = "train_{i}/infer/outputs/*/dft"
# DFT reference in DeepH format (one subfolder per structure, with hamiltonian.h5).
DFT_ROOT = Path("/path/to/your/test_set/dft")
AVERAGE_HAM_DIR = ENSEMBLE_ROOT / "avg_ham"
STRUCTURE_PATTERN = "*"  # glob on the structure folder names, e.g. "twisted_*"
OUTPUT_JSON = "error_vs_dft.json"
# -----------------------------------------------------------------------------

cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

model_dirs = []
for i in range(1, N_MODELS + 1):
    model_dirs.extend(sorted(ENSEMBLE_ROOT.glob(MODEL_OUTPUT_GLOB.format(i=i))))

# Use the same k-grid and ill-conditioning treatment as for the uncertainty,
# so that error and uncertainty refer to the same states.
calc = BandUncertaintyCalculator(grid_mesh=(3, 3, 1), ill_method="orbital_removal")
calc.compare_averaged_to_dft_parallel(
    model_dirs=model_dirs,
    average_hamiltonian_dir=AVERAGE_HAM_DIR,
    dft_root=DFT_ROOT,
    structure_pattern=STRUCTURE_PATTERN,
    max_workers=cpus,
    output_path=OUTPUT_JSON,
)
