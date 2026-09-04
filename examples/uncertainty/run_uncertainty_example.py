from pathlib import Path
from hamilflow.uncertainty import (
    discover_structures,
    link_ensemble_files,
    average_predicted_hamiltonians,
    BandUncertaintyCalculator,
    MatrixUncertaintyCalculator,
)

# Adjust these paths for your environment
STRUCTURES_GLOB = "/path/to/dft/str*"
MODEL_DIRS = [
    Path("train_1/infer/outputs/2026-08-05_17-08-34/dft/"),
    Path("train_2/infer/outputs/2026-08-05_17-08-46/dft/"),
]
DFT_DIR = Path("/path/to/DFT/reference")
ENSEMBLE_DIR = Path("ensemble_avg_train")
DST_PARENT = Path("/path/to/dst_parent")

# Discover structures (example)
structures = discover_structures(STRUCTURES_GLOB)

# Link ensemble files into dst (optional)
link_ensemble_files(structures, ENSEMBLE_DIR, DST_PARENT)

# Average example (single structure example)
# pred_files = [d / structures[0].name / "hamiltonian.h5" for d in MODEL_DIRS]
# average_predicted_hamiltonians(pred_files, ENSEMBLE_DIR / structures[0].name / "hamiltonian.h5")

# Compute band uncertainty. Each k-point/band's result carries both
# "sigma_eV" (std of per-model eigenvalues around their mean) and
# "sigma_eV_avg_ham" (RMS deviation from the eigenvalues of the models'
# averaged Hamiltonian, diagonalized once per structure). average_hamiltonian_dir
# is required -- the averaged hamiltonian.h5 is read from there if already
# present, otherwise computed and written there (shared with MatrixUncertaintyCalculator).
calc = BandUncertaintyCalculator()
output = calc.compute(MODEL_DIRS, ENSEMBLE_DIR)
print(output)

# Compute Hamiltonian matrix-element uncertainty (std of |model - average|,
# streamed chunk-by-chunk per model, never loading a full entries array).
# Leave average_hamiltonian_dir unset to compute the average on the fly via
# average_predicted_hamiltonians; pass a directory to reuse/persist it.
matrix_calc = MatrixUncertaintyCalculator()
matrix_output = matrix_calc.compute(MODEL_DIRS, average_hamiltonian_dir=ENSEMBLE_DIR)
print(matrix_output)

# Parallel version -- one worker process per structure.
# matrix_output = matrix_calc.compute_parallel(MODEL_DIRS, average_hamiltonian_dir=ENSEMBLE_DIR)
