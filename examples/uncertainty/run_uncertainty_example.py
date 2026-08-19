from pathlib import Path
from hamilflow.uncertainty import (
    discover_structures,
    link_ensemble_files,
    average_predicted_hamiltonians,
    BandUncertaintyCalculator,
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

# Compute band uncertainty
calc = BandUncertaintyCalculator()
output = calc.compute(MODEL_DIRS, DFT_DIR)
print(output)
