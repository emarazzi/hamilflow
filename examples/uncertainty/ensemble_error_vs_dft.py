"""Error of the ensemble-averaged Hamiltonian against DFT.

For structures with a DFT reference (e.g. a test set), this compares the
band energies of the ensemble-averaged Hamiltonian to those of the DFT
Hamiltonian on a k-grid, and writes per-structure, per-k absolute errors (eV).
Together with the uncertainty from ``ensemble_band_uncertainty.py`` on the
same structures, it tells how well the ensemble spread tracks the true error
(see ``UncertaintyCalibrator`` in ``hamilflow.uncertainty.band_postprocess``).

Run ``ensemble_band_uncertainty.py`` first with the same settings: it prepares
the model folders and writes the averaged Hamiltonians reused here. Keep
KPOINTS_SETTINGS, FORCE_2D, ILL_METHOD and ILL_THRESHOLD identical in both scripts, so that
error and uncertainty refer to the same states.

The error is always computed by diagonalizing the averaged and the DFT
Hamiltonians and covers all bands, whatever METHOD was used for the
uncertainty. If the uncertainty was computed with METHOD = "projected", it
only covers its band window: compare the two with the band-edge-weighted
statistics (``weighted_max_eV`` / ``weighted_mean_eV``), with the window
wider than e_cut plus a few tau, and without ``bands`` / ``vbm_fraction``,
whose band indices refer to different band lists in the two files.

One worker process runs per allocated CPU; the diagonalizations use the GPU when available
(one GPU and a few CPUs per job, HAMILFLOW_GPU as in ``ensemble_band_uncertainty.py``;
SLURM job script: ``job_gpu.sh``, running this script instead).

Run with:  python ensemble_error_vs_dft.py   (e.g. inside a SLURM job)
"""

import logging
import os
import sys
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
# k-point mesh, chosen per structure from its cell (see
# dft_workflows.kpoints.get_ksampling): a uniform linear spacing in 1/Angstrom
# (2*pi included), k_i = ceil(|b_i| / kspacing), with k_i = 1 along every axis
# found to be vacuum (gaps >= vacuum_gap Angstrom; slabs, wires, molecules).
# A coarser spacing than the DFT's is usually enough for the error statistics.
# Other forms: {"reciprocal_density": 100}, {"grid_density": 1000}, {"k_grid": [3, 3, 1]}.
KPOINTS_SETTINGS = {"kspacing": 0.3, "vacuum_gap": 8.3}
FORCE_2D = False  # force k_z = 1 on top of the vacuum detection (rarely needed now)
# Treatment of an ill-conditioned overlap: "canonical" cuts, at each k-point, the
# eigen-directions of S(k) with eigenvalue below ILL_THRESHOLD (S-only, so the
# DFT and all members are solved in the same subspace). 3e-3 is the value tested
# for the carbon tight projected basis: it removes the eV-scale "ghost" states
# that model errors along near-singular directions of S produce, while moving the
# DFT bands by tens of meV near the gap (see IllConditionedHandler). It needs no
# orbital selection, so it is far cheaper overall than "orbital_removal" (which
# removes whole orbitals until S(k) is well conditioned on the k-grid, distorting
# the bands much more), though each diagonalization costs ~2x a plain one.
# None = no treatment. The uncertainty run (ensemble_band_uncertainty.py) and the
# error run (ensemble_error_vs_dft.py) must use the same ILL_METHOD and
# ILL_THRESHOLD: UncertaintyCalibrator's consistency check rejects a mismatch.
ILL_METHOD = "canonical"
ILL_THRESHOLD = 3e-3
OUTPUT_JSON = "error_vs_dft.json"
# -----------------------------------------------------------------------------

# Show hamilflow's progress messages on stdout (the SLURM .out file) and save
# them, with any warnings, to a log file next to OUTPUT_JSON.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(Path(OUTPUT_JSON).with_suffix(".log"))],
)
logging.captureWarnings(True)

cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

model_dirs = []
for i in range(1, N_MODELS + 1):
    model_dirs.extend(sorted(ENSEMBLE_ROOT.glob(MODEL_OUTPUT_GLOB.format(i=i))))

calc = BandUncertaintyCalculator(
    kpoints_settings=KPOINTS_SETTINGS,
    kpoints_force_2d=FORCE_2D,
    ill_method=ILL_METHOD,
    ill_threshold=ILL_THRESHOLD,
)
calc.compare_averaged_to_dft_parallel(
    model_dirs=model_dirs,
    average_hamiltonian_dir=AVERAGE_HAM_DIR,
    dft_root=DFT_ROOT,
    structure_pattern=STRUCTURE_PATTERN,
    max_workers=cpus,
    output_path=OUTPUT_JSON,
)
