"""Overlap-only FHI-aims -> DeepH -> projected DeepH inputs workflow (jobflow-remote).

Same chain as ``aims_to_projected_deeph.py``, but only the overlap matrix S is
needed, e.g. to run DeepH inference on new structures (the model predicts H,
but still needs S in the projected basis). Compared with the full workflow:

  * AIMS stops right after writing the real-space matrices and the SCF
    criteria are loose (S does not depend on the density), so runs are short;
  * the converted ``hamiltonian.h5`` keeps its layout but is filled with zeros
    (``dummy_hamiltonian`` is forced by the flow);
  * the projections act on S only (``overlap_only=True``), and each stage
    writes a dummy ``hamiltonian.h5`` (``write_dummy_hamiltonian=True``),
    since the next stage reads the structure metadata from it.

Before running, edit the "User settings" section below (see
``aims_to_projected_deeph.py`` for the removal-plan format).

Run with:  python aims_overlap_only_to_projected_deeph.py
"""

from pathlib import Path

from jobflow_remote import set_run_config, submit_flow

from hamilflow.dft_workflows.flows_base import (
    ConvertAimsToDeephConfig,
    ProjectDeephInputsConfig,
)
from hamilflow.dft_workflows.flows_core import GenerateOvlOnlyAimsToProjectedDeephData

# --- User settings -----------------------------------------------------------
WORKDIR = Path("/path/to/your/work/folder")
STRUCTURES_FILE = WORKDIR / "structures.extxyz"
SCHUR_REMOVAL_PLAN = WORKDIR / "removal_plan_schur.json"
TRUNCATE_REMOVAL_PLAN = WORKDIR / "removal_plan_truncate.json"

KPOINTS_SETTINGS = {"reciprocal_density": 1000}
FORCE_2D = True  # set k_z = 1 (slabs / 2D materials)

SLURM_ACCOUNT = "your_account"
JOBFLOW_PROJECT = "your_project"
PRE_RUN = "source /path/to/your/venv/bin/activate"
# -----------------------------------------------------------------------------

convert_config = ConvertAimsToDeephConfig(output_dir=WORKDIR / "deeph_full_ovl")

# n_workers=None: automatic. For Schur, one worker using all cores in BLAS when
# many orbitals (>= 128) are removed, otherwise one worker per core.
schur_config = ProjectDeephInputsConfig(
    output_root=WORKDIR / "deeph_projected_schur_ovl",
    removal_plan=SCHUR_REMOVAL_PLAN,
    reduction_mode="schur",
    user_kpoints_settings=KPOINTS_SETTINGS,
    force_2d=FORCE_2D,
    overlap_only=True,
    write_dummy_hamiltonian=True,
    n_workers=None,
)

truncate_config = ProjectDeephInputsConfig(
    output_root=WORKDIR / "deeph_projected_truncated_ovl",
    removal_plan=TRUNCATE_REMOVAL_PLAN,
    reduction_mode="truncate",
    user_kpoints_settings=KPOINTS_SETTINGS,
    force_2d=FORCE_2D,
    overlap_only=True,
    write_dummy_hamiltonian=True,
    n_workers=None,
)

maker = GenerateOvlOnlyAimsToProjectedDeephData(
    structures_path=STRUCTURES_FILE,
    aims_kwargs={
        "species_dir": "tight",
        "density_update_method": "density_matrix",
        "output_rs_matrices": "plain",
        "occupation_type": "gaussian 0.01",
        "use_local_index": "false",
        "KS_method": "parallel",
        "collect_eigenvectors": "false",
        # Only S matters: no MBD and loose SCF criteria. The flow adds
        # output_rs_matrices_stop to stop AIMS once the matrices are written.
        "sc_accuracy_etot": "1.0e-03 eV",
        "sc_accuracy_rho": 1.0e-2,
    },
    user_kpoints_settings=KPOINTS_SETTINGS,
    force_2d=FORCE_2D,
    collected_runs_root=WORKDIR / "aims_output_ovl",
    aims_to_deeph_config=convert_config,
    projection_config=schur_config,
    second_projection_config=truncate_config,
)
flow = maker.make()

# --- Per-job SLURM resources (matched on the job-name prefix) -----------------
# AIMS: same parallel layout as a full run, much shorter wall time.
resources_aims = {"nodes": 1, "ntasks": 128, "mem": 0, "partition": "your_partition",
                  "time": "02:00:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="aims_static_", worker="submit",
                      resources=resources_aims, dynamic=False)

resources_conv = {"nodes": 1, "ntasks": 1, "cpus_per_task": 1, "mem": "30G",
                  "partition": "your_partition", "time": "00:40:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="convert_", worker="submit", resources=resources_conv)

# Schur on S only: about half the work and memory of the full run.
resources_schur = {"nodes": 1, "ntasks": 1, "cpus_per_task": 32, "mem": "40G",
                   "partition": "your_partition", "time": "01:00:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="projection_schur", worker="submit", resources=resources_schur)

resources_trunc = {"nodes": 1, "ntasks": 1, "cpus_per_task": 8, "mem": "30G",
                   "partition": "your_partition", "time": "00:10:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="projection_trunc", worker="submit", resources=resources_trunc)

# Jobs without their own run config (the light collect_ jobs) go to the
# default worker, here a local "shell" worker.
submit_flow(
    flow,
    worker="shell",
    exec_config={"pre_run": PRE_RUN},
    project=JOBFLOW_PROJECT,
)
