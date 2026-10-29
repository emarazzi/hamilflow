"""Full FHI-aims -> DeepH -> projected DeepH inputs workflow (jobflow-remote).

For every structure in ``structures.extxyz`` the flow chains, independently:

    aims_static_<name>               FHI-aims SCF, writing real-space H and S
    -> collect_<name>                move the run into ``aims_output/<name>``
    -> convert_<name>                AIMS -> DeepH format, in ``deeph_full/``
    -> projection_schur_<name>       stage 1: fold orbitals out via Schur complement
    -> projection_truncate_<name>    stage 2: drop further orbitals by truncation

The final DeepH training inputs are written in ``deeph_projected_truncated/``.

Before running, edit the "User settings" section below:
  * ``WORKDIR``: folder holding ``structures.extxyz`` and the two removal plans;
    all outputs are written there too.
  * the SLURM resources (account, partitions, wall times, memory);
  * the jobflow-remote project/worker names and the environment activation.

Removal plans are JSON files with a list of rules, e.g.

    {"rules": [{"target_elements": ["Mo"], "remove_shells": ["1s", "2s", "2p"]}]}

Each rule may set ``target_elements``, ``target_atom_indices``,
``remove_orbitals`` (families such as "s", "p", "d", "f") and
``remove_shells`` (shells such as "1s", "2p").

Run with:  python aims_to_projected_deeph.py
"""

from pathlib import Path

from jobflow_remote import set_run_config, submit_flow

from hamilflow.dft_workflows.flows_base import (
    ConvertAimsToDeephConfig,
    GenerateAimsDFTData,
    ProjectDeephInputsConfig,
)
from hamilflow.dft_workflows.flows_core import GenerateAimsToProjectedDeephData

# --- User settings -----------------------------------------------------------
WORKDIR = Path("/path/to/your/work/folder")
STRUCTURES_FILE = WORKDIR / "structures.extxyz"
SCHUR_REMOVAL_PLAN = WORKDIR / "removal_plan_schur.json"
TRUNCATE_REMOVAL_PLAN = WORKDIR / "removal_plan_truncate.json"

# Same k-point density for the AIMS runs and for the projections.
KPOINTS_SETTINGS = {"reciprocal_density": 1000}
FORCE_2D = True  # set k_z = 1 (slabs / 2D materials)

SLURM_ACCOUNT = "your_account"
JOBFLOW_PROJECT = "your_project"
PRE_RUN = "source /path/to/your/venv/bin/activate"
# -----------------------------------------------------------------------------

convert_config = ConvertAimsToDeephConfig(output_dir=WORKDIR / "deeph_full")

aims_dft = GenerateAimsDFTData(
    structures_path=STRUCTURES_FILE,
    aims_kwargs={
        "species_dir": "tight",
        "density_update_method": "density_matrix",
        "output_rs_matrices": "plain",  # required: writes the real-space H and S
        "occupation_type": "gaussian 0.01",
        "sc_accuracy_etot": "1.0e-07 eV",
        "sc_accuracy_rho": 1.0e-5,
        "many_body_dispersion_nl": "",
        "use_local_index": "false",
        "KS_method": "parallel",
        "collect_eigenvectors": "false",
    },
    user_kpoints_settings=KPOINTS_SETTINGS,
    force_2d=FORCE_2D,
    collected_runs_root=WORKDIR / "aims_output",
    aims_to_deeph_config=convert_config,
)

# n_workers=None picks the number of workers automatically from the CPUs
# allocated to the job (SLURM --cpus-per-task).
schur_config = ProjectDeephInputsConfig(
    output_root=WORKDIR / "deeph_projected_schur",
    removal_plan=SCHUR_REMOVAL_PLAN,
    reduction_mode="schur",
    user_kpoints_settings=KPOINTS_SETTINGS,
    force_2d=FORCE_2D,
    n_workers=None,
)

truncate_config = ProjectDeephInputsConfig(
    output_root=WORKDIR / "deeph_projected_truncated",
    removal_plan=TRUNCATE_REMOVAL_PLAN,
    reduction_mode="truncate",
    user_kpoints_settings=KPOINTS_SETTINGS,
    force_2d=FORCE_2D,
    n_workers=None,
)

maker = GenerateAimsToProjectedDeephData(
    dft_data_flow=aims_dft,
    projection_config=schur_config,
    second_projection_config=truncate_config,  # optional, drop it for a single stage
)
flow = maker.make()

# --- Per-job SLURM resources (matched on the job-name prefix) -----------------
resources_aims = {"nodes": 1, "ntasks": 64, "partition": "your_partition",
                  "time": "12:00:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="aims_static_", worker="submit",
                      resources=resources_aims, dynamic=False)

# The conversion is serial but memory hungry for large basis sets.
resources_conv = {"nodes": 1, "ntasks": 1, "cpus_per_task": 1, "mem": "30G",
                  "partition": "your_partition", "time": "00:40:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="convert_", worker="submit", resources=resources_conv)

# The Schur projection is dominated by dense matrix products: give it many cores.
# Check the real memory peak with `sacct -j <jobid> --format=JobName,Elapsed,MaxRSS`
# after a first run and adjust.
resources_schur = {"nodes": 1, "ntasks": 1, "cpus_per_task": 32, "mem": "60G",
                   "partition": "your_partition", "time": "02:00:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="projection_schur", worker="submit", resources=resources_schur)

resources_trunc = {"nodes": 1, "ntasks": 1, "cpus_per_task": 8, "mem": "60G",
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
