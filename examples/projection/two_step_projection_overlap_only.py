"""Two-step projection of existing DeepH inputs, overlap matrix only (jobflow-remote).

Same as ``two_step_projection.py`` but only S is projected
(``overlap_only=True``), e.g. to prepare inference inputs where H will be
predicted by the model. Each stage writes a dummy, zero-filled
``hamiltonian.h5`` (``write_dummy_hamiltonian=True``), because the next stage
reads the structure metadata from it.

Before running, edit the "User settings" section below.

Run with:  python two_step_projection_overlap_only.py
"""

from pathlib import Path

from jobflow_remote import set_run_config, submit_flow

from hamilflow.dft_workflows.flows_base import ProjectDeephInputsConfig
from hamilflow.dft_workflows.flows_core import GenerateTwoStepProjectedDeephInputs

# --- User settings -----------------------------------------------------------
WORKDIR = Path("/path/to/your/work/folder")
DEEPH_INPUTS_ROOT = WORKDIR / "deeph_full_ovl"  # one subfolder per structure
SCHUR_REMOVAL_PLAN = WORKDIR / "removal_plan_schur.json"
TRUNCATE_REMOVAL_PLAN = WORKDIR / "removal_plan_truncate.json"

KPOINTS_SETTINGS = {"reciprocal_density": 1000}
FORCE_2D = True  # set k_z = 1 (slabs / 2D materials)

SLURM_ACCOUNT = "your_account"
JOBFLOW_PROJECT = "your_project"
PRE_RUN = "source /path/to/your/venv/bin/activate"
# -----------------------------------------------------------------------------

# n_workers=None -> chosen automatically from the CPUs allocated to the job.
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

maker = GenerateTwoStepProjectedDeephInputs(
    deeph_inputs_root=DEEPH_INPUTS_ROOT,
    first_projection_config=schur_config,
    second_projection_config=truncate_config,
)
flow = maker.make()

# --- Per-job SLURM resources (matched on the job-name prefix) -----------------
# About half the memory of the full projection, since only S is handled.
resources_schur = {"nodes": 1, "ntasks": 1, "cpus_per_task": 32, "mem": "120G",
                   "partition": "your_partition", "time": "01:00:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="projection_schur", worker="submit", resources=resources_schur)

resources_trunc = {"nodes": 1, "ntasks": 1, "cpus_per_task": 8, "mem": "80G",
                   "partition": "your_partition", "time": "00:10:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="projection_trunc", worker="submit", resources=resources_trunc)

submit_flow(
    flow,
    worker="submit",
    # PYTHONFAULTHANDLER prints a traceback if a worker crashes (e.g. OOM in BLAS).
    exec_config={"pre_run": f"{PRE_RUN}; export PYTHONFAULTHANDLER=1"},
    project=JOBFLOW_PROJECT,
)
