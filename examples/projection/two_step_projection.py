"""Two-step projection of existing DeepH inputs (jobflow-remote).

Starts from DeepH-format data that is already on disk, e.g. the ``deeph_full``
folder written by the conversion step of
``dft_workflows/aims_to_projected_deeph.py``, with one subfolder per structure.
For each structure it runs:

    projection_schur_<name>      stage 1: fold orbitals out via Schur complement
    -> projection_truncate_<name> stage 2: drop further orbitals by truncation

Stage 2 reads the output of stage 1, so the two output folders must differ.
See ``dft_workflows/aims_to_projected_deeph.py`` for the removal-plan format.

Before running, edit the "User settings" section below.

Run with:  python two_step_projection.py
"""

from pathlib import Path

from jobflow_remote import set_run_config, submit_flow

from hamilflow.dft_workflows.flows_base import ProjectDeephInputsConfig
from hamilflow.dft_workflows.flows_core import GenerateTwoStepProjectedDeephInputs

# --- User settings -----------------------------------------------------------
WORKDIR = Path("/path/to/your/work/folder")
DEEPH_INPUTS_ROOT = WORKDIR / "deeph_full"  # one subfolder per structure
SCHUR_REMOVAL_PLAN = WORKDIR / "removal_plan_schur.json"
TRUNCATE_REMOVAL_PLAN = WORKDIR / "removal_plan_truncate.json"

KPOINTS_SETTINGS = {"reciprocal_density": 1000}
FORCE_2D = True  # set k_z = 1 (slabs / 2D materials)

SLURM_ACCOUNT = "your_account"
JOBFLOW_PROJECT = "your_project"
PRE_RUN = "source /path/to/your/venv/bin/activate"
# -----------------------------------------------------------------------------

# n_workers=None -> one worker per CPU allocated to the job (SLURM
# --cpus-per-task). The k-mesh is streamed in fixed-size chunks, so memory does
# not grow with the mesh size; it is roughly
#   sparse H/S of the input + n_workers * ~0.3 GB + 2-3x the output entries.
schur_config = ProjectDeephInputsConfig(
    output_root=WORKDIR / "deeph_projected_schur",
    removal_plan=SCHUR_REMOVAL_PLAN,
    structure_pattern="*",  # glob on the structure folder names
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

maker = GenerateTwoStepProjectedDeephInputs(
    deeph_inputs_root=DEEPH_INPUTS_ROOT,
    first_projection_config=schur_config,
    second_projection_config=truncate_config,
)
flow = maker.make()

# --- Per-job SLURM resources (matched on the job-name prefix) -----------------
# All k-space stages run in parallel threads, so extra cores help. Memory
# requests here are generous: check the real peak with
#   sacct -j <jobid> --format=JobName,Elapsed,MaxRSS
# after a first run and lower them.
resources_schur = {"nodes": 1, "ntasks": 1, "cpus_per_task": 32, "mem": "240G",
                   "partition": "your_partition", "time": "02:00:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="projection_schur", worker="submit", resources=resources_schur)

resources_trunc = {"nodes": 1, "ntasks": 1, "cpus_per_task": 8, "mem": "120G",
                   "partition": "your_partition", "time": "00:10:00", "account": SLURM_ACCOUNT}
flow = set_run_config(flow, name_filter="projection_trunc", worker="submit", resources=resources_trunc)

submit_flow(
    flow,
    worker="submit",
    # PYTHONFAULTHANDLER prints a traceback if a worker crashes (e.g. OOM in BLAS).
    exec_config={"pre_run": f"{PRE_RUN}; export PYTHONFAULTHANDLER=1"},
    project=JOBFLOW_PROJECT,
)
