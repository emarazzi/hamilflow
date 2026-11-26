"""Supervised training of a k-fold DeepH ensemble (jobflow-remote): one submission, rounds until done.

Reads the ``dataset_split_{1..k}.json`` files that make_kfold_splits.py wrote in CYCLE_DIR. The
supervisor writes ``train_{j}/train_r{n}.toml`` for each round n, runs every member's
``deeph-train`` under a watchdog on the GPU worker (it stops the run before the SLURM time limit, on
divergence, or when the member converged), and after each round decides on the light worker, from
the logs of all members, whether each member resumes, gets a new patience/cooldown, restarts at a
lower LR, or is done. See the README section "Supervised ensemble training" for the policy, the files
and the recovery commands, and use ``hamilflow-train-supervisor status|review|approve|stop|finish|
summary CYCLE_DIR`` to follow and steer it.

Never ``scancel`` / ``jf job cancel`` a train job: ``hamilflow-train-supervisor stop CYCLE_DIR``.

Before running, edit the "User settings" section below:
  * ``CYCLE_DIR``, ``K`` and the shared train.toml overrides (as in write_train_tomls.py);
  * the policy (``ACTIONS``) and the limits;
  * ``TRAIN_COMMAND`` (DeepH lives in its own venv), the SLURM resources and the worker names.

Run with:  python supervised_training.py
"""

from pathlib import Path

from jobflow_remote import submit_flow

from hamilflow.training.supervisor import SupervisorConfig
from hamilflow.training.supervisor.jobs import TrainingSupervisorMaker, warm_start_from

# --- User settings -----------------------------------------------------------
CYCLE_DIR = Path("/path/to/your/models/cycle2")
K = 5
# Warm start from the previous active-learning cycle, trained with this supervisor: each member starts
# from its final run there (best params, new optimizer), on the new data. None: start from scratch.
PREVIOUS_CYCLE_DIR = None  # e.g. Path("/path/to/your/models/cycle1")

JOBFLOW_PROJECT = "your_project"
SLURM_ACCOUNT = "your_account"
GPU_WORKER = "your_gpu_worker"  # runs deeph-train
LIGHT_WORKER = "shell"  # the login node, a local-shell worker: it must see CYCLE_DIR; keep `jf runner` up
PRE_RUN = "source /path/to/your/hamilflow-venv/bin/activate"  # hamilflow + jobflow-remote

# deeph-train lives in another venv: the watchdog starts it with this command ("{toml}" is the round's
# train_r{n}.toml) in a clean environment (no PYTHONPATH/PYTHONHOME/VIRTUAL_ENV of the hamilflow venv).
TRAIN_COMMAND = ["bash", "-c", "source /path/to/your/deeph-venv/bin/activate && exec deeph-train {toml}"]

OVERRIDES = {
    "system": {"device": "gpu*2:0-1", "random_seed": 137, "float_type": "tf32", "jax_memory_preallocate": False},
    "data": {
        "inputs_dir": "/path/to/your/training/inputs",
        "outputs_dir": "./outputs",  # each round writes ./outputs_r{n}
        "graph": {"dataset_name": "your_dataset", "graph_type": "HS", "storage_type": "disk",
                  "common_orbital_types": "s2p2d2f1"},
    },
    "model": {
        "net_type": "eagle", "target_type": "H", "loss_type": "mse",
        "advanced": {"net_irreps": "16x0e+16x1e+20x2e+16x3e+8x4e+4x5e+2x6e", "gaussian_basis_rmax": 12.0,
                     "num_blocks": 3, "consider_parity": False, "standardize_gauge": True},
    },
    "process": {
        "train": {
            "max_epoch": 10000,
            "dataloader": {"batch_size": 1},
            "optimizer": {"type": "adamw", "init_learning_rate": 1e-3},
            "scheduler": {
                "type": "reduce_on_plateau",
                # the floor the training ends at; the supervisor keeps it (relative to this lr0) through
                # every retune and restart
                "min_learning_rate_scale": 1e-7,
                "patience": 20,
                "cooldown": 10,
            },
        },
    },
}

# Per action: "auto" (act), "ask" (stop the round with a proposal: review, then approve), "off".
ACTIONS = {
    "resume": "auto",  # walltime / stop -> resume from latest, optimizer kept (off: the member is done)
    "stop_converged": "auto",  # converged -> done (off: keep training)
    "retune": "ask",  # pooled patience/cooldown suggestion -> new optimizer from best, LR carried over
    "divergence": "ask",  # val > 10 x best for 5 epochs -> restart from best at half the LR (off: give up)
    "stop_overfit": "ask",  # val/train grew -> done (best-val checkpoint)
    "finished_not_converged": "ask",  # deeph-train ended at min scale / max_epoch without converging
}
# A nan/inf loss always stops the run, whatever the policy; restarting it follows "divergence".

CONFIG = SupervisorConfig(
    actions=ACTIONS,
    train_command=TRAIN_COMMAND,
    # converged: the last conv_stages reduced-LR stages each improved the best val by < conv_tol, or the
    # LR scale (relative to the cycle's first lr0) is <= done_scale. With done_scale 1e-3 above the floor
    # (min_learning_rate_scale), members stop before DeepH's own end: lower it to train down to the floor.
    conv_tol=0.02,
    conv_stages=2,
    done_scale=1e-3,
    # SLURM time limit of the train jobs (below) minus this margin is when the watchdog stops deeph-train;
    # nothing else protects the job from the time limit (no SLURM signal reaches it)
    walltime_margin=900.0,
    poll_s=300.0,
    latest_interval=10,  # a latest checkpoint every 10 epochs: a stop loses at most that many
    max_rounds=10,
)

# GPU train jobs: the SLURM time limit must exceed walltime_margin (15 min) + start-up (graph and JIT,
# minutes). The watchdog reads the end time from SLURM, so nothing else needs to match it.
# Set "mem" explicitly (the SLURM default per CPU can be too small), and on a partition that mixes GPU types
# ask for a type with enough memory (e.g. "gpu:TeslaA100:2"): a plain "gpu:2" may land on an 11 GB card,
# where deeph-train runs out of GPU memory. watchdog_J.log records the node and GPU each member got.
RESOURCES_TRAIN = {"nodes": 1, "ntasks": 1, "cpus_per_task": 32, "mem": "64G", "gres": "gpu:2",
                   "partition": "your_gpu_partition", "time": "2-00:00:00", "account": SLURM_ACCOUNT}
# The supervise job parses logs and writes small files (seconds): the local-shell worker needs no resources.
RESOURCES_SUPERVISE = None
EXEC_CONFIG = {"pre_run": PRE_RUN}
# -----------------------------------------------------------------------------


def build_flow():
    """Write round 1's TOMLs and state.json, and return the round-1 flow (nothing is submitted)."""
    maker = TrainingSupervisorMaker(
        CONFIG, OVERRIDES, K,
        resources_train=RESOURCES_TRAIN, resources_supervise=RESOURCES_SUPERVISE,
        train_worker=GPU_WORKER, supervise_worker=LIGHT_WORKER,
        exec_config_train=EXEC_CONFIG, exec_config_supervise=EXEC_CONFIG,
        project=JOBFLOW_PROJECT,
    )
    # members of a supervised cycle end in different rounds: warm_start_from reads each one's final run
    continue_from = None if PREVIOUS_CYCLE_DIR is None else warm_start_from(PREVIOUS_CYCLE_DIR)
    return maker.make(CYCLE_DIR, continue_from=continue_from)


def main():
    flow = build_flow()
    # the light worker is the flow's default: the store_inputs jobs jobflow adds at each round run there
    submit_flow(flow, worker=LIGHT_WORKER, exec_config=EXEC_CONFIG, project=JOBFLOW_PROJECT)
    print(f"Submitted the supervised training of {CYCLE_DIR}: hamilflow-train-supervisor status {CYCLE_DIR}")


if __name__ == "__main__":
    main()
