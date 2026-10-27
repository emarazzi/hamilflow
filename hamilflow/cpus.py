from __future__ import annotations

import os


def available_cpu_count() -> int:
    """
    Cores actually usable by this process, respecting cgroup/cpuset limits
    (e.g. SLURM's --cpus-per-task). os.cpu_count() reports the node's full
    physical core count regardless of allocation, which under a SLURM
    cpuset causes BLAS thread limits to be set far too high.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # sched_getaffinity is POSIX-only (no Windows/macOS support).
        return os.cpu_count() or 1
