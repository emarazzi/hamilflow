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


def available_memory_bytes() -> int:
    """
    Memory this job may use: SLURM's allocation (--mem, or --mem-per-cpu times the allocated
    CPUs) if set, else the cgroup limit, else the memory currently available on the machine.
    """
    if os.environ.get("SLURM_MEM_PER_NODE"):
        return int(os.environ["SLURM_MEM_PER_NODE"]) * 2**20
    if os.environ.get("SLURM_MEM_PER_CPU"):
        return int(os.environ["SLURM_MEM_PER_CPU"]) * 2**20 * available_cpu_count()
    available = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        if limit != "max":
            available = min(int(limit), available) if available else int(limit)
    except (OSError, ValueError):
        pass
    return available or 8 * 2**30


def reset_peak_rss() -> None:
    """Restart this process's peak resident memory (`peak_rss_bytes`) from the current one
    (Linux; no-op elsewhere)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass


def peak_rss_bytes() -> int | None:
    """Peak resident memory of this process since it started or since `reset_peak_rss` (Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None
