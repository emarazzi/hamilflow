"""
Dense linear algebra of the eigensolvers, on the CPU (SciPy/LAPACK) or on a GPU (CuPy/cuSOLVER).

`get_backend(n)` returns the GPU backend for an n x n problem when CuPy is installed, a CUDA
device is visible and n >= `GPU_MIN_DIM`, and the CPU backend otherwise. The environment
variable ``HAMILFLOW_GPU`` overrides this: ``0`` never uses the GPU, ``1`` requires it (an error
if it is not available), ``auto`` (default) uses it when available.

On an A100, against one CPU core: the generalized eigenproblem of n = 1856 / 2204 / 7076 is
50-120x faster and the orbital removal 6-45x, with eigenvalues equal to 1e-11 eV.

CUDA is initialized in the process that first asks for the GPU, so the worker processes that
`compute_parallel` forks each create their own context; several of them can share one GPU.
Every backend function takes and returns arrays of its own `xp`; `to_device` / `to_host` convert
from and to NumPy.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import scipy.linalg as sla

logger = logging.getLogger(__name__)

# Below this size the CPU is fast enough and the GPU's overhead dominates.
GPU_MIN_DIM = 256

_CPU = SimpleNamespace(
    name="cpu",
    xp=np,
    cholesky=lambda a: sla.cholesky(a, lower=True, check_finite=False),
    solve_triangular=lambda a, b, trans=0, overwrite_b=False: sla.solve_triangular(
        a, b, lower=True, trans=trans, overwrite_b=overwrite_b, check_finite=False),
    cho_solve=lambda L, b: sla.cho_solve((L, True), b, check_finite=False),
    # MRRR (see `eigensolver.eigh_generalized`)
    eigh=lambda a: sla.eigh(a, driver="evr", check_finite=False),
    eigvalsh=lambda a: sla.eigh(a, eigvals_only=True, driver="evr", check_finite=False),
    lowest_eigh=lambda a: sla.eigh(a, subset_by_index=[0, 0], check_finite=False),
    to_device=np.asarray,
    to_host=np.asarray,
)

_gpu_cache: dict[int, SimpleNamespace | None] = {}  # pid -> GPU backend (None: no GPU)

# Set while the processes started from this one have their GPU use reported by it (see
# `gpu_use_reported`). An environment variable, not a module flag, so that it reaches workers
# started by spawn as well as by fork. Not forkserver workers (the Linux default from Python
# 3.14) if the server was started earlier: those keep its environment and log at INFO.
_GPU_REPORTED_ENV = "HAMILFLOW_GPU_USE_REPORTED"


@contextmanager
def gpu_use_reported() -> Iterator[None]:
    """For the duration, the processes started from this one (and this one) log their GPU set-up
    at DEBUG instead of INFO: the caller reports the GPU use of the whole calculation itself, at
    INFO once (e.g. `_run_parallel`'s scheduling summary) instead of once per worker process."""
    previous = os.environ.get(_GPU_REPORTED_ENV)
    os.environ[_GPU_REPORTED_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            del os.environ[_GPU_REPORTED_ENV]
        else:
            os.environ[_GPU_REPORTED_ENV] = previous


def gpu_in_use() -> bool:
    """Whether this process has set up the GPU backend (without setting it up)."""
    return _gpu_cache.get(os.getpid()) is not None


def _make_gpu() -> SimpleNamespace | None:
    mode = os.environ.get("HAMILFLOW_GPU", "auto").strip().lower()
    if mode in ("0", "off", "false", "no", "none"):
        logger.debug(f"HAMILFLOW_GPU={mode}: dense linear algebra on the CPU")
        return None
    try:
        import cupy as cp
        import cupyx
        import cupyx.scipy.linalg as csla

        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("no CUDA device visible")
        # a CuPy wheel built for another CUDA major version typically imports fine and fails
        # here, at its first kernel or cuSOLVER call
        cp.linalg.eigh(cp.eye(2))
    except Exception as err:  # ImportError, CUDARuntimeError, ...
        not_installed = isinstance(err, ModuleNotFoundError) and err.name in ("cupy", "cupyx")
        msg = f"the GPU backend is unavailable: {type(err).__name__}: {err}"
        if not not_installed:
            msg += (". `pip install hamilflow[gpu]` installs cupy-cuda12x (CUDA 12); if your CUDA is another "
                    "major version, replace it with the matching wheel, e.g. "
                    "`pip uninstall cupy-cuda12x && pip install cupy-cuda11x`")
        if mode in ("1", "on", "true", "yes"):
            raise RuntimeError(f"HAMILFLOW_GPU={mode} but {msg}") from err
        if not not_installed:  # installed but broken: say so instead of silently using the CPU
            warnings.warn(f"hamilflow: {msg}. Running on the CPU (HAMILFLOW_GPU=0 silences this).",
                          RuntimeWarning, stacklevel=2)
        else:
            logger.debug("CuPy is not installed: dense linear algebra on the CPU")
        return None

    # once per process (the backend is cached), at INFO unless the caller reports it instead
    level = logging.DEBUG if os.environ.get(_GPU_REPORTED_ENV) else logging.INFO
    logger.log(level, f"Using the GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}) for dense "
                      f"linear algebra of size >= {GPU_MIN_DIM} (pid={os.getpid()})")

    def cholesky(a):
        # raises numpy.linalg.LinAlgError (= scipy.linalg.LinAlgError) if a is not positive definite
        with cupyx.errstate(linalg="raise"):
            return cp.linalg.cholesky(a)

    def solve_triangular(a, b, trans=0, overwrite_b=False):
        # in place only if b is Fortran-ordered and a is too (else cuBLAS gets copies)
        return csla.solve_triangular(a, b, lower=True, trans=trans, overwrite_b=overwrite_b)

    def lowest_eigh(a):  # cuSOLVER has no cheap single-eigenpair driver
        w, v = cp.linalg.eigh(a)
        return w[:1], v[:, :1]

    return SimpleNamespace(
        name="gpu",
        xp=cp,
        cholesky=cholesky,
        solve_triangular=solve_triangular,
        cho_solve=lambda L, b: solve_triangular(L, solve_triangular(L, b), trans="C"),
        eigh=cp.linalg.eigh,
        eigvalsh=cp.linalg.eigvalsh,
        lowest_eigh=lowest_eigh,
        to_device=cp.asarray,
        to_host=cp.asnumpy,
    )


def gpu_backend() -> SimpleNamespace | None:
    """The GPU backend of this process, or None if there is none (see the module docstring)."""
    pid = os.getpid()
    if pid not in _gpu_cache:
        _gpu_cache[pid] = _make_gpu()
    return _gpu_cache[pid]


def get_backend(n: int) -> SimpleNamespace:
    """Backend for an n x n problem: the GPU if available and n >= `GPU_MIN_DIM`, else the CPU."""
    if n >= GPU_MIN_DIM:
        gpu = gpu_backend()
        if gpu is not None:
            return gpu
    return _CPU


def release_gpu_memory() -> int | None:
    """Return the GPU memory cached by CuPy's pool to the device, so that the other processes
    sharing the GPU can use it. Returns the bytes the pool held until then -- since it is only
    emptied here, the peak device memory of the work since the last call (CUDA context not
    included) -- or None without a GPU backend in this process."""
    if _gpu_cache.get(os.getpid()) is None:
        return None
    import cupy as cp

    pool = cp.get_default_memory_pool()
    held = int(pool.total_bytes())
    pool.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return held


def workers_gpu_memory() -> int | None:
    """Total memory (bytes) of the GPU that worker processes started from this one would use, or
    None if they would not use a GPU (see `gpu_backend`). Asks nvidia-smi instead of CUDA, so
    that this process does not initialize CUDA, which forked workers could then not use."""
    import importlib.util
    import subprocess

    if os.environ.get("HAMILFLOW_GPU", "auto").strip().lower() in ("0", "off", "false", "no", "none"):
        return None
    if importlib.util.find_spec("cupy") is None:
        return None
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if not device or device == "-1":
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
                              "-i", device], capture_output=True, text=True, check=True, timeout=30).stdout
        return int(float(out.split()[0]) * 2**20)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
