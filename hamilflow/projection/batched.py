from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from deepx_dock.CONSTANT import (
    DEEPX_HAMILTONIAN_FILENAME,
    DEEPX_OVERLAP_FILENAME,
    DEEPX_POSCAR_FILENAME,
)
from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj

from .kspace import (
    apply_custom_kspace_transform,
    apply_truncation_kspace_transform,
    build_uniform_kmesh,
)
from .io import write_reduced_info_json
from .models import ProjectionConfig, ProjectionResult, RemovalPlanLike
from .removal import coerce_removal_plan, resolve_indices_from_rules


_WORKER_STATE: dict[str, object] = {}


def _iter_slices(length: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, length, batch_size):
        stop = min(start + batch_size, length)
        yield slice(start, stop)


def _worker_init(
    input_dir: str,
    reduction_mode: str,
    remove_indices: list[int],
    Rijk_list: np.ndarray,
    reduced_nb: int,
    r_batch_size: int,
    k_weight: float,
) -> None:
    _WORKER_STATE["obj"] = HamiltonianObj(Path(input_dir))
    _WORKER_STATE["reduction_mode"] = reduction_mode
    _WORKER_STATE["remove_indices"] = list(remove_indices)
    _WORKER_STATE["Rijk_list"] = np.asarray(Rijk_list)
    _WORKER_STATE["reduced_nb"] = int(reduced_nb)
    _WORKER_STATE["r_batch_size"] = int(r_batch_size)
    _WORKER_STATE["k_weight"] = float(k_weight)


def _worker_compute_k_batch(
    task_id: int,
    ks_batch: np.ndarray,
    partial_dir: str,
) -> tuple[str, str]:
    obj = _WORKER_STATE["obj"]
    reduction_mode = _WORKER_STATE["reduction_mode"]
    remove_indices = _WORKER_STATE["remove_indices"]
    Rijk_list = _WORKER_STATE["Rijk_list"]
    reduced_nb = int(_WORKER_STATE["reduced_nb"])
    r_batch_size = int(_WORKER_STATE["r_batch_size"])
    k_weight = float(_WORKER_STATE["k_weight"])

    if not isinstance(obj, HamiltonianObj):
        raise RuntimeError("Worker HamiltonianObj is not initialized.")
    if not isinstance(reduction_mode, str):
        raise RuntimeError("Worker reduction_mode is not initialized.")
    if not isinstance(remove_indices, list):
        raise RuntimeError("Worker remove_indices is not initialized.")
    if not isinstance(Rijk_list, np.ndarray):
        raise RuntimeError("Worker Rijk_list is not initialized.")

    Sk_batch, Hk_batch = obj.Sk_and_Hk(ks_batch)

    if reduction_mode == "schur":
        Hk_new, Sk_new = apply_custom_kspace_transform(Hk_batch, Sk_batch, remove_indices=remove_indices)
    elif reduction_mode == "truncate":
        Hk_new, Sk_new = apply_truncation_kspace_transform(Hk_batch, Sk_batch, remove_indices=remove_indices)
    else:
        raise ValueError(f"Unsupported reduction_mode '{reduction_mode}'. Expected 'schur' or 'truncate'.")

    nR = int(Rijk_list.shape[0])
    Hk_flat = Hk_new.reshape(Hk_new.shape[0], -1)
    Sk_flat = Sk_new.reshape(Sk_new.shape[0], -1)

    partial_root = Path(partial_dir)
    hr_path = partial_root / f"partial_hr_{task_id:06d}.npy"
    sr_path = partial_root / f"partial_sr_{task_id:06d}.npy"

    hr_memmap = np.lib.format.open_memmap(
        hr_path,
        mode="w+",
        dtype=np.complex128,
        shape=(nR, reduced_nb, reduced_nb),
    )
    sr_memmap = np.lib.format.open_memmap(
        sr_path,
        mode="w+",
        dtype=np.complex128,
        shape=(nR, reduced_nb, reduced_nb),
    )
    hr_memmap.fill(0)
    sr_memmap.fill(0)

    for r_slice in _iter_slices(nR, r_batch_size):
        R_batch = Rijk_list[r_slice]
        phase = np.exp(-2j * np.pi * np.matmul(R_batch, ks_batch.T))
        coeff = phase * k_weight

        hr_batch = np.matmul(coeff, Hk_flat).reshape(r_slice.stop - r_slice.start, reduced_nb, reduced_nb)
        sr_batch = np.matmul(coeff, Sk_flat).reshape(r_slice.stop - r_slice.start, reduced_nb, reduced_nb)

        hr_memmap[r_slice] += hr_batch
        sr_memmap[r_slice] += sr_batch

    hr_memmap.flush()
    sr_memmap.flush()
    del hr_memmap
    del sr_memmap

    return str(hr_path), str(sr_path)


def _build_per_atom_reduced_indices(
    keep_global: list[int],
    atom_num_orbits_cumsum: np.ndarray,
) -> list[np.ndarray]:
    keep_arr = np.asarray(sorted(set(int(i) for i in keep_global)), dtype=int)
    per_atom_reduced_indices: list[np.ndarray] = []
    n_atoms = len(atom_num_orbits_cumsum) - 1
    for ia in range(n_atoms):
        a0 = int(atom_num_orbits_cumsum[ia])
        a1 = int(atom_num_orbits_cumsum[ia + 1])
        per_atom_reduced_indices.append(np.where((keep_arr >= a0) & (keep_arr < a1))[0])
    return per_atom_reduced_indices


def _build_chunk_metadata(
    Rijk_list: np.ndarray,
    atom_pairs: np.ndarray,
    atom_num_orbits_cumsum: np.ndarray,
    keep_global: list[int],
):
    per_atom_reduced_indices = _build_per_atom_reduced_indices(keep_global, atom_num_orbits_cumsum)
    r_to_idx = {tuple(int(v) for v in r): i for i, r in enumerate(Rijk_list)}

    atom_pairs = np.asarray(atom_pairs, dtype=np.int64)
    chunk_shapes = np.zeros((len(atom_pairs), 2), dtype=np.int64)
    chunk_boundaries = np.zeros(len(atom_pairs) + 1, dtype=np.int64)

    for i_ap, ap in enumerate(atom_pairs):
        ia = int(ap[3])
        ja = int(ap[4])
        ii = per_atom_reduced_indices[ia]
        jj = per_atom_reduced_indices[ja]
        chunk_shapes[i_ap] = np.array((ii.size, jj.size), dtype=np.int64)
        chunk_boundaries[i_ap + 1] = chunk_boundaries[i_ap] + int(ii.size * jj.size)

    total_entries = int(chunk_boundaries[-1])
    return atom_pairs, chunk_shapes, chunk_boundaries, per_atom_reduced_indices, r_to_idx, total_entries


def _write_reduced_matrix_h5_streaming(
    out_path: Path,
    mats_R: np.ndarray,
    Rijk_list: np.ndarray,
    atom_pairs: np.ndarray,
    atom_num_orbits_cumsum: np.ndarray,
    keep_global: list[int],
) -> Path:
    atom_pairs, chunk_shapes, chunk_boundaries, per_atom_reduced_indices, r_to_idx, total_entries = _build_chunk_metadata(
        Rijk_list=Rijk_list,
        atom_pairs=atom_pairs,
        atom_num_orbits_cumsum=atom_num_orbits_cumsum,
        keep_global=keep_global,
    )

    with h5py.File(out_path, "w") as f:
        f.create_dataset("atom_pairs", data=atom_pairs)
        f.create_dataset("chunk_shapes", data=chunk_shapes)
        f.create_dataset("chunk_boundaries", data=chunk_boundaries)
        entries = f.create_dataset(
            "entries",
            shape=(total_entries,),
            dtype=np.float64,
            chunks=True,
        )

        for i_ap, ap in enumerate(atom_pairs):
            Rijk = (int(ap[0]), int(ap[1]), int(ap[2]))
            r_idx = r_to_idx[Rijk]
            ia = int(ap[3])
            ja = int(ap[4])
            ii = per_atom_reduced_indices[ia]
            jj = per_atom_reduced_indices[ja]
            block = mats_R[r_idx][np.ix_(ii, jj)]
            start = int(chunk_boundaries[i_ap])
            stop = int(chunk_boundaries[i_ap + 1])
            entries[start:stop] = np.asarray(np.real(block).reshape(-1), dtype=np.float64)

    return out_path


def _hermitize_real_space_blocks_inplace(mats_R: np.ndarray, Rijk_list: np.ndarray) -> None:
    r_to_idx = {tuple(int(v) for v in r): i for i, r in enumerate(Rijk_list)}
    visited: set[tuple[int, int, int]] = set()

    for r, i_r in r_to_idx.items():
        if r in visited:
            continue

        r_neg = (-r[0], -r[1], -r[2])
        i_neg = r_to_idx.get(r_neg)

        A = mats_R[i_r].copy()
        if i_neg is None or i_neg == i_r:
            mats_R[i_r] = 0.5 * (A + np.conjugate(A.T))
            visited.add(r)
            continue

        B = mats_R[i_neg].copy()
        A_sym = 0.5 * (A + np.conjugate(B.T))
        mats_R[i_r] = A_sym
        mats_R[i_neg] = np.conjugate(A_sym.T)

        visited.add(r)
        visited.add(r_neg)


def _finalize_projection_outputs(
    config: ProjectionConfig,
    obj: HamiltonianObj,
    hr_mats: np.ndarray,
    sr_mats: np.ndarray,
    keep_global: list[int],
    remove_indices: list[int],
    plan_meta: dict[str, object],
    nb_total: int,
    k_batch_size: int,
    r_batch_size: int,
    parallel_workers: int | None = None,
) -> ProjectionResult:
    _hermitize_real_space_blocks_inplace(hr_mats, np.asarray(obj.Rijk_list))
    _hermitize_real_space_blocks_inplace(sr_mats, np.asarray(obj.Rijk_list))

    overlap_imag_max = float(np.max(np.abs(np.imag(sr_mats)))) if sr_mats.size > 0 else 0.0

    hamiltonian_path = _write_reduced_matrix_h5_streaming(
        config.output_dir / DEEPX_HAMILTONIAN_FILENAME,
        hr_mats,
        np.asarray(obj.Rijk_list),
        obj.atom_pairs,
        obj.atom_num_orbits_cumsum,
        keep_global,
    )
    overlap_path = _write_reduced_matrix_h5_streaming(
        config.output_dir / DEEPX_OVERLAP_FILENAME,
        sr_mats,
        np.asarray(obj.Rijk_list),
        obj.atom_pairs,
        obj.atom_num_orbits_cumsum,
        keep_global,
    )

    reduced_orbital_counts = []
    csum = obj.atom_num_orbits_cumsum
    keep_arr = np.array(keep_global)
    for ia in range(len(csum) - 1):
        a0 = int(csum[ia])
        a1 = int(csum[ia + 1])
        reduced_orbital_counts.append(int(np.sum((keep_arr >= a0) & (keep_arr < a1))))

    metadata = {
        "reduction_mode": config.reduction_mode,
        "removed_global_indices": remove_indices,
        "kept_global_indices": keep_global,
        "original_orbits_quantity": int(nb_total),
        "reduced_orbits_quantity": int(len(keep_global)),
        "reduced_orbitals_per_atom": reduced_orbital_counts,
        "rule_plan_resolution": plan_meta,
        "overlap_imag_max_before_real_cast": overlap_imag_max,
        "k_batch_size": int(k_batch_size),
        "r_batch_size": int(r_batch_size),
    }
    if parallel_workers is not None:
        metadata["parallel_k_workers"] = int(parallel_workers)

    info_path = write_reduced_info_json(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        elements=[str(el) for el in obj.elements],
        removed_indices=remove_indices,
    )

    meta_path = config.output_dir / "reduced_basis_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fw:
        json.dump(metadata, fw, indent=2)
        fw.write("\n")

    return ProjectionResult(
        output_dir=config.output_dir,
        hamiltonian_path=hamiltonian_path,
        overlap_path=overlap_path,
        info_path=info_path,
        meta_path=meta_path,
        metadata=metadata,
    )


def run_projection_batched(
    config: ProjectionConfig,
    removal_plan: RemovalPlanLike,
    k_batch_size: int = 16,
    r_batch_size: int = 32,
) -> ProjectionResult:
    """
    Run the k->R orbital-reduction projection with batched k-point accumulation.

    This keeps the batched implementation separate from the single-shot path and
    reduces peak RAM by only materializing k-point batches and a small R batch at a time.
    """
    if k_batch_size <= 0:
        raise ValueError(f"k_batch_size must be positive, got {k_batch_size}")
    if r_batch_size <= 0:
        raise ValueError(f"r_batch_size must be positive, got {r_batch_size}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.input_dir / DEEPX_POSCAR_FILENAME, config.output_dir / DEEPX_POSCAR_FILENAME)

    obj = HamiltonianObj(config.input_dir)

    plan_model = coerce_removal_plan(removal_plan)
    rm, plan_meta = resolve_indices_from_rules(
        elements=[str(el) for el in obj.elements],
        elements_orbital_map={k: [int(v) for v in vals] for k, vals in obj.elements_orbital_map.items()},
        plan=plan_model,
    )
    rm = sorted(set(rm))

    ks = build_uniform_kmesh(config.kgrid)
    total_nk = int(ks.shape[0])
    sample_Hk = obj.Sk_and_Hk(ks[:1])[1]
    nb_total = int(sample_Hk.shape[-1])
    keep_global = [i for i in range(nb_total) if i not in rm]
    reduced_nb = len(keep_global)

    if obj.Rijk_list is None:
        raise ValueError("Rijk_list is None")
    Rijk_list = np.asarray(obj.Rijk_list)
    nR = int(Rijk_list.shape[0])

    if reduced_nb == 0:
        raise ValueError("Cannot remove all orbitals")

    with tempfile.TemporaryDirectory(prefix="hamilflow_projection_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        hr_memmap = np.lib.format.open_memmap(
            tmpdir_path / "HR_new.npy",
            mode="w+",
            dtype=np.complex128,
            shape=(nR, reduced_nb, reduced_nb),
        )
        sr_memmap = np.lib.format.open_memmap(
            tmpdir_path / "SR_new.npy",
            mode="w+",
            dtype=np.complex128,
            shape=(nR, reduced_nb, reduced_nb),
        )

        k_weight = 1.0 / total_nk
        for k_slice in _iter_slices(total_nk, k_batch_size):
            ks_batch = ks[k_slice]
            Sk_batch, Hk_batch = obj.Sk_and_Hk(ks_batch)

            if config.reduction_mode == "schur":
                Hk_new, Sk_new = apply_custom_kspace_transform(Hk_batch, Sk_batch, remove_indices=rm)
            elif config.reduction_mode == "truncate":
                Hk_new, Sk_new = apply_truncation_kspace_transform(Hk_batch, Sk_batch, remove_indices=rm)
            else:
                raise ValueError(
                    f"Unsupported reduction_mode '{config.reduction_mode}'. Expected 'schur' or 'truncate'."
                )

            Hk_flat = Hk_new.reshape(Hk_new.shape[0], -1)
            Sk_flat = Sk_new.reshape(Sk_new.shape[0], -1)

            for r_slice in _iter_slices(nR, r_batch_size):
                R_batch = Rijk_list[r_slice]
                phase = np.exp(-2j * np.pi * np.matmul(R_batch, ks_batch.T))
                coeff = phase * k_weight

                hr_batch = np.matmul(coeff, Hk_flat).reshape(r_slice.stop - r_slice.start, reduced_nb, reduced_nb)
                sr_batch = np.matmul(coeff, Sk_flat).reshape(r_slice.stop - r_slice.start, reduced_nb, reduced_nb)

                hr_memmap[r_slice] += hr_batch
                sr_memmap[r_slice] += sr_batch

        return _finalize_projection_outputs(
            config=config,
            obj=obj,
            hr_mats=hr_memmap,
            sr_mats=sr_memmap,
            keep_global=keep_global,
            remove_indices=rm,
            plan_meta=plan_meta,
            nb_total=nb_total,
            k_batch_size=k_batch_size,
            r_batch_size=r_batch_size,
        )


def run_projection_batched_parallel(
    config: ProjectionConfig,
    removal_plan: RemovalPlanLike,
    k_batch_size: int = 16,
    r_batch_size: int = 32,
    n_workers: int = 2,
) -> ProjectionResult:
    """
    Run k-batched projection with process-level parallelism over k-point batches.

    Each worker computes one k-batch partial contribution and writes it to disk;
    the main process reduces partials into final real-space matrices.
    """
    if n_workers <= 0:
        raise ValueError(f"n_workers must be positive, got {n_workers}")
    if n_workers == 1:
        return run_projection_batched(
            config=config,
            removal_plan=removal_plan,
            k_batch_size=k_batch_size,
            r_batch_size=r_batch_size,
        )

    if k_batch_size <= 0:
        raise ValueError(f"k_batch_size must be positive, got {k_batch_size}")
    if r_batch_size <= 0:
        raise ValueError(f"r_batch_size must be positive, got {r_batch_size}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.input_dir / DEEPX_POSCAR_FILENAME, config.output_dir / DEEPX_POSCAR_FILENAME)

    obj = HamiltonianObj(config.input_dir)

    plan_model = coerce_removal_plan(removal_plan)
    rm, plan_meta = resolve_indices_from_rules(
        elements=[str(el) for el in obj.elements],
        elements_orbital_map={k: [int(v) for v in vals] for k, vals in obj.elements_orbital_map.items()},
        plan=plan_model,
    )
    rm = sorted(set(rm))

    ks = build_uniform_kmesh(config.kgrid)
    total_nk = int(ks.shape[0])
    sample_Hk = obj.Sk_and_Hk(ks[:1])[1]
    nb_total = int(sample_Hk.shape[-1])
    keep_global = [i for i in range(nb_total) if i not in rm]
    reduced_nb = len(keep_global)

    if obj.Rijk_list is None:
        raise ValueError("Rijk_list is None")
    Rijk_list = np.asarray(obj.Rijk_list)
    nR = int(Rijk_list.shape[0])

    if reduced_nb == 0:
        raise ValueError("Cannot remove all orbitals")

    k_slices = list(_iter_slices(total_nk, k_batch_size))
    if len(k_slices) == 1:
        return run_projection_batched(
            config=config,
            removal_plan=removal_plan,
            k_batch_size=k_batch_size,
            r_batch_size=r_batch_size,
        )

    with tempfile.TemporaryDirectory(prefix="hamilflow_projection_parallel_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        partial_dir = tmpdir_path / "partials"
        partial_dir.mkdir(parents=True, exist_ok=True)

        hr_memmap = np.lib.format.open_memmap(
            tmpdir_path / "HR_new.npy",
            mode="w+",
            dtype=np.complex128,
            shape=(nR, reduced_nb, reduced_nb),
        )
        sr_memmap = np.lib.format.open_memmap(
            tmpdir_path / "SR_new.npy",
            mode="w+",
            dtype=np.complex128,
            shape=(nR, reduced_nb, reduced_nb),
        )
        hr_memmap.fill(0)
        sr_memmap.fill(0)

        ctx = mp.get_context("spawn")
        k_weight = 1.0 / total_nk
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                str(config.input_dir),
                config.reduction_mode,
                rm,
                Rijk_list,
                reduced_nb,
                r_batch_size,
                k_weight,
            ),
        ) as pool:
            futures = [
                pool.submit(
                    _worker_compute_k_batch,
                    task_id,
                    ks[k_slice],
                    str(partial_dir),
                )
                for task_id, k_slice in enumerate(k_slices)
            ]

            for future in as_completed(futures):
                hr_path_str, sr_path_str = future.result()
                hr_part = np.load(hr_path_str, mmap_mode="r")
                sr_part = np.load(sr_path_str, mmap_mode="r")
                hr_memmap += hr_part
                sr_memmap += sr_part
                del hr_part
                del sr_part
                Path(hr_path_str).unlink(missing_ok=True)
                Path(sr_path_str).unlink(missing_ok=True)

        return _finalize_projection_outputs(
            config=config,
            obj=obj,
            hr_mats=hr_memmap,
            sr_mats=sr_memmap,
            keep_global=keep_global,
            remove_indices=rm,
            plan_meta=plan_meta,
            nb_total=nb_total,
            k_batch_size=k_batch_size,
            r_batch_size=r_batch_size,
            parallel_workers=n_workers,
        )