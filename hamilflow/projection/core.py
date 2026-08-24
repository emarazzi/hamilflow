from __future__ import annotations

import json
import shutil

import numpy as np
from pymatgen.core import Structure

from deepx_dock.CONSTANT import (
    DEEPX_HAMILTONIAN_FILENAME,
    DEEPX_OVERLAP_FILENAME,
    DEEPX_POSCAR_FILENAME,
)
from hamilflow.sparse_hamiltonian import SparseHamiltonianObj

from .io import dump_reduced_matrix_h5, hermitize_real_space_blocks, write_reduced_info_json
from .kspace import (
    available_cpu_count,
    build_uniform_kmesh,
    stream_project_to_real_space,
)
from .models import ProjectionConfig, ProjectionResult, RemovalPlanLike
from .removal import coerce_removal_plan, resolve_indices_from_rules


def _resolve_projection_kgrid(config: ProjectionConfig) -> tuple[int, int, int]:
    if config.user_kpoints_settings not in (None, {}):
        from ..dft_workflows.kpoints import get_ksampling

        structure = Structure.from_file(config.input_dir / DEEPX_POSCAR_FILENAME)
        ksampling = get_ksampling(
            structure=structure,
            user_kpoints_settings=config.user_kpoints_settings,
            force_2d=config.force_2d,
        )
        if not ksampling or "k_grid" not in ksampling:
            raise ValueError(
                "user_kpoints_settings must resolve to a uniform k_grid for projection."
            )
        k_grid = ksampling["k_grid"]
        if len(k_grid) != 3:
            raise ValueError(f"Resolved k_grid must contain three integers, got: {k_grid}")
        return (int(k_grid[0]), int(k_grid[1]), int(k_grid[2]))

    if config.force_2d:
        return (int(config.kgrid[0]), int(config.kgrid[1]), 1)
    return (int(config.kgrid[0]), int(config.kgrid[1]), int(config.kgrid[2]))


def run_projection(
    config: ProjectionConfig,
    removal_plan: RemovalPlanLike,
) -> ProjectionResult:
    """
    Run the k->R orbital-reduction projection.

    The reduction plan can be provided as a model, dict/list payload, or JSON file path.
    By default, both Hamiltonian and overlap are written to files in config.output_dir.
    When overlap_only=True, only the overlap is processed. If write_dummy_hamiltonian=True,
    a zero-filled hamiltonian.h5 is also written to enable sequential projections.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config.input_dir / DEEPX_POSCAR_FILENAME, config.output_dir / DEEPX_POSCAR_FILENAME)
    
    # HamiltonianObj requires hamiltonian.h5 to exist to load metadata (elements, atom_pairs, etc.)
    # even in overlap_only mode where we don't process the Hamiltonian itself
    hamiltonian_file = config.input_dir / DEEPX_HAMILTONIAN_FILENAME
    if not hamiltonian_file.exists():
        raise FileNotFoundError(
            f"Hamiltonian file '{DEEPX_HAMILTONIAN_FILENAME}' not found in {config.input_dir}.\n"
            f"Note: overlap_only mode still requires this file to exist for loading structure metadata "
            f"(elements, atom pairs, etc.), even though the Hamiltonian won't be processed or written to output.\n"
            f"To generate a dummy hamiltonian.h5 file with matching structure, you can use:\n"
            f"  deepx_dock write --output hamiltonian.h5 [other args]"
        )
    
    obj = SparseHamiltonianObj(config.input_dir, load_hamiltonian=not config.overlap_only)

    plan_model = coerce_removal_plan(removal_plan)
    rm, plan_meta = resolve_indices_from_rules(
        elements=[str(el) for el in obj.elements],
        elements_orbital_map={k: [int(v) for v in vals] for k, vals in obj.elements_orbital_map.items()},
        plan=plan_model,
    )

    rm = sorted(set(rm))
    n_workers = config.n_workers if config.n_workers is not None else available_cpu_count()

    resolved_kgrid = _resolve_projection_kgrid(config)
    ks = build_uniform_kmesh(resolved_kgrid)
    nb = int(obj.orbits_quantity) * (2 if obj.spinful else 1)
    keep_global = [i for i in range(nb) if i not in rm]

    if obj.Rijk_list is None:
        raise ValueError("Rijk_list is None")

    # Streams the k-mesh in small chunks (chunk -> transform -> IFT ->
    # accumulate) instead of materializing dense Hk/Sk for the whole mesh at
    # once; see stream_project_to_real_space for why this is required, not
    # just an optimization.
    HR_new, SR_new = stream_project_to_real_space(
        obj,
        ks,
        remove_indices=rm,
        reduction_mode=config.reduction_mode,
        Rijk_list=obj.Rijk_list,
        n_workers=n_workers,
        overlap_only=config.overlap_only,
    )

    # Process and store Hamiltonian only if not overlap_only mode
    if not config.overlap_only:
        assert HR_new is not None
        HR_new = hermitize_real_space_blocks(HR_new, obj.Rijk_list)
        hamiltonian_path = dump_reduced_matrix_h5(
            config.output_dir / DEEPX_HAMILTONIAN_FILENAME,
            HR_new,
            obj.Rijk_list,
            obj.atom_pairs,
            obj.atom_num_orbits_cumsum,
            keep_global,
        )
    elif config.overlap_only and config.write_dummy_hamiltonian:
        # Write dummy zeros hamiltonian for chaining overlaps in sequence
        HR_dummy = np.zeros_like(SR_new, dtype=np.float64)
        hamiltonian_path = dump_reduced_matrix_h5(
            config.output_dir / DEEPX_HAMILTONIAN_FILENAME,
            HR_dummy,
            obj.Rijk_list,
            obj.atom_pairs,
            obj.atom_num_orbits_cumsum,
            keep_global,
        )
    else:
        hamiltonian_path = None

    # Process and store Overlap
    SR_new = hermitize_real_space_blocks(SR_new, obj.Rijk_list)

    overlap_imag_max = float(np.max(np.abs(np.imag(SR_new)))) if SR_new.size > 0 else 0.0
    SR_new = np.asarray(np.real(SR_new), dtype=np.float64)

    overlap_path = dump_reduced_matrix_h5(
        config.output_dir / DEEPX_OVERLAP_FILENAME,
        SR_new,
        obj.Rijk_list,
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
        "overlap_only": config.overlap_only,
        "write_dummy_hamiltonian": config.write_dummy_hamiltonian,
        "resolved_kgrid": list(resolved_kgrid),
        "removed_global_indices": rm,
        "kept_global_indices": keep_global,
        "original_orbits_quantity": int(nb),
        "reduced_orbits_quantity": int(len(keep_global)),
        "reduced_orbitals_per_atom": reduced_orbital_counts,
        "rule_plan_resolution": plan_meta,
        "overlap_imag_max_before_real_cast": overlap_imag_max,
    }

    info_path = write_reduced_info_json(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        elements=[str(el) for el in obj.elements],
        removed_indices=rm,
        reduction_mode=config.reduction_mode,
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
