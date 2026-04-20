from __future__ import annotations

import json
import shutil

import numpy as np

from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj
from deepx_dock.CONSTANT import (
    DEEPX_HAMILTONIAN_FILENAME,
    DEEPX_OVERLAP_FILENAME,
    DEEPX_POSCAR_FILENAME,
)

from .io import dump_reduced_matrix_h5, hermitize_real_space_blocks, write_reduced_info_json
from .kspace import (
    apply_custom_kspace_transform,
    apply_truncation_kspace_transform,
    build_uniform_kmesh,
    hk_and_sk_to_real,
)
from .models import ProjectionConfig, ProjectionResult, RemovalPlanLike
from .removal import coerce_removal_plan, resolve_indices_from_rules


def run_projection(
    config: ProjectionConfig,
    removal_plan: RemovalPlanLike,
) -> ProjectionResult:
    """
    Run the k->R orbital-reduction projection.

    The reduction plan can be provided as a model, dict/list payload, or JSON file path.
    Hamiltonian and overlap are always written to files in config.output_dir.
    """
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
    Sk, Hk = obj.Sk_and_Hk(ks)
    nb = Sk.shape[-1]
    keep_global = [i for i in range(nb) if i not in rm]

    if config.reduction_mode == "schur":
        Hk_new, Sk_new = apply_custom_kspace_transform(Hk, Sk, remove_indices=rm)
    elif config.reduction_mode == "truncate":
        Hk_new, Sk_new = apply_truncation_kspace_transform(Hk, Sk, remove_indices=rm)
    else:
        raise ValueError(
            f"Unsupported reduction_mode '{config.reduction_mode}'. Expected 'schur' or 'truncate'."
        )

    if obj.Rijk_list is None:
        raise ValueError("Rijk_list is None")
    HR_new, SR_new = hk_and_sk_to_real(
        ks=ks,
        Hk=Hk_new,
        Sk=Sk_new,
        Rijk_list=obj.Rijk_list,
    )

    HR_new = hermitize_real_space_blocks(HR_new, obj.Rijk_list)
    SR_new = hermitize_real_space_blocks(SR_new, obj.Rijk_list)

    overlap_imag_max = float(np.max(np.abs(np.imag(SR_new)))) if SR_new.size > 0 else 0.0
    SR_new = np.asarray(np.real(SR_new), dtype=np.float64)

    hamiltonian_path = dump_reduced_matrix_h5(
        config.output_dir / DEEPX_HAMILTONIAN_FILENAME,
        HR_new,
        obj.Rijk_list,
        obj.atom_pairs,
        obj.atom_num_orbits_cumsum,
        keep_global,
    )
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
