"""Structure -> projected overlap in the DeepH basis the models are trained on, without FHI-aims.

    full-basis S from the two-centre table (`assemble`)  ->  Schur (1s)  ->  truncate (g)

Two projection paths, same removal plans as production:

* ``"A"`` (parity path): the production k-space `hamilflow.projection.run_projection`, overlap
  only, Schur then truncation, default k-grid (R span). CPU, slow for large cells.
* ``"B"`` (fast path): `schur_rs.project_overlap_rs`, real space, Neumann series for S_mm^-1.

Intended use: screening a structure pool (rank structures by model spread before running the
real overlap-only FHI-aims chain). FHI-aims' own S differs from any two-centre S by its grid
integration noise (~1e-4 off-site, ~1e-3 on-site for tight C), so the built S is not a
replacement for the real one where parity matters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

from hamilflow.projection import ProjectionConfig, run_projection
from hamilflow.projection.models import RemovalPlanLike

from .assemble import assemble_overlap
from .io import DeephStructure
from .schur_rs import project_overlap_rs
from .tables import TwoCentreTable

logger = logging.getLogger(__name__)

FULL, SCHUR, TRUNCATED = "deeph_full", "deeph_projected_schur", "deeph_projected_truncated"


def project_full_overlap(
    full_dir: Path | str,
    out_root: Path | str,
    name: str,
    schur_plan: RemovalPlanLike,
    truncate_plan: RemovalPlanLike,
    path: Literal["A", "B"] = "B",
    n_terms: int = 2,
    n_workers: int | None = None,
) -> Path:
    """Project a full-basis overlap-only DeepH directory; returns the truncated output dir
    (``out_root/deeph_projected_truncated[_A|_B]/name``)."""
    out_root = Path(out_root)
    if path == "A":
        schur_dir = out_root / f"{SCHUR}_A" / name
        trunc_dir = out_root / f"{TRUNCATED}_A" / name
        run_projection(ProjectionConfig(input_dir=Path(full_dir), output_dir=schur_dir, reduction_mode="schur",
                                        overlap_only=True, write_dummy_hamiltonian=True, n_workers=n_workers),
                       schur_plan)
        run_projection(ProjectionConfig(input_dir=schur_dir, output_dir=trunc_dir, reduction_mode="truncate",
                                        overlap_only=True, write_dummy_hamiltonian=True, n_workers=n_workers),
                       truncate_plan)
        return trunc_dir
    if path == "B":
        trunc_dir = out_root / f"{TRUNCATED}_B" / name
        project_overlap_rs(full_dir, trunc_dir, schur_plan, truncate_plan, n_terms=n_terms)
        return trunc_dir
    raise ValueError(f"path must be 'A' or 'B', got {path!r}")


def build_projected_overlap(
    structure: DeephStructure,
    name: str,
    table: TwoCentreTable,
    out_root: Path | str,
    schur_plan: RemovalPlanLike,
    truncate_plan: RemovalPlanLike,
    path: Literal["A", "B"] = "B",
    xp: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Assemble ``out_root/deeph_full/name`` and project it (see `project_full_overlap`)."""
    t0 = time.monotonic()
    full_dir = Path(out_root) / FULL / name
    stats = assemble_overlap(structure, table, full_dir, **({} if xp is None else {"xp": xp}))
    t1 = time.monotonic()
    out = project_full_overlap(full_dir, out_root, name, schur_plan, truncate_plan, path=path, **kwargs)
    return {**stats, "projected_dir": str(out), "assemble_seconds": t1 - t0,
            "project_seconds": time.monotonic() - t1}
