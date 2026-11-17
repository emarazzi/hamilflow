"""
End-to-end test of the AIMS -> DeepH -> projected DeepH workflow on graphene:

    aims_static -> collect -> convert -> projection_schur (1s) -> projection_truncate (1g)

``real``: runs the FHI-aims executable, with underconverged settings (2x2x1 k-grid, loose SCF): about 40 s on 4
cores. It needs ``AIMS_CMD`` in the atomate2 config file and ``AIMS_SPECIES_DIR`` in the pymatgen config
(``~/.config/.pmgrc.yaml``) and fails, not skips, without them. Marked ``aims``, which is deselected by default:
run it with ``pytest -m aims``.

``mocked``: placeholder for the same pipeline without FHI-aims, e.g. by patching atomate2's ``run_aims`` to copy
the outputs of a recorded real run. Not implemented yet.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME, DEEPX_INFO_FILENAME, DEEPX_OVERLAP_FILENAME

DATA = Path(__file__).parent / "data" / "aims_pipeline"
STRUCTURE = "graphene"
# The truncate plan removes the 1g shell: only the tight C species has a g shell
SPECIES = "tight"
UNDERCONVERGED_AIMS_KWARGS = {
    "species_dir": SPECIES,
    "xc": "pbe",
    # Loose SCF: ~6 cycles instead of ~11, the run time being the tight-grid integrations of every cycle
    "sc_accuracy_rho": 1e-2,
    "sc_accuracy_eev": 1.0,
    "sc_accuracy_etot": "1e-1 eV",
    "sc_iter_limit": 100,
}
KGRID = (2, 2, 1)
L_OF = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


def require_aims_config() -> None:
    """Fail loudly unless atomate2 knows how to run FHI-aims and pymatgen where its species are."""
    from atomate2.settings import Atomate2Settings
    from monty.serialization import loadfn
    from pymatgen.core import SETTINGS as PMG_SETTINGS

    config_file = Path(Atomate2Settings().CONFIG_FILE).expanduser()
    if not config_file.is_file():
        pytest.fail(f"atomate2 config file {config_file} does not exist: add AIMS_CMD to it to run the aims tests")
    if "AIMS_CMD" not in (loadfn(config_file) or {}):
        pytest.fail(f"no AIMS_CMD in the atomate2 config file {config_file}: add it to run the aims tests")

    species_dir = PMG_SETTINGS.get("AIMS_SPECIES_DIR")
    if species_dir is None:
        pytest.fail("no AIMS_SPECIES_DIR in the pymatgen config (~/.config/.pmgrc.yaml): add it to run the aims tests")
    species_file = Path(species_dir) / SPECIES / "06_C_default"
    if not species_file.is_file():
        pytest.fail(f"AIMS_SPECIES_DIR has no {SPECIES} C species: {species_file} does not exist")


def build_pipeline(root: Path):
    from hamilflow.dft_workflows import (
        ConvertAimsToDeephConfig,
        GenerateAimsDFTData,
        GenerateAimsToProjectedDeephData,
        ProjectDeephInputsConfig,
    )

    dft = GenerateAimsDFTData(
        structures_path=DATA / "structures",
        aims_kwargs=UNDERCONVERGED_AIMS_KWARGS,
        kgrid=KGRID,
        collected_runs_root=root / "aims_output",
        aims_to_deeph_config=ConvertAimsToDeephConfig(output_dir=root / "deeph_full"),
    )
    return GenerateAimsToProjectedDeephData(
        dft_data_flow=dft,
        projection_config=ProjectDeephInputsConfig(
            output_root=root / "deeph_projected_schur",
            removal_plan=DATA / "removal_plan_schur.json",
            reduction_mode="schur",
            n_workers=1,
        ),
        second_projection_config=ProjectDeephInputsConfig(
            output_root=root / "deeph_projected_truncated",
            removal_plan=DATA / "removal_plan_truncate.json",
            reduction_mode="truncate",
            n_workers=1,
        ),
    ).make()


@pytest.fixture(
    params=[
        pytest.param("real", marks=pytest.mark.aims),
        pytest.param("mocked", marks=pytest.mark.skip(reason="mocked FHI-aims not implemented yet")),
    ]
)
def pipeline_root(request, tmp_path) -> Path:
    """Run the whole pipeline on graphene under tmp_path, with real or mocked FHI-aims, and return tmp_path."""
    from jobflow.managers.local import run_locally

    if request.param == "real":
        require_aims_config()
    else:
        raise NotImplementedError("mocked FHI-aims")

    run_locally(build_pipeline(tmp_path), root_dir=tmp_path / "jobs", create_folders=True, ensure_success=True)
    return tmp_path


def read_info(deeph_dir: Path) -> dict:
    return json.loads((deeph_dir / DEEPX_INFO_FILENAME).read_text())


def remove_first_shell(orbital_map: list[int], shell: str) -> list[int]:
    """Drop the first shell of angular momentum l from a per-element list of shell l values ("1s", "1g", ...)."""
    assert shell[0] == "1", "only first shells are handled here"
    l = L_OF[shell[1]]
    i = orbital_map.index(l)
    return orbital_map[:i] + orbital_map[i + 1 :]


def check_h5_layout(deeph_dir: Path, orbital_map: dict[str, list[int]], elements: list[str]) -> None:
    """Every H and S block has the shape set by the orbitals of its two atoms, and the two files match."""
    n_orb = {el: sum(2 * l + 1 for l in ls) for el, ls in orbital_map.items()}
    files = {}
    for name in (DEEPX_OVERLAP_FILENAME, DEEPX_HAMILTONIAN_FILENAME):
        with h5py.File(deeph_dir / name, "r") as f:
            files[name] = {key: f[key][:] for key in ("atom_pairs", "chunk_boundaries", "chunk_shapes")}
            files[name]["n_entries"] = f["entries"].shape[0]
    S, H = files[DEEPX_OVERLAP_FILENAME], files[DEEPX_HAMILTONIAN_FILENAME]

    np.testing.assert_array_equal(S["atom_pairs"], H["atom_pairs"])
    np.testing.assert_array_equal(S["chunk_shapes"], H["chunk_shapes"])
    for data in (S, H):
        expected = [(n_orb[elements[i]], n_orb[elements[j]]) for i, j in data["atom_pairs"][:, 3:5]]
        np.testing.assert_array_equal(data["chunk_shapes"], expected)
        np.testing.assert_array_equal(np.diff(data["chunk_boundaries"]), np.prod(data["chunk_shapes"], axis=1))
        assert data["chunk_boundaries"][-1] == data["n_entries"]
    assert (S["atom_pairs"][:, :3] == 0).all(axis=1).any(), "no on-site R = 0 block"


@pytest.mark.slow
@pytest.mark.integration
def test_aims_to_projected_deeph_pipeline(pipeline_root):
    schur_plan = json.loads((DATA / "removal_plan_schur.json").read_text())["rules"]
    truncate_plan = json.loads((DATA / "removal_plan_truncate.json").read_text())["rules"]
    assert [rule["remove_shells"] for rule in schur_plan + truncate_plan] == [["1s"], ["1g"]]

    # collect: the AIMS run is moved under aims_output/<structure>
    collected = pipeline_root / "aims_output" / STRUCTURE
    assert collected.is_dir()
    assert any((collected / name).exists() for name in ("aims.out", "aims.out.gz"))
    assert "Have a nice day" in _read_maybe_gz(collected, "aims.out")

    # convert: full tight basis, two C atoms
    full_dir = pipeline_root / "deeph_full" / STRUCTURE
    full = read_info(full_dir)
    assert full["atoms_quantity"] == 2
    full_map = full["elements_orbital_map"]
    assert set(full_map) == {"C"}
    assert 4 in full_map["C"], "the tight C basis must have a g shell for the truncate stage"
    check_h5_layout(full_dir, full_map, ["C", "C"])

    # Schur: the 1s shell is folded out
    schur_dir = pipeline_root / "deeph_projected_schur" / STRUCTURE
    schur = read_info(schur_dir)
    schur_expected = {"C": remove_first_shell(full_map["C"], "1s")}
    assert schur["elements_orbital_map"] == schur_expected
    assert schur["orbits_quantity"] == full["orbits_quantity"] - 2 * 1
    check_h5_layout(schur_dir, schur_expected, ["C", "C"])

    # truncate: the 1g shell is dropped
    trunc_dir = pipeline_root / "deeph_projected_truncated" / STRUCTURE
    trunc = read_info(trunc_dir)
    trunc_expected = {"C": remove_first_shell(schur_expected["C"], "1g")}
    assert trunc["elements_orbital_map"] == trunc_expected
    assert trunc["orbits_quantity"] == schur["orbits_quantity"] - 2 * 9
    check_h5_layout(trunc_dir, trunc_expected, ["C", "C"])

    for info in (schur, trunc):
        assert info["atoms_quantity"] == 2
        assert info["spinful"] == full["spinful"]


def _read_maybe_gz(directory: Path, name: str) -> str:
    import gzip

    if (directory / name).exists():
        return (directory / name).read_text()
    with gzip.open(directory / f"{name}.gz", "rt") as f:
        return f.read()
