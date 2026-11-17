"""
End-to-end test of the AIMS -> DeepH -> projected DeepH workflow on graphene, light C basis:

    aims_static -> collect -> convert -> projection_schur (1s) -> projection_truncate (1d)

Both variants run the same case: underconverged settings (2x2x1 k-grid, loose SCF) and the light C species file
stored in data/aims_pipeline/species, so the generated inputs do not depend on the local FHI-aims installation.

``real``: runs the FHI-aims executable, a few seconds on 4 cores. It needs ``AIMS_CMD`` in the atomate2 config
file and fails, not skips, without it. Marked ``aims``, which is deselected by default: run it with
``pytest -m aims``.

``mocked``: no FHI-aims. atomate2's ``run_aims`` is replaced by a check of the generated ``control.in`` and
``geometry.in`` followed by a copy of the outputs of a recorded real run, then the rest of the pipeline runs for
real. Re-record the outputs with ``python record_mock_aims.py`` whenever the case's aims settings change.
"""

import gzip
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME, DEEPX_INFO_FILENAME, DEEPX_OVERLAP_FILENAME

DATA = Path(__file__).parent / "data" / "aims_pipeline"
MOCK_OUTPUTS = DATA / "mock_outputs"
# What the converter and atomate2's parser read from a run directory, stored gzipped in MOCK_OUTPUTS
MOCK_OUTPUT_FILES = ("aims.out", "basis-indices.out", "rs_indices.out", "rs_overlap.out", "rs_hamiltonian.out")

STRUCTURE = "graphene"
ELEMENTS = ["C", "C"]
UNDERCONVERGED_AIMS_KWARGS = {
    "xc": "pbe",
    # Loose SCF: ~6 cycles instead of ~11, the run time being the grid integrations of every cycle
    "sc_accuracy_rho": 1e-2,
    "sc_accuracy_eev": 1.0,
    "sc_accuracy_etot": "1e-1 eV",
    "sc_iter_limit": 100,
}
KGRID = (2, 2, 1)
L_OF = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}


@dataclass(frozen=True)
class PipelineCase:
    species_dir: Path  # <species dir>/<basis>, holding 06_C_default
    schur_plan: Path
    schur_shell: str  # the shell removed by schur_plan
    truncate_plan: Path
    truncated_shell: str  # the shell removed by truncate_plan

    def aims_kwargs(self) -> dict:
        return {"species_dir": str(self.species_dir), **UNDERCONVERGED_AIMS_KWARGS}


# Light C: 1s, 2s, 2p, then 2p, 3d, 2s hydrogenic functions, i.e. 14 orbitals per atom (39 in tight)
CASE = PipelineCase(
    species_dir=DATA / "species" / "light",
    schur_plan=DATA / "removal_plan_schur.json",
    schur_shell="1s",
    truncate_plan=DATA / "removal_plan_truncate.json",
    truncated_shell="1d",
)


def require_aims_cmd() -> None:
    """Fail loudly unless the atomate2 config file tells atomate2 how to run FHI-aims."""
    from atomate2.settings import Atomate2Settings
    from monty.serialization import loadfn

    config_file = Path(Atomate2Settings().CONFIG_FILE).expanduser()
    if not config_file.is_file():
        pytest.fail(f"atomate2 config file {config_file} does not exist: add AIMS_CMD to it to run the aims tests")
    if "AIMS_CMD" not in (loadfn(config_file) or {}):
        pytest.fail(f"no AIMS_CMD in the atomate2 config file {config_file}: add it to run the aims tests")


def build_pipeline(root: Path, case: PipelineCase):
    from hamilflow.dft_workflows import (
        ConvertAimsToDeephConfig,
        GenerateAimsDFTData,
        GenerateAimsToProjectedDeephData,
        ProjectDeephInputsConfig,
    )

    dft = GenerateAimsDFTData(
        structures_path=DATA / "structures",
        aims_kwargs=case.aims_kwargs(),
        kgrid=KGRID,
        collected_runs_root=root / "aims_output",
        aims_to_deeph_config=ConvertAimsToDeephConfig(output_dir=root / "deeph_full"),
    )
    return GenerateAimsToProjectedDeephData(
        dft_data_flow=dft,
        projection_config=ProjectDeephInputsConfig(
            output_root=root / "deeph_projected_schur",
            removal_plan=case.schur_plan,
            reduction_mode="schur",
            n_workers=1,
        ),
        second_projection_config=ProjectDeephInputsConfig(
            output_root=root / "deeph_projected_truncated",
            removal_plan=case.truncate_plan,
            reduction_mode="truncate",
            n_workers=1,
        ),
    ).make()


# --- mocked FHI-aims ----------------------------------------------------------------------------------------------


def parse_control_in(text: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Global ``keyword value`` lines of a control.in, and the lines of each species block.

    Comments and blank lines are dropped; a species block runs from its ``species`` line to the next one.
    """
    settings: dict[str, str] = {}
    species: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        keyword, _, value = line.partition(" ")
        if keyword == "species":
            current = value.strip()
            species[current] = []
        elif current is None:
            settings[keyword] = " ".join(value.split())
        else:
            species[current].append(" ".join(line.split()))
    return settings, species


def check_control_in(path: Path, case: PipelineCase) -> None:
    """The settings the pipeline depends on, and the C species block of the case's species file."""
    settings, species = parse_control_in(path.read_text())
    expected = {
        "xc": "pbe",
        "k_grid": " ".join(str(k) for k in KGRID),
        "output_rs_matrices": "plain",  # the real-space H and S the converter reads
        "use_local_index": "false",
        "density_update_method": "density_matrix",
        "sc_accuracy_rho": "0.01",
        "sc_iter_limit": "100",
    }
    assert {key: settings.get(key) for key in expected} == expected

    # pymatgen rewrites the species file with its own number format (nucleus 6 -> 6.0, 1e-4 -> 0.0001)
    assert list(species) == ["C"]
    reference = parse_control_in((case.species_dir / "06_C_default").read_text())[1]["C"]
    assert normalize_numbers(species["C"]) == normalize_numbers(reference), (
        f"the C species block is not the one of {case.species_dir}"
    )


def normalize_numbers(lines: list[str]) -> list[tuple]:
    def token(word: str) -> str | float:
        try:
            return float(word)
        except ValueError:
            return word

    return [tuple(token(word) for word in line.split()) for line in lines]


def check_geometry_in(path: Path) -> None:
    """Same lattice and atoms as the input structure, up to the rigid shift applied to coordinates near 0.5."""
    from pymatgen.core import Structure
    from pymatgen.io.aims.inputs import AimsGeometryIn

    written = AimsGeometryIn.from_file(path).structure
    reference = Structure.from_file(DATA / "structures" / STRUCTURE / "POSCAR")
    np.testing.assert_allclose(written.lattice.matrix, reference.lattice.matrix, atol=1e-6)
    assert [str(site.specie) for site in written] == [str(site.specie) for site in reference]
    shift = written.frac_coords - reference.frac_coords
    shift -= np.round(shift)
    np.testing.assert_allclose(shift, np.broadcast_to(shift[0], shift.shape), atol=1e-6)


def make_fake_run_aims(case: PipelineCase):
    """A stand-in for atomate2's run_aims: check the inputs written in the job directory, then copy the outputs."""

    def fake_run_aims(*args, **kwargs) -> None:
        cwd = Path.cwd()
        check_control_in(cwd / "control.in", case)
        check_geometry_in(cwd / "geometry.in")
        for name in MOCK_OUTPUT_FILES:
            with gzip.open(MOCK_OUTPUTS / f"{name}.gz", "rb") as fi, open(cwd / name, "wb") as fo:
                shutil.copyfileobj(fi, fo)

    return fake_run_aims


# --- the pipeline run -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineRun:
    root: Path
    case: PipelineCase


@pytest.fixture(
    params=[
        pytest.param("real", marks=pytest.mark.aims),
        pytest.param("mocked"),
    ]
)
def pipeline_run(request, tmp_path, monkeypatch) -> PipelineRun:
    """Run the whole pipeline on graphene under tmp_path, with real or mocked FHI-aims."""
    from jobflow.managers.local import run_locally

    case = CASE
    if request.param == "real":
        require_aims_cmd()
    else:
        import atomate2.aims.jobs.base
        import atomate2.aims.run

        fake = make_fake_run_aims(case)
        monkeypatch.setattr(atomate2.aims.run, "run_aims", fake)
        monkeypatch.setattr(atomate2.aims.jobs.base, "run_aims", fake)

    run_locally(build_pipeline(tmp_path, case), root_dir=tmp_path / "jobs", create_folders=True, ensure_success=True)
    return PipelineRun(tmp_path, case)


def read_info(deeph_dir: Path) -> dict:
    return json.loads((deeph_dir / DEEPX_INFO_FILENAME).read_text())


def remove_first_shell(orbital_map: list[int], shell: str) -> list[int]:
    """Drop the first shell of angular momentum l from a per-element list of shell l values ("1s", "1g", ...)."""
    assert shell[0] == "1", "only first shells are handled here"
    i = orbital_map.index(L_OF[shell[1]])
    return orbital_map[:i] + orbital_map[i + 1 :]


def shell_size(shell: str) -> int:
    """Orbitals in one shell: 2l + 1."""
    return 2 * L_OF[shell[1]] + 1


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
def test_aims_to_projected_deeph_pipeline(pipeline_run):
    root, case = pipeline_run.root, pipeline_run.case
    for plan, shell in ((case.schur_plan, case.schur_shell), (case.truncate_plan, case.truncated_shell)):
        rules = json.loads(plan.read_text())["rules"]
        assert [rule["remove_shells"] for rule in rules] == [[shell]]

    # collect: the AIMS run is moved under aims_output/<structure>
    collected = root / "aims_output" / STRUCTURE
    assert collected.is_dir()
    assert "Have a nice day" in _read_maybe_gz(collected, "aims.out")

    # convert: the full basis, two C atoms
    full_dir = root / "deeph_full" / STRUCTURE
    full = read_info(full_dir)
    assert full["atoms_quantity"] == len(ELEMENTS)
    full_map = full["elements_orbital_map"]
    assert set(full_map) == {"C"}
    assert L_OF[case.truncated_shell[1]] in full_map["C"], f"the C basis has no {case.truncated_shell} shell"
    check_h5_layout(full_dir, full_map, ELEMENTS)

    # Schur: the 1s shell is folded out, then truncate drops its shell
    previous_info, previous_map = full, full_map
    for stage, shell in (("deeph_projected_schur", case.schur_shell), ("deeph_projected_truncated", case.truncated_shell)):
        stage_dir = root / stage / STRUCTURE
        info = read_info(stage_dir)
        expected_map = {"C": remove_first_shell(previous_map["C"], shell)}
        assert info["elements_orbital_map"] == expected_map, stage
        assert info["orbits_quantity"] == previous_info["orbits_quantity"] - len(ELEMENTS) * shell_size(shell), stage
        assert info["atoms_quantity"] == len(ELEMENTS)
        assert info["spinful"] == full["spinful"]
        check_h5_layout(stage_dir, expected_map, ELEMENTS)
        previous_info, previous_map = info, expected_map


def _read_maybe_gz(directory: Path, name: str) -> str:
    if (directory / name).exists():
        return (directory / name).read_text()
    with gzip.open(directory / f"{name}.gz", "rt") as f:
        return f.read()
