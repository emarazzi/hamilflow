import pytest
from ase.build import bulk
from ase.io import write
from atomate2.aims.jobs.core import StaticMaker
from pymatgen.io.aims.sets.core import StaticSetGenerator

from hamilflow.dft_workflows import load_structures
from hamilflow.dft_workflows.flows_base import GenerateAimsDFTData
from hamilflow.dft_workflows.jobs import build_aims_dft_job


@pytest.fixture
def trajectory(tmp_path):
    frames = []
    for i, k_grid in enumerate([[4, 5, 6], [2, 2, 1]]):
        atoms = bulk("Si")
        atoms.info["index"] = i
        atoms.info["k_grid"] = k_grid
        frames.append(atoms)
    path = tmp_path / "frames.extxyz"
    write(path, frames, format="extxyz")
    return path


def _user_params(aims_job):
    return aims_job.maker.input_set_generator.user_params


def test_kgrid_read_from_trajectory_info(trajectory):
    maker = StaticMaker(input_set_generator=StaticSetGenerator())
    grids = [
        _user_params(build_aims_dft_job(s, name, maker, kgrid_from_structure=True))["k_grid"]
        for name, s in load_structures(trajectory)
    ]
    assert grids == [[4, 5, 6], [2, 2, 1]]


def test_kgrid_from_structure_respects_force_2d(trajectory):
    maker = StaticMaker(input_set_generator=StaticSetGenerator())
    name, s = load_structures(trajectory)[0]
    aims_job = build_aims_dft_job(s, name, maker, kgrid_from_structure=True, force_2d=True)
    assert _user_params(aims_job)["k_grid"] == [4, 5, 1]


def test_kgrid_from_structure_missing_entry_raises():
    from pymatgen.io.ase import AseAtomsAdaptor

    s = AseAtomsAdaptor.get_structure(bulk("Si"))
    maker = StaticMaker(input_set_generator=StaticSetGenerator())
    with pytest.raises(ValueError, match="no 'k_grid'"):
        build_aims_dft_job(s, "si", maker, kgrid_from_structure=True)


def test_kgrid_from_structure_conflicts_with_kgrid(trajectory):
    with pytest.raises(ValueError, match="kgrid_from_structure"):
        GenerateAimsDFTData(structures_path=trajectory, kgrid=(4, 4, 4), kgrid_from_structure=True)


def test_flow_passes_kgrid_from_structure(trajectory):
    flow = GenerateAimsDFTData(structures_path=trajectory, kgrid_from_structure=True)
    grids = [
        _user_params(flow.make_structure_jobs(name, s)[0][0])["k_grid"]
        for name, s in flow.structure_entries()
    ]
    assert grids == [[4, 5, 6], [2, 2, 1]]


# --- projections -----------------------------------------------------------------


def _projection_kgrids(flow, prefix):
    """``kgrid`` argument of every projection job whose name starts with ``prefix``."""
    return {
        job.name: job.function_kwargs["kgrid"]
        for structure_flow in flow.jobs
        for job in structure_flow.jobs
        if job.name.startswith(prefix)
    }


def _projection_config(tmp_path, stage, mode, **kwargs):
    from hamilflow.dft_workflows.flows_base import ProjectDeephInputsConfig

    return ProjectDeephInputsConfig(
        output_root=tmp_path / stage, removal_plan={"rules": []}, reduction_mode=mode, **kwargs
    )


def test_projections_use_kgrid_from_structure(trajectory, tmp_path):
    from hamilflow.dft_workflows.flows_base import ConvertAimsToDeephConfig
    from hamilflow.dft_workflows.flows_core import GenerateAimsToProjectedDeephData

    dft = GenerateAimsDFTData(
        structures_path=trajectory,
        kgrid_from_structure=True,
        aims_to_deeph_config=ConvertAimsToDeephConfig(output_dir=tmp_path / "deeph"),
    )
    flow = GenerateAimsToProjectedDeephData(
        dft_data_flow=dft,
        projection_config=_projection_config(tmp_path, "p1", "schur", kgrid_from_structure=True),
        second_projection_config=_projection_config(
            tmp_path, "p2", "truncate", kgrid=(3, 3, 3)
        ),
    ).make()
    assert _projection_kgrids(flow, "projection_schur") == {
        "projection_schur_0": (4, 5, 6),
        "projection_schur_1": (2, 2, 1),
    }
    # The second stage keeps its own kgrid.
    assert set(_projection_kgrids(flow, "projection_truncate").values()) == {(3, 3, 3)}


def test_projection_kgrid_from_structure_conflicts_with_kpoints_settings(tmp_path):
    with pytest.raises(ValueError, match="kgrid_from_structure"):
        _projection_config(
            tmp_path, "p1", "schur",
            kgrid_from_structure=True, user_kpoints_settings={"reciprocal_density": 100},
        )


def test_projection_only_flow_rejects_kgrid_from_structure(tmp_path):
    from hamilflow.dft_workflows.flows_base import GenerateProjectedDeephInputs

    (tmp_path / "deeph" / "s0").mkdir(parents=True)
    flow = GenerateProjectedDeephInputs(
        deeph_inputs_root=tmp_path / "deeph",
        projection_config=_projection_config(tmp_path, "p1", "schur", kgrid_from_structure=True),
    )
    with pytest.raises(ValueError, match="do not carry"):
        flow.make()


def test_projection_jobs_default_to_no_kgrid(tmp_path):
    """Without kgrid or user_kpoints_settings, the jobs leave the grid to the projection (R span)."""
    from hamilflow.dft_workflows.flows_base import GenerateProjectedDeephInputs

    (tmp_path / "deeph" / "s0").mkdir(parents=True)
    flow = GenerateProjectedDeephInputs(
        deeph_inputs_root=tmp_path / "deeph",
        projection_config=_projection_config(tmp_path, "p1", "schur"),
    ).make()
    kwargs = flow.jobs[0].function_kwargs
    assert kwargs["kgrid"] is None
    assert kwargs["user_kpoints_settings"] is None
