"""
The parallel drivers: same results as the serial ones, and the scheduling of `_run_parallel`.

Scheduling tests use fake memory estimates and small functions as the work of each structure, so what they
check is the order and overlap of the forked processes, whatever a real structure would take.
"""

import itertools
import json
import os
import shutil
import time

import numpy as np
import pytest
from deepx_dock.CONSTANT import DEEPX_HAMILTONIAN_FILENAME

from hamilflow.uncertainty import band_uncertainty
from hamilflow.uncertainty.band_uncertainty import HOST_MATRICES, BandUncertaintyCalculator

NAME = "tMoS2-001"
MESH = (2, 2, 1)
GB = 2**30


def make_ensemble(structure, root, names=(NAME,), n_models=3):
    model_dirs = []
    for i in range(n_models):
        model = root / f"model_{i}"
        for name in names:
            folder = model / name
            folder.mkdir(parents=True)
            for path in structure.path.iterdir():
                if path.name != DEEPX_HAMILTONIAN_FILENAME:
                    shutil.copy(path, folder / path.name)
            H_R = structure.H_R + 0.02 * structure.perturbation(seed=i)
            structure.write_hamiltonian(H_R, folder / DEEPX_HAMILTONIAN_FILENAME)
        model_dirs.append(model)
    return model_dirs


@pytest.fixture
def structure(make_synthetic_structure):
    return make_synthetic_structure("input")


def assert_same_results(a, b):
    assert set(a) == set(b)
    for name in a:
        for key in a[name]:
            if key != "kpoints":
                assert (
                    a[name][key] == pytest.approx(b[name][key], abs=1e-9)
                    if isinstance(a[name][key], float | list)
                    else a[name][key] == b[name][key]
                )
        for k, entry in a[name]["kpoints"].items():
            for key, value in entry.items():
                np.testing.assert_allclose(value, b[name]["kpoints"][k][key], atol=1e-9)


# ---------------------------------- same results ----------------------------------


@pytest.mark.parametrize("method", ["exact", "projected"])
def test_parallel_compute_matches_serial(structure, tmp_path, method):
    names = (NAME, "tMoS2-002")
    model_dirs = make_ensemble(structure, tmp_path, names)
    calc = BandUncertaintyCalculator(grid_mesh=MESH, method=method)
    serial = calc.compute(model_dirs, tmp_path / "average", n_jobs=1)
    parallel = calc.compute_parallel(model_dirs, tmp_path / "average", max_workers=2, memory_gb=100)
    assert_same_results(serial, parallel)


def test_parallel_comparison_matches_serial(structure, tmp_path):
    names = (NAME, "tMoS2-002")
    model_dirs = make_ensemble(structure, tmp_path, names)
    dft_root = tmp_path / "dft"
    for name in names:
        shutil.copytree(structure.path, dft_root / name)
    calc = BandUncertaintyCalculator(grid_mesh=MESH)
    serial = calc.compare_averaged_to_dft(model_dirs, tmp_path / "average", dft_root, n_jobs=1)
    parallel = calc.compare_averaged_to_dft_parallel(
        model_dirs, tmp_path / "average", dft_root, max_workers=2, memory_gb=100
    )
    assert_same_results(serial, parallel)


def test_parallel_compute_writes_its_output_and_resumes(structure, tmp_path):
    model_dirs = make_ensemble(structure, tmp_path)
    output = tmp_path / "out.json"
    calc = BandUncertaintyCalculator(grid_mesh=MESH)
    calc.compute_parallel(model_dirs, tmp_path / "average", max_workers=1, output_path=output, memory_gb=100)
    stored = json.loads(output.read_text())
    assert set(stored) == {NAME}
    stored[NAME]["marker"] = True
    output.write_text(json.dumps(stored))
    assert calc.compute_parallel(model_dirs, tmp_path / "average", output_path=output)[NAME]["marker"] is True


def test_comparison_resumes_too(structure, tmp_path):
    model_dirs = make_ensemble(structure, tmp_path)
    dft_root = tmp_path / "dft"
    shutil.copytree(structure.path, dft_root / NAME)
    output = tmp_path / "out.json"
    output.write_text(json.dumps({NAME: {"marker": True}}))
    calc = BandUncertaintyCalculator(grid_mesh=MESH)
    result = calc.compare_averaged_to_dft_parallel(model_dirs, tmp_path / "average", dft_root, output_path=output)
    assert result == {NAME: {"marker": True}}


# ---------------------------------- memory estimate ----------------------------------


def test_memory_estimate_follows_its_model(structure, monkeypatch):
    calc = BandUncertaintyCalculator(grid_mesh=(1, 1, 1))  # one k-point, dimension 6
    matrix = 16.0 * 6**2
    a, b = HOST_MATRICES["eigenvalues"]
    assert calc._memory_estimate(structure.path, False, False) == ((a + b) * matrix, 0.0)
    a, b = HOST_MATRICES["eigenvectors"]
    assert calc._memory_estimate(structure.path, False, True)[0] == (a + b) * matrix


def test_memory_estimate_on_a_gpu_counts_only_problems_that_go_there(structure, monkeypatch):
    calc = BandUncertaintyCalculator(grid_mesh=(1, 1, 1))
    assert calc._memory_estimate(structure.path, True, False)[1] == 0.0  # dimension 6 is below GPU_MIN_DIM
    monkeypatch.setattr(band_uncertainty, "GPU_MIN_DIM", 1)
    gpu = calc._memory_estimate(structure.path, True, False)[1]
    assert gpu == sum(band_uncertainty.GPU_MATRICES) * 16.0 * 6**2


def test_memory_estimate_reads_no_matrix_entries(structure, tmp_path):
    """It runs before scheduling on every structure, so it must not depend on loading a large file."""
    import h5py

    folder = tmp_path / "s"
    shutil.copytree(structure.path, folder)
    with h5py.File(folder / DEEPX_HAMILTONIAN_FILENAME, "r+") as f:
        del f["entries"]
    assert BandUncertaintyCalculator(grid_mesh=(1, 1, 1))._memory_estimate(folder, False, False)[0] > 0


# ---------------------------------- scheduling ----------------------------------


@pytest.fixture
def scheduler(monkeypatch):
    """`run(work, sizes_gb, max_workers, memory_gb)`: each structure's estimate is `size` GB, plus 1 GB per process."""
    monkeypatch.setattr(band_uncertainty, "workers_gpu_memory", lambda: None)
    calc = BandUncertaintyCalculator()

    def run(work, sizes_gb, max_workers=1, memory_gb=100.0, output_path=None):
        monkeypatch.setattr(calc, "_memory_estimate", lambda d, on_gpu, vectors: (sizes_gb[d] * GB, 0.0))
        names = list(sizes_gb)
        output = {}
        try:
            calc._run_parallel(work, names, names, max_workers, output, output_path, memory_gb, None, False)
        finally:
            run.output = output
        return output

    return run


def work_returning_name(name):
    return name, {"pid": os.getpid()}


def test_every_structure_runs_in_its_own_process(scheduler):
    output = scheduler(work_returning_name, {"a": 0.0, "b": 0.0})
    assert output["a"]["pid"] != output["b"]["pid"]
    assert os.getpid() not in {output["a"]["pid"], output["b"]["pid"]}


def test_largest_structures_start_first(scheduler):
    assert list(scheduler(work_returning_name, {"a": 1.0, "b": 3.0, "c": 2.0})) == ["b", "c", "a"]


def test_equal_estimates_run_in_name_order(scheduler):
    assert list(scheduler(work_returning_name, {"b": 1.0, "c": 1.0, "a": 1.0})) == ["a", "b", "c"]


def test_structures_over_budget_run_last_smallest_first(scheduler, caplog):
    # with the 1 GB per process: a needs 2, c needs 3, b needs 4 of a budget of 2.5
    with caplog.at_level("WARNING"):
        order = list(scheduler(work_returning_name, {"a": 1.0, "b": 3.0, "c": 2.0}, memory_gb=2.5))
    assert order == ["a", "c", "b"]
    assert "2 structures exceed the memory budget" in caplog.text


def wait_for_the_other(folder):
    def work(name):
        (folder / name).write_text("started")
        other = "b" if name == "a" else "a"
        deadline = time.monotonic() + 20
        while not (folder / other).exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{other} never started while {name} was running")
            time.sleep(0.01)
        return name, {}

    return work


def test_structures_that_fit_together_run_at_the_same_time(scheduler, tmp_path):
    """Each waits until it sees the other started, so this can only finish if they overlap."""
    output = scheduler(wait_for_the_other(tmp_path), {"a": 0.0, "b": 0.0}, max_workers=2, memory_gb=10.0)
    assert set(output) == {"a", "b"}


def test_max_workers_limits_the_structures_at_once(scheduler, tmp_path):
    (tmp_path / "times").mkdir()

    def work(name):
        start = time.monotonic()
        time.sleep(0.1)
        (tmp_path / "times" / name).write_text(f"{start} {time.monotonic()}")
        return name, {}

    scheduler(work, {"a": 0.0, "b": 0.0, "c": 0.0}, max_workers=1, memory_gb=100.0)
    spans = sorted(tuple(map(float, p.read_text().split())) for p in (tmp_path / "times").iterdir())
    assert all(first[1] <= second[0] for first, second in itertools.pairwise(spans))


def test_structures_that_do_not_fit_together_never_overlap(scheduler, tmp_path):
    (tmp_path / "times").mkdir()

    def work(name):
        start = time.monotonic()
        time.sleep(0.1)
        (tmp_path / "times" / name).write_text(f"{start} {time.monotonic()}")
        return name, {}

    # each needs 2 GB: both fit the budget alone, not together, although 2 workers are allowed
    scheduler(work, {"a": 1.0, "b": 1.0}, max_workers=2, memory_gb=3.0)
    spans = sorted(tuple(map(float, p.read_text().split())) for p in (tmp_path / "times").iterdir())
    assert spans[0][1] <= spans[1][0]


def test_smaller_structure_fills_the_gap_next_to_a_large_one(scheduler, tmp_path):
    """b (the largest) and c do not fit together, b and a do: a must start while b runs, before c."""

    def work(name):
        if name == "b":
            deadline = time.monotonic() + 20
            while not (tmp_path / "a").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("a never started next to b")
                time.sleep(0.01)
        (tmp_path / name).write_text("started")
        return name, {}

    output = scheduler(work, {"a": 0.0, "b": 2.0, "c": 1.0}, max_workers=2, memory_gb=4.5)
    assert set(output) == {"a", "b", "c"}


def test_memory_of_finished_structures_is_available_again(scheduler, tmp_path):
    """a and b run together, then c and d must too: each pair waits for the other, so a budget not
    handed back when a structure finishes would leave c waiting for d forever."""

    def work(name):
        (tmp_path / name).write_text("started")
        other = {"a": "b", "b": "a", "c": "d", "d": "c"}[name]
        deadline = time.monotonic() + 10
        while not (tmp_path / other).exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{other} never started while {name} was running")
            time.sleep(0.01)
        return name, {}

    sizes = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    assert set(scheduler(work, sizes, max_workers=2, memory_gb=4.5)) == set(sizes)


def test_failed_structure_does_not_stop_the_others(scheduler):
    def work(name):
        if name == "bad":
            raise ValueError("this structure is broken")
        return name, {}

    with pytest.raises(RuntimeError) as error:
        scheduler(work, {"bad": 2.0, "good": 1.0})
    assert "1 of 2 structures failed" in str(error.value)
    assert "ValueError: this structure is broken" in str(error.value)
    assert set(scheduler.output) == {"good"}


def test_process_killed_without_a_word_is_reported_as_failed(scheduler):
    def work(name):
        if name == "killed":
            os._exit(9)
        return name, {}

    with pytest.raises(RuntimeError, match="process exited with code 9"):
        scheduler(work, {"killed": 2.0, "good": 1.0})
    assert set(scheduler.output) == {"good"}


def test_finished_structures_are_on_disk_when_another_fails(scheduler, tmp_path):
    output = tmp_path / "out.json"

    def work(name):
        if name == "bad":
            raise ValueError("broken")
        return name, {"done": True}

    with pytest.raises(RuntimeError):
        scheduler(work, {"good": 2.0, "bad": 1.0}, output_path=output)
    assert json.loads(output.read_text()) == {"good": {"done": True}}


def test_infinite_budget_switches_the_limit_off(scheduler):
    assert set(scheduler(work_returning_name, {"a": 1e6}, memory_gb=float("inf"))) == {"a"}
