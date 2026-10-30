import importlib

import pytest

SUBPACKAGES = [
    "band_structures",
    "candidate_selection",
    "dft_workflows",
    "projection",
    "sparse_hamiltonian",
    "training",
    "uncertainty",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"hamilflow.{name}")
