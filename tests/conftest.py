import pytest

from hamilflow.sparse_hamiltonian.dense_backend import gpu_backend


def pytest_collection_modifyitems(items):
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if gpu_items and gpu_backend() is None:
        skip_gpu = pytest.mark.skip(reason="no GPU backend available")
        for item in gpu_items:
            item.add_marker(skip_gpu)
