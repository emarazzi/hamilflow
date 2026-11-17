"""
Record the FHI-aims outputs replayed by the mocked variant of test_aims_pipeline.py.

Runs the real FHI-aims job of the test case (light C basis, AIMS_CMD from the atomate2 config) and stores the files
the pipeline reads from the run directory, gzipped, in data/aims_pipeline/mock_outputs. Run it again whenever the
aims settings of the test case change, then rerun ``pytest tests/dft_workflows/test_aims_pipeline.py``.

    python tests/dft_workflows/record_mock_aims.py
"""

import gzip
import shutil
import sys
import tempfile
from pathlib import Path

from jobflow.managers.local import run_locally

sys.path.insert(0, str(Path(__file__).parent))
from test_aims_pipeline import (
    CASE,
    DATA,
    KGRID,
    MOCK_OUTPUT_FILES,
    MOCK_OUTPUTS,
    STRUCTURE,
)

from hamilflow.dft_workflows import GenerateAimsDFTData


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="record_mock_aims_") as tmp:
        root = Path(tmp)
        flow = GenerateAimsDFTData(
            structures_path=DATA / "structures",
            aims_kwargs=CASE.aims_kwargs(),
            kgrid=KGRID,
            collected_runs_root=root / "aims_output",
        ).make()
        run_locally(flow, root_dir=root / "jobs", create_folders=True, ensure_success=True)

        run_dir = root / "aims_output" / STRUCTURE
        MOCK_OUTPUTS.mkdir(parents=True, exist_ok=True)
        for name in MOCK_OUTPUT_FILES:
            target = MOCK_OUTPUTS / f"{name}.gz"
            if (run_dir / name).exists():
                with open(run_dir / name, "rb") as fi, gzip.open(target, "wb", compresslevel=9) as fo:
                    shutil.copyfileobj(fi, fo)
            else:  # atomate2 gzips some outputs itself
                shutil.copyfile(run_dir / f"{name}.gz", target)
            print(f"{target.stat().st_size:>10d}  {target}")


if __name__ == "__main__":
    main()
