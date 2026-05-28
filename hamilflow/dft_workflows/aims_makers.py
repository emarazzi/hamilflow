from __future__ import annotations

from jobflow import Maker, Response, job
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from monty.serialization import dumpfn
from pymatgen.io.aims.sets.base import AimsInputGenerator
from atomate2 import SETTINGS
from atomate2.aims.files import (
    cleanup_aims_outputs,
    copy_aims_outputs,
    write_aims_input_set,
)
from atomate2.aims.job.base import _FILES_TO_ZIP
from atomate2.aims.run import run_aims
from atomate2.common.files import gzip_output_folder
from emmet.core.structure import MoleculeMetadata, StructureMetadata
from emmet.core.task import BaseTaskDocument
from pydantic import Field
from typing_extensions import Self
if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

@dataclass
class OvlOnlyAimsMaker(Maker):
    """
    FHI-aims job maker for overlap-only calculations.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .AimsInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.write_aims_input_set`.
    copy_aims_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.copy_aims_outputs`.
    run_aims_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.run_aims`.
    task_document_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict[str, Any]
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict[str, Any]
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    store_output_data: bool
        Whether the job output (TaskDoc) should be stored in the JobStore through
        the response.
    """

    name: str = "base"
    input_set_generator: AimsInputGenerator = field(default_factory=AimsInputGenerator)
    write_input_set_kwargs: dict[str, Any] = field(default_factory=dict)
    copy_aims_kwargs: dict[str, Any] = field(default_factory=dict)
    run_aims_kwargs: dict[str, Any] = field(default_factory=dict)
    task_document_kwargs: dict[str, Any] = field(default_factory=dict)
    stop_children_kwargs: dict[str, Any] = field(default_factory=dict)
    write_additional_data: dict[str, Any] = field(default_factory=dict)
    store_output_data: bool = True

    @job
    def make(
        self,
        structure: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """Run an FHI-aims calculation.

        Parameters
        ----------
        structure : Structure or Molecule
            A pymatgen Structure object to create the calculation for.
        prev_dir : str or Path or None
            A previous FHI-aims calculation directory to copy output files from.
        """
        # copy previous inputs if needed (governed by self.copy_aims_kwargs)
        if prev_dir is not None:
            copy_aims_outputs(prev_dir, **self.copy_aims_kwargs)

        # write aims input files
        self.write_input_set_kwargs["prev_dir"] = prev_dir
        write_aims_input_set(
            structure, self.input_set_generator, **self.write_input_set_kwargs
        )

        # write any additional data
        for filename, data in self.write_additional_data.items():
            dumpfn(data, filename.replace(":", "."))

        # run FHI-aims
        run_aims(**self.run_aims_kwargs)

        # parse FHI-aims outputs
        task_doc = ModAimsTaskDoc.from_directory(Path.cwd())
        task_doc.task_label = self.name

        # decide whether child jobs should proceed
        #stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        # cleanup files to save disk space
        cleanup_aims_outputs(directory=Path.cwd())

        # gzip folder
        gzip_output_folder(
            directory=Path.cwd(),
            setting=SETTINGS.VASP_ZIP_FILES,
            files_list=_FILES_TO_ZIP,
        )

        return Response(
            #stop_children=stop_children,
            output=task_doc if self.store_output_data else None,
        )


class ModAimsTaskDoc(BaseTaskDocument, StructureMetadata, MoleculeMetadata):
    """ 
    Dummy task document class used for overlap-only calculations. 
    
    Parameters
    ----------
    dir_name: str
        The directory for this FHI-aims task
    """

    dir_name: str | Path | None = Field(None, description="The directory for this FHI-aims task")
    task_label: str | None = Field(None, description="A description of the task")

    @classmethod
    def from_directory(
        cls,
        directory: str | Path
    ) -> Self:
        """Create a ModAimsTaskDoc from a directory. This is a dummy implementation that only sets the dir_name field."""
        dir_name = Path(directory)
        data = {
            "dir_name": dir_name,
        }
        doc = cls(**data)
        return doc