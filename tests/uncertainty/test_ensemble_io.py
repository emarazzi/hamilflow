"""Placing ensemble files: which files end up where, and what is never overwritten."""

import logging
import os
from pathlib import Path

import pytest

from hamilflow.uncertainty.ensemble_io import discover_structures, link_ensemble_files


@pytest.fixture
def layout(tmp_path):
    """A source structure with its files, and the folder of one ensemble member for it."""
    source = tmp_path / "dft" / "str1"
    source.mkdir(parents=True)
    for name in ("POSCAR", "info.json", "overlap.h5"):
        (source / name).write_text(f"source {name}")
    member = tmp_path / "models" / "member_1" / "str1"
    member.mkdir(parents=True)
    (member / "hamiltonian_pred.h5").write_text("predicted")
    return source, member, tmp_path / "models"


def test_discover_returns_absolute_paths_of_the_matches(tmp_path, monkeypatch):
    (tmp_path / "str1").mkdir()
    (tmp_path / "str2").mkdir()
    (tmp_path / "other").mkdir()
    monkeypatch.chdir(tmp_path)
    assert sorted(discover_structures("str*")) == [tmp_path / "str1", tmp_path / "str2"]


def test_discover_warns_when_nothing_matches(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert discover_structures(str(tmp_path / "nothing*")) == []
    assert "No structures match" in caplog.text


def test_prediction_is_copied_to_the_hamiltonian_name(layout):
    source, member, parent = layout
    link_ensemble_files([source], Path("member_1"), parent)
    assert (member / "hamiltonian.h5").read_text() == "predicted"
    assert not (member / "hamiltonian.h5").is_symlink()
    assert (member / "hamiltonian_pred.h5").exists()


def test_existing_hamiltonian_is_not_overwritten(layout):
    source, member, parent = layout
    (member / "hamiltonian.h5").write_text("existing")
    link_ensemble_files([source], Path("member_1"), parent)
    assert (member / "hamiltonian.h5").read_text() == "existing"


def test_member_without_prediction_gets_no_hamiltonian(layout):
    source, member, parent = layout
    (member / "hamiltonian_pred.h5").unlink()
    link_ensemble_files([source], Path("member_1"), parent)
    assert not (member / "hamiltonian.h5").exists()


def test_structure_files_are_linked_to_the_source(layout):
    source, member, parent = layout
    link_ensemble_files([source], Path("member_1"), parent)
    for name in ("POSCAR", "info.json", "overlap.h5"):
        assert (member / name).is_symlink()
        assert (member / name).resolve() == (source / name).resolve()
        assert (member / name).read_text() == f"source {name}"


def test_existing_structure_files_are_left_alone(layout):
    source, member, parent = layout
    (member / "POSCAR").write_text("mine")
    link_ensemble_files([source], Path("member_1"), parent)
    assert (member / "POSCAR").read_text() == "mine"
    assert not (member / "POSCAR").is_symlink()


def test_files_missing_from_the_source_are_skipped(layout):
    source, member, parent = layout
    (source / "overlap.h5").unlink()
    link_ensemble_files([source], Path("member_1"), parent)
    assert not (member / "overlap.h5").exists()
    assert (member / "POSCAR").exists()


def test_custom_file_names_are_honoured(layout):
    source, member, parent = layout
    (member / "pred.h5").write_text("other prediction")
    (source / "extra.txt").write_text("extra")
    link_ensemble_files([source], Path("member_1"), parent, "pred.h5", "ham.h5", symlink_files=("extra.txt",))
    assert (member / "ham.h5").read_text() == "other prediction"
    assert (member / "extra.txt").is_symlink()
    assert not (member / "POSCAR").exists()


def test_files_are_copied_when_symlinks_are_unavailable(layout, monkeypatch):
    source, member, parent = layout

    def no_symlinks(self, target):
        raise OSError("not supported")

    monkeypatch.setattr(Path, "symlink_to", no_symlinks)
    link_ensemble_files([source], Path("member_1"), parent)
    assert (member / "POSCAR").read_text() == "source POSCAR"
    assert not (member / "POSCAR").is_symlink()


def test_structures_without_a_member_folder_are_skipped_with_a_warning(layout, tmp_path, caplog):
    source, member, parent = layout
    other = tmp_path / "dft" / "str2"
    other.mkdir()
    with caplog.at_level(logging.WARNING):
        link_ensemble_files([source, other], Path("member_1"), parent)
    assert "1 structures have no folder" in caplog.text
    assert "str2" in caplog.text
    assert not (parent / "member_1" / "str2").exists()
    assert (member / "POSCAR").exists()


def test_running_twice_changes_nothing(layout):
    source, member, parent = layout
    link_ensemble_files([source], Path("member_1"), parent)
    link_ensemble_files([source], Path("member_1"), parent)
    assert (member / "hamiltonian.h5").read_text() == "predicted"
    assert (member / "POSCAR").read_text() == "source POSCAR"


@pytest.mark.xfail(
    strict=True,
    reason="a relative structure path is stored as the link target, which resolves against the member folder",
)
def test_structure_given_by_a_relative_path_gives_working_links(layout, tmp_path, monkeypatch):
    _, member, _ = layout
    monkeypatch.chdir(tmp_path)
    link_ensemble_files([Path("dft") / "str1"], Path("member_1"), Path("models"))
    assert os.path.exists(member / "POSCAR")
