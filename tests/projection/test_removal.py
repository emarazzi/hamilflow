"""Removal plans: parsing, and which global orbital indices each rule selects."""

import json
from pathlib import Path

import pytest

from hamilflow.projection.models import RemovalPlan, RemovalRule
from hamilflow.projection.removal import coerce_removal_plan, resolve_indices_from_rules

EXAMPLES = Path(__file__).parents[2] / "examples" / "projection"

# Mo: 1s 0, 2s 1, 1p 2-4, 1d 5-9. First S: 1s 10, 1p 11-13. Second S: 1s 14, 1p 15-17
ELEMENTS = ["Mo", "S", "S"]
ORBITAL_MAP = {"Mo": [0, 0, 1, 2], "S": [0, 1]}
MO = list(range(10))
S1 = list(range(10, 14))
S2 = list(range(14, 18))

# The FHI-aims basis of a real MoS2 structure: 49 orbitals per Mo and 39 per S, 1778 in total
REAL_ORBITAL_MAP = {"Mo": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 4], "S": [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4]}
REAL_ELEMENTS = ["Mo"] * 14 + ["S"] * 28


def resolve(*rules, elements=ELEMENTS, orbital_map=ORBITAL_MAP):
    return resolve_indices_from_rules(elements, orbital_map, coerce_removal_plan(list(rules)))[0]


def test_family_removes_every_shell_of_that_l_on_every_atom():
    # s: Mo 0 and 1, S 10 and 14
    assert resolve({"remove_orbitals": ["s"]}) == [0, 1, 10, 14]
    # p: Mo 2-4, S 11-13 and 15-17
    assert resolve({"remove_orbitals": ["p"]}) == [2, 3, 4, 11, 12, 13, 15, 16, 17]


def test_shell_counts_per_l_in_the_order_of_the_basis():
    assert resolve({"remove_shells": ["1s"], "target_elements": ["Mo"]}) == [0]
    assert resolve({"remove_shells": ["2s"], "target_elements": ["Mo"]}) == [1]
    assert resolve({"remove_shells": ["1d"]}) == [5, 6, 7, 8, 9]


def test_shell_number_is_the_position_in_the_basis_not_the_principal_quantum_number():
    """The documented trap: the first p of the basis is '1p', although no 1p exists physically."""
    assert resolve({"remove_shells": ["1p"], "target_elements": ["Mo"]}) == [2, 3, 4]


def test_shell_counting_restarts_on_each_atom():
    assert resolve({"remove_shells": ["1s"]}) == [0, 10, 14]


def test_shell_counts_are_independent_across_l_when_shells_interleave():
    assert resolve_indices_from_rules(
        ["X"], {"X": [0, 1, 0, 1]}, coerce_removal_plan([{"remove_shells": ["2s", "2p"]}])
    )[0] == [4, 5, 6, 7]


def test_family_and_shell_in_one_rule_are_combined():
    assert resolve({"target_elements": ["Mo"], "remove_orbitals": ["d"], "remove_shells": ["1s"]}) == [0, 5, 6, 7, 8, 9]


def test_elements_select_only_their_atoms():
    assert resolve({"target_elements": ["S"], "remove_orbitals": ["s"]}) == [10, 14]


def test_atom_indices_select_only_those_atoms():
    assert resolve({"target_atom_indices": [2], "remove_orbitals": ["s", "p"]}) == S2


def test_elements_and_atom_indices_are_intersected():
    assert resolve({"target_elements": ["S"], "target_atom_indices": [0, 1], "remove_orbitals": ["s"]}) == [10]
    assert resolve({"target_elements": ["Mo"], "target_atom_indices": [1], "remove_orbitals": ["s"]}) == []


def test_atom_index_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        resolve({"target_atom_indices": [3], "remove_orbitals": ["s"]})
    with pytest.raises(ValueError, match="out of range"):
        resolve({"target_atom_indices": [-1], "remove_orbitals": ["s"]})


def test_rules_are_united_sorted_and_without_duplicates():
    assert resolve({"target_elements": ["S"], "remove_orbitals": ["s"]}, {"remove_orbitals": ["s"]}) == [0, 1, 10, 14]


def test_rule_that_removes_nothing_selects_nothing():
    assert resolve({"target_elements": ["Mo"]}) == []


def test_whole_basis_can_be_selected():
    assert resolve({"remove_orbitals": ["s", "p", "d"]}) == MO + S1 + S2


@pytest.mark.parametrize("token", ["1s", " 1s ", "1S"])
def test_shell_selector_tolerates_space_and_case(token):
    assert resolve({"target_elements": ["Mo"], "remove_shells": [token]}) == [0]


@pytest.mark.parametrize("label", [" s ", "S"])
def test_family_label_tolerates_space_and_case(label):
    assert resolve({"target_elements": ["Mo"], "remove_orbitals": [label]}) == [0, 1]


@pytest.mark.parametrize("token", ["s", "1", "1sp", "s1", ""])
def test_malformed_shell_selector_is_rejected(token):
    with pytest.raises(ValueError, match="Invalid shell selector"):
        resolve({"remove_shells": [token]})


def test_unsupported_orbital_labels_are_rejected():
    with pytest.raises(ValueError, match="Unsupported orbital label"):
        resolve({"remove_shells": ["1x"]})
    with pytest.raises(ValueError, match="Unsupported orbital family"):
        resolve({"remove_orbitals": ["x"]})


def test_metadata_describes_what_each_rule_resolved_to():
    plan = coerce_removal_plan([{"target_atom_indices": [1], "remove_shells": ["1p"]}])
    _, meta = resolve_indices_from_rules(ELEMENTS, ORBITAL_MAP, plan)
    assert meta[0]["rule_index"] == 0
    assert meta[0]["input_rule"]["remove_shells"] == ["1p"]
    assert meta[0]["resolved"]["selected_atoms"] == [1]
    assert meta[0]["resolved"]["resolved_shells"] == [
        {
            "atom_index": 1,
            "element": "S",
            "shell_position": 1,
            "shell": "1p",
            "global_index_start": 11,
            "global_index_stop": 14,
        }
    ]


def test_all_plan_forms_give_the_same_plan(tmp_path):
    rules = [{"target_elements": ["Mo"], "remove_shells": ["1s"]}]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"rules": rules}))
    expected = RemovalPlan(rules=[RemovalRule(target_elements=["Mo"], remove_shells=["1s"])])

    assert coerce_removal_plan(rules) == expected
    assert coerce_removal_plan({"rules": rules}) == expected
    assert coerce_removal_plan(path) == expected
    assert coerce_removal_plan(str(path)) == expected
    assert coerce_removal_plan(expected) is expected


def test_comment_keys_are_ignored():
    plan = coerce_removal_plan({"_comment": ["doc"], "rules": [{"_comment": "doc", "remove_orbitals": ["s"]}]})
    assert plan == RemovalPlan(rules=[RemovalRule(remove_orbitals=["s"])])


def test_missing_and_null_fields_are_empty():
    assert coerce_removal_plan([{"target_elements": None}]) == RemovalPlan(rules=[RemovalRule()])


def test_atom_indices_may_be_numeric_strings():
    assert coerce_removal_plan([{"target_atom_indices": ["2"]}]).rules[0].target_atom_indices == [2]


@pytest.mark.parametrize("index", [True, 1.5, None])
def test_atom_index_of_another_type_is_rejected(index):
    with pytest.raises(ValueError, match="target_atom_indices"):
        coerce_removal_plan([{"target_atom_indices": [index]}])


def test_rule_field_that_is_not_a_list_is_rejected():
    with pytest.raises(ValueError, match="must be a list"):
        coerce_removal_plan([{"remove_shells": "1s"}])


@pytest.mark.parametrize("payload", [{"rules": "1s"}, {"remove_shells": ["1s"]}])
def test_plan_without_a_list_of_rules_is_rejected(payload):
    with pytest.raises(ValueError, match="must be a list"):
        coerce_removal_plan(payload)


def test_plan_of_an_unsupported_type_is_rejected():
    with pytest.raises(TypeError, match="Unsupported removal_plan"):
        coerce_removal_plan(3)


def test_schur_example_removes_the_mo_and_s_core_of_a_real_basis():
    """Two electrons per orbital: the plan must remove exactly the [Kr] of each Mo and the [Ne] of each S."""
    indices, _ = resolve_indices_from_rules(
        REAL_ELEMENTS, REAL_ORBITAL_MAP, coerce_removal_plan(EXAMPLES / "removal_plan_schur.json")
    )
    assert 2 * len([i for i in indices if i < 14 * 49]) == 14 * 36
    assert 2 * len([i for i in indices if i >= 14 * 49]) == 28 * 10
    assert 14 * 49 + 28 * 39 == 1778
    # the first Mo: 4 s, then 2 kept s, then 3 p, one kept p, then 1 d
    assert [i for i in indices if i < 49] == [0, 1, 2, 3, *range(6, 15), *range(18, 23)]


def test_truncate_example_removes_the_polarisation_shells_of_a_real_basis():
    indices, _ = resolve_indices_from_rules(
        REAL_ELEMENTS, REAL_ORBITAL_MAP, coerce_removal_plan(EXAMPLES / "removal_plan_truncate.json")
    )
    # Mo g is its last 9 orbitals; S f and g are its last 7 + 9
    assert [i for i in indices if i < 49] == list(range(40, 49))
    assert [i for i in indices if 14 * 49 <= i < 14 * 49 + 39] == list(range(14 * 49 + 23, 14 * 49 + 39))
    assert len(indices) == 14 * 9 + 28 * 16
