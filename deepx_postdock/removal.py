from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .models import RemovalPlan, RemovalPlanLike, RemovalRule


L_TO_LABEL = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}


def _parse_shell_selector(token: str) -> tuple[int, int]:
    """Parse shell selector strings like '1s', '2p' into (n, l)."""
    m = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z])\s*", token)
    if m is None:
        raise ValueError(f"Invalid shell selector '{token}'. Use forms like '1s', '2p'.")
    n = int(m.group(1))
    orb = m.group(2).lower()
    l_candidates = [ll for ll, label in L_TO_LABEL.items() if label == orb]
    if not l_candidates:
        raise ValueError(f"Unsupported orbital label in selector '{token}'.")
    return n, l_candidates[0]


def _normalize_orbital_labels(labels: list[str]) -> set[int]:
    """Map labels like ['s', 'p'] to a set of angular momentum integers l."""
    out: set[int] = set()
    inv_map = {v: k for k, v in L_TO_LABEL.items()}
    for lb in labels:
        key = lb.strip().lower()
        if key == "":
            continue
        if key not in inv_map:
            raise ValueError(f"Unsupported orbital family '{lb}'. Supported: {sorted(inv_map)}")
        out.add(inv_map[key])
    return out


def _coerce_to_int_list(values: list[object], key: str) -> list[int]:
    out: list[int] = []
    for v in values:
        if isinstance(v, bool):
            raise ValueError(f"Rule field '{key}' contains bool value, expected integer index")
        if isinstance(v, (int, np.integer, str)):
            out.append(int(v))
        else:
            raise ValueError(f"Rule field '{key}' contains unsupported value type: {type(v).__name__}")
    return out


def _normalize_rule(rule_obj: dict[str, object]) -> RemovalRule:
    def get_list_field(name: str) -> list[object]:
        value = rule_obj.get(name, [])
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"Rule field '{name}' must be a list, got {type(value).__name__}")
        return value

    return RemovalRule(
        target_elements=[str(x) for x in get_list_field("target_elements")],
        target_atom_indices=_coerce_to_int_list(get_list_field("target_atom_indices"), "target_atom_indices"),
        remove_orbitals=[str(x) for x in get_list_field("remove_orbitals")],
        remove_shells=[str(x) for x in get_list_field("remove_shells")],
    )


def _normalize_plan_payload(payload: Any) -> RemovalPlan:
    rules_payload = payload.get("rules", payload) if isinstance(payload, dict) else payload
    if not isinstance(rules_payload, list):
        raise ValueError("removal plan must be a list or an object with key 'rules' (list)")
    rules = [_normalize_rule(r) for r in rules_payload]
    return RemovalPlan(rules=rules)


def coerce_removal_plan(removal_plan: RemovalPlanLike) -> RemovalPlan:
    """Accept a plan from path/json/dict/list/model and return normalized RemovalPlan."""
    if isinstance(removal_plan, RemovalPlan):
        return removal_plan

    if isinstance(removal_plan, (str, Path)):
        with open(Path(removal_plan), "r", encoding="utf-8") as fr:
            payload = json.load(fr)
        return _normalize_plan_payload(payload)

    if isinstance(removal_plan, (dict, list)):
        return _normalize_plan_payload(removal_plan)

    raise TypeError(f"Unsupported removal_plan type: {type(removal_plan).__name__}")


def resolve_indices_from_rules(
    elements: list[str],
    elements_orbital_map: dict[str, list[int]],
    plan: RemovalPlan,
) -> tuple[list[int], list[dict[str, object]]]:
    """Resolve a normalized removal plan into global orbital indices and metadata."""
    all_indices: list[int] = []
    per_rule_meta: list[dict[str, object]] = []
    n_atoms = len(elements)

    for i_rule, rule in enumerate(plan.rules):
        atom_filter = set(range(n_atoms))
        if rule.target_elements:
            target_set = {x.strip() for x in rule.target_elements if x.strip()}
            atom_filter = {ia for ia, el in enumerate(elements) if el in target_set}

        if rule.target_atom_indices:
            atom_index_set = set(int(i) for i in rule.target_atom_indices)
            bad_idx = [i for i in atom_index_set if i < 0 or i >= n_atoms]
            if bad_idx:
                raise ValueError(f"target_atom_indices out of range [0, {n_atoms - 1}]: {sorted(bad_idx)}")
            atom_filter = atom_filter.intersection(atom_index_set)

        rm_l_set = _normalize_orbital_labels(rule.remove_orbitals)
        rm_shell_set = {_parse_shell_selector(tok) for tok in rule.remove_shells}

        rm_i: list[int] = []
        resolved_shells: list[dict[str, object]] = []
        g0 = 0
        for ia, el in enumerate(elements):
            shell_ls = [int(v) for v in elements_orbital_map[el]]
            shell_count_by_l: dict[int, int] = {}

            for shell_pos, l in enumerate(shell_ls):
                shell_n = shell_count_by_l.get(l, 0) + 1
                shell_count_by_l[l] = shell_n
                shell_dim = 2 * l + 1
                shell_start = g0
                shell_stop = g0 + shell_dim

                if ia in atom_filter and (l in rm_l_set or (shell_n, l) in rm_shell_set):
                    rm_i.extend(range(shell_start, shell_stop))
                    resolved_shells.append(
                        {
                            "atom_index": ia,
                            "element": el,
                            "shell_position": shell_pos,
                            "shell": f"{shell_n}{L_TO_LABEL.get(l, f'l{l}')}",
                            "global_index_start": shell_start,
                            "global_index_stop": shell_stop,
                        }
                    )

                g0 = shell_stop

        meta_i = {
            "selected_atoms": sorted(atom_filter),
            "remove_orbital_families": [L_TO_LABEL[l] for l in sorted(rm_l_set)],
            "remove_shells": sorted([f"{n}{L_TO_LABEL[l]}" for n, l in rm_shell_set]),
            "resolved_shells": resolved_shells,
        }

        all_indices.extend(rm_i)
        per_rule_meta.append(
            {
                "rule_index": i_rule,
                "input_rule": {
                    "target_elements": list(rule.target_elements),
                    "target_atom_indices": list(rule.target_atom_indices),
                    "remove_orbitals": list(rule.remove_orbitals),
                    "remove_shells": list(rule.remove_shells),
                },
                "resolved": meta_i,
            }
        )

    return sorted(set(all_indices)), per_rule_meta
