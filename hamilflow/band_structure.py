from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from plotly import graph_objects as go
from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo

from deepx_dock.compute.eigen.hamiltonian import HamiltonianObj


def get_band_conf_from_file(
    workdir: str | Path = ".",
    num_band: int = 50,
    e_min: float = -0.5,
    maxiter: int = 300,
    k_path_filename: str = "K_PATH",
) -> dict[str, object]:
    data_path = Path(workdir).resolve()
    with open(data_path / k_path_filename, "r", encoding="utf-8") as f:
        k_list_spell = f.read()
    return {
        "k_list_spell": k_list_spell,
        "num_band": num_band,
        "lowest_band_energy": e_min,
        "maxiter": maxiter,
    }


def get_band_conf_from_struc(
    struc,
    num_band: int = 50,
    e_min: float = -0.5,
    maxiter: int = 300,
    density: int = 40,
) -> dict[str, object]:
    """Build a band configuration from a structure object and an auto-generated high-symmetry path."""
    kpath = KPathSetyawanCurtarolo(struc)
    if kpath.kpath is None:
        raise ValueError(
            "Automatic high-symmetry k-path generation failed for this structure. "
            "Use get_band_conf_from_file(...) with a K_PATH fallback."
        )

    path_symbols = kpath.kpath["path"]
    kpoint_coords = kpath.kpath["kpoints"]

    recp_lattice = struc.lattice.reciprocal_lattice

    k_list_lines = []
    for section in path_symbols:
        for i in range(len(section) - 1):
            kpt1_symbol = section[i]
            kpt2_symbol = section[i + 1]

            kpt1_frac = np.array(kpoint_coords[kpt1_symbol])
            kpt2_frac = np.array(kpoint_coords[kpt2_symbol])

            kpt1_cart = recp_lattice.get_cartesian_coords(kpt1_frac)
            kpt2_cart = recp_lattice.get_cartesian_coords(kpt2_frac)

            distance = np.linalg.norm(kpt2_cart - kpt1_cart)
            n = max(2, int(round(distance * density)))

            kpt1_str = f"{kpt1_frac[0]:.6f} {kpt1_frac[1]:.6f} {kpt1_frac[2]:.6f}"
            kpt2_str = f"{kpt2_frac[0]:.6f} {kpt2_frac[1]:.6f} {kpt2_frac[2]:.6f}"

            line = (
                f"{n} {kpt1_str} {kpt2_str} "
                f"{kpt1_symbol.replace('\\', '')} {kpt2_symbol.replace('\\', '')}"
            )
            k_list_lines.append(line)

    k_list_spell = "\n".join(k_list_lines)
    return {
        "k_list_spell": k_list_spell,
        "num_band": num_band,
        "lowest_band_energy": e_min,
        "maxiter": maxiter,
    }


def get_hamiltonian(workdir: str | Path = ".") -> HamiltonianObj:
    data_path = Path(workdir).resolve()
    return HamiltonianObj(data_path)


def get_hsk_symbol_list(bd_gen) -> list[str]:
    """Prepare pretty labels for high-symmetry k-points."""
    hsk_symbol_list = ["" for _ in range(bd_gen.k_path_quantity + 1)]
    hsk_symbol_list[0] = bd_gen.hsk_symbol_list[0][0]
    hsk_symbol_list[-1] = bd_gen.hsk_symbol_list[-1][1]
    for i_path in range(1, bd_gen.k_path_quantity):
        if bd_gen.hsk_symbol_list[i_path][0] == bd_gen.hsk_symbol_list[i_path - 1][1]:
            hsk_symbol_list[i_path] = bd_gen.hsk_symbol_list[i_path][0]
        else:
            hsk_symbol_list[i_path] = (
                f"{bd_gen.hsk_symbol_list[i_path - 1][1]}|{bd_gen.hsk_symbol_list[i_path][0]}"
            )

    greek_symbol_map = {
        "Gamma": r"\Gamma",
        "Delta": r"\Delta",
        "Theta": r"\Theta",
        "Lambda": r"\Lambda",
        "Xi": r"\Xi",
        "Pi": r"\Pi",
        "Sigma": r"\Sigma",
        "Phi": r"\Phi",
        "Psi": r"\Psi",
        "Omega": r"\Omega",
    }
    keys = [re.escape(k) for k in greek_symbol_map]
    pattern = r"\b(" + "|".join(keys) + r")\b"

    for i, symbol in enumerate(hsk_symbol_list):
        symbol = re.sub(pattern, lambda match: greek_symbol_map[match.group(1)], symbol, flags=re.IGNORECASE)
        symbol = re.sub(r"_\d+", lambda x: f"_{x.group(0)[2:]}", symbol)
        hsk_symbol_list[i] = symbol

    return hsk_symbol_list


def plot_band(
    bd_gen,
    *,
    show_fig: bool = True,
    save_path: str | Path | None = None,
) -> go.Figure:
    """Plot an electronic band structure with high-symmetry separators and Fermi level."""
    fig = go.Figure()

    ymin = np.round(np.min(bd_gen.band_data) / 20 - 1) * 20
    ymax = np.round(np.max(bd_gen.band_data) / 20 + 1) * 20
    xmin = 0.0
    xmax = bd_gen.kpoints_distance_list[-1]

    for band_index in range(bd_gen.band_quantity):
        scatter = go.Scatter(
            x=bd_gen.kpoints_distance_list,
            y=bd_gen.band_data[band_index],
            mode="lines",
            line=dict(color="red", width=2),
            showlegend=False,
        )
        fig.add_trace(scatter)

    for x in bd_gen.hsk_distance_list:
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[ymin, ymax],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[xmin, xmax],
            y=[0.0, 0.0],
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            showlegend=False,
        )
    )

    fig.update_layout(
        height=600,
        width=700,
        xaxis={
            "showline": True,
            "linewidth": 2,
            "linecolor": "black",
            "mirror": True,
            "showgrid": False,
            "tickmode": "array",
            "tickvals": bd_gen.hsk_distance_list,
            "ticktext": get_hsk_symbol_list(bd_gen),
            "range": [xmin, xmax],
        },
        yaxis={
            "showline": True,
            "linewidth": 2,
            "linecolor": "black",
            "mirror": True,
            "showgrid": False,
            "title": "Energy (eV)",
            "range": [ymin, ymax],
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output))
    if show_fig:
        fig.show()
    return fig