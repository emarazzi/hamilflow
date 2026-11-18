"""Two-centre tables: channels, rotation to arbitrary bonds (checked on analytic Gaussian overlaps),
cutoffs, save/load, outer-radius parsing."""

import numpy as np
import pytest

from hamilflow.overlap.rotations import real_sph_harm, rotation_from_z
from hamilflow.overlap.tables import (
    TwoCentreTable,
    bond_blocks_to_channels,
    channel_list,
    channels_to_bond_blocks,
    wigner_blockdiag,
)

ALPHA, BETA = 0.9, 0.6  # Gaussian exponents, 1/Å^2
C1 = np.sqrt(3 / (4 * np.pi))


def gaussian_sp_overlap(d, alpha=ALPHA, beta=BETA):
    """Analytic overlaps (unnormalized) between {s, p} Gaussians on A = 0 (exponent ALPHA) and on
    B = d (BETA): s = exp(-a r^2), p_k = r_k exp(-a r^2), p ordered (y, z, x) like the real
    harmonics. Product centre P = BETA d / (ALPHA + BETA)."""
    g = alpha + beta
    K = (np.pi / g) ** 1.5 * np.exp(-alpha * beta / g * (d @ d))
    P = beta * d / g
    PA, PB = P, P - d
    out = np.zeros((4, 4))
    out[0, 0] = K
    out[0, 1:] = PB * K
    out[1:, 0] = PA * K
    out[1:, 1:] = (np.outer(PA, PB) + np.eye(3) / (2 * g)) * K
    perm = [0, 2, 3, 1]  # s, then p as (y, z, x)
    return out[np.ix_(perm, perm)]


def gaussian_table(r, alpha=ALPHA, beta=BETA):
    shells = (0, 1)
    z = np.array([0.0, 0.0, 1.0])
    blocks = np.array([gaussian_sp_overlap(ri * z, alpha, beta) for ri in r])
    vals, resid = bond_blocks_to_channels(blocks, shells, shells)
    assert resid.max() < 1e-14
    return TwoCentreTable("C", "C", shells, shells, r, vals, np.array([2.0, 2.0]), np.array([2.0, 2.0]),
                          onsite={"C": np.eye(4)})


def test_channels():
    assert channel_list((0, 1), (1,)).tolist() == [[0, 0, 0], [1, 0, 0], [1, 0, 1]]
    rng = np.random.default_rng(0)
    shells_a, shells_b = (0, 1, 2, 4), (1, 3)
    vals = rng.normal(size=(5, len(channel_list(shells_a, shells_b))))
    blocks = channels_to_bond_blocks(vals, shells_a, shells_b)
    back, resid = bond_blocks_to_channels(blocks, shells_a, shells_b)
    np.testing.assert_allclose(back, vals)
    assert resid.max() == 0


def test_gaussian_overlaps_for_arbitrary_bonds():
    """Table along z, rotated onto random bonds, reproduces the analytic overlaps."""
    table = gaussian_table(np.linspace(0.5, 4.0, 400))
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(30, 3))
    vecs *= rng.uniform(0.8, 3.5, size=(30, 1)) / np.linalg.norm(vecs, axis=1, keepdims=True)
    ref = np.array([gaussian_sp_overlap(v) for v in vecs])
    np.testing.assert_allclose(table.blocks(vecs), ref, atol=2e-7)
    # s on A, p on B along the bond: <s_A | p_B> points against d, i.e. (y, z, x) of -d
    sp = table.blocks(vecs)[:, 0, 1:]
    np.testing.assert_allclose(sp / np.linalg.norm(sp, axis=1, keepdims=True),
                               -real_sph_harm(1, vecs) / C1, atol=1e-6)


def test_reverse_bond_is_transpose():
    """Same functions on both atoms: S(-d) = S(d)^T (parity of the channels)."""
    table = gaussian_table(np.linspace(0.5, 4.0, 200), alpha=ALPHA, beta=ALPHA)
    vecs = np.random.default_rng(2).normal(size=(10, 3)) * 1.3
    np.testing.assert_allclose(table.blocks(-vecs), np.swapaxes(table.blocks(vecs), 1, 2), atol=1e-12)


def test_bond_frame_inverse_rotation():
    """Rotating an assembled block back to the bond frame gives the table channels."""
    table = gaussian_table(np.linspace(0.5, 4.0, 200))
    vecs = np.random.default_rng(3).normal(size=(8, 3)) * 1.5
    M = wigner_blockdiag(table.shells_a, rotation_from_z(vecs))
    B0 = np.swapaxes(M, 1, 2) @ table.blocks(vecs) @ M
    vals, resid = bond_blocks_to_channels(B0, table.shells_a, table.shells_b)
    assert resid.max() < 1e-13
    np.testing.assert_allclose(vals, table.evaluate(np.linalg.norm(vecs, axis=1)), atol=1e-13)


def test_cutoffs_and_range():
    r = np.linspace(1.0, 10.0, 50)
    shells = (0, 1)  # channels (s,s,0), (s,p,0), (p,s,0), (p,p,0), (p,p,1)
    radii = np.array([2.0, 4.5])
    table = TwoCentreTable("C", "C", shells, shells, r, np.ones((50, 5)), radii, radii)
    assert table.pair_cutoff == 9.0
    assert table.channel_cutoff.tolist() == [4.0, 6.5, 6.5, 9.0, 9.0]
    d = np.array([3.99, 4.0, 6.4, 6.5, 8.99, 9.0, 12.0])
    np.testing.assert_array_equal(table.evaluate(d) != 0, d[:, None] < table.channel_cutoff[None, :])
    with pytest.raises(ValueError, match="below the table range"):
        table.evaluate(np.array([0.5]))


def test_save_load(tmp_path):
    table = gaussian_table(np.linspace(0.5, 4.0, 20))
    path = table.save(tmp_path / "t.npz")
    back = TwoCentreTable.load(path)
    assert back.shells_a == table.shells_a and back.element_a == "C"
    np.testing.assert_array_equal(back.values, table.values)
    np.testing.assert_array_equal(back.onsite["C"], table.onsite["C"])
    d = np.array([0.7, 1.33, 3.9])
    np.testing.assert_array_equal(back.evaluate(d), table.evaluate(d))
