from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .gaussian_fchk import MolData

try:
    from pyscf import gto as pyscf_gto
except ImportError:  # pragma: no cover
    pyscf_gto = None


__all__ = [
    "pyscf_available",
    "compute_ao_overlap",
]


_ELEMENT_SYMBOLS = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


@dataclass(frozen=True)
class _PrimitiveShell:
    atom_index: int
    shell_index: int
    role: str
    angular_momentum: int
    primitive_index: int
    coefficient: float


@dataclass(frozen=True)
class _PreparedCartMole:
    mol: any
    transform: np.ndarray


def pyscf_available() -> bool:
    return pyscf_gto is not None


def _atomic_symbol(z_value: int) -> str:
    if z_value <= 0 or z_value >= len(_ELEMENT_SYMBOLS):
        raise ValueError(f"Unsupported atomic number: {z_value}")
    return _ELEMENT_SYMBOLS[z_value]


def _atom_label(z_value: int, atom_index: int) -> str:
    return f"{_atomic_symbol(z_value)}@{atom_index}"


def _cartesian_component_count(l_value: int) -> int:
    return (l_value + 1) * (l_value + 2) // 2


def _gaussian_basis_count(shell_type: int) -> int:
    if shell_type == -1:
        return 4
    if shell_type >= 0:
        return _cartesian_component_count(shell_type)
    return 2 * abs(shell_type) + 1


def _canonical_cart_order(l_value: int) -> list[tuple[int, int, int]]:
    order: list[tuple[int, int, int]] = []
    for lx in reversed(range(l_value + 1)):
        for ly in reversed(range(l_value + 1 - lx)):
            lz = l_value - lx - ly
            order.append((lx, ly, lz))
    return order


def _gaussian_cart_order(l_value: int) -> list[tuple[int, int, int]]:
    if l_value == 0:
        return [(0, 0, 0)]
    if l_value == 1:
        return [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if l_value == 2:
        return [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    if l_value == 3:
        return [
            (3, 0, 0),
            (0, 3, 0),
            (0, 0, 3),
            (1, 2, 0),
            (2, 1, 0),
            (2, 0, 1),
            (1, 0, 2),
            (0, 1, 2),
            (0, 2, 1),
            (1, 1, 1),
        ]
    return _canonical_cart_order(l_value)


def _gaussian_cart_reorder(l_value: int) -> np.ndarray:
    canonical = _canonical_cart_order(l_value)
    gaussian = _gaussian_cart_order(l_value)
    canonical_index = {lmn: idx for idx, lmn in enumerate(canonical)}

    transform = np.zeros((len(gaussian), len(canonical)))
    for row, lmn in enumerate(gaussian):
        transform[row, canonical_index[lmn]] = 1.0
    return transform


def _pyscf_spherical_order(l_value: int) -> list[int]:
    return list(range(-l_value, l_value + 1))


def _gaussian_spherical_order(l_value: int) -> list[int]:
    order = [0]
    for m_value in range(1, l_value + 1):
        order.extend([m_value, -m_value])
    return order


def _gaussian_spherical_reorder(l_value: int) -> np.ndarray:
    pyscf_order = _pyscf_spherical_order(l_value)
    gaussian_order = _gaussian_spherical_order(l_value)
    pyscf_index = {m_value: idx for idx, m_value in enumerate(pyscf_order)}

    transform = np.zeros((len(gaussian_order), len(pyscf_order)))
    for row, m_value in enumerate(gaussian_order):
        transform[row, pyscf_index[m_value]] = 1.0
    return transform


def _primitive_basis_entry(l_value: int, exponent: float) -> list:
    return [int(l_value), [float(exponent), 1.0]]


def _expand_to_primitive_basis(
    mol: MolData,
) -> tuple[list[tuple[str, np.ndarray]], dict[str, list[list]], list[_PrimitiveShell]]:
    atoms: list[tuple[str, np.ndarray]] = []
    basis: dict[str, list[list]] = {}
    primitive_shells: list[_PrimitiveShell] = []

    for atom_index, atom in enumerate(mol.atoms):
        label = _atom_label(int(atom["Z"]), atom_index)
        atoms.append((label, np.asarray(atom["center"], dtype=float)))
        basis[label] = []

    for shell_index, shell in enumerate(mol.shells):
        atom_index = shell.atom_idx
        if atom_index is None:
            raise ValueError(f"Shell {shell_index} is missing atom_idx")
        label = atoms[atom_index][0]

        if shell.shell_type == -1:
            for primitive_index, (exponent, coefficient) in enumerate(zip(shell.exponents, shell.coeffs)):
                basis[label].append(_primitive_basis_entry(0, float(exponent)))
                primitive_shells.append(
                    _PrimitiveShell(
                        atom_index=atom_index,
                        shell_index=shell_index,
                        role="sp_s",
                        angular_momentum=0,
                        primitive_index=primitive_index,
                        coefficient=float(coefficient),
                    )
                )
            for primitive_index, (exponent, coefficient) in enumerate(zip(shell.exponents, shell.coeffs_sp)):
                basis[label].append(_primitive_basis_entry(1, float(exponent)))
                primitive_shells.append(
                    _PrimitiveShell(
                        atom_index=atom_index,
                        shell_index=shell_index,
                        role="sp_p",
                        angular_momentum=1,
                        primitive_index=primitive_index,
                        coefficient=float(coefficient),
                    )
                )
            continue

        l_value = abs(shell.shell_type)
        for primitive_index, (exponent, coefficient) in enumerate(zip(shell.exponents, shell.coeffs)):
            basis[label].append(_primitive_basis_entry(l_value, float(exponent)))
            primitive_shells.append(
                _PrimitiveShell(
                    atom_index=atom_index,
                    shell_index=shell_index,
                    role="full",
                    angular_momentum=l_value,
                    primitive_index=primitive_index,
                    coefficient=float(coefficient),
                )
            )

    return atoms, basis, primitive_shells


def _sorted_primitive_shells(primitive_shells: list[_PrimitiveShell], n_atoms: int) -> list[_PrimitiveShell]:
    ordered: list[_PrimitiveShell] = []
    for atom_index in range(n_atoms):
        atom_shells = [shell for shell in primitive_shells if shell.atom_index == atom_index]
        atom_shells.sort(key=lambda shell: shell.angular_momentum)
        ordered.extend(atom_shells)
    return ordered


def _build_pyscf_cart_mole(mol: MolData, atoms, basis):
    if pyscf_gto is None:
        raise ImportError("PySCF is required for the PySCF AO-overlap backend")

    pyscf_mol = pyscf_gto.Mole()
    pyscf_mol.verbose = 0
    pyscf_mol.atom = atoms
    pyscf_mol.basis = basis
    pyscf_mol.unit = "Bohr"
    pyscf_mol.cart = True
    pyscf_mol.spin = int(mol.n_alpha - mol.n_beta)
    pyscf_mol.charge = int(sum(int(atom["Z"]) for atom in mol.atoms) - mol.n_alpha - mol.n_beta)
    pyscf_mol.build()
    return pyscf_mol


def _primitive_block_transform(shell_type: int, raw_self_block: np.ndarray) -> np.ndarray:
    l_value = abs(shell_type)
    source_size = _cartesian_component_count(l_value)

    if l_value == 0:
        return np.array([[1.0]])
    if l_value == 1:
        return np.eye(3)

    if shell_type > 0:
        diagonal = np.diag(raw_self_block)
        if np.any(diagonal <= 0.0):
            raise ValueError("Encountered non-positive diagonal in primitive cart self-overlap block")
        normalize = np.diag(1.0 / np.sqrt(diagonal))
        return _gaussian_cart_reorder(l_value) @ normalize

    cart2sph = pyscf_gto.cart2sph(l_value, normalized="sp")
    if cart2sph.shape != (source_size, 2 * l_value + 1):
        raise ValueError(f"Unexpected cart2sph shape for l={l_value}: {cart2sph.shape}")
    return _gaussian_spherical_reorder(l_value) @ cart2sph.T


def _build_fchk_transform(mol: MolData, primitive_shells: list[_PrimitiveShell], raw_self_overlap: np.ndarray) -> np.ndarray:
    source_offsets: dict[tuple[int, str, int], int] = {}
    source_total = 0
    for shell in primitive_shells:
        source_offsets[(shell.shell_index, shell.role, shell.primitive_index)] = source_total
        source_total += _cartesian_component_count(shell.angular_momentum)

    transform = np.zeros((mol.n_basis, source_total))
    row_offset = 0

    for shell_index, shell in enumerate(mol.shells):
        if shell.shell_type == -1:
            for primitive_index, coefficient in enumerate(shell.coeffs):
                source_offset = source_offsets[(shell_index, "sp_s", primitive_index)]
                transform[row_offset, source_offset] += float(coefficient)
            row_offset += 1

            for primitive_index, coefficient in enumerate(shell.coeffs_sp):
                source_offset = source_offsets[(shell_index, "sp_p", primitive_index)]
                transform[row_offset : row_offset + 3, source_offset : source_offset + 3] += float(coefficient) * np.eye(3)
            row_offset += 3
            continue

        l_value = abs(shell.shell_type)
        source_size = _cartesian_component_count(l_value)
        target_size = _gaussian_basis_count(shell.shell_type)
        for primitive_index, coefficient in enumerate(shell.coeffs):
            source_offset = source_offsets[(shell_index, "full", primitive_index)]
            raw_block = raw_self_overlap[source_offset : source_offset + source_size, source_offset : source_offset + source_size]
            transform[row_offset : row_offset + target_size, source_offset : source_offset + source_size] += (
                float(coefficient) * _primitive_block_transform(shell.shell_type, raw_block)
            )
        row_offset += target_size

    if row_offset != mol.n_basis:
        raise ValueError(f"AO transform row mismatch: built {row_offset}, expected {mol.n_basis}")
    return transform


def _prepare_cart_mole(mol: MolData) -> _PreparedCartMole:
    atoms, basis, primitive_shells = _expand_to_primitive_basis(mol)
    ordered_shells = _sorted_primitive_shells(primitive_shells, mol.n_atoms)
    pyscf_mol = _build_pyscf_cart_mole(mol, atoms, basis)
    raw_self_overlap = pyscf_mol.intor_symmetric("int1e_ovlp_cart")
    transform = _build_fchk_transform(mol, ordered_shells, raw_self_overlap)

    if pyscf_mol.nao_nr() != transform.shape[1]:
        raise ValueError(
            f"PySCF cart AO count mismatch: Mole has {pyscf_mol.nao_nr()}, transform expects {transform.shape[1]}"
        )

    return _PreparedCartMole(mol=pyscf_mol, transform=transform)


def compute_ao_overlap(mol_bra: MolData, mol_ket: Optional[MolData] = None) -> np.ndarray:
    if pyscf_gto is None:
        raise ImportError("PySCF is not available in the current Python environment")

    if mol_ket is None:
        mol_ket = mol_bra

    prepared_bra = _prepare_cart_mole(mol_bra)
    if mol_ket is mol_bra:
        s_cart = prepared_bra.mol.intor_symmetric("int1e_ovlp_cart")
        prepared_ket = prepared_bra
    else:
        prepared_ket = _prepare_cart_mole(mol_ket)
        s_cart = pyscf_gto.intor_cross("int1e_ovlp_cart", prepared_bra.mol, prepared_ket.mol)

    return prepared_bra.transform @ s_cart @ prepared_ket.transform.T
