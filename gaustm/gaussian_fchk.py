from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .basis_utils import GaussianShell as BasisShell


__all__ = ["BasisShell", "MolData", "read_fchk"]


@dataclass
class MolData:
    method: str
    n_atoms: int
    n_basis: int
    n_mo: int
    n_alpha: int
    n_beta: int
    atoms: list[dict]
    shells: list[BasisShell]
    mo_alpha: np.ndarray
    mo_beta: Optional[np.ndarray] = None


def read_fchk(filepath: str) -> MolData:
    with open(filepath, "r") as handle:
        lines = handle.readlines()

    sections: dict[str, tuple[int, bool, str, int | str]] = {}

    for idx, line in enumerate(lines[2:], start=2):
        if len(line) < 43:
            continue
        name = line[:40].strip()
        type_code = line[43:44] if len(line) > 43 else ""
        rest = line[44:].strip()

        if "N=" in rest:
            sections[name] = (idx, True, type_code, int(rest.split("N=")[1].strip()))
        elif type_code in ("I", "R"):
            sections[name] = (idx, False, type_code, rest)

    def read_scalar_int(name: str) -> int:
        section = sections.get(name)
        if section is None or section[1]:
            return 0
        return int(section[3])

    def read_array(name: str, dtype=float) -> np.ndarray:
        section = sections.get(name)
        if section is None or not section[1]:
            return np.array([])

        idx, _, _, count = section
        values: list[str] = []
        line_idx = idx + 1
        while len(values) < count and line_idx < len(lines):
            values.extend(lines[line_idx].split())
            line_idx += 1

        if dtype == int:
            return np.array([int(value) for value in values[:count]])
        return np.array([float(value.replace("D", "E").replace("d", "e")) for value in values[:count]])

    n_atoms = read_scalar_int("Number of atoms")
    n_basis = read_scalar_int("Number of basis functions")
    n_alpha = read_scalar_int("Number of alpha electrons")
    n_beta = read_scalar_int("Number of beta electrons")
    n_indep = read_scalar_int("Number of independent functions")
    n_mo = n_indep if n_indep > 0 else n_basis

    atomic_numbers = read_array("Atomic numbers", dtype=int)
    coordinates = read_array("Current cartesian coordinates")
    atoms = [
        {
            "Z": int(atomic_numbers[i]),
            "center": coordinates[3 * i : 3 * i + 3].copy(),
        }
        for i in range(n_atoms)
    ]

    shell_types = read_array("Shell types", dtype=int)
    n_prims = read_array("Number of primitives per shell", dtype=int)
    shell_to_atom = read_array("Shell to atom map", dtype=int)
    primitive_exponents = read_array("Primitive exponents")
    coefficients = read_array("Contraction coefficients")
    sp_coefficients = read_array("P(S=P) Contraction coefficients")

    shells: list[BasisShell] = []
    prim_offset = 0
    for shell_index, shell_type in enumerate(shell_types):
        n_prim = int(n_prims[shell_index])
        atom_index = int(shell_to_atom[shell_index]) - 1
        coeffs_sp = None
        if int(shell_type) == -1 and len(sp_coefficients) > 0:
            coeffs_sp = sp_coefficients[prim_offset : prim_offset + n_prim].copy()

        shells.append(
            BasisShell(
                atom_idx=atom_index,
                center=atoms[atom_index]["center"].copy(),
                shell_type=int(shell_type),
                exponents=primitive_exponents[prim_offset : prim_offset + n_prim].copy(),
                coeffs=coefficients[prim_offset : prim_offset + n_prim].copy(),
                coeffs_sp=coeffs_sp,
            )
        )
        prim_offset += n_prim

    mo_alpha = read_array("Alpha MO coefficients").reshape(n_mo, n_basis)
    mo_beta_flat = read_array("Beta MO coefficients")
    mo_beta = mo_beta_flat.reshape(n_mo, n_basis) if len(mo_beta_flat) > 0 else None
    method = "U" if mo_beta is not None else "R"

    return MolData(
        method=method,
        n_atoms=n_atoms,
        n_basis=n_basis,
        n_mo=n_mo,
        n_alpha=n_alpha,
        n_beta=n_beta,
        atoms=atoms,
        shells=shells,
        mo_alpha=mo_alpha,
        mo_beta=mo_beta,
    )
