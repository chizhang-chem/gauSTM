from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Optional

import numpy as np
from pyscf import gto as pyscf_gto


__all__ = [
    "GaussianShell",
    "adapt_gaussian_shell",
    "gaussian_cart_order",
    "gaussian_basis_count",
    "gaussian_shell_components",
    "norm_cart_gto",
    "overlap_1d_table",
    "c2s_matrix",
]


_MISSING = object()


def _as_float_1d(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return array


def _as_float_xyz(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return array


def _get_field(obj: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
    elif hasattr(obj, name):
        return getattr(obj, name)

    if default is _MISSING:
        raise AttributeError(f"Missing field '{name}'")
    return default


def cartesian_component_count(l_value: int) -> int:
    return (l_value + 1) * (l_value + 2) // 2


@dataclass
class GaussianShell:
    """Gaussian/FCHK-like shell descriptor.

    shell_type follows Gaussian conventions:
      0=S, 1=P, -1=SP, 2=6D, -2=5D, 3=10F, -3=7F.
    """

    shell_type: int
    exponents: np.ndarray
    coeffs: np.ndarray
    center: np.ndarray
    coeffs_sp: Optional[np.ndarray] = None
    atom_idx: Optional[int] = None

    def __post_init__(self) -> None:
        self.shell_type = int(self.shell_type)
        self.exponents = _as_float_1d(self.exponents, name="exponents")
        self.coeffs = _as_float_1d(self.coeffs, name="coeffs")
        self.center = _as_float_xyz(self.center, name="center")
        if len(self.exponents) != len(self.coeffs):
            raise ValueError("exponents and coeffs must have the same length")
        if np.any(self.exponents <= 0):
            raise ValueError("all Gaussian exponents must be positive")
        if self.shell_type == -1:
            if self.coeffs_sp is None:
                raise ValueError("SP shells require coeffs_sp")
            self.coeffs_sp = _as_float_1d(self.coeffs_sp, name="coeffs_sp")
            if len(self.coeffs_sp) != len(self.exponents):
                raise ValueError("exponents and coeffs_sp must have the same length")
        elif self.coeffs_sp is not None:
            self.coeffs_sp = _as_float_1d(self.coeffs_sp, name="coeffs_sp")

    @property
    def angular_momentum(self) -> int:
        return 1 if self.shell_type == -1 else abs(self.shell_type)

    @property
    def n_basis(self) -> int:
        return gaussian_basis_count(self.shell_type)

    @property
    def n_prim(self) -> int:
        return len(self.exponents)

    @property
    def n_cart(self) -> int:
        if self.shell_type == -1:
            return 4
        return cartesian_component_count(abs(self.shell_type))


def adapt_gaussian_shell(shell: Any) -> GaussianShell:
    if isinstance(shell, GaussianShell):
        return shell

    coeffs_sp = _get_field(shell, "coeffs_sp", None)
    if coeffs_sp is not None:
        coeffs_sp = np.asarray(coeffs_sp, dtype=float)

    atom_idx = _get_field(shell, "atom_idx", None)
    return GaussianShell(
        shell_type=int(_get_field(shell, "shell_type")),
        exponents=np.asarray(_get_field(shell, "exponents"), dtype=float),
        coeffs=np.asarray(_get_field(shell, "coeffs"), dtype=float),
        center=np.asarray(_get_field(shell, "center"), dtype=float),
        coeffs_sp=coeffs_sp,
        atom_idx=None if atom_idx is None else int(atom_idx),
    )


def gaussian_basis_count(shell_type: int) -> int:
    if shell_type == -1:
        return 4
    if shell_type >= 0:
        return cartesian_component_count(shell_type)
    return 2 * abs(shell_type) + 1


def _canonical_cart_order(l_value: int) -> list[tuple[int, int, int]]:
    order: list[tuple[int, int, int]] = []
    for lx in reversed(range(l_value + 1)):
        for ly in reversed(range(l_value + 1 - lx)):
            lz = l_value - lx - ly
            order.append((lx, ly, lz))
    return order


def gaussian_cart_order(l_value: int) -> list[tuple[int, int, int]]:
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
    # Gaussian ordering above F is only needed internally here for pure shells.
    # Use the canonical lx/ly/lz ordering and let PySCF provide the cart->sph map.
    return _canonical_cart_order(l_value)


def _double_factorial(n_value: int) -> int:
    result = 1
    while n_value > 0:
        result *= n_value
        n_value -= 2
    return result


def norm_cart_gto(alpha: float, lx: int, ly: int, lz: int) -> float:
    l_total = lx + ly + lz
    n2 = (
        (2.0 * alpha / math.pi) ** 1.5
        * (4.0 * alpha) ** l_total
        / (
            _double_factorial(2 * lx - 1)
            * _double_factorial(2 * ly - 1)
            * _double_factorial(2 * lz - 1)
        )
    )
    return math.sqrt(n2)


def overlap_1d_table(n1: int, n2: int, pa: float, pb: float, gamma: float) -> np.ndarray:
    table = np.zeros((n1 + 1, n2 + 1))
    table[0, 0] = 1.0

    for i_value in range(1, n1 + 1):
        table[i_value, 0] = pa * table[i_value - 1, 0]
        if i_value > 1:
            table[i_value, 0] += (i_value - 1) / (2.0 * gamma) * table[i_value - 2, 0]

    for j_value in range(1, n2 + 1):
        for i_value in range(n1 + 1):
            table[i_value, j_value] = pb * table[i_value, j_value - 1]
            if j_value > 1:
                table[i_value, j_value] += (j_value - 1) / (2.0 * gamma) * table[i_value, j_value - 2]
            if i_value > 0:
                table[i_value, j_value] += i_value / (2.0 * gamma) * table[i_value - 1, j_value - 1]

    return table


def _c2s_matrix_table(l_value: int) -> np.ndarray:
    cart = gaussian_cart_order(l_value)
    cart_index = {lmn: i for i, lmn in enumerate(cart)}
    n_cart = len(cart)
    n_sph = 2 * l_value + 1
    transform = np.zeros((n_sph, n_cart))

    m_order = [0]
    for m_value in range(1, l_value + 1):
        m_order.extend([m_value, -m_value])
    m_to_row = {m_value: i for i, m_value in enumerate(m_order)}

    def set_value(m_value: int, lmn: tuple[int, int, int], value: float) -> None:
        transform[m_to_row[m_value], cart_index[lmn]] = value

    if l_value == 2:
        set_value(0, (0, 0, 2), 1.0)
        set_value(0, (2, 0, 0), -0.5)
        set_value(0, (0, 2, 0), -0.5)
        set_value(1, (1, 0, 1), 1.0)
        set_value(-1, (0, 1, 1), 1.0)
        set_value(2, (2, 0, 0), math.sqrt(3.0) / 2.0)
        set_value(2, (0, 2, 0), -math.sqrt(3.0) / 2.0)
        set_value(-2, (1, 1, 0), 1.0)
        return transform

    if l_value == 3:
        sqrt_5 = math.sqrt(5.0)
        sqrt_3 = math.sqrt(3.0)
        set_value(0, (0, 0, 3), 1.0)
        set_value(0, (2, 0, 1), -3.0 / (2.0 * sqrt_5))
        set_value(0, (0, 2, 1), -3.0 / (2.0 * sqrt_5))
        set_value(1, (1, 0, 2), 1.0 / sqrt_5)
        set_value(1, (3, 0, 0), -sqrt_3 / 4.0)
        set_value(1, (1, 2, 0), -sqrt_3 / (4.0 * sqrt_5))
        set_value(-1, (0, 1, 2), 1.0 / sqrt_5)
        set_value(-1, (0, 3, 0), -sqrt_3 / 4.0)
        set_value(-1, (2, 1, 0), -sqrt_3 / (4.0 * sqrt_5))
        set_value(2, (2, 0, 1), math.sqrt(3.0 / 8.0))
        set_value(2, (0, 2, 1), -math.sqrt(3.0 / 8.0))
        set_value(-2, (1, 1, 1), 1.0 / math.sqrt(2.0))
        set_value(3, (3, 0, 0), sqrt_5 / 4.0)
        set_value(3, (1, 2, 0), -3.0 / 4.0)
        set_value(-3, (0, 3, 0), -sqrt_5 / 4.0)
        set_value(-3, (2, 1, 0), 3.0 / 4.0)
        return transform

    raise ValueError(f"Cartesian-to-spherical lookup only supports l=2,3, got l={l_value}")


@lru_cache(maxsize=None)
def _gaussian_spherical_reorder(l_value: int) -> np.ndarray:
    pyscf_order = list(range(-l_value, l_value + 1))
    gaussian_order = [0]
    for m_value in range(1, l_value + 1):
        gaussian_order.extend([m_value, -m_value])
    pyscf_index = {m_value: idx for idx, m_value in enumerate(pyscf_order)}

    transform = np.zeros((len(gaussian_order), len(pyscf_order)))
    for row, m_value in enumerate(gaussian_order):
        transform[row, pyscf_index[m_value]] = 1.0
    return transform


@lru_cache(maxsize=None)
def _c2s_matrix(l_value: int) -> np.ndarray:
    if l_value == 0:
        return np.array([[1.0]])
    if l_value == 1:
        return np.eye(3)
    if l_value <= 3:
        return _c2s_matrix_table(l_value)

    cart2sph = pyscf_gto.cart2sph(l_value, normalized="sp")
    if cart2sph.shape != (cartesian_component_count(l_value), 2 * l_value + 1):
        raise ValueError(f"Unexpected cart2sph shape for l={l_value}: {cart2sph.shape}")
    return _gaussian_spherical_reorder(l_value) @ cart2sph.T


def c2s_matrix(l_value: int) -> np.ndarray:
    return _c2s_matrix(l_value)


def _gaussian_shell_components(shell: GaussianShell) -> list[tuple[tuple[int, int, int], tuple[tuple[float, float], ...]]]:
    if shell.shell_type == -1:
        s_prims = tuple(zip(shell.exponents, shell.coeffs))
        p_prims = tuple(zip(shell.exponents, shell.coeffs_sp))
        return [
            ((0, 0, 0), s_prims),
            ((1, 0, 0), p_prims),
            ((0, 1, 0), p_prims),
            ((0, 0, 1), p_prims),
        ]

    order = gaussian_cart_order(abs(shell.shell_type))
    prims = tuple(zip(shell.exponents, shell.coeffs))
    return [(lmn, prims) for lmn in order]


def gaussian_shell_components(shell: Any) -> list[tuple[tuple[int, int, int], tuple[tuple[float, float], ...]]]:
    return _gaussian_shell_components(adapt_gaussian_shell(shell))
