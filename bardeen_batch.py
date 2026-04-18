from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

import basis_utils as bint


_cart_lmn_order = bint.gaussian_cart_order
_n_bf_in_shell = bint.gaussian_basis_count
_norm_cart_gto = bint.norm_cart_gto
_get_shell_bf_info = bint.gaussian_shell_components
_c2s_matrix = bint.c2s_matrix


@dataclass(frozen=True)
class PrimitiveTerm:
    ao_index: int
    la: int
    lb: int
    lc: int
    alpha: float
    coeff_norm: float
    ax: float
    ay: float
    az: float


@dataclass(frozen=True)
class BardeenBatchPlan:
    tip_components: tuple[tuple[int, int, int], ...]
    n_cart_total: int
    n_basis: int
    transform: np.ndarray | None
    terms: tuple[PrimitiveTerm, ...]


def _load_cupy(required: bool = False):
    try:
        import cupy as cp  # type: ignore
    except ImportError:
        if required:
            raise RuntimeError(
                "GPU backend requested but CuPy is not installed in the active Python environment."
            ) from None
        return None
    return cp


def resolve_backend(name: str):
    normalized = (name or "legacy").strip().lower()
    if normalized in {"legacy", "cpu", "batched"}:
        return normalized, np
    if normalized == "auto":
        cp = _load_cupy(required=False)
        if cp is None:
            return "batched", np
        return "gpu", cp
    if normalized == "gpu":
        return "gpu", _load_cupy(required=True)
    raise ValueError(f"Unsupported STM backend '{name}'")


def to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    cp = _load_cupy(required=False)
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def build_bardeen_batch_plan(mol, tip_lmn) -> BardeenBatchPlan:
    l_tip = int(sum(tip_lmn))
    tip_components = (tuple(int(x) for x in tip_lmn),) if l_tip > 0 else ((0, 0, 0),)

    cart_offsets = []
    sph_offsets = []
    n_cart_total = 0
    n_sph_total = 0
    for sh in mol.shells:
        cart_offsets.append(n_cart_total)
        sph_offsets.append(n_sph_total)
        l_value = abs(sh.shell_type)
        if sh.shell_type == -1:
            n_cart_total += 4
            n_sph_total += 4
        elif sh.shell_type >= 0:
            n_shell = (l_value + 1) * (l_value + 2) // 2
            n_cart_total += n_shell
            n_sph_total += n_shell
        else:
            n_cart_total += (l_value + 1) * (l_value + 2) // 2
            n_sph_total += 2 * l_value + 1

    terms = []
    for ish, sh in enumerate(mol.shells):
        ao_offset = cart_offsets[ish]
        ax, ay, az = sh.center
        for local_ao, (lmn_ao, prims) in enumerate(_get_shell_bf_info(sh)):
            la, lb, lc = lmn_ao
            ao_index = ao_offset + local_ao
            for alpha_ao, coeff_ao in prims:
                coeff_norm = coeff_ao * _norm_cart_gto(alpha_ao, la, lb, lc)
                terms.append(
                    PrimitiveTerm(
                        ao_index=ao_index,
                        la=la,
                        lb=lb,
                        lc=lc,
                        alpha=float(alpha_ao),
                        coeff_norm=float(coeff_norm),
                        ax=float(ax),
                        ay=float(ay),
                        az=float(az),
                    )
                )

    has_pure = any(sh.shell_type < -1 for sh in mol.shells)
    transform = None
    n_basis = n_cart_total
    if has_pure:
        transform = np.zeros((n_sph_total, n_cart_total))
        for ish, sh in enumerate(mol.shells):
            cart_offset = cart_offsets[ish]
            sph_offset = sph_offsets[ish]
            l_value = abs(sh.shell_type)
            if sh.shell_type >= 0 or sh.shell_type == -1:
                n_shell = _n_bf_in_shell(sh.shell_type)
                transform[sph_offset : sph_offset + n_shell, cart_offset : cart_offset + n_shell] = np.eye(n_shell)
            else:
                n_cart = (l_value + 1) * (l_value + 2) // 2
                n_sph = 2 * l_value + 1
                transform[sph_offset : sph_offset + n_sph, cart_offset : cart_offset + n_cart] = _c2s_matrix(l_value)
        n_basis = n_sph_total

    return BardeenBatchPlan(
        tip_components=tip_components,
        n_cart_total=n_cart_total,
        n_basis=n_basis,
        transform=transform,
        terms=tuple(terms),
    )


def _overlap_1d_value_batch(n1: int, n2: int, pa, pb, gamma: float, xp):
    n_points = pa.shape[0]
    table = xp.zeros((n1 + 1, n2 + 1, n_points), dtype=pa.dtype)
    table[0, 0, :] = 1.0
    inv_2gamma = 1.0 / (2.0 * gamma)

    for i_value in range(1, n1 + 1):
        table[i_value, 0, :] = pa * table[i_value - 1, 0, :]
        if i_value > 1:
            table[i_value, 0, :] += (i_value - 1) * inv_2gamma * table[i_value - 2, 0, :]

    for j_value in range(1, n2 + 1):
        for i_value in range(n1 + 1):
            table[i_value, j_value, :] = pb * table[i_value, j_value - 1, :]
            if j_value > 1:
                table[i_value, j_value, :] += (j_value - 1) * inv_2gamma * table[i_value, j_value - 2, :]
            if i_value > 0:
                table[i_value, j_value, :] += i_value * inv_2gamma * table[i_value - 1, j_value - 1, :]

    return table[n1, n2, :]


def _gaussian_moment_1d_batch(alpha_ao: float, center_ao: float, n_ao: int, tip_alpha: float, tip_coord, tip_l: int, xp):
    gamma = alpha_ao + tip_alpha
    mu = alpha_ao * tip_alpha / gamma
    p_center = (alpha_ao * center_ao + tip_alpha * tip_coord) / gamma
    ab = center_ao - tip_coord
    prefactor = math.sqrt(math.pi / gamma) * xp.exp(-mu * ab**2)
    pa = p_center - center_ao
    pb = p_center - tip_coord
    return prefactor * _overlap_1d_value_batch(n_ao, tip_l, pa, pb, gamma, xp)


def _poly_gauss_batch(dz, n_value: int, alpha: float, xp):
    return xp.power(dz, n_value) * xp.exp(-alpha * dz**2)


def _dpoly_gauss_batch(dz, n_value: int, alpha: float, xp):
    exp_term = xp.exp(-alpha * dz**2)
    result = -2.0 * alpha * xp.power(dz, n_value + 1) * exp_term
    if n_value > 0:
        result += n_value * xp.power(dz, n_value - 1) * exp_term
    return result


def compute_bardeen_ao_tip_batch(plan: BardeenBatchPlan, tip_alpha: float, tip_centers, z0: float, xp=np):
    tip_centers = xp.asarray(tip_centers, dtype=float)
    if tip_centers.ndim == 1:
        tip_centers = tip_centers.reshape(1, 3)

    xt = tip_centers[:, 0]
    yt = tip_centers[:, 1]
    zt = tip_centers[:, 2]
    n_points = tip_centers.shape[0]

    t_cart = xp.zeros((n_points, plan.n_cart_total, len(plan.tip_components)), dtype=tip_centers.dtype)

    for tip_index, (tl, tm, tn) in enumerate(plan.tip_components):
        dz_tip = z0 - zt
        g_tip = _poly_gauss_batch(dz_tip, tn, tip_alpha, xp)
        dg_tip = _dpoly_gauss_batch(dz_tip, tn, tip_alpha, xp)

        for term in plan.terms:
            ix = _gaussian_moment_1d_batch(term.alpha, term.ax, term.la, tip_alpha, xt, tl, xp)
            iy = _gaussian_moment_1d_batch(term.alpha, term.ay, term.lb, tip_alpha, yt, tm, xp)

            dz_ao = z0 - term.az
            exp_ao = math.exp(-term.alpha * dz_ao**2)
            g_ao = (dz_ao**term.lc) * exp_ao
            dg_ao = -2.0 * term.alpha * (dz_ao ** (term.lc + 1)) * exp_ao
            if term.lc > 0:
                dg_ao += term.lc * (dz_ao ** (term.lc - 1)) * exp_ao

            tz = g_ao * dg_tip - g_tip * dg_ao
            t_cart[:, term.ao_index, tip_index] += -0.5 * term.coeff_norm * ix * iy * tz

    if plan.transform is None:
        return t_cart

    transform = xp.asarray(plan.transform, dtype=t_cart.dtype)
    return xp.einsum("sc,pck->psk", transform, t_cart)
