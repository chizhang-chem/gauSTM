#!/usr/bin/env python3
"""
transition_dipole.py — delta-SCF 态间跃迁偶极矩计算
====================================================
基于 Löwdin 公式，计算两个 Slater 行列式之间的跃迁偶极矩:

    <U|mu|V> = det(d_uv) * Tr(mu_MO  d_uv^{-1})

对自旋非限制 (UKS/UHF):
    <U|V>    = det(d_alpha) * det(d_beta)
    <U|mu|V> = <U|V> * [ Tr(mu_a  d_a^{-1}) + Tr(mu_b  d_b^{-1}) ]

用法:
    python transition_dipole.py  state_U.fchk  state_V.fchk
"""

import sys
import numpy as np
from numpy.linalg import det, inv
from integrals import read_fchk, compute_ao_integrals


def compute_transition_dipole(mol_U, mol_V, S_ao, D_ao):
    """
    计算两个同电子数 delta-SCF 态之间的跃迁偶极矩。

    参数:
        mol_U, mol_V: 两个态的 MolData
        S_ao: AO 重叠矩阵 [n_basis, n_basis]
        D_ao: AO 偶极矩阵 [3, n_basis, n_basis]

    返回:
        tdm: 跃迁偶极矩 [mu_x, mu_y, mu_z] (a.u.)
        overlap: <U|V>
        info: dict 包含中间量
    """
    Na_U, Nb_U = mol_U.n_alpha, mol_U.n_beta
    Na_V, Nb_V = mol_V.n_alpha, mol_V.n_beta

    if Na_U != Na_V or Nb_U != Nb_V:
        raise ValueError(
            f"电子数不匹配: U=({Na_U}a,{Nb_U}b), V=({Na_V}a,{Nb_V}b)。"
            f"跃迁偶极矩要求两态电子数相同。")

    # --- alpha 自旋 ---
    Ca_U = mol_U.mo_alpha[:Na_U, :]     # [Na, n_basis]
    Ca_V = mol_V.mo_alpha[:Na_V, :]

    # MO 交叉重叠: d_a(k,l) = <u_k|v_l>
    d_alpha = Ca_U @ S_ao @ Ca_V.T      # [Na, Na]

    # MO 交叉偶极积分: mu_a(k,l) = <u_k|r|v_l>
    mu_alpha = np.zeros((3, Na_U, Na_V))
    for c in range(3):
        mu_alpha[c] = Ca_U @ D_ao[c] @ Ca_V.T

    det_alpha = det(d_alpha)
    inv_alpha = inv(d_alpha)

    # --- beta 自旋 ---
    Cb_U = mol_U.mo_beta[:Nb_U, :] if mol_U.mo_beta is not None else mol_U.mo_alpha[:Nb_U, :]
    Cb_V = mol_V.mo_beta[:Nb_V, :] if mol_V.mo_beta is not None else mol_V.mo_alpha[:Nb_V, :]

    d_beta = Cb_U @ S_ao @ Cb_V.T
    mu_beta = np.zeros((3, Nb_U, Nb_V))
    for c in range(3):
        mu_beta[c] = Cb_U @ D_ao[c] @ Cb_V.T

    det_beta = det(d_beta)
    inv_beta = inv(d_beta)

    # --- Löwdin 公式 ---
    overlap = det_alpha * det_beta

    # <U|mu_c|V> / <U|V> = Tr(mu_a_c @ inv_a) + Tr(mu_b_c @ inv_b)
    # <U|mu_c|V> = <U|V> * [Tr(mu_a_c @ inv_a) + Tr(mu_b_c @ inv_b)]
    tdm = np.zeros(3)
    for c in range(3):
        tr_a = np.trace(mu_alpha[c] @ inv_alpha)
        tr_b = np.trace(mu_beta[c]  @ inv_beta)
        tdm[c] = overlap * (tr_a + tr_b)

    info = {
        'det_alpha': det_alpha,
        'det_beta':  det_beta,
        'overlap':   overlap,
        'd_alpha':   d_alpha,
        'd_beta':    d_beta,
    }

    return tdm, overlap, info


def orthogonalize_tdm(tdm_UV, mu_UU, mu_VV, S_UV):
    """
    对非正交 delta-SCF 态进行正交化校正。

    Gram-Schmidt: |V'> = (|V> - S_UV |U>) / sqrt(1 - |S_UV|^2)

    正交化后的跃迁偶极矩:
        <U|mu|V'> = (<U|mu|V> - S_UV * <U|mu|U>) / sqrt(1 - |S_UV|^2)

    参数:
        tdm_UV: <U|mu|V> [3]
        mu_UU:  <U|mu|U> [3]
        mu_VV:  <V|mu|V> [3] (未使用，留作将来 Löwdin 对称正交化)
        S_UV:   <U|V>

    返回:
        tdm_orth: 正交化后的跃迁偶极矩 [3]
    """
    norm = np.sqrt(1.0 - abs(S_UV) ** 2)
    if norm < 1e-12:
        print("WARNING: 两态近乎平行 (|<U|V>| ~ 1)，正交化不稳定", file=sys.stderr)
        return tdm_UV
    return (tdm_UV - S_UV * mu_UU) / norm


def main():
    if len(sys.argv) < 3:
        print("用法: python transition_dipole.py  state_U.fchk  state_V.fchk",
              file=sys.stderr)
        sys.exit(1)

    fchk_U = sys.argv[1]
    fchk_V = sys.argv[2]

    print(f"读取态 U: {fchk_U}", file=sys.stderr)
    mol_U = read_fchk(fchk_U)
    print(f"  method={mol_U.method}  Na={mol_U.n_alpha}  Nb={mol_U.n_beta}  "
          f"Nbasis={mol_U.n_basis}  Nmo={mol_U.n_mo}", file=sys.stderr)

    print(f"读取态 V: {fchk_V}", file=sys.stderr)
    mol_V = read_fchk(fchk_V)
    print(f"  method={mol_V.method}  Na={mol_V.n_alpha}  Nb={mol_V.n_beta}  "
          f"Nbasis={mol_V.n_basis}  Nmo={mol_V.n_mo}", file=sys.stderr)

    # 偶极矩参考原点取坐标原点
    origin = np.zeros(3)

    # --- 计算 AO 积分 ---
    # 两个 fchk 来自不同 SCF，但共享相同的 AO 基组 (同一分子，同一基组)
    # S_ao 和 D_ao 只依赖于基组，不依赖 MO 系数
    print("计算 AO 重叠积分和偶极积分...", file=sys.stderr)
    S_ao, D_ao = compute_ao_integrals(mol_U, mol_U, dipole_origin=origin)
    print(f"  AO overlap 对角线范围: [{np.diag(S_ao).min():.6f}, "
          f"{np.diag(S_ao).max():.6f}]", file=sys.stderr)

    # --- 验证 AO 基组一致性 ---
    if mol_U.n_basis != mol_V.n_basis:
        print(f"ERROR: 基函数数目不同 ({mol_U.n_basis} vs {mol_V.n_basis})",
              file=sys.stderr)
        sys.exit(1)

    # --- 计算跃迁偶极矩 ---
    print("计算跃迁偶极矩 (Löwdin 公式)...", file=sys.stderr)
    tdm, overlap, info = compute_transition_dipole(mol_U, mol_V, S_ao, D_ao)

    # --- 计算两态各自的偶极矩 (用于正交化) ---
    tdm_UU, _, _ = compute_transition_dipole(mol_U, mol_U, S_ao, D_ao)
    tdm_VV, _, _ = compute_transition_dipole(mol_V, mol_V, S_ao, D_ao)

    # --- 正交化校正 ---
    tdm_orth = orthogonalize_tdm(tdm, tdm_UU, tdm_VV, overlap)

    # --- 输出 ---
    au_to_debye = 2.541746  # 1 a.u. = 2.541746 Debye

    print("\n" + "=" * 60)
    print("  delta-SCF 跃迁偶极矩计算结果")
    print("=" * 60)
    print(f"\n  <U|V> = {overlap:.10f}")
    print(f"  det(d_alpha) = {info['det_alpha']:.10f}")
    print(f"  det(d_beta)  = {info['det_beta']:.10f}")

    print(f"\n  态 U 偶极矩 <U|mu|U> (a.u.):")
    print(f"    mu_x = {tdm_UU[0]:12.6f}")
    print(f"    mu_y = {tdm_UU[1]:12.6f}")
    print(f"    mu_z = {tdm_UU[2]:12.6f}")
    print(f"    |mu| = {np.linalg.norm(tdm_UU):12.6f}  "
          f"({np.linalg.norm(tdm_UU) * au_to_debye:.6f} Debye)")

    print(f"\n  态 V 偶极矩 <V|mu|V> (a.u.):")
    print(f"    mu_x = {tdm_VV[0]:12.6f}")
    print(f"    mu_y = {tdm_VV[1]:12.6f}")
    print(f"    mu_z = {tdm_VV[2]:12.6f}")
    print(f"    |mu| = {np.linalg.norm(tdm_VV):12.6f}  "
          f"({np.linalg.norm(tdm_VV) * au_to_debye:.6f} Debye)")

    print(f"\n  跃迁偶极矩 <U|mu|V> (未正交化, a.u.):")
    print(f"    mu_x = {tdm[0]:12.6f}")
    print(f"    mu_y = {tdm[1]:12.6f}")
    print(f"    mu_z = {tdm[2]:12.6f}")
    print(f"    |mu| = {np.linalg.norm(tdm):12.6f}  "
          f"({np.linalg.norm(tdm) * au_to_debye:.6f} Debye)")

    print(f"\n  跃迁偶极矩 (Gram-Schmidt 正交化后, a.u.):")
    print(f"    mu_x = {tdm_orth[0]:12.6f}")
    print(f"    mu_y = {tdm_orth[1]:12.6f}")
    print(f"    mu_z = {tdm_orth[2]:12.6f}")
    print(f"    |mu| = {np.linalg.norm(tdm_orth):12.6f}  "
          f"({np.linalg.norm(tdm_orth) * au_to_debye:.6f} Debye)")

    # 振子强度 f = 2/3 * dE * |mu|^2
    print(f"\n  注: 振子强度 f = (2/3) * Delta_E * |mu_orth|^2")
    print(f"       需要提供两态能量差 Delta_E (Hartree)")
    print("=" * 60)


if __name__ == '__main__':
    main()
