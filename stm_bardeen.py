#!/usr/bin/env python3
"""
stm_bardeen.py - Bardeen 近似 STM 模拟
======================================
基于 Fortran g_post_stm_bardeen_mod.f 重写。
fchk 解析基于 g_initial_stm_mod.f。
AO overlap 使用 PySCF 后端。

MO 系数约定: mo_alpha[i, mu] = MO i 中 AO mu 的系数 (n_mo x n_basis)
与 Fortran MOAlpha(i, mu) 一致。

用法:
    python stm_bardeen.py input.finp > output.m
"""

import sys
import math
import numpy as np
from numpy.linalg import slogdet
from dataclasses import dataclass, field
from pathlib import Path
from multiprocessing import Pool, cpu_count

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import basis_utils as bint
from bardeen_batch import (
    build_bardeen_batch_plan,
    compute_bardeen_ao_tip_batch,
    resolve_backend as resolve_scan_backend,
    to_numpy,
)
from gaussian_fchk import MolData, read_fchk
from gaussian_tddft import build_state_determinants, read_tddft_log
from pyscf_overlap import compute_ao_overlap as compute_ao_overlap_backend

# =====================================================================
# 常量
# =====================================================================
AUAA = 0.529177249       # 1 Bohr = 0.529177249 Angstrom

# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class STMConfig:
    LBardeen: bool = False
    Dyson: bool = False
    TDA: bool = False
    PrintDyson: bool = False
    Gauss: bool = False
    LSmear: bool = False
    NSmear: int = 0
    SmearX: float = 1.0
    RScale: float = 1.0
    MFchk: str = ""
    MFFchk: str = ""
    CITHLD: float = 1e-6
    TDLog: str = ""
    TDChannel: str = "both"
    TDUseY: bool = True
    TDState: int | None = None
    TDStateMin: int = 1
    TDStateMax: int = 0
    LScan: int = 0
    NScan: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))
    VScan: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    RZ0: float = 1000.0
    TAlpha: float = 0.0
    TLMN: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))
    TCenter: np.ndarray = field(default_factory=lambda: np.zeros(3))
    LMOA: bool = False
    LMOB: bool = False
    NMOA: int = 0
    NMOB: int = 0
    MOAIndex: list = field(default_factory=list)
    MOBIndex: list = field(default_factory=list)
    debug: bool = False
    n_cores: int = 10
    backend: str = "legacy"
    row_batch: int = 8

# =====================================================================
# finp 输入文件解析
# =====================================================================
def parse_finp(filepath: str) -> STMConfig:
    with open(filepath, 'r') as f:
        lines = f.readlines()

    cfg = STMConfig()
    kv = {}
    blocks = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#') or line.startswith('!'):
            i += 1; continue
        if line.lower().startswith('%block'):
            block_name = line.split()[1].strip()
            block_lines = []
            i += 1
            while i < len(lines):
                bl = lines[i].strip()
                if bl.lower().startswith('%endblock') or bl.startswith('%'):
                    break
                block_lines.append(bl)
                i += 1
            blocks[block_name.lower()] = block_lines
            i += 1; continue
        parts = line.split()
        if len(parts) >= 2:
            kv[parts[0].lower()] = parts[1]
        i += 1

    def gbool(key, d=False):
        v = kv.get(key.lower())
        return v.upper() in ('T', 'TRUE', '.TRUE.', 'YES', '1') if v else d
    def get_int(key, d=0):
        v = kv.get(key.lower())
        return int(v) if v else d
    def gfloat(key, d=0.0):
        v = kv.get(key.lower())
        return float(v.replace('D', 'E').replace('d', 'e')) if v else d
    def gstr(key, d=''):
        return kv.get(key.lower(), d)

    cfg.debug = gbool('debug')
    cfg.LBardeen = gbool('stm.bardeen')
    cfg.Dyson = gbool('stm.dyson')
    cfg.TDA = gbool('stm.tda', True)
    cfg.PrintDyson = gbool('stm.printdyson')
    cfg.CITHLD = gfloat('stm.cithld', 1e-6)
    cfg.TDLog = gstr('stm.tdlog')
    cfg.TDChannel = gstr('stm.tdchannel', 'both').lower()
    cfg.TDUseY = gbool('stm.tdusey', True)
    tdstate_raw = kv.get('stm.tdstate')
    if tdstate_raw is None:
        cfg.TDState = None
    else:
        tdstate_val = int(tdstate_raw)
        cfg.TDState = tdstate_val if tdstate_val > 0 else None
    cfg.TDStateMin = get_int('stm.tdstatemin', 1)
    cfg.TDStateMax = get_int('stm.tdstatemax', 0)
    cfg.MFchk = gstr('stm.molfchk')
    cfg.MFFchk = gstr('stm.molfinalfchk')
    cfg.Gauss = gbool('stm.gauss')
    cfg.LSmear = gbool('stm.smear')
    cfg.backend = gstr('stm.backend', 'legacy').lower()
    cfg.row_batch = max(1, get_int('stm.rowbatch', 8))
    if cfg.LSmear:
        cfg.NSmear = get_int('stm.nsmear')
        cfg.SmearX = gfloat('stm.smearx', 1.0) / AUAA
        cfg.RScale = gfloat('stm.rscale', 1.0)
    cfg.LScan = get_int('stm.scan')
    cfg.NScan = np.array([get_int('stm.scana'), get_int('stm.scanb'),
                          get_int('stm.scanc')], dtype=int)
    if 'stm.gausstip' in blocks:
        bl = blocks['stm.gausstip']
        cfg.TAlpha = float(bl[0].strip().replace('D', 'E').replace('d', 'e'))
        cfg.TLMN = np.array([int(x) for x in bl[1].split()[:3]], dtype=int)
        cfg.TCenter = np.array([float(x) for x in bl[2].split()[:3]]) / AUAA
    if 'stm.vscan' in blocks:
        for j, line in enumerate(blocks['stm.vscan']):
            cfg.VScan[:, j] = np.array([float(x) for x in line.split()[:3]]) / AUAA
    cfg.RZ0 = gfloat('stm.rz0', 1000.0)
    if cfg.RZ0 != 1000.0:
        cfg.RZ0 /= AUAA
    if not cfg.Dyson:
        cfg.LMOA = gbool('stm.moalpha')
        if not cfg.LMOA:
            cfg.LMOB = gbool('stm.mobeta')
    return cfg


# =====================================================================
# AO 重叠 (PySCF 后端)
# =====================================================================
def compute_ao_overlap(mol_bra, mol_ket=None):
    return compute_ao_overlap_backend(mol_bra, mol_ket)


# =====================================================================
# Bardeen 面积分 (AO 基)
# =====================================================================
_cart_lmn_order = bint.gaussian_cart_order
_n_bf_in_shell = bint.gaussian_basis_count
_norm_cart_gto = bint.norm_cart_gto
_overlap_1d_table = bint.overlap_1d_table
_get_shell_bf_info = bint.gaussian_shell_components
_c2s_matrix = bint.c2s_matrix


def _gaussian_moment_1d(a1, c1, n1, a2, c2, n2):
    g = a1 + a2
    mu = a1*a2/g
    P = (a1*c1 + a2*c2)/g
    AB = c1 - c2
    pf = math.sqrt(math.pi/g) * math.exp(-mu * AB**2)
    S = _overlap_1d_table(n1, n2, P-c1, P-c2, g)
    return pf * S[n1, n2]


def _poly_gauss(dz, n, alpha):
    return dz**n * math.exp(-alpha * dz**2)


def _dpoly_gauss(dz, n, alpha):
    g = math.exp(-alpha * dz**2)
    result = -2*alpha * dz**(n+1) * g
    if n > 0:
        result += n * dz**(n-1) * g
    return result


def _bardeen_prim(a, Ax, Ay, Az, la, lb, lc,
                  at, xt, yt, zt, lt, mt, nt, z0):
    Ix = _gaussian_moment_1d(a, Ax, la, at, xt, lt)
    Iy = _gaussian_moment_1d(a, Ay, lb, at, yt, mt)
    dz_ao = z0 - Az
    dz_tip = z0 - zt
    g_ao = _poly_gauss(dz_ao, lc, a)
    dg_ao = _dpoly_gauss(dz_ao, lc, a)
    g_tip = _poly_gauss(dz_tip, nt, at)
    dg_tip = _dpoly_gauss(dz_tip, nt, at)
    Tz = g_ao * dg_tip - g_tip * dg_ao
    return -0.5 * Ix * Iy * Tz


def compute_bardeen_ao_tip(mol, tip_alpha, tip_center, tip_lmn, z0):
    """返回 THMAO[n_basis, n_comp]"""
    L_tip = int(sum(tip_lmn))
    tip_list = [tuple(int(x) for x in tip_lmn)] if L_tip > 0 else [(0,0,0)]
    n_comp = len(tip_list)
    xt, yt, zt = tip_center

    cart_offsets, sph_offsets = [], []
    n_cart_total, n_sph_total = 0, 0
    for sh in mol.shells:
        cart_offsets.append(n_cart_total)
        sph_offsets.append(n_sph_total)
        L = abs(sh.shell_type)
        if sh.shell_type == -1:
            n_cart_total += 4; n_sph_total += 4
        elif sh.shell_type >= 0:
            n = (L+1)*(L+2)//2
            n_cart_total += n; n_sph_total += n
        else:
            n_cart_total += (L+1)*(L+2)//2
            n_sph_total += 2*L+1

    T_cart = np.zeros((n_cart_total, n_comp))
    for ish, sh in enumerate(mol.shells):
        off = cart_offsets[ish]
        bf_info = _get_shell_bf_info(sh)
        Ax, Ay, Az = sh.center
        for i_ao, (lmn_ao, prims) in enumerate(bf_info):
            la, lb, lc = lmn_ao
            for i_tip, (tl, tm, tn) in enumerate(tip_list):
                val = 0.0
                for alpha_ao, c_ao in prims:
                    N = _norm_cart_gto(alpha_ao, la, lb, lc)
                    val += c_ao * N * _bardeen_prim(
                        alpha_ao, Ax, Ay, Az, la, lb, lc,
                        tip_alpha, xt, yt, zt, tl, tm, tn, z0)
                T_cart[off + i_ao, i_tip] = val

    has_pure = any(sh.shell_type < -1 for sh in mol.shells)
    if not has_pure:
        return T_cart

    Tr = np.zeros((n_sph_total, n_cart_total))
    for ish, sh in enumerate(mol.shells):
        co, so = cart_offsets[ish], sph_offsets[ish]
        L = abs(sh.shell_type)
        if sh.shell_type >= 0 or sh.shell_type == -1:
            n = _n_bf_in_shell(sh.shell_type)
            Tr[so:so+n, co:co+n] = np.eye(n)
        else:
            Tr[so:so+2*L+1, co:co+(L+1)*(L+2)//2] = _c2s_matrix(L)
    return Tr @ T_cart


# =====================================================================
# Dyson 轨道 (对照 Fortran g_post_stm_bardeen_mod.f)
# =====================================================================
def compute_dyson(initial, final, S_ao):
    """
    MO 重叠矩阵约定:
      OVIMOA[p,q] = sum_{mu,nu} MOFA[p,mu] * S[mu,nu] * MOAlpha[q,nu]
                  = final.mo_alpha @ S_ao @ initial.mo_alpha.T
    与 Fortran GenGAOtoMO_ds 一致。

    返回: (MODyson, NDyson, is_IP, is_EA, active_alpha, offset)
    """
    n_mo = initial.n_mo

    # MO overlap: (n_mo x n_mo), OVIMOA[p,q] = <final_MO_p | initial_MO_q>
    OVIMOA = final.mo_alpha @ S_ao.T @ initial.mo_alpha.T

    Cf_b = final.mo_beta if final.mo_beta is not None else final.mo_alpha
    Ci_b = initial.mo_beta if initial.mo_beta is not None else initial.mo_alpha
    OVIMOB = Cf_b @ S_ao.T @ Ci_b.T

    Nai, Nbi = initial.n_alpha, initial.n_beta
    Naf, Nbf = final.n_alpha, final.n_beta
    is_IP = (Nai + Nbi - 1 == Naf + Nbf)
    is_EA = (Nai + Nbi + 1 == Naf + Nbf)

    if not is_IP and not is_EA:
        raise ValueError(f"Cannot determine IP/EA: init={Nai+Nbi}, final={Naf+Nbf}")

    # ------------------------------------------------------------------
    # IP beta: NIMOA==NFMOA, NIMOB==NFMOB+1
    # ------------------------------------------------------------------
    if is_IP and Nai == Naf and Nbi == Nbf + 1:
        NDyson = Nbi
        sign0, logdet0 = slogdet(OVIMOA[:Naf, :Naf])
        # Fortran: DetMOB = OVIMOB(1:NFMOB, 2:NFMOB+1)
        M = OVIMOB[:Nbf, 1:Nbf+1].copy()
        logD = np.zeros(NDyson)
        signs = np.zeros(NDyson)
        sign_m, logdet_m = slogdet(M)
        logD[0] = logdet0 + logdet_m
        signs[0] = sign0 * sign_m
        # Fortran: Do I=2,NFMOB+1; DetMOB(:,I-1)=OVIMOB(:,I-1)
        for I in range(2, Nbf + 2):  # I = 2..Nbf+1
            M[:, I-2] = OVIMOB[:Nbf, I-2]
            sign_m, logdet_m = slogdet(M)
            logD[I-1] = logdet0 + logdet_m
            signs[I-1] = sign0 * sign_m
        logD_max = np.max(logD[np.isfinite(logD)])
        D = signs * np.exp(logD - logD_max)
        return D.reshape(-1, 1), NDyson, True, False, False, 0

    # ------------------------------------------------------------------
    # IP alpha: NIMOA==NFMOA+1, NIMOB==NFMOB
    # ------------------------------------------------------------------
    elif is_IP and Nai == Naf + 1 and Nbi == Nbf:
        NDyson = Nai
        # Fortran: DetA0 = det(OVIMOB(1:NFMOB,1:NFMOB))  [spectator beta]
        sign0, logdet0 = slogdet(OVIMOB[:Nbf, :Nbf])
        # Fortran: DetMOA = OVIMOA(1:NFMOA, 2:NFMOA+1)
        M = OVIMOA[:Naf, 1:Naf+1].copy()
        logD = np.zeros(NDyson)
        signs = np.zeros(NDyson)
        sign_m, logdet_m = slogdet(M)
        logD[0] = logdet0 + logdet_m
        signs[0] = sign0 * sign_m
        for I in range(2, Naf + 2):
            M[:, I-2] = OVIMOA[:Naf, I-2]
            sign_m, logdet_m = slogdet(M)
            logD[I-1] = logdet0 + logdet_m
            signs[I-1] = sign0 * sign_m
        logD_max = np.max(logD[np.isfinite(logD)])
        D = signs * np.exp(logD - logD_max)
        return D.reshape(-1, 1), NDyson, True, False, True, 0

    # ------------------------------------------------------------------
    # EA alpha: NIMOA+1==NFMOA, NIMOB==NFMOB
    # ------------------------------------------------------------------
    elif is_EA and Nai + 1 == Naf and Nbi == Nbf:
        NDyson = n_mo - Nai
        # Fortran: DetA0 = det(OVIMOB(1:NFMOB,1:NFMOB))  [spectator beta]
        sign0, logdet0 = slogdet(OVIMOB[:Nbf, :Nbf])
        # Fortran: DetMOA = OVIMOA(1:NFMOA, 1:NFMOA)
        M = OVIMOA[:Naf, :Naf].copy()
        logD = np.zeros(NDyson)
        signs = np.zeros(NDyson)
        sign_m, logdet_m = slogdet(M)
        logD[0] = logdet0 + logdet_m
        signs[0] = sign0 * sign_m
        # Fortran: Do I=2,NDyson; DetMOA(:,NFMOA) = OVIMOA(:, NFMOA+I-1)
        for I in range(2, NDyson + 1):
            M[:, Naf-1] = OVIMOA[:Naf, Naf + I - 2]
            sign_m, logdet_m = slogdet(M)
            logD[I-1] = logdet0 + logdet_m
            signs[I-1] = sign0 * sign_m
        logD_max = np.max(logD[np.isfinite(logD)])
        D = signs * np.exp(logD - logD_max)
        return D.reshape(-1, 1), NDyson, False, True, True, Nai

    # ------------------------------------------------------------------
    # EA beta: NIMOA==NFMOA, NIMOB+1==NFMOB
    # ------------------------------------------------------------------
    elif is_EA and Nai == Naf and Nbi + 1 == Nbf:
        NDyson = n_mo - Nbi
        # Fortran: DetA0 = det(OVIMOA(1:NFMOA,1:NFMOA))  [spectator alpha]
        sign0, logdet0 = slogdet(OVIMOA[:Naf, :Naf])
        # Fortran: DetMOB = OVIMOB(1:NFMOB, 1:NFMOB)
        M = OVIMOB[:Nbf, :Nbf].copy()
        logD = np.zeros(NDyson)
        signs = np.zeros(NDyson)
        sign_m, logdet_m = slogdet(M)
        logD[0] = logdet0 + logdet_m
        signs[0] = sign0 * sign_m
        for I in range(2, NDyson + 1):
            M[:, Nbf-1] = OVIMOB[:Nbf, Nbf + I - 2]
            sign_m, logdet_m = slogdet(M)
            logD[I-1] = logdet0 + logdet_m
            signs[I-1] = sign0 * sign_m
        logD_max = np.max(logD[np.isfinite(logD)])
        D = signs * np.exp(logD - logD_max)
        return D.reshape(-1, 1), NDyson, False, True, False, Nbi

    else:
        raise ValueError("Cannot determine Dyson branch")


# =====================================================================
# 并行扫描 worker
# =====================================================================
def _mo_overlap_matrices(initial, final, S_ao):
    OVIMOA = final.mo_alpha @ S_ao.T @ initial.mo_alpha.T
    Cf_b = final.mo_beta if final.mo_beta is not None else final.mo_alpha
    Ci_b = initial.mo_beta if initial.mo_beta is not None else initial.mo_alpha
    OVIMOB = Cf_b @ S_ao.T @ Ci_b.T
    return OVIMOA, OVIMOB


def _reference_occupations(mol):
    return list(range(mol.n_alpha)), list(range(mol.n_beta))


def _replace_occupied(occupied, removed, added):
    occ_set = set(occupied)
    if removed not in occ_set:
        raise ValueError(f"Orbital {removed + 1} is not occupied in the reference determinant")
    if added in occ_set:
        raise ValueError(f"Orbital {added + 1} is already occupied in the reference determinant")
    occ_set.remove(removed)
    occ_set.add(added)
    return sorted(occ_set)


def _excited_configuration_occupations(final, component):
    occ_alpha, occ_beta = _reference_occupations(final)
    occupied = component.occupied - 1
    virtual = component.virtual - 1
    if component.spin.upper() == 'A':
        return _replace_occupied(occ_alpha, occupied, virtual), occ_beta
    if component.spin.upper() == 'B':
        return occ_alpha, _replace_occupied(occ_beta, occupied, virtual)
    raise ValueError(f"Unsupported spin label '{component.spin}'")


def _slogdet_safe(matrix):
    sign, logdet = slogdet(matrix)
    if sign == 0 or not np.isfinite(logdet):
        return 0.0, -np.inf
    return float(sign), float(logdet)


def _normalized_component_vector(indices, logs, signs, n_mo):
    vec = np.zeros(n_mo)
    finite = np.isfinite(logs)
    if not np.any(finite):
        return vec, -np.inf
    logmax = float(np.max(logs[finite]))
    for idx, value_log, value_sign in zip(indices, logs, signs):
        if np.isfinite(value_log):
            vec[idx] = value_sign * np.exp(value_log - logmax)
    return vec, logmax


def compute_dyson_configuration(initial, final, OVIMOA, OVIMOB, final_occ_alpha, final_occ_beta):
    initial_occ_alpha, initial_occ_beta = _reference_occupations(initial)
    final_occ_alpha = list(final_occ_alpha)
    final_occ_beta = list(final_occ_beta)

    Nai, Nbi = len(initial_occ_alpha), len(initial_occ_beta)
    Naf, Nbf = len(final_occ_alpha), len(final_occ_beta)

    alpha_vec = np.zeros(initial.n_mo)
    beta_vec = np.zeros(initial.n_mo)

    if Nai + Nbi + 1 == Naf + Nbf:
        if Naf == Nai + 1 and Nbf == Nbi:
            sign0, logdet0 = _slogdet_safe(OVIMOB[np.ix_(final_occ_beta, initial_occ_beta)])
            candidates = [idx for idx in range(initial.n_mo) if idx not in initial_occ_alpha]
            logs = np.full(len(candidates), -np.inf)
            signs = np.zeros(len(candidates))
            if sign0 != 0.0:
                for i, candidate in enumerate(candidates):
                    cols = initial_occ_alpha + [candidate]
                    sign_m, logdet_m = _slogdet_safe(OVIMOA[np.ix_(final_occ_alpha, cols)])
                    if sign_m != 0.0:
                        logs[i] = logdet0 + logdet_m
                        signs[i] = sign0 * sign_m
            alpha_vec, logmax = _normalized_component_vector(candidates, logs, signs, initial.n_mo)
            return alpha_vec, beta_vec, 'alpha-EA', logmax

        if Naf == Nai and Nbf == Nbi + 1:
            sign0, logdet0 = _slogdet_safe(OVIMOA[np.ix_(final_occ_alpha, initial_occ_alpha)])
            candidates = [idx for idx in range(initial.n_mo) if idx not in initial_occ_beta]
            logs = np.full(len(candidates), -np.inf)
            signs = np.zeros(len(candidates))
            if sign0 != 0.0:
                for i, candidate in enumerate(candidates):
                    cols = initial_occ_beta + [candidate]
                    sign_m, logdet_m = _slogdet_safe(OVIMOB[np.ix_(final_occ_beta, cols)])
                    if sign_m != 0.0:
                        logs[i] = logdet0 + logdet_m
                        signs[i] = sign0 * sign_m
            beta_vec, logmax = _normalized_component_vector(candidates, logs, signs, initial.n_mo)
            return alpha_vec, beta_vec, 'beta-EA', logmax

    if Nai + Nbi - 1 == Naf + Nbf:
        if Naf == Nai - 1 and Nbf == Nbi:
            sign0, logdet0 = _slogdet_safe(OVIMOB[np.ix_(final_occ_beta, initial_occ_beta)])
            candidates = list(initial_occ_alpha)
            logs = np.full(len(candidates), -np.inf)
            signs = np.zeros(len(candidates))
            if sign0 != 0.0:
                for i, candidate in enumerate(candidates):
                    cols = [idx for idx in initial_occ_alpha if idx != candidate]
                    sign_m, logdet_m = _slogdet_safe(OVIMOA[np.ix_(final_occ_alpha, cols)])
                    if sign_m != 0.0:
                        logs[i] = logdet0 + logdet_m
                        signs[i] = sign0 * sign_m
            alpha_vec, logmax = _normalized_component_vector(candidates, logs, signs, initial.n_mo)
            return alpha_vec, beta_vec, 'alpha-IP', logmax

        if Naf == Nai and Nbf == Nbi - 1:
            sign0, logdet0 = _slogdet_safe(OVIMOA[np.ix_(final_occ_alpha, initial_occ_alpha)])
            candidates = list(initial_occ_beta)
            logs = np.full(len(candidates), -np.inf)
            signs = np.zeros(len(candidates))
            if sign0 != 0.0:
                for i, candidate in enumerate(candidates):
                    cols = [idx for idx in initial_occ_beta if idx != candidate]
                    sign_m, logdet_m = _slogdet_safe(OVIMOB[np.ix_(final_occ_beta, cols)])
                    if sign_m != 0.0:
                        logs[i] = logdet0 + logdet_m
                        signs[i] = sign0 * sign_m
            beta_vec, logmax = _normalized_component_vector(candidates, logs, signs, initial.n_mo)
            return alpha_vec, beta_vec, 'beta-IP', logmax

    raise ValueError(
        f"Cannot determine determinant Dyson branch: init=({Nai},{Nbi}) final=({Naf},{Nbf})"
    )


_worker_data = {}

def _worker_init(
    mol,
    cfg,
    C_active,
    MODyson,
    NDyson,
    offset,
    n_states,
    n_comp,
    tddft_mode=False,
    state_coeff_alpha=None,
    state_coeff_beta=None,
):
    _worker_data['mol'] = mol
    _worker_data['cfg'] = cfg
    _worker_data['C_active'] = C_active
    _worker_data['MODyson'] = MODyson
    _worker_data['NDyson'] = NDyson
    _worker_data['offset'] = offset
    _worker_data['n_states'] = n_states
    _worker_data['n_comp'] = n_comp
    _worker_data['tddft_mode'] = tddft_mode
    _worker_data['state_coeff_alpha'] = state_coeff_alpha
    _worker_data['state_coeff_beta'] = state_coeff_beta


def _worker_compute_row(j):
    mol = _worker_data['mol']
    cfg = _worker_data['cfg']
    C_active = _worker_data['C_active']
    MODyson = _worker_data['MODyson']
    NDyson = _worker_data['NDyson']
    offset = _worker_data['offset']
    n_states = _worker_data['n_states']
    n_comp = _worker_data['n_comp']
    tddft_mode = _worker_data['tddft_mode']
    state_coeff_alpha = _worker_data['state_coeff_alpha']
    state_coeff_beta = _worker_data['state_coeff_beta']
    Nx = cfg.NScan[0]

    row_X = np.zeros(Nx + 1)
    row_Y = np.zeros(Nx + 1)
    row_THM = np.zeros((Nx + 1, n_states, n_comp))

    for i in range(Nx + 1):
        tip = cfg.TCenter + i * cfg.VScan[:, 0] + j * cfg.VScan[:, 1]
        row_X[i] = tip[0] * AUAA
        row_Y[i] = tip[1] * AUAA

        THMAO = compute_bardeen_ao_tip(mol, cfg.TAlpha, tip, cfg.TLMN, cfg.RZ0)

        if tddft_mode:
            THMMO_a = mol.mo_alpha @ THMAO
            Cb = mol.mo_beta if mol.mo_beta is not None else mol.mo_alpha
            THMMO_b = Cb @ THMAO
            row_THM[i, :, :] = state_coeff_alpha.T @ THMMO_a + state_coeff_beta.T @ THMMO_b
        elif cfg.Dyson:
            THMMO = C_active @ THMAO
            for k1 in range(n_comp):
                for i1 in range(NDyson):
                    row_THM[i, :, k1] += MODyson[i1, :] * THMMO[i1 + offset, k1]
        else:
            if cfg.NMOA > 0:
                THMMO = mol.mo_alpha @ THMAO
                for i1 in range(cfg.NMOA):
                    row_THM[i, i1, :] = THMMO[cfg.MOAIndex[i1] - 1, :]
            elif cfg.NMOB > 0:
                Cb = mol.mo_beta if mol.mo_beta is not None else mol.mo_alpha
                THMMO = Cb @ THMAO
                for i1 in range(cfg.NMOB):
                    row_THM[i, i1, :] = THMMO[cfg.MOBIndex[i1] - 1, :]

    return j, row_X, row_Y, row_THM


def _project_thmao_batch(
    thmao_batch,
    coeff_matrix,
    cfg,
    modyson,
    n_dyson,
    offset,
    selected_indices,
    xp,
    tddft_mode=False,
    coeff_matrix_beta=None,
    state_coeff_alpha=None,
    state_coeff_beta=None,
):
    thmmo_batch = xp.einsum("mb,pbc->pmc", coeff_matrix, thmao_batch)
    if tddft_mode:
        thmmo_beta_batch = xp.einsum("mb,pbc->pmc", coeff_matrix_beta, thmao_batch)
        return (
            xp.einsum("ms,pmc->psc", state_coeff_alpha, thmmo_batch)
            + xp.einsum("ms,pmc->psc", state_coeff_beta, thmmo_beta_batch)
        )
    if cfg.Dyson:
        active = thmmo_batch[:, offset:offset + n_dyson, :]
        return xp.einsum("is,pic->psc", modyson, active)
    return thmmo_batch[:, selected_indices, :]


def _run_scan_batched(
    initial,
    cfg,
    C_active,
    MODyson,
    NDyson,
    offset,
    n_states,
    n_comp,
    Nx,
    Ny,
    msg,
    tddft_mode=False,
    state_coeff_alpha=None,
    state_coeff_beta=None,
):
    backend_name, xp = resolve_scan_backend(cfg.backend)
    row_batch = min(max(1, cfg.row_batch), Ny + 1)

    x_grid = np.zeros((Nx + 1, Ny + 1))
    y_grid = np.zeros((Nx + 1, Ny + 1))
    thm_grid = np.zeros((Nx + 1, Ny + 1, n_states, n_comp))

    plan = build_bardeen_batch_plan(initial, cfg.TLMN)
    total = (Nx + 1) * (Ny + 1)
    msg(f"* 2D scan: {Nx+1}x{Ny+1} = {total} points, backend={backend_name}, row_batch={row_batch}\n")

    if tddft_mode:
        coeff_matrix = xp.asarray(initial.mo_alpha)
        beta_orbitals = initial.mo_beta if initial.mo_beta is not None else initial.mo_alpha
        coeff_matrix_beta = xp.asarray(beta_orbitals)
        state_coeff_alpha = xp.asarray(state_coeff_alpha)
        state_coeff_beta = xp.asarray(state_coeff_beta)
        modyson_matrix = None
        selected_indices = None
    elif cfg.Dyson:
        coeff_matrix = xp.asarray(C_active)
        coeff_matrix_beta = None
        modyson_matrix = xp.asarray(MODyson)
        selected_indices = None
        state_coeff_alpha = None
        state_coeff_beta = None
    else:
        coeff_matrix_beta = None
        state_coeff_alpha = None
        state_coeff_beta = None
        if cfg.NMOA > 0:
            coeff_matrix = xp.asarray(initial.mo_alpha)
            selected_indices = xp.asarray(np.asarray(cfg.MOAIndex, dtype=int) - 1)
        else:
            beta_orbitals = initial.mo_beta if initial.mo_beta is not None else initial.mo_alpha
            coeff_matrix = xp.asarray(beta_orbitals)
            selected_indices = xp.asarray(np.asarray(cfg.MOBIndex, dtype=int) - 1)
        modyson_matrix = None

    import time
    t_start = time.time()
    completed = 0

    for row_start in range(0, Ny + 1, row_batch):
        row_stop = min(row_start + row_batch, Ny + 1)
        tip_centers = []
        for j in range(row_start, row_stop):
            for i in range(Nx + 1):
                tip = cfg.TCenter + i * cfg.VScan[:, 0] + j * cfg.VScan[:, 1]
                x_grid[i, j] = tip[0] * AUAA
                y_grid[i, j] = tip[1] * AUAA
                tip_centers.append(tip)

        thmao_batch = compute_bardeen_ao_tip_batch(
            plan,
            cfg.TAlpha,
            np.asarray(tip_centers, dtype=float),
            cfg.RZ0,
            xp=xp,
        )
        thm_batch = _project_thmao_batch(
            thmao_batch,
            coeff_matrix,
            cfg,
            modyson_matrix,
            NDyson,
            offset,
            selected_indices,
            xp,
            tddft_mode=tddft_mode,
            coeff_matrix_beta=coeff_matrix_beta,
            state_coeff_alpha=state_coeff_alpha,
            state_coeff_beta=state_coeff_beta,
        )
        thm_batch = to_numpy(thm_batch)

        point_offset = 0
        for j in range(row_start, row_stop):
            next_offset = point_offset + Nx + 1
            thm_grid[:, j, :, :] = thm_batch[point_offset:next_offset, :, :]
            point_offset = next_offset

        completed += row_stop - row_start
        elapsed = time.time() - t_start
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = (Ny + 1 - completed) / rate if rate > 0 else 0.0
        msg(f"  row {completed}/{Ny+1} | {elapsed:.1f}s elapsed | {rate:.1f} rows/s | ETA {eta:.0f}s\n")

    t_scan = time.time() - t_start
    return x_grid, y_grid, thm_grid, t_scan


def print_matlab_matrix(name, M, nr, nc):
    print(f"{name}=[...")
    for i in range(nr):
        row = " ".join(f"{M[i,j]:.10E}" for j in range(nc))
        end = ";..." if i < nr-1 else "];"
        print(f"{row}{end}")


def run_stm(cfg):
    import time
    t_total_start = time.time()

    msg = sys.stderr.write

    msg(f"* Algorithm for STM is Bardeen "
        f"{'with Gaussian tip' if cfg.Gauss else ''}\n")

    initial = read_fchk(cfg.MFchk)
    msg(f"* Initial: {initial.method} Na={initial.n_alpha} Nb={initial.n_beta} "
        f"Nbasis={initial.n_basis} Nmo={initial.n_mo}\n")

    MODyson = None
    NDyson = 0
    offset = 0
    is_IP = False
    is_EA = False
    active_alpha = True
    tddft_mode = cfg.TDA and bool(cfg.TDLog)
    state_coeff_alpha = None
    state_coeff_beta = None
    state_labels = []
    state_comments = []

    if tddft_mode:
        if not cfg.MFFchk:
            raise ValueError('TDDFT determinant mode requires STM.MolFinalfchk')

        final = read_fchk(cfg.MFFchk)
        msg(f"* Final:   {final.method} Na={final.n_alpha} Nb={final.n_beta}\n")
        msg("* Computing AO overlap...\n")
        S_ao = compute_ao_overlap(initial, final)
        diag = np.diag(S_ao)
        msg(f"* AO overlap done. diag range=[{diag.min():.6f}, {diag.max():.6f}]\n")

        msg(f"* TD log: {cfg.TDLog}\n")
        all_states = read_tddft_log(cfg.TDLog)
        if cfg.TDState is not None:
            td_states = [state for state in all_states if state.index == cfg.TDState]
            if not td_states:
                raise ValueError(f"Requested STM.TDState={cfg.TDState} was not found in {cfg.TDLog}")
        else:
            td_states = [
                state for state in all_states
                if state.index >= max(1, cfg.TDStateMin)
                and (cfg.TDStateMax <= 0 or state.index <= cfg.TDStateMax)
            ]
            if not td_states:
                raise ValueError('No TDDFT states remain after STM.TDStateMin/Max filtering')

        det_states = build_state_determinants(
            final,
            td_states,
            coeff_threshold=cfg.CITHLD,
            include_deexcitations=cfg.TDUseY,
        )

        OVIMOA, OVIMOB = _mo_overlap_matrices(initial, final, S_ao)
        state_coeff_alpha = np.zeros((initial.n_mo, len(det_states)))
        state_coeff_beta = np.zeros((initial.n_mo, len(det_states)))

        if cfg.TDChannel != 'both':
            msg(f"! STM.TDChannel={cfg.TDChannel} is ignored in determinant TDDFT Dyson mode.\n")

        for state_idx, det_state in enumerate(det_states):
            state = det_state.state
            component_count = len(det_state.components)
            coeff_norm = math.sqrt(sum(comp.coefficient * comp.coefficient for comp in det_state.components))
            max_coeff = max((abs(comp.coefficient) for comp in det_state.components), default=0.0)
            msg(
                f"* TD state {state.index:3d}: ncfg={component_count} coeff_norm={coeff_norm:.4f} "
                f"max|c|={max_coeff:.4f} E={state.energy_ev:.4f} eV f={state.oscillator_strength:.4f}\n"
            )

            term_vectors = []
            state_logmax = -np.inf
            branch_name = None

            for component in det_state.components:
                final_occ_alpha, final_occ_beta = _excited_configuration_occupations(final, component)
                comp_alpha, comp_beta, branch_name_local, comp_logmax = compute_dyson_configuration(
                    initial,
                    final,
                    OVIMOA,
                    OVIMOB,
                    final_occ_alpha,
                    final_occ_beta,
                )
                branch_name = branch_name or branch_name_local
                term_vectors.append((component, comp_alpha, comp_beta, comp_logmax))
                if np.isfinite(comp_logmax):
                    state_logmax = max(state_logmax, comp_logmax)
                msg(
                    f"    {component.occupied:3d}{component.spin} -> {component.virtual:3d}{component.spin} "
                    f"{component.coefficient: .5f}\n"
                )

            if not det_state.components:
                msg(
                    f"! TD state {state.index:3d} has no determinant components above CITHLD={cfg.CITHLD:.2e}\n"
                )
            elif not np.isfinite(state_logmax):
                msg(
                    f"! TD state {state.index:3d} determinant Dyson coefficients vanished numerically\n"
                )
            else:
                for component, comp_alpha, comp_beta, comp_logmax in term_vectors:
                    if not np.isfinite(comp_logmax):
                        continue
                    scale = component.coefficient * math.exp(comp_logmax - state_logmax)
                    state_coeff_alpha[:, state_idx] += scale * comp_alpha
                    state_coeff_beta[:, state_idx] += scale * comp_beta

            state_labels.append(f"Ex{state.index:03d}")
            state_comments.append(
                f"TDDFT excited state No.:{state.index:3d} E={state.energy_ev:.4f} eV "
                f"f={state.oscillator_strength:.4f} ncfg={component_count} branch={branch_name or 'n/a'}"
            )

    elif cfg.Dyson:
        final = read_fchk(cfg.MFFchk)
        msg(f"* Final:   {final.method} Na={final.n_alpha} Nb={final.n_beta}\n")

        msg("* Computing AO overlap...\n")
        S_ao = compute_ao_overlap(initial, final)
        diag = np.diag(S_ao)
        msg(f"* AO overlap done. diag range=[{diag.min():.6f}, {diag.max():.6f}]\n")

        msg("* Computing Dyson coefficients...\n")
        MODyson, NDyson, is_IP, is_EA, active_alpha, offset = \
            compute_dyson(initial, final, S_ao)

        branch = ("beta-IP" if is_IP and not active_alpha else
                  "alpha-IP" if is_IP and active_alpha else
                  "alpha-EA" if is_EA and active_alpha else "beta-EA")
        msg(f"* Branch: {branch}, NDyson={NDyson}, offset={offset}\n")

        if cfg.PrintDyson:
            print(f"{'*Ex No.':>7s}{'   MO No.':>9s}{'           Dyson':>16s}")
            for p in range(NDyson):
                mo_idx = p + 1 if is_IP else p + offset + 1
                print(f"{0:7d}{mo_idx:9d}{MODyson[p,0]:16.8f}")

    MaxMZ = max(a['center'][2] for a in initial.atoms)
    if MaxMZ > cfg.TCenter[2] - 0.01:
        msg(f"!*** Error. Max mol Z={MaxMZ*AUAA:.4f} >= tip Z={cfg.TCenter[2]*AUAA:.4f}\n")
        sys.exit(1)

    if cfg.RZ0 == 1000.0:
        cfg.RZ0 = (MaxMZ + cfg.TCenter[2]) / 2.0
    msg(f"* Max mol Z={MaxMZ*AUAA:.4f} A, z0={cfg.RZ0*AUAA:.4f} A\n")

    L_tip = int(sum(cfg.TLMN))
    tip_list = [tuple(int(x) for x in cfg.TLMN)] if L_tip > 0 else [(0,0,0)]
    n_comp = len(tip_list)
    Nx, Ny, Nz = cfg.NScan

    n_states = MODyson.shape[1] if MODyson is not None else 0
    C_active = None

    if tddft_mode:
        n_states = len(state_labels)
    elif cfg.Dyson:
        if initial.method != 'U':
            C_active = initial.mo_alpha
        elif active_alpha:
            C_active = initial.mo_alpha
        else:
            C_active = initial.mo_beta if initial.mo_beta is not None else initial.mo_alpha
    else:
        if cfg.LMOA and cfg.NMOA == 0:
            cfg.NMOA = 2
            cfg.MOAIndex = [initial.n_alpha, initial.n_alpha + 1]
        if cfg.LMOB and cfg.NMOB == 0:
            cfg.NMOB = 2
            cfg.MOBIndex = [initial.n_beta, initial.n_beta + 1]
        n_states = cfg.NMOA if cfg.NMOA > 0 else cfg.NMOB

    if abs(cfg.LScan) == 2:
        X = np.zeros((Nx+1, Ny+1))
        Y = np.zeros((Nx+1, Ny+1))
        THM = np.zeros((Nx+1, Ny+1, n_states, n_comp))

        total = (Nx+1) * (Ny+1)
        backend_name = (cfg.backend or "legacy").lower()
        n_workers = min(cfg.n_cores, cpu_count())

        if backend_name == "legacy":
            msg(f"* 2D scan: {Nx+1}x{Ny+1} = {total} points, {n_workers} cores\n")

            t_start = time.time()
            completed = 0

            if n_workers > 1:
                with Pool(processes=n_workers,
                           initializer=_worker_init,
                           initargs=(initial, cfg, C_active, MODyson,
                                     NDyson, offset, n_states, n_comp,
                                     tddft_mode, state_coeff_alpha, state_coeff_beta)) as pool:
                    for result in pool.imap_unordered(_worker_compute_row, range(Ny + 1)):
                        j, row_X, row_Y, row_THM = result
                        X[:, j] = row_X
                        Y[:, j] = row_Y
                        THM[:, j, :, :] = row_THM
                        completed += 1
                        if completed % 20 == 0 or completed == Ny + 1:
                            elapsed = time.time() - t_start
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (Ny + 1 - completed) / rate if rate > 0 else 0
                            msg(f"  row {completed}/{Ny+1} | {elapsed:.1f}s elapsed | "
                                f"{rate:.1f} rows/s | ETA {eta:.0f}s\n")
            else:
                _worker_init(initial, cfg, C_active, MODyson, NDyson, offset, n_states, n_comp,
                             tddft_mode, state_coeff_alpha, state_coeff_beta)
                for j in range(Ny + 1):
                    _, row_X, row_Y, row_THM = _worker_compute_row(j)
                    X[:, j] = row_X
                    Y[:, j] = row_Y
                    THM[:, j, :, :] = row_THM
                    completed += 1
                    if completed % 10 == 0 or completed == Ny + 1:
                        elapsed = time.time() - t_start
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (Ny + 1 - completed) / rate if rate > 0 else 0
                        msg(f"  row {completed}/{Ny+1} | {elapsed:.1f}s | "
                            f"{rate:.2f} rows/s | ETA {eta:.0f}s\n")

            t_scan = time.time() - t_start
        else:
            X, Y, THM, t_scan = _run_scan_batched(
                initial,
                cfg,
                C_active,
                MODyson,
                NDyson,
                offset,
                n_states,
                n_comp,
                Nx,
                Ny,
                msg,
                tddft_mode=tddft_mode,
                state_coeff_alpha=state_coeff_alpha,
                state_coeff_beta=state_coeff_beta,
            )
            n_workers = 1

        msg(f"* Scan completed in {t_scan:.1f}s ({total/t_scan:.1f} points/s)\n")

        msg(f"* Alpha of tip wavefunction: {cfg.TAlpha:.6f}\n")
        print_matlab_matrix("X", X, Nx+1, Ny+1)
        print_matlab_matrix("Y", Y, Nx+1, Ny+1)

        for k1 in range(n_comp):
            lmn = tip_list[k1]
            if tddft_mode:
                for s, label in enumerate(state_labels):
                    data = THM[:, :, s, k1]
                    if cfg.LSmear:
                        try:
                            from scipy.ndimage import gaussian_filter
                            data = gaussian_filter(data**2 * cfg.RScale,
                                                   sigma=max(1, cfg.NSmear))
                        except ImportError:
                            pass
                    print(f"%* TLMN:{lmn[0]:2d}{lmn[1]:2d}{lmn[2]:2d} {state_comments[s]}")
                    print_matlab_matrix(f"{label}LMN{k1+1}", data, Nx+1, Ny+1)
            elif cfg.Dyson:
                for s in range(n_states):
                    data = THM[:, :, s, k1]
                    if cfg.LSmear:
                        try:
                            from scipy.ndimage import gaussian_filter
                            data = gaussian_filter(data**2 * cfg.RScale,
                                                   sigma=max(1, cfg.NSmear))
                        except ImportError:
                            pass
                    print(f"%* TLMN:{lmn[0]:2d}{lmn[1]:2d}{lmn[2]:2d}"
                          f" Excited state No.:{s:3d}")
                    print_matlab_matrix(f"Ex{s:03d}LMN{k1+1}", data, Nx+1, Ny+1)
            else:
                indices = cfg.MOAIndex if cfg.NMOA > 0 else cfg.MOBIndex
                prefix = "MOA" if cfg.NMOA > 0 else "MOB"
                for si, mo_i in enumerate(indices):
                    data = THM[:, :, si, k1]
                    print(f"%* TLMN:{lmn[0]:2d}{lmn[1]:2d}{lmn[2]:2d}"
                          f" Molecular orbital No.:{mo_i:3d}")
                    print_matlab_matrix(f"{prefix}{mo_i:03d}LMN{k1+1}",
                                        data, Nx+1, Ny+1)

        msg("* Done.\n")
        t_total = time.time() - t_total_start
        msg(f"* Total time: {t_total:.1f}s\n")
        msg(f"* Summary: {initial.n_basis} basis, {initial.n_mo} MOs, "
            f"{(Nx+1)*(Ny+1)} points, {n_workers} host workers, backend={cfg.backend}\n")
    else:
        msg(f"* Scan dimension {cfg.LScan} not yet implemented.\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python stm_bardeen.py input.finp [n_cores] [backend]", file=sys.stderr)
        print("  n_cores: number of cores (default: 10, max: cpu_count)", file=sys.stderr)
        print("  backend: legacy | cpu | batched | gpu | auto", file=sys.stderr)
        sys.exit(1)

    n_cores = 10
    backend_override = None
    if len(sys.argv) >= 3:
        try:
            n_cores = int(sys.argv[2])
            if n_cores < 1:
                n_cores = 1
        except ValueError:
            backend_override = sys.argv[2].lower()
    if len(sys.argv) >= 4:
        backend_override = sys.argv[3].lower()

    cfg = parse_finp(sys.argv[1])
    if backend_override is not None:
        cfg.backend = backend_override
    cfg.n_cores = n_cores  # 添加到配置中
    sys.stderr.write(f"* Input: {sys.argv[1]}\n")
    sys.stderr.write(f"* Bardeen={cfg.LBardeen} Dyson={cfg.Dyson} "
                     f"TDA={cfg.TDA} Gauss={cfg.Gauss}\n")
    sys.stderr.write(f"* Backend={cfg.backend} row_batch={cfg.row_batch}\n")
    sys.stderr.write(f"* Molfchk={cfg.MFchk}\n")
    if cfg.MFFchk:
        sys.stderr.write(f"* MolFinalfchk={cfg.MFFchk}\n")
    if cfg.TDLog:
        sys.stderr.write(f"* TDLog={cfg.TDLog}\n")
        sys.stderr.write(
            f"* TDChannel={cfg.TDChannel} TDUseY={cfg.TDUseY} TDState={cfg.TDState if cfg.TDState is not None else 'range'} "
            f"TDStateMin={cfg.TDStateMin} TDStateMax={cfg.TDStateMax}\n"
        )
    sys.stderr.write(f"* Tip: alpha={cfg.TAlpha} LMN=({cfg.TLMN[0]},"
                     f"{cfg.TLMN[1]},{cfg.TLMN[2]}) "
                     f"center=({cfg.TCenter[0]*AUAA:.3f},"
                     f"{cfg.TCenter[1]*AUAA:.3f},"
                     f"{cfg.TCenter[2]*AUAA:.3f}) A\n")
    sys.stderr.write(f"* Scan={cfg.LScan}D grid=({cfg.NScan[0]+1}x"
                     f"{cfg.NScan[1]+1})\n")

    if cfg.LBardeen:
        run_stm(cfg)
    else:
        sys.stderr.write("* Only STM.Bardeen=T is supported.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
