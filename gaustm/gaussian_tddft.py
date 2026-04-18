from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from .gaussian_fchk import MolData


__all__ = [
    'TDTransition',
    'TDExcitedState',
    'TDDeterminantComponent',
    'TDDeterminantState',
    'build_state_determinants',
    'read_tddft_log',
]


HEADER_RE = re.compile(
    r'^\s*Excited State\s+(\d+):\s+(\S+)\s+'
    r'([+-]?\d+(?:\.\d+)?)\s+eV\s+([+-]?\d+(?:\.\d+)?)\s+nm\s+'
    r'f=\s*([+-]?\d+(?:\.\d+)?)\s+<S\*\*2>=\s*([+-]?\d+(?:\.\d+)?)'
)
TRANSITION_RE = re.compile(
    r'^\s*(\d+)([AB]?)\s*(->|<-)\s*(\d+)([AB]?)\s*'
    r'([+-]?\d+(?:\.\d+)?(?:[DEde][+-]?\d+)?)\s*$'
)
TOTAL_ENERGY_RE = re.compile(
    r'^\s*Total Energy,\s+E\(TD-HF/TD-DFT\)\s*=\s*'
    r'([+-]?\d+(?:\.\d+)?(?:[DEde][+-]?\d+)?)'
)


@dataclass
class TDTransition:
    occupied: int
    virtual: int
    spin: str
    amplitude: float
    is_deexcitation: bool = False


@dataclass
class TDExcitedState:
    index: int
    label: str
    energy_ev: float
    wavelength_nm: float
    oscillator_strength: float
    s2: float | None = None
    total_energy: float | None = None
    transitions: list[TDTransition] = field(default_factory=list)


@dataclass
class TDDeterminantComponent:
    occupied: int
    virtual: int
    spin: str
    coefficient: float


@dataclass
class TDDeterminantState:
    state: TDExcitedState
    components: list[TDDeterminantComponent]



def _parse_float(value: str) -> float:
    return float(value.replace('D', 'E').replace('d', 'e'))



def read_tddft_log(filepath: str) -> list[TDExcitedState]:
    states: list[TDExcitedState] = []
    current: TDExcitedState | None = None

    for raw_line in Path(filepath).read_text(errors='ignore').splitlines():
        header_match = HEADER_RE.match(raw_line)
        if header_match:
            if current is not None:
                states.append(current)
            current = TDExcitedState(
                index=int(header_match.group(1)),
                label=header_match.group(2),
                energy_ev=float(header_match.group(3)),
                wavelength_nm=float(header_match.group(4)),
                oscillator_strength=float(header_match.group(5)),
                s2=float(header_match.group(6)),
            )
            continue

        if current is None:
            continue

        trans_match = TRANSITION_RE.match(raw_line)
        if trans_match:
            left_spin = trans_match.group(2) or trans_match.group(5) or 'A'
            right_spin = trans_match.group(5) or trans_match.group(2) or left_spin
            if left_spin != right_spin:
                raise ValueError(
                    f"Mixed-spin TDDFT transition is not supported: '{raw_line.strip()}'"
                )
            current.transitions.append(
                TDTransition(
                    occupied=int(trans_match.group(1)),
                    virtual=int(trans_match.group(4)),
                    spin=left_spin.upper(),
                    amplitude=_parse_float(trans_match.group(6)),
                    is_deexcitation=(trans_match.group(3) == '<-'),
                )
            )
            continue

        energy_match = TOTAL_ENERGY_RE.match(raw_line)
        if energy_match:
            current.total_energy = _parse_float(energy_match.group(1))

    if current is not None:
        states.append(current)

    if not states:
        raise ValueError(f'No Gaussian excited-state block was found in {filepath}')
    return states



def _validate_transition(mol: MolData, transition: TDTransition) -> None:
    spin = transition.spin.upper()
    if spin == 'A':
        n_occ = mol.n_alpha
        n_mo = mol.n_mo
    elif spin == 'B':
        n_occ = mol.n_beta
        n_mo = mol.n_mo
    else:
        raise ValueError(f"Unsupported spin label '{transition.spin}'")

    if transition.occupied < 1 or transition.occupied > n_occ:
        raise ValueError(
            f"{spin} occupied orbital {transition.occupied} is outside 1..{n_occ}"
        )
    if transition.virtual <= n_occ or transition.virtual > n_mo:
        raise ValueError(
            f"{spin} virtual orbital {transition.virtual} must be in {n_occ + 1}..{n_mo}"
        )



def build_state_determinants(
    mol: MolData,
    states: list[TDExcitedState],
    coeff_threshold: float = 1e-6,
    include_deexcitations: bool = True,
) -> list[TDDeterminantState]:
    det_states: list[TDDeterminantState] = []

    for state in states:
        grouped: dict[tuple[str, int, int], float] = {}
        for transition in state.transitions:
            if abs(transition.amplitude) < coeff_threshold:
                continue
            if transition.is_deexcitation and not include_deexcitations:
                continue
            _validate_transition(mol, transition)
            key = (transition.spin.upper(), transition.occupied, transition.virtual)
            grouped[key] = grouped.get(key, 0.0) + transition.amplitude

        components = [
            TDDeterminantComponent(
                spin=spin,
                occupied=occupied,
                virtual=virtual,
                coefficient=coefficient,
            )
            for (spin, occupied, virtual), coefficient in sorted(grouped.items())
            if abs(coefficient) >= coeff_threshold
        ]
        det_states.append(TDDeterminantState(state=state, components=components))

    return det_states
