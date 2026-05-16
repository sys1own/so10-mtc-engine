"""Targeted SO(10)_312 parent-block solver without full-spectrum enumeration.

The parent sector has billions of integrable primaries, so this module never
constructs ``P_+^312(D5)``.  It evaluates only the four audited anomaly-free
blocks by combining the Kac-Peterson modular S rows from ``core.py`` with an
explicit Weyl-denominator quantum-dimension generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Final, Mapping, TypeAlias

import mpmath

from mtc_engine.core import DEFAULT_DPS, SO10_312, Weight
from mtc_engine.exact_traces import TauLike, TruncatedVZWCharacter, finite_screen_budget, truncated_vzw_character

ExactScalar: TypeAlias = int | str | Decimal | Fraction | mpmath.mpf

PARENT_SUBSPACE_DPS: Final[int] = DEFAULT_DPS
PARENT_SUBSPACE_NOISE_WALL: Final[mpmath.mpf] = mpmath.mpf("1e-199")
AUDITED_PARENT_BLOCK_ORDER: Final[tuple[str, ...]] = (
    "so10_v",
    "so10_adjoint",
    "so10_spinor_plus",
    "so10_spinor_minus",
)
PARENT_BLOCK_WEIGHTS: Final[Mapping[str, Weight]] = {
    "so10_v": (1, 0, 0, 0, 0),
    "so10_adjoint": (0, 1, 0, 0, 0),
    "so10_spinor_plus": (0, 0, 0, 1, 0),
    "so10_spinor_minus": (0, 0, 0, 0, 1),
}

mpmath.mp.dps = max(mpmath.mp.dps, PARENT_SUBSPACE_DPS)


@dataclass(frozen=True)
class ParentBlockSolution:
    """Solved data for one audited SO(10)_312 parent primary block."""

    label: str
    weight: Weight
    quantum_dimension: mpmath.mpf
    core_quantum_dimension: mpmath.mpf
    weyl_denominator_numerator: mpmath.mpf
    weyl_denominator_vacuum: mpmath.mpf
    finite_character: TruncatedVZWCharacter
    dimension_residual: mpmath.mpf
    character_dimension_residual: mpmath.mpf

    @property
    def passed(self) -> bool:
        """Return whether Weyl and character dimensions agree below the wall."""

        return (
            self.dimension_residual <= PARENT_SUBSPACE_NOISE_WALL
            and self.character_dimension_residual <= PARENT_SUBSPACE_NOISE_WALL
        )


@dataclass(frozen=True)
class ParentSubspaceReport:
    """Targeted 4-block parent subspace report."""

    capacity_bits: mpmath.mpf
    full_parent_primary_count: int
    block_order: tuple[str, ...]
    blocks: tuple[ParentBlockSolution, ...]
    modular_s_submatrix: tuple[tuple[mpmath.mpc, ...], ...]

    @property
    def passed(self) -> bool:
        """Return whether all targeted parent-block checks pass."""

        return bool(
            self.block_order == AUDITED_PARENT_BLOCK_ORDER
            and self.full_parent_primary_count > SO10_312.max_full_fusion_primaries
            and len(self.modular_s_submatrix) == len(AUDITED_PARENT_BLOCK_ORDER)
            and all(len(row) == len(AUDITED_PARENT_BLOCK_ORDER) for row in self.modular_s_submatrix)
            and all(block.passed for block in self.blocks)
        )


def audited_parent_blocks() -> tuple[tuple[str, Weight], ...]:
    """Return the four and only four audited parent blocks in stable order."""

    return tuple((label, PARENT_BLOCK_WEIGHTS[label]) for label in AUDITED_PARENT_BLOCK_ORDER)


def parent_modular_s_submatrix(*, dps: int | None = None) -> tuple[tuple[mpmath.mpc, ...], ...]:
    """Evaluate the targeted 4x4 SO(10)_312 modular S submatrix.

    This calls the Kac-Peterson row formula directly for the named weights and
    never requests the multi-billion-element parent primary spectrum.
    """

    blocks = audited_parent_blocks()
    with mpmath.workdps(_resolved_dps(dps)):
        return tuple(
            tuple(SO10_312.modular_s_entry(left_weight, right_weight, dps=dps) for _, right_weight in blocks)
            for _, left_weight in blocks
        )


def weyl_denominator_for_shifted_weight(
    shifted_weight: Weight,
    *,
    dps: int | None = None,
) -> mpmath.mpf:
    """Evaluate the simply-laced Weyl denominator product at affine level 312."""

    if len(shifted_weight) != SO10_312.rank:
        raise ValueError(f"Shifted weight {shifted_weight} does not have D5 rank {SO10_312.rank}.")
    with mpmath.workdps(_resolved_dps(dps)):
        denominator = mpmath.mpf(SO10_312.denominator)
        product = mpmath.mpf("1")
        for root in SO10_312.positive_roots:
            product *= mpmath.mpf("2") * mpmath.sin(
                mpmath.pi * mpmath.mpf(_root_pairing(shifted_weight, root)) / denominator
            )
        return product


def weyl_character_quantum_dimension(weight: Weight, *, dps: int | None = None) -> mpmath.mpf:
    """Compute a parent quantum dimension by Weyl's denominator formula."""

    if not SO10_312.is_integrable(weight):
        raise ValueError(f"Weight {weight} is not integrable for {SO10_312.name}.")
    with mpmath.workdps(_resolved_dps(dps)):
        numerator = weyl_denominator_for_shifted_weight(_add_weights(weight, SO10_312.rho), dps=dps)
        vacuum = weyl_denominator_for_shifted_weight(SO10_312.rho, dps=dps)
        if vacuum == 0:
            raise ArithmeticError("Vacuum Weyl denominator vanished unexpectedly.")
        return numerator / vacuum


def parent_quantum_dimensions(*, dps: int | None = None) -> Mapping[str, mpmath.mpf]:
    """Return Weyl-denominator quantum dimensions for the four parent blocks."""

    return {
        label: weyl_character_quantum_dimension(weight, dps=dps)
        for label, weight in audited_parent_blocks()
    }


def parent_character_generators(
    *,
    tau: TauLike | None = None,
    capacity_bits: ExactScalar | None = None,
    dps: int | None = None,
) -> Mapping[str, TruncatedVZWCharacter]:
    """Generate finite-screen parent characters only for the audited block set."""

    bits = _default_capacity_bits(capacity_bits, dps=dps)
    return {
        label: truncated_vzw_character(SO10_312, weight, tau=tau, capacity_bits=bits, dps=dps)
        for label, weight in audited_parent_blocks()
    }


def solve_parent_subspace(
    *,
    tau: TauLike | None = None,
    capacity_bits: ExactScalar | None = None,
    dps: int | None = None,
) -> ParentSubspaceReport:
    """Solve the audited SO(10)_312 4-block subspace without enumeration."""

    bits = _default_capacity_bits(capacity_bits, dps=dps)
    s_submatrix = parent_modular_s_submatrix(dps=dps)
    characters = parent_character_generators(tau=tau, capacity_bits=bits, dps=dps)
    blocks: list[ParentBlockSolution] = []
    with mpmath.workdps(_resolved_dps(dps)):
        for label, weight in audited_parent_blocks():
            numerator = weyl_denominator_for_shifted_weight(_add_weights(weight, SO10_312.rho), dps=dps)
            vacuum = weyl_denominator_for_shifted_weight(SO10_312.rho, dps=dps)
            quantum_dimension = numerator / vacuum
            core_dimension = SO10_312.quantum_dimension(weight, dps=dps)
            character = characters[label]
            blocks.append(
                ParentBlockSolution(
                    label=label,
                    weight=weight,
                    quantum_dimension=quantum_dimension,
                    core_quantum_dimension=core_dimension,
                    weyl_denominator_numerator=numerator,
                    weyl_denominator_vacuum=vacuum,
                    finite_character=character,
                    dimension_residual=abs(quantum_dimension - core_dimension),
                    character_dimension_residual=abs(quantum_dimension - character.quantum_dimension),
                )
            )
    return ParentSubspaceReport(
        capacity_bits=bits,
        full_parent_primary_count=SO10_312.integrable_weight_count(),
        block_order=AUDITED_PARENT_BLOCK_ORDER,
        blocks=tuple(blocks),
        modular_s_submatrix=s_submatrix,
    )


def _default_capacity_bits(capacity_bits: ExactScalar | None, *, dps: int | None = None) -> mpmath.mpf:
    if capacity_bits is None:
        return finite_screen_budget(dps=dps).bits
    return _coerce_mpf(capacity_bits)


def _coerce_mpf(value: ExactScalar | mpmath.mpf) -> mpmath.mpf:
    if isinstance(value, float):
        raise TypeError("Binary floats are not accepted in the parent subspace solver.")
    if isinstance(value, Fraction):
        return mpmath.mpf(value.numerator) / mpmath.mpf(value.denominator)
    if isinstance(value, Decimal):
        return mpmath.mpf(str(value))
    return mpmath.mpf(value)


def _resolved_dps(dps: int | None) -> int:
    if dps is None:
        return max(PARENT_SUBSPACE_DPS, int(mpmath.mp.dps))
    return max(PARENT_SUBSPACE_DPS, int(dps))


def _add_weights(left: Weight, right: Weight) -> Weight:
    return tuple(left[index] + right[index] for index in range(len(left)))


def _root_pairing(weight: Weight, root: Weight) -> int:
    return sum(label * coefficient for label, coefficient in zip(weight, root))


__all__ = [
    "AUDITED_PARENT_BLOCK_ORDER",
    "PARENT_BLOCK_WEIGHTS",
    "PARENT_SUBSPACE_DPS",
    "PARENT_SUBSPACE_NOISE_WALL",
    "ParentBlockSolution",
    "ParentSubspaceReport",
    "audited_parent_blocks",
    "parent_character_generators",
    "parent_modular_s_submatrix",
    "parent_quantum_dimensions",
    "solve_parent_subspace",
    "weyl_character_quantum_dimension",
    "weyl_denominator_for_shifted_weight",
]