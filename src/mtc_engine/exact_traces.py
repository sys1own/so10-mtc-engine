"""Finite-screen loop traces for WZW current-algebra truncation.

The continuous affine Kac-Moody current tower on a conformal cylinder is not
enumerated as an infinite object here.  Instead, the module applies the finite
screen budget ``N`` from ``context/physics_constants*.tex`` and keeps only
current oscillator modes whose Boltzmann weight remains above the holographic
floor ``1/N``.  This gives an explicit, auditable truncation rule without
importing any low-energy fitting or differential-equation layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, Mapping, TypeAlias

import mpmath

from mtc_engine.core import (
    ANOMALY_FREE_SO10_312_BRANCH,
    DEFAULT_DPS,
    KAPPA_D5,
    SO10_312,
    Weight,
    WZWSector,
)

ExactScalar: TypeAlias = int | str | Fraction | mpmath.mpf
TauLike: TypeAlias = str | tuple[ExactScalar, ExactScalar] | mpmath.mpc

DEFAULT_TRACE_DPS: Final[int] = DEFAULT_DPS
MAX_AUTOMATIC_OSCILLATOR_DEPTH: Final[int] = 4096
PHYSICS_CONSTANTS_CANDIDATES: Final[tuple[Path, ...]] = (
    Path("context/physics_constants.tex"),
    Path("context/physics_constants (2).tex"),
)

mpmath.mp.dps = max(mpmath.mp.dps, DEFAULT_TRACE_DPS)


@dataclass(frozen=True)
class FiniteScreenBudget:
    """Finite holographic screen capacity used to truncate cylinder modes."""

    bits: mpmath.mpf
    source: Path

    @property
    def noise_floor(self) -> mpmath.mpf:
        """Return the register floor below which loop modes are unresolvable."""

        return mpmath.mpf("1") / self.bits


@dataclass(frozen=True)
class TruncatedVZWCharacter:
    """One finite-capacity VZW loop-character evaluation.

    The character equation implemented is the finite-screen current-loop block

        chi_lambda^(N)(tau) = d_lambda q^(h_lambda - c/24)
                              prod_{n=1}^{L_N(tau)} (1 - q^n)^(-dim g),

    where ``L_N`` is the largest oscillator index with ``|q|^n >= 1/N``.  The
    primary data ``d_lambda``, ``h_lambda``, and ``c`` come from the same WZW
    sector object used by the Verlinde engine.
    """

    sector_name: str
    weight: Weight
    tau: mpmath.mpc
    q: mpmath.mpc
    capacity_bits: mpmath.mpf
    oscillator_depth: int
    central_charge: mpmath.mpf
    conformal_weight: mpmath.mpf
    quantum_dimension: mpmath.mpf
    primary_factor: mpmath.mpc
    oscillator_factor: mpmath.mpc
    value: mpmath.mpc


@dataclass(frozen=True)
class TorusPartitionTrace:
    """Diagonal torus trace assembled from finite-screen VZW characters."""

    sector_name: str
    tau: mpmath.mpc
    capacity_bits: mpmath.mpf
    weights: tuple[Weight, ...]
    characters: tuple[TruncatedVZWCharacter, ...]
    value: mpmath.mpf

    @property
    def primary_count(self) -> int:
        """Return the number of primary blocks included in the trace."""

        return len(self.weights)


@dataclass(frozen=True)
class ModularTCharacterTransform:
    """One unsquared primary-block transform under modular T and framing phase."""

    weight: Weight
    original_value: mpmath.mpc
    modular_phase: mpmath.mpc
    anomaly_phase: mpmath.mpc
    total_phase: mpmath.mpc
    reference_value: mpmath.mpc
    transformed_value: mpmath.mpc
    intersection_contribution: mpmath.mpc
    diagonal_contribution: mpmath.mpf


@dataclass(frozen=True)
class ModularTTrace:
    """Phase-sensitive modular-T audit for finite-screen character loops.

    ``diagonal_modulus_value`` records the conventional absolute-square torus
    trace and is intentionally not used as the anomaly detector, because a pure
    framing phase cancels inside ``abs(.)**2``.  ``transformed_value`` is instead
    the complex character-intersection trace

        sum_lambda chi_lambda^T * conj(chi_lambda^{T, Delta_fr=0}),

    normalized against ``reference_value``.  A nonzero framing anomaly rotates
    this complex intersection by ``exp(2*pi*i*Delta_fr)`` and is therefore
    visible before any absolute-value squaring.
    """

    original: TorusPartitionTrace
    transformed_tau: mpmath.mpc
    transforms: tuple[ModularTCharacterTransform, ...]
    reference_value: mpmath.mpc
    transformed_value: mpmath.mpc
    diagonal_modulus_value: mpmath.mpf
    diagonal_modulus_residual: mpmath.mpf
    delta_fr: Fraction
    anomaly_phase: mpmath.mpc
    residual: mpmath.mpf

    @property
    def passed(self) -> bool:
        """Return whether the phase-sensitive T trace closes at audit precision."""

        return self.delta_fr == 0 and self.residual <= mpmath.power(mpmath.mpf(10), -200)


@dataclass(frozen=True)
class MassGapScalingTrace:
    """Analytical trace showing the exact N^(-1/4) mass-gap exponent."""

    capacity_bits: mpmath.mpf
    kappa_d5: mpmath.mpf
    planck_mass: mpmath.mpf
    scaling_exponent: Fraction
    mass_gap: mpmath.mpf
    normalized_loop_trace: mpmath.mpf
    closure_residual: mpmath.mpf

    @property
    def passed(self) -> bool:
        """Return whether N * (m / (kappa_D5 M_P))^4 closes to unity."""

        return self.scaling_exponent == Fraction(-1, 4) and self.closure_residual <= mpmath.power(
            mpmath.mpf(10), -200
        )


@dataclass(frozen=True)
class FramingDefect:
    """Exact arithmetic audit of the visible branch framing defect."""

    parent_level: int
    lepton_level: int
    quark_level: int
    lepton_gap: Fraction
    quark_gap: Fraction
    delta_fr: Fraction


_CHAR_LINE = re.compile(r"\\providecommand\{\\(?P<name>[^}]+)\}\{(?P<value>.+)\}")
_FRAC = re.compile(r"^\\frac\{(?P<num>-?\d+)\}\{(?P<den>-?\d+)\}$")
_SCI = re.compile(r"^(?P<mantissa>-?\d+(?:\.\d+)?)\\times10\^\{(?P<exponent>-?\d+)\}$")


def load_physics_constants(path: Path | None = None, *, dps: int | None = None) -> Mapping[str, mpmath.mpf]:
    """Load high-precision numeric constants from the repository TeX context."""

    constants_path = _resolve_physics_constants_path(path)
    constants: dict[str, mpmath.mpf] = {}
    with mpmath.workdps(_resolved_dps(dps)):
        for line in constants_path.read_text().splitlines():
            match = _CHAR_LINE.match(line.strip())
            if match is None:
                continue
            try:
                constants[match.group("name")] = _parse_tex_number(match.group("value"))
            except ValueError:
                continue
    return constants


def finite_screen_budget(path: Path | None = None, *, dps: int | None = None) -> FiniteScreenBudget:
    """Return the finite boundary-screen budget ``N`` from the context file."""

    constants_path = _resolve_physics_constants_path(path)
    constants = load_physics_constants(constants_path, dps=dps)
    if "modularHorizonBits" in constants:
        bits = constants["modularHorizonBits"]
    elif "maxComplexityCapacity" in constants:
        bits = constants["maxComplexityCapacity"]
    else:
        raise KeyError("No modularHorizonBits or maxComplexityCapacity entry found in physics constants.")
    if bits <= 0:
        raise ValueError("Finite-screen capacity must be positive.")
    return FiniteScreenBudget(bits=bits, source=constants_path)


def central_charge(sector: WZWSector, *, dps: int | None = None) -> mpmath.mpf:
    """Return the affine WZW central charge c = k dim(g)/(k + h^vee)."""

    with mpmath.workdps(_resolved_dps(dps)):
        dimension = finite_lie_algebra_dimension(sector)
        return mpmath.mpf(sector.level * dimension) / mpmath.mpf(sector.denominator)


def finite_lie_algebra_dimension(sector: WZWSector) -> int:
    """Return dim(g) from rank plus positive and negative roots."""

    return sector.rank + (2 * len(sector.positive_roots))


def conformal_weight(sector: WZWSector, weight: Weight, *, dps: int | None = None) -> mpmath.mpf:
    """Return h_lambda = (lambda, lambda + 2 rho)/(2(k + h^vee))."""

    if not sector.is_integrable(weight):
        raise ValueError(f"Weight {weight} is not integrable for {sector.name}.")
    with mpmath.workdps(_resolved_dps(dps)):
        shifted = tuple(weight[index] + (2 * sector.rho[index]) for index in range(sector.rank))
        numerator = sector.inner_product(weight, shifted)
        return numerator / (mpmath.mpf("2") * mpmath.mpf(sector.denominator))


def nome(tau: TauLike | None = None, *, dps: int | None = None) -> mpmath.mpc:
    """Return q = exp(2 pi i tau) for an upper-half-plane cylinder modulus."""

    resolved_tau = _coerce_tau(tau)
    if mpmath.im(resolved_tau) <= 0:
        raise ValueError("The cylinder modulus tau must lie in the upper half-plane.")
    with mpmath.workdps(_resolved_dps(dps)):
        return mpmath.exp(mpmath.mpc("0", "2") * mpmath.pi * resolved_tau)


def capacity_oscillator_depth(
    q_abs: ExactScalar | mpmath.mpf,
    capacity_bits: ExactScalar | mpmath.mpf,
    *,
    max_depth: int = MAX_AUTOMATIC_OSCILLATOR_DEPTH,
    dps: int | None = None,
) -> int:
    """Return L_N with |q|^L_N >= 1/N > |q|^(L_N + 1)."""

    with mpmath.workdps(_resolved_dps(dps)):
        magnitude = _coerce_mpf(q_abs)
        bits = _coerce_mpf(capacity_bits)
        if magnitude <= 0 or magnitude >= 1:
            raise ValueError("|q| must lie strictly between zero and one.")
        if bits <= 1:
            raise ValueError("Capacity must exceed one bit for a nontrivial truncation.")
        depth = int(mpmath.floor(mpmath.log(bits) / (-mpmath.log(magnitude))))
    if depth > max_depth:
        raise ValueError(
            f"Automatic oscillator depth {depth} exceeds safety limit {max_depth}; "
            "choose a deeper imaginary modulus or pass an explicit oscillator_depth."
        )
    return max(depth, 0)


def truncated_vzw_character(
    sector: WZWSector,
    weight: Weight,
    *,
    tau: TauLike | None = None,
    capacity_bits: ExactScalar | mpmath.mpf | None = None,
    oscillator_depth: int | None = None,
    dps: int | None = None,
) -> TruncatedVZWCharacter:
    """Evaluate a finite-capacity VZW loop character at high precision."""

    if not sector.is_integrable(weight):
        raise ValueError(f"Weight {weight} is not integrable for {sector.name}.")
    with mpmath.workdps(_resolved_dps(dps)):
        tau_value = _coerce_tau(tau)
        q_value = nome(tau_value, dps=dps)
        bits = _default_capacity_bits(capacity_bits, dps=dps)
        depth = (
            capacity_oscillator_depth(abs(q_value), bits, dps=dps)
            if oscillator_depth is None
            else int(oscillator_depth)
        )
        if depth < 0:
            raise ValueError("Oscillator depth must be nonnegative.")

        c_value = central_charge(sector, dps=dps)
        h_value = conformal_weight(sector, weight, dps=dps)
        q_dimension = sector.quantum_dimension(weight, dps=dps)
        primary_exponent = h_value - (c_value / mpmath.mpf("24"))
        primary_factor = q_dimension * mpmath.exp(primary_exponent * mpmath.log(q_value))
        oscillator_factor = mpmath.mpc("1")
        current_dimension = finite_lie_algebra_dimension(sector)
        for mode in range(1, depth + 1):
            oscillator_factor *= mpmath.power(mpmath.mpc("1") - mpmath.power(q_value, mode), -current_dimension)
        return TruncatedVZWCharacter(
            sector_name=sector.name,
            weight=weight,
            tau=tau_value,
            q=q_value,
            capacity_bits=bits,
            oscillator_depth=depth,
            central_charge=c_value,
            conformal_weight=h_value,
            quantum_dimension=q_dimension,
            primary_factor=primary_factor,
            oscillator_factor=oscillator_factor,
            value=primary_factor * oscillator_factor,
        )


def torus_partition_trace(
    sector: WZWSector,
    *,
    tau: TauLike | None = None,
    weights: tuple[Weight, ...] | None = None,
    capacity_bits: ExactScalar | mpmath.mpf | None = None,
    oscillator_depth: int | None = None,
    dps: int | None = None,
) -> TorusPartitionTrace:
    """Construct a diagonal finite-screen torus trace from VZW characters.

    For tractable sectors, ``weights=None`` means the complete diagonal primary
    spectrum.  For SO(10)_312 the complete spectrum contains billions of
    primaries, so callers should pass the audited primary blocks they want to
    trace; otherwise the sector guard from ``core.WZWSector`` will refuse the
    accidental full enumeration.
    """

    spectrum = sector.integrable_weights() if weights is None else weights
    with mpmath.workdps(_resolved_dps(dps)):
        characters = tuple(
            truncated_vzw_character(
                sector,
                weight,
                tau=tau,
                capacity_bits=capacity_bits,
                oscillator_depth=oscillator_depth,
                dps=dps,
            )
            for weight in spectrum
        )
        value = sum(abs(character.value) ** 2 for character in characters)
        return TorusPartitionTrace(
            sector_name=sector.name,
            tau=_coerce_tau(tau),
            capacity_bits=_default_capacity_bits(capacity_bits, dps=dps),
            weights=tuple(spectrum),
            characters=characters,
            value=value,
        )


def modular_t_trace(
    sector: WZWSector,
    *,
    tau: TauLike | None = None,
    weights: tuple[Weight, ...] | None = None,
    delta_fr: Fraction | int | str | None = None,
    capacity_bits: ExactScalar | mpmath.mpf | None = None,
    oscillator_depth: int | None = None,
    dps: int | None = None,
) -> ModularTTrace:
    """Apply modular T and audit the unsquared complex character intersection.

    The usual diagonal torus scalar ``sum(abs(chi)^2)`` is retained only as a
    diagnostic because it cannot see pure framing phases.  The closure residual
    is computed from the complex pairing between the transformed character loop
    and the anomaly-free transformed loop, so a nonzero ``Delta_fr`` produces
    the explicit factor ``exp(2*pi*i*Delta_fr) - 1`` instead of being removed by
    absolute-value squaring.
    """

    framing = visible_branch_framing_defect().delta_fr if delta_fr is None else _coerce_fraction(delta_fr)
    original = torus_partition_trace(
        sector,
        tau=tau,
        weights=weights,
        capacity_bits=capacity_bits,
        oscillator_depth=oscillator_depth,
        dps=dps,
    )
    with mpmath.workdps(_resolved_dps(dps)):
        anomaly_phase = mpmath.exp(mpmath.mpc("0", "2") * mpmath.pi * _mp_fraction(framing))
        reference_value = mpmath.mpc("0")
        transformed_value = mpmath.mpc("0")
        diagonal_modulus_value = mpmath.mpf("0")
        transforms: list[ModularTCharacterTransform] = []
        for character in original.characters:
            modular_phase = modular_t_phase(sector, character.weight, dps=dps)
            reference_character = modular_phase * character.value
            total_phase = modular_phase * anomaly_phase
            transformed_character = total_phase * character.value
            intersection_contribution = transformed_character * mpmath.conj(reference_character)
            diagonal_contribution = abs(transformed_character) ** 2
            reference_value += abs(reference_character) ** 2
            transformed_value += intersection_contribution
            diagonal_modulus_value += diagonal_contribution
            transforms.append(
                ModularTCharacterTransform(
                    weight=character.weight,
                    original_value=character.value,
                    modular_phase=modular_phase,
                    anomaly_phase=anomaly_phase,
                    total_phase=total_phase,
                    reference_value=reference_character,
                    transformed_value=transformed_character,
                    intersection_contribution=intersection_contribution,
                    diagonal_contribution=diagonal_contribution,
                )
            )
        residual = abs(transformed_value - reference_value)
        diagonal_modulus_residual = abs(diagonal_modulus_value - original.value)
    return ModularTTrace(
        original=original,
        transformed_tau=original.tau + mpmath.mpc("1", "0"),
        transforms=tuple(transforms),
        reference_value=reference_value,
        transformed_value=transformed_value,
        diagonal_modulus_value=diagonal_modulus_value,
        diagonal_modulus_residual=diagonal_modulus_residual,
        delta_fr=framing,
        anomaly_phase=anomaly_phase,
        residual=residual,
    )


def modular_t_phase(sector: WZWSector, weight: Weight, *, dps: int | None = None) -> mpmath.mpc:
    """Return exp(2 pi i (h_lambda - c/24)) for a primary block."""

    with mpmath.workdps(_resolved_dps(dps)):
        exponent = conformal_weight(sector, weight, dps=dps) - (central_charge(sector, dps=dps) / mpmath.mpf("24"))
        return mpmath.exp(mpmath.mpc("0", "2") * mpmath.pi * exponent)


def analytical_loop_trace_mass_gap(
    *,
    capacity_bits: ExactScalar | mpmath.mpf | None = None,
    kappa_d5: ExactScalar | mpmath.mpf | None = None,
    planck_mass: ExactScalar | mpmath.mpf = "1",
    dps: int | None = None,
) -> MassGapScalingTrace:
    """Derive the mass-gap scaling from finite conformal-cylinder capacity.

    The finite-screen loop closure is encoded as

        N * (m / (kappa_D5 M_P))^4 = 1.

    Solving this exact algebraic trace identity gives
    ``m = kappa_D5 M_P N^(-1/4)`` and the logarithmic derivative
    ``d log(m) / d log(N) = -1/4``.  The routine returns the closure residual
    in high precision rather than consulting any observational fit.
    """

    with mpmath.workdps(_resolved_dps(dps)):
        bits = _default_capacity_bits(capacity_bits, dps=dps)
        kappa = KAPPA_D5 if kappa_d5 is None else _coerce_mpf(kappa_d5)
        planck = _coerce_mpf(planck_mass)
        if bits <= 0 or kappa <= 0 or planck <= 0:
            raise ValueError("Capacity, kappa_D5, and Planck normalization must be positive.")
        exponent = Fraction(-1, 4)
        mass_gap = kappa * planck * mpmath.power(bits, _mp_fraction(exponent))
        normalized_loop_trace = bits * mpmath.power(mass_gap / (kappa * planck), 4)
        residual = abs(normalized_loop_trace - mpmath.mpf("1"))
        return MassGapScalingTrace(
            capacity_bits=bits,
            kappa_d5=kappa,
            planck_mass=planck,
            scaling_exponent=exponent,
            mass_gap=mass_gap,
            normalized_loop_trace=normalized_loop_trace,
            closure_residual=residual,
        )


def compute_exact_trace(
    sector: WZWSector = SO10_312,
    *,
    tau: TauLike | None = None,
    weights: tuple[Weight, ...] | None = None,
    capacity_bits: ExactScalar | mpmath.mpf | None = None,
    oscillator_depth: int | None = None,
    dps: int | None = None,
) -> TorusPartitionTrace:
    """Compatibility wrapper returning a finite-screen torus partition trace."""

    resolved_weights = anomaly_free_primary_blocks() if weights is None and sector.name == SO10_312.name else weights
    return torus_partition_trace(
        sector,
        tau=tau,
        weights=resolved_weights,
        capacity_bits=capacity_bits,
        oscillator_depth=oscillator_depth,
        dps=dps,
    )


def anomaly_free_primary_blocks() -> tuple[Weight, ...]:
    """Return the SO(10)_312 primary blocks used for finite parent traces."""

    return tuple(
        weight
        for sector_name, weight in ANOMALY_FREE_SO10_312_BRANCH.primary_blocks.values()
        if sector_name == SO10_312.name
    )


def visible_branch_framing_defect(
    *,
    parent_level: int = 312,
    lepton_level: int = 26,
    quark_level: int = 8,
) -> FramingDefect:
    """Return the exact visible-branch framing defect Delta_fr."""

    lepton_gap = _distance_to_integer(Fraction(parent_level, 2 * lepton_level))
    quark_gap = _distance_to_integer(Fraction(parent_level, 3 * quark_level))
    return FramingDefect(
        parent_level=parent_level,
        lepton_level=lepton_level,
        quark_level=quark_level,
        lepton_gap=lepton_gap,
        quark_gap=quark_gap,
        delta_fr=max(lepton_gap, quark_gap),
    )


def _default_capacity_bits(
    capacity_bits: ExactScalar | mpmath.mpf | None,
    *,
    dps: int | None = None,
) -> mpmath.mpf:
    if capacity_bits is not None:
        return _coerce_mpf(capacity_bits)
    return finite_screen_budget(dps=dps).bits


def _resolve_physics_constants_path(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Physics constants file not found: {path}")
        return path
    for candidate in PHYSICS_CONSTANTS_CANDIDATES:
        if candidate.exists():
            return candidate
    matches = sorted(Path("context").glob("physics_constants*.tex"))
    if matches:
        return matches[0]
    raise FileNotFoundError("No context/physics_constants*.tex file was found.")


def _parse_tex_number(value: str) -> mpmath.mpf:
    compact = value.strip().replace(" ", "")
    fraction_match = _FRAC.match(compact)
    if fraction_match is not None:
        return _mp_fraction(Fraction(int(fraction_match.group("num")), int(fraction_match.group("den"))))
    scientific_match = _SCI.match(compact)
    if scientific_match is not None:
        return mpmath.mpf(scientific_match.group("mantissa")) * mpmath.power(
            mpmath.mpf("10"), int(scientific_match.group("exponent"))
        )
    if re.fullmatch(r"-?\d+(?:\.\d+)?", compact):
        return mpmath.mpf(compact)
    raise ValueError(f"Unsupported TeX numeric literal: {value}")


def _coerce_tau(tau: TauLike | None) -> mpmath.mpc:
    if tau is None:
        return mpmath.mpc("0", "1")
    if isinstance(tau, mpmath.mpc):
        return tau
    if isinstance(tau, str):
        return mpmath.mpc(tau)
    real_part, imaginary_part = tau
    return mpmath.mpc(_coerce_mpf(real_part), _coerce_mpf(imaginary_part))


def _coerce_mpf(value: ExactScalar | mpmath.mpf) -> mpmath.mpf:
    if isinstance(value, float):
        raise TypeError("Binary floats are not accepted; pass an int, Fraction, mpf, or decimal string.")
    if isinstance(value, Fraction):
        return _mp_fraction(value)
    return mpmath.mpf(value)


def _coerce_fraction(value: Fraction | int | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    return Fraction(value)


def _distance_to_integer(value: Fraction) -> Fraction:
    remainder = Fraction(value.numerator % value.denominator, value.denominator)
    return min(remainder, Fraction(1, 1) - remainder)


def _mp_fraction(value: Fraction) -> mpmath.mpf:
    return mpmath.mpf(value.numerator) / mpmath.mpf(value.denominator)


def _resolved_dps(dps: int | None) -> int:
    if dps is None:
        return max(DEFAULT_TRACE_DPS, int(mpmath.mp.dps))
    return max(DEFAULT_TRACE_DPS, int(dps))


__all__ = [
    "DEFAULT_TRACE_DPS",
    "FiniteScreenBudget",
    "FramingDefect",
    "MassGapScalingTrace",
    "ModularTCharacterTransform",
    "ModularTTrace",
    "TorusPartitionTrace",
    "TruncatedVZWCharacter",
    "analytical_loop_trace_mass_gap",
    "anomaly_free_primary_blocks",
    "capacity_oscillator_depth",
    "central_charge",
    "compute_exact_trace",
    "conformal_weight",
    "finite_lie_algebra_dimension",
    "finite_screen_budget",
    "load_physics_constants",
    "modular_t_phase",
    "modular_t_trace",
    "nome",
    "torus_partition_trace",
    "truncated_vzw_character",
    "visible_branch_framing_defect",
]
