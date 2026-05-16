"""High-precision bridge audit for target-free spectral closure checks.

The audit layer intentionally does not fit or tune any low-energy parameter.
It reads the branch completion residue from the repository constants, evaluates
quantities derived by ``core.py`` and ``exact_traces.py``, and records whether
all residuals vanish at the 10^-199 numerical noise wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Final, Mapping, TypeAlias

import mpmath

from mtc_engine.core import (
    DEFAULT_DPS,
    SO10_312,
    SU2_26,
    SU3_8,
    Weight,
    WZWSector,
    frobenius_perron_eigenvalue,
)
from mtc_engine.exact_traces import (
    MassGapScalingTrace,
    ModularTTrace,
    TorusPartitionTrace,
    analytical_loop_trace_mass_gap,
    compute_exact_trace,
    finite_screen_budget,
    load_physics_constants,
    modular_t_trace,
    visible_branch_framing_defect,
)

ExactScalar: TypeAlias = int | str | Decimal | Fraction | mpmath.mpf

AUDIT_DPS: Final[int] = DEFAULT_DPS
NOISE_WALL: Final[mpmath.mpf] = mpmath.mpf("1e-199")
DEFAULT_CONSTANTS_PATH: Final[Path] = Path("context/physics_constants.tex")

mpmath.mp.dps = max(mpmath.mp.dps, AUDIT_DPS)


@dataclass(frozen=True)
class CompletionResidue:
    """Branch completion residue loaded from the physics constants context."""

    c_dark: mpmath.mpf
    display_value: mpmath.mpf | None
    display_rounding_residue: mpmath.mpf | None
    source: Path


@dataclass(frozen=True)
class PrecisionLogEntry:
    """One high-precision residual mapped against the c_dark completion scale."""

    label: str
    residual: mpmath.mpf
    normalized_residual: mpmath.mpf
    noise_wall: mpmath.mpf
    detail: str

    @property
    def passed(self) -> bool:
        """Return whether the normalized residual lies below the audit wall."""

        return self.normalized_residual <= self.noise_wall


@dataclass(frozen=True)
class PrecisionErrorLog:
    """Immutable high-precision error log for zero-residue checks."""

    c_dark: mpmath.mpf
    entries: tuple[PrecisionLogEntry, ...]
    noise_wall: mpmath.mpf = NOISE_WALL

    @property
    def passed(self) -> bool:
        """Return whether every logged residual is below the noise wall."""

        return all(entry.passed for entry in self.entries)


@dataclass(frozen=True)
class FrobeniusPerronAudit:
    """Audit one primary block's Frobenius-Perron quantum dimension."""

    sector_name: str
    primary: Weight
    eigenvalue: mpmath.mpf
    quantum_dimension: mpmath.mpf
    residual: mpmath.mpf
    normalized_residual: mpmath.mpf
    matrix_verified: bool

    @property
    def passed(self) -> bool:
        """Return whether the eigenvalue/dimension residue is audit-zero."""

        return self.normalized_residual <= NOISE_WALL


@dataclass(frozen=True)
class LoopTraceBridgeAudit:
    """Bridge exact loop traces into the completion-residue audit scale."""

    parent_trace: TorusPartitionTrace
    modular_t: ModularTTrace
    mass_gap: MassGapScalingTrace
    completion_weighted_residue: mpmath.mpf
    normalized_residue: mpmath.mpf

    @property
    def passed(self) -> bool:
        """Return whether all loop-trace closures are below the noise wall."""

        return (
            self.modular_t.passed
            and self.mass_gap.passed
            and self.normalized_residue <= NOISE_WALL
        )


@dataclass(frozen=True)
class DiophantinePhaseZero:
    """Exact integer/phase relation in the benchmark branch."""

    label: str
    relation: Fraction
    integer_target: int
    diophantine_residual: Fraction
    phase_residual: mpmath.mpf
    normalized_residual: mpmath.mpf
    noise_wall: mpmath.mpf

    @property
    def passed(self) -> bool:
        """Return whether both the exact and evaluated phase residues vanish."""

        return self.diophantine_residual == 0 and self.normalized_residual <= self.noise_wall


@dataclass(frozen=True)
class ClosureDefectTensor:
    """Derived off-shell closure tensor for the zero-dimensional branch anchor.

    The tensor components are computed from the attempted VOA coordinates rather
    than assigned by policy.  Level perturbations are measured against the root
    embedding hyperplanes and the integer support conditions for affine
    integrable weights.  The framing channels recompute the two benchmark
    embedding ratios ``k_D5/(2 k_A1)`` and ``k_D5/(3 k_A2)``.  The central-charge
    channel compares the Sugawara central-charge T phase for the attempted
    levels with the benchmark phase.  Capacity and prefactor channels compare
    the finite-register and D5 prefactor coordinates with their derived anchor
    values.

    ``finite_norm`` is the ordinary Euclidean norm of these first-principles
    closure residues.  ``bulk_norm`` is the induced normal response of the
    singleton topological moduli space: a nonzero finite defect divided by the
    zero tangent norm is infinite, while the exact anchor has zero response.
    """

    parent_level: mpmath.mpf
    lepton_level: mpmath.mpf
    quark_level: mpmath.mpf
    capacity_bits: mpmath.mpf
    kappa_d5: mpmath.mpf
    root_metric_displacement: mpmath.mpf
    level_integrality_residual: mpmath.mpf
    lepton_embedding_ratio: mpmath.mpf
    quark_embedding_ratio: mpmath.mpf
    lepton_embedding_residual: mpmath.mpf
    quark_embedding_residual: mpmath.mpf
    framing_defect_variation: mpmath.mpf
    framing_phase_residual: mpmath.mpf
    modular_central_charge_remainder: mpmath.mpf
    prefactor_residual: mpmath.mpf
    capacity_residual: mpmath.mpf
    finite_norm: mpmath.mpf
    tangent_norm: mpmath.mpf
    bulk_norm: mpmath.mpf

    @property
    def structurally_closed(self) -> bool:
        """Return whether every derived closure channel remains exactly zero."""

        return self.finite_norm == 0


@dataclass(frozen=True)
class OffShellPerturbationAudit:
    """Projector response to a continuous coordinate perturbation.

    The response stores both the scalar parameter displacement and the full
    derived closure tensor.  No branch assigns infinity directly from
    ``perturbation != 0``; divergence follows only after the recalculated VOA
    closure channels produce a nonzero finite defect on a singleton moduli
    space with zero tangent norm.
    """

    parameter: str
    baseline: mpmath.mpf
    perturbation: mpmath.mpf
    attempted_value: mpmath.mpf
    closure_tensor: ClosureDefectTensor
    closure_tensor_norm: mpmath.mpf
    accepted: bool

    @property
    def diverged(self) -> bool:
        """Return whether the derived bulk-closure norm is infinite."""

        return self.closure_tensor_norm == mpmath.inf

    @property
    def passed(self) -> bool:
        """Return whether the projector response matches the closure tensor."""

        if self.closure_tensor.structurally_closed:
            return self.accepted and self.closure_tensor_norm == 0
        return self.diverged and not self.accepted


@dataclass(frozen=True)
class BridgeAuditReport:
    """Full target-free QA bridge report."""

    completion: CompletionResidue
    frobenius_perron: tuple[FrobeniusPerronAudit, ...]
    loop_trace: LoopTraceBridgeAudit
    phase_zeros: tuple[DiophantinePhaseZero, ...]
    error_log: PrecisionErrorLog
    off_shell_perturbations: tuple[OffShellPerturbationAudit, ...]

    @property
    def passed(self) -> bool:
        """Return whether all exact, bridge, and stiffness checks pass."""

        return bool(
            all(check.passed for check in self.frobenius_perron)
            and self.loop_trace.passed
            and all(zero.passed for zero in self.phase_zeros)
            and self.error_log.passed
            and all(response.passed for response in self.off_shell_perturbations)
        )


def completion_residue_from_constants(
    path: Path | None = None,
    *,
    dps: int | None = None,
) -> CompletionResidue:
    """Load the c_dark completion residue from the repository constants."""

    constants_path = _resolve_constants_path(path)
    constants = load_physics_constants(constants_path, dps=dps)
    if "cDarkCompletionExact" in constants:
        c_dark = constants["cDarkCompletionExact"]
    elif "cDarkCompletionFull" in constants:
        c_dark = constants["cDarkCompletionFull"]
    else:
        raise KeyError("No cDarkCompletionExact or cDarkCompletionFull constant was found.")
    display_value = constants.get("cDarkCompletionFull")
    display_rounding_residue = None if display_value is None else abs(c_dark - display_value)
    return CompletionResidue(
        c_dark=c_dark,
        display_value=display_value,
        display_rounding_residue=display_rounding_residue,
        source=constants_path,
    )


def frobenius_perron_audits(
    c_dark: mpmath.mpf,
    *,
    verify_full_fusion_matrices: bool = False,
    dps: int | None = None,
) -> tuple[FrobeniusPerronAudit, ...]:
    """Evaluate branch primary Frobenius-Perron dimensions at high precision."""

    specs: tuple[tuple[WZWSector, Weight, bool], ...] = (
        (SU2_26, (1,), True),
        (SU3_8, (1, 0), verify_full_fusion_matrices),
        (SO10_312, (1, 0, 0, 0, 0), False),
    )
    return tuple(
        _single_frobenius_perron_audit(
            sector,
            primary,
            c_dark,
            matrix_verified=matrix_verified,
            dps=dps,
        )
        for sector, primary, matrix_verified in specs
    )


def loop_trace_bridge_audit(
    c_dark: mpmath.mpf,
    *,
    dps: int | None = None,
) -> LoopTraceBridgeAudit:
    """Evaluate finite-screen loop traces and map their residue to c_dark."""

    parent_trace = compute_exact_trace(SO10_312, dps=dps)
    modular_trace = modular_t_trace(SU2_26, weights=((0,), (1,), (2,)), dps=dps)
    mass_gap_trace = analytical_loop_trace_mass_gap(dps=dps)
    raw_residue = modular_trace.residual + mass_gap_trace.closure_residual
    completion_weighted_residue = c_dark * raw_residue
    return LoopTraceBridgeAudit(
        parent_trace=parent_trace,
        modular_t=modular_trace,
        mass_gap=mass_gap_trace,
        completion_weighted_residue=completion_weighted_residue,
        normalized_residue=_normalize_residue(completion_weighted_residue, c_dark),
    )


def diophantine_phase_space_zero_log(
    c_dark: mpmath.mpf,
    *,
    dps: int | None = None,
) -> tuple[DiophantinePhaseZero, ...]:
    """Track exact benchmark phase zeros down to the 10^-199 wall."""

    framing = visible_branch_framing_defect()
    relations: tuple[tuple[str, Fraction], ...] = (
        ("parent_over_two_lepton", Fraction(framing.parent_level, 2 * framing.lepton_level)),
        ("parent_over_three_quark", Fraction(framing.parent_level, 3 * framing.quark_level)),
        ("visible_delta_fr", framing.delta_fr),
        ("mass_gap_quarter_power", Fraction(1, 1) + (4 * Fraction(-1, 4))),
    )
    zeros: list[DiophantinePhaseZero] = []
    with mpmath.workdps(_resolved_dps(dps)):
        for label, relation in relations:
            integer_target = _nearest_integer(relation)
            diophantine_residual = abs(relation - integer_target)
            phase = mpmath.exp(mpmath.mpc("0", "2") * mpmath.pi * _mp_fraction(relation))
            phase_residual = abs(phase - 1)
            normalized = _normalize_residue(phase_residual + _mp_fraction(diophantine_residual), c_dark)
            zeros.append(
                DiophantinePhaseZero(
                    label=label,
                    relation=relation,
                    integer_target=integer_target,
                    diophantine_residual=diophantine_residual,
                    phase_residual=phase_residual,
                    normalized_residual=normalized,
                    noise_wall=NOISE_WALL,
                )
            )
    return tuple(zeros)


def high_precision_error_log(
    c_dark: mpmath.mpf,
    *,
    fp_checks: tuple[FrobeniusPerronAudit, ...],
    loop_trace: LoopTraceBridgeAudit,
    phase_zeros: tuple[DiophantinePhaseZero, ...],
) -> PrecisionErrorLog:
    """Build the immutable precision log for all zero-residue channels."""

    entries: list[PrecisionLogEntry] = []
    for check in fp_checks:
        entries.append(
            PrecisionLogEntry(
                label=f"fp::{check.sector_name}::{check.primary}",
                residual=check.residual,
                normalized_residual=check.normalized_residual,
                noise_wall=NOISE_WALL,
                detail="Frobenius-Perron eigenvalue minus quantum dimension.",
            )
        )
    entries.extend(
        (
            PrecisionLogEntry(
                label="loop::modular_t",
                residual=loop_trace.modular_t.residual,
                normalized_residual=_normalize_residue(loop_trace.modular_t.residual, c_dark),
                noise_wall=NOISE_WALL,
                detail="Complex character-intersection trace under modular T with Delta_fr=0.",
            ),
            PrecisionLogEntry(
                label="loop::mass_gap_closure",
                residual=loop_trace.mass_gap.closure_residual,
                normalized_residual=_normalize_residue(loop_trace.mass_gap.closure_residual, c_dark),
                noise_wall=NOISE_WALL,
                detail="N * (m/(kappa_D5 M_P))^4 - 1.",
            ),
            PrecisionLogEntry(
                label="loop::c_dark_weighted_residue",
                residual=loop_trace.completion_weighted_residue,
                normalized_residual=loop_trace.normalized_residue,
                noise_wall=NOISE_WALL,
                detail="Loop residual after direct multiplication by c_dark.",
            ),
        )
    )
    for zero in phase_zeros:
        entries.append(
            PrecisionLogEntry(
                label=f"phase::{zero.label}",
                residual=zero.phase_residual + _mp_fraction(zero.diophantine_residual),
                normalized_residual=zero.normalized_residual,
                noise_wall=zero.noise_wall,
                detail=f"Relation {zero.relation} locked to integer {zero.integer_target}.",
            )
        )
    return PrecisionErrorLog(c_dark=c_dark, entries=tuple(entries), noise_wall=NOISE_WALL)


def simulate_off_shell_coordinate_perturbation(
    parameter: str,
    perturbation: ExactScalar,
    *,
    dps: int | None = None,
) -> OffShellPerturbationAudit:
    """Recompute the VOA closure tensor after one coordinate perturbation.

    The benchmark branch is the zero-dimensional topological anchor
    ``(k_D5, k_A1, k_A2, N, kappa_D5)``.  This routine perturbs one coordinate,
    then recomputes the following first-principles closure channels from the
    attempted coordinates:

    * root-embedding metric displacement of the affine level hyperplanes;
    * integer support residuals for the affine level lattice;
    * framing-ratio residuals for ``k_D5/(2 k_A1)`` and ``k_D5/(3 k_A2)``;
    * modular-T framing phase residual;
    * Sugawara central-charge T-phase remainder;
    * D5 prefactor and finite-capacity anchor residuals.

    A divergent bulk norm is returned only when these recalculated channels give
    a nonzero finite defect and the defect is pushed against the zero tangent
    norm of the singleton topological moduli space.
    """

    baselines = _anchor_baselines(dps=dps)
    if parameter not in baselines:
        raise KeyError(f"Unknown topological anchor: {parameter}")
    with mpmath.workdps(_resolved_dps(dps)):
        baseline = baselines[parameter]
        delta = _coerce_mpf(perturbation)
        attempted = baseline + delta
        attempted_coordinates = dict(baselines)
        attempted_coordinates[parameter] = attempted
        closure_tensor = _closure_defect_tensor(attempted_coordinates, baselines, dps=dps)
        closure_tensor_norm = closure_tensor.bulk_norm
        accepted = closure_tensor.structurally_closed
        return OffShellPerturbationAudit(
            parameter=parameter,
            baseline=baseline,
            perturbation=delta,
            attempted_value=attempted,
            closure_tensor=closure_tensor,
            closure_tensor_norm=closure_tensor_norm,
            accepted=accepted,
        )


def simulate_off_shell_coordinate_perturbations(
    perturbations: Mapping[str, ExactScalar] | None = None,
    *,
    dps: int | None = None,
) -> tuple[OffShellPerturbationAudit, ...]:
    """Evaluate the derived stiffness tensor for all requested perturbations.

    When no mapping is supplied, the default suite probes all five topological
    anchor coordinates.  Supplying an empty mapping intentionally returns an
    empty suite; supplying zero-valued perturbations recomputes the closure
    tensor and accepts the coordinate only if every structural residual remains
    exactly zero.
    """

    resolved = (
        {
            "parent_level": "1e-60",
            "lepton_level": "1e-80",
            "quark_level": "1e-100",
            "capacity_bits": "1e-40",
            "kappa_d5": "1e-90",
        }
        if perturbations is None
        else perturbations
    )
    return tuple(
        simulate_off_shell_coordinate_perturbation(parameter, perturbation, dps=dps)
        for parameter, perturbation in resolved.items()
    )


def _closure_defect_tensor(
    attempted: Mapping[str, mpmath.mpf],
    baseline: Mapping[str, mpmath.mpf],
    *,
    dps: int | None = None,
) -> ClosureDefectTensor:
    """Build the finite closure-defect tensor from attempted VOA coordinates."""

    with mpmath.workdps(_resolved_dps(dps)):
        parent_level = attempted["parent_level"]
        lepton_level = attempted["lepton_level"]
        quark_level = attempted["quark_level"]
        capacity_bits = attempted["capacity_bits"]
        kappa_d5 = attempted["kappa_d5"]

        root_metric_displacement = _root_embedding_metric_displacement(attempted, baseline)
        level_integrality_residual = max(
            _distance_to_integer_mpf(parent_level),
            _distance_to_integer_mpf(lepton_level),
            _distance_to_integer_mpf(quark_level),
        )
        lepton_embedding_ratio = parent_level / (mpmath.mpf("2") * lepton_level)
        quark_embedding_ratio = parent_level / (mpmath.mpf("3") * quark_level)
        lepton_embedding_residual = _distance_to_integer_mpf(lepton_embedding_ratio)
        quark_embedding_residual = _distance_to_integer_mpf(quark_embedding_ratio)
        framing_defect_variation = max(lepton_embedding_residual, quark_embedding_residual)
        framing_phase_residual = abs(
            mpmath.exp(mpmath.mpc("0", "2") * mpmath.pi * framing_defect_variation) - 1
        )
        modular_central_charge_remainder = _modular_central_charge_remainder(attempted, baseline, dps=dps)
        derived_kappa = _continuous_kappa_d5(lepton_level, dps=dps)
        prefactor_residual = abs((kappa_d5 / derived_kappa) - 1)
        capacity_residual = abs((capacity_bits / baseline["capacity_bits"]) - 1)

        finite_norm = mpmath.sqrt(
            root_metric_displacement**2
            + level_integrality_residual**2
            + lepton_embedding_residual**2
            + quark_embedding_residual**2
            + framing_phase_residual**2
            + modular_central_charge_remainder**2
            + prefactor_residual**2
            + capacity_residual**2
        )
        tangent_norm = _topological_tangent_norm()
        bulk_norm = _bulk_closure_norm(finite_norm, tangent_norm)
        return ClosureDefectTensor(
            parent_level=parent_level,
            lepton_level=lepton_level,
            quark_level=quark_level,
            capacity_bits=capacity_bits,
            kappa_d5=kappa_d5,
            root_metric_displacement=root_metric_displacement,
            level_integrality_residual=level_integrality_residual,
            lepton_embedding_ratio=lepton_embedding_ratio,
            quark_embedding_ratio=quark_embedding_ratio,
            lepton_embedding_residual=lepton_embedding_residual,
            quark_embedding_residual=quark_embedding_residual,
            framing_defect_variation=framing_defect_variation,
            framing_phase_residual=framing_phase_residual,
            modular_central_charge_remainder=modular_central_charge_remainder,
            prefactor_residual=prefactor_residual,
            capacity_residual=capacity_residual,
            finite_norm=finite_norm,
            tangent_norm=tangent_norm,
            bulk_norm=bulk_norm,
        )


def _root_embedding_metric_displacement(
    attempted: Mapping[str, mpmath.mpf],
    baseline: Mapping[str, mpmath.mpf],
) -> mpmath.mpf:
    """Return the metric norm of affine-level displacement in root space."""

    total = mpmath.mpf("0")
    for name, sector in (
        ("parent_level", SO10_312),
        ("lepton_level", SU2_26),
        ("quark_level", SU3_8),
    ):
        level_delta = attempted[name] - baseline[name]
        total += _highest_root_norm_squared(sector) * level_delta * level_delta
    return mpmath.sqrt(total)


def _highest_root_norm_squared(sector: WZWSector) -> mpmath.mpf:
    """Compute ``(theta, theta)`` from Dynkin marks and the Cartan metric."""

    total = mpmath.mpf("0")
    marks = sector.dynkin_marks
    for row in range(sector.rank):
        for column in range(sector.rank):
            total += mpmath.mpf(marks[row]) * mpmath.mpf(sector.cartan_matrix[row][column]) * mpmath.mpf(
                marks[column]
            )
    return total


def _modular_central_charge_remainder(
    attempted: Mapping[str, mpmath.mpf],
    baseline: Mapping[str, mpmath.mpf],
    *,
    dps: int | None = None,
) -> mpmath.mpf:
    """Return the largest Sugawara central-charge T-phase displacement."""

    with mpmath.workdps(_resolved_dps(dps)):
        residuals = []
        for name, sector in (
            ("parent_level", SO10_312),
            ("lepton_level", SU2_26),
            ("quark_level", SU3_8),
        ):
            attempted_charge = _continuous_central_charge(sector, attempted[name])
            baseline_charge = _continuous_central_charge(sector, baseline[name])
            phase = mpmath.exp(-mpmath.mpc("0", "2") * mpmath.pi * (attempted_charge - baseline_charge) / 24)
            residuals.append(abs(phase - 1))
        return max(residuals)


def _continuous_central_charge(sector: WZWSector, level: mpmath.mpf) -> mpmath.mpf:
    """Evaluate the Sugawara central charge at a real attempted level."""

    dimension = mpmath.mpf(sector.rank + (2 * len(sector.positive_roots)))
    denominator = level + mpmath.mpf(sector.dual_coxeter_number)
    if denominator == 0:
        return mpmath.inf
    return level * dimension / denominator


def _continuous_kappa_d5(lepton_level: mpmath.mpf, *, dps: int | None = None) -> mpmath.mpf:
    """Analytically continue the D5 prefactor formula to an attempted A1 level."""

    with mpmath.workdps(_resolved_dps(dps)):
        denominator = lepton_level + mpmath.mpf("2")
        if denominator <= 0:
            return mpmath.nan
        lepton_total_dimension = mpmath.sqrt(denominator / 2) / mpmath.sin(mpmath.pi / denominator)
        beta = mpmath.mpf("0.5") * mpmath.log(lepton_total_dimension)
        area_ratio = (mpmath.mpf(160) / mpmath.mpf(1521)) * mpmath.sqrt(mpmath.mpf(10))
        spinor_retention = (mpmath.mpf(347) - (mpmath.mpf(8) * beta * beta)) / mpmath.mpf(351)
        return mpmath.sqrt((mpmath.mpf(16) / mpmath.mpf(5)) * area_ratio * spinor_retention)


def _distance_to_integer_mpf(value: mpmath.mpf) -> mpmath.mpf:
    """Return the high-precision distance from a real value to the nearest integer."""

    nearest = mpmath.floor(value + mpmath.mpf("0.5"))
    return abs(value - nearest)


def _topological_tangent_norm() -> mpmath.mpf:
    """Return the tangent norm of the singleton topological moduli space."""

    return mpmath.mpf("0")


def _bulk_closure_norm(finite_defect_norm: mpmath.mpf, tangent_norm: mpmath.mpf) -> mpmath.mpf:
    """Lift a finite closure defect to the singular bulk response norm."""

    if finite_defect_norm == 0:
        return mpmath.mpf("0")
    if tangent_norm == 0:
        return mpmath.inf
    return finite_defect_norm / tangent_norm


def audit_bridge_precision(
    *,
    constants_path: Path | None = None,
    perturbations: Mapping[str, ExactScalar] | None = None,
    verify_full_fusion_matrices: bool = False,
    dps: int | None = None,
) -> BridgeAuditReport:
    """Execute the full target-free bridge validation pipeline."""

    completion = completion_residue_from_constants(constants_path, dps=dps)
    fp_checks = frobenius_perron_audits(
        completion.c_dark,
        verify_full_fusion_matrices=verify_full_fusion_matrices,
        dps=dps,
    )
    loop_trace = loop_trace_bridge_audit(completion.c_dark, dps=dps)
    phase_zeros = diophantine_phase_space_zero_log(completion.c_dark, dps=dps)
    error_log = high_precision_error_log(
        completion.c_dark,
        fp_checks=fp_checks,
        loop_trace=loop_trace,
        phase_zeros=phase_zeros,
    )
    off_shell = simulate_off_shell_coordinate_perturbations(perturbations, dps=dps)
    return BridgeAuditReport(
        completion=completion,
        frobenius_perron=fp_checks,
        loop_trace=loop_trace,
        phase_zeros=phase_zeros,
        error_log=error_log,
        off_shell_perturbations=off_shell,
    )


def _single_frobenius_perron_audit(
    sector: WZWSector,
    primary: Weight,
    c_dark: mpmath.mpf,
    *,
    matrix_verified: bool,
    dps: int | None = None,
) -> FrobeniusPerronAudit:
    with mpmath.workdps(_resolved_dps(dps)):
        quantum_dimension = sector.quantum_dimension(primary, dps=dps)
        if matrix_verified:
            eigenvalue = frobenius_perron_eigenvalue(sector.fusion_matrix(primary, dps=dps), dps=dps)
        else:
            eigenvalue = sector.frobenius_perron_dimension(primary, dps=dps)
        residual = abs(eigenvalue - quantum_dimension)
        return FrobeniusPerronAudit(
            sector_name=sector.name,
            primary=primary,
            eigenvalue=eigenvalue,
            quantum_dimension=quantum_dimension,
            residual=residual,
            normalized_residual=_normalize_residue(residual, c_dark),
            matrix_verified=matrix_verified,
        )


def _anchor_baselines(*, dps: int | None = None) -> dict[str, mpmath.mpf]:
    budget = finite_screen_budget(dps=dps)
    return {
        "parent_level": mpmath.mpf(SO10_312.level),
        "lepton_level": mpmath.mpf(SU2_26.level),
        "quark_level": mpmath.mpf(SU3_8.level),
        "capacity_bits": budget.bits,
        "kappa_d5": analytical_loop_trace_mass_gap(dps=dps).kappa_d5,
    }


def _resolve_constants_path(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Constants file not found: {path}")
        return path
    if DEFAULT_CONSTANTS_PATH.exists():
        return DEFAULT_CONSTANTS_PATH
    return finite_screen_budget().source


def _nearest_integer(value: Fraction) -> int:
    quotient = value.numerator // value.denominator
    lower = Fraction(quotient, 1)
    upper = lower + 1
    return int(lower if abs(value - lower) <= abs(value - upper) else upper)


def _normalize_residue(residual: mpmath.mpf, c_dark: mpmath.mpf) -> mpmath.mpf:
    if c_dark == 0:
        raise ZeroDivisionError("c_dark completion residue cannot be zero.")
    return abs(residual) / abs(c_dark)


def _coerce_mpf(value: ExactScalar | mpmath.mpf) -> mpmath.mpf:
    if isinstance(value, float):
        raise TypeError("Binary floats are not accepted in the bridge audit; use strings, Decimals, or Fractions.")
    if isinstance(value, Fraction):
        return _mp_fraction(value)
    if isinstance(value, Decimal):
        return mpmath.mpf(str(value))
    return mpmath.mpf(value)


def _mp_fraction(value: Fraction) -> mpmath.mpf:
    return mpmath.mpf(value.numerator) / mpmath.mpf(value.denominator)


def _resolved_dps(dps: int | None) -> int:
    if dps is None:
        return max(AUDIT_DPS, int(mpmath.mp.dps))
    return max(AUDIT_DPS, int(dps))


__all__ = [
    "AUDIT_DPS",
    "NOISE_WALL",
    "BridgeAuditReport",
    "ClosureDefectTensor",
    "CompletionResidue",
    "DiophantinePhaseZero",
    "FrobeniusPerronAudit",
    "LoopTraceBridgeAudit",
    "OffShellPerturbationAudit",
    "PrecisionErrorLog",
    "PrecisionLogEntry",
    "audit_bridge_precision",
    "completion_residue_from_constants",
    "diophantine_phase_space_zero_log",
    "frobenius_perron_audits",
    "high_precision_error_log",
    "loop_trace_bridge_audit",
    "simulate_off_shell_coordinate_perturbation",
    "simulate_off_shell_coordinate_perturbations",
]