"""Command-line driver for the Spectral-Invariants-MTC bridge audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import mpmath

from mtc_engine.bridge_audit import (
    NOISE_WALL,
    BridgeAuditReport,
    ExactScalar,
    audit_bridge_precision,
)

MINIMUM_DPS = 250
DEFAULT_RENDER_DIGITS = 80


def build_parser() -> argparse.ArgumentParser:
    """Build the professional command-line interface for the audit driver."""

    parser = argparse.ArgumentParser(
        prog="mtc-audit",
        description=(
            "Run the Spectral-Invariants-MTC high-precision bridge audit and "
            "render a target-free QA ledger."
        ),
    )
    parser.add_argument(
        "--verify-full-fusion-matrices",
        action="store_true",
        help=(
            "Construct full tractable fusion matrices where enabled. This is slower "
            "than the default Verlinde/Frobenius-Perron dimension check."
        ),
    )
    parser.add_argument(
        "--dps",
        type=int,
        default=MINIMUM_DPS,
        help=(
            "Decimal digits for the mpmath context. Values below 250 are promoted "
            "to preserve the holographic noise-floor audit."
        ),
    )
    parser.add_argument(
        "--constants-path",
        type=Path,
        default=None,
        help="Optional repository-relative path to a physics_constants TeX file.",
    )
    parser.add_argument(
        "--perturbation",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Override or add an off-shell perturbation using a decimal string, for "
            "example --perturbation kappa_d5=1e-90. May be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=DEFAULT_RENDER_DIGITS,
        help=(
            "Significant digits to render for mpmath values. Formatting never casts "
            "through binary float."
        ),
    )
    parser.add_argument(
        "--show-precision-log",
        action="store_true",
        help="Always include the full precision-error log, even when the audit passes.",
    )
    return parser


def render_report(
    report: BridgeAuditReport,
    *,
    digits: int = DEFAULT_RENDER_DIGITS,
    include_precision_log: bool = False,
) -> str:
    """Render a transparent BridgeAuditReport without implicit float casts.

    Category A reports independently derived topological inputs: exact branch
    arithmetic and tractable Frobenius-Perron extractions. Category B reports
    internal consistency checks: large-sector algebraic self-comparisons,
    diagonal/loop trace closures, mass-gap identities, and stiffness probes.
    """

    resolved_digits = _resolved_digits(digits)
    lines: list[str] = [
        "Spectral-Invariants-MTC :: Transparent Precision QA Ledger",
        "==========================================================",
        f"Status                    : {_status(report.passed)}",
        f"mpmath dps                : {mpmath.mp.dps}",
        f"noise wall                : {_format_mpf(NOISE_WALL, resolved_digits)}",
        f"constants source          : {report.completion.source.as_posix()}",
        f"c_dark exact              : {_format_mpf(report.completion.c_dark, resolved_digits)}",
    ]
    if report.completion.display_value is not None:
        lines.append(f"c_dark display            : {_format_mpf(report.completion.display_value, resolved_digits)}")
    if report.completion.display_rounding_residue is not None:
        lines.append(
            "display rounding residue  : "
            f"{_format_mpf(report.completion.display_rounding_residue, resolved_digits)}"
        )

    lines.extend(
        (
            "",
            "Category A: Independent Topological Foundations",
            "================================================",
            "These entries are non-circular algebraic inputs or tractable direct checks.",
            "",
            "Diophantine Phase-Space Zeros",
            "------------------------------",
            "Exact Visible-Branch Framing Arithmetic",
            "---------------------------------------",
        )
    )
    for zero in report.phase_zeros:
        if zero.label == "mass_gap_quarter_power":
            continue
        lines.extend(
            (
                f"[{_status(zero.passed)}] {zero.label}",
                f"  relation                : {zero.relation}",
                f"  integer target          : {zero.integer_target}",
                f"  diophantine residual    : {zero.diophantine_residual}",
                f"  phase residual          : {_format_mpf(zero.phase_residual, resolved_digits)}",
                f"  normalized residual     : {_format_mpf(zero.normalized_residual, resolved_digits)}",
            )
        )

    lines.extend(("", "Frobenius-Perron Checks", "------------------------"))
    lines.extend(("Tractable Frobenius-Perron Extractions", "--------------------------------------"))
    for check in report.frobenius_perron:
        if check.sector_name == "SO(10)_312":
            continue
        lines.extend(
            (
                f"[{_status(check.passed)}] {check.sector_name} primary={check.primary}",
                f"  matrix verification     : {'full fusion matrix' if check.matrix_verified else 'Verlinde/Weyl dimension formula'}",
                f"  eigenvalue              : {_format_mpf(check.eigenvalue, resolved_digits)}",
                f"  quantum dimension       : {_format_mpf(check.quantum_dimension, resolved_digits)}",
                f"  residual                : {_format_mpf(check.residual, resolved_digits)}",
                f"  residual / c_dark       : {_format_mpf(check.normalized_residual, resolved_digits)}",
            )
        )

    loop = report.loop_trace
    lines.extend(
        (
            "",
            "Category B: Internal Consistency & Sanity Checks",
            "================================================",
            "These entries validate closure identities, large-sector algebraic "
            "self-consistency, and derived rigidity behavior.",
            "",
            "Massive Parent-Sector Self-Consistency",
            "--------------------------------------",
        )
    )
    for check in report.frobenius_perron:
        if check.sector_name != "SO(10)_312":
            continue
        lines.extend(
            (
                f"[{_status(check.passed)}] {check.sector_name} primary={check.primary}",
                "  classification          : internal algebraic self-comparison",
                "  note                    : bypasses full parent fusion matrix diagonalization",
                f"  matrix verification     : {'full fusion matrix' if check.matrix_verified else 'Verlinde/Weyl dimension formula'}",
                f"  eigenvalue              : {_format_mpf(check.eigenvalue, resolved_digits)}",
                f"  quantum dimension       : {_format_mpf(check.quantum_dimension, resolved_digits)}",
                f"  residual                : {_format_mpf(check.residual, resolved_digits)}",
                f"  residual / c_dark       : {_format_mpf(check.normalized_residual, resolved_digits)}",
            )
        )

    lines.extend(
        (
            "",
            "Loop Trace Residuals",
            "--------------------",
            "Loop Trace and Diagonal-Trace Identities",
            "----------------------------------------",
            f"parent trace sector       : {loop.parent_trace.sector_name}",
            f"parent primary count      : {loop.parent_trace.primary_count}",
            f"parent torus trace        : {_format_mpf(loop.parent_trace.value, resolved_digits)}",
            f"modular-T reference trace : {_format_mpf(loop.modular_t.reference_value, resolved_digits)}",
            f"modular-T transformed     : {_format_mpf(loop.modular_t.transformed_value, resolved_digits)}",
            f"modular-T residual        : {_format_mpf(loop.modular_t.residual, resolved_digits)}",
            f"diagonal modulus residual : {_format_mpf(loop.modular_t.diagonal_modulus_residual, resolved_digits)}",
            f"mass-gap exponent         : {loop.mass_gap.scaling_exponent}",
            f"mass-gap value            : {_format_mpf(loop.mass_gap.mass_gap, resolved_digits)}",
            f"mass-gap closure residual : {_format_mpf(loop.mass_gap.closure_residual, resolved_digits)}",
            f"c_dark weighted residual  : {_format_mpf(loop.completion_weighted_residue, resolved_digits)}",
            f"weighted residual / c_dark: {_format_mpf(loop.normalized_residue, resolved_digits)}",
            f"loop status               : {_status(loop.passed)}",
        )
    )

    lines.extend(("", "Mass-Gap Scaling Identity", "-------------------------"))
    for zero in report.phase_zeros:
        if zero.label != "mass_gap_quarter_power":
            continue
        lines.extend(
            (
                f"[{_status(zero.passed)}] {zero.label}",
                "  classification          : internal algebraic exponent identity",
                f"  relation                : {zero.relation}",
                f"  integer target          : {zero.integer_target}",
                f"  diophantine residual    : {zero.diophantine_residual}",
                f"  phase residual          : {_format_mpf(zero.phase_residual, resolved_digits)}",
                f"  normalized residual     : {_format_mpf(zero.normalized_residual, resolved_digits)}",
            )
        )

    lines.extend(("", "Off-Shell Continuous Perturbation Responses", "--------------------------------------------"))
    lines.extend(("Off-Shell Rigidity Sanity Checks", "--------------------------------"))
    for response in report.off_shell_perturbations:
        lines.extend(
            (
                f"[{_status(response.passed)}] {response.parameter}",
                f"  baseline                : {_format_mpf(response.baseline, resolved_digits)}",
                f"  perturbation            : {_format_mpf(response.perturbation, resolved_digits)}",
                f"  attempted value         : {_format_mpf(response.attempted_value, resolved_digits)}",
                f"  finite defect norm      : {_format_mpf(response.closure_tensor.finite_norm, resolved_digits)}",
                f"  root metric displacement: {_format_mpf(response.closure_tensor.root_metric_displacement, resolved_digits)}",
                f"  integer support residual: {_format_mpf(response.closure_tensor.level_integrality_residual, resolved_digits)}",
                f"  lepton embedding ratio  : {_format_mpf(response.closure_tensor.lepton_embedding_ratio, resolved_digits)}",
                f"  lepton embedding resid. : {_format_mpf(response.closure_tensor.lepton_embedding_residual, resolved_digits)}",
                f"  quark embedding ratio   : {_format_mpf(response.closure_tensor.quark_embedding_ratio, resolved_digits)}",
                f"  quark embedding resid.  : {_format_mpf(response.closure_tensor.quark_embedding_residual, resolved_digits)}",
                f"  framing defect variation: {_format_mpf(response.closure_tensor.framing_defect_variation, resolved_digits)}",
                f"  framing phase residual  : {_format_mpf(response.closure_tensor.framing_phase_residual, resolved_digits)}",
                f"  central charge remainder: {_format_mpf(response.closure_tensor.modular_central_charge_remainder, resolved_digits)}",
                f"  prefactor residual      : {_format_mpf(response.closure_tensor.prefactor_residual, resolved_digits)}",
                f"  capacity residual       : {_format_mpf(response.closure_tensor.capacity_residual, resolved_digits)}",
                f"  tangent norm            : {_format_mpf(response.closure_tensor.tangent_norm, resolved_digits)}",
                f"  closure tensor norm     : {_format_mpf(response.closure_tensor_norm, resolved_digits)}",
                f"  diverged                : {response.diverged}",
                f"  accepted                : {response.accepted}",
            )
        )

    if include_precision_log or not report.passed:
        lines.extend(("", "Precision Error Log", "-------------------"))
        for entry in report.error_log.entries:
            lines.extend(
                (
                    f"[{_status(entry.passed)}] {entry.label}",
                    f"  residual                : {_format_mpf(entry.residual, resolved_digits)}",
                    f"  normalized residual     : {_format_mpf(entry.normalized_residual, resolved_digits)}",
                    f"  wall                    : {_format_mpf(entry.noise_wall, resolved_digits)}",
                    f"  detail                  : {entry.detail}",
                )
            )

    lines.append("")
    lines.append("[SPECTRAL CLOSURE VALIDATED]" if report.passed else "[SPECTRAL CLOSURE FAILED]")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bridge audit CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.dps <= 0:
        parser.error("--dps must be positive")
    if args.digits <= 0:
        parser.error("--digits must be positive")
    resolved_dps = max(MINIMUM_DPS, args.dps)
    mpmath.mp.dps = resolved_dps

    try:
        perturbations = _parse_perturbations(args.perturbation)
        report = audit_bridge_precision(
            constants_path=args.constants_path,
            perturbations=perturbations or None,
            verify_full_fusion_matrices=args.verify_full_fusion_matrices,
            dps=resolved_dps,
        )
    except Exception as exc:
        print(f"[SPECTRAL CLOSURE FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        render_report(
            report,
            digits=args.digits,
            include_precision_log=args.show_precision_log or not report.passed,
        )
    )
    return 0 if report.passed else 1


def _parse_perturbations(values: Sequence[str]) -> Mapping[str, ExactScalar]:
    perturbations: dict[str, ExactScalar] = {}
    for raw_value in values:
        if "=" not in raw_value:
            raise ValueError(f"Perturbation must use NAME=VALUE syntax: {raw_value}")
        name, value = raw_value.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Perturbation must include both name and value: {raw_value}")
        perturbations[name] = value
    return perturbations


def _format_mpf(value: object, digits: int) -> str:
    """Format high-precision values without converting through float."""

    if isinstance(value, mpmath.ctx_mp_python.mpc):
        real = _format_mpf(mpmath.re(value), digits)
        imaginary = _format_mpf(abs(mpmath.im(value)), digits)
        sign = "+" if mpmath.im(value) >= 0 else "-"
        return f"({real} {sign} {imaginary}j)"
    if isinstance(value, mpmath.ctx_mp_python.mpf):
        if mpmath.isinf(value):
            return "+inf" if value > 0 else "-inf"
        if mpmath.isnan(value):
            return "nan"
        return mpmath.nstr(value, n=digits, strip_zeros=False, min_fixed=-8, max_fixed=8)
    return str(value)


def _resolved_digits(digits: int) -> int:
    if digits <= 0:
        raise ValueError("Render digits must be positive.")
    return int(digits)


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


if __name__ == "__main__":
    raise SystemExit(main())