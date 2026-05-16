"""CLI rendering tests for the precision bridge audit driver."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import sys

import mpmath

sys.path.insert(0, "src")

from mtc_engine.bridge_audit import audit_bridge_precision  # noqa: E402
from mtc_engine.run_audit import main, render_report  # noqa: E402


def test_render_report_contains_required_qa_ledger_sections() -> None:
    """The renderer exposes all audit channels and the success sentinel."""

    report = audit_bridge_precision()
    rendered = render_report(report, digits=30)

    assert "Category A: Independent Topological Foundations" in rendered
    assert "Category B: Internal Consistency & Sanity Checks" in rendered
    assert "Exact Visible-Branch Framing Arithmetic" in rendered
    assert "Frobenius-Perron Checks" in rendered
    assert "Tractable Frobenius-Perron Extractions" in rendered
    assert "Loop Trace Residuals" in rendered
    assert "Loop Trace and Diagonal-Trace Identities" in rendered
    assert "Mass-Gap Scaling Identity" in rendered
    assert "Off-Shell Continuous Perturbation Responses" in rendered
    assert "Off-Shell Rigidity Sanity Checks" in rendered
    assert "Massive Parent-Sector Self-Consistency" in rendered
    assert "internal algebraic self-comparison" in rendered
    assert "[SPECTRAL CLOSURE VALIDATED]" in rendered


def test_render_report_exposes_full_closure_tensor_channels() -> None:
    """The stiffness renderer prints the derived tensor channels, not only infinity."""

    report = audit_bridge_precision(perturbations={"parent_level": Decimal("1e-50")})
    rendered = render_report(report, digits=30)

    assert "lepton embedding ratio" in rendered
    assert "lepton embedding resid." in rendered
    assert "quark embedding ratio" in rendered
    assert "quark embedding resid." in rendered
    assert "framing phase residual" in rendered
    assert "tangent norm" in rendered
    assert "diverged                : True" in rendered


def test_render_report_dumps_precision_log_on_failure() -> None:
    """Failed reports include the precision log required for diagnostics."""

    report = audit_bridge_precision()
    first_entry = report.error_log.entries[0]
    failed_entry = replace(
        first_entry,
        normalized_residual=mpmath.mpf("1e-10"),
        residual=mpmath.mpf("1e-10"),
    )
    failed_log = replace(report.error_log, entries=(failed_entry, *report.error_log.entries[1:]))
    failed_report = replace(report, error_log=failed_log)
    rendered = render_report(failed_report, digits=30)

    assert "Precision Error Log" in rendered
    assert "[SPECTRAL CLOSURE FAILED]" in rendered


def test_render_report_accepts_explicit_zero_perturbation() -> None:
    """The ledger reports an on-shell zero perturbation as passing."""

    report = audit_bridge_precision(perturbations={"parent_level": Decimal("0")})
    rendered = render_report(report, digits=30)

    assert "Status                    : PASS" in rendered
    assert "[PASS] parent_level" in rendered
    assert "accepted                : True" in rendered


def test_main_returns_zero_for_valid_default_audit(capsys) -> None:
    """The CLI exits successfully when spectral closure validates."""

    exit_code = main(["--digits", "25"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[SPECTRAL CLOSURE VALIDATED]" in captured.out
    assert "Frobenius-Perron Checks" in captured.out