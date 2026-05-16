"""Finite-screen exact trace tests."""

from __future__ import annotations

from fractions import Fraction
import sys

import mpmath

sys.path.insert(0, "src")

from mtc_engine.core import SO10_312, SU2_26  # noqa: E402
from mtc_engine.exact_traces import (  # noqa: E402
    analytical_loop_trace_mass_gap,
    anomaly_free_primary_blocks,
    capacity_oscillator_depth,
    central_charge,
    compute_exact_trace,
    conformal_weight,
    finite_lie_algebra_dimension,
    finite_screen_budget,
    modular_t_trace,
    nome,
    truncated_vzw_character,
    visible_branch_framing_defect,
)


def test_context_budget_is_loaded_at_high_precision() -> None:
    """The finite screen budget comes from the TeX constants file."""

    budget = finite_screen_budget()

    assert budget.bits == mpmath.mpf("3.3") * mpmath.power(mpmath.mpf("10"), 122)
    assert budget.noise_floor < mpmath.mpf("1e-122")


def test_vzw_character_data_uses_sector_invariants() -> None:
    """The truncated loop character exposes c, h, d, and finite mode depth."""

    character = truncated_vzw_character(SU2_26, (1,))

    assert finite_lie_algebra_dimension(SU2_26) == 3
    assert central_charge(SU2_26) == mpmath.mpf("39") / mpmath.mpf("14")
    assert conformal_weight(SU2_26, (1,)) == mpmath.mpf("3") / mpmath.mpf("112")
    assert character.oscillator_depth == capacity_oscillator_depth(abs(nome()), character.capacity_bits)
    assert abs(character.value) > 0


def test_modular_t_trace_closes_for_vanishing_framing_anomaly() -> None:
    """The benchmark branch has Delta_fr = 0 and a phase-sensitive zero residual."""

    framing = visible_branch_framing_defect()
    trace = modular_t_trace(SU2_26, weights=((0,), (1,), (2,)), dps=250)

    assert framing.delta_fr == 0
    assert trace.passed
    assert trace.residual < mpmath.mpf("1e-200")
    assert trace.diagonal_modulus_residual < mpmath.mpf("1e-200")
    assert abs(trace.transformed_value - trace.reference_value) < mpmath.mpf("1e-200")


def test_modular_t_trace_detects_nonzero_framing_before_modulus_square() -> None:
    """A nonzero Delta_fr rotates the unsquared character intersection trace."""

    trace = modular_t_trace(SU2_26, weights=((0,), (1,), (2,)), delta_fr=Fraction(1, 13), dps=250)
    expected_rotation = trace.anomaly_phase * trace.reference_value
    expected_residual = abs((trace.anomaly_phase - 1) * trace.reference_value)

    assert trace.delta_fr == Fraction(1, 13)
    assert not trace.passed
    assert abs(trace.transformed_value - expected_rotation) < mpmath.mpf("1e-200")
    assert abs(trace.residual - expected_residual) < mpmath.mpf("1e-200")
    assert trace.residual > mpmath.mpf("1e-199")
    assert trace.diagonal_modulus_residual < mpmath.mpf("1e-200")
    assert all(
        abs(transform.transformed_value - trace.anomaly_phase * transform.reference_value) < mpmath.mpf("1e-200")
        for transform in trace.transforms
    )


def test_so10_parent_trace_uses_audited_primary_blocks_by_default() -> None:
    """The compatibility wrapper avoids accidental full SO(10)_312 enumeration."""

    trace = compute_exact_trace(SO10_312)

    assert trace.sector_name == "SO(10)_312"
    assert trace.weights == anomaly_free_primary_blocks()
    assert trace.primary_count == 4
    assert trace.value > 0


def test_loop_trace_derives_exact_negative_quarter_scaling() -> None:
    """The finite-capacity closure gives m = kappa_D5 M_P N^(-1/4)."""

    trace = analytical_loop_trace_mass_gap(planck_mass="1")

    assert trace.scaling_exponent == Fraction(-1, 4)
    assert trace.passed
    assert abs(trace.normalized_loop_trace - 1) < mpmath.mpf("1e-200")