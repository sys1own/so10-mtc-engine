"""Rigidity tests for spectral invariant preservation."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import sys

import mpmath
import pytest

sys.path.insert(0, "src")

from mtc_engine.bridge_audit import (  # noqa: E402
    NOISE_WALL,
    audit_bridge_precision,
    completion_residue_from_constants,
    diophantine_phase_space_zero_log,
    simulate_off_shell_coordinate_perturbation,
)
from mtc_engine.core import (  # noqa: E402
    ANOMALY_FREE_SO10_312_BRANCH,
    DEFAULT_DPS,
    KAPPA_D5,
    SO10_312,
    SU2_26,
    SU3_8,
    WZWSector,
    frobenius_perron_eigenvalue,
)
from mtc_engine.exact_traces import finite_screen_budget  # noqa: E402
from mtc_engine.parent_subspace import (  # noqa: E402
    AUDITED_PARENT_BLOCK_ORDER,
    PARENT_BLOCK_WEIGHTS,
    PARENT_SUBSPACE_NOISE_WALL,
    solve_parent_subspace,
    weyl_character_quantum_dimension,
)


def test_default_precision_floor_is_250_digits() -> None:
    """The module must initialize mpmath above the holographic noise floor."""

    assert DEFAULT_DPS == 250
    assert mpmath.mp.dps >= 250


def test_sector_root_and_primary_counts_are_codified() -> None:
    """The anomaly-free branch exposes the expected algebraic sectors."""

    assert SU2_26.integrable_weight_count() == 27
    assert SU3_8.integrable_weight_count() == 45
    assert len(SO10_312.positive_roots) == 20
    assert len(SO10_312.weyl_group) == 1920
    assert SO10_312.is_integrable((1, 0, 0, 0, 0))


def test_su2_verlinde_matrix_has_expected_perron_root() -> None:
    """The SU(2)_26 fundamental fusion matrix recovers its quantum dimension."""

    fusion_matrix = SU2_26.fusion_matrix((1,))
    perron_root = frobenius_perron_eigenvalue(fusion_matrix)
    quantum_dimension = SU2_26.quantum_dimension((1,))

    assert SU2_26.fusion_coefficient((1,), (1,), (0,)) == 1
    assert SU2_26.fusion_coefficient((1,), (1,), (2,)) == 1
    assert abs(perron_root - quantum_dimension) < mpmath.mpf("1e-200")


def test_su3_verlinde_product_contains_singlet_and_adjoint() -> None:
    """The SU(3)_8 Verlinde product 3 x anti-3 yields 1 + 8."""

    assert SU3_8.fusion_coefficient((1, 0), (0, 1), (0, 0)) == 1
    assert SU3_8.fusion_coefficient((1, 0), (0, 1), (1, 1)) == 1


def test_so10_parent_is_guarded_against_accidental_full_enumeration() -> None:
    """SO(10)_312 is defined, but its full primary spectrum is intentionally lazy."""

    assert SO10_312.integrable_weight_count() > SO10_312.max_full_fusion_primaries
    with pytest.raises(ValueError, match="refusing to enumerate"):
        SO10_312.integrable_weights()


def test_kappa_d5_is_derived_from_quantum_dimensions() -> None:
    """The D5 prefactor is generated from level-set invariants without fitting."""

    audit = ANOMALY_FREE_SO10_312_BRANCH.verify_kappa_d5()

    assert audit.passed
    assert abs(KAPPA_D5 - mpmath.mpf("0.98877105126637883272731137446867755258")) < mpmath.mpf("1e-38")


def test_benchmark_exact_closure() -> None:
    """Default bridge audit closure must pass without conditional tolerancing."""

    report = audit_bridge_precision()

    assert report.passed is True
    assert report.loop_trace.passed is True
    assert report.error_log.passed is True
    assert all(check.passed for check in report.frobenius_perron)
    assert all(zero.passed for zero in report.phase_zeros)


@pytest.mark.parametrize(
    ("parameter", "perturbation"),
    (
        ("parent_level", Decimal("1e-50")),
        ("lepton_level", Decimal("1e-70")),
        ("quark_level", Decimal("1e-90")),
        ("capacity_bits", Decimal("1e-40")),
        ("kappa_d5", Decimal("1e-110")),
    ),
)
def test_stiff_jacobian_singularity(parameter: str, perturbation: Decimal) -> None:
    """Tiny continuous off-shell coordinate tweaks diverge after closure recomputation."""

    response = simulate_off_shell_coordinate_perturbation(parameter, perturbation)

    assert response.accepted is False
    assert response.diverged is True
    assert response.closure_tensor_norm == mpmath.inf
    assert response.closure_tensor.finite_norm > 0
    assert response.closure_tensor.tangent_norm == 0


def test_level_perturbation_recomputes_root_lattice_and_central_charge() -> None:
    """Off-shell levels break integer support, framing, and T-phase closure."""

    response = simulate_off_shell_coordinate_perturbation("parent_level", Decimal("1e-50"))
    tensor = response.closure_tensor

    assert tensor.root_metric_displacement > 0
    assert tensor.level_integrality_residual > 0
    assert tensor.framing_defect_variation > 0
    assert tensor.framing_phase_residual > 0
    assert tensor.modular_central_charge_remainder > 0
    assert tensor.prefactor_residual == 0
    assert tensor.capacity_residual == 0


def test_prefactor_and_capacity_perturbations_use_derived_anchor_residuals() -> None:
    """Nonnumeric root channels stay zero while anchor residuals force divergence."""

    prefactor = simulate_off_shell_coordinate_perturbation("kappa_d5", Decimal("1e-110"))
    capacity = simulate_off_shell_coordinate_perturbation("capacity_bits", Decimal("1e-40"))

    assert prefactor.closure_tensor.root_metric_displacement == 0
    assert prefactor.closure_tensor.level_integrality_residual == 0
    assert prefactor.closure_tensor.framing_defect_variation == 0
    assert prefactor.closure_tensor.modular_central_charge_remainder == 0
    assert prefactor.closure_tensor.prefactor_residual > 0
    assert prefactor.diverged
    assert capacity.closure_tensor.capacity_residual > 0
    assert capacity.closure_tensor.prefactor_residual == 0
    assert capacity.diverged


def test_diophantine_phase_zeros() -> None:
    """Every benchmark phase relation is an exact Fraction-level zero."""

    c_dark = completion_residue_from_constants().c_dark

    for zero in diophantine_phase_space_zero_log(c_dark):
        exact_residue = zero.relation - Fraction(zero.integer_target, 1)
        assert exact_residue == Fraction(0, 1)
        assert zero.diophantine_residual == abs(exact_residue)
        assert zero.diophantine_residual == Fraction(0, 1)
        assert zero.normalized_residual <= NOISE_WALL


def test_bridge_audit_maps_derived_residues_to_c_dark() -> None:
    """The bridge audit validates derived structures against the c_dark scale."""

    report = audit_bridge_precision()

    assert report.completion.source.as_posix() == "context/physics_constants.tex"
    assert report.completion.display_value == mpmath.mpf("3.300805139659")
    assert report.loop_trace.normalized_residue <= NOISE_WALL
    assert all(check.normalized_residual <= NOISE_WALL for check in report.frobenius_perron)


def test_completion_residue_uses_exact_fraction_before_display_decimal() -> None:
    """The displayed c_dark decimal is not used as a fitted target."""

    completion = completion_residue_from_constants()

    assert completion.c_dark != completion.display_value
    assert completion.display_rounding_residue is not None
    assert completion.display_rounding_residue < mpmath.mpf("1e-12")


def test_zero_perturbation_remains_on_shell() -> None:
    """The stiffness projector only diverges for actual off-shell movement."""

    response = simulate_off_shell_coordinate_perturbation("parent_level", Decimal("0"))

    assert response.accepted is True
    assert response.diverged is False
    assert response.passed is True
    assert response.closure_tensor_norm == 0


def test_zero_perturbation_report_remains_valid() -> None:
    """Explicit on-shell perturbation overrides do not invalidate report closure."""

    report = audit_bridge_precision(perturbations={"parent_level": Decimal("0")})

    assert report.passed is True
    assert len(report.off_shell_perturbations) == 1
    assert report.off_shell_perturbations[0].passed is True


def test_binary_float_perturbations_are_rejected() -> None:
    """Bridge QA forbids accidental 64-bit perturbation shortcuts."""

    with pytest.raises(TypeError, match="Binary floats"):
        simulate_off_shell_coordinate_perturbation("kappa_d5", 1e-30)

def test_parent_subspace_targets_only_audited_blocks() -> None:
    """The parent solver exposes exactly the four anomaly-free SO(10)_312 blocks."""

    report = solve_parent_subspace()

    assert report.passed is True
    assert report.block_order == AUDITED_PARENT_BLOCK_ORDER
    assert tuple(block.label for block in report.blocks) == AUDITED_PARENT_BLOCK_ORDER
    assert tuple(block.weight for block in report.blocks) == tuple(
        PARENT_BLOCK_WEIGHTS[label] for label in AUDITED_PARENT_BLOCK_ORDER
    )
    assert len(report.modular_s_submatrix) == 4
    assert all(len(row) == 4 for row in report.modular_s_submatrix)
    assert report.full_parent_primary_count == 6_563_729_615


def test_parent_subspace_dimensions_are_finite_screen_rigid() -> None:
    """Targeted Weyl-denominator dimensions remain fixed under the screen budget."""

    budget = finite_screen_budget()
    report = solve_parent_subspace(capacity_bits=budget.bits)

    assert report.capacity_bits == budget.bits
    for block in report.blocks:
        direct_dimension = weyl_character_quantum_dimension(block.weight)
        assert block.finite_character.capacity_bits == budget.bits
        assert abs(direct_dimension - block.quantum_dimension) <= PARENT_SUBSPACE_NOISE_WALL
        assert block.dimension_residual <= PARENT_SUBSPACE_NOISE_WALL
        assert block.character_dimension_residual <= PARENT_SUBSPACE_NOISE_WALL
        assert abs(block.quantum_dimension - SO10_312.quantum_dimension(block.weight)) <= PARENT_SUBSPACE_NOISE_WALL


def test_parent_subspace_solver_never_full_enumerates_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The targeted parent solver must not instantiate the full D5 primary spectrum."""

    def fail_full_enumeration(*_args, **_kwargs):
        raise AssertionError("full parent enumeration attempted")

    monkeypatch.setattr(WZWSector, "integrable_weights", fail_full_enumeration)

    report = solve_parent_subspace()

    assert report.passed is True
    assert tuple(block.label for block in report.blocks) == AUDITED_PARENT_BLOCK_ORDER