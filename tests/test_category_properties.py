"""High-precision category-property regressions for tractable WZW sectors."""

from __future__ import annotations

from functools import cache
import sys

import mpmath

sys.path.insert(0, "src")

from mtc_engine.core import SU2_26, SU3_8, frobenius_perron_eigenvalue  # noqa: E402


DPS = 250
NOISE_WALL = mpmath.mpf("1e-200")
SMALL_SECTORS = (SU2_26, SU3_8)

IntegerMatrix = tuple[tuple[int, ...], ...]


def test_s_matrix_unitarity() -> None:
    """The full tractable modular S matrices are unitary at the noise wall."""

    with mpmath.workdps(DPS):
        for sector in SMALL_SECTORS:
            primaries = sector.integrable_weights()
            s_matrix = sector.modular_s_matrix(primaries=primaries, dps=DPS)

            for row_index in range(len(primaries)):
                for column_index in range(len(primaries)):
                    entry = mpmath.mpc("0")
                    for summation_index in range(len(primaries)):
                        entry += s_matrix[row_index][summation_index] * mpmath.conj(
                            s_matrix[column_index][summation_index]
                        )
                    target = mpmath.mpf("1") if row_index == column_index else mpmath.mpf("0")
                    assert abs(entry - target) <= NOISE_WALL


def test_verlinde_ring_associativity_and_commutativity() -> None:
    """Fusion matrices commute and satisfy associativity exactly over integers."""

    for sector in SMALL_SECTORS:
        _, fusion_matrices = _fusion_matrices_for(sector.name)
        for left_matrix_index, left_matrix in enumerate(fusion_matrices):
            for right_matrix_index, right_matrix in enumerate(fusion_matrices):
                left_then_right = _matrix_product(left_matrix, right_matrix)
                right_then_left = _matrix_product(right_matrix, left_matrix)
                assert left_then_right == right_then_left

                structure_coefficients = fusion_matrices[left_matrix_index][right_matrix_index]
                left_associated = _linear_combination(structure_coefficients, fusion_matrices)
                assert left_associated == right_then_left


def test_frobenius_perron_dimensions_preserve_fusion_multiplication() -> None:
    """Quantum dimensions form a fusion-ring character at extreme precision."""

    with mpmath.workdps(DPS):
        for sector in SMALL_SECTORS:
            primaries, fusion_matrices = _fusion_matrices_for(sector.name)
            dimensions = tuple(sector.quantum_dimension(primary, dps=DPS) for primary in primaries)

            for left_index, left_dimension in enumerate(dimensions):
                for right_index, right_dimension in enumerate(dimensions):
                    preserved_dimension = mpmath.mpf("0")
                    for output_index, coefficient in enumerate(fusion_matrices[left_index][right_index]):
                        if coefficient:
                            preserved_dimension += mpmath.mpf(coefficient) * dimensions[output_index]
                    assert abs((left_dimension * right_dimension) - preserved_dimension) <= NOISE_WALL

            for fundamental_weight in sector.fundamental_weights:
                fundamental_index = primaries.index(fundamental_weight)
                eigenvalue = frobenius_perron_eigenvalue(
                    fusion_matrices[fundamental_index],
                    dps=DPS,
                )
                assert abs(eigenvalue - dimensions[fundamental_index]) <= NOISE_WALL


@cache
def _fusion_matrices_for(sector_name: str) -> tuple[tuple[tuple[int, ...], ...], tuple[IntegerMatrix, ...]]:
    if sector_name == SU2_26.name:
        return _su2_fusion_matrices()
    if sector_name == SU3_8.name:
        return _su3_fusion_matrices()
    raise ValueError(f"Unsupported tractable sector: {sector_name}")


def _su2_fusion_matrices() -> tuple[tuple[tuple[int, ...], ...], tuple[IntegerMatrix, ...]]:
    primaries = SU2_26.integrable_weights()
    matrices: dict[tuple[int, ...], IntegerMatrix] = {
        (0,): _identity_matrix(len(primaries)),
        (1,): SU2_26.fusion_matrix((1,), primaries=primaries, dps=DPS),
    }

    for label in range(2, SU2_26.level + 1):
        matrices[(label,)] = _matrix_subtract(
            _matrix_product(matrices[(1,)], matrices[(label - 1,)]),
            matrices[(label - 2,)],
        )
        _assert_nonnegative_integer_matrix(matrices[(label,)])

    return primaries, tuple(matrices[primary] for primary in primaries)


def _su3_fusion_matrices() -> tuple[tuple[tuple[int, ...], ...], tuple[IntegerMatrix, ...]]:
    primaries = SU3_8.integrable_weights()
    matrices: dict[tuple[int, ...], IntegerMatrix] = {
        (0, 0): _identity_matrix(len(primaries)),
        (1, 0): SU3_8.fusion_matrix((1, 0), primaries=primaries, dps=DPS),
        (0, 1): SU3_8.fusion_matrix((0, 1), primaries=primaries, dps=DPS),
    }

    for total_level in range(2, SU3_8.level + 1):
        boundary_weight = (0, total_level)
        boundary_matrix = _matrix_product(matrices[(0, 1)], matrices[(0, total_level - 1)])
        if total_level >= 2:
            boundary_matrix = _matrix_subtract(boundary_matrix, matrices[(1, total_level - 2)])
        matrices[boundary_weight] = boundary_matrix
        _assert_nonnegative_integer_matrix(boundary_matrix)

        for left_label in range(1, total_level + 1):
            right_label = total_level - left_label
            matrix = _matrix_product(matrices[(1, 0)], matrices[(left_label - 1, right_label)])
            if left_label >= 2:
                matrix = _matrix_subtract(matrix, matrices[(left_label - 2, right_label + 1)])
            if right_label >= 1:
                matrix = _matrix_subtract(matrix, matrices[(left_label - 1, right_label - 1)])
            matrices[(left_label, right_label)] = matrix
            _assert_nonnegative_integer_matrix(matrix)

    return primaries, tuple(matrices[primary] for primary in primaries)


def _identity_matrix(size: int) -> IntegerMatrix:
    return tuple(tuple(1 if row == column else 0 for column in range(size)) for row in range(size))


def _matrix_product(left: IntegerMatrix, right: IntegerMatrix) -> IntegerMatrix:
    size = len(left)
    rows: list[tuple[int, ...]] = []
    for left_row in left:
        output_row = [0 for _ in range(size)]
        for middle_index, coefficient in enumerate(left_row):
            if coefficient == 0:
                continue
            right_row = right[middle_index]
            for column_index, right_entry in enumerate(right_row):
                if right_entry != 0:
                    output_row[column_index] += coefficient * right_entry
        rows.append(tuple(output_row))
    return tuple(rows)


def _linear_combination(coefficients: tuple[int, ...], matrices: tuple[IntegerMatrix, ...]) -> IntegerMatrix:
    size = len(matrices[0])
    rows = [[0 for _ in range(size)] for _ in range(size)]
    for matrix_index, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        matrix = matrices[matrix_index]
        for row_index, matrix_row in enumerate(matrix):
            output_row = rows[row_index]
            for column_index, matrix_entry in enumerate(matrix_row):
                if matrix_entry != 0:
                    output_row[column_index] += coefficient * matrix_entry
    return tuple(tuple(row) for row in rows)


def _matrix_subtract(left: IntegerMatrix, right: IntegerMatrix) -> IntegerMatrix:
    size = len(left)
    return tuple(
        tuple(left[row][column] - right[row][column] for column in range(size))
        for row in range(size)
    )


def _assert_nonnegative_integer_matrix(matrix: IntegerMatrix) -> None:
    assert all(isinstance(entry, int) and entry >= 0 for row in matrix for entry in row)