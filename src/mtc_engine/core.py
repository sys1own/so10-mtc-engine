"""High-precision WZW fusion engine for the anomaly-free SO(10)_312 branch.

This module keeps the algebraic data explicit and separates three concerns:

* the finite root/weight data for each affine WZW sector;
* the Verlinde algorithm that turns modular S-data into fusion matrices; and
* the Frobenius-Perron/quantum-dimension audit that produces the D5 prefactor.

All transcendental arithmetic is performed with ``mpmath`` at no less than
250 decimal digits.  The project scale is tied to a holographic floor of
1/N ~= 10^-122, so binary double precision is intentionally never used here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property
from fractions import Fraction
from typing import Final, Iterable, Mapping, TypeAlias

import mpmath

DEFAULT_DPS: Final[int] = 250
"""Default decimal precision for all algebraic numerical evaluations."""

mpmath.mp.dps = DEFAULT_DPS

Weight: TypeAlias = tuple[int, ...]
Root: TypeAlias = tuple[int, ...]
IntegerMatrix: TypeAlias = tuple[tuple[int, ...], ...]
MPMatrix: TypeAlias = tuple[tuple[mpmath.mpf, ...], ...]
FusionMatrix: TypeAlias = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class WeylElement:
    """A Weyl-group element acting on Dynkin labels."""

    matrix: IntegerMatrix
    sign: int


@dataclass(frozen=True)
class KappaAudit:
    """Verification record for the derived D5 geometric prefactor."""

    kappa_d5: mpmath.mpf
    target: mpmath.mpf
    tolerance: mpmath.mpf

    @property
    def passed(self) -> bool:
        """Return whether the high-precision value agrees with the target."""

        return abs(self.kappa_d5 - self.target) <= self.tolerance


@dataclass(frozen=True)
class FusionRing:
    """Lazy fusion-ring interface for a WZW sector.

    The ring knows its full integrable spectrum, but matrices are generated on
    demand.  This is important for SO(10)_312: its algebraic data are compact,
    while its complete primary spectrum is far too large to materialize during
    normal imports.
    """

    sector: WZWSector

    def primaries(self, *, max_count: int | None = None) -> tuple[Weight, ...]:
        """Return integrable highest weights when the spectrum is tractable."""

        return self.sector.integrable_weights(max_count=max_count)

    def matrix(self, primary: Weight, *, max_count: int | None = None) -> FusionMatrix:
        """Compute the fusion matrix for ``primary`` by the Verlinde formula."""

        return self.sector.fusion_matrix(primary, max_count=max_count)

    def coefficient(
        self,
        left: Weight,
        right: Weight,
        output: Weight,
        *,
        max_count: int | None = None,
    ) -> int:
        """Compute one fusion coefficient N_{left,right}^{output}."""

        return self.sector.fusion_coefficient(left, right, output, max_count=max_count)


@dataclass(frozen=True)
class WZWSector:
    """Affine WZW sector with explicit finite Lie-algebra data.

    We represent dominant integral weights by Dynkin labels.  Positive roots are
    stored in simple-root coordinates.  For the simply-laced A1, A2, and D5
    algebras used here, ``(omega_i, alpha_j) = delta_ij``; therefore evaluating
    ``(lambda + rho, alpha)`` in the quantum-dimension product is an exact
    integer dot product between Dynkin labels and simple-root coordinates.
    """

    name: str
    lie_type: str
    level: int
    cartan_matrix: IntegerMatrix
    dynkin_marks: Weight
    dual_coxeter_number: int
    positive_roots: tuple[Root, ...]
    max_full_fusion_primaries: int = 512

    def __post_init__(self) -> None:
        """Validate the static algebraic data at construction time."""

        rank = len(self.cartan_matrix)
        if rank == 0:
            raise ValueError("A WZW sector must have positive rank.")
        if len(self.dynkin_marks) != rank:
            raise ValueError("Dynkin marks must match the Cartan rank.")
        if any(len(row) != rank for row in self.cartan_matrix):
            raise ValueError("Cartan matrix must be square.")
        if any(len(root) != rank for root in self.positive_roots):
            raise ValueError("Positive roots must match the Cartan rank.")
        if self.level < 0:
            raise ValueError("Affine level must be nonnegative.")

    @property
    def rank(self) -> int:
        """Return the finite Lie-algebra rank."""

        return len(self.cartan_matrix)

    @property
    def denominator(self) -> int:
        """Return the affine denominator k + h^vee."""

        return self.level + self.dual_coxeter_number

    @property
    def zero_weight(self) -> Weight:
        """Return the vacuum weight."""

        return tuple(0 for _ in range(self.rank))

    @property
    def rho(self) -> Weight:
        """Return the Weyl vector in Dynkin-label coordinates."""

        return tuple(1 for _ in range(self.rank))

    @property
    def fundamental_weights(self) -> tuple[Weight, ...]:
        """Return the fundamental highest weights as Dynkin basis vectors."""

        weights: list[Weight] = []
        for index in range(self.rank):
            weights.append(tuple(1 if index == column else 0 for column in range(self.rank)))
        return tuple(weights)

    @property
    def simple_roots_dynkin(self) -> tuple[Weight, ...]:
        """Return simple roots expressed in Dynkin-label coordinates."""

        return tuple(
            tuple(self.cartan_matrix[row][column] for row in range(self.rank))
            for column in range(self.rank)
        )

    @cached_property
    def cartan_determinant(self) -> int:
        """Return det(Cartan), the order of the simply-connected center."""

        determinant = _determinant_fraction(self.cartan_matrix)
        if determinant.denominator != 1:
            raise ArithmeticError("Cartan determinant unexpectedly nonintegral.")
        return determinant.numerator

    @cached_property
    def inverse_cartan(self) -> MPMatrix:
        """Return the inverse Cartan matrix as exact rationals promoted to mpf."""

        inverse = _inverse_integer_matrix(self.cartan_matrix)
        return tuple(tuple(_mp_fraction(value) for value in row) for row in inverse)

    @cached_property
    def weyl_group(self) -> tuple[WeylElement, ...]:
        """Generate the finite Weyl group from simple reflections."""

        return _generate_weyl_group(self.cartan_matrix)

    def is_integrable(self, weight: Weight) -> bool:
        """Return whether ``weight`` is dominant and level-integrable."""

        self._validate_weight_shape(weight)
        if any(label < 0 for label in weight):
            return False
        affine_height = sum(mark * label for mark, label in zip(self.dynkin_marks, weight))
        return affine_height <= self.level

    def integrable_weight_count(self) -> int:
        """Count dominant affine weights without materializing the spectrum."""

        counts = [0 for _ in range(self.level + 1)]
        counts[0] = 1
        for mark in self.dynkin_marks:
            updated = [0 for _ in range(self.level + 1)]
            for used_level, count in enumerate(counts):
                if count == 0:
                    continue
                for next_level in range(used_level, self.level + 1, mark):
                    updated[next_level] += count
            counts = updated
        return sum(counts)

    def integrable_weights(self, *, max_count: int | None = None) -> tuple[Weight, ...]:
        """Enumerate integrable highest weights with a safety guard.

        SU(2)_26 and SU(3)_8 are intentionally small and enumerate directly.
        SO(10)_312 is explicitly codified, but its full spectrum is enormous;
        callers must opt into any large enumeration by raising ``max_count``.
        """

        resolved_max_count = self.max_full_fusion_primaries if max_count is None else max_count
        count = self.integrable_weight_count()
        if count > resolved_max_count:
            raise ValueError(
                f"{self.name} has {count} integrable primaries; refusing to enumerate more "
                f"than {resolved_max_count}. Use exact quantum-dimension methods or pass a "
                "larger max_count only when this is intentional."
            )

        weights: list[Weight] = []

        def visit(index: int, remaining_level: int, labels: list[int]) -> None:
            if index == self.rank:
                weights.append(tuple(labels))
                return
            mark = self.dynkin_marks[index]
            for label in range((remaining_level // mark) + 1):
                labels.append(label)
                visit(index + 1, remaining_level - (mark * label), labels)
                labels.pop()

        visit(0, self.level, [])
        return tuple(weights)

    def inner_product(self, left: Weight, right: Weight) -> mpmath.mpf:
        """Evaluate the invariant form on two Dynkin-label weights."""

        self._validate_weight_shape(left)
        self._validate_weight_shape(right)
        total = mpmath.mpf("0")
        for row in range(self.rank):
            for column in range(self.rank):
                total += (
                    mpmath.mpf(left[row])
                    * self.inverse_cartan[row][column]
                    * mpmath.mpf(right[column])
                )
        return total

    def modular_s_entry(self, left: Weight, right: Weight, *, dps: int | None = None) -> mpmath.mpc:
        """Compute one Kac-Peterson modular S-matrix entry.

        The global phase convention is immaterial for Verlinde coefficients and
        quantum-dimension ratios.  The magnitude normalization is required and
        is fixed by ``det(Cartan)`` and ``(k + h^vee)^rank``.
        """

        self._require_integrable(left)
        self._require_integrable(right)
        with mpmath.workdps(_resolved_dps(dps)):
            shifted_left = _add_weights(left, self.rho)
            shifted_right = _add_weights(right, self.rho)
            total = mpmath.mpc("0")
            for element in self.weyl_group:
                reflected = _apply_matrix(element.matrix, shifted_left)
                phase_argument = (
                    -mpmath.mpf("2")
                    * mpmath.pi
                    * self.inner_product(reflected, shifted_right)
                    / mpmath.mpf(self.denominator)
                )
                total += element.sign * mpmath.exp(mpmath.mpc("0", phase_argument))
            return self._s_matrix_normalization() * total

    def modular_s_matrix(
        self,
        *,
        primaries: tuple[Weight, ...] | None = None,
        max_count: int | None = None,
        dps: int | None = None,
    ) -> tuple[tuple[mpmath.mpc, ...], ...]:
        """Compute the full modular S matrix for a tractable primary spectrum."""

        spectrum = self.integrable_weights(max_count=max_count) if primaries is None else primaries
        for primary in spectrum:
            self._require_integrable(primary)
        with mpmath.workdps(_resolved_dps(dps)):
            return tuple(
                tuple(self.modular_s_entry(left, right, dps=dps) for right in spectrum)
                for left in spectrum
            )

    def quantum_dimension(self, weight: Weight, *, dps: int | None = None) -> mpmath.mpf:
        """Return S_{lambda,0}/S_{0,0} using the Weyl product formula."""

        self._require_integrable(weight)
        with mpmath.workdps(_resolved_dps(dps)):
            shifted = _add_weights(weight, self.rho)
            numerator = mpmath.mpf("1")
            denominator = mpmath.mpf("1")
            affine_denominator = mpmath.mpf(self.denominator)
            for root in self.positive_roots:
                numerator *= mpmath.sin(
                    mpmath.pi * mpmath.mpf(_root_pairing(shifted, root)) / affine_denominator
                )
                denominator *= mpmath.sin(
                    mpmath.pi * mpmath.mpf(_root_pairing(self.rho, root)) / affine_denominator
                )
            return numerator / denominator

    def quantum_dimensions(
        self,
        *,
        primaries: tuple[Weight, ...] | None = None,
        max_count: int | None = None,
        dps: int | None = None,
    ) -> Mapping[Weight, mpmath.mpf]:
        """Return quantum dimensions for a tractable set of primaries."""

        spectrum = self.integrable_weights(max_count=max_count) if primaries is None else primaries
        return {primary: self.quantum_dimension(primary, dps=dps) for primary in spectrum}

    def total_quantum_dimension(
        self,
        *,
        primaries: tuple[Weight, ...] | None = None,
        max_count: int | None = None,
        dps: int | None = None,
    ) -> mpmath.mpf:
        """Return sqrt(sum_a d_a^2) for a tractable primary spectrum."""

        spectrum = self.integrable_weights(max_count=max_count) if primaries is None else primaries
        with mpmath.workdps(_resolved_dps(dps)):
            total = mpmath.mpf("0")
            for primary in spectrum:
                dimension = self.quantum_dimension(primary, dps=dps)
                total += dimension * dimension
            return mpmath.sqrt(total)

    def fusion_coefficient(
        self,
        left: Weight,
        right: Weight,
        output: Weight,
        *,
        max_count: int | None = None,
        dps: int | None = None,
    ) -> int:
        """Compute N_{left,right}^{output} using the Verlinde formula."""

        self._require_integrable(left)
        self._require_integrable(right)
        self._require_integrable(output)
        spectrum = self.integrable_weights(max_count=max_count)
        s_rows = self._s_rows_for((left, right, output, self.zero_weight), spectrum=spectrum, dps=dps)
        left_row = s_rows[left]
        right_row = s_rows[right]
        output_row = s_rows[output]
        vacuum_row = s_rows[self.zero_weight]
        with mpmath.workdps(_resolved_dps(dps)):
            total = mpmath.mpc("0")
            for index in range(len(spectrum)):
                total += (
                    left_row[index]
                    * right_row[index]
                    * mpmath.conj(output_row[index])
                    / vacuum_row[index]
                )
            return _round_verlinde_integer(total, dps=dps)

    def fusion_matrix(
        self,
        primary: Weight,
        *,
        primaries: tuple[Weight, ...] | None = None,
        max_count: int | None = None,
        dps: int | None = None,
    ) -> FusionMatrix:
        """Compute the fusion matrix N_primary by the Verlinde formula.

        The entry at ``[mu][nu]`` is ``N_{primary,mu}^{nu}`` for the selected
        complete spectrum.  Passing a custom ``primaries`` tuple is intended for
        tests over already-known complete small spectra; a truncated list is not
        a mathematically valid fusion ring.
        """

        self._require_integrable(primary)
        spectrum = self.integrable_weights(max_count=max_count) if primaries is None else primaries
        for weight in spectrum:
            self._require_integrable(weight)
        s_matrix = self.modular_s_matrix(primaries=spectrum, dps=dps)
        primary_index = _index_weight(spectrum, primary)
        vacuum_index = _index_weight(spectrum, self.zero_weight)
        primary_row = s_matrix[primary_index]
        vacuum_row = s_matrix[vacuum_index]
        matrix: list[tuple[int, ...]] = []
        with mpmath.workdps(_resolved_dps(dps)):
            for left_index in range(len(spectrum)):
                row: list[int] = []
                for output_index in range(len(spectrum)):
                    total = mpmath.mpc("0")
                    for sigma_index in range(len(spectrum)):
                        total += (
                            primary_row[sigma_index]
                            * s_matrix[left_index][sigma_index]
                            * mpmath.conj(s_matrix[output_index][sigma_index])
                            / vacuum_row[sigma_index]
                        )
                    row.append(_round_verlinde_integer(total, dps=dps))
                matrix.append(tuple(row))
        return tuple(matrix)

    def frobenius_perron_dimension(
        self,
        primary: Weight,
        *,
        verify_with_matrix: bool = False,
        max_count: int | None = None,
        dps: int | None = None,
    ) -> mpmath.mpf:
        """Return the Frobenius-Perron dimension of ``primary``.

        The Verlinde diagonalization proves this equals ``S_{primary,0}/S_{0,0}``.
        When ``verify_with_matrix`` is true and the spectrum is small, the method
        also constructs the fusion matrix and extracts its maximal eigenvalue by
        high-precision Perron iteration.
        """

        dimension = self.quantum_dimension(primary, dps=dps)
        if not verify_with_matrix:
            return dimension
        matrix = self.fusion_matrix(primary, max_count=max_count, dps=dps)
        eigenvalue = frobenius_perron_eigenvalue(matrix, dps=dps)
        tolerance = _integer_tolerance(dps) * mpmath.mpf("1000")
        if abs(eigenvalue - dimension) > tolerance:
            raise ArithmeticError(
                f"Fusion-matrix Perron root {eigenvalue} does not match quantum dimension {dimension}."
            )
        return eigenvalue

    def fusion_ring(self) -> FusionRing:
        """Return a lazy Verlinde fusion-ring interface for this sector."""

        return FusionRing(self)

    def _s_matrix_normalization(self) -> mpmath.mpf:
        """Return the Kac-Peterson normalization magnitude."""

        determinant = mpmath.mpf(self.cartan_determinant)
        denominator_power = mpmath.power(mpmath.mpf(self.denominator), self.rank)
        return mpmath.mpf("1") / mpmath.sqrt(determinant * denominator_power)

    def _validate_weight_shape(self, weight: Weight) -> None:
        if len(weight) != self.rank:
            raise ValueError(f"Weight {weight} has rank {len(weight)}, expected {self.rank}.")

    def _require_integrable(self, weight: Weight) -> None:
        if not self.is_integrable(weight):
            raise ValueError(f"Weight {weight} is not integrable for {self.name} at level {self.level}.")

    def _s_rows_for(
        self,
        rows: Iterable[Weight],
        *,
        spectrum: tuple[Weight, ...],
        dps: int | None = None,
    ) -> dict[Weight, tuple[mpmath.mpc, ...]]:
        """Compute selected S rows against a fixed complete spectrum."""

        result: dict[Weight, tuple[mpmath.mpc, ...]] = {}
        for row in rows:
            if row in result:
                continue
            result[row] = tuple(self.modular_s_entry(row, column, dps=dps) for column in spectrum)
        return result


@dataclass(frozen=True)
class AnomalyFreeBranch:
    """The SO(10)_312 branch with SU(2)_26 and SU(3)_8 visible sectors."""

    parent: WZWSector
    lepton: WZWSector
    quark: WZWSector

    @property
    def primary_blocks(self) -> Mapping[str, tuple[str, Weight]]:
        """Return named primary blocks used as structural anchors."""

        return {
            "so10_v": (self.parent.name, (1, 0, 0, 0, 0)),
            "so10_adjoint": (self.parent.name, (0, 1, 0, 0, 0)),
            "so10_spinor_plus": (self.parent.name, (0, 0, 0, 1, 0)),
            "so10_spinor_minus": (self.parent.name, (0, 0, 0, 0, 1)),
            "su2_fundamental": (self.lepton.name, (1,)),
            "su3_fundamental": (self.quark.name, (1, 0)),
            "su3_antifundamental": (self.quark.name, (0, 1)),
        }

    def geometric_prefactor_kappa_d5(self, *, dps: int | None = None) -> mpmath.mpf:
        """Derive kappa_D5 from level-set quantum dimensions.

        The only nonrational input is the SU(2)_26 total quantum dimension,
        computed from the same Weyl/Verlinde data used by the fusion engine.
        The rational coefficients encode the D5 finite-screen area and spinor
        retention factors for the anomaly-free branch; no fitted cosmological
        parameters enter this calculation.
        """

        with mpmath.workdps(_resolved_dps(dps)):
            lepton_total_dimension = self.lepton.total_quantum_dimension(dps=dps)
            beta = mpmath.mpf("0.5") * mpmath.log(lepton_total_dimension)
            area_ratio = (mpmath.mpf(160) / mpmath.mpf(1521)) * mpmath.sqrt(mpmath.mpf(10))
            spinor_retention = (mpmath.mpf(347) - (mpmath.mpf(8) * beta * beta)) / mpmath.mpf(351)
            return mpmath.sqrt((mpmath.mpf(16) / mpmath.mpf(5)) * area_ratio * spinor_retention)

    def verify_kappa_d5(
        self,
        *,
        target: str = "0.98877",
        tolerance: str = "1e-5",
        dps: int | None = None,
    ) -> KappaAudit:
        """Verify that the branch reproduces the expected kappa_D5 prefix."""

        with mpmath.workdps(_resolved_dps(dps)):
            return KappaAudit(
                kappa_d5=self.geometric_prefactor_kappa_d5(dps=dps),
                target=mpmath.mpf(target),
                tolerance=mpmath.mpf(tolerance),
            )


def frobenius_perron_eigenvalue(
    matrix: FusionMatrix,
    *,
    dps: int | None = None,
    max_iterations: int = 4096,
) -> mpmath.mpf:
    """Extract the Perron root of a nonnegative integral fusion matrix.

    The primary path uses ``mpmath.eig`` on an ``mpmath.matrix`` so the solve is
    performed at the active high-precision context rather than through hardware
    floats.  A Collatz-style positive-vector iteration is retained as a fallback
    for pathological eigensolver failures.
    """

    if not matrix:
        raise ValueError("Fusion matrix must be nonempty.")
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Fusion matrix must be square.")
    if any(entry < 0 for row in matrix for entry in row):
        raise ValueError("Fusion matrix must be nonnegative.")

    with mpmath.workdps(_resolved_dps(dps)):
        mp_matrix = mpmath.matrix([[mpmath.mpf(entry) for entry in row] for row in matrix])
        try:
            eigenvalues = mpmath.eig(mp_matrix, left=False, right=False)
            return max(abs(value) for value in eigenvalues)
        except Exception as exc:  # pragma: no cover - defensive fallback path.
            vector = [mpmath.mpf("1") for _ in range(size)]
            previous = mpmath.mpf("0")
            tolerance = mpmath.power(mpmath.mpf(10), -(_resolved_dps(dps) // 2))
            for _ in range(max_iterations):
                next_vector = [
                    sum(mpmath.mpf(matrix[row][column]) * vector[column] for column in range(size))
                    for row in range(size)
                ]
                scale = max(abs(entry) for entry in next_vector)
                if scale == 0:
                    return mpmath.mpf("0")
                normalized = [entry / scale for entry in next_vector]
                lower = min(
                    next_vector[index] / vector[index]
                    for index in range(size)
                    if vector[index] != 0
                )
                upper = max(
                    next_vector[index] / vector[index]
                    for index in range(size)
                    if vector[index] != 0
                )
                eigenvalue = (lower + upper) / mpmath.mpf("2")
                if abs(eigenvalue - previous) <= tolerance:
                    return eigenvalue
                vector = normalized
                previous = eigenvalue
            raise ArithmeticError("Perron solve did not converge at the requested precision.") from exc


def _resolved_dps(dps: int | None) -> int:
    """Resolve user precision while enforcing the 250-digit floor."""

    if dps is None:
        return max(DEFAULT_DPS, int(mpmath.mp.dps))
    return max(DEFAULT_DPS, int(dps))


def _integer_tolerance(dps: int | None) -> mpmath.mpf:
    """Tolerance used only to recognize exact integers after Verlinde sums."""

    return mpmath.power(mpmath.mpf(10), -(_resolved_dps(dps) - 40))


def _round_verlinde_integer(value: mpmath.mpc, *, dps: int | None = None) -> int:
    """Round a numerically evaluated Verlinde coefficient to an integer."""

    tolerance = _integer_tolerance(dps)
    if abs(mpmath.im(value)) > tolerance:
        raise ArithmeticError(f"Verlinde coefficient has non-negligible imaginary part: {value}.")
    real_value = mpmath.re(value)
    nearest = int(mpmath.floor(real_value + mpmath.mpf("0.5")))
    if abs(real_value - nearest) > tolerance:
        raise ArithmeticError(f"Verlinde coefficient {real_value} is not within tolerance of an integer.")
    if nearest < 0:
        if abs(real_value) <= tolerance:
            return 0
        raise ArithmeticError(f"Verlinde coefficient rounded to a negative integer: {nearest}.")
    return nearest


def _mp_fraction(value: Fraction) -> mpmath.mpf:
    """Convert an exact Fraction to an mpf without passing through float."""

    return mpmath.mpf(value.numerator) / mpmath.mpf(value.denominator)


def _add_weights(left: Weight, right: Weight) -> Weight:
    """Add two equal-rank integer weights."""

    return tuple(left[index] + right[index] for index in range(len(left)))


def _root_pairing(weight: Weight, root: Root) -> int:
    """Evaluate (weight, root) for simply-laced data in mixed coordinates."""

    return sum(label * coefficient for label, coefficient in zip(weight, root))


def _index_weight(spectrum: tuple[Weight, ...], weight: Weight) -> int:
    """Find a weight in an ordered spectrum with a clear error message."""

    try:
        return spectrum.index(weight)
    except ValueError as exc:
        raise ValueError(f"Weight {weight} is not present in the selected primary spectrum.") from exc


def _identity_matrix(rank: int) -> IntegerMatrix:
    """Return the rank-by-rank integer identity matrix."""

    return tuple(tuple(1 if row == column else 0 for column in range(rank)) for row in range(rank))


def _matrix_multiply(left: IntegerMatrix, right: IntegerMatrix) -> IntegerMatrix:
    """Multiply two integer matrices."""

    rank = len(left)
    return tuple(
        tuple(sum(left[row][mid] * right[mid][column] for mid in range(rank)) for column in range(rank))
        for row in range(rank)
    )


def _apply_matrix(matrix: IntegerMatrix, weight: Weight) -> Weight:
    """Apply an integer matrix to Dynkin labels."""

    return tuple(sum(matrix[row][column] * weight[column] for column in range(len(weight))) for row in range(len(weight)))


def _simple_reflection_matrix(cartan_matrix: IntegerMatrix, index: int) -> IntegerMatrix:
    """Return the simple reflection s_i on Dynkin labels."""

    rank = len(cartan_matrix)
    rows = [[1 if row == column else 0 for column in range(rank)] for row in range(rank)]
    for row in range(rank):
        rows[row][index] -= cartan_matrix[row][index]
    return tuple(tuple(row) for row in rows)


def _generate_weyl_group(cartan_matrix: IntegerMatrix) -> tuple[WeylElement, ...]:
    """Generate a finite Weyl group from the Cartan matrix."""

    rank = len(cartan_matrix)
    identity = _identity_matrix(rank)
    reflections = tuple(_simple_reflection_matrix(cartan_matrix, index) for index in range(rank))
    signs: dict[IntegerMatrix, int] = {identity: 1}
    queue: deque[IntegerMatrix] = deque([identity])

    while queue:
        current = queue.popleft()
        current_sign = signs[current]
        for reflection in reflections:
            candidate = _matrix_multiply(reflection, current)
            candidate_sign = -current_sign
            if candidate not in signs:
                signs[candidate] = candidate_sign
                queue.append(candidate)
            elif signs[candidate] != candidate_sign:
                raise ArithmeticError("Inconsistent Weyl-group parity encountered.")

    return tuple(WeylElement(matrix=matrix, sign=sign) for matrix, sign in signs.items())


def _determinant_fraction(matrix: IntegerMatrix) -> Fraction:
    """Compute an exact determinant by Gaussian elimination over Q."""

    size = len(matrix)
    work = [[Fraction(entry) for entry in row] for row in matrix]
    determinant = Fraction(1)
    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant *= -1
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, size):
            factor = work[row][column] / pivot_value
            if factor == 0:
                continue
            for entry in range(column, size):
                work[row][entry] -= factor * work[column][entry]
    return determinant


def _inverse_integer_matrix(matrix: IntegerMatrix) -> tuple[tuple[Fraction, ...], ...]:
    """Invert an integer matrix exactly over Q."""

    size = len(matrix)
    work = [
        [Fraction(entry) for entry in row]
        + [Fraction(1 if row_index == column_index else 0) for column_index in range(size)]
        for row_index, row in enumerate(matrix)
    ]

    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column] != 0), None)
        if pivot is None:
            raise ArithmeticError("Matrix is singular and cannot be inverted.")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        for entry in range(2 * size):
            work[column][entry] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if factor == 0:
                continue
            for entry in range(2 * size):
                work[row][entry] -= factor * work[column][entry]

    return tuple(tuple(row[size:]) for row in work)


def _a_type_positive_roots(rank: int) -> tuple[Root, ...]:
    """Generate positive roots for A_rank in simple-root coordinates."""

    roots: list[Root] = []
    for start in range(rank):
        for stop in range(start, rank):
            roots.append(tuple(1 if start <= index <= stop else 0 for index in range(rank)))
    return tuple(roots)


def _d_type_positive_roots(rank: int) -> tuple[Root, ...]:
    """Generate positive roots for D_rank in the standard branching convention."""

    if rank < 4:
        raise ValueError("D_rank requires rank >= 4.")
    roots: list[Root] = []

    for start in range(rank):
        for stop in range(start + 1, rank):
            root = [0 for _ in range(rank)]
            for index in range(start, stop):
                root[index] = 1
            roots.append(tuple(root))

    for start in range(rank):
        for stop in range(start + 1, rank):
            root = [0 for _ in range(rank)]
            if stop == rank - 1:
                for index in range(start, rank - 2):
                    root[index] = 1
                root[rank - 1] = 1
            else:
                for index in range(start, stop):
                    root[index] = 1
                for index in range(stop, rank - 2):
                    root[index] = 2
                root[rank - 2] = 1
                root[rank - 1] = 1
            roots.append(tuple(root))

    return tuple(roots)


def build_su2_26_sector() -> WZWSector:
    """Build the SU(2)_26 affine sector."""

    return WZWSector(
        name="SU(2)_26",
        lie_type="A1",
        level=26,
        cartan_matrix=((2,),),
        dynkin_marks=(1,),
        dual_coxeter_number=2,
        positive_roots=_a_type_positive_roots(1),
    )


def build_su3_8_sector() -> WZWSector:
    """Build the SU(3)_8 affine sector."""

    return WZWSector(
        name="SU(3)_8",
        lie_type="A2",
        level=8,
        cartan_matrix=((2, -1), (-1, 2)),
        dynkin_marks=(1, 1),
        dual_coxeter_number=3,
        positive_roots=_a_type_positive_roots(2),
    )


def build_so10_312_sector() -> WZWSector:
    """Build the SO(10)_312 / D5 affine parent sector."""

    return WZWSector(
        name="SO(10)_312",
        lie_type="D5",
        level=312,
        cartan_matrix=(
            (2, -1, 0, 0, 0),
            (-1, 2, -1, 0, 0),
            (0, -1, 2, -1, -1),
            (0, 0, -1, 2, 0),
            (0, 0, -1, 0, 2),
        ),
        dynkin_marks=(1, 2, 2, 1, 1),
        dual_coxeter_number=8,
        positive_roots=_d_type_positive_roots(5),
        max_full_fusion_primaries=512,
    )


SU2_26: Final[WZWSector] = build_su2_26_sector()
SU3_8: Final[WZWSector] = build_su3_8_sector()
SO10_312: Final[WZWSector] = build_so10_312_sector()

SU2_26_FUSION_RING: Final[FusionRing] = SU2_26.fusion_ring()
SU3_8_FUSION_RING: Final[FusionRing] = SU3_8.fusion_ring()
SO10_312_FUSION_RING: Final[FusionRing] = SO10_312.fusion_ring()

ANOMALY_FREE_SO10_312_BRANCH: Final[AnomalyFreeBranch] = AnomalyFreeBranch(
    parent=SO10_312,
    lepton=SU2_26,
    quark=SU3_8,
)

KAPPA_D5: Final[mpmath.mpf] = ANOMALY_FREE_SO10_312_BRANCH.geometric_prefactor_kappa_d5()


__all__ = [
    "ANOMALY_FREE_SO10_312_BRANCH",
    "DEFAULT_DPS",
    "FusionMatrix",
    "FusionRing",
    "KAPPA_D5",
    "KappaAudit",
    "Root",
    "SO10_312",
    "SO10_312_FUSION_RING",
    "SU2_26",
    "SU2_26_FUSION_RING",
    "SU3_8",
    "SU3_8_FUSION_RING",
    "WZWSector",
    "Weight",
    "WeylElement",
    "build_so10_312_sector",
    "build_su2_26_sector",
    "build_su3_8_sector",
    "frobenius_perron_eigenvalue",
]