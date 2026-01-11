"""Multi-class Fisher's exact test implementation.

This module provides an implementation of Fisher's exact test for contingency
tables larger than 2x2, supporting 2x2, 3x3, and 4x4 matrices.
"""

import logging
import math
from collections.abc import Callable
from decimal import Decimal
from enum import Enum
from functools import lru_cache
from typing import Literal

import numpy as np
from constraint import Problem


@lru_cache(maxsize=512)
def _cached_factorial(n: int) -> Decimal:
    """Cached factorial computation using Decimal for precision."""
    return Decimal(math.factorial(n))

logger = logging.getLogger(__name__)

# Constants
_SUPPORTED_DIMENSIONS: tuple[int, ...] = (2, 3, 4)
_MIN_CELL_COUNT_WARNING: int = 5


class _Alternative(str, Enum):
    """Internal enum for alternative hypothesis (for match-case)."""

    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"


class _NanPolicy(str, Enum):
    """Internal enum for NaN handling policy (for match-case)."""

    PROPAGATE = "propagate"
    ASSUME_ZERO = "assume-zero"
    RAISE = "raise"


def _calculate_matrix_pval(table: np.ndarray) -> float:
    """Calculate p-value for the given contingency table.

    Uses the hypergeometric distribution formula to compute the probability
    of observing the given distribution of counts.

    Parameters
    ----------
    table : np.ndarray
        A 2D contingency table of non-negative integer counts.

    Returns
    -------
    float
        The probability of observing the given distribution.
    """
    sum_rows = table.sum(axis=0)
    sum_columns = table.sum(axis=1)
    sum_all = int(table.sum())

    # Use Python lists instead of np.append (O(n) vs O(n²))
    numerator: list[Decimal] = []
    denominator: list[Decimal] = []

    # Row and column marginal factorials go in numerator
    for item in sum_rows:
        numerator.append(_cached_factorial(int(item)))

    for item in sum_columns:
        numerator.append(_cached_factorial(int(item)))

    # Cell factorials go in denominator
    for item in table.flat:
        denominator.append(_cached_factorial(int(item)))

    # Total factorial goes in denominator
    denominator.append(_cached_factorial(sum_all))

    # Use math.prod for Decimal (faster than np.prod for Python objects)
    p_value = float(math.prod(numerator) / math.prod(denominator))
    return p_value


def _dict_to_array(matrix_dict: dict[tuple[int, int], int | float]) -> np.ndarray:
    """Convert a dictionary into a numpy matrix.

    Parameters
    ----------
    matrix_dict : dict[tuple[int, int], int | float]
        Dictionary mapping (row, column) indices to values.

    Returns
    -------
    np.ndarray
        The corresponding 2D numpy array.
    """
    max_row = max(idx[0] for idx in matrix_dict.keys())
    max_col = max(idx[1] for idx in matrix_dict.keys())
    matrix_shape = (max_row + 1, max_col + 1)

    matrix = np.zeros(matrix_shape)
    for idx, val in matrix_dict.items():
        matrix[idx] = val

    return matrix


def _get_sum_constraint(size: int, expected_sum: int | float) -> Callable[..., bool]:
    """Create a sum constraint function for the given size.

    Parameters
    ----------
    size : int
        Number of variables to sum (2, 3, or 4).
    expected_sum : int | float
        The expected sum of the variables.

    Returns
    -------
    Callable[..., bool]
        A constraint function that returns True if the sum matches.

    Raises
    ------
    ValueError
        If size is not in supported dimensions.
    """
    match size:
        case 2:
            return lambda a, b: a + b == expected_sum
        case 3:
            return lambda a, b, c: a + b + c == expected_sum
        case 4:
            return lambda a, b, c, d: a + b + c + d == expected_sum
        case _:
            raise ValueError(f"Unsupported dimension {size}, must be one of {_SUPPORTED_DIMENSIONS}")


def _add_constraints_for_shape(
    problem: Problem,
    table: np.ndarray,
    sum_rows: np.ndarray,
    sum_columns: np.ndarray,
) -> None:
    """Add row and column sum constraints to the constraint problem.

    Parameters
    ----------
    problem : Problem
        The constraint satisfaction problem to add constraints to.
    table : np.ndarray
        The contingency table.
    sum_rows : np.ndarray
        Sum of each row.
    sum_columns : np.ndarray
        Sum of each column.
    """
    size = table.shape[0]

    # Add row constraints
    for row_idx in range(size):
        variables = tuple((row_idx, col_idx) for col_idx in range(size))
        constraint = _get_sum_constraint(size, sum_rows[row_idx])
        problem.addConstraint(constraint, variables)

    # Add column constraints
    for col_idx in range(size):
        variables = tuple((row_idx, col_idx) for row_idx in range(size))
        constraint = _get_sum_constraint(size, sum_columns[col_idx])
        problem.addConstraint(constraint, variables)


def _get_possible_tables(table: np.ndarray) -> list[np.ndarray]:
    """Calculate all possible tables with the same marginal sums.

    Parameters
    ----------
    table : np.ndarray
        A square numpy matrix of n dimensions.

    Returns
    -------
    list[np.ndarray]
        A list of all possible permutations of observations that maintain
        the same row and column sums as the input table.
    """
    problem = Problem()
    sum_rows = table.sum(axis=1)
    sum_columns = table.sum(axis=0)

    # Add variables for each cell
    for i, row in enumerate(table):
        for j, _ in enumerate(row):
            max_val = int(min(sum_columns[j], sum_rows[i]))
            domain = list(range(max_val + 1))
            problem.addVariable((i, j), domain)

    # Add constraints based on table shape
    _add_constraints_for_shape(problem, table, sum_rows, sum_columns)

    # Convert solutions to arrays
    solutions = problem.getSolutions()
    valid_matrices: list[np.ndarray] = [_dict_to_array(solution) for solution in solutions]

    return valid_matrices


def _calculate_odds_ratio(table: np.ndarray) -> float:
    """Calculate the odds ratio of a given square matrix.

    Parameters
    ----------
    table : np.ndarray
        A square numpy matrix of n dimensions.

    Returns
    -------
    float
        The calculated odds ratio (product of diagonal / product of off-diagonal).
    """
    # Vectorized: extract diagonal and off-diagonal elements
    diagonal = np.diag(table)
    off_diagonal_mask = ~np.eye(table.shape[0], dtype=bool)
    off_diagonal = table[off_diagonal_mask]

    odds_ratio = np.prod(diagonal) / np.prod(off_diagonal)
    return float(odds_ratio)


def _validate_table(
    table: np.ndarray,
    *,
    nan_policy: _NanPolicy,
) -> np.ndarray | None:
    """Validate and prepare the contingency table.

    Parameters
    ----------
    table : np.ndarray
        The contingency table to validate.
    nan_policy : _NanPolicy
        Policy for handling NaN values.

    Returns
    -------
    np.ndarray | None
        The validated (and possibly modified) table, or None if NaN propagation
        should occur.

    Raises
    ------
    ValueError
        If the table is not square, contains negative values, has unsupported
        dimensions, or contains NaN with raise policy.
    """
    # Check for square matrix
    if table.shape[0] != table.shape[1]:
        raise ValueError(f"Table must be square, got shape {table.shape}")

    # Check supported dimensions
    if table.shape[0] not in _SUPPORTED_DIMENSIONS:
        raise ValueError(f"Unsupported dimension {table.shape[0]}, must be one of {_SUPPORTED_DIMENSIONS}")

    # Check for negative values
    if np.any(table < 0):
        raise ValueError("All values in table must be non-negative")

    # Handle NaN values
    if np.isnan(table).any():
        match nan_policy:
            case _NanPolicy.RAISE:
                raise ValueError("Table contains NaN values and nan_policy is 'raise'")
            case _NanPolicy.PROPAGATE:
                logger.warning("Table contains NaN values, propagating NaN result")
                return None
            case _NanPolicy.ASSUME_ZERO:
                logger.warning("Table contains NaN values, assuming these are 0")
                table = table.copy()
                table[np.isnan(table)] = 0

    # Warning for large cell counts
    if table.shape[0] == 4:
        logger.warning("Matrices of 4 rows/columns are slow to compute")

    if np.all(table >= _MIN_CELL_COUNT_WARNING):
        logger.warning(
            "Fisher's exact test is typically used when cell counts are less than %d",
            _MIN_CELL_COUNT_WARNING,
        )

    return table


def _filter_pvalues_by_alternative(
    p_values: np.ndarray,
    observed_pval: float,
    alternative: _Alternative,
) -> float:
    """Filter p-values based on the alternative hypothesis.

    Parameters
    ----------
    p_values : np.ndarray
        Sorted array of p-values from all possible tables.
    observed_pval : float
        The p-value of the observed table.
    alternative : _Alternative
        The alternative hypothesis.

    Returns
    -------
    float
        The final p-value based on the alternative hypothesis.
    """
    match alternative:
        case _Alternative.GREATER:
            # Use np.unique for set behavior (deduplicate)
            p_values_allowed = np.unique(p_values[p_values <= observed_pval])
            logger.warning("It is recommended to use two-sided rather than %s", alternative)
        case _Alternative.LESS:
            # Use np.unique for set behavior (deduplicate)
            p_values_allowed = np.unique(p_values[p_values >= observed_pval])
            logger.warning("It is recommended to use two-sided rather than %s", alternative)
        case _Alternative.TWO_SIDED:
            # No deduplication needed for two-sided
            p_values_allowed = p_values[p_values <= observed_pval]

    return float(np.sum(p_values_allowed))


def multiclass_fisher_exact(
    table: list[list[int | float]] | np.ndarray,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    nan_policy: Literal["propagate", "assume-zero", "raise"] = "propagate",
) -> tuple[float, float]:
    """Perform Fisher's exact test on a square contingency table.

    This function computes the p-value and odds ratio for a contingency table
    using Fisher's exact test. Unlike scipy's implementation, this supports
    tables larger than 2x2 (up to 4x4).

    Parameters
    ----------
    table : list[list[int | float]] | np.ndarray
        The square contingency table of counts. Must be 2x2, 3x3, or 4x4.
    alternative : {"two-sided", "less", "greater"}, optional
        The alternative hypothesis to test. Default is "two-sided".
    nan_policy : {"propagate", "assume-zero", "raise"}, optional
        How to handle NaN values in the input:
        - "propagate": Return (NaN, NaN) if NaN values are present.
        - "assume-zero": Replace NaN values with 0.
        - "raise": Raise a ValueError if NaN values are present.
        Default is "propagate".

    Returns
    -------
    tuple[float, float]
        A tuple of (odds_ratio, p_value).

    Raises
    ------
    ValueError
        If the table is not square, contains negative values, has unsupported
        dimensions, or contains NaN with nan_policy="raise".

    Examples
    --------
    >>> table = [[1, 9], [11, 3]]
    >>> odds_ratio, pval = multiclass_fisher_exact(table)
    """
    # Convert to numpy array if needed
    if isinstance(table, list):
        logger.debug("Passed a list, converting to numpy array")
        table = np.array(table)

    # Convert string parameters to internal enums
    alt_enum = _Alternative(alternative)
    nan_enum = _NanPolicy(nan_policy)

    # Validate the table
    validated_table = _validate_table(table, nan_policy=nan_enum)
    if validated_table is None:
        return (np.nan, np.nan)

    # Calculate p-value for the observed table
    observed_pval = _calculate_matrix_pval(validated_table.astype(int))

    # Get all possible tables with the same marginal sums
    possible_tables = _get_possible_tables(validated_table)

    # Calculate p-values for all possible tables
    all_pvalues = np.array([_calculate_matrix_pval(t.astype(int)) for t in possible_tables])
    sorted_pvalues = np.sort(all_pvalues)

    # Filter p-values based on alternative hypothesis
    final_pval = _filter_pvalues_by_alternative(sorted_pvalues, observed_pval, alt_enum)

    # Calculate odds ratio
    odds_ratio = _calculate_odds_ratio(validated_table)

    return odds_ratio, final_pval


# Keep public aliases for backwards compatibility
calculate_matrix_pval = _calculate_matrix_pval
dict_to_array = _dict_to_array
get_possible_tables = _get_possible_tables
calculate_odds_ratio = _calculate_odds_ratio
