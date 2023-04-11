import logging
import math
import numpy as np
from typing import List
from typing import Literal
from typing import Union
from typing import Tuple
from constraint import Problem
from decimal import Decimal


def calculate_matrix_pval(table: np.ndarray) -> float:
    """
    Calculates the p-value corresponding to the likelihood of observing the given distribution.

    :param table: A numpy matrix of n dimensions
    :return: A float representing the p-value
    """
    sum_rows = table.sum(0)
    sum_columns = table.sum(1)
    sum_all = table.sum()
    numerator = np.array([], dtype=np.float64)
    denominator = np.array([], dtype=np.float64)
    for item in sum_rows:
        value = Decimal(math.factorial(item))
        numerator = np.append(numerator, value)
    for item in sum_columns:
        value = Decimal(math.factorial(item))
        numerator = np.append(numerator, value)
    for item in table.flat.__array__():
        value = Decimal(math.factorial(item))
        denominator = np.append(denominator, value)
    value = math.factorial(sum_all)
    denominator = np.append(denominator, value)
    p_value = float(np.prod(numerator) / (np.prod(denominator)))
    return p_value


def dict_to_array(matrix_dict: dict) -> np.ndarray:
    """
    Converts a dictionary into a numpy matrix

    :param matrix_dict: Dictionary where {row_index_column_index:value}
    :return: The corresponding Numpy array
    """
    # Determine the shape of the matrix from the dictionary keys
    max_row = max([idx[0] for idx in matrix_dict.keys()])
    max_col = max([idx[1] for idx in matrix_dict.keys()])
    matrix_shape = (max_row + 1, max_col + 1)
    # Create a matrix of zeros with the determined shape
    matrix = np.zeros(matrix_shape)
    # Fill in the values from the dictionary
    for idx, val in matrix_dict.items():
        matrix[idx] = val
    return matrix


def get_possible_tables(table: np.ndarray) -> List[np.ndarray]:
    """
    Calculates all possible permutations of observation of the matrix, still maintaining row and column sum

    :param table: A symmetric numpy matrix of n dimensions.
    :return: A list of all possible permutations of observations
    """
    problem = Problem()
    sum_rows = table.sum(axis=1)
    sum_columns = table.sum(axis=0)
    for i, row in enumerate(table):
        for j, column in enumerate(row):
            arr = list(range(min(sum_columns[j], sum_rows[i]) + 1))
            problem.addVariable((i, j), arr)
    if table.shape == (2, 2):
        # Assuming
        # [
        # [00, 01],
        # [10, 11]
        # ]
        problem.addConstraint(lambda a, b: sum([a, b]) == sum_rows[0], ((0, 0), (0, 1)))
        problem.addConstraint(lambda a, b: sum([a, b]) == sum_rows[1], ((1, 0), (1, 1)))

        problem.addConstraint(lambda a, b: sum([a, b]) == sum_columns[0], ((0, 0), (1, 0)))
        problem.addConstraint(lambda a, b: sum([a, b]) == sum_columns[1], ((0, 1), (1, 1)))
    elif table.shape == (3, 3):
        # Assuming
        # [
        # [00, 01, 02],
        # [10, 11, 12]
        # [20, 21, 22]
        # ]
        problem.addConstraint(lambda a, b, c: sum([a, b, c]) == sum_rows[0], ((0, 0), (0, 1), (0, 2)))
        problem.addConstraint(lambda a, b, c: sum([a, b, c]) == sum_rows[1], ((1, 0), (1, 1), (1, 2)))
        problem.addConstraint(lambda a, b, c: sum([a, b, c]) == sum_rows[2], ((2, 0), (2, 1), (2, 2)))

        problem.addConstraint(lambda a, b, c: sum([a, b, c]) == sum_columns[0], ((0, 0), (1, 0), (2, 0)))
        problem.addConstraint(lambda a, b, c: sum([a, b, c]) == sum_columns[1], ((0, 1), (1, 1), (2, 1)))
        problem.addConstraint(lambda a, b, c: sum([a, b, c]) == sum_columns[2], ((0, 2), (1, 2), (2, 2)))
    elif table.shape == (4, 4):
        # Assuming
        # [
        # [00, 01, 02, 03],
        # [10, 11, 12, 13],
        # [20, 21, 22, 23],
        # [30, 31, 32, 33]
        # ]
        problem.addConstraint(lambda a, b, c, d: sum([a, b, c, d]) == sum_rows[0], ((0, 0), (0, 1), (0, 2), (0, 3)))
        problem.addConstraint(lambda a, b, c, d: sum([a, b, c, d]) == sum_rows[1], ((1, 0), (1, 1), (1, 2), (1, 3)))
        problem.addConstraint(lambda a, b, c, d: sum([a, b, c, d]) == sum_rows[2], ((2, 0), (2, 1), (2, 2), (2, 3)))
        problem.addConstraint(lambda a, b, c, d: sum([a, b, c, d]) == sum_rows[3], ((3, 0), (3, 1), (3, 2), (3, 3)))

        problem.addConstraint(lambda a, b, c, d: sum([a, b, c, d]) == sum_columns[0], ((0, 0), (1, 0), (2, 0), (3, 0)))
        problem.addConstraint(lambda a, b, c, d: sum([a, b, c, d]) == sum_columns[1], ((0, 1), (1, 1), (2, 1), (3, 1)))
        problem.addConstraint(lambda a, b, c, d: sum([a, b, c, d]) == sum_columns[2], ((0, 2), (1, 2), (2, 2), (3, 2)))
        problem.addConstraint(lambda a, b, c, d: sum([a, b, c, d]) == sum_columns[3], ((0, 3), (1, 3), (2, 3), (3, 3)))
    solutions = problem.getSolutions()
    valid_matrixes = []
    for solution in solutions:
        valid_matrix = dict_to_array(solution)
        valid_matrixes.append(valid_matrix)
    return valid_matrixes


def calculate_odds_ratio(table: np.ndarray) -> float:
    """
    Calculates the odds ratio of a given symmetric matrix.

    :param table:  A symmetric numpy matrix of n dimensions.
    :return: The calculated odds ratio
    """
    diagonal_product = []
    off_diagonal_product = []
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if i == j:
                diagonal_product.append(table[i][j])
            else:
                off_diagonal_product.append(table[i][j])
    odds_ratio = np.prod(diagonal_product) / np.prod(off_diagonal_product)
    return odds_ratio


def multiclass_fisher_exact(table: Union[List[list], np.array],
                            alternative: Literal["two-sided", "less", "greater"] = 'two-sided',
                            nan_policy: Literal["propagate", "assume-zero", "raise"] = 'propagate') -> Tuple[
    float, float]:
    """
    Main function to run the fishers test on symmetrical matrix, returning a tuple of the pval and the odds ratio

    :param table: The symmetric contingency table of counts. It can be a list of lists or a numpy array.
    :param alternative: This parameter specifies the alternative hypothesis to be tested. Valid options are 'two-sided', 'less', or 'greater'.
    :param nan_policy:Specifies how to handle the presence of NaN values in the input data. Valid options are 'propagate', 'assume-zero', or 'raise'.
    :return: A tuple containing the computed odds ratio and p-value. The odds ratio is a float and the p-value is a float or NaN.
    """
    # if not passed a np.array
    if isinstance(table, list):
        logging.debug('Passed a list, converting to numpy array')
        table = np.array(table)
    # if not symmetrical
    if table.shape[0] != table.shape[1]:
        logging.error(f'Matrix has shape of {table.shape}, data must be symmetrical')
        raise ValueError
    # warning that this calculation is slow
    if table.shape[0] == 4:
        logging.warning('Matrices of 4 rows/columns are slow to run')
    # only certain dimensions are supported
    if table.shape[0] not in [2, 3, 4]:
        logging.warning(
            f'Currently only Matrices of {[2, 3, 4]} rows/columns are supported and you passed {table.shape}')
    # handling of negative values
    if np.any(table < 0):
        logging.error("All values in `table` must be non-negative.")
        raise ValueError
    # NaN handling
    if np.any(table == np.nan):
        if nan_policy == 'raise':
            logging.error(f'{nan_policy=} and array contains NaN values')
            raise ValueError
        elif nan_policy == 'propagate':
            logging.warning(f'{nan_policy=} and array contains NaN values')
            raise (np.nan, np.nan)
        elif nan_policy == 'assume-zero':
            logging.warning(f'{nan_policy=} and array contains NaN values, assuming these are 0')
            table[np.isnan(table)] = 0
    pval = calculate_matrix_pval(table.astype(int))
    tables = get_possible_tables(table)
    p_values = []
    for single_table in tables:
        p_values.append(calculate_matrix_pval(single_table.astype(int)))
    p_values = np.sort(p_values)
    # for greater add on those p values greater than the observed
    if alternative == 'greater':
        p_values_allowed = list(set([x for x in p_values if x <= pval]))
        final_pval = np.sum(p_values_allowed)
        logging.warning(f'It is recommended to use two-sided rather than {alternative}')
    # for less add on those p values less than the observed
    elif alternative == 'less':
        p_values_allowed = list(set([x for x in p_values if x >= pval]))
        final_pval = np.sum(p_values_allowed)
        logging.warning(f'It is recommended to use two-sided rather than {alternative}')
    # for two_sided add on those p values greater on either side that are more extreme
    elif alternative == 'two-sided':
        p_values_allowed = [x for x in p_values if x <= pval]
        final_pval = np.sum(p_values_allowed)
    else:
        logging.error(f'{alternative=} passed, must be one of {Literal["two-sided", "less", "greater"]}')
        raise ValueError
    odds_ratio = calculate_odds_ratio(table)
    return odds_ratio, final_pval
