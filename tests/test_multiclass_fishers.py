import numpy as np
import pytest
from scipy.stats import fisher_exact
from multi_class_fishers.multi_class_fishers import calculate_matrix_pval
from multi_class_fishers.multi_class_fishers import dict_to_array
from multi_class_fishers.multi_class_fishers import get_possible_tables
from multi_class_fishers.multi_class_fishers import calculate_odds_ratio
from multi_class_fishers.multi_class_fishers import multiclass_fisher_exact

two_by_two_arrays = [np.array([[1, 9],
                               [11, 3]]),
                     np.array([[0, 10],
                               [12, 2]]),
                     np.array([[10, 15],
                               [20, 25]]),
                     np.array([[12, 8],
                               [16, 22]]),
                     ]
three_by_three_arrays = [np.array([[1, 9, 12],
                                   [11, 3, 90],
                                   [11, 3, 90]]),
                         np.array([[5, 60, 0],
                                   [1, 500, 1],
                                   [0, 10, 1]]),
                         np.array([[3, 4, 2],
                                   [5, 6, 4],
                                   [1, 2, 5]]),
                         np.array([[2, 6, 8],
                                   [5, 1, 3],
                                   [9, 7, 4]])
                         ]
four_by_four_arrays = [np.array([[10, 12, 15, 20],
                                 [9, 11, 14, 18],
                                 [8, 10, 12, 16],
                                 [6, 8, 9, 12]]),
                       np.array([[7, 11, 14, 16],
                                 [5, 9, 12, 14],
                                 [4, 8, 11, 13],
                                 [3, 7, 9, 11]])]

perfect_3_by_3 = np.array([[10, 0, 0],
                            [0, 10, 0],
                            [0, 0, 10]])
antiperfect_3_by_3 = np.array([[5, 5, 5],
                                [5, 5, 5],
                                [5, 5, 5]])


@pytest.mark.parametrize("table, expected", [
    (table, expected) for table, expected in
    (zip(two_by_two_arrays + three_by_three_arrays + four_by_four_arrays, [0.0013460761879122358,
                                                                           3.3651904697805894e-05,
                                                                           0.1872081310600528,
                                                                           0.09638972091264894,
                                                                           8.379313228244736e-09,
                                                                           2.7731805637144637e-06,
                                                                           0.0035199656200600848,
                                                                           0.00020072553142454845,
                                                                           2.7911468584095127e-07,
                                                                           7.346252199641538e-07]
         ))
])
def test_calculate_matrix_pval(table, expected):
    np.testing.assert_allclose(calculate_matrix_pval(table), expected)


@pytest.mark.parametrize("matrix_dict, expected", [
    ({(0, 0): 10, (0, 1): 0, (1, 0): 2, (1, 1): 12},
     np.array([[10., 0.],
               [2., 12.]])),
    ({(0, 1): 15, (1, 1): 0, (2, 1): 0, (0, 0): 7, (0, 2): 0, (1, 0): 16, (1, 2): 88, (2, 0): 0, (2, 2): 104},
     np.array([[7., 15., 0.],
               [16., 0., 88.],
               [0., 0., 104.]]))])
def test_dict_to_array(matrix_dict, expected):
    np.testing.assert_array_equal(dict_to_array(matrix_dict), expected)


@pytest.mark.parametrize("table, expected", [
    (table, expected) for table, expected in
    (zip(two_by_two_arrays + three_by_three_arrays + four_by_four_arrays, [11, 11, 26, 21, 36924, 168, 1949, 6215]
         ))
]
                         )
def test_get_possible_tables(table, expected):
    tables = get_possible_tables(table)
    assert len(tables) == expected


@pytest.mark.parametrize("table, expected", [
    (table, expected) for table, expected in
    (zip(two_by_two_arrays + three_by_three_arrays + four_by_four_arrays, [0.030303030303030304, 0.0,
                                                                           0.8333333333333334,
                                                                           2.0625,
                                                                           7.652280379553107e-05,
                                                                           np.inf,
                                                                           0.28125,
                                                                           0.0001763668430335097,
                                                                           1.9870044290529433e-05,
                                                                           -1.6016488961729118e-05]

         ))
]
                         )
def test_calculate_odds_ratio(table, expected):
    np.testing.assert_allclose(calculate_odds_ratio(table), expected)


@pytest.mark.parametrize("table, alternative, nan_policy, expected_output",
                         [(two_by_two_arrays[0],
                           'greater',
                           'propagate',
                           (0.030303030303030304, 0.0013797280926100416)),
                          (two_by_two_arrays[1],
                           'two-sided',
                           'propagate',
                           (0.0, 6.730380939561179e-05)),
                          (three_by_three_arrays[0],
                           'two-sided',
                           'propagate',
                           (7.652280379553107e-05, 1.0719448466611434e-05)),
                          (three_by_three_arrays[2],
                           'two-sided',
                           'propagate',
                           (0.28125, 0.5297232810816128)),
                          (perfect_3_by_3,
                           'two-sided',
                           'propagate',
                           (np.inf, 1.0808869515760631e-12)),
                          (antiperfect_3_by_3,
                           'two-sided',
                           'propagate',
                           (0.008, 1.0))

                          ]
                         )
def test_multiclass_fisher_exact(table,
                                 alternative,
                                 nan_policy,
                                 expected_output):
    odds_ratio, pval = multiclass_fisher_exact(table=table,
                                               nan_policy=nan_policy,
                                               alternative=alternative,
                                               )
    print(odds_ratio, pval)
    np.testing.assert_allclose(odds_ratio, expected_output[0])
    np.testing.assert_allclose(pval, expected_output[1])


@pytest.mark.parametrize("table",
                         two_by_two_arrays

                         )
def test_numical_equivilency_multiclass_fisher_exact_greater(table,
                                                             alternative='two-sided',
                                                             ):
    mc_odds_ratio, mc_pval = multiclass_fisher_exact(table=table,
                                                     alternative=alternative,
                                                     )
    sp_odds_ratio, sp_pval = fisher_exact(table, alternative=alternative)

    np.testing.assert_allclose(sp_pval, mc_pval)
    np.testing.assert_allclose(sp_pval, mc_pval)

# 0.9999663480953022 g
# 0.0013797280926100418 l
# 0.0027594561852200836 tw
