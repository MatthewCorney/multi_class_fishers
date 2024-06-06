# Introduction
This package provides a function multiclass_fisher_exact to run the Fisher's exact test on a symmetrical contingency table of counts. 
The function takes either a n-dimensional array as input and returns a tuple containing the calculated odds ratio and p-value.

currently, matrix's of 2x2, 3x3 and 4x4 are supported

The function works by calculating the pvalue of the provided matrix,
and all the possible corresponding matrixes which still preserve row and column sums
This is achieved by backtracking

# Dependencies
The function requires the following Python packages:

- numpy
- scipy
- python-constraint
- Decimal

# Usage
The function takes three parameters:

- table: The symmetric contingency table of counts. It can be a list of lists or a numpy array.
- alternative: This parameter specifies the alternative hypothesis to be tested. Valid options are 'two-sided', 'less', or 'greater'.
- nan_policy: Specifies how to handle the presence of NaN values in the input data. Valid options are 'propagate', 'assume-zero', or 'raise'.

If table is a list of lists, it will be converted to a numpy array before being processed. 
The function will raise an error if the input data is not symmetrical or if it contains negative values.

The alternative parameter specifies the type of test to be performed. 
If 'two-sided', a two-sided test is performed. If 'less', a one-sided test for values less than the observed value is performed. 
If 'greater', a one-sided test for values greater than the observed value is performed.In general, it is recommend that two-sided is used

The nan_policy parameter specifies how to handle NaN values in the input data. 
If 'propagate', NaN values will be propagated through the calculation and the function will return NaN values. 
If 'assume-zero', NaN values will be treated as zero. 
If 'raise', the function will raise an error if NaN values are present in the input data.

License
This package is distributed under the MIT License. Please see the LICENSE file for details.

# Example
```
pip install git+https://github.com/MatthewCorney/multi_class_fishers.git
```

```
from multi_class_fishers.multi_class_fishers import multiclass_fisher_exact

# create a 3x3 symmetrical contingency table of counts
table=np.array([[1, 9, 12],
                [11, 3, 90],
                [11, 3, 90]])

# calculate the odds ratio and p-value using the default two-sided alternative hypothesis
odds_ratio, p_value = multi_class_fishers(table)

# print the results
print("Odds Ratio:", odds_ratio)
print("P-Value:", p_value)
```

```
Odds Ratio: 7.652280379553107e-05
P-Value: 1.0719448466611434e-05
```