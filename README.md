### Fairness measures API
The aim of this API is to give an accessible and practical way to calculate fairness both in dataset and in classification algorithms.

## Usage
`from fairness_measures_api import fairness_measures_api`

`fairness_measures_api(d, 'r', 'y', 'y_hat', g0, g1)`

where 
- *d* is a panda dataset
- *r* is the name of the score variable
- *y* is the name of the ground truth success variable
- *y_hat* is the name of the predicted success variable
- *g0* is first group
- *g1* is second group

methods are 
- true_statistical_parity
- statistical_parity
- total_accuracy
- calibration
- ofi
