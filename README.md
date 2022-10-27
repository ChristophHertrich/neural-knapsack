# Neural Knapsack

This repository contains the source code related to the publication

> Hertrich, Christoph and Skutella, Martin:
> Provably Good Solutions to the Knapsack Problem via Neural Networks of Bounded Size.
> [(arXiv:2005.14105)](https://arxiv.org/abs/2005.14105).

and the corresponding chapter in the dissertation

> Hertrich, Christoph:
> Facets of Neural Network Complexity.
> [(link to dissertation)](https://doi.org/10.14279/depositonce-15271).

There exist two different code versions. For reproducing the experiments in arXiv.v2 and in the dissertation, please use the code provided in the subfolder 'code_v1'. For a future journal version, please use the code provided in the subfolder 'code_v2'.

### Requirements

(with the versions used in our paper)

Note that numpy and tensorflow only guarantee the same outcome of random experiments with explicit seeds if exactly these versions are used.

#### For code_v1
* python (3.8.5)
* numpy (1.19.1)
* tensorflow (2.2.0)
* statsmodels (0.11.1)
* matplotlib (3.3.1)

#### For code_v2
* python (3.8.13)
* numpy (1.22.3)
* tensorflow (2.3.0)
* matplotlib (3.5.1)

### Reproducing the experiments of our paper

#### Reproducing experiments with threshold 0.005

For reproducing the experiment with threshold 0.005 one only needs to run the corresponding version of 'knapsack_experiments.py' and 'knapsack_analyze.py' in this order. The latter will output a plot as included in our paper. Moreover, in the case of code_v1, it will also print a summary of the least squares regression including the reported p-value. We do not perform regression and statistical tests in code_v2.

#### Reproducing experiments with threshold 0.0025 in code_v2

For reproducing the experiment with threshold 0.0025, please modify the following parameters in the header of 'knapsack_experiments.py':
```python
results_dir = './results/t0.0025_seed257/'
pstars = range(1,31,1)
break_at_threshold = 0.0025
```

Please also modify the following parameters of 'knapsack_analyze.py':
```python
threshold = 0.0025
filename = './results/t0.0025_seed257/results.json'
```

Then proceed as above.

#### Reproducing experiments with threshold 0.00375 in code_v2

For reproducing the experiment with threshold 0.00375, please modify the following parameters in the header of 'knapsack_experiments.py':
```python
results_dir = './results/t0.00375_seed257/'
pstars = range(2,61,2)
break_at_threshold = 0.00375
```

Please also modify the following parameters of 'knapsack_analyze.py':
```python
threshold = 0.00375
filename = './results/t0.00375_seed257/results.json'
```

Then proceed as above.

#### Reproducing experiments with threshold 0.0025 in code_v1

For reproducing the experiment with threshold 0.0025, please modify the following parameters in the header of 'knapsack_experiments.py':
```python
results_dir = './results/t0.0025_seed257/'
pstars = range(1,26,1)
break_at_threshold = 0.0025
```

Please also modify the following parameters of 'knapsack_analyze.py':
```python
threshold = 0.0025
filename = './results/t0.0025_seed257/results.json'
```

Then proceed as above.

#### Reproducing experiments with threshold 0.00375 in code_v1

For reproducing the experiment with threshold 0.00375, please modify the following parameters in the header of 'knapsack_experiments.py':
```python
results_dir = './results/t0.00375_seed257/'
pstars = range(2,51,2)
break_at_threshold = 0.00375
```

Please also modify the following parameters of 'knapsack_analyze.py':
```python
threshold = 0.00375
filename = './results/t0.00375_seed257/results.json'
```

Then proceed as above.

### Questions?

If you have any questions, please do not hesitate to contact us.
