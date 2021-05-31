# Neural Knapsack

This repository contains the source code related to the publication

> Hertrich, Christoph and Skutella, Martin:
> Provably Good Solutions to the Knapsack Problem via Neural Networks of Bounded Size.
> [arXiv:2005.14105](https://arxiv.org/abs/2005.14105).

### Requirements
(with the versions used in our paper)
* python (3.8.5)
* numpy (1.19.1)
* tensorflow (2.2.0)
* statsmodels (0.11.1)
* matplotlib (3.3.1)

Note that numpy and tensorflow only guarantee the same outcome of random experiments with explicit seeds if exactly these versions are used.

### Reproducing the experiments of our paper

##### Section 5 and Appendix A

For reproducing the experiment  with threshold 0.005 one only needs to run 'knapsack_experiments.py' and 'knapsack_analyze.py' in this order. The latter will output a plot as included in our paper, as well as, a summary of the least squares regression including the reported p-value.

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

If you have any questions, please do not hesitate to contact us.
