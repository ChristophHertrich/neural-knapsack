import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import knapsack_commons as kc

# parameters for running:
results_dir = './results/t0.005_seed257/'
pstars = range(3,91,3)
widths = range(1,10001,1)
break_at_threshold = 0.005
prime_for_seed = 257
patience = 2
epochs = 100
steps_per_epoch = 1000
eval_n = steps_per_epoch
batch_size = 32
verbose = 0
reg = None

def experiments(results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_filename = results_dir + 'results.json'
    results = []
    for pstar in pstars:
        for width in widths:
            seed = prime_for_seed * pstar + width
            result = kc.one_experiment(seed, pstar, width, reg, batch_size, steps_per_epoch, epochs, patience, verbose, eval_n)
            results.append(result)
            with open(results_filename, 'w') as fout:
                json.dump(results, fout, indent = 4)
            if break_at_threshold is not None and result['loss'] < break_at_threshold:
                break
    return results

results = experiments(results_dir)