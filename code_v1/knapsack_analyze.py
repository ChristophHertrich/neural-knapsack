import knapsack_commons as kc
import matplotlib.pyplot as plt
import numpy as np

threshold = 0.005
filename = './results/t0.005_seed257/results.json'
prec_name = 'loss'
throw_away_start = 0
throw_away_end = 0
plot_reg_line = True
include_title = True
degree = 2
use_toy_data = False
toy_data_x = range(2,18)
toy_data_y = [3, 7, 10, 14, 18, 24, 38, 38, 58, 62, 70, 70, 70, 86, 110, 126]

def extract_pstar_width_function(results, threshold):
    function = {}
    for result in results:
        pstar = result['pstar']
        width = result['width']
        mse = result[prec_name]
        if mse <= threshold:
            if not pstar in function or function[pstar] > width:
                function[pstar] = width
    function = dict(sorted(function.items()))
    return list(function.keys()), list(function.values())
    
results = kc.read_results(filename)
if use_toy_data:
    pstars, widths = toy_data_x, toy_data_y
else:
    pstars, widths = extract_pstar_width_function(results, threshold)
pstars, widths = pstars[throw_away_start:], widths[throw_away_start:]
if throw_away_end > 0:
    pstars, widths = pstars[:-throw_away_end], widths[:-throw_away_end]
fig, ax = plt.subplots()
ax.plot(pstars, widths, 'ok', label='Observations')
params = kc.perform_lin_reg(pstars, widths, degree)
if plot_reg_line:
    x = np.arange(min(pstars), max(pstars)+1)
    y = params[0] + params[1]*x
    if degree == 2:
        y += params[2]*(x**2)
    ax.plot(x, y, '-k', label='Quadratic Regression Line')
    ax.legend()
ax.set_xlabel('$p*$')
ax.set_ylabel('width')
if include_title:
    ax.set_title('Required width for MSE loss $\leq '+str(threshold)+'$')

fig.savefig('test.pdf')