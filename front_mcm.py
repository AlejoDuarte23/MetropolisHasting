import numpy as np
from tqdm.auto import tqdm
from mcmc_engine import MCMCEngine
import warnings
from tabulate import tabulate
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 


# Inputs
def model(x, z):
    y = x[0] * z**2 + x[1]
    return y

def prior(x, lim_pri):
    for i in range(len(x)):
        if not lim_pri[i][0] <= x[i] <= lim_pri[i][1]:
            return 0
    return 1

def transition_model(x, std):
    return np.random.normal(x, std, len(x))

def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = 0.1
    return accept < np.exp(x_new - x)


def print_results(results):
    table_data = []

    for i, result in enumerate(results):
        chain_id, (accepted, _, _) = result
        means = accepted.mean(axis=0)
        table_data.append([f"Chain {i + 1}", means[0], means[1], means[2]])

    headers = ["Chain", "Mean a", "Mean b", "Mean sigma"]
    print(tabulate(table_data, headers=headers, floatfmt=".4f"))




    

    
    

if __name__ == '__main__':
    # ...
    zo = np.linspace(0, 10, 100)
    error = np.random.normal(0, 1, 100)
    y = 2.23 * zo**2 + 4.56 + error

    std_tr = [0.5, 0.5, 0.01]
    pri_lim = [[0, 10], [0, 10], [0, 10]]

    xoo = [2, 4, 0.1]
    iterations = 100000
    
    std_tr = [0.5, 0.5, 0.01]
    pri_lim = [[0, 10], [0, 10], [0, 10]]

    engine = MCMCEngine(model, prior, transition_model, acceptance, std_tr, pri_lim)

    num_chains = 4
    initial_conditions = [np.random.uniform(1, 5, 3) for _ in range(num_chains)]

    results = engine.run_chains_parallel(initial_conditions, iterations, y, num_chains)

    print_results(results)

    def plot_trace_hist(results):
        n_params = 3
        n_chains = len(results)
        fig, axes = plt.subplots(n_params, n_chains, figsize=(12, 8))
        
        for i, result in enumerate(results):
            chain_id, (accepted, _, _) = result
            
            for j in range(n_params):
                # Plot the trace
                axes[j, i].plot(accepted[:, j], alpha=0.7)
                # Plot the histogram
                axes[j, i].hist(accepted[:, j], bins=30, alpha=0.3, density=True)
                # Set the title for the top row
                if j == 0:
                    axes[j, i].set_title(f"Chain {i + 1}")
                # Set the y-label for the first column
                if i == 0:
                    axes[j, i].set_ylabel([f"a", f"b", f"sigma"][j])
    
        plt.tight_layout()
        plt.show()
    
    plot_trace_hist(results)