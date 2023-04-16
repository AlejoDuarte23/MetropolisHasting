import numpy as np
from scipy import stats
from tqdm.auto import tqdm
import multiprocessing
from multiprocessing import Pool, freeze_support, Value
from ctypes import c_int
import warnings
from tabulate import tabulate
warnings.filterwarnings('ignore')


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

# Engine
def likelihood(x, y, model):
    yhat = model(x[:-1], y)
    err = np.linalg.norm(yhat - y)
    lik = -err / (2 * x[-1]**2)
    return lik

def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = 0.1
    return accept < np.exp(x_new - x)

def metropolis_hastings(likelihood_computer, prior, transition_model, param_init, iterations, data, acceptance_rule):
    x = param_init
    accepted = []
    rejected = []
    prob = []

    for i in range(iterations):
        x_new = transition_model(x)
        x_lik = likelihood_computer(x, data, model)
        x_new_lik = likelihood_computer(x_new, data, model)

        if acceptance_rule(x_lik + np.log(prior(x)), x_new_lik + np.log(prior(x_new))):
            x = x_new
            accepted.append(x_new[:])
        else:
            rejected.append(x_new)
            prob.append(x_new_lik)

    return np.array(accepted), np.array(rejected), prob


def run_chain(args):
    index, likelihood, prior, transition_model, param_init, iterations, data, acceptance_rule, std_tr, pri_lim = args
    tr_model = lambda x: transition_model(x, std_tr)
    pri_model = lambda x: prior(x, pri_lim)
    result = metropolis_hastings(likelihood, pri_model, tr_model, param_init, iterations, data, acceptance_rule)
    return index, result



def print_results(results):
    headers = ["Chain", "Mean_a", "Std_a", "Mean_b", "Std_b", "Mean_sigma", "Std_sigma"]
    table_data = []

    for i, result in enumerate(results):
        accepted, _, _ = result
        means = accepted.mean(axis=0)
        stds = accepted.std(axis=0)
        table_data.append([i + 1, means[0], stds[0], means[1], stds[1], means[2], stds[2]])

    print(tabulate(table_data, headers=headers, floatfmt=".4f"))


# Main Execution
if __name__ == '__main__':
    freeze_support()


    zo = np.linspace(0, 10, 100)
    error = np.random.normal(0, 1, 100)
    y = 2.23 * zo**2 + 4.56 + error

    std_tr = [0.5, 0.5, 0.01]
    pri_lim = [[0, 10], [0, 10], [0, 10]]

    xoo = [2, 4, 0.1]
    iterations = 100000

    # Run multiple chains with different settings, parameters, and initial conditions
    num_chains = 4
    initial_conditions = [np.random.uniform(1, 5, 3) for _ in range(num_chains)]

    chain_args = [(i, likelihood, prior, transition_model, init_cond, iterations, y, acceptance, std_tr, pri_lim) for i, init_cond in enumerate(initial_conditions)]



    with Pool(num_chains) as pool:
        with tqdm(total=num_chains, desc="Chains") as pbar:
            results = [None] * num_chains
            for index, result in pool.imap_unordered(run_chain, chain_args):
                results[index] = result
                pbar.update()
                pbar.set_description(f"Chain {index + 1}/{num_chains} completed")

    print_results(results)
