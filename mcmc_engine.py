import numpy as np
from multiprocessing import Pool


class MCMCEngine:
    def __init__(self, model, prior, transition_model, acceptance, std_tr, pri_lim):
        self.model = model
        self.prior = prior
        self.transition_model = transition_model
        self.acceptance = acceptance
        self.std_tr = std_tr
        self.pri_lim = pri_lim
        
    def prior(self, x):
        for i in range(len(x)):
            if not self.pri_lim[i][0] <= x[i] <= self.pri_lim[i][1]:
                return 0
        return 1

    def transition_model(self, x):
        return np.random.normal(x, self.std_tr, len(x))

    def likelihood(self, x, y):
        yhat = self.model(x[:-1], y)
        err = np.linalg.norm(yhat - y)
        lik = -err / (2 * x[-1]**2)
        return lik

    def acceptance(self, x, x_new):
        if x_new > x:
            return True
        else:
            accept = 0.1
        return accept < np.exp(x_new - x)

    def metropolis_hastings(self, likelihood_computer, prior, transition_model, param_init, iterations, data, acceptance_rule):
        x = param_init
        accepted = []
        rejected = []
        prob = []

        for i in range(iterations):
            x_new = transition_model(x)
            x_lik = likelihood_computer(x, data)
            x_new_lik = likelihood_computer(x_new, data)

            if acceptance_rule(x_lik + np.log(prior(x)), x_new_lik + np.log(prior(x_new))):
                x = x_new
                accepted.append(x_new[:])
            else:
                rejected.append(x_new)
                prob.append(x_new_lik)

        return np.array(accepted), np.array(rejected), prob


    def run_chain(self, index, param_init, iterations, data):
        tr_model = lambda x: self.transition_model(x, self.std_tr)
        pri_model = lambda x: self.prior(x, self.pri_lim)
        result = self.metropolis_hastings(self.likelihood, pri_model, tr_model, param_init, iterations, data, self.acceptance)
        return index, result
    
    def _run_chain_wrapper(self, args):
        i, init_cond, iterations, data = args
        return self.run_chain(i, init_cond, iterations, data)

    def run_chains_parallel(self, initial_conditions, iterations, data, num_chains):
        chain_args = [(i, init_cond, iterations, data) for i, init_cond in enumerate(initial_conditions)]

        with Pool(num_chains) as pool:
            results = pool.map(self._run_chain_wrapper, chain_args)

        return results