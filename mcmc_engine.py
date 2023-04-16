import numpy as np
from multiprocessing import Pool


class MCMCEngine:
    def __init__(self, model, prior, transition_model, std_tr, pri_lim):
        self.model = model
        self.prior = prior
        self.transition_model = transition_model
        # self.acceptance = acceptance
        self.std_tr = std_tr
        self.pri_lim = pri_lim
        
    def prior(x, lim_pri):
        for i in range(len(x)):
            if not (lim_pri[i][0] <= x[i].item() <= lim_pri[i][1]):
                return 0
        return 1




    def transition_model(self, x):
        return np.random.normal(x, self.std_tr, len(x))

    def likelihood(self, x, y):
        yhat = self.model(x)
        err = np.linalg.norm(yhat - y)
        lik = -err / (2 * 0.1**2)
        return lik

    def acceptance(self, x_lik, x_prior, x_new_lik, x_new_prior):
        accept_prob = min(1, (x_new_lik * x_new_prior) / (x_lik * x_prior))
        return np.random.rand() < accept_prob

    def metropolis_hastings(self, likelihood_computer, prior, transition_model, param_init, iterations, data, acceptance_rule):
        x = param_init
        accepted = []
        rejected = []
        prob = []

        for i in range(iterations):
            x_new = transition_model(x)
            x_lik = likelihood_computer(x, data)
            x_new_lik = likelihood_computer(x_new, data)
            x_prior = prior(x)
            x_new_prior = prior(x_new)

            if acceptance_rule(x_lik, x_prior, x_new_lik, x_new_prior):
                x = x_new
                accepted.append(x_new[:])
            else:
                rejected.append(x_new)
                prob.append(x_new_lik)

        return np.array(accepted), np.array(rejected), prob


    def run_chain(self, index, param_init, iterations, data):
        tr_model = lambda x: self.transition_model(x, self.std_tr)
        pri_model = lambda x: self.prior(x, self.pri_lim)
        result = self.metropolis_hastings(self.likelihood, pri_model, tr_model, param_init, iterations, data,
                                          lambda x_lik, x_prior, x_new_lik, x_new_prior: self.acceptance(x_lik, x_prior, x_new_lik, x_new_prior))
        return index, result
        
    def _run_chain_wrapper(self, args):
        i, init_cond, iterations, data = args
        return self.run_chain(i, init_cond, iterations, data)

    def run_chains_parallel(self, initial_conditions, iterations, data, num_chains):
        chain_args = [(i, init_cond, iterations, data) for i, init_cond in enumerate(initial_conditions)]

        with Pool(num_chains) as pool:
            results = pool.map(self._run_chain_wrapper, chain_args)

        return results