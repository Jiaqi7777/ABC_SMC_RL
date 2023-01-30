import numpy as np
import scipy.stats as stats
from copy import deepcopy

class RandomWalk:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def move(self, current_para):
        ratio = 1
        return current_para + stats.norm(0, self.sigma), ratio
   

class MCMC:
    def __init__(self, parameters, proposal=None, steps = 10, tractability=False):
        self.parameters = parameters
        self.proposal = proposal
        self.steps = steps
        self._tractability = tractability

    def get_log_prior(self, parameters, sigma=1):
        """log(p(para))"""
        return np.log(stats.norm.pdf(parameters, scale=sigma)).sum()

    def get_log_likelihood(self, obs, samples, sigma=1):
        """log(p(evidence|para))"""
        if self._tractability:
            raise NotImplementedError("Not implemented for tractable likelihood")
            likelihood = stats.norm.pdf(evidence, loc=9.8, scale=sigma)
            log_likelihood = np.log(likelihood).sum()
            return log_likelihood
        else:
            likelihood = stats.norm.pdf(obs['rewards'], loc=samples, scale=sigma)
            return np.log(likelihood).sum()
        
    def get_log_posterior(self, para, *args):
        log_prior = self.get_log_prior(para)
        log_likelihood = self.get_log_likelihood(*args)
        return log_prior + log_likelihood

    def random_walk(self, current_para, sigma=1):
        ratio = 1
        return current_para + np.random.normal(size=(current_para.shape), scale=sigma), ratio

    def update(self, tables, obs, samples):
        samples = np.array(samples)
        return self.metropolis_hastings(tables, obs, samples)

    def accept(self, current_para, proposed_para, ratio, *args):
        current_log_posterior = self.get_log_posterior(current_para, *args)
        proposed_log_posterior = self.get_log_posterior(proposed_para, *args)
        r = proposed_log_posterior - current_log_posterior - np.log(ratio)
        if r > 1:
            return True
        else:
            alpha = np.random.uniform(0, 1)
            if alpha < np.exp(r):
                return True
            else:
                return False

    def metropolis_hastings(self, paras, obs, samples, move=random_walk):
        new_paras = deepcopy(paras)
        for i, table in enumerate(paras):
            current_para = table
            for _ in range (self.steps):
                proposed_para, ratio = self.random_walk(current_para)
                current_para = proposed_para
            if self.accept(current_para, proposed_para, ratio, obs, samples[:, i]):
                new_paras[i] = proposed_para
        return new_paras

