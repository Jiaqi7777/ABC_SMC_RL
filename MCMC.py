import numpy as np
import scipy.stats as stats
from copy import deepcopy
   
def generate_samples(model, obs, para):
    samples = []
    for s0, a, s1 in zip(obs['state0'], obs['action'], obs['state1']):
        samples.append(model.q_value(para, s0, a) - model.gamma * model.v_value(para, s1))
    return np.array(samples)

def get_log_prior(parameters, sigma=1):
        """log(p(para))"""
        parameters = parameters.reshape(-1)
        sigma = sigma * np.identity(len(parameters))
        return stats.multivariate_normal.logpdf(parameters, cov=sigma).sum()

def get_log_likelihood(obs, samples, sigma=1, tractability=False):
    """log(p(evidence|para))"""
    if tractability:
        raise NotImplementedError("Not implemented for tractable likelihood")
        likelihood = stats.norm.pdf(evidence, loc=9.8, scale=sigma)
        log_likelihood = np.log(likelihood).sum()
        return log_likelihood
    else:
        #sigma = sigma * np.identity(len(samples))
        lld = stats.norm.logpdf(obs['rewards'], loc=samples, scale=sigma).sum()
        return lld
        
    
def get_log_posterior(para, *args):
    log_prior = get_log_prior(para)
    log_likelihood = get_log_likelihood(*args)
    return log_prior + log_likelihood

class Kernel:
    def __init__(self, model=None, sigma=0.1, prior=get_log_prior, likelihood=get_log_likelihood, tractability=False):
        self.sigma = sigma
        self.model=model
        self.prior = prior
        self.likelihood = likelihood
        self.tractability = tractability

    def posterior(self, para, *args):
        return self.prior(para) + self.likelihood(*args, tractability=self.tractability)

    def move(self):
        raise NotImplementedError

class RandomWalk(Kernel):
    def __init__(self, *args, model=None):
        super(RandomWalk, self).__init__(*args)   
        self.model=model

    def move(self, current_para, sigma=0):
        move_ratio = 1
        return current_para , move_ratio
        #+ np.random.normal(size=(current_para.shape), scale=sigma), move_ratio

    def accept(self, current_para, obs, samples):
        proposed_para, move_ratio = self.move(current_para)
        proposed_samples = generate_samples(self.model, obs, proposed_para)
        print(proposed_samples-samples)
        current_log_posterior = self.posterior(current_para, obs, samples)
        proposed_log_posterior = self.posterior(proposed_para, obs, proposed_samples)
        return proposed_log_posterior - current_log_posterior - np.log(move_ratio), proposed_para
        

class MCMC:
    def __init__(self, steps = 10, kernal=RandomWalk()):
        self.steps = steps
        self.kernel = kernal
        self.accepted = 0

    def reset(self):
        self.accepted = 0

    def update(self, paras, obs, samples):
        for i, current_para in enumerate(paras):
            new_paras = deepcopy(paras)
            acceptance_ratio, proposed_para = self.kernel.accept(current_para, obs, samples[:, i])
            if acceptance_ratio > 1:
                accept = True
            else:
                alpha = np.random.uniform(0, 1)
                accept = alpha < np.exp(acceptance_ratio)
            if accept:
                new_paras[i] = proposed_para
                self.accepted += 1
                print('accept with ratio ', np.exp(acceptance_ratio))
        
        return new_paras
        

if __name__ == '__main__':
    from GridWrold import *
    from model import *
    n_particle = 5    
    r = []
    env = GridWorld(n_cell=9, starting_position=5, goal_position=8)
    model = Tabular(env=env, n_particle=n_particle, prior='normal')
    s0 = (1, 2)
    s1 = (2, 3)
    action = 1
    obs = {'rewards':[1,2,3], 'state0':[(1,),(1,),(2,)],'state1':[(7,),(8,),(8,)], 'action':[-1,0,1]}
    samples_l = []
    for j in range(len(obs['state0'])):
        samples_l.append(model.r_hat(obs['state0'][j], obs['state1'][j], obs['action'][j]))
    samples_l = np.array(samples_l)
    for i in range(n_particle):
        propsed_samples = generate_samples(model, obs, model.get_parameter()[i])
        print(i, samples_l[:, i], propsed_samples)