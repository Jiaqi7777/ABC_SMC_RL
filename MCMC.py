import numpy as np
import scipy.stats as stats
from copy import deepcopy
from parameter import *
   
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
        data_length = min(len(obs['rewards']), len(samples))
        lld = stats.norm.logpdf(obs['rewards'][:data_length], loc=samples[:data_length], scale=sigma).sum()
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
    def __init__(self, *args, model=None, sigma=0.5):
        print('sigma', sigma)
        super(RandomWalk, self).__init__(*args)   
        self.model=model
        self.sigma = sigma

    def move(self, current_para):
        move_ratio = 1
        return current_para + np.random.normal(size=(current_para.shape), scale=self.sigma), move_ratio

    def accept(self, current_para, obs, samples):
        proposed_para, move_ratio = self.move(current_para)
        proposed_samples = generate_samples(self.model, obs, proposed_para)
        #print(proposed_samples-samples)
        current_log_posterior = self.posterior(current_para, obs, samples)
        proposed_log_posterior = self.posterior(proposed_para, obs, proposed_samples)
        return proposed_log_posterior - current_log_posterior - np.log(move_ratio), proposed_para, proposed_samples
        

class MCMC:
    def __init__(self, steps = 10, kernal=RandomWalk(sigma=sigma)):
        self.steps = steps
        self.kernel = kernal
        self.accepted = 0

    def reset(self):
        self.accepted = 0

    def update(self, paras, obs, samples):
        new_paras = deepcopy(paras)
        for i, current_para in enumerate(paras):
            acceptance_ratio, proposed_para, proposed_samples = self.kernel.accept(current_para, obs, samples[:, i])
            if acceptance_ratio > 1:
                accept = True
            else:
                alpha = np.random.uniform(0, 1)
                accept = alpha < np.exp(acceptance_ratio)
            if accept:
                new_paras[i] = proposed_para
                self.accepted += 1
                samples[:, i] = proposed_samples
                # print('accept with ratio ', np.exp(acceptance_ratio))
        return new_paras, samples
        

if __name__ == '__main__':
    from GridWorld import *
    from model import *
    from QLearning import *
    n_particle = 5    
    sigma = 1
    r = []
    random.seed(10)
    env = GridWorld((10,12), (1,2), obstacles=True)
    model = Tabular(env=env, n_particle=n_particle, prior='normal')
    kernel = RandomWalk(model=model, sigma=sigma)
    mcmc = MCMC(kernal=kernel)
    
    
    print(env.reset())
    # env.plot_env()
    env.expert()
    # env.plot_env(env.expert_traj + env.R)
    
    obs = env.expert_obs._buffers
    R = [obs['rewards'][0]]
    samples_l = np.expand_dims(model.r_hat(obs['state0'][0], obs['state1'][0], obs['action'][0]), axis=0)
    for j in range(1, len(obs['state0'])):
        samples_l = np.append(samples_l, np.expand_dims(model.r_hat(obs['state0'][j], obs['state1'][j], obs['action'][j]), axis=0), axis=0)
        R += obs['rewards'][j]
        if j % 1 == 0:
            new_parameter, samples_l = mcmc.update(model.get_parameter(), obs, samples_l)
            model.set_parameter(new_parameter)
    model.plot_policy(show=True, additional_info = env.R)
    
        
    